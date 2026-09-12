"""Push the retrained checkpoint + model card to star-ga/mind-mem-4b.

Publication is gated on two attested evaluation reports (product-conformance
and held-out paraphrase) that bind to the *same* checkpoint — the one this
script is actually about to upload.

Integrity scope: the gate checks that recorded bytes and recorded results are
mutually consistent and unchanged between evaluation and publication.  It is
not proof that the evaluation ran, nor that the artifact originates with an
authorised party (see :mod:`eval_receipt`).  Do not describe a passing gate as
a quality guarantee — it is a refusal to publish an unattested, mismatched or
failing candidate.

Requires a HuggingFace token with **write** scope via ``HF_TOKEN=...``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ID = os.environ.get("MM_HF_REPO_ID", "star-ga/mind-mem-4b")
OUT_DIR = Path(os.environ.get("MM_TRAIN_ROOT", "/data/checkpoints/mm-workspace/train-output"))
# Source dir holding the trained weights. For QLoRA this is `adapter/`; for
# a full fine-tune (v3.9 onward) this is `full-ft/`. Override via MM_WEIGHTS_DIR.
_DEFAULT_WEIGHTS_DIR = (OUT_DIR / "full-ft") if any((OUT_DIR / "full-ft" / name).is_file() for name in ("model.safetensors", "model.safetensors.index.json")) else (OUT_DIR / "adapter")
WEIGHTS_DIR = Path(os.environ.get("MM_WEIGHTS_DIR", str(_DEFAULT_WEIGHTS_DIR)))

# ---------------------------------------------------------------------------
# Fixed pass criteria.  These live here, in code, and are never read from a
# report: a report that carries its own thresholds is a report that can lower
# them.  Scope note: the `main` suite measures product conformance against a
# corpus the model trained on; `holdout` measures paraphrases held out of that
# corpus.  Neither is an uncontaminated capability benchmark and this gate
# makes no such claim.
# ---------------------------------------------------------------------------
MAIN_THRESHOLDS: dict[str, float] = {
    "tool_call": 0.95,
    "block_schema": 0.98,
    "workflow": 0.90,
    "v39_new_tools": 0.90,
    "v39_transform_hash": 0.95,
    "v39_transport_guard": 0.95,
    "v311_new_tools": 0.90,
    "v311_explain_field": 0.95,
    "v312_quality_gate_strict_mode": 0.90,
    "v312_lineage_staleness": 0.90,
    "v4_surfaces": 0.90,
}
HOLDOUT_THRESHOLDS: dict[str, float] = {"v4_holdout": 0.90, "v312_holdout": 0.90}

#: Files whose presence in the upload plan is required per checkpoint layout.
_FULLFT_REQUIRED = ("config.json",)
_LORA_REQUIRED = ("adapter_config.json", "adapter_model.safetensors")
_TOKENIZER_REQUIRED_ANY = ("tokenizer.json", "tokenizer.model")


def probe_sets() -> dict[str, dict[str, list]]:
    """The probe *definitions* themselves, imported from the harness modules.

    Importing the lists (not a hardcoded number) is what keeps the expectation
    from drifting when probes are added, and gives the gate the item content it
    needs to check digests rather than counts alone.
    """
    try:
        import eval_harness as H
        import eval_holdout as O
    except Exception as exc:  # pragma: no cover - environment failure path
        sys.exit(
            "refusing upload: cannot resolve the probe definitions from the eval "
            f"harness ({exc}); the gate will not accept report-supplied probes"
        )
    return {
        "main": {
            "tool_call": H.TOOL_CALL_QUESTIONS,
            "block_schema": H.BLOCK_SCHEMA_QUESTIONS,
            "workflow": H.WORKFLOW_QUESTIONS,
            "v39_new_tools": H.V39_NEW_TOOLS,
            "v39_transform_hash": H.V39_TRANSFORMHASH_PROMPTS,
            "v39_transport_guard": H.V39_TRANSPORT_PROMPTS,
            "v311_new_tools": H.V311_NEW_TOOLS,
            "v311_explain_field": H.V311_EXPLAIN_FIELD,
            "v312_quality_gate_strict_mode": H.V312_QUALITY_GATE_STRICT_MODE,
            "v312_lineage_staleness": H.V312_LINEAGE_STALENESS,
            "v4_surfaces": H.V4_SURFACES,
        },
        "holdout": {
            "v4_holdout": O.V4_HOLDOUT,
            "v312_holdout": O.V312_HOLDOUT,
        },
    }


def _expected_probe_specs() -> dict[str, dict[str, dict]]:
    """Per-group ``{n_items, digest}`` computed from the probe definitions."""
    from eval_receipt import probe_definition_digests

    return {suite: probe_definition_digests(groups) for suite, groups in probe_sets().items()}


def _expected_probe_counts() -> dict[str, dict[str, int]]:
    """Per-group counts derived from the same definitions (count-only view)."""
    return {
        suite: {group: spec["n_items"] for group, spec in groups.items()}
        for suite, groups in _expected_probe_specs().items()
    }


class GateError(Exception):
    """A publication precondition was not met."""


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GateError(f"{label} must be a non-bool integer, got {value!r}")
    return value


def _strict_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{label} must be a non-bool number, got {value!r}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise GateError(f"{label} must be finite, got {value!r}")
    return number


def _check_group(report: dict, group: str, expected: int, threshold: float) -> None:
    bench = report.get(group)
    if not isinstance(bench, dict):
        raise GateError(f"report is missing required probe group {group!r}")
    hits = _strict_int(bench.get("hits"), f"{group}.hits")
    total = _strict_int(bench.get("total"), f"{group}.total")
    if total <= 0:
        raise GateError(f"{group}.total must be > 0")
    if total != expected:
        raise GateError(
            f"{group}.total is {total} but the harness defines {expected} probes — "
            "partial or stale run"
        )
    if not 0 <= hits <= total:
        raise GateError(f"{group}.hits={hits} is out of range for total={total}")
    accuracy = _strict_float(bench.get("accuracy"), f"{group}.accuracy")
    if abs(accuracy - hits / total) > 1e-9:
        raise GateError(f"{group}.accuracy does not match hits/total")
    if accuracy < threshold:
        raise GateError(
            f"{group} scored {hits}/{total} ({accuracy:.2%}) below the fixed "
            f"threshold {threshold:.0%}"
        )


def _check_raw_items(report: dict, suite: str) -> None:
    """Recompute scores from every response using the fixed probe definitions."""
    from eval_harness import score_probe

    for group, probes in probe_sets()[suite].items():
        bench = report[group]
        items = bench.get("items")
        if not isinstance(items, list) or len(items) != len(probes):
            raise GateError(f"{group} must record exactly {len(probes)} raw probe outcomes")
        hits = 0
        for index, (item, probe) in enumerate(zip(items, probes)):
            if not isinstance(item, dict):
                raise GateError(f"{group} raw item {index} is malformed")
            if _strict_int(item.get("index"), f"{group}.items.index") != index:
                raise GateError(f"{group} raw item order/index differs from the probe definitions")
            if item.get("prompt") != probe[0] or not isinstance(item.get("response"), str):
                raise GateError(f"{group} raw prompt/response is missing or substituted")
            passed, _ = score_probe(group, probe, item["response"])
            if type(item.get("passed")) is not bool or item["passed"] != passed:
                raise GateError(f"{group} raw item verdict disagrees with its response")
            hits += passed
        if hits != bench["hits"]:
            raise GateError(f"{group} aggregate hits disagree with the raw responses")


def _check_probe_records(receipt: dict, expected: dict[str, dict]) -> None:
    """Gate on probe *identity* (digest) as well as size.

    A count alone is substitutable: eleven easier questions score the same
    shape as eleven real ones.  The digest is taken over the definitions the
    harness currently declares, so a substituted probe set fails here.
    """
    probes = receipt.get("probes")
    if not isinstance(probes, dict) or not probes:
        raise GateError("receipt records no probes")
    missing = sorted(set(expected) - set(probes))
    extra = sorted(set(probes) - set(expected))
    if missing or extra:
        raise GateError(f"probe groups mismatch (missing={missing}, unexpected={extra})")
    for group, spec in expected.items():
        record = probes[group]
        if not isinstance(record, dict):
            raise GateError(f"probe record for {group!r} is malformed")
        n_items = _strict_int(record.get("n_items"), f"probes.{group}.n_items")
        n_done = _strict_int(record.get("n_completed"), f"probes.{group}.n_completed")
        if n_items != spec["n_items"]:
            raise GateError(
                f"probes.{group}.n_items={n_items} but the harness defines {spec['n_items']}"
            )
        if record.get("digest") != spec["digest"]:
            raise GateError(
                f"probes.{group} digest does not match the harness probe definitions "
                "— the attested probe set is not the one this gate expects"
            )
        if n_done != n_items:
            raise GateError(
                f"probes.{group} completed {n_done}/{n_items} — run did not finish"
            )


def _check_layout(receipt: dict) -> str:
    """Refuse a checkpoint whose weight set is not fully present in the manifest."""
    from eval_receipt import base_binding_is_release_ready, weight_set_is_closed

    selection = receipt.get("selection") or {}
    kind = selection.get("kind")
    files = (selection.get("model") or {}).get("files") or {}
    tokenizer_files = (selection.get("tokenizer") or {}).get("files") or {}
    if not files:
        raise GateError("receipt records no model files")
    if kind == "full-ft":
        for name in _FULLFT_REQUIRED:
            if name not in files:
                raise GateError(f"full-FT checkpoint manifest is missing {name}")
        ok, reason = weight_set_is_closed(
            Path((selection["model"])["path"]), files, "full-FT checkpoint"
        )
        if not ok:
            raise GateError(reason)
        # The evaluator selects full-FT safetensors only, and the uploader
        # publishes that layout. The generic base-checkpoint closure helper
        # also accepts legacy .bin weights; accepting those here would let
        # the payload filter omit every weight while reporting a valid plan.
        if not any(name in files for name in ("model.safetensors", "model.safetensors.index.json")):
            raise GateError("full-FT release requires safetensors weights or a safetensors shard index")
    elif kind == "lora":
        for name in _LORA_REQUIRED:
            if name not in files:
                raise GateError(f"adapter manifest is missing {name}")
        ok, reason = base_binding_is_release_ready(receipt)
        if not ok:
            raise GateError(reason)
    else:
        raise GateError(f"unsupported checkpoint layout {kind!r}")
    if "tokenizer_config.json" not in tokenizer_files or not any(
        name in tokenizer_files for name in _TOKENIZER_REQUIRED_ANY
    ):
        raise GateError("tokenizer manifest is incomplete")
    return kind


def _validate_report(path: Path, suite: str, thresholds: dict[str, float],
                     expected: dict[str, dict]) -> dict:
    """Full quality + binding validation of one attested report."""
    from eval_receipt import bindings_match, receipt_is_valid, report_matches_receipt

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        receipt = report["receipt"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise GateError(f"missing or invalid evaluation receipt at {path}: {exc}") from exc
    if not isinstance(receipt, dict):
        raise GateError(f"receipt at {path} is not an object")
    if receipt.get("schema") != "mind-mem/eval-receipt@2":
        raise GateError(
            f"receipt at {path} has schema {receipt.get('schema')!r}; this gate "
            "requires mind-mem/eval-receipt@2"
        )
    if receipt.get("suite") != suite:
        raise GateError(f"receipt at {path} attests suite {receipt.get('suite')!r}, expected {suite!r}")
    run = receipt.get("run") or {}
    if run.get("status") != "completed" or not run.get("run_id"):
        raise GateError(f"receipt at {path} does not attest a completed run")
    if not receipt.get("complete"):
        raise GateError(f"evaluation receipt at {path} is incomplete")
    if not receipt_is_valid(receipt):
        raise GateError(f"evaluation receipt at {path} fails its own digest — tampered")
    if not report_matches_receipt(report):
        raise GateError(
            f"report payload at {path} does not match the digest inside its receipt — "
            "results were edited after evaluation"
        )
    ok, reason = bindings_match(receipt)
    if not ok:
        raise GateError(f"binding check for {path}: {reason}")
    _check_probe_records(receipt, expected)
    _check_layout(receipt)
    for group, threshold in thresholds.items():
        _check_group(report, group, expected[group]["n_items"], threshold)
    _check_raw_items(report, suite)
    return report


def _require_eval_receipts() -> dict:
    """Refuse publication unless both suites pass and bind the upload checkpoint.

    Returns the validated ``{"main": report, "holdout": report}`` pair.
    """
    expected = _expected_probe_specs()
    report_paths = {
        "main": OUT_DIR / "eval_report.json",
        "holdout": Path(
            os.environ.get(
                "MM_HOLDOUT_REPORT",
                str(OUT_DIR.parent / "full-ft" / "eval_holdout_report.json"),
            )
        ),
    }
    thresholds = {"main": MAIN_THRESHOLDS, "holdout": HOLDOUT_THRESHOLDS}
    reports: dict[str, dict] = {}
    for suite, path in report_paths.items():
        try:
            reports[suite] = _validate_report(path, suite, thresholds[suite], expected[suite])
        except GateError as exc:
            sys.exit(f"refusing upload: {exc}")

    selections = {
        suite: report["receipt"]["selection"] for suite, report in reports.items()
    }
    try:
        target = Path(WEIGHTS_DIR).resolve(strict=True)
    except OSError as exc:
        sys.exit(f"refusing upload: upload weights dir is unusable: {exc}")
    for suite, selection in selections.items():
        attested = Path(selection["model"]["path"]).resolve()
        if attested != target:
            sys.exit(
                f"refusing upload: {suite} suite attests checkpoint {attested} but the "
                f"upload plan would publish {target}"
            )
    _require_identical_selection(selections["main"], selections["holdout"])
    # Holdout's overlap check is against the training corpus, not a separate
    # benchmark dataset. Both suites must bind those same bytes. Identical
    # copies at different paths are valid (for example a staged pod bundle).
    main_dataset = reports["main"]["receipt"]["dataset"]
    holdout_dataset = reports["holdout"]["receipt"]["dataset"]
    if any(main_dataset[key] != holdout_dataset[key] for key in ("sha256", "bytes")):
        sys.exit("refusing upload: evaluation suites attest different corpus contents")
    return reports


def _require_identical_selection(main: dict, holdout: dict) -> None:
    """Both suites must have evaluated the *same* thing, base included.

    Comparing only model+tokenizer accepted two suites that ran the same LoRA
    adapter against two different base checkpoints — the adapter bytes match,
    the numbers come from two different models, and the pair reads as one
    candidate.  Kind, model, tokenizer and the whole base block (path, ref,
    scope and every base file hash) must agree.
    """
    if main.get("kind") != holdout.get("kind"):
        sys.exit(
            f"refusing upload: suites attest different checkpoint kinds "
            f"({main.get('kind')!r} vs {holdout.get('kind')!r})"
        )
    if main["model"]["files"] != holdout["model"]["files"]:
        sys.exit(
            "refusing upload: the two evaluation suites attest different checkpoint "
            "contents at the same path"
        )
    if main["tokenizer"]["files"] != holdout["tokenizer"]["files"]:
        sys.exit("refusing upload: the two evaluation suites attest different tokenizers")
    main_base = main.get("base") or {}
    holdout_base = holdout.get("base") or {}
    for field in ("path", "ref", "scope"):
        if main_base.get(field) != holdout_base.get(field):
            sys.exit(
                f"refusing upload: the two evaluation suites attest different base "
                f"checkpoints (base {field}: {main_base.get(field)!r} vs "
                f"{holdout_base.get(field)!r})"
            )
    if (main_base.get("files") or {}) != (holdout_base.get("files") or {}):
        sys.exit(
            "refusing upload: the two evaluation suites attest different base "
            "checkpoint contents — the adapter was evaluated against two different "
            "sets of base weights"
        )


# The evaluation manifests bind entire input directories. Release only the
# checkpoint payload: training logs, optimiser state and unrelated files are
# not model assets. Tokenizers may be selected from a separate base snapshot.
_TOKENIZER_NAMES = {
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "vocab.txt",
    "merges.txt", "chat_template.jinja",
}
_MODEL_NAMES = _TOKENIZER_NAMES | {
    "config.json", "generation_config.json", "model.safetensors",
    "model.safetensors.index.json", "adapter_config.json", "adapter_model.safetensors",
}


def _is_payload(name: str, *, tokenizer: bool = False) -> bool:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        return False
    if name in (_TOKENIZER_NAMES if tokenizer else _MODEL_NAMES):
        return True
    if path.parts[0] == "chat_templates" and path.suffix == ".jinja":
        return True
    return not tokenizer and len(path.parts) == 1 and name.startswith("model-") and name.endswith(".safetensors")


def _release_files(reports: dict) -> dict[str, tuple[Path, dict]]:
    selection = reports["main"]["receipt"]["selection"]
    result: dict[str, tuple[Path, dict]] = {}
    for component in ("model", "tokenizer"):
        recorded = selection[component]
        indexed_shards: set[str] = set()
        if component == "model" and "model.safetensors.index.json" in recorded["files"]:
            index_path = Path(recorded["path"]) / "model.safetensors.index.json"
            try:
                indexed_shards = set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values())
            except (OSError, ValueError, KeyError, TypeError) as exc:
                sys.exit(f"refusing upload: unusable shard index: {exc}")
            for shard in indexed_shards:
                if not isinstance(shard, str) or Path(shard).is_absolute() or ".." in Path(shard).parts:
                    sys.exit("refusing upload: unsafe shard path")
                if shard not in recorded["files"]:
                    sys.exit(f"refusing upload: missing attested shard {shard}")
        for name, metadata in recorded["files"].items():
            if name not in indexed_shards and not _is_payload(name, tokenizer=component == "tokenizer"):
                continue
            prior = result.get(name)
            if prior and any(prior[1][key] != metadata[key] for key in ("bytes", "sha256")):
                sys.exit(f"refusing upload: conflicting upload contents for {name}")
            result[name] = (Path(recorded["path"]) / name, metadata)
    return result


def _attested_manifest(reports: dict) -> dict[str, dict]:
    return {name: metadata for name, (_, metadata) in _release_files(reports).items()}


def _discover_upload_paths(reports: dict | None = None) -> list[tuple[Path, str]]:
    """Build the complete payload from validated selections, or a dry-run scan."""
    if reports is not None:
        uploads = [(path, name) for name, (path, _) in sorted(_release_files(reports).items())]
    else:
        uploads = [
            (path, path.relative_to(WEIGHTS_DIR).as_posix())
            for path in sorted(WEIGHTS_DIR.rglob("*"))
            if path.is_file() and _is_payload(path.relative_to(WEIGHTS_DIR).as_posix())
        ]
    card = OUT_DIR / "README.md"
    if card.is_file():
        uploads.append((card, "README.md"))
    gguf = OUT_DIR / "mind-mem-4b-Q4_K_M.gguf"
    if gguf.is_file():
        print(f"note: omitting {gguf.name} — derived artifacts have no converter-parent binding")
    return uploads


def _verify_upload_plan(uploads: list[tuple[Path, str]], reports: dict) -> None:
    """Require every selected payload file and an exact report-derived card."""
    from build_model_card import render_release_card
    from eval_receipt import sha256_file

    manifest = _attested_manifest(reports)
    names = [remote for _, remote in uploads]
    if len(names) != len(set(names)):
        sys.exit("refusing upload: duplicate upload destination")
    unexpected = set(names) - set(manifest) - {"README.md"}
    if unexpected:
        sys.exit(f"refusing upload: {sorted(unexpected)} not covered by the evaluation receipt")
    missing = set(manifest) | {"README.md"}
    missing -= set(names)
    if missing:
        sys.exit(f"refusing upload: missing required upload files: {sorted(missing)}")
    for local, remote in uploads:
        if remote == "README.md":
            try:
                card = local.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                sys.exit(f"refusing upload: model card unreadable: {exc}")
            if card != render_release_card(reports):
                sys.exit("refusing upload: model card does not match the validated evaluation pair; regenerate it")
            continue
        entry = manifest[remote]
        try:
            if local.stat().st_size != entry["bytes"] or sha256_file(local) != entry["sha256"]:
                sys.exit(f"refusing upload: {remote} changed between evaluation and publication")
        except OSError as exc:
            sys.exit(f"refusing upload: cannot re-read {remote}: {exc}")


@contextlib.contextmanager
def _stage_verified_uploads(uploads: list[tuple[Path, str]], reports: dict):
    """Upload private verified copies, so later source edits cannot change them."""
    with tempfile.TemporaryDirectory(prefix="mind-mem-release-") as temporary:
        staged = []
        for source, remote in uploads:
            destination = Path(temporary) / remote
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            destination.chmod(0o400)
            staged.append((destination, remote))
        _verify_upload_plan(staged, reports)
        yield staged


def _remote_release_plan(api, uploads: list[tuple[Path, str]]) -> tuple[str, list[str]]:
    """Bind replacement to a remote commit and remove obsolete checkpoint assets.

    A stale monolithic weight file can take precedence over newly uploaded
    shards. Keep unrelated repository documents; retire only recognized model
    payloads and the legacy derived GGUF that the new receipt does not attest.
    """
    parent = api.repo_info(repo_id=REPO_ID).sha
    if not parent:
        sys.exit("refusing upload: the destination has no commit identity")
    existing = api.list_repo_files(repo_id=REPO_ID, revision=parent)
    selected = {name for _, name in uploads}
    obsolete = []
    for name in existing:
        legacy_weight = (
            name in {"pytorch_model.bin", "pytorch_model.bin.index.json", "adapter_model.bin", "mind-mem-4b-Q4_K_M.gguf"}
            or ("/" not in name and name.startswith("pytorch_model-") and name.endswith(".bin"))
        )
        if name not in selected and (_is_payload(name) or legacy_weight):
            obsolete.append(name)
    return parent, sorted(obsolete)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--commit-message", default="Retrain mind-mem-4b")
    parser.add_argument(
        "--token",
        default="",
        help=(
            "HuggingFace write token; prefer HF_TOKEN in the environment, since a "
            "token on the command line is visible in the process table and shell "
            "history. The value is never printed."
        ),
    )
    args = parser.parse_args()
    token = args.token or os.environ.get("HF_TOKEN", "")

    if args.dry_run:
        uploads = _discover_upload_paths()
        print("Upload plan (UNVERIFIED — dry run does not run the evaluation gate):")
        for local, remote in uploads:
            print(f"  {local}  →  {REPO_ID}:{remote}  ({local.stat().st_size} bytes)")
        print("UNVERIFIED: rerun without --dry-run to require attested, passing evals.")
        return

    reports = _require_eval_receipts()  # gate runs before any plan is trusted
    uploads = _discover_upload_paths(reports)
    if not uploads:
        sys.exit(f"no files to upload — {WEIGHTS_DIR} is empty. Train first.")
    _verify_upload_plan(uploads, reports)

    print("Upload plan (attested):")
    for local, remote in uploads:
        print(f"  {local}  →  {REPO_ID}:{remote}  ({local.stat().st_size} bytes)")

    if not token:
        sys.exit(
            "no HF token provided. Set HF_TOKEN (write scope) for star-ga/mind-mem-4b, "
            "or pass --token."
        )

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, create_commit

    api = HfApi(token=token)
    try:
        who = api.whoami()
        role = who.get("auth", {}).get("accessToken", {}).get("role", "?")
        if role != "write":
            sys.exit(f"token role is {role!r} — need 'write' for {REPO_ID}.")
    except Exception as exc:
        sys.exit(f"token check failed: {exc}")

    parent, obsolete = _remote_release_plan(api, uploads)
    for remote in obsolete:
        print(f"  retire obsolete release payload: {remote}")
    # Snapshot after validation, then rehash the copies before the client can
    # reopen them. Replacements and stale-asset removals form one atomic commit;
    # parent_commit refuses a concurrent change to the remote branch.
    with _stage_verified_uploads(uploads, reports) as staged:
        ops = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)) for local, remote in staged]
        ops += [CommitOperationDelete(path_in_repo=name) for name in obsolete]
        result = create_commit(
            repo_id=REPO_ID,
            operations=ops,
            commit_message=args.commit_message,
            token=token,
            parent_commit=parent,
        )
    print(f"\ncommit: {result.commit_url}")
    print(f"files:  https://huggingface.co/{REPO_ID}/tree/main")


if __name__ == "__main__":
    main()
