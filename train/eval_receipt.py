"""Content-addressed receipts binding a 4B evaluation report to its inputs.

Integrity scope (read this before quoting a receipt as evidence):

* What a receipt establishes: the *recorded* bytes (model weights, tokenizer,
  base checkpoint, dataset, eval source files) were identical at capture time
  and at verification time, and the *recorded* report payload — every score,
  every raw item outcome, every count — digests to the value stored inside the
  receipt body.  Call this property **bound-to-recorded-inputs**.
* What it does NOT establish: that those bytes were the ones the loader read,
  that the evaluation actually executed, that the numbers were produced by that
  model, or that any of it originated with an authorised party.  The receipt is
  written by the same process it attests, so anyone able to write the receipt
  can write a self-consistent falsehood.  Signing does not fix this either — a
  signer can sign a false statement; it only narrows *who* could have written
  it.  Never describe this module as proof of authentic origin.

The evaluation itself lives in :mod:`eval_harness` / :mod:`eval_holdout`.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "mind-mem/eval-receipt@2"

#: Artifacts written *into* a model directory by the eval itself.  Excluded
#: only at the manifest root — a nested ``subdir/eval_report.json`` is real
#: model content and stays in the manifest.
_NON_MODEL_ARTIFACTS = {"eval_report.json", "eval_holdout_report.json"}

#: Base-checkpoint binding strengths.  Only ``full`` is release-grade; the
#: gate refuses ``index-only`` (see :func:`base_binding_is_release_ready`).
BASE_SCOPE_FULL = "full"
BASE_SCOPE_INDEX_ONLY = "index-only"


class ReceiptError(ValueError):
    """Raised when a receipt cannot be built or is structurally unusable."""


# ---------------------------------------------------------------------------
# Hashing / manifests
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _entries(root: Path) -> Iterable[tuple[str, Path]]:
    if root.is_file():
        yield root.name, root
        return
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if rel in _NON_MODEL_ARTIFACTS:
            continue  # root-level eval output only; nested files stay
        if path.is_file():  # follows symlinks by design (HF cache layout)
            yield rel, path


def file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    """Hash a file or directory, resolving symlinks to their targets.

    HuggingFace snapshot directories are trees of symlinks into the blob
    cache, so a manifest that *skipped* links would record "no weights".  We
    hash the resolved target and record the link identity, which means a
    swapped target changes the manifest.  A target outside the model root is
    normal for a cache checkout and is recorded, not rejected.
    """
    manifest: dict[str, dict[str, Any]] = {}
    root_resolved = root.resolve()
    for name, path in _entries(root):
        target = path.resolve()
        entry: dict[str, Any] = {
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
        if path.is_symlink():
            entry["link_target"] = str(target)
            entry["link_target_inside_root"] = str(target).startswith(
                str(root_resolved) + os.sep
            )
        manifest[name] = entry
    return manifest


# ---------------------------------------------------------------------------
# Source provenance
# ---------------------------------------------------------------------------


def _git(repo_root: Path, rel_paths: tuple[str, ...] = ()) -> dict[str, Any]:
    """HEAD plus a dirty flag that counts staged *and* untracked changes.

    Scoped to ``rel_paths`` (the attested source files) so an unrelated
    scratch file elsewhere in the tree does not masquerade as source truth.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", *rel_paths],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "scope": list(rel_paths)}
    return {"commit": commit, "dirty": bool(status), "scope": list(rel_paths)}


def _versions(names: tuple[str, ...]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


# ---------------------------------------------------------------------------
# Loader-selected inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSelection:
    """The exact paths a loader selected, captured *before* it loads them.

    ``kind`` is ``"full-ft"`` or ``"lora"``.  For LoRA, ``base_path`` is the
    resolved base checkpoint directory; ``base_scope`` states how much of it
    was hashed.  A selection that cannot name a resolved local base directory
    must say so (``base_path=None``) rather than imply a binding it lacks.
    """

    kind: str
    model_path: Path
    tokenizer_path: Path
    base_path: Path | None = None
    base_scope: str = BASE_SCOPE_FULL
    base_ref: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)


def _source_manifest(repo_root: Path, source_paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Hash the evaluator/loader sources that are about to be used.

    Missing files are recorded as ``present: False`` rather than silently
    dropped — a helper the runner declared but that is absent must not read as
    "nothing to attest".
    """
    source: dict[str, dict[str, Any]] = {}
    for path in sorted({Path(p).resolve() for p in source_paths}):
        try:
            rel = str(path.relative_to(repo_root))
        except ValueError as exc:  # source outside the attested repo
            raise ReceiptError(f"source path escapes repo_root: {path}") from exc
        if path.is_file():
            source[rel] = {
                "present": True,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        else:
            source[rel] = {"present": False, "bytes": None, "sha256": None}
    return source


#: Sources that define the evaluation and the loader path it uses.  Required
#: files are always attested (a missing one makes the receipt incomplete);
#: optional helpers are attested only when this checkout has them, so a helper
#: that does not exist here does not read as a missing attested file.
REQUIRED_EVAL_SOURCES = (
    "train/eval_harness.py",
    "train/eval_holdout.py",
    "train/eval_receipt.py",
    "train/build_corpus.py",
    "train/_causal_lm_import.py",
    "src/mind_mem/causal_lm_loader.py",
)
OPTIONAL_EVAL_SOURCES: tuple[str, ...] = ()


def eval_source_paths(repo_root: Path) -> tuple[Path, ...]:
    """The evaluator/loader sources to capture before the evaluation runs."""
    root = Path(repo_root).resolve()
    paths = [root / rel for rel in REQUIRED_EVAL_SOURCES]
    paths += [root / rel for rel in OPTIONAL_EVAL_SOURCES if (root / rel).is_file()]
    return tuple(paths)


def _dataset_manifest(dataset_root: Path) -> dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    present = dataset_root.is_file()
    return {
        "path": str(dataset_root),
        "present": present,
        "sha256": sha256_file(dataset_root) if present else None,
        "bytes": dataset_root.stat().st_size if present else None,
    }


def probe_definition_digests(probe_sets: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Digest each probe *definition* (the items themselves), plus its size.

    The digest is what makes a probe record non-substitutable: a run that
    swapped the questions for easier ones keeps the count and changes this.
    """
    digests: dict[str, dict[str, Any]] = {}
    for name in sorted(probe_sets):
        items = probe_sets[name]
        try:
            n_items = len(items)
        except TypeError as exc:
            raise ReceiptError(f"probe set {name!r} is not sized") from exc
        digests[name] = {"n_items": n_items, "digest": _digest(items)}
    return digests


def capture_inputs(
    selection: ModelSelection,
    *,
    repo_root: Path,
    dataset_root: Path,
    source_paths: Iterable[Path],
    probe_sets: Mapping[str, Any],
) -> dict[str, Any]:
    """Manifest every input BEFORE it is used.

    Weights, tokenizer and base are captured before the loader runs; the
    dataset, the evaluator/loader source files and the probe *definitions* are
    captured before the evaluation reads them.  Capturing the dataset and
    sources only at :func:`build_receipt` time (i.e. after the run) would leave
    the interval in which they matter unattested.

    This is an equality check between two endpoints, capture and finalization.
    It does not establish that the bytes were unchanged *throughout* the window
    between them.
    """
    repo_root = Path(repo_root).resolve()
    captured: dict[str, Any] = {
        "kind": selection.kind,
        "model": {
            "path": str(Path(selection.model_path).resolve()),
            "files": file_manifest(Path(selection.model_path)),
        },
        "tokenizer": {
            "path": str(Path(selection.tokenizer_path).resolve()),
            "files": file_manifest(Path(selection.tokenizer_path)),
        },
        "base": None,
        "dataset": _dataset_manifest(dataset_root),
        "source": {
            "repo_root": str(repo_root),
            "files": _source_manifest(repo_root, source_paths),
        },
        "probes": probe_definition_digests(probe_sets),
    }
    if selection.kind == "lora":
        base = {
            "path": str(Path(selection.base_path).resolve()) if selection.base_path else None,
            "ref": selection.base_ref,
            "scope": selection.base_scope,
            "files": (
                file_manifest(Path(selection.base_path))
                if selection.base_path and selection.base_scope == BASE_SCOPE_FULL
                else {}
            ),
        }
        captured["base"] = base
    if selection.notes:
        captured["notes"] = dict(selection.notes)
    return captured


def _recapture(captured: Mapping[str, Any]) -> dict[str, Any]:
    """Re-read every captured path for the verify-after half of the check."""
    again = json.loads(json.dumps(captured))
    for key in ("model", "tokenizer"):
        again[key]["files"] = file_manifest(Path(again[key]["path"]))
    base = again.get("base")
    if base and base.get("path") and base.get("scope") == BASE_SCOPE_FULL:
        base["files"] = file_manifest(Path(base["path"]))
    again["dataset"] = _dataset_manifest(Path(again["dataset"]["path"]))
    source = again["source"]
    repo_root = Path(source["repo_root"])
    source["files"] = _source_manifest(
        repo_root, [repo_root / rel for rel in sorted(source["files"])]
    )
    return again


# ---------------------------------------------------------------------------
# Receipt construction
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    return str(uuid.uuid4())


def _exact_count(value: Any, label: str) -> int:
    """A count must already BE an integer.  ``int()`` would launder ``3.9``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReceiptError(f"{label} must be a non-bool integer, got {value!r}")
    if value < 0:
        raise ReceiptError(f"{label} must not be negative, got {value!r}")
    return value


def build_receipt(
    *,
    repo_root: Path,
    suite: str,
    captured: Mapping[str, Any],
    probe_counts: Mapping[str, tuple[int, int]],
    probe_sets: Mapping[str, Any],
    command: str,
    run_id: str,
    started_at: str,
    ended_at: str,
    status: str,
) -> dict[str, Any]:
    """Build a receipt from inputs captured before they were used.

    Dataset, source and probe definitions come from ``captured`` (taken before
    the evaluation read them) and are re-read here; a mutation in between makes
    the receipt incomplete.  ``probe_counts`` maps group name ->
    ``(n_items, n_completed)``; a crashed or truncated run reports
    ``n_completed < n_items`` and is therefore visibly incomplete instead of
    indistinguishable from a finished one.  ``status`` is ``"completed"`` for a
    run that executed every probe.
    """
    repo_root = repo_root.resolve()
    for key in ("dataset", "source", "probes"):
        if key not in captured:
            raise ReceiptError(
                f"captured inputs lack {key!r}: capture_inputs() must run before the "
                "evaluation, not after it"
            )

    verified = _recapture(captured)
    capture_verified = verified == json.loads(json.dumps(dict(captured)))

    captured_probes = captured["probes"]
    finalization = probe_definition_digests(probe_sets)
    probes_unchanged = finalization == json.loads(json.dumps(dict(captured_probes)))

    probes: dict[str, dict[str, Any]] = {}
    for name in sorted(probe_counts):
        n_items, n_completed = probe_counts[name]
        defined = captured_probes.get(name) or {}
        probes[name] = {
            "n_items": _exact_count(n_items, f"probes.{name}.n_items"),
            "n_completed": _exact_count(n_completed, f"probes.{name}.n_completed"),
            "digest": defined.get("digest"),
            "defined_n_items": defined.get("n_items"),
        }
    counts_match_definitions = all(
        rec["digest"] is not None and rec["n_items"] == rec["defined_n_items"]
        for rec in probes.values()
    ) and set(probes) == set(captured_probes)

    dataset = json.loads(json.dumps(captured["dataset"]))
    source_files = json.loads(json.dumps(captured["source"]["files"]))
    source_present = {rel: m for rel, m in source_files.items() if m.get("present")}

    model_block = {
        k: json.loads(json.dumps(v))
        for k, v in captured.items()
        if k not in ("dataset", "source", "probes")
    }
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "suite": suite,
        "command": command,
        "run": {
            "run_id": run_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "status": status,
        },
        "source": {
            "repo_root": str(repo_root),
            "git": _git(repo_root, tuple(sorted(source_files))),
            "files": source_files,
        },
        "selection": model_block,
        "capture_verified": bool(capture_verified),
        "probes_match_definitions": bool(probes_unchanged and counts_match_definitions),
        "dataset": dataset,
        "probes": probes,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _versions(("torch", "transformers", "peft", "bitsandbytes")),
        },
        "report_sha256": None,  # filled by finalize_report
        "integrity_scope": (
            "bound-to-recorded-inputs: recorded bytes and recorded report "
            "payload are self-consistent; this is not proof of execution or "
            "of authentic origin"
        ),
    }
    receipt["complete"] = bool(
        receipt["source"]["git"]["commit"]
        and source_present
        and len(source_present) == len(source_files)
        and model_block["model"]["files"]
        and probes_unchanged
        and counts_match_definitions
        and dataset["present"]
        and probes
        and capture_verified
        and status == "completed"
        and all(p["n_items"] > 0 and p["n_completed"] == p["n_items"] for p in probes.values())
    )
    return receipt


def report_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    """The canonical digest domain: the whole report except the receipt."""
    return {k: v for k, v in report.items() if k != "receipt"}


def finalize_report(report: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    """Bind the entire report payload into the receipt, then seal the receipt.

    Every result, raw item outcome and count in ``report`` enters the digest
    domain, so editing a score after the fact invalidates the receipt.
    """
    sealed = dict(receipt)
    sealed["report_sha256"] = _digest(report_payload(report))
    sealed["receipt_sha256"] = _digest(
        {k: v for k, v in sealed.items() if k != "receipt_sha256"}
    )
    return {**report, "receipt": sealed}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def require_complete(receipt: Mapping[str, Any]) -> None:
    if not receipt.get("complete"):
        raise ReceiptError(
            "evaluation receipt is incomplete: model/source/dataset/probe binding is required"
        )


def receipt_is_valid(receipt: Mapping[str, Any]) -> bool:
    """Check the receipt's self-digest before it is used as a release gate."""
    if not isinstance(receipt, Mapping):
        return False
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str):
        return False
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    return claimed == _digest(body)


def report_matches_receipt(report: Mapping[str, Any]) -> bool:
    """The scores a human reads must be the scores the receipt digested."""
    try:
        claimed = report["receipt"]["report_sha256"]
    except (KeyError, TypeError):
        return False
    if not isinstance(claimed, str):
        return False
    try:
        return claimed == _digest(report_payload(report))
    except (TypeError, ValueError):
        return False


def bindings_match(receipt: Mapping[str, Any]) -> tuple[bool, str]:
    """Re-hash every recorded path.  Returns ``(ok, reason)``.

    ``reason`` distinguishes "verification failed" from "verification could
    not run" so an operator reading the refusal does not reach for an
    override when the real problem is an unreadable path.
    """
    try:
        source = receipt["source"]
        repo_root = Path(source["repo_root"]).resolve()
        scope = tuple(source["git"].get("scope") or ())
        if _git(repo_root, scope) != source["git"]:
            return False, "repository state changed since the receipt was written"
        for rel, meta in source["files"].items():
            path = repo_root / rel
            if not path.is_file():
                return False, f"attested source file is missing: {rel}"
            if sha256_file(path) != meta["sha256"] or path.stat().st_size != meta["bytes"]:
                return False, f"attested source file changed: {rel}"
        dataset = receipt["dataset"]
        dataset_path = Path(dataset["path"])
        if not dataset.get("present") or not dataset_path.is_file():
            return False, "attested dataset is missing"
        if (
            sha256_file(dataset_path) != dataset["sha256"]
            or dataset_path.stat().st_size != dataset["bytes"]
        ):
            return False, "attested dataset changed"
        selection = receipt["selection"]
        for key in ("model", "tokenizer"):
            recorded = selection[key]
            if file_manifest(Path(recorded["path"])) != recorded["files"]:
                return False, f"attested {key} bytes changed since evaluation"
        base = selection.get("base")
        if base and base.get("path") and base.get("scope") == BASE_SCOPE_FULL:
            if file_manifest(Path(base["path"])) != base["files"]:
                return False, "attested base checkpoint bytes changed since evaluation"
        if not receipt.get("capture_verified"):
            return False, "inputs changed between capture and finalization"
        if not receipt.get("probes_match_definitions"):
            return False, (
                "probe records do not match the probe definitions captured before "
                "the run (substituted or resized probe set)"
            )
        return True, "ok"
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"receipt is structurally unusable: {exc}"
    except OSError as exc:
        return False, f"verification could not run (not a failed check): {exc}"


#: Weight-set layouts a checkpoint may present.  A stray ``extras/foo.bin``
#: is not one of them — "some file ends in .bin" was accepted as a base
#: manifest before, which let an unrelated blob stand in for the weights.
_SINGLE_WEIGHTS = ("model.safetensors", "pytorch_model.bin")
_WEIGHT_INDEXES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


def weight_set_is_closed(
    root: Path, files: Mapping[str, Any], label: str
) -> tuple[bool, str]:
    """A named single weight file, or an index whose every shard is manifested.

    ``root`` is the directory the manifest was taken from; the index is read
    from disk because the shard list lives inside it, and every file it names
    must appear in ``files`` (closure), otherwise the manifest attests only
    part of the weights.
    """
    if any(name in files for name in _SINGLE_WEIGHTS):
        return True, "ok"
    index_name = next((name for name in _WEIGHT_INDEXES if name in files), None)
    if index_name is None:
        return False, (
            f"{label} manifest has no recognised weight file "
            f"(expected one of {list(_SINGLE_WEIGHTS)} or a shard index)"
        )
    try:
        index = json.loads((Path(root) / index_name).read_text(encoding="utf-8"))
        shards = sorted(set((index.get("weight_map") or {}).values()))
    except (OSError, ValueError) as exc:
        return False, f"{label} shard index is unreadable: {exc}"
    if not shards:
        return False, f"{label} shard index lists no shards"
    absent = [s for s in shards if s not in files]
    if absent:
        return False, f"{label} shard index lists files absent from the manifest: {absent}"
    return True, "ok"


def _adapter_base_binding(receipt: Mapping[str, Any]) -> tuple[bool, str]:
    """``adapter_config.json`` must name the base the selection recorded.

    An adapter whose config points at a different base than the one the
    receipt attests is either mis-recorded or evaluated against something else;
    either way it is not publishable.
    """
    selection = receipt.get("selection") or {}
    model = selection.get("model") or {}
    base = selection.get("base") or {}
    config_path = Path(model.get("path", "")) / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"adapter_config.json is unreadable: {exc}"
    declared = config.get("base_model_name_or_path")
    if not isinstance(declared, str) or not declared:
        return False, "adapter_config.json declares no base_model_name_or_path"
    candidates = {c for c in (base.get("path"), base.get("ref")) if c}
    resolved = {str(Path(c).resolve()) for c in candidates if Path(c).exists()}
    declared_resolved = str(Path(declared).resolve()) if Path(declared).exists() else declared
    if declared in candidates or declared_resolved in (candidates | resolved):
        return True, "ok"
    return False, (
        f"adapter_config.json binds base {declared!r} but the receipt attests "
        f"{sorted(candidates)} — the adapter was not evaluated against the "
        "recorded base"
    )


def base_binding_is_release_ready(receipt: Mapping[str, Any]) -> tuple[bool, str]:
    """A LoRA candidate needs a fully manifested, resolved base checkpoint."""
    selection = receipt.get("selection") or {}
    if selection.get("kind") != "lora":
        return True, "ok"
    base = selection.get("base") or {}
    if not base.get("path"):
        return False, (
            "adapter is not bound to a base checkpoint: no resolved local base "
            "directory was recorded — re-run the eval with MM_BASE_MODEL pointed "
            "at a local snapshot"
        )
    if base.get("scope") != BASE_SCOPE_FULL:
        return False, (
            f"base binding scope is {base.get('scope')!r}; index-only binding is "
            "not release-ready — a full base manifest is required"
        )
    files = base.get("files") or {}
    if "config.json" not in files:
        return False, "base checkpoint manifest is insufficient (missing ['config.json'])"
    ok, reason = weight_set_is_closed(Path(base["path"]), files, "base checkpoint")
    if not ok:
        return False, reason
    if "tokenizer_config.json" not in files or not any(
        n in files for n in ("tokenizer.json", "tokenizer.model")
    ):
        return False, "base checkpoint manifest carries no tokenizer files"
    return _adapter_base_binding(receipt)
