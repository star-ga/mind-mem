"""Publication-gate controls for the 4B evaluation attestation.

These tests drive the *actual* uploader validator
(:func:`train.upload_to_hf._require_eval_receipts`) — no re-implementation of
the checks, no HuggingFace API calls (the hub import lives inside ``main`` and
is never reached).  The four negative controls that rejected the previous
candidate are included verbatim in intent: failing scores, a receipt for a
different checkpoint than the one uploaded, scores edited after the fact, and a
config-only checkpoint with no probes.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "train"


@contextlib.contextmanager
def _train_on_path():
    """Put ``train/`` on ``sys.path`` only for the duration of the block.

    The gate's own runtime imports (``import eval_receipt`` inside a function)
    resolve through ``sys.modules`` once these modules are loaded, so the path
    entry does not have to stay — leaving it there would change import
    resolution for every other test in the suite.
    """
    original = list(sys.path)
    sys.path.insert(0, str(TRAIN))
    try:
        yield
    finally:
        sys.path[:] = original


with _train_on_path():
    H = importlib.import_module("eval_harness")
    HO = importlib.import_module("eval_holdout")
    E = importlib.import_module("eval_receipt")
    U = importlib.import_module("upload_to_hf")
    B = importlib.import_module("build_model_card")

PROBE_SETS = U.probe_sets()
EXPECTED = U._expected_probe_counts()
SOURCE_PATHS = E.eval_source_paths(REPO)


# ---------------------------------------------------------------------------
# Fixtures — a valid, same-checkpoint, two-suite pair with fixed probe counts
# ---------------------------------------------------------------------------


def _checkpoint(root: Path, payload: bytes = b"weights-A", *, sharded: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    if sharded:
        (root / "model-00001-of-00001.safetensors").write_bytes(payload)
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"w": "model-00001-of-00001.safetensors"}}),
            encoding="utf-8",
        )
    else:
        (root / "model.safetensors").write_bytes(payload)
    return root


def _dataset(path: Path) -> Path:
    path.write_text('{"messages":[]}\n', encoding="utf-8")
    return path


def _bench(total: int, hits: int | None = None) -> dict:
    hits = total if hits is None else hits
    return {"hits": hits, "total": total, "accuracy": hits / total, "misses": []}


def _suite_report(
    suite: str,
    model: Path,
    dataset: Path,
    *,
    hits: dict[str, int] | None = None,
    captured: dict | None = None,
    selection: E.ModelSelection | None = None,
    probe_groups: dict[str, list] | None = None,
    completed: dict[str, int] | None = None,
) -> dict:
    # The positive fixture uses the *actual* probe definitions, so the digests
    # in the receipt are the ones the gate recomputes from the harness.  A
    # placeholder set (``{group: [group]}``) would keep the counts right and
    # make every digest check pass vacuously.
    probe_groups = PROBE_SETS[suite] if probe_groups is None else probe_groups
    counts = {group: len(items) for group, items in probe_groups.items()}
    completed = completed or {}
    selection = selection or E.ModelSelection(kind="full-ft", model_path=model, tokenizer_path=model)
    if captured is None:
        captured = E.capture_inputs(
            selection,
            repo_root=REPO,
            dataset_root=dataset,
            source_paths=SOURCE_PATHS,
            probe_sets=probe_groups,
        )
    hits = hits or {}
    report = {}
    for group, probes in probe_groups.items():
        bench = _bench(len(probes), hits.get(group))
        bench["items"] = []
        for index, probe in enumerate(probes):
            tokens = [probe[1]] if isinstance(probe[1], str) else probe[1]
            response = " ".join(tokens) if index < bench["hits"] else ""
            bench["items"].append({"index": index, "prompt": probe[0], "response": response, "passed": index < bench["hits"]})
        report[group] = bench
    receipt = E.build_receipt(
        repo_root=REPO,
        suite=suite,
        captured=captured,
        probe_counts={g: (t, completed.get(g, t)) for g, t in counts.items()},
        probe_sets=probe_groups,
        command=f"pytest::{suite}",
        run_id=E.new_run_id(),
        started_at=E.now_iso(),
        ended_at=E.now_iso(),
        status="completed",
    )
    return E.finalize_report(report, receipt)


def _capture(
    selection: E.ModelSelection,
    dataset: Path,
    *,
    suite: str = "main",
    probe_groups: dict[str, list] | None = None,
) -> dict:
    return E.capture_inputs(
        selection,
        repo_root=REPO,
        dataset_root=dataset,
        source_paths=SOURCE_PATHS,
        probe_sets=PROBE_SETS[suite] if probe_groups is None else probe_groups,
    )


def _write(path: Path, report: dict) -> Path:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """A valid publishable candidate: two suites, one checkpoint."""
    model = _checkpoint(tmp_path / "full-ft")
    dataset = _dataset(tmp_path / "corpus.jsonl")
    main_path = _write(tmp_path / "eval_report.json", _suite_report("main", model, dataset))
    holdout_path = _write(tmp_path / "eval_holdout_report.json", _suite_report("holdout", model, dataset))
    monkeypatch.setattr(U, "OUT_DIR", tmp_path)
    monkeypatch.setattr(U, "WEIGHTS_DIR", model)
    monkeypatch.setenv("MM_HOLDOUT_REPORT", str(holdout_path))
    (tmp_path / "README.md").write_text(B.render_release_card(U._require_eval_receipts()), encoding="utf-8")
    return {
        "tmp": tmp_path,
        "model": model,
        "dataset": dataset,
        "main": main_path,
        "holdout": holdout_path,
    }


def _reject(match: str) -> "pytest.RaisesContext":
    return pytest.raises(SystemExit, match=match)


# ---------------------------------------------------------------------------
# Positive control
# ---------------------------------------------------------------------------


def test_valid_two_suite_same_checkpoint_is_accepted(workspace) -> None:
    reports = U._require_eval_receipts()
    assert set(reports) == {"main", "holdout"}
    assert reports["main"]["receipt"]["suite"] == "main"
    # Fixed counts came from the harness probe lists, not from the report.
    assert reports["main"]["tool_call"]["total"] == len(H.TOOL_CALL_QUESTIONS)
    assert reports["holdout"]["v4_holdout"]["total"] == len(HO.V4_HOLDOUT)


def test_suites_cannot_attest_different_training_corpora(workspace) -> None:
    other = workspace["tmp"] / "other-corpus.jsonl"
    other.write_text('{"messages":["different training material"]}\n')
    _write(workspace["holdout"], _suite_report("holdout", workspace["model"], other))
    with _reject("different corpus contents"):
        U._require_eval_receipts()


def test_identical_corpus_bytes_at_another_path_are_accepted(workspace) -> None:
    other = workspace["tmp"] / "copied-corpus.jsonl"
    other.write_bytes(workspace["dataset"].read_bytes())
    _write(workspace["holdout"], _suite_report("holdout", workspace["model"], other))
    assert set(U._require_eval_receipts()) == {"main", "holdout"}


def test_verified_staging_survives_later_source_replacement(workspace) -> None:
    reports = U._require_eval_receipts()
    uploads = U._discover_upload_paths(reports)
    with U._stage_verified_uploads(uploads, reports) as staged:
        (workspace["model"] / "model.safetensors").write_bytes(b"later source edit")
        staged_model = next(path for path, name in staged if name == "model.safetensors")
        assert staged_model.read_bytes() == b"weights-A"
        U._verify_upload_plan(staged, reports)
    assert not staged_model.exists()


def test_staging_rejects_source_changed_before_copy(workspace) -> None:
    reports = U._require_eval_receipts()
    uploads = U._discover_upload_paths(reports)
    (workspace["model"] / "model.safetensors").write_bytes(b"changed before staging")
    with _reject("changed between evaluation and publication"), U._stage_verified_uploads(uploads, reports):
        pytest.fail("unverified copied bytes reached publication")


def test_publication_atomically_replaces_stale_weights_from_pinned_parent(workspace, monkeypatch) -> None:
    """Drive the real uploader against a fake hub; no network or credentials."""
    import types

    model = workspace["model"]
    (model / "model.safetensors").rename(model / "model-00001-of-00001.safetensors")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "model-00001-of-00001.safetensors"}}))
    for suite in ("main", "holdout"):
        _write(workspace[suite], _suite_report(suite, model, workspace["dataset"]))
    reports = U._require_eval_receipts()
    (workspace["tmp"] / "README.md").write_text(B.render_release_card(reports))
    remote = {
        "model.safetensors": b"old monolithic model that would mask new shards",
        "adapter_model.safetensors": b"old adapter",
        "mind-mem-4b-Q4_K_M.gguf": b"unattested old conversion",
        "NOTICE": b"preserve unrelated repository document",
    }
    parent = "a" * 40
    events = []

    class FakeApi:
        def __init__(self, token):
            assert token == "fixture-only-token"

        def whoami(self):
            return {"auth": {"accessToken": {"role": "write"}}}

        def repo_info(self, *, repo_id):
            return types.SimpleNamespace(sha=parent)

        def list_repo_files(self, *, repo_id, revision):
            assert revision == parent
            return list(remote)

    def create_commit(**kwargs):
        assert kwargs["parent_commit"] == parent
        # The source can change after the client's operation preparation; the
        # committed byte stream must still come from private verified copies.
        (model / "model-00001-of-00001.safetensors").write_bytes(b"source changed during upload")
        for operation in kwargs["operations"]:
            events.append(operation.kind)
            if operation.kind == "add":
                remote[operation.path_in_repo] = Path(operation.path_or_fileobj).read_bytes()
            else:
                del remote[operation.path_in_repo]
        return types.SimpleNamespace(commit_url="https://example.invalid/fixture-commit")

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = FakeApi
    fake_hub.CommitOperationAdd = lambda **kw: types.SimpleNamespace(kind="add", **kw)
    fake_hub.CommitOperationDelete = lambda **kw: types.SimpleNamespace(kind="delete", **kw)
    fake_hub.create_commit = create_commit
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setenv("HF_TOKEN", "fixture-only-token")
    monkeypatch.setattr(sys, "argv", ["upload_to_hf.py"])
    U.main()
    assert "add" in events and "delete" in events
    assert remote["model-00001-of-00001.safetensors"] == b"weights-A"
    assert "model.safetensors" not in remote
    assert "adapter_model.safetensors" not in remote
    assert "mind-mem-4b-Q4_K_M.gguf" not in remote
    assert remote["NOTICE"] == b"preserve unrelated repository document"


def test_upload_plan_verifies_against_the_attestation(workspace) -> None:
    reports = U._require_eval_receipts()
    uploads = U._discover_upload_paths()
    U._verify_upload_plan(uploads, reports)  # must not raise
    assert "model.safetensors" in {remote for _, remote in uploads}


@pytest.mark.parametrize("omitted", ["model.safetensors", "config.json", "tokenizer.json"])
def test_release_plan_cannot_omit_attested_payload(workspace, omitted) -> None:
    reports = U._require_eval_receipts()
    uploads = [(p, name) for p, name in U._discover_upload_paths() if name != omitted]
    with _reject("missing required upload"):
        U._verify_upload_plan(uploads, reports)


def test_release_plan_cannot_duplicate_remote_names(workspace) -> None:
    reports = U._require_eval_receipts()
    uploads = U._discover_upload_paths()
    with _reject("duplicate upload"):
        U._verify_upload_plan(uploads + uploads[:1], reports)


def test_sentencepiece_tokenizer_is_in_upload_plan(workspace) -> None:
    (workspace["model"] / "tokenizer.json").unlink()
    (workspace["model"] / "tokenizer.model").write_bytes(b"sentencepiece fixture")
    for suite in ("main", "holdout"):
        _write(workspace[suite], _suite_report(suite, workspace["model"], workspace["dataset"]))
    reports = U._require_eval_receipts()
    (workspace["tmp"] / "README.md").write_text(B.render_release_card(reports), encoding="utf-8")
    uploads = U._discover_upload_paths(reports)
    assert "tokenizer.model" in {name for _, name in uploads}
    U._verify_upload_plan(uploads, reports)


def test_conflicting_model_and_tokenizer_paths_are_refused(workspace) -> None:
    reports = U._require_eval_receipts()
    # Distinct roots can legitimately overlap names, but conflicting contents
    # cannot both be published to the same repository path.
    reports["main"]["receipt"]["selection"]["tokenizer"]["files"]["tokenizer.json"] = {
        "sha256": "f" * 64,
        "bytes": 3,
    }
    with _reject("conflicting upload"):
        U._attested_manifest(reports)


def test_arbitrary_card_with_valid_digest_is_refused(workspace) -> None:
    reports = U._require_eval_receipts()
    digest = reports["main"]["receipt"]["report_sha256"]
    (workspace["tmp"] / "README.md").write_text(f"Accuracy: 100000%.\n{digest}\n")
    with _reject("model card"):
        U._verify_upload_plan(U._discover_upload_paths(), reports)


def test_indexed_shards_with_nonstandard_names_are_uploaded(workspace) -> None:
    model = workspace["model"]
    (model / "model.safetensors").rename(model / "text-weights.safetensors")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "text-weights.safetensors"}}))
    for suite in ("main", "holdout"):
        _write(workspace[suite], _suite_report(suite, model, workspace["dataset"]))
    reports = U._require_eval_receipts()
    (workspace["tmp"] / "README.md").write_text(B.render_release_card(reports))
    uploads = U._discover_upload_paths(reports)
    assert "text-weights.safetensors" in {name for _, name in uploads}
    U._verify_upload_plan(uploads, reports)
    with _reject("missing required upload"):
        U._verify_upload_plan([(p, n) for p, n in uploads if n != "text-weights.safetensors"], reports)


def test_separate_tokenizer_snapshot_supplies_the_uploaded_tokenizer(workspace) -> None:
    model = workspace["model"]
    tokenizer = workspace["tmp"] / "selected-tokenizer"
    tokenizer.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (model / name).rename(tokenizer / name)
    (tokenizer / "added_tokens.json").write_text('{"special": 4}')
    selection = E.ModelSelection(kind="full-ft", model_path=model, tokenizer_path=tokenizer)
    for suite in ("main", "holdout"):
        _write(workspace[suite], _suite_report(suite, model, workspace["dataset"], selection=selection))
    reports = U._require_eval_receipts()
    (workspace["tmp"] / "README.md").write_text(B.render_release_card(reports))
    uploads = U._discover_upload_paths(reports)
    assert (tokenizer / "added_tokens.json", "added_tokens.json") in uploads
    U._verify_upload_plan(uploads, reports)


def test_loader_source_is_required_and_drift_is_detected(tmp_path) -> None:
    import shutil

    root = tmp_path / "source"
    for source in SOURCE_PATHS:
        destination = root / source.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    source_paths = E.eval_source_paths(root)
    loader = root / "src/mind_mem/causal_lm_loader.py"
    assert loader in source_paths
    model = _checkpoint(tmp_path / "model")
    captured = E.capture_inputs(
        E.ModelSelection(kind="full-ft", model_path=model, tokenizer_path=model),
        repo_root=root,
        dataset_root=_dataset(tmp_path / "corpus.jsonl"),
        source_paths=source_paths,
        probe_sets={"g": [1]},
    )
    assert captured["source"]["files"]["src/mind_mem/causal_lm_loader.py"]["present"]
    loader.write_text("# replaced loader\n")
    receipt = E.build_receipt(
        repo_root=root,
        suite="main",
        captured=captured,
        probe_counts={"g": (1, 1)},
        probe_sets={"g": [1]},
        command="test",
        run_id=E.new_run_id(),
        started_at=E.now_iso(),
        ended_at=E.now_iso(),
        status="completed",
    )
    assert receipt["capture_verified"] is False


@pytest.mark.parametrize("sharded", [False, True])
def test_unsupported_bin_fullft_cannot_pass_publication(workspace, sharded) -> None:
    model = workspace["model"]
    name = "pytorch_model-00001-of-00001.bin" if sharded else "pytorch_model.bin"
    (model / "model.safetensors").rename(model / name)
    if sharded:
        (model / "pytorch_model.bin.index.json").write_text(json.dumps({"weight_map": {"w": name}}))
    for suite in ("main", "holdout"):
        _write(workspace[suite], _suite_report(suite, model, workspace["dataset"]))
    with _reject("full-FT release requires safetensors"):
        U._require_eval_receipts()


# ---------------------------------------------------------------------------
# The four controls that rejected the previous candidate
# ---------------------------------------------------------------------------


def test_control_failed_scores_are_refused(workspace) -> None:
    report = _suite_report("main", workspace["model"], workspace["dataset"], hits={"tool_call": 0})
    _write(workspace["main"], report)
    with _reject("below the fixed threshold"):
        U._require_eval_receipts()


def test_control_receipt_for_a_different_checkpoint_than_uploaded(workspace, monkeypatch) -> None:
    other = _checkpoint(workspace["tmp"] / "other", payload=b"weights-B")
    monkeypatch.setattr(U, "WEIGHTS_DIR", other)
    with _reject("attests checkpoint"):
        U._require_eval_receipts()


def test_control_editing_report_scores_after_evaluation(workspace) -> None:
    report = json.loads(workspace["main"].read_text(encoding="utf-8"))
    report["tool_call"] = _bench(report["tool_call"]["total"])
    report["workflow"]["hits"] = 0  # the human-readable score, receipt untouched
    _write(workspace["main"], report)
    with _reject("does not match the digest inside its receipt"):
        U._require_eval_receipts()


def test_control_config_only_checkpoint_with_no_probes_is_incomplete(tmp_path) -> None:
    lonely = tmp_path / "config-only"
    lonely.mkdir()
    (lonely / "config.json").write_text("{}\n", encoding="utf-8")
    selection = E.ModelSelection(kind="full-ft", model_path=lonely, tokenizer_path=lonely)
    receipt = E.build_receipt(
        repo_root=REPO,
        suite="main",
        captured=E.capture_inputs(
            selection,
            repo_root=REPO,
            dataset_root=_dataset(tmp_path / "corpus.jsonl"),
            source_paths=SOURCE_PATHS,
            probe_sets={},
        ),
        probe_counts={},
        probe_sets={},
        command="pytest",
        run_id=E.new_run_id(),
        started_at=E.now_iso(),
        ended_at=E.now_iso(),
        status="completed",
    )
    assert receipt["complete"] is False


# ---------------------------------------------------------------------------
# Structural / numeric integrity
# ---------------------------------------------------------------------------


def test_missing_probe_group_is_refused(workspace) -> None:
    groups = dict(PROBE_SETS["main"])
    groups.pop("v4_surfaces")
    _write(
        workspace["main"],
        _suite_report("main", workspace["model"], workspace["dataset"], probe_groups=groups),
    )
    with _reject("probe groups mismatch"):
        U._require_eval_receipts()


def test_count_inconsistent_with_harness_is_refused(workspace) -> None:
    groups = dict(PROBE_SETS["main"])
    groups["tool_call"] = list(groups["tool_call"])[:-1]  # a truncated probe set
    _write(
        workspace["main"],
        _suite_report("main", workspace["model"], workspace["dataset"], probe_groups=groups),
    )
    with _reject("but the harness defines"):
        U._require_eval_receipts()


def test_partial_run_status_is_refused(workspace) -> None:
    report = json.loads(workspace["main"].read_text(encoding="utf-8"))
    receipt = report["receipt"]
    receipt["probes"]["tool_call"]["n_completed"] -= 1
    report = E.finalize_report(E.report_payload(report), receipt)
    _write(workspace["main"], report)
    with _reject("run did not finish"):
        U._require_eval_receipts()


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("hits", True, r"tool_call\.hits must be a non-bool integer"),
        ("total", False, r"tool_call\.total must be a non-bool integer"),
        ("accuracy", float("nan"), r"tool_call\.accuracy must be finite"),
        ("hits", 1.5, r"tool_call\.hits must be a non-bool integer"),
    ],
)
def test_bool_and_nan_numerics_are_refused(workspace, field, value, reason) -> None:
    """Each control must be refused for its own reason, not incidentally."""
    report = json.loads(workspace["main"].read_text(encoding="utf-8"))
    payload = E.report_payload(report)
    payload["tool_call"][field] = value
    # Re-seal so the digest matches: the numeric checks, not the tamper check,
    # are what must reject these.
    _write(workspace["main"], E.finalize_report(payload, dict(report["receipt"])))
    with _reject(reason):
        U._require_eval_receipts()


def test_hits_exceeding_total_is_refused(workspace) -> None:
    report = json.loads(workspace["main"].read_text(encoding="utf-8"))
    payload = E.report_payload(report)
    total = payload["tool_call"]["total"]
    payload["tool_call"] = {"hits": total + 1, "total": total, "accuracy": 1.0, "misses": []}
    _write(workspace["main"], E.finalize_report(payload, dict(report["receipt"])))
    with _reject("out of range"):
        U._require_eval_receipts()


def test_receipt_digest_tamper_is_refused(workspace) -> None:
    report = json.loads(workspace["main"].read_text(encoding="utf-8"))
    report["receipt"]["command"] = "python3 train/eval_harness.py --totally-fine"
    _write(workspace["main"], report)
    with _reject("fails its own digest"):
        U._require_eval_receipts()


# ---------------------------------------------------------------------------
# Checkpoint binding
# ---------------------------------------------------------------------------


def test_two_suites_on_different_checkpoints_are_refused(workspace) -> None:
    other = _checkpoint(workspace["tmp"] / "b", payload=b"weights-B")
    _write(workspace["holdout"], _suite_report("holdout", other, workspace["dataset"]))
    with _reject("attests checkpoint"):
        U._require_eval_receipts()


def test_model_mutated_after_evaluation_is_refused(workspace) -> None:
    (workspace["model"] / "model.safetensors").write_bytes(b"swapped-after-eval")
    with _reject("bytes changed since evaluation"):
        U._require_eval_receipts()


def test_model_mutated_between_capture_and_finalization_is_refused(workspace) -> None:
    model, dataset = workspace["model"], workspace["dataset"]
    selection = E.ModelSelection(kind="full-ft", model_path=model, tokenizer_path=model)
    captured = _capture(selection, dataset)
    (model / "model.safetensors").write_bytes(b"swapped-mid-run")
    report = _suite_report("main", model, dataset, captured=captured, selection=selection)
    assert report["receipt"]["capture_verified"] is False
    _write(workspace["main"], report)
    with _reject("incomplete"):
        U._require_eval_receipts()


def test_symlinked_checkpoint_is_supported_and_target_swap_is_detected(tmp_path, monkeypatch) -> None:
    blobs = _checkpoint(tmp_path / "cache")  # HF-style blob store
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
        (snapshot / name).symlink_to(blobs / name)
    dataset = _dataset(tmp_path / "corpus.jsonl")
    main = _write(tmp_path / "eval_report.json", _suite_report("main", snapshot, dataset))
    holdout = _write(tmp_path / "eval_holdout_report.json", _suite_report("holdout", snapshot, dataset))
    monkeypatch.setattr(U, "OUT_DIR", tmp_path)
    monkeypatch.setattr(U, "WEIGHTS_DIR", snapshot)
    monkeypatch.setenv("MM_HOLDOUT_REPORT", str(holdout))
    receipt = json.loads(main.read_text(encoding="utf-8"))["receipt"]
    entry = receipt["selection"]["model"]["files"]["model.safetensors"]
    assert entry["link_target"].startswith(str(blobs))  # link identity recorded
    assert entry["link_target_inside_root"] is False  # cache target is allowed
    U._require_eval_receipts()  # symlinked checkpoints stay publishable

    (blobs / "model.safetensors").write_bytes(b"swapped-target")
    with _reject("bytes changed since evaluation"):
        U._require_eval_receipts()


# ---------------------------------------------------------------------------
# Layout completeness
# ---------------------------------------------------------------------------


def test_checkpoint_without_weights_is_refused(workspace) -> None:
    model, dataset = workspace["model"], workspace["dataset"]
    (model / "model.safetensors").unlink()
    _write(workspace["main"], _suite_report("main", model, dataset))
    _write(workspace["holdout"], _suite_report("holdout", model, dataset))
    with _reject("has no recognised weight file"):
        U._require_eval_receipts()


def test_shard_index_missing_a_listed_shard_is_refused(tmp_path, monkeypatch) -> None:
    model = _checkpoint(tmp_path / "full-ft", sharded=True)
    (model / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00001.safetensors",
                    "b": "model-00002-of-00002.safetensors",  # never written
                }
            }
        ),
        encoding="utf-8",
    )
    dataset = _dataset(tmp_path / "corpus.jsonl")
    _write(tmp_path / "eval_report.json", _suite_report("main", model, dataset))
    holdout = _write(tmp_path / "eval_holdout_report.json", _suite_report("holdout", model, dataset))
    monkeypatch.setattr(U, "OUT_DIR", tmp_path)
    monkeypatch.setattr(U, "WEIGHTS_DIR", model)
    monkeypatch.setenv("MM_HOLDOUT_REPORT", str(holdout))
    with _reject("absent from the manifest"):
        U._require_eval_receipts()


def test_lora_without_resolved_base_is_refused(tmp_path, monkeypatch) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"lora")
    (adapter / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    dataset = _dataset(tmp_path / "corpus.jsonl")
    selection = E.ModelSelection(
        kind="lora",
        model_path=adapter,
        tokenizer_path=adapter,
        base_path=None,
        base_ref="Qwen/Qwen3.5-4B",
        base_scope=E.BASE_SCOPE_INDEX_ONLY,
    )
    for suite, name in (("main", "eval_report.json"), ("holdout", "eval_holdout_report.json")):
        _write(tmp_path / name, _suite_report(suite, adapter, dataset, selection=selection))
    monkeypatch.setattr(U, "OUT_DIR", tmp_path)
    monkeypatch.setattr(U, "WEIGHTS_DIR", adapter)
    monkeypatch.setenv("MM_HOLDOUT_REPORT", str(tmp_path / "eval_holdout_report.json"))
    with _reject("not bound to a base checkpoint"):
        U._require_eval_receipts()


def test_index_only_base_binding_is_not_release_ready() -> None:
    receipt = {
        "selection": {
            "kind": "lora",
            "base": {"path": "/some/base", "scope": E.BASE_SCOPE_INDEX_ONLY, "files": {}},
        }
    }
    ok, reason = E.base_binding_is_release_ready(receipt)
    assert ok is False and "index-only" in reason


# ---------------------------------------------------------------------------
# Publication boundary
# ---------------------------------------------------------------------------


def test_derived_gguf_is_omitted_from_the_upload_plan(workspace, capsys) -> None:
    (workspace["tmp"] / "mind-mem-4b-Q4_K_M.gguf").write_bytes(b"gguf")
    uploads = U._discover_upload_paths()
    assert all(remote != "mind-mem-4b-Q4_K_M.gguf" for _, remote in uploads)
    assert "converter-parent binding" in capsys.readouterr().out


def test_model_card_must_carry_the_current_report_digest(workspace) -> None:
    card = workspace["tmp"] / "README.md"
    card.write_text("mind-mem-4b scored great.\n", encoding="utf-8")
    reports = U._require_eval_receipts()
    with _reject("model card does not match"):
        U._verify_upload_plan(U._discover_upload_paths(), reports)
    digest = reports["main"]["receipt"]["report_sha256"]
    card.write_text(f"report_sha256: {digest}\n", encoding="utf-8")
    with _reject("model card does not match"):
        U._verify_upload_plan(U._discover_upload_paths(), reports)
    card.write_text(B.render_release_card(reports), encoding="utf-8")
    U._verify_upload_plan(U._discover_upload_paths(), reports)


def test_file_swapped_at_publication_boundary_is_refused(workspace) -> None:
    reports = U._require_eval_receipts()
    uploads = U._discover_upload_paths()
    (workspace["model"] / "model.safetensors").write_bytes(b"swapped-at-publish")
    with _reject("changed between evaluation and publication"):
        U._verify_upload_plan(uploads, reports)


def test_unattested_file_in_the_plan_is_refused(workspace) -> None:
    reports = U._require_eval_receipts()
    stray = workspace["model"] / "generation_config.json"
    stray.write_text("{}\n", encoding="utf-8")  # appeared after evaluation
    with _reject("not covered by the evaluation receipt"):
        U._verify_upload_plan(U._discover_upload_paths(), reports)


def test_dry_run_prints_an_unverified_plan(workspace, capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["upload_to_hf.py", "--dry-run"])
    U.main()
    out = capsys.readouterr().out
    assert "UNVERIFIED" in out


def test_bindings_match_reports_a_reason(workspace) -> None:
    receipt = json.loads(workspace["main"].read_text(encoding="utf-8"))["receipt"]
    ok, reason = E.bindings_match(receipt)
    assert ok is True and reason == "ok"
    receipt["dataset"]["path"] = str(workspace["tmp"] / "gone.jsonl")
    ok, reason = E.bindings_match(receipt)
    assert ok is False and "dataset is missing" in reason


# ---------------------------------------------------------------------------
# Base-checkpoint identity — the defect root measured: two suites evaluating
# the same adapter against DIFFERENT base weights were accepted, because the
# validator compared only model and tokenizer manifests.
# ---------------------------------------------------------------------------


def _adapter(root: Path, base: Path, *, payload: bytes = b"adapter-A") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base.resolve()), "peft_type": "LORA"}),
        encoding="utf-8",
    )
    (root / "adapter_model.safetensors").write_bytes(payload)
    (root / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    return root


def _lora_selection(adapter: Path, base: Path) -> E.ModelSelection:
    return E.ModelSelection(
        kind="lora",
        model_path=adapter,
        tokenizer_path=adapter,
        base_path=base,
        base_scope=E.BASE_SCOPE_FULL,
        base_ref=str(base),
    )


def _lora_workspace(tmp_path, monkeypatch, *, holdout_base: Path | None = None):
    """One adapter, two suites; ``holdout_base`` lets a suite use another base."""
    base = _checkpoint(tmp_path / "base-A", payload=b"base-A-weights")
    adapter = _adapter(tmp_path / "adapter", base)
    dataset = _dataset(tmp_path / "corpus.jsonl")
    main = _write(
        tmp_path / "eval_report.json",
        _suite_report("main", adapter, dataset, selection=_lora_selection(adapter, base)),
    )
    holdout = _write(
        tmp_path / "eval_holdout_report.json",
        _suite_report(
            "holdout",
            adapter,
            dataset,
            selection=_lora_selection(adapter, holdout_base or base),
        ),
    )
    monkeypatch.setattr(U, "OUT_DIR", tmp_path)
    monkeypatch.setattr(U, "WEIGHTS_DIR", adapter)
    monkeypatch.setenv("MM_HOLDOUT_REPORT", str(holdout))
    return {"base": base, "adapter": adapter, "dataset": dataset, "main": main, "holdout": holdout, "tmp": tmp_path}


def test_lora_same_adapter_and_same_base_across_suites_is_accepted(tmp_path, monkeypatch) -> None:
    """Positive control: without it the different-base refusal proves nothing."""
    _lora_workspace(tmp_path, monkeypatch)
    reports = U._require_eval_receipts()
    assert reports["main"]["receipt"]["selection"]["kind"] == "lora"


def test_control_same_adapter_evaluated_against_different_bases_is_refused(tmp_path, monkeypatch) -> None:
    other_base = _checkpoint(tmp_path / "base-B", payload=b"base-B-weights")
    space = _lora_workspace(tmp_path, monkeypatch, holdout_base=other_base)
    # End to end the pair is refused; here the adapter's own declared base is
    # what fires first, because only one of the two suites can agree with it.
    with _reject("base"):
        U._require_eval_receipts()
    # The cross-suite comparison is exercised directly below, since a fixture
    # cannot make both suites agree with adapter_config *and* disagree on the
    # base — asserting only the end-to-end refusal would leave it untested.
    main_sel = json.loads(space["main"].read_text())["receipt"]["selection"]
    holdout_sel = json.loads(space["holdout"].read_text())["receipt"]["selection"]
    with _reject("different base checkpoints"):
        U._require_identical_selection(main_sel, holdout_sel)


def test_identical_selections_pass_the_cross_suite_check(tmp_path, monkeypatch) -> None:
    """Positive control for the comparison the different-base test drives."""
    space = _lora_workspace(tmp_path, monkeypatch)
    main_sel = json.loads(space["main"].read_text())["receipt"]["selection"]
    holdout_sel = json.loads(space["holdout"].read_text())["receipt"]["selection"]
    U._require_identical_selection(main_sel, holdout_sel)  # must not exit


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda s: s.update(kind="full-ft"), "different checkpoint kinds"),
        (lambda s: s["model"]["files"].update({"extra.bin": {"sha256": "x", "bytes": 1}}), "different checkpoint contents"),
        (lambda s: s["tokenizer"]["files"].pop("tokenizer.json"), "different tokenizers"),
        (lambda s: s["base"].update(path="/elsewhere/base"), "different base checkpoints"),
        (lambda s: s["base"].update(ref="other-repo/base"), "different base checkpoints"),
        (lambda s: s["base"].update(scope=E.BASE_SCOPE_INDEX_ONLY), "different base checkpoints"),
        (lambda s: s["base"]["files"]["model.safetensors"].update(sha256="0" * 64), "different base checkpoint contents"),
    ],
)
def test_every_selection_field_is_compared(tmp_path, monkeypatch, mutate, expected) -> None:
    space = _lora_workspace(tmp_path, monkeypatch)
    main_sel = json.loads(space["main"].read_text())["receipt"]["selection"]
    holdout_sel = json.loads(space["holdout"].read_text())["receipt"]["selection"]
    mutate(holdout_sel)
    with _reject(expected):
        U._require_identical_selection(main_sel, holdout_sel)


def test_forged_base_manifest_is_caught_by_rehashing(tmp_path, monkeypatch) -> None:
    """Same base path, one receipt claiming different base bytes."""
    space = _lora_workspace(tmp_path, monkeypatch)
    report = json.loads(space["holdout"].read_text(encoding="utf-8"))
    receipt = report["receipt"]
    files = receipt["selection"]["base"]["files"]
    files["model.safetensors"]["sha256"] = "0" * 64
    _write(space["holdout"], E.finalize_report(E.report_payload(report), receipt))
    with _reject("base checkpoint bytes changed"):
        U._require_eval_receipts()


def test_adapter_config_pointing_at_another_base_is_refused(tmp_path, monkeypatch) -> None:
    space = _lora_workspace(tmp_path, monkeypatch)
    elsewhere = _checkpoint(tmp_path / "base-C", payload=b"base-C-weights")
    (space["adapter"] / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(elsewhere.resolve())}),
        encoding="utf-8",
    )
    # Re-attest so the byte-level check is not what fires: the adapter's own
    # declared base must agree with the base the receipt records.
    selection = _lora_selection(space["adapter"], space["base"])
    for suite, path in (("main", space["main"]), ("holdout", space["holdout"])):
        _write(path, _suite_report(suite, space["adapter"], space["dataset"], selection=selection))
    with _reject("was not evaluated against the recorded base"):
        U._require_eval_receipts()


def test_base_shard_index_missing_a_shard_is_refused(tmp_path, monkeypatch) -> None:
    base = _checkpoint(tmp_path / "base-A", sharded=True)
    (base / "model-00001-of-00001.safetensors").unlink()
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    adapter = _adapter(tmp_path / "adapter", base)
    dataset = _dataset(tmp_path / "corpus.jsonl")
    selection = _lora_selection(adapter, base)
    main = _write(tmp_path / "eval_report.json", _suite_report("main", adapter, dataset, selection=selection))
    holdout = _write(tmp_path / "eval_holdout_report.json", _suite_report("holdout", adapter, dataset, selection=selection))
    monkeypatch.setattr(U, "OUT_DIR", tmp_path)
    monkeypatch.setattr(U, "WEIGHTS_DIR", adapter)
    monkeypatch.setenv("MM_HOLDOUT_REPORT", str(holdout))
    assert main.is_file()
    with _reject("absent from the manifest"):
        U._require_eval_receipts()


def test_a_stray_bin_file_is_not_a_base_weight_set(tmp_path, monkeypatch) -> None:
    """ "Something ends in .bin" used to satisfy the base-weight requirement."""
    base = tmp_path / "base-A"
    base.mkdir()
    (base / "config.json").write_text("{}\n", encoding="utf-8")
    (base / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (base / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (base / "extras").mkdir()
    (base / "extras" / "optimizer.bin").write_bytes(b"not weights")
    adapter = _adapter(tmp_path / "adapter", base)
    dataset = _dataset(tmp_path / "corpus.jsonl")
    selection = _lora_selection(adapter, base)
    _write(tmp_path / "eval_report.json", _suite_report("main", adapter, dataset, selection=selection))
    holdout = _write(tmp_path / "eval_holdout_report.json", _suite_report("holdout", adapter, dataset, selection=selection))
    monkeypatch.setattr(U, "OUT_DIR", tmp_path)
    monkeypatch.setattr(U, "WEIGHTS_DIR", adapter)
    monkeypatch.setenv("MM_HOLDOUT_REPORT", str(holdout))
    with _reject("no recognised weight file"):
        U._require_eval_receipts()


# ---------------------------------------------------------------------------
# Capture scope — dataset, sources and probe definitions
# ---------------------------------------------------------------------------


def test_probe_digest_mismatch_with_the_right_count_is_refused(workspace) -> None:
    groups = {g: list(items) for g, items in PROBE_SETS["main"].items()}
    groups["tool_call"] = [("substituted", "easier") for _ in groups["tool_call"]]
    assert len(groups["tool_call"]) == len(PROBE_SETS["main"]["tool_call"])
    _write(
        workspace["main"],
        _suite_report("main", workspace["model"], workspace["dataset"], probe_groups=groups),
    )
    with _reject("digest does not match the harness probe definitions"):
        U._require_eval_receipts()


def test_probe_set_swapped_between_capture_and_finalization_is_refused(workspace) -> None:
    model, dataset = workspace["model"], workspace["dataset"]
    selection = E.ModelSelection(kind="full-ft", model_path=model, tokenizer_path=model)
    captured = _capture(selection, dataset)
    swapped = {g: list(items) for g, items in PROBE_SETS["main"].items()}
    swapped["workflow"] = [("swapped", "after-capture") for _ in swapped["workflow"]]
    report = _suite_report("main", model, dataset, captured=captured, selection=selection, probe_groups=swapped)
    assert report["receipt"]["probes_match_definitions"] is False
    _write(workspace["main"], report)
    with _reject("incomplete"):
        U._require_eval_receipts()


def test_dataset_mutated_between_capture_and_finalization_is_refused(workspace) -> None:
    model, dataset = workspace["model"], workspace["dataset"]
    selection = E.ModelSelection(kind="full-ft", model_path=model, tokenizer_path=model)
    captured = _capture(selection, dataset)
    dataset.write_text('{"messages":["swapped mid-run"]}\n', encoding="utf-8")
    report = _suite_report("main", model, dataset, captured=captured, selection=selection)
    assert report["receipt"]["capture_verified"] is False
    _write(workspace["main"], report)
    with _reject("incomplete"):
        U._require_eval_receipts()


def test_source_mutated_between_capture_and_finalization_is_detected(tmp_path) -> None:
    """Sources are captured before use, so a mid-run edit is visible.

    Uses a throwaway repo root: the real evaluator sources cannot be mutated
    from a test, and asserting on them would prove nothing.
    """
    root = tmp_path / "repo"
    (root / "train").mkdir(parents=True)
    source = root / "train/eval_harness.py"
    source.write_text("# before\n", encoding="utf-8")
    dataset = _dataset(tmp_path / "corpus.jsonl")
    model = _checkpoint(tmp_path / "full-ft")
    selection = E.ModelSelection(kind="full-ft", model_path=model, tokenizer_path=model)
    captured = E.capture_inputs(
        selection,
        repo_root=root,
        dataset_root=dataset,
        source_paths=(source,),
        probe_sets={"g": [1, 2]},
    )
    assert captured["source"]["files"]["train/eval_harness.py"]["present"] is True
    unchanged = E.build_receipt(
        repo_root=root,
        suite="main",
        captured=captured,
        probe_counts={"g": (2, 2)},
        probe_sets={"g": [1, 2]},
        command="pytest",
        run_id=E.new_run_id(),
        started_at=E.now_iso(),
        ended_at=E.now_iso(),
        status="completed",
    )
    assert unchanged["capture_verified"] is True  # positive control
    source.write_text("# edited mid-run\n", encoding="utf-8")
    mutated = E.build_receipt(
        repo_root=root,
        suite="main",
        captured=captured,
        probe_counts={"g": (2, 2)},
        probe_sets={"g": [1, 2]},
        command="pytest",
        run_id=E.new_run_id(),
        started_at=E.now_iso(),
        ended_at=E.now_iso(),
        status="completed",
    )
    assert mutated["capture_verified"] is False
    assert mutated["complete"] is False


def test_declared_but_missing_source_file_is_not_silently_dropped(tmp_path) -> None:
    root = tmp_path / "repo"
    (root / "train").mkdir(parents=True)
    dataset = _dataset(tmp_path / "corpus.jsonl")
    model = _checkpoint(tmp_path / "full-ft")
    selection = E.ModelSelection(kind="full-ft", model_path=model, tokenizer_path=model)
    captured = E.capture_inputs(
        selection,
        repo_root=root,
        dataset_root=dataset,
        source_paths=(root / "train/eval_receipt.py",),
        probe_sets={"g": [1]},
    )
    assert captured["source"]["files"]["train/eval_receipt.py"]["present"] is False
    receipt = E.build_receipt(
        repo_root=root,
        suite="main",
        captured=captured,
        probe_counts={"g": (1, 1)},
        probe_sets={"g": [1]},
        command="pytest",
        run_id=E.new_run_id(),
        started_at=E.now_iso(),
        ended_at=E.now_iso(),
        status="completed",
    )
    assert receipt["complete"] is False


def test_counts_are_not_coerced(tmp_path) -> None:
    """``int(3.9)`` would launder a bogus count into a plausible one."""
    dataset = _dataset(tmp_path / "corpus.jsonl")
    model = _checkpoint(tmp_path / "full-ft")
    selection = E.ModelSelection(kind="full-ft", model_path=model, tokenizer_path=model)
    captured = _capture(selection, dataset)
    with pytest.raises(E.ReceiptError, match="non-bool integer"):
        E.build_receipt(
            repo_root=REPO,
            suite="main",
            captured=captured,
            probe_counts={g: (len(items), 3.9) for g, items in PROBE_SETS["main"].items()},
            probe_sets=PROBE_SETS["main"],
            command="pytest",
            run_id=E.new_run_id(),
            started_at=E.now_iso(),
            ended_at=E.now_iso(),
            status="completed",
        )


def test_interrupted_run_stays_incomplete(workspace) -> None:
    """A bounded/killed run must not be able to present itself as finished."""
    report = _suite_report("main", workspace["model"], workspace["dataset"], completed={"v4_surfaces": 1})
    assert report["receipt"]["complete"] is False
    _write(workspace["main"], report)
    with _reject("incomplete"):
        U._require_eval_receipts()


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_token_cli_argument_is_supported_and_never_printed(workspace, capsys, monkeypatch) -> None:
    secret = "hf_fixture_token_value"
    monkeypatch.setattr(sys, "argv", ["upload_to_hf.py", "--dry-run", "--token", secret])
    U.main()  # must not raise SystemExit(2) from argparse
    out = capsys.readouterr()
    assert secret not in out.out and secret not in out.err


@pytest.mark.parametrize("mutation", ["missing", "truncated", "duplicate", "prompt", "verdict", "response", "aggregate"])
def test_complete_raw_probe_outcomes_are_required(workspace, mutation):
    report = json.loads(workspace["main"].read_text())
    bench = report["tool_call"]
    if mutation == "missing":
        del bench["items"]
    elif mutation == "truncated":
        bench["items"].pop()
    elif mutation == "duplicate":
        bench["items"][1] = dict(bench["items"][0])
    elif mutation == "prompt":
        bench["items"][0]["prompt"] = "a substituted easy prompt"
    elif mutation == "verdict":
        bench["items"][0]["passed"] = 1
    elif mutation == "response":
        bench["items"][0]["response"] = "no matching tool"
    else:
        bench["items"][0]["response"] = "no matching tool"
        bench["items"][0]["passed"] = False
    # Re-seal so the raw-item validator, rather than the payload digest check,
    # must reject this self-consistent but internally contradictory report.
    _write(workspace["main"], E.finalize_report(E.report_payload(report), report["receipt"]))
    with _reject("raw"):
        U._require_eval_receipts()


def test_runner_keeps_successful_response_beyond_miss_excerpt(monkeypatch):
    response = "x" * 220 + " verify_chain"
    monkeypatch.setattr(H, "_chat", lambda *_: response)
    bench = H._bench_probes(None, None, "tool_call", [("check integrity", "verify_chain")])
    assert bench["hits"] == 1
    assert bench["misses"] == []
    assert bench["items"] == [{"index": 0, "prompt": "check integrity", "response": response, "passed": True}]


def test_fixed_predicates_preserve_case_and_transport_exclusions():
    assert H.score_probe("tool_call", ("q", "verify_chain"), "VERIFY_CHAIN")[0]
    assert not H.score_probe("block_schema", ("q", ["TransformHash"]), "transformhash")[0]
    assert H.score_probe("v39_transport_guard", ("q", ["/mcp"], ["/invented"]), "/mcp")[0]
    assert not H.score_probe("v39_transport_guard", ("q", ["/mcp"], ["/invented"]), "/mcp /invented")[0]
