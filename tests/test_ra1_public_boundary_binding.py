"""Public axis/prefetch RA.1 boundaries must retain one request snapshot.

These tests mutate only the workspace policy file after the real retrieval has
returned and before the real ``mind_mem.recall.attest_and_record`` helper runs.
They do not replace retrieval, scoring, or result data.  The mutation and the
helper call are asserted so a test cannot pass without crossing the production
boundary under review.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pytest

import mind_mem.recall as recall_module
from mind_mem import prefetch
from mind_mem.mcp.infra.constants import MCP_SCHEMA_VERSION
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools import public
from mind_mem.pipeline_hash import current_pipeline_hash
from mind_mem.recall_digests import query_hash, served_set_digest
from mind_mem.served_ledger import ServedRunV2, context_digest, read_served_runs, row_hash

QUERY = "deterministic compiler"


def _seed_workspace(root: Path, *, anticipation: bool = False) -> str:
    (root / "decisions").mkdir(parents=True)
    for name in ("tasks", "entities", "intelligence"):
        (root / name).mkdir()
    (root / "decisions" / "DECISIONS.md").write_text(
        "[D-RA1-PUBLIC-001]\nStatement: deterministic compiler retrieval context\nStatus: active\nDate: 2026-01-01\n\n",
        encoding="utf-8",
        newline="\n",
    )
    cache: dict[str, Any] = {"enabled": False}
    if anticipation:
        cache["anticipation"] = {
            "enabled": True,
            "min_corpus_stems": 0,
            "novel_ratio_threshold": 1.0,
        }
    (root / "mind-mem.json").write_text(
        json.dumps(
            {
                "cache": cache,
                "extraction": {"backend": "unknown-a", "model": "a"},
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    return str(root)


def _config_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {"cache": {"enabled": False}, "extraction": {"backend": "unknown-a", "model": "a"}},
        {"cache": {"enabled": False}, "extraction": {"backend": "unknown-b", "model": "b"}},
    )


def _seed_many_recall_workspace(root: str) -> None:
    """Use the shared supported decisions fixture with enough hits to test limits."""
    from _recall_clock_sentinel import write_workspace

    blocks = [("D-RA1-LIMIT-000", "the deterministic compiler emits byte identical artifacts", "2026-01-01")]
    for number in range(14):
        unique = " ".join(f"stem{number}x{index}" for index in range(25))
        blocks.append(
            (
                f"D-RA1-LIMIT-{number + 1:03d}",
                f"recall scoring takes the instant as an input {unique}",
                "2026-01-01",
            )
        )
    write_workspace(root, tuple(blocks))


def _assert_coherent_or_unproven(
    workspace: str,
    payload: dict[str, Any],
    *,
    hash_a: str,
    generation_a: str,
) -> None:
    """Reject the mixed row and allow an explicit fail-closed result."""
    attestation = payload.get("attestation")
    rows = read_served_runs(workspace)
    assert payload.get("results"), payload
    assert isinstance(attestation, dict), payload

    if attestation.get("served_proof") == "unproven":
        assert attestation.get("served_seq") is None, attestation
        assert attestation.get("served_row_hash") is None, attestation
        assert attestation.get("ledger_error"), attestation
        assert rows == (), rows
        return

    assert attestation.get("served_proof") == "recorded", attestation
    assert attestation.get("config_hash") == hash_a, attestation
    assert attestation.get("served_seq") is not None, attestation
    assert len(rows) == 1, rows
    row = rows[0]
    assert isinstance(row, ServedRunV2), row
    assert row.pipeline_hash == hash_a, row
    assert row.context_digest == context_digest(
        workspace=workspace,
        config_hash=hash_a,
        generation=generation_a,
        index_anchor=row.index_anchor,
    )
    assert row.context_digest == attestation.get("served_context_digest"), attestation


@pytest.fixture(autouse=True)
def _reset_anticipation_cache() -> Any:
    prefetch.reset_cache()
    yield
    prefetch.reset_cache()


@pytest.mark.parametrize("mode", ["axis", "prefetch"])
def test_public_boundary_snapshot_does_not_record_live_hash_with_old_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """A policy change at the production handoff cannot mint a mixed row."""
    workspace = _seed_workspace(tmp_path / mode)
    config_path = Path(workspace) / "mind-mem.json"
    config_a, config_b = _config_pair()
    hash_a = current_pipeline_hash(workspace)
    calls: list[dict[str, Any]] = []
    original: Callable[..., Any] = recall_module.attest_and_record

    def transition(*args: Any, **kwargs: Any) -> Any:
        calls.append(
            {
                "helper": f"{original.__module__}.{original.__name__}",
                "workspace": args[0] if args else None,
                "query": args[1] if len(args) > 1 else None,
                "generation": kwargs.get("generation"),
            }
        )
        config_path.write_text(json.dumps(config_b), encoding="utf-8", newline="\n")
        changed_hash = current_pipeline_hash(args[0])
        assert changed_hash != hash_a
        calls[-1]["post_change_hash"] = changed_hash
        return original(*args, **kwargs)

    monkeypatch.setattr(recall_module, "attest_and_record", transition)

    with use_workspace(workspace):
        if mode == "axis":
            raw = public.recall(QUERY, mode="axis", axes="lexical", limit=5)
        else:
            raw = public.recall(
                QUERY,
                mode="prefetch",
                signals=QUERY,
                limit=5,
            )
    generation_a = prefetch.anticipation_generation_identity(config_a, str(MCP_SCHEMA_VERSION))
    assert generation_a is not None

    payload = json.loads(raw)
    assert len(calls) == 1, calls
    assert calls[0]["helper"] == "mind_mem.recall.attest_and_record", calls
    assert calls[0]["workspace"] == workspace, calls
    assert calls[0]["query"] == QUERY, calls
    assert calls[0]["post_change_hash"] != hash_a, calls
    assert json.loads(config_path.read_text(encoding="utf-8")) == config_b
    if isinstance(payload.get("attestation"), dict) and payload["attestation"].get("served_proof") == "recorded":
        assert calls[0]["generation"] == generation_a, calls
    _assert_coherent_or_unproven(
        workspace,
        payload,
        hash_a=hash_a,
        generation_a=generation_a,
    )


def test_public_prefetch_positive_has_results_and_recorded_row(tmp_path: Path) -> None:
    """The positive control proves a real public prefetch serve is recorded."""
    workspace = _seed_workspace(tmp_path / "positive")

    with use_workspace(workspace):
        payload = json.loads(public.recall(QUERY, mode="prefetch", signals=QUERY, limit=5))

    assert payload.get("results"), payload
    attestation = payload.get("attestation")
    assert isinstance(attestation, dict), payload
    assert attestation.get("served_proof") == "recorded", payload
    rows = read_served_runs(workspace)
    assert len(rows) == 1, rows
    assert isinstance(rows[0], ServedRunV2), rows
    assert tuple(hit["_id"] for hit in payload["results"]) == rows[0].ids


@pytest.mark.parametrize("mode", ["axis", "prefetch"])
def test_public_boundary_proof_failure_is_explicit_and_unrecorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """A failed proof probe must not disappear as a null attestation."""
    workspace = _seed_workspace(tmp_path / f"failure-{mode}")
    calls: list[str] = []
    original: Callable[..., Any] = recall_module.attest_and_record

    def observed_failure(*args: Any, **kwargs: Any) -> Any:
        calls.append(f"{original.__module__}.{original.__name__}")
        return original(*args, **kwargs)

    def fail_flags(*_args: Any, **_kwargs: Any) -> tuple[bool, bool]:
        raise RuntimeError("CONFIG_HASH_PROBE_FAIL")

    monkeypatch.setattr(recall_module, "attest_and_record", observed_failure)
    monkeypatch.setattr(recall_module, "resolve_vector_flags", fail_flags)

    with use_workspace(workspace):
        if mode == "axis":
            raw = public.recall(QUERY, mode="axis", axes="lexical", limit=5)
        else:
            raw = public.recall(QUERY, mode="prefetch", signals=QUERY, limit=5)

    payload = json.loads(raw)
    assert calls == ["mind_mem.recall.attest_and_record"], calls
    assert payload.get("results"), payload
    attestation = payload.get("attestation")
    assert isinstance(attestation, dict), payload
    assert attestation.get("served_proof") == "unproven", payload
    assert attestation.get("served_seq") is None, payload
    assert attestation.get("served_row_hash") is None, payload
    assert "CONFIG_HASH_PROBE_FAIL" in (attestation.get("ledger_error") or ""), payload
    assert read_served_runs(workspace) == (), payload


def test_public_anticipation_positive_is_explicitly_unattested(tmp_path: Path) -> None:
    """A local hit has a serving receipt, but no false corpus-read attestation."""
    workspace = _seed_workspace(tmp_path / "anticipation", anticipation=True)

    with use_workspace(workspace):
        warm = json.loads(public.recall(QUERY, mode="prefetch", signals=QUERY, limit=5))
        served = json.loads(public.recall(QUERY, mode="auto", limit=5))

    assert warm.get("results"), warm
    assert served.get("backend") == "anticipation_cache", served
    assert served.get("results"), served
    assert served.get("attestation") is None, served
    assert any("no recall attestation" in warning for warning in served.get("warnings", [])), served
    assert all(hit.get("_retrieval_source") == "anticipation_cache" for hit in served["results"])
    receipt = served.get("serving_receipt")
    assert isinstance(receipt, dict), served
    assert receipt.get("served_proof") == "recorded", receipt
    assert receipt.get("served_serve_kind") == "anticipation", receipt
    rows = read_served_runs(workspace)
    assert len(rows) == 2, rows
    row = rows[-1]
    assert isinstance(row, ServedRunV2), row
    ids = tuple(hit.get("_id", "") for hit in served["results"])
    assert row.ids == ids, row
    assert row.query_hash == query_hash(QUERY), row
    assert row.served_digest == served_set_digest(ids), row
    assert receipt.get("served_seq") == row.seq, receipt
    assert receipt.get("served_row_hash") == row_hash(row), receipt
    assert receipt.get("query_hash") == row.query_hash, receipt
    assert receipt.get("results_digest") == row.served_digest, receipt
    assert receipt.get("config_hash") == row.pipeline_hash, receipt
    assert receipt.get("index_anchor") == row.index_anchor, receipt
    assert receipt.get("scoring_instant") == row.scoring_instant, receipt
    assert receipt.get("served_context_digest") == row.context_digest, receipt
    assert receipt.get("run_id") == row.run_id, receipt
    assert len(receipt["run_id"]) == 64, receipt

    # The local answer is not a corpus-read attestation, but its recorded
    # occurrence is still eligible for the existing explicit outcome join.
    # This proves the receipt is directly usable by a caller and that it names
    # the same answer identity as the row rather than asking callers to
    # reimplement the run-id preimage.
    from mind_mem.outcome_attribution import report_outcome

    outcome = report_outcome(workspace, [ids[0]], "success", run_id=receipt["run_id"])
    assert outcome["run_id"] == row.run_id, outcome
    assert outcome["block_ids"] == [ids[0]], outcome
    config = json.loads((Path(workspace) / "mind-mem.json").read_text(encoding="utf-8"))
    config_hash = current_pipeline_hash(workspace)
    generation = prefetch.anticipation_generation_identity(config, str(MCP_SCHEMA_VERSION))
    assert generation is not None
    assert row.context_digest == context_digest(
        workspace=workspace,
        config_hash=config_hash,
        generation=generation,
        index_anchor=row.index_anchor,
    )


def test_public_rejected_anticipation_hit_falls_through_without_receipt(tmp_path: Path) -> None:
    """A novel-term rejection uses the normal public retrieval path."""
    workspace = _seed_workspace(tmp_path / "anticipation-rejected", anticipation=True)

    with use_workspace(workspace):
        warm = json.loads(public.recall(QUERY, mode="prefetch", signals=QUERY, limit=5))
        rejected = json.loads(public.recall("kubernetes ingress certificate", mode="auto", limit=5))

    assert warm.get("results"), warm
    assert rejected.get("backend") != "anticipation_cache", rejected
    assert "serving_receipt" not in rejected, rejected
    rows = read_served_runs(workspace)
    assert len(rows) == 2, rows
    assert all(row.serve_kind == "attested" for row in rows if isinstance(row, ServedRunV2)), rows


def test_public_anticipation_unresolved_generation_falls_through_without_downgrade(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unrepresentable cache generation cannot take the local or v1 path."""
    workspace = _seed_workspace(tmp_path / "anticipation-unresolved-generation", anticipation=True)

    with use_workspace(workspace):
        warm = json.loads(public.recall(QUERY, mode="prefetch", signals=QUERY, limit=5))
        monkeypatch.setattr(prefetch, "anticipation_generation_identity", lambda *_args, **_kwargs: None)
        served = json.loads(public.recall(QUERY, mode="auto", limit=5))

    assert warm.get("results"), warm
    assert served.get("backend") != "anticipation_cache", served
    assert "serving_receipt" not in served, served
    assert served.get("attestation", {}).get("served_proof") == "unproven", served
    assert served["attestation"].get("served_seq") is None, served
    assert len(read_served_runs(workspace)) == 1, read_served_runs(workspace)


def test_public_anticipation_malformed_result_id_stays_unproven(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed local hit cannot produce a ledger row for a smaller answer."""
    workspace = _seed_workspace(tmp_path / "anticipation-malformed-id", anticipation=True)

    with use_workspace(workspace):
        public.recall(QUERY, mode="prefetch", signals=QUERY, limit=5)
        cache = prefetch.get_cache()
        original_lookup = cache.lookup

        def malformed_lookup(*args: Any, **kwargs: Any) -> Any:
            decision = original_lookup(*args, **kwargs)
            return replace(decision, served=({"text": "missing-id"},))

        monkeypatch.setattr(cache, "lookup", malformed_lookup)
        served = json.loads(public.recall(QUERY, mode="auto", limit=5))

    assert served.get("backend") == "anticipation_cache", served
    receipt = served.get("serving_receipt")
    assert receipt.get("served_proof") == "unproven", receipt
    assert "missing a non-empty string _id" in receipt.get("ledger_error", ""), receipt
    assert len(read_served_runs(workspace)) == 1, read_served_runs(workspace)


def test_anticipation_malformed_envelope_gets_explicit_unproven_receipt(tmp_path: Path) -> None:
    """A valid anticipation envelope with malformed results remains accountable."""
    from mind_mem.mcp.tools.recall import _record_anticipation_run

    workspace = _seed_workspace(tmp_path / "anticipation-malformed-envelope", anticipation=True)
    raw = json.dumps({"backend": "anticipation_cache", "results": {"wrong": "shape"}, "answer": "preserve"})

    with use_workspace(workspace):
        served = json.loads(
            _record_anticipation_run(
                raw,
                workspace,
                query=QUERY,
                config_hash=current_pipeline_hash(workspace),
                index_anchor="0" * 64,
                scoring_instant="2026-09-13",
                generation="PV:malformed-envelope",
            )
        )

    assert served["answer"] == "preserve", served
    receipt = served.get("serving_receipt")
    assert receipt.get("served_proof") == "unproven", receipt
    assert receipt.get("served_seq") is None, receipt
    assert receipt.get("served_row_hash") is None, receipt
    assert "results must be a list" in receipt.get("ledger_error", ""), receipt
    assert read_served_runs(workspace) == ()


def test_public_anticipation_receipt_respects_disabled_ledger(tmp_path: Path) -> None:
    """Opting out reports an unproven local serve and writes no row."""
    workspace = _seed_workspace(tmp_path / "anticipation-disabled", anticipation=True)
    config_path = Path(workspace) / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["served_ledger"] = {"enabled": False}
    config_path.write_text(json.dumps(config), encoding="utf-8", newline="\n")

    with use_workspace(workspace):
        public.recall(QUERY, mode="prefetch", signals=QUERY, limit=5)
        served = json.loads(public.recall(QUERY, mode="auto", limit=5))

    assert served.get("backend") == "anticipation_cache", served
    receipt = served.get("serving_receipt")
    assert receipt.get("served_proof") == "unproven", receipt
    assert receipt.get("ledger_error") == "disabled", receipt
    assert "run_id" not in receipt, receipt
    assert read_served_runs(workspace) == ()


def test_public_anticipation_receipt_reports_a_corrupt_ledger(tmp_path: Path) -> None:
    """A local answer survives a broken ledger, with an explicit refusal."""
    workspace = _seed_workspace(tmp_path / "anticipation-corrupt", anticipation=True)
    ledger_dir = Path(workspace) / ".mind-mem-ledger"

    with use_workspace(workspace):
        public.recall(QUERY, mode="prefetch", signals=QUERY, limit=5)
        assert json.loads(public.recall(QUERY, mode="auto", limit=5))["backend"] == "anticipation_cache"
        ledger_dir.unlink() if ledger_dir.is_file() else None
        if ledger_dir.is_dir():
            import shutil

            shutil.rmtree(ledger_dir)
        ledger_dir.write_text("not a directory\n", encoding="utf-8")
        served = json.loads(public.recall(QUERY, mode="auto", limit=5))

    assert served.get("backend") == "anticipation_cache", served
    receipt = served.get("serving_receipt")
    assert receipt.get("served_proof") == "unproven", receipt
    assert receipt.get("served_seq") is None, receipt
    assert receipt.get("served_row_hash") is None, receipt
    assert receipt.get("ledger_error") and receipt.get("ledger_error") != "disabled", receipt
    assert "run_id" not in receipt, receipt


def test_public_anticipation_receipt_refuses_unresolved_config_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cache hit never downgrades to a v1 row when its snapshot hash is unknown."""
    import mind_mem.pipeline_hash as pipeline_hash

    workspace = _seed_workspace(tmp_path / "anticipation-unresolved", anticipation=True)
    with use_workspace(workspace):
        public.recall(QUERY, mode="prefetch", signals=QUERY, limit=5)
        before = read_served_runs(workspace)

        def fail_hash(*_args: Any, **_kwargs: Any) -> str:
            raise RuntimeError("CONFIG_HASH_UNRESOLVED_CONTROL")

        monkeypatch.setattr(pipeline_hash, "current_pipeline_hash", fail_hash)
        served = json.loads(public.recall(QUERY, mode="auto", limit=5))

    assert served.get("backend") == "anticipation_cache", served
    receipt = served.get("serving_receipt")
    assert receipt.get("served_proof") == "unproven", receipt
    assert receipt.get("served_seq") is None, receipt
    assert receipt.get("served_row_hash") is None, receipt
    assert receipt.get("ledger_error"), receipt
    assert read_served_runs(workspace) == before


def test_public_anticipation_lookup_uses_the_captured_limit_after_disk_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A config edit before cache selection cannot change this request's ceiling."""
    from mind_mem.mcp.tools.recall import _recall_impl
    from mind_mem.mcp.tools.recall import prefetch as prefetch_tool

    workspace = str(tmp_path / "anticipation-limit-snapshot")
    _seed_many_recall_workspace(workspace)
    config_path = Path(workspace) / "mind-mem.json"
    config = {
        "cache": {"enabled": False, "anticipation": {"enabled": True}},
        "limits": {"max_recall_results": 3, "max_prefetch_results": 20},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8", newline="\n")

    with use_workspace(workspace):
        filled = json.loads(prefetch_tool("recall,scoring,instant,input", limit=20))
        assert filled.get("count", 0) > 3, filled
        captured_hash = current_pipeline_hash(workspace)
        import mind_mem.pipeline_hash as pipeline_hash

        original_hash = pipeline_hash.current_pipeline_hash
        hash_calls = 0

        def capture_then_mutate(root: str) -> str:
            nonlocal hash_calls
            result = original_hash(root)
            hash_calls += 1
            if hash_calls == 1:
                changed = dict(config)
                changed["limits"] = {"max_recall_results": 1, "max_prefetch_results": 20}
                config_path.write_text(json.dumps(changed), encoding="utf-8", newline="\n")
            return result

        monkeypatch.setattr(pipeline_hash, "current_pipeline_hash", capture_then_mutate)
        served = json.loads(_recall_impl("recall scoring instant", limit=100, scoring_instant="2026-01-02"))

    assert hash_calls == 1
    assert served.get("backend") == "anticipation_cache", served
    assert 1 < served.get("count", 0) <= 3, served
    receipt = served.get("serving_receipt")
    assert receipt.get("served_proof") == "recorded", receipt
    assert receipt.get("config_hash") == captured_hash, receipt
    rows = read_served_runs(workspace)
    assert rows[-1].pipeline_hash == captured_hash, rows
