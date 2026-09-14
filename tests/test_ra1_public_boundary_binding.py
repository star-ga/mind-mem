"""Public axis/prefetch RA.1 boundaries must retain one request snapshot.

These tests mutate only the workspace policy file after the real retrieval has
returned and before the real ``mind_mem.recall.attest_and_record`` helper runs.
They do not replace retrieval, scoring, or result data.  The mutation and the
helper call are asserted so a test cannot pass without crossing the production
boundary under review.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

import mind_mem.recall as recall_module
from mind_mem import prefetch
from mind_mem.mcp.infra.constants import MCP_SCHEMA_VERSION
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools import public
from mind_mem.pipeline_hash import current_pipeline_hash
from mind_mem.served_ledger import ServedRunV2, context_digest, read_served_runs

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
    """The deliberate anticipation boundary says it has no recall proof."""
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
