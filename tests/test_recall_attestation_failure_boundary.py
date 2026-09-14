"""Failure-boundary controls for the post-cache recall attestation rail."""

from __future__ import annotations

import builtins
import json

from mind_mem.recall_attestation import IndexAnchorResolution


def _deriver_failure(monkeypatch):
    import mind_mem.recall_attestation as attestation

    def fail(*args, **kwargs):
        raise RuntimeError("test deriver failure")

    monkeypatch.setattr(attestation, "derive_recall_attestation_for_workspace", fail)


def _block_served_ledger_import(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mind_mem.served_ledger":
            raise ImportError("test blocked served ledger")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)


def test_public_recall_cache_miss_and_hit_with_deriver_failure(monkeypatch, tmp_path):
    """Both ordinary cache paths retain answers but publish unproven proof."""
    import mind_mem.mcp.tools.recall as recall_tool
    from mind_mem.recall_cache import reset_singleton

    decisions = tmp_path / "decisions"
    decisions.mkdir()
    (decisions / "DECISIONS.md").write_text(
        "# DECISIONS\n\n---\n\n[D-20260101-001]\nStatement: The capital of France is Paris.\nStatus: active\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(recall_tool, "_workspace", lambda: str(tmp_path))
    monkeypatch.setattr(
        recall_tool,
        "_load_config",
        lambda workspace: {"cache": {"enabled": True}, "recall": {"vector_enabled": False}},
    )
    monkeypatch.setattr(
        recall_tool,
        "_resolve_chain_head_resolution",
        lambda workspace: IndexAnchorResolution.genesis(),
    )
    _deriver_failure(monkeypatch)
    _block_served_ledger_import(monkeypatch)
    reset_singleton()

    original_uncached = recall_tool._recall_impl_uncached
    uncached_calls = 0

    def counted_uncached(*args, **kwargs):
        nonlocal uncached_calls
        uncached_calls += 1
        return original_uncached(*args, **kwargs)

    monkeypatch.setattr(recall_tool, "_recall_impl_uncached", counted_uncached)

    first = json.loads(recall_tool._recall_impl("capital", limit=5, backend="bm25"))
    second = json.loads(recall_tool._recall_impl("capital", limit=5, backend="bm25"))

    for envelope in (first, second):
        assert envelope["count"] == 1
        assert envelope["attestation"]["served_proof"] == "unproven"
        assert envelope["attestation"]["served_seq"] is None
        assert envelope["attestation"]["served_row_hash"] is None
    assert uncached_calls == 1, "second call must be a real cache hit"


def test_derivation_and_ledger_failure_replaces_carried_proof(monkeypatch):
    """A second ledger failure cannot preserve carried proof from the input."""
    import mind_mem.mcp.tools.recall as recall_tool

    _deriver_failure(monkeypatch)
    _block_served_ledger_import(monkeypatch)
    raw = json.dumps(
        {
            "backend": "scan",
            "results": [{"_id": "B-1"}],
            "attestation": {"served_proof": "recorded", "served_seq": 4},
        }
    )

    envelope = json.loads(
        recall_tool._apply_attestation(
            raw,
            "scan",
            "2026-09-14",
            "query",
            config_hash="CFG",
            index_anchor="HEAD",
        )
    )
    assert envelope["attestation"]["served_proof"] == "unproven"
    assert envelope["attestation"]["served_seq"] is None
    assert envelope["attestation"]["served_row_hash"] is None


def test_malformed_and_non_block_envelopes_are_preserved(monkeypatch):
    """The guard handles malformed and presentation-only inputs without proof."""
    import mind_mem.mcp.tools.recall as recall_tool

    _deriver_failure(monkeypatch)
    malformed = "not json"
    bundle = json.dumps({"facts": [], "relations": []})

    assert recall_tool._apply_attestation(malformed, "scan", "2026-09-14", "q") == malformed
    assert recall_tool._apply_attestation(bundle, "scan", "2026-09-14", "q") == bundle
