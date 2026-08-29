"""Tests for the per-run recall attestation (recall_attestation.py).

Nothing attested a recall result before this: a ranked list carried no
runtime evidence of *which path* produced it. ``RecallAttestation`` is the
first such artifact — a runtime signal about how an answer was produced,
generalising the ``.degraded`` marker.

The three load-bearing wedge rails each get a test here:

1. DERIVABLE — every field recomputed from a recorded run signal, never a
   string a producer typed.
2. NEVER PERSISTED — deriving writes nothing to the block store / audit chain.
3. DETERMINISTIC — same run state -> same attestation bytes (no clock/rand).

Plus: the degraded marker is folded in, the MCP envelope surfaces it, and the
derived legs match what actually ran.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from mind_mem.hybrid_recall import _as_results
from mind_mem.recall_attestation import (
    GENESIS_ANCHOR,
    RECALL_ATTEST_TAG,
    RecallAttestation,
    _resolve_index_anchor,
    build_recall_attestation,
    derive_legs,
    derive_recall_attestation,
    derive_recall_attestation_for_workspace,
)

#: Every record below answers a question — v2 binds it, so every builder
#: call has to name one. Held constant except where a test varies it.
QUERY = "what did we decide about the ingest gate"


def _make_workspace() -> str:
    ws = tempfile.mkdtemp(prefix="mm_att_")
    os.makedirs(os.path.join(ws, "decisions"), exist_ok=True)
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as f:
        f.write(
            "# DECISIONS\n\n---\n\n"
            "[D-20260101-001]\nStatement: The capital of France is Paris.\nStatus: active\n\n---\n\n"
            "[D-20260101-002]\nStatement: Python is used for data science.\nStatus: active\n"
        )
    return ws


# ---------------------------------------------------------------------------
# Rail 1 — DERIVABLE (legs read from real run flags, not a self-declared string)
# ---------------------------------------------------------------------------


def test_legs_derived_from_real_result_flags_not_declared():
    """The legs come from the recorded result object + backend flags — there is
    no ``legs=`` string parameter anyone can type to assert a path ran."""
    # Healthy two-leg hybrid fusion: vector requested + available + no marker.
    r = _as_results([{"_id": "a", "score": 1.0}], None)
    att = derive_recall_attestation(r, vector_requested=True, vector_available=True, config_hash="CFG", query=QUERY)
    assert att.legs_ran == ("bm25", "hybrid", "vector")
    assert att.legs_degraded == ()

    # The ONLY way to change the legs is to change the recorded run state.
    # There is no free-form leg input on the derive surface.
    import inspect

    sig = inspect.signature(derive_recall_attestation)
    assert "legs_ran" not in sig.parameters
    assert "legs_degraded" not in sig.parameters


def test_legs_match_bm25_only_when_vector_not_requested():
    """A plain BM25 recall (vector never requested) attests bm25 only —
    no phantom vector/hybrid leg, no spurious degradation."""
    ran, degraded = derive_legs([{"_id": "a"}], vector_requested=False, vector_available=False)
    assert ran == ("bm25",)
    assert degraded == ()


def test_legs_vector_degraded_when_requested_but_unavailable():
    """Vector requested but the backend was unavailable → vector is a *degraded*
    leg (requested, not served), not a leg that ran."""
    r = _as_results([{"_id": "a"}], {"leg": "vector", "reason": "unavailable"})
    ran, degraded = derive_legs(r, vector_requested=True, vector_available=False)
    assert "vector" not in ran
    assert "hybrid" not in ran
    assert ran == ("bm25",)
    assert degraded == ("vector",)


def test_graph_leg_detected_from_graph_hop_provenance():
    """The graph leg is derived from the ``_graph_hop`` provenance the graph
    expander stamps on walked hits — a recorded signal."""
    r = _as_results([{"_id": "a"}, {"_id": "b", "_graph_hop": 1}], None)
    att = derive_recall_attestation(r, vector_requested=True, vector_available=True, config_hash="CFG", query=QUERY)
    assert "graph" in att.legs_ran


def test_pg_bm25_fallback_provenance_degrades_vector():
    """A pgvector server-side hybrid that served BM25-only stamps
    ``_retrieval_source == 'bm25_fallback'`` on hits — that recorded provenance
    degrades the vector leg even without a ``.degraded`` marker object."""
    hits = [{"_id": "a", "_retrieval_source": "bm25_fallback"}]
    ran, degraded = derive_legs(hits, vector_requested=True, vector_available=True)
    assert ran == ("bm25",)
    assert degraded == ("vector",)


def test_config_hash_reuses_pipeline_probe_not_reinvented():
    """The effective config hash is the *reused* pipeline-probe SHA, byte-for-byte
    — the module does not compute its own config digest."""
    from mind_mem.pipeline_hash import current_pipeline_hash

    ws = _make_workspace()
    probe = current_pipeline_hash(ws)
    att = derive_recall_attestation_for_workspace(
        _as_results([{"_id": "a"}], None), ws, vector_requested=True, vector_available=True, query=QUERY
    )
    assert att.config_hash == probe


# ---------------------------------------------------------------------------
# Rail 3 — DETERMINISTIC (same run state -> same bytes)
# ---------------------------------------------------------------------------


def test_deterministic_bytes_same_state_same_hash():
    r = _as_results([{"_id": "a"}, {"_id": "b"}], None)
    a1 = derive_recall_attestation(r, vector_requested=True, vector_available=True, config_hash="CFG", query=QUERY)
    a2 = derive_recall_attestation(r, vector_requested=True, vector_available=True, config_hash="CFG", query=QUERY)
    assert a1.attestation_hash == a2.attestation_hash
    assert a1.to_dict() == a2.to_dict()


def test_hash_binds_every_field_tamper_detectable():
    a = build_recall_attestation(
        legs_ran=("bm25", "vector", "hybrid"),
        legs_degraded=(),
        config_hash="CFG",
        degraded=None,
        index_anchor=GENESIS_ANCHOR,
        result_count=3,
        query=QUERY,
    )
    assert a.is_internally_consistent()
    # Mutate a bound field without recomputing the hash → inconsistent.
    import dataclasses

    tampered = dataclasses.replace(a, result_count=999)
    assert not tampered.is_internally_consistent()
    tampered2 = dataclasses.replace(a, config_hash="OTHER")
    assert not tampered2.is_internally_consistent()
    # v2 added two bindings; "every field" has to keep meaning every field.
    assert not dataclasses.replace(a, query_hash="0" * 64).is_internally_consistent()
    assert not dataclasses.replace(a, schema="SOMETHING_ELSE").is_internally_consistent()


def test_different_legs_produce_different_hash():
    common = dict(config_hash="CFG", degraded=None, index_anchor=GENESIS_ANCHOR, result_count=1, query=QUERY)
    bm25_only = build_recall_attestation(legs_ran=("bm25",), legs_degraded=(), **common)
    hybrid = build_recall_attestation(legs_ran=("bm25", "vector", "hybrid"), legs_degraded=(), **common)
    assert bm25_only.attestation_hash != hybrid.attestation_hash


def test_record_is_a_pure_function_of_its_fields_roundtrip():
    """Round-tripping through to_dict/from_dict preserves the hash — proof the
    record is a pure function of its fields (no hidden timestamp/nonce).

    Renamed from ``test_no_clock_or_random_in_preimage_roundtrip``: the preimage
    now deliberately carries the run's ``scoring_instant``, so "no clock in the
    preimage" is no longer the property being asserted here — and leaving that
    name in place would have been a false green. What still holds, and is what
    this test is for, is that nothing *unrecorded* enters the hash.
    See tests/test_recall_attestation_completeness.py for the instant binding."""
    a = derive_recall_attestation(
        _as_results([{"_id": "a"}], None),
        vector_requested=True,
        vector_available=True,
        config_hash="CFG",
        query=QUERY,
    )
    b = RecallAttestation.from_dict(json.loads(json.dumps(a.to_dict())))
    assert b.attestation_hash == a.attestation_hash
    assert b.is_internally_consistent()


# ---------------------------------------------------------------------------
# Degraded marker folded in
# ---------------------------------------------------------------------------


def test_degraded_marker_folded_in_verbatim():
    marker = {"leg": "vector", "reason": "deadline_exceeded"}
    r = _as_results([{"_id": "a"}], marker)
    att = derive_recall_attestation(r, vector_requested=True, vector_available=True, config_hash="CFG", query=QUERY)
    assert att.degraded == marker
    assert att.legs_degraded == ("vector",)
    # The readable marker is bound into the hash — swapping it breaks consistency.
    import dataclasses

    swapped = dataclasses.replace(att, degraded={"leg": "vector", "reason": "healthy_lie"})
    assert not swapped.is_internally_consistent()


def test_multi_query_union_marker_legs_split_out():
    """The multi-query union marker can comma-join legs; each is folded into
    ``legs_degraded`` separately."""
    marker = {"leg": "vector", "reason": "import_failed,unavailable", "variants_degraded": "2", "variants_total": "3"}
    r = _as_results([{"_id": "a"}], marker)
    att = derive_recall_attestation(r, vector_requested=True, vector_available=True, config_hash="CFG", query=QUERY)
    assert att.degraded == marker
    assert "vector" in att.legs_degraded


def test_healthy_recall_has_none_degraded():
    r = _as_results([{"_id": "a"}], None)
    att = derive_recall_attestation(r, vector_requested=True, vector_available=True, config_hash="CFG", query=QUERY)
    assert att.degraded is None
    assert att.legs_degraded == ()


# ---------------------------------------------------------------------------
# Rail 2 — NEVER PERSISTED
# ---------------------------------------------------------------------------


def test_module_exposes_no_persistence_surface():
    """Unlike fold_attestation (which anchors), this module must expose NO
    write / anchor / append entry point — the verdict is runtime-only."""
    import mind_mem.recall_attestation as mod

    # Persistence is an *action*, so only callables can be a write surface.
    # "anchor" as a noun (GENESIS_ANCHOR / index_anchor) is a read-only
    # reference, not a write — exclude the data constants.
    banned = ("attest_fold", "anchor_", "append", "write", "persist", "store_", "save")
    for name in mod.__all__:
        obj = getattr(mod, name)
        if not callable(obj):
            continue
        low = name.lower()
        assert not any(b in low for b in banned), f"{name} looks like a persistence surface"
    # And there is genuinely no anchoring entry point analogous to
    # fold_attestation.attest_fold.
    assert not hasattr(mod, "attest_recall")
    assert not hasattr(mod, "anchor_recall")


def test_derive_writes_nothing_to_workspace(tmp_path):
    """Deriving an attestation must leave the workspace byte-identical — no
    audit chain created, no block store touched, no files written."""
    ws = str(tmp_path)
    os.makedirs(os.path.join(ws, "decisions"), exist_ok=True)
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as f:
        f.write("# DECISIONS\n\n---\n\n[D-1]\nStatement: x.\nStatus: active\n")

    def _snapshot(root):
        snap = {}
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                p = os.path.join(dirpath, name)
                with open(p, "rb") as fh:
                    snap[os.path.relpath(p, root)] = fh.read()
        return snap

    before = _snapshot(ws)
    att = derive_recall_attestation_for_workspace(
        _as_results([{"_id": "D-1"}], {"leg": "vector", "reason": "unavailable"}),
        ws,
        vector_requested=True,
        vector_available=False,
        query=QUERY,
    )
    after = _snapshot(ws)
    assert before == after, "deriving an attestation mutated the workspace"
    # No audit chain dir should have been created by anchor-free derivation.
    assert not os.path.isdir(os.path.join(ws, ".mind-mem-audit"))
    assert att.legs_degraded == ("vector",)


def test_index_anchor_read_only_creates_nothing(tmp_path):
    ws = str(tmp_path)
    # No chain yet → genesis, and no dir created by the read.
    assert _resolve_index_anchor(ws) == GENESIS_ANCHOR
    assert not os.path.isdir(os.path.join(ws, ".mind-mem-audit"))


def test_index_anchor_reads_existing_chain_head(tmp_path):
    ws = str(tmp_path)
    audit_dir = os.path.join(ws, ".mind-mem-audit")
    os.makedirs(audit_dir, exist_ok=True)
    with open(os.path.join(audit_dir, "chain.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"seq": 1, "entry_hash": "a" * 64}) + "\n")
        f.write(json.dumps({"seq": 2, "entry_hash": "b" * 64}) + "\n")
    assert _resolve_index_anchor(ws) == "b" * 64
    # The anchor is bound into the attestation.
    att = derive_recall_attestation_for_workspace(
        _as_results([{"_id": "a"}], None), ws, vector_requested=False, vector_available=False, query=QUERY
    )
    assert att.index_anchor == "b" * 64


# ---------------------------------------------------------------------------
# MCP envelope surfaces the attestation next to `degraded`
# ---------------------------------------------------------------------------


def test_mcp_recall_envelope_surfaces_attestation(monkeypatch):
    import mind_mem.mcp.tools.recall as mcp_recall

    ws = _make_workspace()
    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)

    class _FakeHB:
        vector_enabled = True
        vector_available = True

        @staticmethod
        def from_config(config):
            return _FakeHB()

        def search(self, query, workspace, limit=10, active_only=False, **kwargs):
            return _as_results([{"_id": "D-1", "score": 1.0}], None)

    import mind_mem.hybrid_recall as hr

    monkeypatch.setattr(hr, "HybridBackend", _FakeHB)
    # Attestation is injected POST-cache by ``_recall_impl`` (Finding 2). Disable
    # the recall cache so the assertion exercises a clean derivation.
    monkeypatch.setattr(mcp_recall, "_load_config", lambda ws: {"cache": {"enabled": False}})

    raw = mcp_recall._recall_impl("capital of France", limit=5, backend="hybrid")
    envelope = json.loads(raw)
    att = envelope.get("attestation")
    assert att is not None, "envelope must carry an attestation"
    assert att["schema"] == RECALL_ATTEST_TAG
    assert att["legs_ran"] == ["bm25", "hybrid", "vector"]
    assert att["degraded"] is None
    assert att["derivation"] == "derived"
    assert "attestation_hash" in att


def test_mcp_envelope_attestation_reflects_degradation(monkeypatch):
    import mind_mem.mcp.tools.recall as mcp_recall

    ws = _make_workspace()
    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)

    class _FakeHB:
        vector_enabled = True
        vector_available = True

        @staticmethod
        def from_config(config):
            return _FakeHB()

        def search(self, query, workspace, limit=10, active_only=False, **kwargs):
            return _as_results(
                [{"_id": "D-1", "score": 1.0}],
                {"leg": "vector", "reason": "deadline_exceeded"},
            )

    import mind_mem.hybrid_recall as hr

    monkeypatch.setattr(hr, "HybridBackend", _FakeHB)
    monkeypatch.setattr(mcp_recall, "_load_config", lambda ws: {"cache": {"enabled": False}})

    raw = mcp_recall._recall_impl("capital of France", limit=5, backend="hybrid")
    envelope = json.loads(raw)
    # The legacy degraded field still fires loud (bug #139 not regressed).
    assert envelope.get("degraded") == {"leg": "vector", "reason": "deadline_exceeded"}
    # And the attestation records the same fact structurally.
    att = envelope["attestation"]
    assert att["legs_degraded"] == ["vector"]
    assert "vector" not in att["legs_ran"]
    assert att["degraded"] == {"leg": "vector", "reason": "deadline_exceeded"}


def test_mcp_envelope_attestation_internally_consistent(monkeypatch):
    """The envelope attestation is a fully-formed, internally-consistent record
    derived post-cache on the public recall path (Finding 2 placement)."""
    import mind_mem.mcp.tools.recall as mcp_recall

    ws = _make_workspace()
    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)

    class _FakeHB:
        vector_enabled = True
        vector_available = True

        @staticmethod
        def from_config(config):
            return _FakeHB()

        def search(self, query, workspace, limit=10, active_only=False, **kwargs):
            return _as_results([{"_id": "D-1"}], None)

    import mind_mem.hybrid_recall as hr

    monkeypatch.setattr(hr, "HybridBackend", _FakeHB)
    monkeypatch.setattr(mcp_recall, "_load_config", lambda ws: {"cache": {"enabled": False}})

    raw = mcp_recall._recall_impl("q", limit=5, backend="hybrid")
    att = RecallAttestation.from_dict(json.loads(raw)["attestation"])
    assert isinstance(att, RecallAttestation)
    assert att.is_internally_consistent()
    assert att.derivation == "derived"


def test_attestation_reflects_config_toggle_on_cache_hit(monkeypatch):
    """Finding 2: config drift (vector toggled) WITHOUT a governance event must
    not replay a stale attestation on a cache hit. The recall-cache key omits the
    config hash, so a second identical query hits the cache — but the post-cache
    derivation must bind the CURRENT pipeline's config_hash + legs, never the
    cached ones."""
    import mind_mem.mcp.tools.recall as mcp_recall
    from mind_mem.recall_cache import reset_singleton

    reset_singleton()
    ws = _make_workspace()
    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)

    # Live, mutable pipeline config that both the fake backend and the vector-
    # flag probe read. Cache stays ENABLED (that is the whole point).
    state = {"recall": {"vector_enabled": False}, "cache": {"enabled": True}}
    monkeypatch.setattr(mcp_recall, "_load_config", lambda w: state)

    class _FakeHB:
        def __init__(self, enabled):
            self.vector_enabled = enabled
            self.vector_available = enabled

        @staticmethod
        def from_config(config):
            return _FakeHB(bool(config.get("recall", {}).get("vector_enabled", False)))

        def search(self, query, workspace, limit=10, active_only=False, **kwargs):
            return _as_results([{"_id": "D-1"}], None)

    import mind_mem.hybrid_recall as hr

    monkeypatch.setattr(hr, "HybridBackend", _FakeHB)
    # Config hash a pure function of the live flag so the assertion is
    # independent of the pipeline-probe internals.
    monkeypatch.setattr(
        "mind_mem.pipeline_hash.current_pipeline_hash",
        lambda w: "CFG_ON" if state["recall"]["vector_enabled"] else "CFG_OFF",
    )

    # Call 1 — vector OFF, populates the cache.
    raw1 = mcp_recall._recall_impl("same query", limit=5, backend="hybrid")
    env1 = json.loads(raw1)
    att1 = env1["attestation"]
    assert att1["config_hash"] == "CFG_OFF"
    assert "vector" not in att1["legs_ran"]

    # Toggle vector ON — NO cache invalidation (simulating silent config drift).
    state["recall"]["vector_enabled"] = True

    # Call 2 — SAME cache key → cache hit, but attestation reflects the NEW pipeline.
    raw2 = mcp_recall._recall_impl("same query", limit=5, backend="hybrid")
    env2 = json.loads(raw2)
    att2 = env2["attestation"]
    assert att2["config_hash"] == "CFG_ON", "config_hash must be the CURRENT pipeline's, not the cached one"
    assert "vector" in att2["legs_ran"], "legs_ran must reflect the NEW (vector-on) pipeline"
    # Prove we truly served the cached envelope (same result payload).
    assert env2["count"] == env1["count"]
    assert RecallAttestation.from_dict(att2).is_internally_consistent()


# ---------------------------------------------------------------------------
# Guardrails on the record itself
# ---------------------------------------------------------------------------


def test_negative_result_count_rejected():
    with pytest.raises(ValueError):
        build_recall_attestation(
            legs_ran=("bm25",),
            legs_degraded=(),
            config_hash="CFG",
            degraded=None,
            index_anchor=GENESIS_ANCHOR,
            result_count=-1,
            query=QUERY,
        )


def test_leg_tuples_normalised_order_independent():
    a = build_recall_attestation(
        legs_ran=("vector", "bm25", "bm25"),
        legs_degraded=(),
        config_hash="CFG",
        degraded=None,
        index_anchor=GENESIS_ANCHOR,
        result_count=1,
        query=QUERY,
    )
    b = build_recall_attestation(
        legs_ran=("bm25", "vector"),
        legs_degraded=(),
        config_hash="CFG",
        degraded=None,
        index_anchor=GENESIS_ANCHOR,
        result_count=1,
        query=QUERY,
    )
    assert a.legs_ran == ("bm25", "vector")
    assert a.attestation_hash == b.attestation_hash
