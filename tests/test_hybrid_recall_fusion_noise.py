#!/usr/bin/env python3
"""Regression gate for the hybrid-recall NOISE bug (empty BM25 arm → 1/(k+1) floor).

The bug: hybrid recall returned a uniform ~0.016 = 1/(60+1) — the RRF
single-arm floor — because one lexical arm queried an existing-but-EMPTY
index while the vector arm was healthy. The degrade was ASYMMETRIC: a failed
vector arm surfaced a ``degraded`` marker, but an empty BM25 arm was silently
indistinguishable from a legitimate zero-match, so fusion collapsed to noise
with no signal.

These tests lock in the fix:
  * ``rrf_fuse`` annotates ``fusion_sources`` (which arms + ranks a hit came
    from) so a single-arm noise hit is distinguishable from a fused hit.
  * an empty BM25 arm while the store has blocks now degrades LOUD via the
    same ``.degraded`` plumbing the vector leg uses.
  * ``recall.strict_hybrid=true`` turns that structural failure into a raise.
  * a genuinely two-arm fused recall clears the 1/(k+1) single-arm floor.
"""

from __future__ import annotations

import json
import os

import pytest

from mind_mem.hybrid_recall import BM25LegError, HybridBackend, rrf_fuse
from mind_mem.sqlite_index import _db_path, _init_schema, build_index

_RRF_K = 60
_SINGLE_ARM_FLOOR = 1.0 / (_RRF_K + 1)  # ~0.016393 — the noise the bug produced


# ---------------------------------------------------------------------------
# Unit: rrf_fuse fusion_sources annotation (step 3)
# ---------------------------------------------------------------------------


def test_rrf_fuse_annotates_fusion_sources() -> None:
    bm25 = [{"_id": "A"}, {"_id": "B"}]
    vec = [{"_id": "B"}, {"_id": "C"}]
    fused = rrf_fuse([bm25, vec], [1.0, 1.0], k=_RRF_K, source_names=["bm25", "vector"])
    by_id = {r["_id"]: r for r in fused}

    # Per-arm 1-based ranks recorded for every hit.
    assert by_id["A"]["fusion_sources"] == {"bm25": 1}
    assert by_id["B"]["fusion_sources"] == {"bm25": 2, "vector": 1}
    assert by_id["C"]["fusion_sources"] == {"vector": 2}

    # B is the only genuinely-fused (>=2 arm) hit → ranks first and clears
    # the single-arm floor; single-arm hits sit AT the floor.
    assert fused[0]["_id"] == "B"
    assert by_id["B"]["rrf_score"] > _SINGLE_ARM_FLOOR
    assert by_id["A"]["rrf_score"] == pytest.approx(_SINGLE_ARM_FLOOR, abs=1e-6)


# ---------------------------------------------------------------------------
# Helpers for the end-to-end workspace tests
# ---------------------------------------------------------------------------


def _write_config(ws: str) -> None:
    with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "recall": {
                    "backend": "sqlite",
                    "cross_encoder": {"enabled": False, "auto_enable": False},
                    "dedup": {"enabled": False},
                }
            },
            fh,
        )


def _seed_and_build(ws: str) -> None:
    """Seed a Markdown corpus with ONE block carrying a distinctive token,
    then build the real sqlite FTS index (the healthy BM25 arm)."""
    decisions = os.path.join(ws, "decisions")
    os.makedirs(decisions, exist_ok=True)
    body = (
        "[DEC-1]\nStatement: byteidentical crossplatform determinism wedge\n"
        "Date: 2026-08-18\nStatus: active\nTags: WEDGE\n\n---\n\n"
        "[DEC-2]\nStatement: unrelated alpha planning topic\n"
        "Date: 2026-08-18\nStatus: active\nTags: MISC\n\n---\n\n"
        "[DEC-3]\nStatement: unrelated beta scheduling topic\n"
        "Date: 2026-08-18\nStatus: active\nTags: MISC\n\n---\n\n"
    )
    with open(os.path.join(decisions, "DECISIONS.md"), "w", encoding="utf-8") as fh:
        fh.write(body)
    _write_config(ws)
    build_index(ws, incremental=False)


def _healthy_vector(ids: list[str]):
    """A deterministic stand-in for a healthy vector arm (no embedder)."""

    def _mock(query: str, workspace: str, limit: int = 200, active_only: bool = False, **kwargs):
        return [{"_id": i, "score": 1.0 - n * 0.01} for n, i in enumerate(ids)]

    return _mock


# ---------------------------------------------------------------------------
# E2E: a genuinely two-arm fused recall clears the noise floor
# ---------------------------------------------------------------------------


def test_fused_recall_discriminates(tmp_path, monkeypatch) -> None:
    ws = str(tmp_path)
    _seed_and_build(ws)
    os.environ.setdefault("MIND_MEM_DISABLE_TELEMETRY", "1")

    hb = HybridBackend(config={"vector_enabled": True, "rrf_k": _RRF_K, "bm25_weight": 1.0, "vector_weight": 1.0})
    # Force a healthy vector arm without a real embedder; DEC-1 is top in
    # both arms so it is genuinely fused, not single-arm noise.
    hb._vector_available = True
    monkeypatch.setattr(hb, "_vector_search", _healthy_vector(["DEC-1", "DEC-2", "DEC-3"]))

    result = hb.search("byteidentical", ws, limit=10)

    # (a) The full pipeline ran — no silent degradation.
    assert getattr(result, "degraded", None) is None
    assert len(result) >= 1

    # (b) At least one hit is genuinely fused (>=2 arms).
    fused_hits = [h for h in result if len(h.get("fusion_sources", {})) >= 2]
    assert fused_hits, f"no multi-arm fused hit; sources={[h.get('fusion_sources') for h in result]}"

    # (c) The top hit clears the 1/(k+1) single-arm floor — NOT the uniform
    #     ~0.016 noise the bug produced.
    top = result[0]
    assert set(top["fusion_sources"]) >= {"bm25", "vector"}
    threshold = (hb.bm25_weight + hb.vector_weight) / (_RRF_K + 1) - 1e-3
    assert top["rrf_score"] >= threshold, f"top rrf_score {top['rrf_score']} <= floor {_SINGLE_ARM_FLOOR}"


# ---------------------------------------------------------------------------
# Guard: an empty BM25 arm while the store has blocks degrades LOUD
# ---------------------------------------------------------------------------


def _make_empty_fts_db(ws: str) -> None:
    import sqlite3

    db = _db_path(ws)
    os.makedirs(os.path.dirname(db), exist_ok=True)
    conn = sqlite3.connect(db)
    _init_schema(conn)  # schema present, blocks_fts EMPTY
    conn.commit()
    conn.close()


def test_empty_bm25_arm_degrades_loud(tmp_path, monkeypatch) -> None:
    ws = str(tmp_path)
    _write_config(ws)
    _make_empty_fts_db(ws)
    # Store HAS blocks (the exact bug condition: index empty, store full).
    monkeypatch.setattr("mind_mem.storage.iter_active_blocks", lambda w, config=None: [{"_id": "X"}])

    hb = HybridBackend(config={"vector_enabled": True, "rrf_k": _RRF_K})
    hb._vector_available = True
    monkeypatch.setattr(hb, "_vector_search", _healthy_vector(["V1", "V2"]))

    result = hb.search("anything", ws, limit=10)

    marker = getattr(result, "degraded", None)
    assert marker is not None, "empty BM25 arm must not be silent"
    assert "bm25" in marker.get("leg", "")
    assert "index_empty" in marker.get("reason", "")


def test_legit_zero_match_stays_silent(tmp_path, monkeypatch) -> None:
    """A POPULATED FTS that simply matches nothing must NOT degrade."""
    ws = str(tmp_path)
    _seed_and_build(ws)  # FTS populated with DEC-1..3
    hb = HybridBackend(config={"vector_enabled": True, "rrf_k": _RRF_K})
    hb._vector_available = True
    monkeypatch.setattr(hb, "_vector_search", _healthy_vector(["V1"]))

    # A token that exists in no block → bm25 zero-match, but the index is
    # healthy, so this is legitimate, not a structural failure.
    result = hb.search("zzzznonexistenttoken", ws, limit=10)
    marker = getattr(result, "degraded", None)
    assert marker is None or "bm25" not in marker.get("leg", "")


# ---------------------------------------------------------------------------
# Strict knob: a structurally-empty arm raises instead of degrading (step 4)
# ---------------------------------------------------------------------------


def test_strict_hybrid_raises_on_empty_arm(tmp_path, monkeypatch) -> None:
    ws = str(tmp_path)
    _write_config(ws)
    _make_empty_fts_db(ws)
    monkeypatch.setattr("mind_mem.storage.iter_active_blocks", lambda w, config=None: [{"_id": "X"}])

    # vector disabled → the simple BM25-only path; strict flips degrade→raise.
    hb = HybridBackend(config={"vector_enabled": False, "strict_hybrid": True, "rrf_k": _RRF_K})
    with pytest.raises(BM25LegError):
        hb.search("anything", ws, limit=10)
