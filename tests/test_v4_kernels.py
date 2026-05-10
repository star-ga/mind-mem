"""Tests for the v4 kernel strategy implementations.

Covers all 5 strategies registered in mind_mem.v4.kernels:
surprise_weighted, lineage_first, recent_first, contradicts_first,
graph_walk. Each has graceful-degrade tests when the underlying state
(lineage table, tier table, embeddings, centroid) is absent.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from mind_mem.v4 import FeatureDisabledError
from mind_mem.v4.cognitive_kernel import KernelHit, KernelKind, KernelResult, mind_recall
from mind_mem.v4.kernels import (  # noqa: F401  — import to trigger auto-registration
    contradicts_first_kernel,
    graph_walk_kernel,
    lineage_first_kernel,
    recent_first_kernel,
    surprise_weighted_kernel,
)


@pytest.fixture
def cfg_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = {
        "v4": {
            "cognitive_kernel": {"enabled": True},
            "tier_memory": {"enabled": True},
            "surprise_retrieval": {"enabled": True},
        }
    }
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


@pytest.fixture
def fake_v3_recall(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Stub the v3 recall path so default-kernel returns deterministic hits."""
    hits = [
        {"_id": "B-1", "rrf_score": 0.9},
        {"_id": "B-2", "rrf_score": 0.7},
        {"_id": "B-3", "rrf_score": 0.5},
        {"_id": "B-4", "rrf_score": 0.4},
        {"_id": "B-5", "rrf_score": 0.3},
    ]

    class _Mod:
        @staticmethod
        def recall(_w: str, _q: str) -> list[dict]:
            return list(hits)

    sys.modules["mind_mem._recall_core"] = _Mod  # type: ignore[assignment]
    yield hits
    sys.modules.pop("mind_mem._recall_core", None)


# ---------------------------------------------------------------------------
# Auto-registration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_all_five_strategies_registered(cfg_on: Path, fake_v3_recall: list[dict]) -> None:
    """After importing the kernels module, every named strategy is reachable."""
    for k in (
        KernelKind.SURPRISE_WEIGHTED,
        KernelKind.LINEAGE_FIRST,
        KernelKind.RECENT_FIRST,
        KernelKind.CONTRADICTS_FIRST,
        KernelKind.GRAPH_WALK,
    ):
        out = mind_recall("/tmp/ws", "q", kernel=k)
        assert out.kernel is k


# ---------------------------------------------------------------------------
# surprise_weighted
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_surprise_weighted_degrades_without_inputs(cfg_on: Path, fake_v3_recall: list[dict]) -> None:
    out = surprise_weighted_kernel("/tmp/ws", "q")
    assert out.kernel is KernelKind.SURPRISE_WEIGHTED
    assert out.metadata.get("degraded") is True
    # Falls back to DEFAULT shape — same hit IDs as the v3 stub.
    assert [h.block_id for h in out.hits] == ["B-1", "B-2", "B-3", "B-4", "B-5"]


@pytest.mark.unit
def test_surprise_weighted_reranks_by_distance(cfg_on: Path, fake_v3_recall: list[dict]) -> None:
    centroid = [1.0, 0.0, 0.0]
    embeddings = {
        "B-1": [1.0, 0.0, 0.0],  # surprise = 0 (identical)
        "B-2": [0.0, 1.0, 0.0],  # surprise = 0.5 (orthogonal)
        "B-3": [-1.0, 0.0, 0.0],  # surprise = 1.0 (opposite)
        "B-4": [0.5, 0.5, 0.0],  # surprise ≈ 0.146
        "B-5": [-0.7, 0.7, 0.0],  # surprise ≈ 0.85
    }
    out = surprise_weighted_kernel("/tmp/ws", "q", context_centroid=centroid, candidate_embeddings=embeddings)
    # Ranked descending by surprise.
    assert [h.block_id for h in out.hits] == ["B-3", "B-5", "B-2", "B-4", "B-1"]
    assert out.hits[0].score == pytest.approx(1.0)
    assert out.hits[-1].score == pytest.approx(0.0, abs=1e-9)
    # Reason tag carries the surprise number.
    assert out.hits[0].reason.startswith("surprise_weighted:s=")


@pytest.mark.unit
def test_surprise_weighted_missing_embedding_uses_mild(cfg_on: Path, fake_v3_recall: list[dict]) -> None:
    """Candidate without an embedding entry scores mild surprise (0.5)."""
    out = surprise_weighted_kernel(
        "/tmp/ws",
        "q",
        context_centroid=[1.0, 0.0],
        candidate_embeddings={"B-1": [1.0, 0.0]},
    )
    by_id = {h.block_id: h.score for h in out.hits}
    assert by_id["B-1"] == pytest.approx(0.0)
    assert by_id["B-2"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# lineage_first
# ---------------------------------------------------------------------------


def _make_co_retrieval(workspace: Path, edges: list[tuple[str, str, str]] | None = None) -> None:
    """Create a v3.11-style co_retrieval table.

    edges: list of (mem1_id, mem2_id, kind). When None, only the table
    schema is created (no rows).
    """
    db = workspace / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS co_retrieval ("
            "mem1_id TEXT, mem2_id TEXT, weight REAL DEFAULT 1.0, "
            "kind TEXT NOT NULL DEFAULT 'cooccurrence')"
        )
        if edges:
            conn.executemany(
                "INSERT INTO co_retrieval (mem1_id, mem2_id, kind) VALUES (?, ?, ?)",
                edges,
            )
        conn.commit()


@pytest.mark.unit
def test_lineage_first_degrades_without_table(cfg_on: Path, fake_v3_recall: list[dict]) -> None:
    out = lineage_first_kernel(str(cfg_on), "q")
    assert out.metadata.get("degraded") is True
    assert [h.block_id for h in out.hits] == ["B-1", "B-2", "B-3", "B-4", "B-5"]


@pytest.mark.unit
def test_lineage_first_promotes_well_connected(cfg_on: Path, fake_v3_recall: list[dict]) -> None:
    _make_co_retrieval(
        cfg_on,
        edges=[
            ("B-3", "B-100", "cites"),
            ("B-3", "B-101", "cites"),
            ("B-3", "B-102", "implements"),
            ("B-1", "B-103", "cites"),
        ],
    )
    out = lineage_first_kernel(str(cfg_on), "q")
    by_id = {h.block_id: h.score for h in out.hits}
    # B-3 starts at 0.5 base + 3 outgoing edges → 0.5 * 1.3 = 0.65
    # B-1 starts at 0.9 base + 1 outgoing edge  → 0.9 * 1.1 = 0.99
    assert by_id["B-3"] == pytest.approx(0.5 * 1.3)
    assert by_id["B-1"] == pytest.approx(0.9 * 1.1)
    # B-1 still ranks above B-3 because base_score dominates.
    assert out.hits[0].block_id == "B-1"


# ---------------------------------------------------------------------------
# recent_first
# ---------------------------------------------------------------------------


def _make_tier_table(workspace: Path, rows: list[tuple[str, str]]) -> None:
    """rows: (block_id, last_seen_at_iso)."""
    db = workspace / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS block_recall_tier ("
            "block_id TEXT PRIMARY KEY, tier TEXT NOT NULL DEFAULT 'warm', "
            "last_seen_at TEXT NOT NULL, promoted_count INTEGER NOT NULL DEFAULT 0, "
            "last_surprise REAL, block_version INTEGER NOT NULL DEFAULT 0)"
        )
        conn.executemany(
            "INSERT INTO block_recall_tier (block_id, tier, last_seen_at) VALUES (?, 'warm', ?)",
            rows,
        )
        conn.commit()


@pytest.mark.unit
def test_recent_first_degrades_without_tier_table(cfg_on: Path, fake_v3_recall: list[dict]) -> None:
    out = recent_first_kernel(str(cfg_on), "q")
    assert out.metadata.get("degraded") is True


@pytest.mark.unit
def test_recent_first_degrades_when_tier_table_empty(cfg_on: Path, fake_v3_recall: list[dict]) -> None:
    _make_tier_table(cfg_on, [])
    out = recent_first_kernel(str(cfg_on), "q")
    assert out.metadata.get("degraded") is True


@pytest.mark.unit
def test_recent_first_boosts_recently_seen(cfg_on: Path, fake_v3_recall: list[dict]) -> None:
    _make_tier_table(
        cfg_on,
        [
            ("B-5", "2026-05-10T10:00:00Z"),  # most recent
            ("B-3", "2026-05-10T05:00:00Z"),
            ("B-1", "2026-05-09T12:00:00Z"),  # oldest
        ],
    )
    out = recent_first_kernel(str(cfg_on), "q")
    by_id = {h.block_id: h.score for h in out.hits}
    # B-5 base 0.3 + bonus (1 - 0/3) = 1.0 → 1.3
    # B-3 base 0.5 + bonus (1 - 1/3) ≈ 0.667 → 1.167
    # B-1 base 0.9 + bonus (1 - 2/3) ≈ 0.333 → 1.233
    # B-2, B-4 base + 0
    assert by_id["B-5"] == pytest.approx(0.3 + 1.0)
    assert by_id["B-3"] == pytest.approx(0.5 + (1 - 1.0 / 3.0))
    assert by_id["B-1"] == pytest.approx(0.9 + (1 - 2.0 / 3.0))
    assert by_id["B-2"] == pytest.approx(0.7)  # no bonus
    # B-5 has highest combined score.
    assert out.hits[0].block_id == "B-5"


# ---------------------------------------------------------------------------
# contradicts_first
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_contradicts_first_degrades_without_table(cfg_on: Path, fake_v3_recall: list[dict]) -> None:
    out = contradicts_first_kernel(str(cfg_on), "q")
    assert out.metadata.get("degraded") is True


@pytest.mark.unit
def test_contradicts_first_degrades_on_untyped_lineage(cfg_on: Path, fake_v3_recall: list[dict]) -> None:
    """A v2.6.0 graph with no `kind` column degrades cleanly."""
    db = cfg_on / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE co_retrieval (mem1_id TEXT, mem2_id TEXT, weight REAL DEFAULT 1.0)")
        conn.commit()
    out = contradicts_first_kernel(str(cfg_on), "q")
    assert out.metadata.get("degraded") is True
    assert out.metadata["reason"] == "untyped_lineage"


@pytest.mark.unit
def test_contradicts_first_boosts_contradicts_endpoints(cfg_on: Path, fake_v3_recall: list[dict]) -> None:
    _make_co_retrieval(
        cfg_on,
        edges=[
            ("B-1", "B-99", "contradicts"),  # B-1 boosted
            ("B-99", "B-3", "contradicts"),  # B-3 boosted (mem2 side)
            ("B-2", "B-98", "cites"),  # cites — no boost
        ],
    )
    out = contradicts_first_kernel(str(cfg_on), "q")
    by_id = {h.block_id: h.score for h in out.hits}
    # B-1: base 0.9 + 1.0 = 1.9 ; B-3: base 0.5 + 1.0 = 1.5
    assert by_id["B-1"] == pytest.approx(1.9)
    assert by_id["B-3"] == pytest.approx(1.5)
    assert by_id["B-2"] == pytest.approx(0.7)  # cites doesn't boost
    # B-1 has the largest contradicts boost, ranks first.
    assert out.hits[0].block_id == "B-1"
    # Reason tag distinguishes hit vs miss.
    assert any(h.reason == "contradicts_first:hit" for h in out.hits)
    assert any(h.reason == "contradicts_first:miss" for h in out.hits)


# ---------------------------------------------------------------------------
# graph_walk
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_graph_walk_degrades_without_table(cfg_on: Path, fake_v3_recall: list[dict]) -> None:
    out = graph_walk_kernel(str(cfg_on), "q")
    assert out.metadata.get("degraded") is True


@pytest.mark.unit
def test_graph_walk_uses_default_hits_as_seeds(cfg_on: Path, fake_v3_recall: list[dict]) -> None:
    _make_co_retrieval(
        cfg_on,
        edges=[
            ("B-1", "B-10", "cites"),
            ("B-10", "B-100", "cites"),
        ],
    )
    out = graph_walk_kernel(str(cfg_on), "q", max_hops=2, max_nodes=20)
    # Visited: B-1 (seed, hop 0), B-10 (hop 1), B-100 (hop 2 — but max_hops
    # check fires AT hop 2, so B-100 is enqueued at hop 1+1=2, rejected
    # by the `if hop >= max_hops` guard before it expands. So B-100 IS
    # visited (added to visited dict) but its neighbours aren't expanded.
    ids = {h.block_id for h in out.hits}
    assert "B-1" in ids
    assert "B-10" in ids
    assert "B-100" in ids


@pytest.mark.unit
def test_graph_walk_with_explicit_seeds(cfg_on: Path, fake_v3_recall: list[dict]) -> None:
    _make_co_retrieval(
        cfg_on,
        edges=[
            ("X-1", "X-2", "cites"),
            ("X-1", "X-3", "implements"),
            ("X-2", "X-4", "refines"),
        ],
    )
    out = graph_walk_kernel(str(cfg_on), "q", seed_ids=["X-1"], max_hops=2, max_nodes=20)
    ids = {h.block_id for h in out.hits}
    assert ids == {"X-1", "X-2", "X-3", "X-4"}
    # X-1 (seed) ranks highest at score 1/(0+1) = 1.0.
    assert out.hits[0].block_id == "X-1"
    assert out.hits[0].score == pytest.approx(1.0)


@pytest.mark.unit
def test_graph_walk_respects_max_nodes(cfg_on: Path, fake_v3_recall: list[dict]) -> None:
    edges = [(f"N-{i}", f"N-{i + 1}", "cites") for i in range(50)]
    _make_co_retrieval(cfg_on, edges=edges)
    out = graph_walk_kernel(str(cfg_on), "q", seed_ids=["N-0"], max_hops=10, max_nodes=5)
    assert len(out.hits) <= 5


@pytest.mark.unit
def test_graph_walk_no_seeds_no_default_hits(cfg_on: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty seed list and v3 recall returns nothing → empty result."""

    class _NoOp:
        @staticmethod
        def recall(_w: str, _q: str) -> list[dict]:
            return []

    sys.modules["mind_mem._recall_core"] = _NoOp  # type: ignore[assignment]
    try:
        _make_co_retrieval(cfg_on, edges=[("X", "Y", "cites")])
        out = graph_walk_kernel(str(cfg_on), "q")
        assert out.hits == []
    finally:
        sys.modules.pop("mind_mem._recall_core", None)


# ---------------------------------------------------------------------------
# Dispatcher round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mind_recall_routes_through_each_strategy(cfg_on: Path, fake_v3_recall: list[dict]) -> None:
    """End-to-end: register five → dispatch via mind_recall → get back KernelResult."""
    for k in (
        KernelKind.SURPRISE_WEIGHTED,
        KernelKind.LINEAGE_FIRST,
        KernelKind.RECENT_FIRST,
        KernelKind.CONTRADICTS_FIRST,
        KernelKind.GRAPH_WALK,
    ):
        out = mind_recall(str(cfg_on), "q", kernel=k)
        assert isinstance(out, KernelResult)
        assert out.kernel is k


# Type-only import keeper — silences unused warnings on the explicit
# import the linter would otherwise flag as side-effect-only.
def _kernel_hit_keepalive() -> KernelHit:
    return KernelHit(block_id="x", score=0.0)


@pytest.mark.unit
def test_kernels_are_idempotently_re_registerable(cfg_on: Path, fake_v3_recall: list[dict]) -> None:
    """Importing the kernels module again is safe (replace-not-accumulate)."""
    import importlib

    import mind_mem.v4.kernels as kernels_mod

    importlib.reload(kernels_mod)
    out = mind_recall(str(cfg_on), "q", kernel=KernelKind.LINEAGE_FIRST)
    assert out.kernel is KernelKind.LINEAGE_FIRST


@pytest.mark.unit
def test_flag_off_blocks_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Even though strategies are auto-registered at import time, dispatch is gated."""
    cfg = {"v4": {"cognitive_kernel": {"enabled": False}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    with pytest.raises(FeatureDisabledError):
        mind_recall("/tmp/ws", "q", kernel=KernelKind.LINEAGE_FIRST)
