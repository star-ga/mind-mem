"""Regression tests for the Postgres pgvector recall-engagement audit.

The prior #139 fix threaded a query embedding into ``PostgresBlockStore.
hybrid_search`` but left the real defects in place: on a Postgres workspace
whose ``embedding`` column was never back-filled, the pgvector CTE returned
zero rows, RRF fused BM25 *alone* (top score ``1/(k+1)``), yet every hit was
stamped ``_retrieval_source = "hybrid_pgvector"`` — a silent lie.

These tests assert, WITHOUT a live Postgres (a fake pool/cursor drives the
fusion + labelling logic directly):

* a document present in BOTH the BM25 and the pgvector candidate pools scores
  the genuine TWO-LEG RRF sum (``> 1/(k+1)``, up to ``2/(k+1)``) and is
  labelled ``hybrid_pgvector`` — the proof the vector leg actually engaged;
* the vector leg returning zero rows (un-backfilled column) is labelled
  ``bm25_fallback`` — never ``hybrid_pgvector`` — so the degradation is honest;
* no embedding supplied → ``bm25_only`` (BM25 by design, quiet);
* the MCP ``recall`` tool lifts a ``bm25_fallback`` hit into its ``warnings``
  array — the only degradation surface a caller ever sees.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import mind_mem.block_store_postgres as pg
from mind_mem.block_store_postgres import PostgresBlockStore

RRF_K = 60


class _FakeCursor:
    """Dispatches ``execute`` by SQL shape and replays canned rows."""

    def __init__(self, bm25_rows: list, cos_rows: list, fetch_rows: list) -> None:
        self._bm25 = bm25_rows
        self._cos = cos_rows
        self._fetch = fetch_rows
        self._last: list = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *a: object) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        if "ts_rank" in sql:
            self._last = self._bm25
        elif "embedding <=>" in sql:
            self._last = self._cos
        elif "id = ANY" in sql:
            self._last = self._fetch
        else:
            self._last = []

    def fetchall(self) -> list:
        return list(self._last)


class _FakeConn:
    def __init__(self, cur: _FakeCursor) -> None:
        self._cur = cur

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *a: object) -> bool:
        return False

    def cursor(self, row_factory: Any = None) -> _FakeCursor:
        return self._cur


class _FakePool:
    def __init__(self, cur: _FakeCursor) -> None:
        self._cur = cur

    def connection(self) -> _FakeConn:
        return _FakeConn(self._cur)


class _FakePsycopgRows:
    dict_row = object()


class _FakePsycopg:
    rows = _FakePsycopgRows()


def _make_store(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bm25_rows: list,
    cos_rows: list,
    fetch_rows: list,
) -> PostgresBlockStore:
    """A PostgresBlockStore wired to a fake pool — no live DB, no psycopg."""
    monkeypatch.setattr(pg, "_require_psycopg", lambda: (_FakePsycopg(), None))
    # Return the raw template so the fake cursor can dispatch on plain-string
    # SQL (production returns a psycopg ``Composed``; this keeps the unit test
    # free of a live psycopg while exercising the real fusion + label logic).
    monkeypatch.setattr(pg, "_sql", lambda schema, template: template)
    store = PostgresBlockStore(dsn="postgresql://u:p@localhost:5432/db", schema="mind_mem")
    store._has_vector = True
    store._embedding_dim = 3
    store._schema_ready = True
    cur = _FakeCursor(bm25_rows, cos_rows, fetch_rows)
    monkeypatch.setattr(store, "_ensure_schema", lambda: None)
    monkeypatch.setattr(store, "_get_pool", lambda: _FakePool(cur))
    return store


def _fetch_rows(*ids: str) -> list[dict[str, Any]]:
    return [
        {
            "id": bid,
            "file_path": f"{bid}.md",
            "content": f"body of {bid}",
            "metadata": {"Statement": f"stmt {bid}", "Status": "active"},
            "created_at": None,
            "updated_at": None,
            "active": True,
        }
        for bid in ids
    ]


def test_hybrid_search_two_leg_fusion_scores_and_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """A doc in BOTH pools scores the two-leg RRF sum (> 1/(k+1)) and is
    honestly labelled hybrid_pgvector — the real vector-engagement proof."""
    # "A" ranks #1 in BOTH BM25 and pgvector → 1/(k+1) + 1/(k+1) = 2/(k+1).
    store = _make_store(
        monkeypatch,
        bm25_rows=[("A", 0.9), ("B", 0.5)],
        cos_rows=[("A", 0.10), ("C", 0.20)],
        fetch_rows=_fetch_rows("A", "B", "C"),
    )
    out = store.hybrid_search("q", query_embedding=[0.1, 0.2, 0.3], limit=10, rrf_k=RRF_K)

    by_id = {b["_id"]: b for b in out}
    single_leg = 1.0 / (RRF_K + 1)  # 0.016393…
    two_leg = 2.0 / (RRF_K + 1)  # 0.032787…

    # The two-leg document scores the fused sum — NOT the single-leg 1/(k+1)
    # that a silently-BM25-only path produced.
    assert by_id["A"]["_score"] == pytest.approx(two_leg, abs=1e-9)
    assert by_id["A"]["_score"] > single_leg + 1e-9
    # Vector leg genuinely contributed → honest label on every fused hit.
    assert by_id["A"]["_retrieval_source"] == "hybrid_pgvector"
    assert by_id["B"]["_retrieval_source"] == "hybrid_pgvector"
    # A single-leg doc (B: BM25 rank 2 only) scores exactly 1/(k+2) and
    # strictly below the two-leg doc — the RRF fusion math is real, not a flat
    # 1/(k+1) stamp.
    assert by_id["B"]["_score"] == pytest.approx(1.0 / (RRF_K + 2), abs=1e-9)
    assert by_id["B"]["_score"] < by_id["A"]["_score"]


def test_hybrid_search_labels_bm25_fallback_when_vector_leg_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding supplied + pgvector present but the CTE returns 0 rows
    (un-backfilled column) → bm25_fallback, NEVER hybrid_pgvector."""
    store = _make_store(
        monkeypatch,
        bm25_rows=[("A", 0.9), ("B", 0.5)],
        cos_rows=[],  # every row NULL → no vector contribution
        fetch_rows=_fetch_rows("A", "B"),
    )
    out = store.hybrid_search("q", query_embedding=[0.1, 0.2, 0.3], limit=10, rrf_k=RRF_K)

    assert out, "BM25 rows must still return results"
    assert all(b["_retrieval_source"] == "bm25_fallback" for b in out)
    # And the score is honest single-leg BM25 (1/(k+1)) — the exact symptom.
    assert out[0]["_score"] == pytest.approx(1.0 / (RRF_K + 1), abs=1e-9)


def test_hybrid_search_labels_bm25_only_when_no_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """No embedding supplied → BM25 by design, quietly labelled bm25_only."""
    store = _make_store(
        monkeypatch,
        bm25_rows=[("A", 0.9)],
        cos_rows=[("X", 0.1)],  # must be ignored: do_vector is False
        fetch_rows=_fetch_rows("A"),
    )
    out = store.hybrid_search("q", query_embedding=None, limit=10, rrf_k=RRF_K)
    assert out[0]["_retrieval_source"] == "bm25_only"


def test_vector_search_returns_pgvector_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    """The new vector-only path (audit 1a) returns cosine hits labelled
    pgvector instead of raising ``ValueError: Unknown provider: postgres``."""
    store = _make_store(
        monkeypatch,
        bm25_rows=[],
        cos_rows=[],
        fetch_rows=[],
    )
    # vector_search runs its own cosine SELECT; drive it via the fake cursor.
    cur = _FakeCursor(
        bm25_rows=[],
        cos_rows=[],
        fetch_rows=[],
    )
    cur._cos = [  # dict_row shape (SELECT ... embedding <=> ...)
        {
            "id": "V1",
            "file_path": "V1.md",
            "content": "vec body",
            "metadata": {"Statement": "vec stmt"},
            "created_at": None,
            "updated_at": None,
            "active": True,
            "dist": 0.25,
        }
    ]
    monkeypatch.setattr(store, "_get_pool", lambda: _FakePool(cur))
    out = store.vector_search([0.1, 0.2, 0.3], limit=5)
    assert out and out[0]["_id"] == "V1"
    assert out[0]["_retrieval_source"] == "pgvector"
    assert out[0]["_score"] == pytest.approx(1.0 - 0.25, abs=1e-9)


def test_mcp_recall_warns_on_bm25_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """The honest bm25_fallback label bubbles into the MCP recall tool's
    ``warnings`` array — the only degradation surface a caller sees
    (audit findings 1b + 7)."""
    from mind_mem.mcp.tools import recall as mcp_recall

    ws = str(tmp_path)

    class _FakeHybrid:
        @staticmethod
        def from_config(config: dict) -> _FakeHybrid:
            return _FakeHybrid()

        def search(self, query: str, workspace: str, limit: int = 10, active_only: bool = False) -> list[dict]:
            return [{"_id": "A", "score": 0.0164, "_retrieval_source": "bm25_fallback", "excerpt": "x"}]

    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)
    monkeypatch.setattr(mcp_recall, "_get_limits", lambda w: {"max_recall_results": 10, "query_timeout_seconds": 30})
    monkeypatch.setattr(mcp_recall, "_load_config", lambda w: {"recall": {"backend": "hybrid", "provider": "postgres"}})
    import mind_mem.hybrid_recall as hr

    monkeypatch.setattr(hr, "HybridBackend", _FakeHybrid)
    monkeypatch.setattr(hr, "validate_recall_config", lambda cfg: [])

    out = json.loads(mcp_recall._recall_impl_uncached("some query", limit=5, backend="hybrid"))
    warnings = out.get("warnings", [])
    assert any("BM25-only" in w and "pgvector" in w for w in warnings), f"missing degradation warning: {warnings}"
