"""Regression tests for recall-infra issues #139 and #140.

#139 — the vector/RRF config keys (``rrf_k``, ``bm25_weight``,
       ``vector_enabled``, ``ollama_embed_model``, ``postgres`` …) were absent
       from ``_VALID_RECALL_KEYS`` so a hybrid-authored ``recall`` block logged
       a spurious ``unknown_recall_config_keys`` warning.
       **Narrative correction (v4.2.3, audit finding 6):** that missing-key
       warning was *hygiene only* — ``_load_backend`` never consults
       ``_VALID_RECALL_KEYS`` for backend RESOLUTION (it logs the warning, then
       resolves from ``recall.backend`` / the block-store backend regardless),
       so the whitelist gap did NOT by itself downgrade recall to BM25-only.
       The real, silent BM25-only degradation on a Postgres workspace came from
       the pgvector-engagement defects fixed in v4.2.3 (``VectorBackend.search``
       raising ``ValueError: Unknown provider: postgres`` swallowed to ``[]``;
       an un-backfilled ``embedding`` column mislabelled ``hybrid_pgvector``;
       ``_embed_query`` hardcoding ``embed_ollama``). These tests still assert
       the useful, TRUE properties: the keys are recognised (no spurious
       warning), the config *resolves to hybrid*, and an unavailable-but-
       requested vector backend fails **loud**, never silently.

#140 — the MCP recall tool hung ~30s on a postgres+ollama workspace: the
       embedding HTTP round-trip and the psycopg_pool checkout both had no /
       far-too-generous timeouts (ollama 120s, pool default 30s, libpq
       connect_timeout=0 = forever). These tests assert every one of those
       waits is now bounded and fast-failing.
"""

from __future__ import annotations

import json
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

import mind_mem.block_store_postgres as pg
import mind_mem.hybrid_recall as hr
from mind_mem._recall_constants import _VALID_RECALL_KEYS
from mind_mem.block_store_postgres import PostgresBlockStore
from mind_mem.hybrid_recall import HybridBackend
from mind_mem.recall_vector import VectorBackend, _embed_timeout_seconds

# ---------------------------------------------------------------------------
# #139 — hybrid recall engages; schema keys are valid; unavailable → loud
# ---------------------------------------------------------------------------

# The full vector/RRF/pgvector config surface that a hybrid-backend `recall`
# block legitimately carries. Every key MUST be recognised by the loader,
# otherwise `_load_backend` logs `unknown_recall_config_keys` and the loader
# silently drops back to BM25-only (the #139 defect).
_HYBRID_RECALL_KEYS = [
    "rrf_k",
    "bm25_weight",
    "vector_weight",
    "vector_enabled",
    "onnx_backend",
    "provider",
    "model",
    "ollama_embed_model",
    "postgres",
    "embed_timeout_seconds",
    "vector_deadline_seconds",
]


def test_hybrid_config_keys_are_all_valid() -> None:
    """No hybrid/vector key trips the unknown-key schema-drift warning (#139)."""
    unknown = set(_HYBRID_RECALL_KEYS) - _VALID_RECALL_KEYS
    assert unknown == set(), f"schema drift: keys not in _VALID_RECALL_KEYS: {sorted(unknown)}"


def test_hybrid_recall_block_produces_no_unknown_keys() -> None:
    """A realistic hybrid `recall` config yields zero unknown keys → the loader
    will NOT downgrade to BM25-only."""
    recall_cfg = {
        "backend": "vector",
        "vector_enabled": True,
        "rrf_k": 60,
        "bm25_weight": 1.0,
        "vector_weight": 1.0,
        "provider": "ollama",
        "model": "mxbai-embed-large",
        "ollama_embed_model": "mxbai-embed-large",
        "postgres": {"dsn": "postgresql://localhost/db"},
        "embed_timeout_seconds": 8,
        "vector_deadline_seconds": 14,
    }
    assert set(recall_cfg.keys()) - _VALID_RECALL_KEYS == set()


def test_config_resolves_to_hybrid_not_bm25_fallback() -> None:
    """With vector_enabled=True and the backend importable, the backend
    resolves to hybrid (vector_available=True) — NOT the BM25-only fallback."""
    hb = HybridBackend(
        {
            "vector_enabled": True,
            "rrf_k": 60,
            "bm25_weight": 1.0,
            "vector_weight": 1.0,
        }
    )
    assert hb.vector_available is True


def test_vector_disabled_is_bm25_only_without_warning() -> None:
    """When vector recall is not requested, BM25-only is the correct, quiet
    path — the loud warning must fire ONLY on requested-but-unavailable."""
    with patch.object(hr, "_log") as mlog, patch.object(hr, "metrics"):
        hb = HybridBackend({"vector_enabled": False})
    assert hb.vector_available is False
    warned = [c.args[0] for c in mlog.warning.call_args_list]
    assert "hybrid_vector_requested_but_unavailable" not in warned


def test_vector_requested_but_unavailable_fails_loud() -> None:
    """vector_enabled=True but backend unavailable → LOUD warning + metric,
    never a silent degrade to BM25 (#139 fail-loud rail)."""
    with (
        patch.object(HybridBackend, "_check_vector", return_value=False),
        patch.object(hr, "_log") as mlog,
        patch.object(hr, "metrics") as mmet,
    ):
        hb = HybridBackend({"vector_enabled": True})

    assert hb.vector_available is False
    warned = [c.args[0] for c in mlog.warning.call_args_list]
    assert "hybrid_vector_requested_but_unavailable" in warned
    incremented = [c.args[0] for c in mmet.inc.call_args_list]
    assert "hybrid_vector_requested_but_unavailable" in incremented


# ---------------------------------------------------------------------------
# #140 — every recall wait is bounded and fast-failing
# ---------------------------------------------------------------------------


def test_embed_timeout_default_is_bounded() -> None:
    """The per-request embed timeout defaults to a small, bounded value —
    NOT the old unbounded 120s."""
    assert _embed_timeout_seconds({}) == 8.0
    assert _embed_timeout_seconds(None) == 8.0


def test_embed_timeout_is_configurable_and_clamped() -> None:
    assert _embed_timeout_seconds({"embed_timeout_seconds": 3}) == 3.0
    # Clamp high to a sane upper bound (never re-open the 2-minute hang).
    assert _embed_timeout_seconds({"embed_timeout_seconds": 999}) == 120.0
    # Non-positive / invalid → fall back to default, never 0 (= no timeout).
    assert _embed_timeout_seconds({"embed_timeout_seconds": 0}) == 8.0
    assert _embed_timeout_seconds({"embed_timeout_seconds": "oops"}) == 8.0


def test_vector_deadline_is_bounded_and_configurable() -> None:
    assert HybridBackend({})._vector_deadline_seconds() == 14.0
    assert HybridBackend({"vector_deadline_seconds": 2})._vector_deadline_seconds() == 2.0
    # Clamp high; invalid → default.
    assert HybridBackend({"vector_deadline_seconds": 9999})._vector_deadline_seconds() == 120.0
    assert HybridBackend({"vector_deadline_seconds": -1})._vector_deadline_seconds() == 14.0


def test_embed_ollama_passes_bounded_timeout_to_urlopen() -> None:
    """The embed HTTP round-trip is actually invoked with the bounded timeout
    (proves the bound is wired, not merely computed)."""
    captured: dict[str, object] = {}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self) -> bytes:
            return json.dumps({"embeddings": [[0.1, 0.2, 0.3]]}).encode()

    def _fake_urlopen(req, timeout=None):  # noqa: ANN001
        captured["timeout"] = timeout
        return _FakeResp()

    with patch.object(urllib.request, "urlopen", _fake_urlopen):
        VectorBackend({"provider": "ollama"}).embed_ollama(["hello"])
        assert captured["timeout"] == 8.0
        VectorBackend({"provider": "ollama", "embed_timeout_seconds": 3}).embed_ollama(["hi"])
        assert captured["timeout"] == 3.0


def _install_recording_pool(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch ``_require_psycopg`` to hand back a ConnectionPool that records
    its construction kwargs, and clear the process-wide pool registry."""
    rec: dict = {}

    class _RecordingPool:
        closed = False

        def __init__(self, dsn, **kw):  # noqa: ANN001
            rec["dsn"] = dsn
            rec["kw"] = kw

    monkeypatch.setattr(pg, "_require_psycopg", lambda: (MagicMock(), _RecordingPool))
    pg._pool_registry.clear()
    return rec


def test_postgres_pool_uses_bounded_connect_and_checkout_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pool is built with a bounded libpq connect_timeout AND a checkout
    timeout strictly below psycopg_pool's 30s default (the #140 root cause),
    plus a server-side ``statement_timeout`` so a stuck query is aborted
    rather than waited on forever (audit finding 9)."""
    rec = _install_recording_pool(monkeypatch)
    store = PostgresBlockStore(dsn="postgresql://u:p@localhost:5432/db", schema="mind_mem")
    store._get_pool()

    assert rec["kw"]["kwargs"] == {"connect_timeout": 5, "options": "-c statement_timeout=30000"}
    checkout = rec["kw"]["timeout"]
    assert checkout == 10.0
    assert checkout < 30.0, "checkout timeout must be below psycopg_pool's 30s default"


def test_postgres_pool_respects_dsn_supplied_connect_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the operator already set connect_timeout in the DSN, we do not
    override it (their intent wins) — but the server-side statement_timeout is
    still applied because the DSN carries no ``options``."""
    rec = _install_recording_pool(monkeypatch)
    store = PostgresBlockStore(dsn="postgresql://u:p@h/db?connect_timeout=20", schema="mind_mem")
    store._get_pool()
    assert rec["kw"]["kwargs"] == {"options": "-c statement_timeout=30000"}


def test_postgres_pool_respects_dsn_supplied_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ``options`` already in the DSN (e.g. the ``-csearch_path=`` schema
    form) is never clobbered — statement_timeout is only injected when the
    operator supplied no options of their own (audit finding 9)."""
    rec = _install_recording_pool(monkeypatch)
    store = PostgresBlockStore(dsn="postgresql://u:p@h/db?options=-csearch_path=ws", schema="mind_mem")
    store._get_pool()
    assert "options" not in rec["kw"]["kwargs"] or "search_path" not in rec["kw"]["kwargs"].get("options", "")
    assert rec["kw"]["kwargs"] == {"connect_timeout": 5}


def test_postgres_pool_statement_timeout_env_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    """statement_timeout is env-overridable to any positive value — but a
    non-positive / non-numeric override falls back to the safe default, never
    silently re-opening an unbounded wait (audit finding 9). Operators who
    genuinely want no statement_timeout use their own DSN ``options`` instead
    (see test_postgres_pool_respects_dsn_supplied_options)."""
    monkeypatch.setenv("MIND_MEM_PG_STATEMENT_TIMEOUT_MS", "5000")
    rec = _install_recording_pool(monkeypatch)
    PostgresBlockStore(dsn="postgresql://u:p@h/db", schema="mind_mem")._get_pool()
    assert rec["kw"]["kwargs"] == {"connect_timeout": 5, "options": "-c statement_timeout=5000"}

    # Non-positive / typo → safe default (30s), never 0/forever.
    monkeypatch.setenv("MIND_MEM_PG_STATEMENT_TIMEOUT_MS", "0")
    rec2 = _install_recording_pool(monkeypatch)
    PostgresBlockStore(dsn="postgresql://u:p@h2/db", schema="mind_mem")._get_pool()
    assert rec2["kw"]["kwargs"] == {"connect_timeout": 5, "options": "-c statement_timeout=30000"}


def test_postgres_pool_timeout_env_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators on genuinely slow links can widen the bounds via env — but a
    typo (non-numeric / non-positive) can never re-introduce an unbounded wait."""
    monkeypatch.setenv("MIND_MEM_PG_CONNECT_TIMEOUT", "7")
    monkeypatch.setenv("MIND_MEM_PG_POOL_TIMEOUT", "12")
    rec = _install_recording_pool(monkeypatch)
    PostgresBlockStore(dsn="postgresql://u:p@h/db", schema="mind_mem")._get_pool()
    assert rec["kw"]["kwargs"] == {"connect_timeout": 7, "options": "-c statement_timeout=30000"}
    assert rec["kw"]["timeout"] == 12.0

    # Bad env value → default, never 0/forever.
    monkeypatch.setenv("MIND_MEM_PG_POOL_TIMEOUT", "0")
    monkeypatch.setenv("MIND_MEM_PG_CONNECT_TIMEOUT", "nope")
    rec2 = _install_recording_pool(monkeypatch)
    PostgresBlockStore(dsn="postgresql://u:p@h2/db", schema="mind_mem")._get_pool()
    assert rec2["kw"]["kwargs"] == {"connect_timeout": 5, "options": "-c statement_timeout=30000"}
    assert rec2["kw"]["timeout"] == 10.0
