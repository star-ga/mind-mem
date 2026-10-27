"""The corpus stamp covers the vector-side retrieval inputs too.

Hybrid and vector recall read the local vector index
(``<ws>/<recall.index_path>/index.json``, default ``.mind-mem-vectors``), an
optional custom ``recall.sqlite_vec_db`` and the v4 PQ side store
(``<ws>/index.db``). A vector-only rebuild (``recall_vector.rebuild_index``,
``python -m mind_mem.recall_vector --index``) or an out-of-band edit of the
vector index writes only those files. The stamp used to miss them, so the cache
kept serving the old ranking until the TTL ran out.

Every negative assertion is paired with a positive control proving the cache
genuinely held the answer when nothing changed.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from mind_mem.recall_cache import corpus_stamp, reset_singleton
from tests.test_recall_cache_corpus_stamp import Q, _counting_engine, _seed


@pytest.fixture(autouse=True)
def _fresh_singletons(monkeypatch):
    from mind_mem import prefetch

    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    reset_singleton()
    prefetch.reset_cache()
    yield
    reset_singleton()
    prefetch.reset_cache()


def _write_aged(ws: str, rel: str, payload: bytes) -> None:
    path = os.path.join(ws, *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(payload)
    past = time.time() - 600
    os.utime(path, (past, past))


def _set_recall_cfg(ws: str, **fields: object) -> dict:
    cfg_path = os.path.join(ws, "mind-mem.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg.setdefault("recall", {}).update(fields)
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    past = time.time() - 3600
    os.utime(cfg_path, (past, past))
    return cfg


class TestTheStampSeesTheVectorInputs:
    @pytest.mark.parametrize(
        "rel",
        [
            ".mind-mem-vectors/index.json",
            "index.db",
            "index.db-wal",
        ],
    )
    def test_default_vector_inputs_move_the_stamp(self, tmp_path, rel) -> None:
        ws = _seed(tmp_path, index=True)
        _write_aged(ws, rel, b'{"v": 1}')
        before = corpus_stamp(ws)
        assert before is not None
        _write_aged(ws, rel, b'{"v": 22}')
        after = corpus_stamp(ws)
        assert after is not None
        assert after != before

    def test_a_vector_index_appearing_moves_the_stamp(self, tmp_path) -> None:
        ws = _seed(tmp_path, index=True)
        before = corpus_stamp(ws)
        _write_aged(ws, ".mind-mem-vectors/index.json", b"{}")
        assert corpus_stamp(ws) not in (None, before)

    def test_configured_index_path_moves_the_stamp(self, tmp_path) -> None:
        ws = _seed(tmp_path, index=True)
        cfg = _set_recall_cfg(ws, index_path="custom-vec")
        _write_aged(ws, "custom-vec/index.json", b"{}")
        before = corpus_stamp(ws, cfg)
        assert before is not None
        _write_aged(ws, "custom-vec/index.json", b'{"changed": true}')
        assert corpus_stamp(ws, cfg) not in (None, before)

    def test_configured_sqlite_vec_db_moves_the_stamp(self, tmp_path) -> None:
        ws = _seed(tmp_path, index=True)
        cfg = _set_recall_cfg(ws, sqlite_vec_db="vec/vectors.db")
        _write_aged(ws, "vec/vectors.db", b"a")
        before = corpus_stamp(ws, cfg)
        assert before is not None
        _write_aged(ws, "vec/vectors.db", b"bb")
        assert corpus_stamp(ws, cfg) not in (None, before)

    def test_malformed_recall_config_is_tolerated(self, tmp_path) -> None:
        ws = _seed(tmp_path, index=True)
        base = corpus_stamp(ws)
        for cfg in (None, {}, {"recall": "x"}, {"recall": {"sqlite_vec_db": ""}}):
            assert corpus_stamp(ws, cfg) == base
        # A non-string index_path is not ignored: VectorBackend reads
        # ``str(index_path)``, so the stamp covers that directory too
        # (pinned in tests/test_recall_cache_vector_path_parity.py).
        assert corpus_stamp(ws, {"recall": {"index_path": 7}}) is not None


class TestNoStaleAnswerAfterAVectorOnlyWrite:
    def _recall_twice_around(self, ws: str, monkeypatch, write) -> list[str]:
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.recall import _recall_impl

        calls = _counting_engine(monkeypatch)
        with use_workspace(ws):
            _recall_impl(Q, limit=1, backend="bm25")
            write()
            _recall_impl(Q, limit=1, backend="bm25")
        return calls

    def test_positive_control_no_write_hits(self, tmp_path, monkeypatch) -> None:
        ws = _seed(tmp_path, index=True)
        _write_aged(ws, ".mind-mem-vectors/index.json", b"{}")
        calls = self._recall_twice_around(ws, monkeypatch, lambda: None)
        assert len(calls) == 1, "vacuous: the cache never held the answer"

    def test_out_of_band_vector_index_edit_misses(self, tmp_path, monkeypatch) -> None:
        ws = _seed(tmp_path, index=True)
        _write_aged(ws, ".mind-mem-vectors/index.json", b"{}")
        calls = self._recall_twice_around(ws, monkeypatch, lambda: _write_aged(ws, ".mind-mem-vectors/index.json", b'{"dropped": 1}'))
        assert len(calls) == 2, "stale cached answer served after a vector-only index write"

    def test_configured_vector_index_edit_misses(self, tmp_path, monkeypatch) -> None:
        ws = _seed(tmp_path, index=True)
        _set_recall_cfg(ws, index_path="custom-vec")
        _write_aged(ws, "custom-vec/index.json", b"{}")
        calls = self._recall_twice_around(ws, monkeypatch, lambda: _write_aged(ws, "custom-vec/index.json", b'{"dropped": 1}'))
        assert len(calls) == 2, "stale cached answer served after a configured vector-index write"
