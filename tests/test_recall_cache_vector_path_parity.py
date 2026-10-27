"""The corpus stamp resolves every vector input exactly as its reader does.

``corpus_stamp`` used to join each configured vector path onto the workspace
itself (``os.path.join(root, *rel.split("/"))``). For an absolute
``recall.index_path`` that names ``<ws>/tmp/...`` while ``VectorBackend`` reads
``/tmp/.../index.json`` — so the stamp stat'ed a directory that never exists,
recorded ``absent`` forever, and a vector-only rebuild was served from cache.

These tests pin the stamp to the READERS' own resolvers
(``recall_vector.local_index_file`` via ``VectorBackend._get_index_path`` and
``VectorBackend._sqlite_vec_db_path``) for relative, absolute, ``~`` and
symlinked configured paths, and prove the stamp moves when the file the
reader actually opens changes.
"""

from __future__ import annotations

import os
import time

import pytest

from mind_mem.recall_cache import corpus_stamp, reset_singleton
from mind_mem.recall_vector import VectorBackend
from tests.test_recall_cache_corpus_stamp import _seed
from tests.test_recall_cache_vector_stamp import _set_recall_cfg


@pytest.fixture(autouse=True)
def _fresh_singletons(monkeypatch):
    from mind_mem import prefetch

    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    reset_singleton()
    prefetch.reset_cache()
    yield
    reset_singleton()
    prefetch.reset_cache()


def _write_aged_abs(path: str, payload: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(payload)
    past = time.time() - 600
    os.utime(path, (past, past))


def _spy_stamp(monkeypatch, ws: str, cfg: dict) -> tuple[set[str], set[str]]:
    """Run the stamp and return the (listed dirs, stat'ed paths) it touched."""
    listed: set[str] = set()
    stated: set[str] = set()
    real_listdir, real_stat = os.listdir, os.stat

    def listdir(path="."):
        listed.add(os.path.normpath(os.fspath(path)))
        return real_listdir(path)

    def stat(path, *a, **k):
        stated.add(os.path.normpath(os.fspath(path)))
        return real_stat(path, *a, **k)

    monkeypatch.setattr(os, "listdir", listdir)
    monkeypatch.setattr(os, "stat", stat)
    try:
        assert corpus_stamp(ws, cfg) is not None
    finally:
        monkeypatch.setattr(os, "listdir", real_listdir)
        monkeypatch.setattr(os, "stat", real_stat)
    return listed, stated


def _configured_paths(tmp_path, ws: str) -> dict[str, tuple[str, str]]:
    """(index_path, sqlite_vec_db) per flavour of configured path."""
    outside = tmp_path / "outside"
    outside.mkdir()
    link = os.path.join(ws, "linked-vec")
    os.symlink(str(outside), link)
    return {
        "relative": ("rel-vec", "rel-db/vectors.db"),
        "absolute": (str(tmp_path / "abs-vec"), str(tmp_path / "abs-db" / "vectors.db")),
        "tilde": ("~/tilde-vec", "~/tilde-db/vectors.db"),
        "symlink": ("linked-vec", "linked-vec/vectors.db"),
    }


FLAVOURS = ("relative", "absolute", "tilde", "symlink")


class TestStampAndReaderAgreeOnEveryConfiguredPath:
    @pytest.mark.parametrize("flavour", FLAVOURS)
    def test_stamp_lists_the_dir_the_reader_opens(self, tmp_path, monkeypatch, flavour) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        ws = _seed(tmp_path, index=True)
        index_path, vec_db = _configured_paths(tmp_path, ws)[flavour]
        cfg = _set_recall_cfg(ws, index_path=index_path, sqlite_vec_db=vec_db)
        backend = VectorBackend(cfg["recall"])
        reader_index_dir = os.path.normpath(os.path.dirname(backend._get_index_path(ws)))
        reader_vec_db = os.path.normpath(backend._sqlite_vec_db_path(ws))

        listed, stated = _spy_stamp(monkeypatch, ws, cfg)

        assert reader_index_dir in listed, (flavour, reader_index_dir, sorted(listed))
        assert reader_vec_db in stated, (flavour, reader_vec_db)

    @pytest.mark.parametrize("flavour", FLAVOURS)
    def test_rewriting_the_readers_index_moves_the_stamp(self, tmp_path, monkeypatch, flavour) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        ws = _seed(tmp_path, index=True)
        index_path, vec_db = _configured_paths(tmp_path, ws)[flavour]
        cfg = _set_recall_cfg(ws, index_path=index_path, sqlite_vec_db=vec_db)
        backend = VectorBackend(cfg["recall"])
        index_file = backend._get_index_path(ws)

        _write_aged_abs(index_file, b'{"v": 1}')
        before = corpus_stamp(ws, cfg)
        assert before is not None
        assert corpus_stamp(ws, cfg) == before  # positive control: stable when untouched
        _write_aged_abs(index_file, b'{"v": 22}')
        after = corpus_stamp(ws, cfg)
        assert after is not None
        assert after != before

    @pytest.mark.parametrize("flavour", FLAVOURS)
    def test_rewriting_the_readers_vec_db_moves_the_stamp(self, tmp_path, monkeypatch, flavour) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        ws = _seed(tmp_path, index=True)
        index_path, vec_db = _configured_paths(tmp_path, ws)[flavour]
        cfg = _set_recall_cfg(ws, index_path=index_path, sqlite_vec_db=vec_db)
        db_file = VectorBackend(cfg["recall"])._sqlite_vec_db_path(ws)

        _write_aged_abs(db_file, b"a")
        before = corpus_stamp(ws, cfg)
        assert before is not None
        _write_aged_abs(db_file, b"bb")
        assert corpus_stamp(ws, cfg) not in (None, before)

    def test_coerced_index_path_matches_the_reader(self, tmp_path, monkeypatch) -> None:
        """VectorBackend reads ``str(index_path)``; the stamp must follow it."""
        ws = _seed(tmp_path, index=True)
        cfg = _set_recall_cfg(ws, index_path=7)
        reader_dir = os.path.normpath(os.path.dirname(VectorBackend(cfg["recall"])._get_index_path(ws)))
        listed, _ = _spy_stamp(monkeypatch, ws, cfg)
        assert reader_dir in listed
