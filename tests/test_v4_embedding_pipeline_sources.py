# Copyright 2026 STARGA, Inc.
"""Where v4.embedding_pipeline actually finds block content.

The pipeline documented a ``blocks(id, content)`` table that no write
path in the package ever fills, so on a real workspace it returned an
empty mapping and said nothing about it.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from mind_mem.v4.embedding_pipeline import FLAG, derive_embeddings

_RECALL_REL = ".mind-mem-index/recall.db"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = tmp_path / "mind-mem.json"
    cfg.write_text(json.dumps({"v4": {FLAG: {"enabled": True}}}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))
    return tmp_path


def _write_recall_db(ws: Path, rows: list[tuple[str, dict]]) -> None:
    """Build a v3-shaped recall index (blocks carry json_blob, not content)."""
    db = ws / _RECALL_REL
    db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, type TEXT NOT NULL DEFAULT '', json_blob TEXT NOT NULL DEFAULT '{}')")
        conn.executemany(
            "INSERT INTO blocks (id, type, json_blob) VALUES (?, 'decision', ?)",
            [(bid, json.dumps(blob)) for bid, blob in rows],
        )
        conn.commit()


class TestRecallIndexFallback:
    def test_derives_from_the_v3_recall_index(self, workspace: Path) -> None:
        """Regression: only ``<ws>/index.db`` was consulted, and nothing
        writes that table, so a real workspace derived nothing."""
        _write_recall_db(
            workspace,
            [
                ("D-1", {"_id": "D-1", "Statement": "use JWT for auth", "Tags": ["auth", "api"]}),
                ("D-2", {"_id": "D-2", "Statement": "ship the compiler"}),
            ],
        )
        out = derive_embeddings(workspace, ["D-1", "D-2", "D-missing"], dim=32)
        assert set(out) == {"D-1", "D-2"}
        assert out["D-1"] != out["D-2"]
        assert any(v != 0.0 for v in out["D-1"])

    def test_index_db_content_wins_and_recall_fills_the_rest(self, workspace: Path) -> None:
        with closing(sqlite3.connect(workspace / "index.db")) as conn:
            conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, content TEXT)")
            conn.execute("INSERT INTO blocks (id, content) VALUES ('D-1', 'from index.db')")
            conn.commit()
        _write_recall_db(workspace, [("D-2", {"Statement": "from recall.db"})])
        out = derive_embeddings(workspace, ["D-1", "D-2"], dim=32)
        assert set(out) == {"D-1", "D-2"}

    def test_v3_schema_without_content_column_does_not_raise(self, workspace: Path) -> None:
        """A v3-shaped ``blocks`` table in index.db has no ``content``
        column; the SELECT must not blow up the caller."""
        with closing(sqlite3.connect(workspace / "index.db")) as conn:
            conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, json_blob TEXT)")
            conn.execute("INSERT INTO blocks (id, json_blob) VALUES ('D-1', '{}')")
            conn.commit()
        _write_recall_db(workspace, [("D-1", {"Statement": "real content"})])
        out = derive_embeddings(workspace, ["D-1"], dim=32)
        assert set(out) == {"D-1"}

    def test_unparseable_blob_yields_a_zero_vector(self, workspace: Path) -> None:
        db = workspace / _RECALL_REL
        db.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db)) as conn:
            conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, json_blob TEXT)")
            conn.execute("INSERT INTO blocks (id, json_blob) VALUES ('D-1', '{not json')")
            conn.commit()
        out = derive_embeddings(workspace, ["D-1"], dim=8)
        assert out["D-1"] == [0.0] * 8


class TestEmptyResultIsLogged:
    def test_no_content_anywhere_is_logged(self, workspace: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Regression: an empty mapping came back with no signal at all."""
        with caplog.at_level(logging.WARNING, logger="mind-mem.v4.embedding_pipeline"):
            logging.getLogger("mind-mem.v4.embedding_pipeline").propagate = True
            assert derive_embeddings(workspace, ["D-1"], dim=8) == {}
        assert any("embedding_pipeline_no_content" in r.getMessage() for r in caplog.records)

    def test_no_ids_requested_is_not_logged(self, workspace: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="mind-mem.v4.embedding_pipeline"):
            logging.getLogger("mind-mem.v4.embedding_pipeline").propagate = True
            assert derive_embeddings(workspace, [], dim=8) == {}
        assert not caplog.records
