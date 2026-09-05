# Copyright 2026 STARGA, Inc.
"""One file, one shape, one writer — and a shape mismatch that is never silent.

``<workspace>/<index_path>/index.json`` had **two writers that disagreed**.
``VectorBackend._index_local`` wrote the dict ``{model, dimension, blocks,
embeddings}``; the module-level ``rebuild_index`` — the vector rebuild the MCP
``reindex(include_vectors=True)`` tool actually calls — wrote a bare LIST of
``{_id, embedding, text}`` records to the same path. ``_load_local_index``
reads the dict, so a workspace reindexed through MCP raised
``AttributeError: 'list' object has no attribute 'get'`` on the next dense
query, and the serving path swallowed it to ``[]``: recall silently degraded
to lexical-only with no error anywhere a user would look.

Every assertion here is paired with proof it could have failed:

* a "the dense leg returns results" test passes trivially if no index was ever
  loaded, so each one first proves the file EXISTS on disk **in the shape under
  test**, then proves the reader actually consumed it
  (``last_index_diagnostic``), and only then asserts on candidates;
* the negative case — a list-shaped file — asserts a NAMED diagnostic and a
  real log record, not merely "no crash".
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.usefixtures("captured_vector_logs")

QUERY = "authentication rotation"


# ── log capture ──────────────────────────────────────────────────────────
#
# ``StructuredLogger`` sets ``propagate = False`` on ``mind-mem.<component>``,
# so ``caplog`` sees nothing. Attach to the real logger the module writes to;
# anything less would be asserting against a logger the product does not use.


class _Recorder(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[tuple[str, str, dict[str, Any]]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.levelname, record.getMessage(), getattr(record, "data", None) or {}))

    def events(self, name: str) -> list[dict[str, Any]]:
        return [data for _lvl, msg, data in self.records if msg == name]

    def level_of(self, name: str) -> str | None:
        for lvl, msg, _data in self.records:
            if msg == name:
                return lvl
        return None


@pytest.fixture()
def captured_vector_logs():
    logger = logging.getLogger("mind-mem.recall_vector")
    handler = _Recorder()
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


# ── fixtures ─────────────────────────────────────────────────────────────


def _backend(config: dict | None = None):
    from mind_mem.recall_vector import VectorBackend

    return VectorBackend(config or {})


def _stub_embedder(backend, monkeypatch: pytest.MonkeyPatch) -> None:
    """A deterministic 2-d embedder — no model download, no GPU, no clock.

    The vector under test is the SAME for every text, so similarity ordering is
    not what any assertion below rests on; what rests on it is whether the leg
    produced candidates at all.
    """
    monkeypatch.setattr(backend, "embed", lambda texts: [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(backend, "_embed_for_provider", lambda texts: [[1.0, 0.0] for _ in texts])


def _write_canonical(backend, workspace: str) -> None:
    from mind_mem.recall_vector import canonical_local_index

    backend._save_local_index(
        workspace,
        canonical_local_index(
            "test-model",
            2,
            [
                {
                    "_id": "DEC-20260101-001",
                    "type": "decision",
                    "excerpt": "rotate the signing key",
                    "file": "decisions/DECISIONS.md",
                    "line": 1,
                    "status": "active",
                    "date": "2026-01-01",
                }
            ],
            [[1.0, 0.0]],
        ),
    )


def _write_legacy_list(backend, workspace: str) -> None:
    """Exactly what the pre-fix ``rebuild_index`` left on disk."""
    path = backend._get_index_path(workspace)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            [{"_id": "DEC-20260101-001", "embedding": [1.0, 0.0], "text": "rotate the signing key"}],
            fh,
        )


@pytest.fixture()
def corpus_ws(tmp_path) -> str:
    """A real workspace with two admitted blocks, rendered by the store."""
    from mind_mem.block_store import _render_block
    from mind_mem.init_workspace import init

    root = str(tmp_path / "ws")
    os.makedirs(root)
    init(root)
    decisions = Path(root) / "decisions" / "DECISIONS.md"
    decisions.write_text(
        _render_block(
            {
                "_id": "DEC-20260101-001",
                "Statement": "rotate the signing key every quarter",
                "Date": "2026-01-01",
                "Status": "active",
                "Type": "decision",
            }
        )
        + "\n"
        + _render_block(
            {
                "_id": "DEC-20260101-002",
                "Statement": "authentication rotation is owned by the platform team",
                "Date": "2026-01-02",
                "Status": "active",
                "Type": "decision",
            }
        ),
        encoding="utf-8",
    )
    return root


# ── the canonical shape actually serves ──────────────────────────────────


class TestCanonicalShapeServes:
    def test_index_local_writes_a_dict_the_reader_consumes_and_serves(self, tmp_path, monkeypatch, captured_vector_logs):
        ws = str(tmp_path)
        backend = _backend()
        _stub_embedder(backend, monkeypatch)
        backend._index_local(
            ws,
            [
                {
                    "_id": "DEC-20260101-001",
                    "type": "decision",
                    "excerpt": "rotate the signing key",
                    "file": "decisions/DECISIONS.md",
                    "line": 1,
                    "status": "active",
                    "date": "2026-01-01",
                }
            ],
            [[1.0, 0.0]],
        )

        # 1. The file exists ON DISK in the shape under test.
        on_disk = json.loads(Path(backend._get_index_path(ws)).read_text(encoding="utf-8"))
        assert isinstance(on_disk, dict), "the writer under test did not produce the canonical dict"
        assert isinstance(on_disk["blocks"], list) and isinstance(on_disk["embeddings"], list)

        # 2. The READER consumed it — not "an index was found somewhere".
        loaded = backend._load_local_index(ws)
        assert loaded is not None
        assert backend.last_index_diagnostic == "ok"
        assert captured_vector_logs.events("index_loaded"), "the reader never reported a load"

        # 3. Only now is a non-empty result meaningful.
        hits = backend._search_local(ws, QUERY, 10, False)
        assert hits, "the dense leg served nothing from a canonical index — every assertion above would be vacuous"
        assert hits[0]["_id"] == "DEC-20260101-001"

    def test_the_shape_carries_model_and_dimension(self, tmp_path):
        """The reader needs both to tell a stale index from a current one."""
        ws = str(tmp_path)
        backend = _backend()
        _write_canonical(backend, ws)
        on_disk = json.loads(Path(backend._get_index_path(ws)).read_text(encoding="utf-8"))
        assert on_disk["model"] == "test-model"
        assert on_disk["dimension"] == 2
        assert on_disk["schema"] == "mind-mem/local-vector-index@1"


# ── rebuild_index (the MCP reindex path) writes the same shape ───────────


class TestRebuildIndexWritesTheCanonicalShape:
    """The defect, at its source: this is the writer MCP reindex calls."""

    def _rebuild(self, ws: str, monkeypatch) -> int:
        from mind_mem import recall_vector

        real_init = recall_vector.VectorBackend.__init__

        def _patched(self, config):  # noqa: ANN001
            real_init(self, config)
            _stub_embedder(self, monkeypatch)

        monkeypatch.setattr(recall_vector.VectorBackend, "__init__", _patched)
        return recall_vector.rebuild_index(ws)

    def test_the_file_it_writes_is_the_file_the_reader_reads(self, corpus_ws, monkeypatch, captured_vector_logs):
        n = self._rebuild(corpus_ws, monkeypatch)
        assert n == 2, "positive control: the rebuild indexed nothing, so the shape assertion below is vacuous"

        backend = _backend()
        path = Path(backend._get_index_path(corpus_ws))
        assert path.is_file(), "rebuild_index wrote no index at all"
        on_disk = json.loads(path.read_text(encoding="utf-8"))

        # THE REGRESSION. A bare list here is the shipped defect.
        assert isinstance(on_disk, dict), "rebuild_index wrote a bare list again — the two-writer defect is back"
        assert len(on_disk["blocks"]) == len(on_disk["embeddings"]) == 2

        loaded = backend._load_local_index(corpus_ws)
        assert loaded is not None
        assert backend.last_index_diagnostic == "ok", "the reader could not consume what the MCP path wrote"
        assert not captured_vector_logs.events("index_shape_invalid")
        assert not captured_vector_logs.events("index_legacy_list_shape")

    def test_a_dense_query_after_an_mcp_style_reindex_returns_candidates(self, corpus_ws, monkeypatch):
        """End to end: reindex, then query — the path that returned [] before."""
        assert self._rebuild(corpus_ws, monkeypatch) == 2

        backend = _backend()
        _stub_embedder(backend, monkeypatch)
        hits = backend._search_local(corpus_ws, QUERY, 10, False)
        assert backend.last_index_diagnostic == "ok"
        assert hits, "a dense query after an MCP-style reindex still returns nothing"
        assert {h["_id"] for h in hits} == {"DEC-20260101-001", "DEC-20260101-002"}

    def test_it_populates_the_metadata_the_scoring_path_reads(self, corpus_ws, monkeypatch):
        """A legacy record carried ``_id`` alone — status/date/file were absent,
        so ``active_only`` matched nothing and every hit scored without a
        recency or status boost."""
        assert self._rebuild(corpus_ws, monkeypatch) == 2

        backend = _backend()
        blocks = (backend._load_local_index(corpus_ws) or {})["blocks"]
        by_id = {b["_id"]: b for b in blocks}
        first = by_id["DEC-20260101-001"]
        assert first["status"] == "active"
        assert first["date"] == "2026-01-01"
        assert first["file"].endswith("DECISIONS.md")
        assert first["type"]
        assert first["excerpt"]
        # The field the legacy shape carried is still carried.
        assert "rotate the signing key" in first["text"]

    def test_active_only_now_matches(self, corpus_ws, monkeypatch):
        assert self._rebuild(corpus_ws, monkeypatch) == 2
        backend = _backend()
        _stub_embedder(backend, monkeypatch)
        assert backend._search_local(corpus_ws, QUERY, 10, True), "no block could prove it was active"


# ── the negative case: a list-shaped file is VISIBLE, never silent ───────


class TestLegacyListShapeIsVisible:
    def test_the_reader_no_longer_raises_attributeerror(self, tmp_path, captured_vector_logs):
        """The exact reported reproduction."""
        ws = str(tmp_path)
        backend = _backend()
        _write_legacy_list(backend, ws)

        # Prove the file on disk really is the list shape under test.
        assert isinstance(json.loads(Path(backend._get_index_path(ws)).read_text(encoding="utf-8")), list)

        index = backend._load_local_index(ws)  # used to raise AttributeError
        assert index is not None

    def test_it_is_reported_with_a_named_diagnostic_and_a_log_record(self, tmp_path, captured_vector_logs):
        ws = str(tmp_path)
        backend = _backend()
        _write_legacy_list(backend, ws)
        backend._load_local_index(ws)

        assert backend.last_index_diagnostic == "legacy_list_shape"
        events = captured_vector_logs.events("index_legacy_list_shape")
        assert events, "a stale-shaped index loaded without saying so — that silence is the defect"
        assert events[0]["usable"] == 1
        assert "reindex" in events[0]["remediation"]
        assert captured_vector_logs.level_of("index_legacy_list_shape") == "WARNING"

    def test_the_existing_on_disk_index_keeps_serving(self, tmp_path, monkeypatch):
        """Migrated on read: an already-reindexed workspace is not left dead."""
        ws = str(tmp_path)
        backend = _backend()
        _stub_embedder(backend, monkeypatch)
        _write_legacy_list(backend, ws)

        hits = backend._search_local(ws, QUERY, 10, False)
        assert backend.last_index_diagnostic == "legacy_list_shape"
        assert hits, "a legacy index was adapted to nothing — that is the silent zero-result defect again"
        assert hits[0]["_id"] == "DEC-20260101-001"
        assert hits[0]["excerpt"] == "rotate the signing key"

    def test_what_a_legacy_file_cannot_prove_is_not_invented(self, tmp_path, monkeypatch):
        """Legacy records carry no status, so ``active_only`` stays empty — the
        adapter reports that rather than fabricating ``status: active``."""
        ws = str(tmp_path)
        backend = _backend()
        _stub_embedder(backend, monkeypatch)
        _write_legacy_list(backend, ws)
        assert backend._search_local(ws, QUERY, 10, True) == []

    def test_junk_records_inside_a_legacy_file_are_dropped_not_mispaired(self):
        """Blocks and embeddings are positionally paired; dropping a record on
        one side alone is how a block wears its neighbour's vector."""
        from mind_mem.recall_vector import migrate_legacy_list_index

        index = migrate_legacy_list_index(
            [
                {"_id": "A", "embedding": [1.0, 0.0], "text": "a"},
                "not a record",
                {"_id": "B", "embedding": "not a vector", "text": "b"},
                {"_id": "C", "embedding": [], "text": "c"},
                {"_id": "D", "embedding": [0.0, 1.0], "text": "d"},
            ]
        )
        assert [b["_id"] for b in index["blocks"]] == ["A", "D"]
        assert index["embeddings"] == [[1.0, 0.0], [0.0, 1.0]]
        assert index["dimension"] == 2


# ── the negative case: an unreadable shape is an ERROR, not an empty leg ─


class TestInvalidShapeIsReported:
    @pytest.mark.parametrize("payload", [42, "a string", {"not": "an index"}])
    def test_it_returns_none_with_a_named_error(self, tmp_path, captured_vector_logs, payload):
        ws = str(tmp_path)
        backend = _backend()
        path = backend._get_index_path(ws)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

        assert backend._load_local_index(ws) is None
        assert backend.last_index_diagnostic == "invalid_shape"
        events = captured_vector_logs.events("index_shape_invalid")
        assert events, "an unreadable index produced no diagnostic"
        assert "reindex" in events[0]["remediation"]
        assert captured_vector_logs.level_of("index_shape_invalid") == "ERROR"

    def test_a_missing_index_is_a_different_diagnostic_than_a_broken_one(self, tmp_path):
        backend = _backend()
        assert backend._load_local_index(str(tmp_path)) is None
        assert backend.last_index_diagnostic == "missing"

    def test_corrupt_json_is_a_different_diagnostic_than_a_wrong_shape(self, tmp_path):
        ws = str(tmp_path)
        backend = _backend()
        path = backend._get_index_path(ws)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Path(path).write_text("{corrupt json!!", encoding="utf-8")
        assert backend._load_local_index(ws) is None
        assert backend.last_index_diagnostic == "unreadable"


# ── the swallow: empty-because-broken vs empty-because-no-match ──────────


class TestSearchBatchDistinguishesItsEmpties:
    def test_a_broken_index_names_the_reason(self, tmp_path, monkeypatch, captured_vector_logs):
        from mind_mem import recall_vector

        ws = str(tmp_path)
        path = os.path.join(ws, ".mind-mem-vectors", "index.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Path(path).write_text(json.dumps({"not": "an index"}), encoding="utf-8")

        assert recall_vector.search_batch(ws, QUERY, config={}) == []
        empties = captured_vector_logs.events("search_batch_empty")
        assert empties, "search_batch returned [] without saying why"
        assert empties[0]["reason"] == "invalid_shape"

    def test_an_honest_miss_is_not_labelled_a_failure(self, tmp_path, monkeypatch, captured_vector_logs):
        from mind_mem import recall_vector

        ws = str(tmp_path)
        backend = _backend()
        _write_canonical(backend, ws)
        monkeypatch.setattr(
            recall_vector.VectorBackend,
            "embed",
            lambda self, texts: [[1.0, 0.0] for _ in texts],
        )
        # active_only with a status the index does not carry → a real miss.
        assert recall_vector.search_batch(ws, QUERY, config={}, active_only=False)
        empties = captured_vector_logs.events("search_batch_empty")
        assert not empties, "a served result was reported as an empty leg"
        assert not captured_vector_logs.events("search_batch_failed")

    def test_a_swallowed_exception_names_its_type(self, tmp_path, monkeypatch, captured_vector_logs):
        from mind_mem import recall_vector

        def _boom(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
            raise RuntimeError("index exploded")

        monkeypatch.setattr(recall_vector.VectorBackend, "search", _boom)
        assert recall_vector.search_batch(str(tmp_path), QUERY, config={}) == []
        failures = captured_vector_logs.events("search_batch_failed")
        assert failures, "the swallow left no trace at all"
        assert failures[0]["error_type"] == "RuntimeError"
