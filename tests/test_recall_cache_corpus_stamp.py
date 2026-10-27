"""A recall answer is never served from cache after a write that changes it.

The governed-ledger head (``index_anchor``) in the cache key retires entries on
governed writes, but several writes that change what recall may serve never
move that head: an admin ``reindex`` after a status change on disk,
``delete_memory_item``, ``compact``, ``lint_autofix``, lineage / edge writes,
and any out-of-band edit. Those used to be served stale until the TTL.

The fix is a second corpus coordinate in the key: a stat-only stamp over the
index directory, the knowledge-graph database and every corpus file. Every
negative assertion below is paired with a positive control proving the cache
genuinely held the answer across calls when nothing changed.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from mind_mem.recall_cache import corpus_stamp, make_cache_key, reset_singleton


@pytest.fixture(autouse=True)
def _fresh_singletons(monkeypatch):
    from mind_mem import prefetch

    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    reset_singleton()
    prefetch.reset_cache()
    yield
    reset_singleton()
    prefetch.reset_cache()


def _blk(bid: str, stmt: str, date: str, status: str, extra: str = "") -> str:
    return f"[{bid}]\nType: Decision\nStatement: {stmt}\nDate: {date}\nStatus: {status}\n{extra}\n---\n\n"


def _backdate(ws: str, seconds: int = 3600) -> None:
    """Age every file past the racy-timestamp window, as a quiet store would be."""
    past = time.time() - seconds
    for root, _dirs, files in os.walk(ws):
        for name in files:
            os.utime(os.path.join(root, name), (past, past))


def _seed(tmp_path, *, index: bool) -> str:
    from mind_mem.init_workspace import init
    from mind_mem.sqlite_index import build_index

    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    init(ws)
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as fh:
        fh.write("\n" + _blk("D-20261006-001", "Ilona call is on Tuesday. Ilona call Tuesday at 10.", "2026-10-06", "active"))
    cfg_path = os.path.join(ws, "mind-mem.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    recall_cfg = cfg.get("recall") if isinstance(cfg.get("recall"), dict) else {}
    recall_cfg["vector_enabled"] = False
    cfg["recall"] = recall_cfg
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    if index:
        build_index(ws)
        # The first recall against a fresh index lazily creates the
        # calibration / outcome tables (a one-time recall.db write). A live
        # store is past that point; so is this fixture.
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.recall import _recall_impl

        with use_workspace(ws):
            _recall_impl("warm-up", limit=1, backend="bm25")
        reset_singleton()
    _backdate(ws)
    return ws


def _supersede_on_disk(ws: str) -> None:
    path = os.path.join(ws, "decisions", "DECISIONS.md")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    i = text.index("[D-20261006-001]")
    j = text.index("Status: active", i)
    text = text[:j] + "Status: superseded\nSupersededBy: D-20261008-001" + text[j + len("Status: active") :]
    text += _blk("D-20261008-001", "Ilona call moved to Thursday.", "2026-10-08", "active", "Supersedes: D-20261006-001")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _ids(envelope: str) -> list[tuple[str, str]]:
    return [(h.get("_id"), h.get("status")) for h in json.loads(envelope)["results"]]


def _counting_engine(monkeypatch) -> list[str]:
    from mind_mem.mcp.tools import recall as recall_mod

    calls: list[str] = []
    real = recall_mod._recall_impl_uncached

    def counted(query, *args, **kwargs):
        calls.append(query)
        return real(query, *args, **kwargs)

    monkeypatch.setattr(recall_mod, "_recall_impl_uncached", counted)
    return calls


Q = "Ilona call Tuesday"


# ---------------------------------------------------------------------------
# The key and the stamp
# ---------------------------------------------------------------------------


class TestTheStampIsInTheKey:
    def test_two_stamps_are_two_keys(self) -> None:
        assert make_cache_key("q", corpus_stamp="a") != make_cache_key("q", corpus_stamp="b")

    def test_no_stamp_keeps_the_legacy_key(self) -> None:
        assert make_cache_key("q") == make_cache_key("q", corpus_stamp="")


class TestTheStampSeesEveryRetrievalInput:
    def test_stable_when_nothing_changes(self, tmp_path) -> None:
        ws = _seed(tmp_path, index=True)
        first = corpus_stamp(ws)
        assert first is not None
        assert corpus_stamp(ws) == first

    def test_a_recall_does_not_move_the_stamp(self, tmp_path) -> None:
        """Otherwise the cache would never hit — a correctness-safe but silent regression."""
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.recall import _recall_impl

        ws = _seed(tmp_path, index=True)
        before = corpus_stamp(ws)
        with use_workspace(ws):
            _recall_impl(Q, limit=1, backend="bm25")
        assert corpus_stamp(ws) == before

    @pytest.mark.parametrize(
        "rel",
        [
            "decisions/DECISIONS.md",
            ".mind-mem-index/recall.db",
            "memory/knowledge_graph.db",
            "intelligence/NEW_NOTES.md",
        ],
    )
    def test_any_write_to_a_retrieval_input_moves_the_stamp(self, tmp_path, rel) -> None:
        ws = _seed(tmp_path, index=True)
        before = corpus_stamp(ws)
        path = os.path.join(ws, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "ab") as fh:
            fh.write(b"\n")
        past = time.time() - 600
        os.utime(path, (past, past))
        after = corpus_stamp(ws)
        assert after is not None
        assert after != before

    def test_a_fresh_write_makes_the_state_uncacheable(self, tmp_path) -> None:
        """Racy timestamps: a file modified inside the coarse mtime tick cannot
        vouch for its own content, so the stamp refuses to name it."""
        ws = _seed(tmp_path, index=True)
        with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as fh:
            fh.write("\n")
        assert corpus_stamp(ws) is None


# ---------------------------------------------------------------------------
# The wiring: real MCP recall over a real workspace
# ---------------------------------------------------------------------------


class TestTheCacheStillCaches:
    def test_identical_recalls_with_no_write_hit_the_cache(self, tmp_path, monkeypatch) -> None:
        """POSITIVE CONTROL for every negative test below."""
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.recall import _recall_impl

        ws = _seed(tmp_path, index=True)
        calls = _counting_engine(monkeypatch)
        with use_workspace(ws):
            first = _recall_impl(Q, limit=1, backend="bm25")
            second = _recall_impl(Q, limit=1, backend="bm25")
        assert _ids(first) == [("D-20261006-001", "active")]
        assert _ids(second) == _ids(first)
        assert len(calls) == 1, "the second recall did not come from the cache"


class TestNoStaleAnswerAfterAWrite:
    def test_admin_reindex_after_an_on_disk_status_change(self, tmp_path) -> None:
        """Reviewer finding F4, exactly: quarantine/supersede on disk, reindex, recall."""
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools import memory_ops
        from mind_mem.mcp.tools.recall import _recall_impl

        ws = _seed(tmp_path, index=True)
        with use_workspace(ws):
            before = _recall_impl(Q, limit=1, backend="bm25")
            assert _ids(before) == [("D-20261006-001", "active")], "vacuous: nothing cached"
            _supersede_on_disk(ws)
            json.loads(memory_ops.reindex())
            _backdate(ws)
            after = _recall_impl(Q, limit=1, backend="bm25")
        assert _ids(after) != [("D-20261006-001", "active")], "stale cached answer served after reindex"

    def test_out_of_band_index_rebuild(self, tmp_path) -> None:
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.recall import _recall_impl
        from mind_mem.sqlite_index import build_index

        ws = _seed(tmp_path, index=True)
        with use_workspace(ws):
            before = _recall_impl(Q, limit=1, backend="bm25")
            assert _ids(before) == [("D-20261006-001", "active")]
            _supersede_on_disk(ws)
            build_index(ws)
            _backdate(ws)
            after = _recall_impl(Q, limit=1, backend="bm25")
        assert _ids(after) != [("D-20261006-001", "active")]

    def test_hand_edit_with_the_scan_backend(self, tmp_path) -> None:
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.recall import _recall_impl

        ws = _seed(tmp_path, index=False)
        with use_workspace(ws):
            before = _recall_impl(Q, limit=1, backend="bm25")
            assert _ids(before) == [("D-20261006-001", "active")]
            _supersede_on_disk(ws)
            _backdate(ws)
            after = _recall_impl(Q, limit=1, backend="bm25")
        assert _ids(after) != [("D-20261006-001", "active")]

    def test_a_write_inside_the_racy_window_is_not_served_from_cache(self, tmp_path, monkeypatch) -> None:
        """No backdating: the write lands 'now', exactly as a live tool's would."""
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools import memory_ops
        from mind_mem.mcp.tools.recall import _recall_impl

        ws = _seed(tmp_path, index=True)
        with use_workspace(ws):
            before = _recall_impl(Q, limit=1, backend="bm25")
            assert _ids(before) == [("D-20261006-001", "active")]
            _supersede_on_disk(ws)
            json.loads(memory_ops.reindex())
            after = _recall_impl(Q, limit=1, backend="bm25")
        assert _ids(after) != [("D-20261006-001", "active")]

    def test_delete_memory_item(self, tmp_path) -> None:
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools import memory_ops
        from mind_mem.mcp.tools.recall import _recall_impl

        ws = _seed(tmp_path, index=True)
        with use_workspace(ws):
            before = _recall_impl(Q, limit=1, backend="bm25")
            assert _ids(before) == [("D-20261006-001", "active")]
            deleted = json.loads(memory_ops.delete_memory_item("D-20261006-001", rationale="test"))
            assert "error" not in deleted, deleted
            _backdate(ws)
            after = _recall_impl(Q, limit=1, backend="bm25")
        assert "D-20261006-001" not in [i for i, _ in _ids(after)]
