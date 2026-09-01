"""The external-anchor tools wired onto the audit family in 5.0.0.

``ledger_anchor`` shipped in v2.0.0rc1 with a module docstring claiming an
``anchor_history`` MCP tool was "registered in mcp_server". It was not: nothing
in ``src/`` imported the module, so the whole external-anchoring capability was
unreachable from the product for three minor versions while its own tests
passed. That is the exact defect the reachability gate now blocks, and these
tests pin the surface that fixes it.

The point of an anchor is adversarial: ``verify_merkle`` and ``verify_chain``
prove the store is *internally* consistent, which a holder of the store could
fake by rebuilding a consistent history. An anchor pins a root to a moment
outside the store. So the tests below check the two properties that make it
worth anything -- the root is computed from the index rather than accepted from
the caller, and a damaged trail is reported rather than silently skipped.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.audit import anchor_history, anchor_root


@pytest.fixture(autouse=True)
def _admin_scope(monkeypatch):
    """anchor_root is admin-scoped by design; these tests exercise it as admin.

    Deliberately autouse rather than folded into the workspace fixture: the
    scope requirement is a property of the tool, not of the workspace, and
    TestReachability asserts that classification independently.
    """
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")


@pytest.fixture
def ws(tmp_path):
    """A workspace whose index carries two hashable blocks."""
    w = tmp_path / "ws"
    (w / ".mind-mem-index").mkdir(parents=True)
    # _check_workspace requires the Markdown corpus layout on the default
    # SQLite backend; without decisions/ every ws-gated tool fails closed.
    (w / "decisions").mkdir(parents=True)
    (w / "maintenance").mkdir(parents=True)
    conn = sqlite3.connect(w / ".mind-mem-index" / "recall.db")
    conn.executescript(
        "CREATE TABLE blocks (id TEXT PRIMARY KEY, type TEXT, file TEXT, line INTEGER, "
        "status TEXT, date TEXT, speaker TEXT, tags TEXT, dia_id TEXT, parent_id TEXT, json_blob TEXT);"
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"
        "CREATE TABLE index_meta (block_id TEXT PRIMARY KEY, content_hash TEXT);"
    )
    for bid, h in (("DEC-20200101-001", "a" * 64), ("DEC-20200101-002", "b" * 64)):
        conn.execute(
            "INSERT INTO blocks VALUES (?, 'decision', 'decisions/DECISIONS.md', 1, 'active', "
            "'2020-01-01', '', '', '', '', '{}')",
            (bid,),
        )
        conn.execute("INSERT INTO index_meta VALUES (?, ?)", (bid, h))
    conn.commit()
    conn.close()
    return w


def _anchor(w, **kw):
    with use_workspace(str(w)):
        return json.loads(anchor_root(**kw))


def _history(w, **kw):
    with use_workspace(str(w)):
        return json.loads(anchor_history(**kw))


class TestAnchorRoot:
    def test_records_the_live_root_and_returns_the_entry(self, ws) -> None:
        out = _anchor(ws)
        assert out["ok"] is True
        assert out["leaves"] == 2
        assert isinstance(out["root"], str) and out["root"]
        assert out["entry"]["merkle_root"] == out["root"]

    def test_root_comes_from_the_index_not_the_caller(self, ws) -> None:
        """The tool takes no root argument at all.

        Anchoring a caller-supplied root would anchor the caller's claim about
        the store rather than the store's actual state, which is worth nothing
        against the adversary an anchor exists to catch.
        """
        import inspect

        params = set(inspect.signature(anchor_root).parameters)
        assert "merkle_root" not in params and "root" not in params

    def test_without_a_poster_the_anchor_is_pending_not_confirmed(self, ws) -> None:
        entry = _anchor(ws)["entry"]
        assert entry["status"] == "pending"
        assert entry["tx_hash"] in (None, "")

    def test_a_cleared_transaction_is_recorded_as_confirmed(self, ws) -> None:
        entry = _anchor(ws, chain="eth-l2", tx_hash="0xfeed", block_height=42)["entry"]
        assert entry["status"] == "confirmed"
        assert entry["tx_hash"] == "0xfeed"
        assert entry["chain"] == "eth-l2"
        assert entry["block_height"] == 42

    def test_two_anchors_over_an_unchanged_index_agree_on_the_root(self, ws) -> None:
        assert _anchor(ws)["root"] == _anchor(ws)["root"]

    def test_a_changed_index_produces_a_different_root(self, ws) -> None:
        first = _anchor(ws)["root"]
        conn = sqlite3.connect(ws / ".mind-mem-index" / "recall.db")
        conn.execute(
            "INSERT INTO blocks VALUES ('DEC-20200101-003', 'decision', 'decisions/DECISIONS.md', "
            "1, 'active', '2020-01-01', '', '', '', '', '{}')"
        )
        conn.execute("INSERT INTO index_meta VALUES ('DEC-20200101-003', ?)", ("c" * 64,))
        conn.commit()
        conn.close()
        assert _anchor(ws)["root"] != first

    def test_an_empty_index_refuses_rather_than_anchoring_nothing(self, tmp_path) -> None:
        bare = tmp_path / "bare"
        (bare / ".mind-mem-index").mkdir(parents=True)
        (bare / "decisions").mkdir(parents=True)
        out = _anchor(bare)
        assert out["ok"] is False
        assert "mind-mem-scan" in out["error"]

    @pytest.mark.parametrize(
        "kwargs",
        [{"chain": ""}, {"chain": "   "}, {"block_height": -1}, {"block_height": True}],
    )
    def test_bad_input_is_refused_at_the_boundary(self, ws, kwargs) -> None:
        out = _anchor(ws, **kwargs)
        assert out["ok"] is False
        assert "error" in out


class TestAnchorHistory:
    def test_empty_trail_reports_zero_not_an_error(self, ws) -> None:
        out = _history(ws)
        assert out["ok"] is True
        assert out["count"] == 0
        assert out["entries"] == []
        assert out["latest"] is None

    def test_entries_accumulate_and_latest_tracks_the_newest(self, ws) -> None:
        _anchor(ws, chain="local")
        _anchor(ws, chain="eth-l2", tx_hash="0xabc")
        out = _history(ws)
        assert out["count"] == 2
        assert out["latest"]["chain"] == "eth-l2"

    def test_limit_returns_the_most_recent(self, ws) -> None:
        for i in range(4):
            _anchor(ws, chain=f"c{i}")
        out = _history(ws, limit=2)
        assert len(out["entries"]) == 2
        assert out["entries"][-1]["chain"] == "c3"
        assert out["count"] == 4  # count is the full trail, not the page

    def test_a_damaged_line_is_REPORTED_not_silently_dropped(self, ws) -> None:
        """A trail that quietly discards what it cannot parse is worse than none.

        Truncating a line is exactly the tampering the anchor trail exists to
        reveal, so it has to surface in ``problems``.
        """
        _anchor(ws)
        trail = ws / "maintenance" / "ledger_anchors.jsonl"
        trail.write_text(trail.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
        out = _history(ws)
        assert out["ok"] is True
        assert out["problems"], "a corrupt trail line must be reported"

    @pytest.mark.parametrize("bad", [0, -1, True])
    def test_bad_limit_is_refused(self, ws, bad) -> None:
        out = _history(ws, limit=bad)
        assert out["ok"] is False


class TestReachability:
    def test_both_tools_are_registered_on_the_audit_family(self) -> None:
        """The defect this whole surface fixes was being unregistered."""
        registered = []

        class _Mcp:
            def tool(self, fn):
                registered.append(fn.__name__)
                return fn

        from mind_mem.mcp.tools.audit import register

        register(_Mcp())
        assert "anchor_root" in registered
        assert "anchor_history" in registered

    def test_both_tools_are_classified_in_the_acl(self) -> None:
        """Registered but unclassified is unreachable, not unprivileged."""
        from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS

        assert "anchor_root" in ADMIN_TOOLS, "anchor_root writes the trail"
        assert "anchor_history" in ADMIN_TOOLS | USER_TOOLS
