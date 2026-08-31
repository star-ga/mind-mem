"""Tests for provenance-rich blocks (roadmap Group E).

Covers:
  - block_provenance: sanitize / attach / extract helpers
  - block_store: canonical rendering + write/read round-trip of the
    optional provenance fields, backward compatibility for blocks
    without them
  - block_metadata: set_provenance / get_provenance + additive schema
    migration of pre-Group-E databases
  - propose_update MCP tool: optional provenance params land in
    SIGNALS.md; omitted params keep the legacy output byte-shape
  - recall: provenance fields surfaced in results when present
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from mind_mem.block_metadata import BlockMetadataManager
from mind_mem.block_provenance import (
    MAX_PROVENANCE_VALUE_LEN,
    PROVENANCE_FIELD_NAMES,
    PROVENANCE_FIELDS,
    attach_provenance,
    extract_provenance,
    sanitize_provenance_value,
)
from mind_mem.block_store import MarkdownBlockStore

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_PROV_KWARGS = {
    "actor_id": "agent-7",
    "actor_role": "planner",
    "session_id": "sess-42",
    "tool_id": "mcp.propose_update",
    "purpose": "record architectural decision",
    # T-001 content axis — see tests/test_content_source_provenance.py for
    # its own contracts (vocabulary, no default, fail-closed read).
    "content_source": "agent",
}


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    """Minimal corpus layout for MarkdownBlockStore writes."""
    for d in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


# ---------------------------------------------------------------------------
# sanitize_provenance_value
# ---------------------------------------------------------------------------


class TestSanitize:
    def test_flattens_newlines(self):
        assert sanitize_provenance_value("a\nb\r\nc") == "a b  c"

    def test_strips_and_caps(self):
        long = "x" * (MAX_PROVENANCE_VALUE_LEN + 50)
        out = sanitize_provenance_value("  " + long + "  ")
        assert out == "x" * MAX_PROVENANCE_VALUE_LEN

    def test_header_injection_neutralised(self):
        # A crafted value cannot start a new [ID] header line once flattened.
        out = sanitize_provenance_value("legit\n[D-20260101-001]\nStatus: active")
        assert "\n" not in out and "\r" not in out


# ---------------------------------------------------------------------------
# attach_provenance / extract_provenance
# ---------------------------------------------------------------------------


class TestAttachExtract:
    def test_attach_sets_canonical_fields(self):
        block = {"_id": "D-20260724-001", "Statement": "s"}
        out = attach_provenance(block, **_PROV_KWARGS)
        assert out["ActorId"] == "agent-7"
        assert out["ActorRole"] == "planner"
        assert out["SessionId"] == "sess-42"
        assert out["ToolId"] == "mcp.propose_update"
        assert out["Purpose"] == "record architectural decision"

    def test_attach_is_immutable(self):
        block = {"_id": "D-20260724-001"}
        out = attach_provenance(block, actor_id="a")
        assert "ActorId" not in block
        assert out is not block

    def test_attach_omits_none_and_blank(self):
        out = attach_provenance({}, actor_id=None, actor_role="", purpose="   ")
        assert out == {}

    def test_attach_rejects_non_str(self):
        with pytest.raises(TypeError):
            attach_provenance({}, actor_id=42)  # type: ignore[arg-type]

    def test_round_trip(self):
        out = attach_provenance({}, **_PROV_KWARGS)
        assert extract_provenance(out) == _PROV_KWARGS

    def test_extract_empty_when_absent(self):
        assert extract_provenance({"_id": "D-1", "Statement": "s"}) == {}

    def test_extract_coerces_non_str_values(self):
        # Hand-edited corpus values must not crash recall surfacing.
        assert extract_provenance({"ActorId": 123}) == {"actor_id": "123"}

    def test_field_maps_are_consistent(self):
        assert tuple(PROVENANCE_FIELDS.values()) == PROVENANCE_FIELD_NAMES
        assert set(PROVENANCE_FIELDS) == set(_PROV_KWARGS)


# ---------------------------------------------------------------------------
# BlockStore round-trip
# ---------------------------------------------------------------------------


class TestBlockStoreRoundTrip:
    def test_write_and_read_back_provenance(self, ws: Path, admitted):
        store = MarkdownBlockStore(str(ws))
        block = attach_provenance(
            {"_id": "D-20260724-001", "Statement": "Adopt provenance fields", "Status": "active"},
            **_PROV_KWARGS,
        )
        store.write_block(block)
        got = store.get_by_id("D-20260724-001")
        assert got is not None
        for field, param in zip(PROVENANCE_FIELD_NAMES, _PROV_KWARGS):
            assert got[field] == _PROV_KWARGS[param]

    def test_rendered_file_contains_canonical_lines(self, ws: Path, admitted):
        store = MarkdownBlockStore(str(ws))
        block = attach_provenance({"_id": "D-20260724-002", "Statement": "s", "Status": "active"}, actor_id="agent-7")
        store.write_block(block)
        text = (ws / "decisions" / "DECISIONS.md").read_text(encoding="utf-8")
        assert "ActorId: agent-7" in text

    def test_block_without_provenance_unaffected(self, ws: Path, admitted):
        store = MarkdownBlockStore(str(ws))
        store.write_block({"_id": "D-20260724-003", "Statement": "plain block", "Status": "active"})
        text = (ws / "decisions" / "DECISIONS.md").read_text(encoding="utf-8")
        for field in PROVENANCE_FIELD_NAMES:
            assert f"{field}:" not in text
        got = store.get_by_id("D-20260724-003")
        assert got is not None
        assert extract_provenance(got) == {}


# ---------------------------------------------------------------------------
# BlockMetadataManager provenance sidecar
# ---------------------------------------------------------------------------


class TestBlockMetadataProvenance:
    def test_set_and_get(self, tmp_path: Path):
        mgr = BlockMetadataManager(str(tmp_path / "meta.db"))
        try:
            assert mgr.set_provenance("D-001", **_PROV_KWARGS) is True
            assert mgr.get_provenance("D-001") == _PROV_KWARGS
        finally:
            mgr.close()

    def test_partial_update_preserves_existing(self, tmp_path: Path):
        mgr = BlockMetadataManager(str(tmp_path / "meta.db"))
        try:
            mgr.set_provenance("D-001", actor_id="agent-7", purpose="initial write")
            mgr.set_provenance("D-001", session_id="sess-9")
            assert mgr.get_provenance("D-001") == {
                "actor_id": "agent-7",
                "purpose": "initial write",
                "session_id": "sess-9",
            }
        finally:
            mgr.close()

    def test_get_unknown_block_is_empty(self, tmp_path: Path):
        mgr = BlockMetadataManager(str(tmp_path / "meta.db"))
        try:
            assert mgr.get_provenance("D-does-not-exist") == {}
        finally:
            mgr.close()

    def test_set_nothing_returns_false(self, tmp_path: Path):
        mgr = BlockMetadataManager(str(tmp_path / "meta.db"))
        try:
            assert mgr.set_provenance("D-001") is False
            assert mgr.set_provenance("D-001", actor_id="   ") is False
        finally:
            mgr.close()

    def test_values_sanitized(self, tmp_path: Path):
        mgr = BlockMetadataManager(str(tmp_path / "meta.db"))
        try:
            mgr.set_provenance("D-001", purpose="line1\nline2", actor_id="x" * 500)
            prov = mgr.get_provenance("D-001")
            assert prov["purpose"] == "line1 line2"
            assert len(prov["actor_id"]) == MAX_PROVENANCE_VALUE_LEN
        finally:
            mgr.close()

    def test_migration_of_pre_group_e_database(self, tmp_path: Path):
        """A legacy block_meta DB (no provenance columns) is upgraded in place."""
        db_path = str(tmp_path / "legacy.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE block_meta (
                id TEXT PRIMARY KEY,
                importance REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                keywords TEXT DEFAULT '',
                connections TEXT DEFAULT ''
            );
            """
        )
        conn.execute(
            "INSERT INTO block_meta (id, importance, access_count, keywords) VALUES (?, ?, ?, ?)",
            ("D-legacy", 1.2, 7, "old,keywords"),
        )
        conn.commit()
        conn.close()

        mgr = BlockMetadataManager(db_path)
        try:
            # Legacy row intact, provenance readable (empty) and writable.
            assert mgr.get_provenance("D-legacy") == {}
            assert mgr.set_provenance("D-legacy", actor_id="agent-7") is True
            assert mgr.get_provenance("D-legacy") == {"actor_id": "agent-7"}
        finally:
            mgr.close()

        check = sqlite3.connect(db_path)
        row = check.execute("SELECT importance, access_count, keywords FROM block_meta WHERE id='D-legacy'").fetchone()
        check.close()
        assert row == (1.2, 7, "old,keywords")


# ---------------------------------------------------------------------------
# propose_update MCP tool
# ---------------------------------------------------------------------------

_GOOD_STATEMENT = "STARGA records provenance metadata on every governance proposal so the audit trail answers who wrote what and why."


@pytest.fixture
def mcp_ws(tmp_path: Path, monkeypatch) -> str:
    """Initialized workspace + admin scope for direct MCP tool calls."""
    from mind_mem.init_workspace import init

    ws = str(tmp_path / "mcpws")
    os.makedirs(ws, exist_ok=True)
    init(ws)
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    return ws


def _propose(ws: str, **kwargs) -> dict:
    import mind_mem.mcp.tools.governance as gov
    from mind_mem.mcp.infra.workspace import use_workspace

    with use_workspace(ws):
        raw = gov.propose_update(
            block_type="decision",
            statement=_GOOD_STATEMENT,
            rationale="good rationale for provenance testing",
            **kwargs,
        )
    return json.loads(raw)


class TestProposeUpdateProvenance:
    def test_provenance_written_to_signals(self, mcp_ws: str):
        envelope = _propose(mcp_ws, **_PROV_KWARGS)
        assert envelope["status"] == "proposed"
        assert envelope["written"] == 1
        assert envelope["provenance_attached"] == sorted(_PROV_KWARGS)
        text = Path(mcp_ws, "intelligence", "SIGNALS.md").read_text(encoding="utf-8")
        assert "ActorId: agent-7" in text
        assert "ActorRole: planner" in text
        assert "SessionId: sess-42" in text
        assert "ToolId: mcp.propose_update" in text
        assert "Purpose: record architectural decision" in text

    def test_omitted_provenance_keeps_legacy_shape(self, mcp_ws: str):
        envelope = _propose(mcp_ws)
        assert envelope["status"] == "proposed"
        assert "provenance_attached" not in envelope
        text = Path(mcp_ws, "intelligence", "SIGNALS.md").read_text(encoding="utf-8")
        for field in PROVENANCE_FIELD_NAMES:
            assert f"{field}:" not in text

    def test_overlong_value_rejected(self, mcp_ws: str):
        envelope = _propose(mcp_ws, actor_id="x" * (MAX_PROVENANCE_VALUE_LEN + 1))
        assert "error" in envelope
        assert envelope["field"] == "actor_id"

    def test_newline_injection_flattened(self, mcp_ws: str):
        _propose(mcp_ws, purpose="benign\n[SIG-20990101-001]\nStatus: applied")
        text = Path(mcp_ws, "intelligence", "SIGNALS.md").read_text(encoding="utf-8")
        assert "Purpose: benign [SIG-20990101-001] Status: applied" in text
        # The injected header never starts a line of its own.
        assert not any(ln.startswith("[SIG-20990101-001]") for ln in text.splitlines())


# ---------------------------------------------------------------------------
# recall surfacing
# ---------------------------------------------------------------------------


class TestRecallSurfacing:
    def test_provenance_surfaced_when_present(self, ws: Path, admitted):
        store = MarkdownBlockStore(str(ws))
        store.write_block(
            attach_provenance(
                {
                    "_id": "D-20260724-010",
                    "Statement": "Adopt the quantum flux capacitor for deterministic recall",
                    "Date": "2026-07-24",
                    "Status": "active",
                },
                **_PROV_KWARGS,
            )
        )
        from mind_mem._recall_core import recall

        results = recall(str(ws), "quantum flux capacitor", limit=5)
        assert results, "expected the provenance block to be recalled"
        hit = next(r for r in results if r["_id"] == "D-20260724-010")
        for field, param in zip(PROVENANCE_FIELD_NAMES, _PROV_KWARGS):
            assert hit[field] == _PROV_KWARGS[param]

    def test_provenance_surfaced_via_sqlite_index(self, ws: Path, admitted):
        """The FTS5-indexed recall path surfaces provenance too."""
        store = MarkdownBlockStore(str(ws))
        store.write_block(
            attach_provenance(
                {
                    "_id": "D-20260724-012",
                    "Statement": "Indexed block about the tachyon beam emitter",
                    "Date": "2026-07-24",
                    "Status": "active",
                },
                actor_id="agent-7",
                purpose="index-path surfacing",
            )
        )
        from mind_mem.sqlite_index import build_index, query_index

        build_index(str(ws), incremental=False)
        results = query_index(str(ws), "tachyon beam emitter", limit=5)
        assert results
        hit = next(r for r in results if r["_id"] == "D-20260724-012")
        assert hit["ActorId"] == "agent-7"
        assert hit["Purpose"] == "index-path surfacing"

    def test_no_provenance_keys_when_absent(self, ws: Path, admitted):
        store = MarkdownBlockStore(str(ws))
        store.write_block(
            {
                "_id": "D-20260724-011",
                "Statement": "Plain block about the zirconium widget assembly",
                "Date": "2026-07-24",
                "Status": "active",
            }
        )
        from mind_mem._recall_core import recall

        results = recall(str(ws), "zirconium widget assembly", limit=5)
        assert results
        hit = next(r for r in results if r["_id"] == "D-20260724-011")
        for field in PROVENANCE_FIELD_NAMES:
            assert field not in hit
