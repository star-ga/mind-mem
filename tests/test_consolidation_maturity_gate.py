# Copyright 2026 STARGA, Inc.
"""Acceptance gate for the consolidation maturity gate (Group H, safe half).

Four contracts, one test class each:

1. a *young* block is never consolidated when the gate is on;
2. a *mature* block is consolidated when the gate is on;
3. consolidation never touches either end of a *live contradiction*;
4. with the flag OFF the output is byte-identical to the pre-gate code —
   proven twice: against a frozen golden and against the actual ``main``
   implementation loaded from git.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from mind_mem.cognitive_forget import (
    BlockCognition,
    BlockLifecycle,
    ConsolidationConfig,
    plan_consolidation,
)
from mind_mem.consolidation_maturity_gate import (
    HOLD_CONTRADICTED,
    HOLD_YOUNG,
    MaturityGate,
    MaturityGateConfig,
    collect_contradicted_block_ids,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

MATURE_META = {"Status": "active", "Lifecycle": "durable"}  # score 0.5
YOUNG_META = {"Status": "wip", "Lifecycle": "ephemeral"}  # score 0.15


def _stale_block(block_id: str) -> BlockCognition:
    """A block the ungated cycle always wants to mark (low value + stale)."""
    return BlockCognition(
        block_id=block_id,
        importance=0.1,
        last_accessed="2020-01-01T00:00:00Z",
        access_count=0,
        created_at="2019-01-01T00:00:00Z",
        lifecycle=BlockLifecycle.ACTIVE,
    )


def _gate(**kwargs) -> MaturityGate:
    meta = kwargs.pop("block_meta", None)
    contradicted = kwargs.pop("contradicted_ids", None)
    return MaturityGate(
        MaturityGateConfig(enabled=True, **kwargs),
        block_meta=meta,
        contradicted_ids=contradicted,
    )


class TestYoungBlocksAreHeld:
    def test_young_block_never_consolidated(self) -> None:
        block = _stale_block("DEC-20200101-001")
        ungated = plan_consolidation([block], now=NOW)
        assert ungated.mark == ["DEC-20200101-001"]

        gated = plan_consolidation(
            [block],
            now=NOW,
            gate=_gate(block_meta={"DEC-20200101-001": YOUNG_META}),
        )
        assert gated.as_dict() == {"mark": [], "archive": [], "forget": [], "total": 0}

    def test_unknown_block_is_treated_as_young(self) -> None:
        block = _stale_block("DEC-20200101-002")
        gated = plan_consolidation([block], now=NOW, gate=_gate(block_meta={}))
        assert gated.total == 0

    def test_hold_reason_is_reported(self) -> None:
        gate = _gate(block_meta={"A": YOUNG_META})
        assert gate.hold_reason("A") == HOLD_YOUNG
        assert gate.admits("A") is False


class TestMatureBlocksAreConsolidated:
    def test_mature_block_is_consolidated(self) -> None:
        block = _stale_block("DEC-20200101-003")
        gated = plan_consolidation(
            [block],
            now=NOW,
            gate=_gate(block_meta={"DEC-20200101-003": MATURE_META}),
        )
        assert gated.mark == ["DEC-20200101-003"]

    def test_explicit_maturity_field_wins(self) -> None:
        gate = _gate(block_meta={"A": {"Maturity": 0.9, **YOUNG_META}})
        assert gate.admits("A") is True
        assert gate.score("A") == pytest.approx(0.9)

    def test_edge_corroboration_matures_a_block(self) -> None:
        cfg = MaturityGateConfig(enabled=True, min_maturity=0.5)
        gate = MaturityGate(cfg, block_meta={"A": YOUNG_META}, edge_counts={"A": 5})
        assert gate.admits("A") is True  # 0.15 + 0.5 edge weight

    def test_evaluate_partitions_in_input_order(self) -> None:
        gate = _gate(block_meta={"M": MATURE_META, "Y": YOUNG_META})
        decision = gate.evaluate([_stale_block("M"), _stale_block("Y")])
        assert decision.admitted == ("M",)
        assert decision.held == ("Y",)
        assert decision.reasons["Y"] == HOLD_YOUNG


class TestLiveContradictionIsNeverCrossed:
    def test_mature_but_contradicted_blocks_are_held(self) -> None:
        blocks = [_stale_block("DEC-20200101-010"), _stale_block("DEC-20200101-011")]
        meta = {b.block_id: MATURE_META for b in blocks}

        without = plan_consolidation(blocks, now=NOW, gate=_gate(block_meta=meta))
        assert without.mark == ["DEC-20200101-010", "DEC-20200101-011"]

        gated = plan_consolidation(
            blocks,
            now=NOW,
            gate=_gate(block_meta=meta, contradicted_ids={"DEC-20200101-010", "DEC-20200101-011"}),
        )
        assert gated.total == 0

    def test_contradiction_outranks_maturity(self) -> None:
        gate = _gate(block_meta={"A": {"Maturity": 1.0}}, contradicted_ids={"A"})
        assert gate.hold_reason("A") == HOLD_CONTRADICTED

    def test_contradicted_ids_read_from_lineage_graph(self, tmp_path) -> None:
        ws = tmp_path / "ws"
        (ws / ".mind-mem-index").mkdir(parents=True)
        conn = sqlite3.connect(ws / ".mind-mem-index" / "recall.db")
        conn.executescript(
            "CREATE TABLE co_retrieval (mem1_id TEXT, mem2_id TEXT, weight REAL, "
            "kind TEXT NOT NULL DEFAULT 'cooccurrence', PRIMARY KEY (mem1_id, mem2_id));"
        )
        conn.execute("INSERT INTO co_retrieval VALUES ('A', 'B', 1.0, 'contradicts')")
        conn.execute("INSERT INTO co_retrieval VALUES ('C', 'D', 1.0, 'cooccurrence')")
        conn.commit()
        conn.close()

        assert collect_contradicted_block_ids(str(ws)) == frozenset({"A", "B"})

    def test_missing_workspace_degrades_to_empty_set(self, tmp_path) -> None:
        assert collect_contradicted_block_ids(str(tmp_path / "nope")) == frozenset()

    def test_pre_lineage_schema_degrades_instead_of_raising(self, tmp_path) -> None:
        """A recall.db predating the ``kind`` column must not crash the gate."""
        ws = tmp_path / "ws"
        (ws / ".mind-mem-index").mkdir(parents=True)
        conn = sqlite3.connect(ws / ".mind-mem-index" / "recall.db")
        conn.executescript("CREATE TABLE co_retrieval (mem1_id TEXT, mem2_id TEXT, weight REAL);")
        conn.commit()
        conn.close()

        assert collect_contradicted_block_ids(str(ws)) == frozenset()

    def test_blank_workspace_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            collect_contradicted_block_ids("   ")


class TestFlagOffIsByteIdentical:
    """Zero-regression proof: the default path must not move a single byte."""

    CORPUS = [
        ("B-001", 0.10, "2020-01-01T00:00:00Z", 0, BlockLifecycle.ACTIVE),
        ("B-002", 0.90, "2025-12-31T00:00:00Z", 12, BlockLifecycle.ACTIVE),
        ("B-003", 0.40, None, 3, BlockLifecycle.MERGED),
        ("B-004", 0.20, "2020-06-01T00:00:00Z", 1, BlockLifecycle.ARCHIVED),
        ("B-005", 0.05, "2025-12-30T00:00:00Z", 0, BlockLifecycle.ACTIVE),
    ]
    GOLDEN = '{"archive": ["B-003"], "forget": ["B-004"], "mark": ["B-001"], "total": 3}'

    @staticmethod
    def _build(factory) -> list:
        return [
            factory(
                block_id=bid,
                importance=imp,
                last_accessed=last,
                access_count=count,
                created_at="2019-01-01T00:00:00Z",
                lifecycle=life,
            )
            for bid, imp, last, count, life in TestFlagOffIsByteIdentical.CORPUS
        ]

    def test_default_call_matches_golden(self) -> None:
        plan = plan_consolidation(self._build(BlockCognition), config=ConsolidationConfig(), now=NOW)
        assert json.dumps(plan.as_dict(), sort_keys=True) == self.GOLDEN

    def test_explicit_none_gate_matches_golden(self) -> None:
        plan = plan_consolidation(self._build(BlockCognition), now=NOW, gate=None)
        assert json.dumps(plan.as_dict(), sort_keys=True) == self.GOLDEN

    def test_disabled_gate_object_matches_golden(self) -> None:
        gate = MaturityGate(MaturityGateConfig(), block_meta={})  # enabled defaults to False
        plan = plan_consolidation(self._build(BlockCognition), now=NOW, gate=gate)
        assert json.dumps(plan.as_dict(), sort_keys=True) == self.GOLDEN

    def test_matches_pre_gate_implementation_from_git(self, tmp_path) -> None:
        """Load ``main``'s cognitive_forget verbatim and diff the outputs."""
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            blob = subprocess.run(
                ["git", "-C", repo, "show", "main:src/mind_mem/cognitive_forget.py"],
                capture_output=True,
                check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):  # pragma: no cover
            pytest.skip("git reference revision unavailable")

        ref_path = tmp_path / "pre_gate_cognitive_forget.py"
        ref_path.write_bytes(blob)
        spec = importlib.util.spec_from_file_location("_pre_gate_cognitive_forget", ref_path)
        assert spec is not None and spec.loader is not None
        ref = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ref
        try:
            spec.loader.exec_module(ref)
            assert not hasattr(ref.plan_consolidation, "__wrapped__")
            ref_blocks = [
                ref.BlockCognition(
                    block_id=bid,
                    importance=imp,
                    last_accessed=last,
                    access_count=count,
                    created_at="2019-01-01T00:00:00Z",
                    lifecycle=ref.BlockLifecycle(life.value),
                )
                for bid, imp, last, count, life in self.CORPUS
            ]
            expected = ref.plan_consolidation(ref_blocks, now=NOW).as_dict()
        finally:
            sys.modules.pop(spec.name, None)

        got = plan_consolidation(self._build(BlockCognition), now=NOW).as_dict()
        assert json.dumps(got, sort_keys=True) == json.dumps(expected, sort_keys=True)


class TestConfigValidation:
    @pytest.mark.parametrize("bad", [-0.1, 1.5, "high"])
    def test_min_maturity_out_of_range_is_rejected(self, bad) -> None:
        with pytest.raises(ValueError):
            MaturityGateConfig(enabled=True, min_maturity=bad)

    def test_non_bool_enabled_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            MaturityGateConfig(enabled="yes")  # type: ignore[arg-type]

    def test_gate_requires_a_config_object(self) -> None:
        with pytest.raises(TypeError):
            MaturityGate({"enabled": True})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# MCP tool surface
# ---------------------------------------------------------------------------

# Captured from the pre-change tool on the fixture workspace below.
MCP_GOLDEN = (
    '{\n  "config": {\n    "importance_threshold": 0.25,\n    "stale_days": 14,\n'
    '    "archive_after_days": 60,\n    "grace_days": 30\n  },\n  "plan": {\n'
    '    "mark": [\n      "DEC-20200101-001",\n      "DEC-20200101-002"\n    ],\n'
    '    "archive": [],\n    "forget": [],\n    "total": 2\n  },\n'
    '  "_schema_version": "1.0"\n}'
)


@pytest.fixture()
def mcp_workspace(tmp_path):
    ws = tmp_path / "ws"
    (ws / "decisions").mkdir(parents=True)
    # Canonical index layout, owned by mind_mem.sqlite_index.DB_REL_PATH.
    # The fixture used to build ".sqlite_index/index.db", a path nothing in
    # the product writes, so these tests passed against a phantom corpus.
    (ws / ".mind-mem-index").mkdir(parents=True)
    conn = sqlite3.connect(ws / ".mind-mem-index" / "recall.db")
    conn.executescript(
        "CREATE TABLE blocks (id TEXT PRIMARY KEY, type TEXT, file TEXT, line INTEGER, "
        "status TEXT, date TEXT, speaker TEXT, tags TEXT, dia_id TEXT, parent_id TEXT, json_blob TEXT);"
        "CREATE TABLE block_meta (id TEXT PRIMARY KEY, importance REAL, last_accessed TEXT, access_count INTEGER);"
    )
    rows = [
        ("DEC-20200101-001", "active", json.dumps({"Lifecycle": "durable"})),
        ("DEC-20200101-002", "wip", json.dumps({"Lifecycle": "ephemeral"})),
    ]
    for bid, status, blob in rows:
        conn.execute(
            "INSERT INTO blocks VALUES (?, 'decision', 'decisions/DECISIONS.md', 1, ?, '2020-01-01', '', '', '', '', ?)",
            (bid, status, blob),
        )
        conn.execute("INSERT INTO block_meta VALUES (?, 0.1, '2020-01-02T00:00:00Z', 0)", (bid,))
    conn.commit()
    conn.close()
    return ws


class TestMcpToolSurface:
    def test_flag_off_output_is_byte_identical(self, mcp_workspace) -> None:
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.consolidation import plan_consolidation as tool

        with use_workspace(str(mcp_workspace)):
            assert tool() == MCP_GOLDEN
            assert tool(maturity_gate=False) == MCP_GOLDEN

    def test_flag_on_holds_the_young_block(self, mcp_workspace) -> None:
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.consolidation import plan_consolidation as tool

        with use_workspace(str(mcp_workspace)):
            payload = json.loads(tool(maturity_gate=True, min_maturity=0.5))

        assert payload["plan"]["mark"] == ["DEC-20200101-001"]
        assert payload["maturity_gate"]["held"] == ["DEC-20200101-002"]
        assert payload["maturity_gate"]["reasons"]["DEC-20200101-002"] == HOLD_YOUNG

    def test_flag_on_rejects_a_bad_threshold(self, mcp_workspace) -> None:
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.consolidation import plan_consolidation as tool

        with use_workspace(str(mcp_workspace)):
            payload = json.loads(tool(maturity_gate=True, min_maturity=2.0))
        assert "error" in payload
