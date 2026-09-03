# Copyright 2026 STARGA, Inc.
"""GAP-1 — a detector finding is content, so it enters through the gate.

``intel_scan`` is the third command of the README's 30-second demo
(``mind-mem-scan``). Measured on 5.0.1, on a fresh workspace::

    write_contradictions([...], ws, report)
      -> intelligence/CONTRADICTIONS.md gains  [C-YYYYMMDD-001]  Status: open
      -> evidence_chain +0   hash_chain +0   .mind-mem-audit/chain.jsonl +0
      -> recall(ws, "...") returns it FIRST

A served block, minted by a `open(path, "a")` that never touched the
store, with nothing in any ledger naming where it came from. The pin in
``test_governed_write_paths.PENDING_CORPUS_WRITERS`` called it "lower
severity ... internally-derived content"; provenance is not the axis. The
thesis is *served content with no record*, and a contradiction finding is
served content.

What closes it, and why it is closed by construction rather than by
convention:

1. ``corpus_registry.CORPUS_TABLE`` routes the ``DREF`` prefix, so
   ``write_block`` will ACCEPT a drift finding. Before this the store
   *refused* a ``DREF-`` id — which is precisely why the detector spliced
   the file by hand, and why the fix has to start at the routing table.
2. ``IngestTier.DETECTOR_FINDING`` mints ``Status.OPEN`` and is
   **confined** by ``enums.TIER_ID_PREFIXES`` to ``C-``/``DREF-`` ids.
   ``require_admission`` refuses that receipt for any other prefix and
   for any other status, so the one tier that may mint a status recall
   recognises can reach exactly two corpora and exactly one status.
3. Both detectors funnel through one writer, ``intel_scan._write_findings``,
   which opens ``admit_batch`` and calls ``store.write_block``. Neither
   detector opens a corpus file any more — there is no second way in.
4. The two ``PENDING_CORPUS_WRITERS`` entries are gone, so
   ``test_governed_write_paths`` now enforces this instead of exempting it.

Every negative assertion below is paired with a positive control: the
refusals are shown beside the same write succeeding, and the ledger
counters are shown moving before they are asserted not to move.
"""

from __future__ import annotations

import ast
import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Iterator

import pytest
from _ledger_rows import authorisation_rows, chain_rows, count_chain_authorisations

from mind_mem.admission import UngatedWriteError
from mind_mem.enums import INITIAL_STATUS, TIER_ID_PREFIXES, IngestTier, Status
from mind_mem.governance_gate import get_gate
from mind_mem.init_workspace import init as init_workspace
from mind_mem.intel_scan import IntelReport, write_contradictions, write_drift
from mind_mem.storage import get_block_store

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "mind_mem")


# ---------------------------------------------------------------------------
# Fixtures + measurement helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def ws(tmp_path: Any) -> Iterator[str]:
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    init_workspace(workspace)
    yield workspace


@pytest.fixture
def report() -> IntelReport:
    return IntelReport()


def _today() -> str:
    return datetime.now().strftime("%Y%m%d")


def _contradiction() -> dict[str, Any]:
    return {
        "sig1": {"decision": "D-20260101-001", "sig": {"id": "SIG-20260101-011", "domain": "deploy", "modality": "must"}},
        "sig2": {"decision": "D-20260101-002", "sig": {"id": "SIG-20260101-012", "domain": "deploy", "modality": "must_not"}},
        "severity": "critical",
        "reason": "modality conflict: must vs must_not on axis=deploy",
    }


def _drift() -> dict[str, Any]:
    return {
        "severity": "medium",
        "signal": "stale_decisions",
        "summary": "12 decisions untouched for 90 days",
        "evidence": ["D-20260101-001", "D-20260101-002"],
    }


def _evidence_rows(workspace: str) -> list[dict[str, Any]]:
    path = os.path.join(workspace, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _authorisations(workspace: str) -> list[dict[str, Any]]:
    """Evidence rows that authorised a detector run — see tests/_ledger_rows."""
    return authorisation_rows(_evidence_rows(workspace))


def _hash_chain_rows(workspace: str) -> int:
    path = os.path.join(workspace, "memory", "hash_chain_v2.db")
    if not os.path.isfile(path):
        return 0
    con = sqlite3.connect(path)
    try:
        return int(con.execute("SELECT COUNT(*) FROM hash_chain").fetchone()[0])
    finally:
        con.close()


def _field_audit_rows(workspace: str) -> int:
    path = os.path.join(workspace, ".mind-mem-audit", "chain.jsonl")
    if not os.path.isfile(path):
        return 0
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _finding_block(block_id: str, status: object = None) -> dict[str, Any]:
    """A finding-shaped block. ``status`` defaults to the tier's own row."""
    row = INITIAL_STATUS[IngestTier.DETECTOR_FINDING]
    assert row is not None, "the detector tier must name the status it mints"
    block: dict[str, Any] = {
        "_id": block_id,
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Statement": "modality conflict: must vs must_not on axis=deploy",
        "Status": row.value if status is None else status,
    }
    if status is _ABSENT:
        block.pop("Status")
    return block


_ABSENT = object()


# ---------------------------------------------------------------------------
# The capability must survive — a governed finding is still a served finding
# ---------------------------------------------------------------------------


class TestTheFindingSurvivesTheGate:
    """Governing the write must not cost the product the finding."""

    def test_a_contradiction_is_written_readable_and_served(self, ws: str, report: IntelReport) -> None:
        from mind_mem.recall import recall

        write_contradictions([_contradiction()], ws, report)

        cid = f"C-{_today()}-001"
        block = get_block_store(ws).get_by_id(cid)
        assert block is not None, "the finding must still reach the corpus"
        assert block["Status"] == Status.OPEN.value
        assert block["Objects"] == ["D-20260101-001", "D-20260101-002"], (
            "the list fields validity_gate reads must survive the store round-trip"
        )

        served = [hit.get("_id") for hit in recall(ws, "modality conflict deploy", limit=5)]
        assert cid in served, f"recall stopped serving the contradiction: {served}"

    def test_a_drift_signal_is_written_readable_and_served(self, ws: str, report: IntelReport) -> None:
        from mind_mem.recall import recall

        write_drift([_drift()], ws, report)

        dref = f"DREF-{_today()}-001"
        block = get_block_store(ws).get_by_id(dref)
        assert block is not None, "the DREF prefix must route, or the finding is lost"
        assert block["Status"] == Status.OPEN.value
        assert block["Evidence"] == ["D-20260101-001", "D-20260101-002"]

        served = [hit.get("_id") for hit in recall(ws, "stale decisions untouched", limit=5)]
        assert dref in served, f"recall stopped serving the drift signal: {served}"

    def test_open_is_still_a_status_recall_serves(self) -> None:
        """The regression the tier table could have caused, pinned.

        ``UNADMITTED`` is derived from ``INITIAL_STATUS``. A naive
        derivation (``not is_servable(row)``) would have pulled ``open``
        into the withheld set the moment ``DETECTOR_FINDING`` existed —
        withholding every ``open`` block in every corpus (task loops
        included) to record one scanner. ``enums.mints_quarantine``
        excludes confined tiers for exactly this reason.
        """
        from mind_mem.admissibility import RECOGNISED_STATUSES, UNADMITTED, is_admissible_status

        assert Status.OPEN.value not in UNADMITTED
        assert Status.OPEN.value in RECOGNISED_STATUSES
        assert is_admissible_status(Status.OPEN.value)
        # Positive control: the counter-example still IS withheld.
        assert Status.QUARANTINED.value in UNADMITTED
        assert not is_admissible_status(Status.QUARANTINED.value)


# ---------------------------------------------------------------------------
# ...and now it leaves a record
# ---------------------------------------------------------------------------


class TestEveryFindingLeavesARecord:
    def test_both_gate_ledgers_gain_a_row_per_detector_run(self, ws: str, report: IntelReport) -> None:
        before_evidence = len(_authorisations(ws))
        before_hash = count_chain_authorisations(ws)

        write_contradictions([_contradiction()], ws, report)
        write_drift([_drift()], ws, report)

        rows = _authorisations(ws)
        assert len(rows) - before_evidence == 2, "one admission per detector run, and it is missing"
        assert count_chain_authorisations(ws) - before_hash == 2

        minted = {row["action"]: row for row in rows[before_evidence:]}
        assert set(minted) == {"CONTRADICT", "DRIFT"}, f"the run was recorded under the wrong verbs: {sorted(minted)}"
        for action, row in minted.items():
            assert row["actor"] == "intel_scan"
            assert row["metadata"]["action_verb"] == action
            assert row["metadata"]["findings"] == 1
        assert minted["CONTRADICT"]["target_file"] == "intelligence/CONTRADICTIONS.md"
        assert minted["DRIFT"]["target_file"] == "intelligence/DRIFT.md"

    def test_the_chain_entry_covers_the_ids_that_landed(self, ws: str, report: IntelReport) -> None:
        """A batch entry is only worth its id set; prove the two agree."""
        write_contradictions([_contradiction(), dict(_contradiction(), severity="medium")], ws, report)

        # The LAST chain row is now the scope's CLOSE record, not its
        # authorisation. The authorisation is the row that names the id
        # set the batch covered, so that is the one this test reads.
        row = authorisation_rows(chain_rows(ws))[-1]
        batch_id, action = row["block_id"], row["action"]
        assert action == "CONTRADICT"
        assert batch_id.startswith("intel-scan-contradictions-")

        store = get_block_store(ws)
        for index in (1, 2):
            assert store.get_by_id(f"C-{_today()}-{index:03d}") is not None

    def test_the_field_audit_sidecar_stays_out_of_this_path(self, ws: str, report: IntelReport) -> None:
        """Honest accounting of the THIRD ledger, with a positive control.

        The 5.0.1 measurement said "all three ledgers +0". Two of them —
        the evidence chain and the hash chain — are the gate's, and they
        move now. The third, ``.mind-mem-audit/chain.jsonl``, is the
        field-level sidecar written only by ``FieldAuditor.update_field``;
        no write path in the product appends to it (GAP-8, ledger
        hierarchy). It stays at +0 here, and this test says so out loud
        rather than letting a reader infer three from two.
        """
        from mind_mem.audit_chain import AuditChain

        before = _field_audit_rows(ws)
        write_contradictions([_contradiction()], ws, report)
        assert _field_audit_rows(ws) == before, "unexpected sidecar write — the ledger hierarchy changed"

        # Positive control: the counter CAN see a row, so the equality
        # above is a measurement and not a broken reader.
        AuditChain(ws).append("create_block", "intelligence/CONTRADICTIONS.md", agent="test")
        assert _field_audit_rows(ws) == before + 1


# ---------------------------------------------------------------------------
# The door cannot be avoided, widened, or spent on something else
# ---------------------------------------------------------------------------


class TestTheDoorCannotBeAvoided:
    def test_an_ungated_finding_write_is_refused(self, ws: str) -> None:
        store = get_block_store(ws)
        block = _finding_block(f"C-{_today()}-777")

        with pytest.raises(UngatedWriteError):
            store.write_block(block)

        # Positive control: the same block, inside the scope, lands.
        with get_gate(ws).admit_batch(
            action="CONTRADICT",
            batch_id="b-control",
            block_ids=[block["_id"]],
            content=block["_id"],
            tier=IngestTier.DETECTOR_FINDING,
        ):
            assert store.write_block(block) == block["_id"]
        assert store.get_by_id(block["_id"]) is not None

    @pytest.mark.parametrize("block_id", ["D-20260101-009", "INBOX-20260101-009", "T-20260101-009"])
    def test_the_detector_tier_cannot_write_outside_its_corpora(self, ws: str, block_id: str) -> None:
        """Confinement: the receipt is refused for any prefix but C/DREF."""
        store = get_block_store(ws)
        with get_gate(ws).admit_batch(
            action="CONTRADICT",
            batch_id="b-confined",
            block_ids=[block_id, f"C-{_today()}-001"],
            content="x",
            tier=IngestTier.DETECTOR_FINDING,
        ):
            with pytest.raises(UngatedWriteError, match="may only write ids prefixed"):
                store.write_block(_finding_block(block_id))
            # Positive control: the SAME receipt writes a finding id.
            assert store.write_block(_finding_block(f"C-{_today()}-001"))

    @pytest.mark.parametrize("status", ["active", "superseded", "pending", "", _ABSENT])
    def test_the_detector_tier_mints_exactly_one_status(self, ws: str, status: object) -> None:
        store = get_block_store(ws)
        block_id = f"DREF-{_today()}-002"
        with get_gate(ws).admit_batch(
            action="DRIFT",
            batch_id="b-status",
            block_ids=[block_id],
            content="x",
            tier=IngestTier.DETECTOR_FINDING,
        ):
            with pytest.raises(UngatedWriteError):
                store.write_block(_finding_block(block_id, status=status))
            # Positive control: its own row goes through, spelling and all.
            assert store.write_block(_finding_block(block_id, status="  Open  "))

    def test_an_ingest_tier_still_cannot_mint_a_finding(self, ws: str) -> None:
        """The general rule is untouched: quarantine tiers still refuse
        anything recall would serve, ``open`` included."""
        store = get_block_store(ws)
        block_id = f"C-{_today()}-003"
        with get_gate(ws).admit_batch(
            action="INGEST",
            batch_id="b-ingest",
            block_ids=[block_id],
            content="x",
            tier=IngestTier.EXTERNAL_INGEST,
        ):
            with pytest.raises(UngatedWriteError, match="recall would serve it"):
                store.write_block(_finding_block(block_id))
            # Positive control: that tier's own status still lands.
            assert store.write_block(_finding_block(block_id, status=Status.QUARANTINED.value))


# ---------------------------------------------------------------------------
# Drift guards — the three tables that must agree
# ---------------------------------------------------------------------------


class TestTheTablesStayInStep:
    def test_every_confined_prefix_is_routable_by_the_store(self) -> None:
        """A prefix the store cannot route is a tier that can write nothing.

        The confinement table lives in ``enums`` (a leaf module that
        imports no storage), so it cannot derive itself from the routing
        table. This is the drift guard that keeps the copy honest.
        """
        from mind_mem.block_store import _BLOCK_PREFIX_MAP

        assert TIER_ID_PREFIXES, "the confinement table is empty; the guard would pass over nothing"
        for tier, prefixes in TIER_ID_PREFIXES.items():
            assert prefixes, f"{tier.value} is confined to no prefix at all, so it can write nothing"
            for prefix in prefixes:
                assert prefix in _BLOCK_PREFIX_MAP, f"{tier.value} may mint {prefix!r}, which the store cannot route"

    def test_the_detector_corpora_are_the_two_intelligence_files(self) -> None:
        from mind_mem.block_store import _BLOCK_PREFIX_MAP

        assert _BLOCK_PREFIX_MAP["C"] == ("intelligence", "CONTRADICTIONS.md")
        assert _BLOCK_PREFIX_MAP["DREF"] == ("intelligence", "DRIFT.md")
        assert TIER_ID_PREFIXES[IngestTier.DETECTOR_FINDING] == frozenset({"C", "DREF"})

    def test_the_prefix_maps_stay_in_lockstep(self) -> None:
        from mind_mem.block_store import _BLOCK_PREFIX_MAP as store_map
        from mind_mem.mcp.tools.memory_ops import _BLOCK_PREFIX_MAP as mcp_map

        assert store_map == mcp_map

    def test_the_ids_the_detectors_mint_use_the_confined_prefixes(self, ws: str, report: IntelReport) -> None:
        """The confinement is worth nothing if the writer mints other ids."""
        write_contradictions([_contradiction()], ws, report)
        write_drift([_drift()], ws, report)

        allowed = TIER_ID_PREFIXES[IngestTier.DETECTOR_FINDING]
        minted = [str(block["_id"]) for block in get_block_store(ws).get_all(active_only=False)]
        findings = [bid for bid in minted if bid.split("-", 1)[0] in allowed]
        assert len(findings) == 2, f"the detectors minted {minted}, which the tier could not have written"

    def test_the_finding_tier_is_the_only_confined_one(self) -> None:
        """A second confined tier is a governance decision, not a default.

        Confinement is what buys ``DETECTOR_FINDING`` the right to mint a
        status recall recognises. A new row here would inherit that right
        by omission, so it has to be argued for in the same way
        ``test_quarantine_redteam`` argues for this one.
        """
        assert set(TIER_ID_PREFIXES) == {IngestTier.DETECTOR_FINDING}


# ---------------------------------------------------------------------------
# The hand-landing is gone — structurally, not by inspection
# ---------------------------------------------------------------------------


def _write_mode_opens(tree: ast.AST, qualname: str) -> list[int]:
    """Line numbers of ``open(..., <write mode>)`` inside *qualname*."""
    target: ast.AST | None = None
    stack: list[tuple[ast.AST, list[str]]] = [(tree, [])]
    while stack:
        node, path = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nested = path + [child.name]
                if ".".join(nested) == qualname:
                    target = child
                stack.append((child, nested))
            else:
                stack.append((child, path))
    assert target is not None, f"{qualname} does not exist"

    hits: list[int] = []
    for node in ast.walk(target):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "open":
            continue
        mode = node.args[1].value if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) else None
        for keyword in node.keywords:
            if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                mode = keyword.value.value
        if isinstance(mode, str) and any(char in mode for char in "aw+"):
            hits.append(node.lineno)
    return hits


class TestTheHandLandingIsGone:
    @pytest.mark.parametrize("qualname", ["write_contradictions", "write_drift", "_write_findings", "_recorded_findings"])
    def test_no_finding_writer_opens_a_corpus_file_for_writing(self, qualname: str) -> None:
        with open(os.path.join(_SRC, "intel_scan.py"), "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        assert _write_mode_opens(tree, qualname) == [], f"{qualname} still writes a corpus file directly"

    def test_the_matcher_finds_the_shape_it_is_looking_for(self) -> None:
        """Negative control: the 5.0.1 body must trip the same matcher."""
        pre_fix = (
            "def write_contradictions(c, ws, report):\n"
            '    with open(f"{ws}/intelligence/CONTRADICTIONS.md", "a") as f:\n'
            '        f.write("x")\n'
        )
        assert _write_mode_opens(ast.parse(pre_fix), "write_contradictions") == [2]

    def test_the_writer_opens_the_scope_and_uses_the_store(self) -> None:
        """The other half: it is not enough that the file write is gone."""
        with open(os.path.join(_SRC, "intel_scan.py"), "r", encoding="utf-8") as handle:
            source = handle.read()
        tree = ast.parse(source)
        (writer,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_write_findings"]
        called = {n.func.attr for n in ast.walk(writer) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert "admit_batch" in called, "the finding writer opens no admission scope"
        assert "write_block" in called, "the finding writer does not go through the store"
