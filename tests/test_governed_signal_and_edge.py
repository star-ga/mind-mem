# Copyright 2026 STARGA, Inc.
"""Two ENTER-side holes: a regex that served withheld content, and edges with no record.

The thesis is that nothing ENTERS, LEAVES or DIES without a gate receipt and
a chain record. The DIE half is closed across five stores and five doors. This
file closes two of the four writers that were landing *served* content around
the store.

**GAP-2 — a regex flipped a withheld signal to served.** Measured on a fresh
workspace, before the fix::

    _flip_signal_status(ws, sig, "applied")
    on-disk:  Status: pending -> applied
    recall(): before "not served"  ->  after "served"
    evidence_chain +0   hash_chain +0   audit_chain +0

``pending`` is withheld by ``admissibility.is_admissible_status``; ``applied``
is in ``RECOGNISED_STATUSES`` and therefore served. Moving between them is a
mint of servable content, and the gate's rule (I-4) is that only an approved
proposal may mint one. Reachable from ``mm graph-backfill --approve``.

**The root cause is the interesting part.** ``write_block`` REFUSED every
``SIG`` id — ``corpus_registry.CORPUS_TABLE`` carried the signals row with
``prefix=None``, so the write router had no file for it and raised *"no
canonical file mapping"*. With no store to write through, every signal writer
spliced ``intelligence/SIGNALS.md`` by hand, and an approval had no way to
change a status except a ``re.subn``. The fix removes the reason rather than
adding a check: the row gains its prefix, ``approve_relation_signals`` opens
``admit_proposal`` and writes the block *through the store*, and
``_flip_signal_status`` is deleted. :class:`TestTheSignalPrefixIsRoutable` is
the positive control that the row is really there — without it every test
below would be measuring a refusal, not an admission.

**GAP-3 — knowledge-graph edges entered with no record.** Measured::

    KnowledgeGraph.add_edge(...)
    edge stored;  evidence_chain +0   hash_chain +0   audit_chain +0

…through three doors (``graph_add_edge``, ``approve_edge``,
``approve_relation_signals``) while those edges ARE served — by ``graph_query``,
``traverse_graph`` and ``kg_fusion``. ``add_edge`` now begins with
``require_admission`` on a deterministic :func:`~mind_mem.knowledge_graph.edge_id`,
which is the same seam every ``BlockStore.write_block`` uses, placed at the one
choke point all three doors go through rather than at the three doors.

Every negative assertion here carries a positive control: the signal is shown
*unserved* before and *served* after by the same ``recall`` call, the edge is
shown *retrievable through ``graph_query``* after the scoped write, and the
chain is shown *empty* before it is counted. :class:`TestMutationTwin` restores
each pre-fix shape and reproduces the measured defect — a gate never observed
failing is not a gate.
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Iterator

import pytest
from _ledger_rows import authorisation_rows

from mind_mem.admission import UngatedWriteError
from mind_mem.enums import IngestTier
from mind_mem.governance_gate import GovernanceBypassError, evict_gate, get_gate
from mind_mem.graph_ingest import (
    APPLIED_STATUS,
    EDGE_ORIGIN,
    RELATION_APPROVAL_PREFIX,
    RelationTriple,
    approve_relation_signals,
    pending_relation_signals,
    stage_relation_signals,
)
from mind_mem.knowledge_graph import (
    EDGE_ORIGIN_DIRECT_ADMIN,
    EDGE_ORIGIN_HITL_APPROVED,
    KnowledgeGraph,
    Predicate,
    default_db_path,
    edge_id,
)
from mind_mem.recall import recall

CORPUS = ("decisions", "tasks", "entities", "intelligence", "memory")

#: The staged relation. Terms chosen so the signal's own ``Excerpt`` is the
#: only thing in the corpus that matches :data:`QUERY` — a served hit is then
#: unambiguously this block and not a neighbour dragged in by another leg.
SUBJECT = "pineapple"
OBJECT = "protocol"
SOURCE_BLOCK = "D-20260101-001"
QUERY = "pineapple depends_on protocol"
DATE = "2026-09-02"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[str]:
    """The zero-config default: blocks of record on the Markdown corpus."""
    ws = tmp_path / "ws"
    for sub in CORPUS:
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "intelligence" / "SIGNALS.md").write_text("# Captured Signals\n\n", encoding="utf-8")
    (ws / "decisions" / "DECISIONS.md").write_text("# Decisions\n\n", encoding="utf-8")
    (ws / "mind-mem.json").write_text(
        json.dumps({"recall": {"vector_enabled": False, "provider": "local"}, "block_store": {"backend": "markdown"}}),
        encoding="utf-8",
    )
    try:
        yield str(ws)
    finally:
        evict_gate(str(ws))


def _records(ws: str) -> list[dict]:
    path = os.path.join(ws, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _meta(record: dict) -> dict:
    return record.get("metadata") or {}


def _authorisations(ws: str) -> list[dict]:
    """Evidence rows that *authorised* something — one per governed scope.

    A scope now leaves two rows: the authorisation and the close record
    that says whether the write landed. Every count in this file means
    the first. ``tests/_ledger_rows`` holds that convention once.
    """
    return authorisation_rows(_records(ws))


def _rows_for(ws: str, target: str) -> list[dict]:
    return [r for r in _authorisations(ws) if r.get("target_block_id") == target]


def _signals_text(ws: str) -> str:
    return Path(ws, "intelligence", "SIGNALS.md").read_text(encoding="utf-8")


def _status_of(ws: str, sig_id: str) -> str:
    """Read ``Status:`` for one signal straight off disk."""
    match = re.search(rf"\[{re.escape(sig_id)}\](?:(?!\n\[)[\s\S])*?\nStatus: (\w+)", _signals_text(ws))
    assert match is not None, f"{sig_id} is not in SIGNALS.md at all; the fixture never staged it"
    return match.group(1)


def _served(ws: str) -> list[str]:
    return [str(r.get("_id") or r.get("id") or "") for r in recall(ws, QUERY, limit=10)]


def _edges(ws: str) -> list[Any]:
    with KnowledgeGraph(default_db_path(ws)) as kg:
        return kg._query_edges()


def _stage(ws: str, *, predicate: str = "depends_on", source: str = SOURCE_BLOCK) -> str:
    """Stage one relation signal and return its id."""
    triple = RelationTriple(
        subject=SUBJECT,
        predicate=predicate,
        object=OBJECT,
        source_block_id=source,
        confidence=0.7,
    )
    assert stage_relation_signals(ws, [triple], DATE) == 1
    pending = pending_relation_signals(ws)
    assert len(pending) == 1, f"the fixture staged nothing; every assertion below would be vacuous ({pending})"
    return str(pending[0]["signal_id"])


# ---------------------------------------------------------------------------
# The root cause: a corpus file the store could read and could not write
# ---------------------------------------------------------------------------


class TestTheSignalPrefixIsRoutable:
    """Positive control for the whole file: ``SIG`` resolves to its file.

    If this class goes red, ``write_block`` is refusing signal ids again and
    every governed-approval test below is measuring a refusal it mistook for
    an admission.
    """

    @pytest.mark.unit
    def test_the_registry_routes_sig_to_the_signals_file(self) -> None:
        from mind_mem.block_store import _BLOCK_PREFIX_MAP, _resolve_block_file
        from mind_mem.corpus_registry import CORPUS_TABLE

        assert _BLOCK_PREFIX_MAP["SIG"] == ("intelligence", "SIGNALS.md")
        row = next(entry for entry in CORPUS_TABLE if entry.label == "signals")
        assert row.prefix == "SIG", "the one corpus definition must be where the prefix comes from"
        assert _resolve_block_file("/ws", "SIG-20260902-001") == os.path.join("/ws", "intelligence", "SIGNALS.md")

    @pytest.mark.unit
    def test_the_store_writes_a_sig_block_inside_a_scope(self, workspace: str) -> None:
        """The capability the missing row denied, exercised end to end."""
        from mind_mem.block_store import MarkdownBlockStore

        store = MarkdownBlockStore(workspace)
        block = {"_id": "SIG-20260902-777", "Excerpt": "a governed signal", "Status": "pending", "Date": DATE}
        with get_gate(workspace).admit_proposal(proposal_id="TEST-SIG-WRITE", content="[]", actor="pytest"):
            assert store.write_block(block) == "SIG-20260902-777"
        assert "[SIG-20260902-777]" in _signals_text(workspace)
        assert store.get_by_id("SIG-20260902-777") is not None, "written but unreadable is the I-14 defect, not a fix"

    @pytest.mark.unit
    def test_an_ungated_sig_write_is_still_refused(self, workspace: str) -> None:
        """Routable is not ungoverned: the store seam still holds for SIG."""
        from mind_mem.block_store import MarkdownBlockStore

        store = MarkdownBlockStore(workspace)
        with pytest.raises(UngatedWriteError):
            store.write_block({"_id": "SIG-20260902-778", "Excerpt": "ungated", "Status": "pending"})
        assert "[SIG-20260902-778]" not in _signals_text(workspace), "the refused write landed anyway"


# ---------------------------------------------------------------------------
# GAP-2 — the withheld-to-served transition is a proposal apply
# ---------------------------------------------------------------------------


class TestApprovingARelationSignal:
    @pytest.mark.unit
    def test_a_pending_signal_is_withheld_and_an_approved_one_is_served(self, workspace: str) -> None:
        """The measured transition, with the same call answering both halves.

        ``recall`` before and ``recall`` after — one function, one query, one
        corpus. A test that only asserted the "after" would pass against a
        fixture whose signal was servable all along.
        """
        sig_id = _stage(workspace)
        assert _status_of(workspace, sig_id) == "pending"
        assert _served(workspace) == [], "a pending signal must not be served, or the 'after' proves nothing"

        report = approve_relation_signals(workspace, [sig_id])

        assert report == {"applied": [sig_id], "errors": {}}
        assert _status_of(workspace, sig_id) == APPLIED_STATUS
        assert sig_id in _served(workspace)

    @pytest.mark.unit
    def test_the_approval_writes_one_apply_row_naming_the_signal(self, workspace: str) -> None:
        """+0 rows was the defect. One row, and it says what happened."""
        sig_id = _stage(workspace)
        before = len(_authorisations(workspace))

        approve_relation_signals(workspace, [sig_id])

        rows = _rows_for(workspace, f"{RELATION_APPROVAL_PREFIX}{sig_id}")
        assert len(rows) == 1, f"expected exactly one authorisation record, got {len(rows)}"
        assert len(_authorisations(workspace)) == before + 1
        meta = _meta(rows[0])
        assert rows[0]["action"] == "APPLY"
        assert meta["door"] == "graph_ingest.approve_relation_signals"
        assert meta["signal_id"] == sig_id
        assert meta["status_from"] == "pending"
        assert meta["status_to"] == APPLIED_STATUS
        assert meta["edge_id"] == edge_id(SUBJECT, "depends_on", OBJECT, SOURCE_BLOCK)
        assert meta["ingest_tier"] == IngestTier.PROPOSAL_APPLY.value

    @pytest.mark.unit
    def test_the_edge_and_the_served_status_land_in_the_same_scope(self, workspace: str) -> None:
        sig_id = _stage(workspace)

        approve_relation_signals(workspace, [sig_id])

        edges = _edges(workspace)
        assert len(edges) == 1
        assert edges[0].metadata.get("origin") == EDGE_ORIGIN
        assert edges[0].metadata.get("signal_id") == sig_id
        assert _status_of(workspace, sig_id) == APPLIED_STATUS

    @pytest.mark.unit
    def test_a_refused_authorisation_changes_nothing(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fail closed: no edge, no status change, and it is reported."""
        sig_id = _stage(workspace)

        def _refuse(*_args: Any, **_kwargs: Any) -> Any:
            raise GovernanceBypassError("spec binding drifted")

        monkeypatch.setattr("mind_mem.governance_gate.GovernanceGate.admit_proposal", _refuse)
        report = approve_relation_signals(workspace, [sig_id])

        assert report["applied"] == []
        assert sig_id in report["errors"]
        assert _status_of(workspace, sig_id) == "pending", "the gate refused, so the signal must still be withheld"
        assert _edges(workspace) == []
        assert _served(workspace) == []

    @pytest.mark.unit
    def test_the_status_change_goes_through_the_store_not_a_splice(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """By-construction check that the regex is really gone.

        Break the *store* write and the status must not move. Under the old
        ``re.subn`` implementation the store was never involved, so this test
        could not fail — which is exactly what makes it a fix and not a
        rewording.
        """
        sig_id = _stage(workspace)

        def _boom(self: Any, block: dict) -> str:
            raise ValueError("store write refused by the test")

        monkeypatch.setattr("mind_mem.block_store.MarkdownBlockStore.write_block", _boom)
        report = approve_relation_signals(workspace, [sig_id])

        assert report["applied"] == []
        assert sig_id in report["errors"]
        assert _status_of(workspace, sig_id) == "pending"
        # And the direction the partial failure falls in: still withheld.
        # The edge is written first because it is idempotent, so a failed
        # status write is retryable; a served signal with no edge behind it
        # would not be.
        assert _served(workspace) == []
        assert "_flip_signal_status" not in dir(__import__("mind_mem.graph_ingest", fromlist=["x"]))

    @pytest.mark.unit
    def test_an_unapprovable_signal_mints_no_authorisation(self, workspace: str) -> None:
        """Routing refusals stay outside the scope, as the delete doors do."""
        _stage(workspace)
        before = len(_records(workspace))

        report = approve_relation_signals(workspace, ["SIG-20990101-001"])

        assert report["applied"] == []
        assert "SIG-20990101-001" in report["errors"]
        assert len(_records(workspace)) == before, "an approval that cannot happen left a record claiming it could"

    @pytest.mark.unit
    def test_an_approval_of_nothing_touches_nothing(self, workspace: str) -> None:
        """A run that approves nothing must leave no trace of the machinery.

        ``get_gate`` creates ``memory/`` and both ledger files as a side
        effect of existing, so a gate built up front would have an empty
        ``--approve`` list writing files the pre-5.0.2 code never wrote.
        Inertness is a claim about syscalls, so it is asserted about files.
        """
        ledgers = [Path(workspace, "memory", "evidence_chain.jsonl"), Path(workspace, "memory", "hash_chain_v2.db")]
        assert not any(p.exists() for p in ledgers), "the fixture pre-created a ledger; this test would be vacuous"

        assert approve_relation_signals(workspace, []) == {"applied": [], "errors": {}}
        assert not any(p.exists() for p in ledgers), "an empty approval built the gate anyway"

        report = approve_relation_signals(workspace, ["SIG-20990101-002"])
        assert report["applied"] == [] and "SIG-20990101-002" in report["errors"]
        assert not any(p.exists() for p in ledgers), "an unresolvable id built the gate anyway"

    @pytest.mark.unit
    def test_a_second_approval_mints_no_second_record(self, workspace: str) -> None:
        sig_id = _stage(workspace)
        approve_relation_signals(workspace, [sig_id])
        after_first = len(_authorisations(workspace))

        report = approve_relation_signals(workspace, [sig_id])

        assert report["applied"] == []
        assert sig_id in report["errors"]
        assert len(_authorisations(workspace)) == after_first

    @pytest.mark.unit
    def test_the_cli_door_reaches_the_governed_path(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """``mm graph-backfill --approve`` is the door the defect was reached by."""
        from mind_mem import mm_cli

        sig_id = _stage(workspace)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        assert _served(workspace) == []

        assert mm_cli.main(["graph-backfill", "--approve", sig_id]) == 0

        assert sig_id in _served(workspace)
        assert len(_rows_for(workspace, f"{RELATION_APPROVAL_PREFIX}{sig_id}")) == 1


# ---------------------------------------------------------------------------
# GAP-3 — an edge is content, so it needs a receipt
# ---------------------------------------------------------------------------


class TestGovernedEdges:
    @pytest.mark.unit
    def test_an_ungated_add_edge_is_refused_and_a_scoped_one_lands(self, workspace: str) -> None:
        """The refusal, with the positive control that the call otherwise works."""
        with KnowledgeGraph(default_db_path(workspace)) as kg:
            with pytest.raises(UngatedWriteError):
                kg.add_edge(SUBJECT, Predicate.DEPENDS_ON, OBJECT, source_block_id=SOURCE_BLOCK)
            assert kg._query_edges() == [], "the refused edge was written anyway"

            with get_gate(workspace).admit_proposal(proposal_id="TEST-EDGE", content="[]", actor="pytest"):
                edge = kg.add_edge(SUBJECT, Predicate.DEPENDS_ON, OBJECT, source_block_id=SOURCE_BLOCK)
            assert edge.object == OBJECT
            assert len(kg._query_edges()) == 1

    @pytest.mark.unit
    def test_a_refused_edge_resolves_no_entities(self, workspace: str) -> None:
        """The check runs before ``entities.resolve``, which writes rows itself.

        A refusal that had already half-populated the entity table would leave
        the database in a state no admission covers.
        """
        with KnowledgeGraph(default_db_path(workspace)) as kg:
            with pytest.raises(UngatedWriteError):
                kg.add_edge("brand new subject", Predicate.DEPENDS_ON, "brand new object", source_block_id=SOURCE_BLOCK)
            assert kg.entities.lookup("brand new subject") is None
            assert kg.entities.lookup("brand new object") is None

    @pytest.mark.unit
    def test_a_receipt_for_another_id_does_not_authorise_the_edge(self, workspace: str) -> None:
        """``covers`` binds: a block scope authorises its block, not the graph."""
        with KnowledgeGraph(default_db_path(workspace)) as kg:
            with get_gate(workspace).admit_block(
                action="WRITE",
                block_id="SIG-20260902-900",
                content="a different subject entirely",
                tier=IngestTier.AUTO_CAPTURE,
                actor="pytest",
            ):
                with pytest.raises(UngatedWriteError):
                    kg.add_edge(SUBJECT, Predicate.DEPENDS_ON, OBJECT, source_block_id=SOURCE_BLOCK)
            assert kg._query_edges() == []

    @pytest.mark.unit
    def test_the_admitted_edge_is_served_by_graph_query(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """The reason an edge needs a receipt: the tool hands it to a caller."""
        from mind_mem.mcp.tools.graph import graph_query

        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        with KnowledgeGraph(default_db_path(workspace)) as kg:
            with get_gate(workspace).admit_proposal(proposal_id="TEST-EDGE-SERVED", content="[]", actor="pytest"):
                kg.add_edge(SUBJECT, Predicate.DEPENDS_ON, OBJECT, source_block_id=SOURCE_BLOCK)

        out = json.loads(graph_query(SUBJECT, depth=1))
        assert "error" not in out, out
        served = json.dumps(out)
        assert OBJECT in served, f"graph_query does not serve the edge, so this file's premise is wrong: {out}"

    @pytest.mark.unit
    def test_edge_id_is_deterministic_and_canonical(self) -> None:
        """A receipt can only cover an id the caller can compute in advance."""
        base = edge_id(SUBJECT, "depends_on", OBJECT, SOURCE_BLOCK)
        assert base.startswith("E-") and len(base) == 18
        assert edge_id(f"  {SUBJECT.upper()} ", Predicate.DEPENDS_ON, OBJECT, f" {SOURCE_BLOCK} ") == base
        assert edge_id(SUBJECT, "supports", OBJECT, SOURCE_BLOCK) != base
        assert edge_id(SUBJECT, "depends_on", OBJECT, "D-20260101-002") != base
        with pytest.raises(ValueError):
            edge_id(SUBJECT, "not-a-predicate", OBJECT, SOURCE_BLOCK)
        with pytest.raises(ValueError):
            edge_id(SUBJECT, "depends_on", OBJECT, "   ")


class TestTheEdgeDoors:
    @pytest.fixture
    def mcp_ws(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> str:
        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
        monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)
        return workspace

    @pytest.mark.unit
    def test_the_admin_door_records_that_review_was_bypassed(self, mcp_ws: str) -> None:
        from mind_mem.mcp.tools.graph import graph_add_edge

        out = json.loads(graph_add_edge(SUBJECT, "depends_on", OBJECT, SOURCE_BLOCK))
        assert "error" not in out, out

        eid = edge_id(SUBJECT, "depends_on", OBJECT, SOURCE_BLOCK)
        rows = _rows_for(mcp_ws, eid)
        assert len(rows) == 1, f"the direct admin write left {len(rows)} records"
        assert _meta(rows[0])["door"] == "mcp.graph_add_edge"
        assert _meta(rows[0])["origin"] == EDGE_ORIGIN_DIRECT_ADMIN
        assert len(_edges(mcp_ws)) == 1

    @pytest.mark.unit
    def test_the_admin_door_reports_a_refusal_and_writes_nothing(self, mcp_ws: str, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem.mcp.tools.graph import graph_add_edge

        def _refuse(*_args: Any, **_kwargs: Any) -> Any:
            raise GovernanceBypassError("spec binding drifted")

        monkeypatch.setattr("mind_mem.governance_gate.GovernanceGate.admit_proposal", _refuse)
        out = json.loads(graph_add_edge(SUBJECT, "depends_on", OBJECT, SOURCE_BLOCK))

        assert out["error"] == "Edge refused by governance."
        assert _edges(mcp_ws) == []

    @pytest.mark.unit
    def test_the_approval_door_records_the_proposal_it_committed(self, mcp_ws: str) -> None:
        from mind_mem.mcp.tools.graph import approve_edge, propose_edge

        staged = json.loads(propose_edge(SUBJECT, "depends_on", OBJECT, SOURCE_BLOCK))
        assert "error" not in staged, staged
        pid = staged["proposal_id"]
        assert _edges(mcp_ws) == [], "propose must stage only; the approval below would otherwise prove nothing"

        out = json.loads(approve_edge(pid))

        assert out.get("approved") == pid, out
        rows = _rows_for(mcp_ws, pid)
        assert len(rows) == 1
        assert _meta(rows[0])["door"] == "mcp.approve_edge"
        assert _meta(rows[0])["origin"] == EDGE_ORIGIN_HITL_APPROVED
        assert _meta(rows[0])["edge_id"] == edge_id(SUBJECT, "depends_on", OBJECT, SOURCE_BLOCK)
        assert len(_edges(mcp_ws)) == 1

    @pytest.mark.unit
    def test_an_unknown_proposal_mints_no_authorisation(self, mcp_ws: str) -> None:
        from mind_mem.mcp.tools.graph import approve_edge

        before = len(_records(mcp_ws))
        out = json.loads(approve_edge("EP-0000000000000000"))

        assert "unknown edge proposal" in out["error"]
        assert len(_records(mcp_ws)) == before

    @pytest.mark.unit
    def test_a_rejected_proposal_mints_no_authorisation(self, mcp_ws: str) -> None:
        from mind_mem.mcp.tools.graph import approve_edge, propose_edge, reject_edge

        pid = json.loads(propose_edge(SUBJECT, "depends_on", OBJECT, SOURCE_BLOCK))["proposal_id"]
        assert "error" not in json.loads(reject_edge(pid))
        before = len(_records(mcp_ws))

        out = json.loads(approve_edge(pid))

        assert "rejected" in out["error"]
        assert len(_records(mcp_ws)) == before
        assert _edges(mcp_ws) == []


# ---------------------------------------------------------------------------
# Mutation twins — restore each pre-fix shape and reproduce the measurement
# ---------------------------------------------------------------------------


class TestMutationTwin:
    """A gate never observed failing is not a gate."""

    @pytest.mark.unit
    def test_the_regex_flip_reproduces_the_measured_defect(self, workspace: str) -> None:
        """The deleted ``_flip_signal_status``, verbatim, against this fixture.

        It must serve the signal and move no ledger — which is the defect
        report — and by doing so it proves that
        :meth:`TestApprovingARelationSignal.test_a_pending_signal_is_withheld_and_an_approved_one_is_served`
        is measuring the approval path and not some ambient property of the
        corpus.
        """
        from mind_mem.mind_filelock import FileLock

        def _flip_signal_status(ws: str, signal_id: str, new_status: str) -> bool:
            path = os.path.join(ws, "intelligence", "SIGNALS.md")
            if not os.path.isfile(path):
                return False
            with FileLock(path):
                with open(path, "r", encoding="utf-8") as handle:
                    content = handle.read()
                pattern = re.compile(rf"(\[{re.escape(signal_id)}\](?:(?!\n\[)[\s\S])*?\nStatus: )\w+")
                new_content, hits = pattern.subn(rf"\g<1>{new_status}", content, count=1)
                if hits == 0:
                    return False
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(new_content)
            return True

        sig_id = _stage(workspace)
        before_rows = len(_records(workspace))
        assert _served(workspace) == []

        assert _flip_signal_status(workspace, sig_id, APPLIED_STATUS) is True

        assert _status_of(workspace, sig_id) == APPLIED_STATUS
        assert sig_id in _served(workspace), "the twin must reproduce a WORKING flip, not a broken one"
        assert len(_records(workspace)) == before_rows, "the twin did not actually bypass the gate"

    @pytest.mark.unit
    def test_without_require_admission_the_edge_lands_unrecorded(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """M-4 reproduced: edge stored, all three ledgers +0."""
        monkeypatch.setattr("mind_mem.knowledge_graph.require_admission", lambda *_a, **_k: None)

        before_rows = len(_records(workspace))
        with KnowledgeGraph(default_db_path(workspace)) as kg:
            kg.add_edge(SUBJECT, Predicate.DEPENDS_ON, OBJECT, source_block_id=SOURCE_BLOCK)
            assert len(kg._query_edges()) == 1, "the twin must reproduce a working write, not a broken one"

        assert len(_records(workspace)) == before_rows, "the twin did not actually bypass the gate"


# ---------------------------------------------------------------------------
# The structural half — a new ungoverned edge writer fails the build
# ---------------------------------------------------------------------------
#
# ``tests/test_governed_write_paths.py`` does exactly this for
# ``BlockStore.write_block``: a static AST scan over the source on disk, an
# allowlist that IS the invariant, and three guards so a matcher that silently
# matches nothing cannot report a clean tree. The graph needs the same thing,
# because the receipt on ``add_edge`` only binds the writers a reviewer
# remembers to route through it.
#
# The scanners live here rather than in ``tests/_write_path_scan.py`` because
# that module is owned by the concurrent I-14 unit; folding these two functions
# into it (and this allowlist into that file's) is the right end state and
# costs nothing but a merge.
#
# Like that module, nothing below imports ``mind_mem``: the invariant is
# checked against the text on disk, so a monkeypatch cannot satisfy it.

SRC_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "mind_mem")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Methods that commit an edge to the source-of-truth ``edges`` table.
#:
#: ``approve_edge`` is here beside ``add_edge`` because it is the HITL flow's
#: sole committer: it delegates, so the *receipt* is enforced once in
#: ``add_edge``, but the *scope* has to be opened by whoever calls it, and a
#: caller that forgets is exactly the hole this scan closes.
EDGE_COMMITTERS: frozenset[str] = frozenset({"add_edge", "approve_edge"})

#: Scopes that authorise an edge write.
ADMIT_OPENERS: frozenset[str] = frozenset({"admit_block", "admit_batch", "admit_proposal"})

#: The caller opens its own scope.
LOCAL = "local"
#: The callee runs inside its caller's scope and opens none of its own.
DELEGATES = "delegates"

#: ``(file, enclosing qualname) -> LOCAL | DELEGATES``. THIS IS THE INVARIANT.
#: Do not add an entry without the justification comment.
SANCTIONED_EDGE_COMMITTERS: dict[tuple[str, str], str] = {
    # The HITL flow's committer. It resolves the proposal and hands the tuple
    # to add_edge, which is where the receipt is required; the scope belongs
    # to its door (mcp.tools.graph.approve_edge, below).
    ("src/mind_mem/knowledge_graph.py", "KnowledgeGraph.approve_edge"): DELEGATES,
    # One operator approval of one staged relation signal: admit_proposal
    # covering the edge AND the signal's move to a served status.
    ("src/mind_mem/graph_ingest.py", "approve_relation_signals"): LOCAL,
    # The admin-scoped direct write. Its scope records origin="direct_admin",
    # so the chain says an admin bypassed review rather than merely that an
    # edge appeared.
    ("src/mind_mem/mcp/tools/graph.py", "graph_add_edge"): LOCAL,
    # The approval door. Resolves the proposal first, outside the scope, so an
    # approval that cannot happen mints no authorisation.
    ("src/mind_mem/mcp/tools/graph.py", "approve_edge"): LOCAL,
}


def _iter_src_files() -> tuple[str, ...]:
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(SRC_ROOT):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                found.append(os.path.join(dirpath, name))
    return tuple(sorted(found))


def _parse(path: str) -> ast.Module:
    with open(path, "r", encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=path)


def _relpath(path: str) -> str:
    return os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")


def _qualnames(tree: ast.AST) -> dict[ast.AST, str]:
    """Map every node to the dotted ``Class.func`` name enclosing it."""
    out: dict[ast.AST, str] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}.{child.name}" if prefix else child.name
                out[child] = name
                walk(child, name)
            else:
                out[child] = prefix
                walk(child, prefix)

    walk(tree, "")
    return out


def _is_edge_commit(node: ast.AST) -> bool:
    """True for a call that commits an edge.

    ``add_edge`` alone is not enough: ``causal_graph.CausalGraph.add_edge`` is
    a different method on a different table (block-id dependency edges, no
    provenance anchor, not served by ``graph_query``). It is told apart with
    no import at all — the knowledge-graph signature takes ``source_block_id``
    keyword-only and the causal one has no such parameter.
    """
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr not in EDGE_COMMITTERS:
        return False
    if node.func.attr == "approve_edge":
        return True
    return any(kw.arg == "source_block_id" for kw in node.keywords)


def find_edge_commits(tree: ast.AST, rel: str) -> list[tuple[str, str, int]]:
    """Every edge-committing CALL in *tree*, as ``(file, qualname, line)``."""
    quals = _qualnames(tree)
    found: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if _is_edge_commit(node):
            found.append((rel, quals.get(node, ""), node.lineno))
    return sorted(found)


def _function_node(tree: ast.AST, qual: str) -> ast.AST | None:
    for node, name in _qualnames(tree).items():
        if name == qual and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


def _opens_admission(func: ast.AST) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ADMIT_OPENERS:
            return True
    return False


def _calls_require_admission(func: ast.AST) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "require_admission":
            return True
    return False


class TestNoUngovernedEdgeWriters:
    """The allowlist IS the invariant; the guards keep the scan honest."""

    @pytest.fixture(scope="class")
    def files(self) -> tuple[str, ...]:
        return _iter_src_files()

    @pytest.mark.unit
    def test_the_scan_sees_the_whole_package(self, files: tuple[str, ...]) -> None:
        """A broken walk trips here before any allowlist comparison passes."""
        assert len(files) >= 250, f"only {len(files)} source files scanned; the walk is broken, not the tree"

    @pytest.mark.unit
    def test_the_matcher_finds_a_known_call_site(self, files: tuple[str, ...]) -> None:
        """Positive control: a call site that is definitely present."""
        found = {(rel, qual) for path in files for rel, qual, _line in find_edge_commits(_parse(path), _relpath(path))}
        assert ("src/mind_mem/graph_ingest.py", "approve_relation_signals") in found

    @pytest.mark.unit
    def test_the_matcher_detects_a_synthetic_bypass(self) -> None:
        """Negative control, tree-independent: run it against known-bad source."""
        rogue = "class Rogue:\n    def go(self, kg):\n        kg.add_edge('a', 'depends_on', 'b', source_block_id='D-1')\n"
        assert find_edge_commits(ast.parse(rogue), "synthetic.py") == [("synthetic.py", "Rogue.go", 3)]

    @pytest.mark.unit
    def test_the_matcher_ignores_the_causal_graph(self) -> None:
        """The disambiguation, asserted rather than assumed.

        ``CausalGraph.add_edge`` is a different table with different rules; a
        scan that flagged it would push this file into pinning a method it has
        no opinion about.
        """
        causal = "def go(g):\n    g.add_edge('D-002', 'D-001', 'depends_on', weight=1.0)\n"
        assert find_edge_commits(ast.parse(causal), "synthetic.py") == []

    @pytest.mark.unit
    def test_no_ungoverned_edge_committers(self, files: tuple[str, ...]) -> None:
        unsanctioned = sorted(
            {
                (rel, qual, line)
                for path in files
                for rel, qual, line in find_edge_commits(_parse(path), _relpath(path))
                if (rel, qual) not in SANCTIONED_EDGE_COMMITTERS
            }
        )
        if unsanctioned:
            listing = "\n".join(f"  {rel}:{line}  in  {qual}" for rel, qual, line in unsanctioned)
            pytest.fail(
                "UNGOVERNED EDGE WRITE — these commit a typed edge outside the sanctioned set.\n"
                "An edge is served content (graph_query, traverse_graph, kg_fusion), so it needs\n"
                "an admit_proposal scope at the door and an entry here saying why:\n\n" + listing
            )

    @pytest.mark.unit
    def test_every_sanctioned_caller_opens_a_scope(self, files: tuple[str, ...]) -> None:
        """An allowlist entry is a promise that a scope is actually opened."""
        by_file = {_relpath(path): _parse(path) for path in files}
        failures: list[str] = []
        for (rel, qual), kind in sorted(SANCTIONED_EDGE_COMMITTERS.items()):
            tree = by_file.get(rel)
            if tree is None:
                failures.append(f"  {rel} — allowlisted file is gone; prune the entry")
                continue
            func = _function_node(tree, qual)
            if func is None:
                failures.append(f"  {rel}:{qual} — allowlisted function does not exist")
                continue
            if kind == DELEGATES:
                continue
            if not _opens_admission(func):
                failures.append(f"  {rel}:{qual} — commits an edge but opens no admission scope")
        assert not failures, "SANCTIONED BUT UNADMITTED (edge write):\n" + "\n".join(failures)

    @pytest.mark.unit
    def test_add_edge_itself_requires_a_receipt(self, files: tuple[str, ...]) -> None:
        """The enforcement point, checked at its definition.

        Scan B of the write-path invariant, for the graph: whatever the doors
        do, the committer must refuse an unadmitted write on its own.
        """
        path = os.path.join(SRC_ROOT, "knowledge_graph.py")
        func = _function_node(_parse(path), "KnowledgeGraph.add_edge")
        assert func is not None, "KnowledgeGraph.add_edge is gone; this whole file is measuring nothing"
        assert _calls_require_admission(func), (
            "UNENFORCED EDGE SURFACE — KnowledgeGraph.add_edge accepts a write with no open\n"
            "admission. It must begin with require_admission(edge_id(...)) so a caller that\n"
            "forgot to open a scope raises UngatedWriteError instead of writing."
        )
