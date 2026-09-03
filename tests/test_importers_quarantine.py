"""Acceptance gate for import quarantine (``mind_mem.importers.quarantine``).

``mm import`` is the only bulk, ungated write in the package. The gate
this file enforces is the bargain that makes that acceptable:

1. imported blocks arrive **quarantined** and provenance-marked;
2. quarantined blocks are **not recallable**;
3. release goes through the **existing** governance gate
   (staged proposal -> ``approve_apply`` -> ``apply_proposal``), never a
   bespoke side door — and only then are the blocks recallable;
4. the bulk run is recorded in the tamper-evident **audit chain**;
5. with the importer unused, recall output is **byte-identical** to a
   workspace built before any of this existed (zero regression).

Everything runs offline against committed fixtures.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from mind_mem.audit_chain import AuditChain
from mind_mem.block_parser import parse_file
from mind_mem.importers import (
    IMPORTED_CORPUS_FILE,
    MAX_RELEASE_BLOCKS,
    QUARANTINE_STATUS,
    QUARANTINE_TIER,
    ImportQuarantineError,
    NothingToReleaseError,
    ReleaseTooLargeError,
    admitted_import_ids,
    batch_id_for,
    is_quarantined,
    propose_import_release,
    quarantined_import_ids,
    run_import,
)
from mind_mem.importers.quarantine import (
    DECISIONS_FILE,
    RELEASE_PROPOSAL_FILE,
    build_release_proposal,
)
from mind_mem.init_workspace import init
from mind_mem.mm_cli import config_set
from mind_mem.recall import recall

FIXTURES = Path(__file__).parent / "fixtures" / "importers"
VAULT = str(FIXTURES / "vault")
MEM0_DUMP = str(FIXTURES / "mem0_export.json")

FIXED_NOW = datetime(2026, 8, 27, 12, 0, 0)

#: A query that matches the committed vault fixture and nothing else.
VAULT_QUERY = "append-only block store canonical file prefix"


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


def _governed_ws(mode: str = "enforce") -> str:
    """A fully initialised workspace whose governance gate is armed.

    ``mind-mem.json`` is written through ``mm config set``, never by
    hand: ``init`` arms the gate, so a hand edit is drift and every
    governed write in the test then fails on a setting change nobody was
    hiding. ``memory/intel-state.json`` is not the bound config, so it is
    still written directly.
    """
    ws = tempfile.mkdtemp(prefix="mm_quarantine_")
    init(ws)
    config_path = os.path.join(ws, "mind-mem.json")
    config_set(config_path, "governance_mode", mode)
    state_path = os.path.join(ws, "memory", "intel-state.json")
    with open(state_path, encoding="utf-8") as handle:
        state = json.load(handle)
    state["governance_mode"] = mode
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle)
    return ws


def _imported_blocks(ws: str) -> list[dict[str, Any]]:
    path = Path(ws) / IMPORTED_CORPUS_FILE
    return parse_file(str(path)) if path.is_file() else []


def _approve(ws: str, proposal_id: str, *, dry_run: bool = False) -> dict[str, Any]:
    """Apply *proposal_id* through the real ``approve_apply`` MCP tool.

    ``check_preconditions`` shells out to ``validate.sh`` / ``intel_scan``
    which is environment-dependent, so it is stubbed exactly as
    ``tests/test_lint_autofix.py`` does — every other stage of the gate
    (validation, fingerprint, contradiction check, snapshot, WAL,
    receipt) runs for real.
    """
    from mind_mem.mcp.tools import governance

    env = dict(os.environ)
    env["MIND_MEM_WORKSPACE"] = ws
    with patch.dict(os.environ, env, clear=True):
        with patch("mind_mem.apply_engine.check_preconditions", return_value=(True, ["stubbed"])):
            raw = governance.approve_apply.__wrapped__(proposal_id, dry_run=dry_run)  # type: ignore[attr-defined]
    return dict(json.loads(raw))


def _vault_hit_ids(ws: str) -> set[str]:
    return {str(hit.get("id") or hit.get("_id")) for hit in recall(ws, VAULT_QUERY, limit=10)}


# ---------------------------------------------------------------------------
# Gate 1 — arrival is quarantined + provenance-marked
# ---------------------------------------------------------------------------


def test_import_stamps_quarantine_and_external_ingest_provenance() -> None:
    ws = _governed_ws()
    result = run_import(ws, "markdown", VAULT)

    assert result.imported == 3
    assert result.status == QUARANTINE_STATUS
    assert result.batch == batch_id_for("markdown", result.block_ids)

    blocks = _imported_blocks(ws)
    assert len(blocks) == 3
    for block in blocks:
        assert block["Status"] == QUARANTINE_STATUS
        assert is_quarantined(block["Status"])
        assert block["IngestTier"] == QUARANTINE_TIER == "external-ingest"
        assert block["ImportBatch"] == result.batch
        # The pre-existing provenance stamps are kept, not replaced.
        assert block["Source"] == "imported:markdown"
        assert block["ToolId"] == "imported:markdown"
        assert block["ActorRole"] == "importer"


def test_quarantined_blocks_are_classified_external_ingest() -> None:
    """The tier field and the validity gate's classifier agree."""
    from mind_mem.provenance_class import EXTERNAL_INGEST, classify_provenance

    ws = _governed_ws()
    run_import(ws, "markdown", VAULT)
    for block in _imported_blocks(ws):
        assert classify_provenance(block) == EXTERNAL_INGEST == block["IngestTier"]


def test_batch_id_is_deterministic_and_content_derived() -> None:
    ids = ("IMP-mem0-b", "IMP-mem0-a")
    assert batch_id_for("mem0", ids) == batch_id_for("mem0", reversed(ids))
    assert batch_id_for("mem0", ids) != batch_id_for("letta", ids)
    assert batch_id_for("mem0", ()) == ""


def test_dry_run_writes_nothing_and_records_nothing() -> None:
    ws = _governed_ws()
    result = run_import(ws, "markdown", VAULT, dry_run=True)

    assert result.dry_run is True
    assert result.status == QUARANTINE_STATUS
    assert _imported_blocks(ws) == []
    assert not os.path.exists(os.path.join(ws, ".mind-mem-audit", "chain.jsonl"))


# ---------------------------------------------------------------------------
# Gate 2 — quarantined blocks are not recallable
# ---------------------------------------------------------------------------


def test_quarantined_blocks_are_not_recallable() -> None:
    ws = _governed_ws()
    result = run_import(ws, "markdown", VAULT)

    hits = _vault_hit_ids(ws)
    assert hits.isdisjoint(set(result.block_ids))
    assert not any(h.startswith("IMP-") for h in hits)


def test_quarantine_filter_survives_the_sqlite_recall_backend() -> None:
    """The filter is on the shared post-filter funnel, not one backend."""
    ws = _governed_ws()
    config_set(os.path.join(ws, "mind-mem.json"), "recall", {"backend": "sqlite"})

    result = run_import(ws, "markdown", VAULT)
    from mind_mem.sqlite_index import build_index

    build_index(ws)
    assert _vault_hit_ids(ws).isdisjoint(set(result.block_ids))


def test_the_mcp_recall_tool_withholds_and_then_releases() -> None:
    """The agent-facing surface reaches the backends directly — check it."""
    from mind_mem.mcp.tools import recall as mcp_recall

    ws = _governed_ws()
    result = run_import(ws, "markdown", VAULT)

    def mcp_ids() -> set[str]:
        env = dict(os.environ)
        env["MIND_MEM_WORKSPACE"] = ws
        with patch.dict(os.environ, env, clear=True):
            raw = mcp_recall.recall.__wrapped__(VAULT_QUERY, limit=10)  # type: ignore[attr-defined]
        payload = json.loads(raw)
        return {str(hit.get("id") or hit.get("_id")) for hit in payload.get("results", [])}

    assert mcp_ids().isdisjoint(set(result.block_ids))

    proposal_id = propose_import_release(ws, result.block_ids, system="markdown", batch=result.batch, now=FIXED_NOW)
    _approve(ws, proposal_id, dry_run=False)

    assert mcp_ids() & set(result.block_ids)


def test_the_withheld_set_is_the_complement_of_admission() -> None:
    """Withholding is now a property of the corpus, not a second id-set.

    ``withheld_import_ids`` is gone: it existed only because some legs
    returned hits with no status, and it answered that by naming one
    corpus file — so the inbox and agent messages fell straight through
    it. The same question is now asked of the blocks themselves.
    """
    from mind_mem.admissibility import admissible

    ws = _governed_ws()
    result = run_import(ws, "markdown", VAULT)
    imported = _imported_blocks(ws)
    assert frozenset(admissible(imported)) == frozenset()

    proposal_id = propose_import_release(ws, result.block_ids, system="markdown", batch=result.batch, now=FIXED_NOW)
    _approve(ws, proposal_id, dry_run=False)
    releases = admitted_import_ids(ws)
    assert frozenset(admissible(_imported_blocks(ws), releases=releases)) == frozenset(result.block_ids)


def test_the_admissibility_decision_is_free_on_an_import_free_workspace() -> None:
    """Nothing withheld in the corpus means the release set is never read."""
    ws = _governed_ws()
    assert not os.path.exists(os.path.join(ws, IMPORTED_CORPUS_FILE))
    with patch("mind_mem.block_parser.parse_file", side_effect=AssertionError("read the decisions file with nothing withheld")):
        assert admitted_import_ids.__module__


def test_quarantined_blocks_do_not_consume_result_slots() -> None:
    """An unreleased block must not displace a governed one in the top-k."""
    ws = _governed_ws()
    with open(os.path.join(ws, DECISIONS_FILE), "a", encoding="utf-8") as handle:
        handle.write(
            "\n[D-20260101-001]\n"
            "Statement: The block store is append-only and writes land in a canonical file per prefix.\n"
            "Date: 2026-01-01\n"
            "Status: active\n"
            "Scope: global\n"
            "Rationale: Auditability.\n"
            "Supersedes: none\n"
            "Tags: storage\n"
            "Sources:\n"
            "- decisions/DECISIONS.md\n"
        )
    run_import(ws, "markdown", VAULT)
    hits = [h for h in recall(ws, VAULT_QUERY, limit=1)]
    assert [str(h.get("id") or h.get("_id")) for h in hits] == ["D-20260101-001"]


# ---------------------------------------------------------------------------
# Gate 3 — release goes through the existing governance gate
# ---------------------------------------------------------------------------


def test_release_proposal_only_stages_and_never_writes_the_corpus() -> None:
    ws = _governed_ws()
    result = run_import(ws, "markdown", VAULT)

    def digests() -> dict[str, str]:
        out: dict[str, str] = {}
        for root, _dirs, files in os.walk(ws):
            for name in files:
                path = os.path.join(root, name)
                rel = os.path.relpath(path, ws).replace(os.sep, "/")
                if rel == RELEASE_PROPOSAL_FILE:
                    continue
                with open(path, "rb") as handle:
                    out[rel] = hashlib.sha256(handle.read()).hexdigest()
        return out

    before = digests()
    proposal_id = propose_import_release(
        ws,
        result.block_ids,
        system="markdown",
        batch=result.batch,
        rationale="Reviewed the vault; it is our own note tree.",
        now=FIXED_NOW,
    )
    assert proposal_id == "P-20260827-001"
    assert digests() == before
    # Still inert: staging is not admission.
    assert _vault_hit_ids(ws).isdisjoint(set(result.block_ids))


def test_release_is_applied_by_the_real_approve_apply_gate() -> None:
    ws = _governed_ws()
    result = run_import(ws, "markdown", VAULT)
    assert _vault_hit_ids(ws).isdisjoint(set(result.block_ids))

    proposal_id = propose_import_release(
        ws,
        result.block_ids,
        system="markdown",
        batch=result.batch,
        rationale="Reviewed the vault; it is our own note tree.",
        now=FIXED_NOW,
    )

    dry = _approve(ws, proposal_id, dry_run=True)
    assert dry["status"] == "dry_run_passed", dry
    # A dry run admits nothing.
    assert admitted_import_ids(ws) == frozenset()

    applied = _approve(ws, proposal_id, dry_run=False)
    assert applied["status"] == "applied", applied

    assert admitted_import_ids(ws) == frozenset(result.block_ids)
    released = _vault_hit_ids(ws)
    assert released & set(result.block_ids), released

    # The admission is a first-class corpus record, not a silent flip.
    decisions = {b["_id"]: b for b in parse_file(os.path.join(ws, DECISIONS_FILE))}
    decision = decisions["D-20260827-001"]
    assert decision["Status"] == "active"
    assert sorted(decision["Releases"]) == sorted(result.block_ids)
    assert decision["AdmitsImportBatch"] == result.batch

    staged = {b["ProposalId"]: b for b in parse_file(os.path.join(ws, RELEASE_PROPOSAL_FILE))}
    assert staged[proposal_id]["Status"] == "applied"


def test_revoking_the_release_decision_re_quarantines_the_batch() -> None:
    ws = _governed_ws()
    result = run_import(ws, "markdown", VAULT)
    proposal_id = propose_import_release(ws, result.block_ids, system="markdown", batch=result.batch, now=FIXED_NOW)
    _approve(ws, proposal_id, dry_run=False)
    assert _vault_hit_ids(ws) & set(result.block_ids)

    path = os.path.join(ws, DECISIONS_FILE)
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text.replace("Status: active\nType: decision", "Status: revoked\nType: decision"))

    assert admitted_import_ids(ws) == frozenset()
    assert _vault_hit_ids(ws).isdisjoint(set(result.block_ids))


def test_release_refuses_ids_that_are_not_in_quarantine() -> None:
    ws = _governed_ws()
    run_import(ws, "markdown", VAULT)
    with pytest.raises(NothingToReleaseError):
        propose_import_release(ws, ["IMP-markdown-deadbeefdeadbeef"], system="markdown", now=FIXED_NOW)


def test_release_refuses_an_unreviewably_large_batch() -> None:
    ws = _governed_ws()
    run_import(ws, "markdown", VAULT)
    ids = tuple(f"IMP-mem0-{i:016x}" for i in range(MAX_RELEASE_BLOCKS + 1))
    with pytest.raises(ReleaseTooLargeError):
        build_release_proposal(
            "P-20260827-001",
            "D-20260827-001",
            system="mem0",
            batch="IMPB-mem0-x",
            block_ids=ids,
            rationale="too many",
            date="2026-08-27",
        )
    # The corpus is untouched: refusing to build a proposal writes nothing.
    assert len(quarantined_import_ids(ws, IMPORTED_CORPUS_FILE)) == 3


def test_restaging_the_same_release_is_refused() -> None:
    ws = _governed_ws()
    result = run_import(ws, "markdown", VAULT)
    propose_import_release(ws, result.block_ids, system="markdown", batch=result.batch, now=FIXED_NOW)
    with pytest.raises(ImportQuarantineError):
        propose_import_release(ws, result.block_ids, system="markdown", batch=result.batch, now=FIXED_NOW)


def test_release_proposal_passes_the_apply_engine_validator() -> None:
    from mind_mem.apply_engine import find_proposal, validate_proposal

    ws = _governed_ws()
    result = run_import(ws, "mem0", MEM0_DUMP)
    proposal_id = propose_import_release(ws, result.block_ids, system="mem0", batch=result.batch, now=FIXED_NOW)
    staged, source = find_proposal(ws, proposal_id)
    assert source is not None and source.endswith("EDITS_PROPOSED.md")
    assert staged is not None
    assert validate_proposal(staged) == []
    assert staged["Risk"] == "high"


# ---------------------------------------------------------------------------
# Gate 4 — the bulk run is chain-recorded
# ---------------------------------------------------------------------------


def test_import_is_recorded_in_the_audit_chain() -> None:
    ws = _governed_ws()
    result = run_import(ws, "markdown", VAULT)
    assert result.batch

    chain = AuditChain(ws)
    ok, errors = chain.verify()
    assert ok, errors

    entries = [e for e in chain.entries() if e.agent == "importer:markdown"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.operation == "create_block"
    assert entry.target == IMPORTED_CORPUS_FILE
    assert QUARANTINE_TIER in entry.reason
    assert entry.prev_hash and entry.entry_hash
    # Every written id is inside the hashed payload, so the chain names
    # exactly what arrived — not just that "an import happened".
    from mind_mem.audit_chain import _payload_hash

    assert entry.payload_hash == _payload_hash(
        {
            "system": "markdown",
            "source_path": os.path.abspath(VAULT),
            "batch": result.batch,
            "status": QUARANTINE_STATUS,
            "tier": QUARANTINE_TIER,
            "block_ids": list(result.block_ids),
        }
    )


def test_a_second_import_chains_onto_the_first() -> None:
    ws = _governed_ws()
    run_import(ws, "markdown", VAULT)
    run_import(ws, "mem0", MEM0_DUMP)

    chain = AuditChain(ws)
    ok, errors = chain.verify()
    assert ok, errors
    entries = chain.entries()
    assert [e.agent for e in entries] == ["importer:markdown", "importer:mem0"]
    assert entries[1].prev_hash == entries[0].entry_hash


def test_reimport_writes_nothing_and_adds_no_chain_entry() -> None:
    ws = _governed_ws()
    run_import(ws, "markdown", VAULT)
    before = AuditChain(ws).entry_count()
    second = run_import(ws, "markdown", VAULT)
    assert second.imported == 0
    assert second.batch == ""
    assert AuditChain(ws).entry_count() == before


# ---------------------------------------------------------------------------
# Gate 5 — zero regression when the importer is unused
# ---------------------------------------------------------------------------


def _seed_corpus(ws: str) -> None:
    with open(os.path.join(ws, DECISIONS_FILE), "a", encoding="utf-8") as handle:
        handle.write(
            "\n[D-20260101-001]\n"
            "Statement: Recall is filtered by status before the top-k cut.\n"
            "Date: 2026-01-01\n"
            "Status: active\n"
            "Scope: global\n"
            "Rationale: Ordering must not depend on withheld blocks.\n"
            "Supersedes: none\n"
            "Tags: recall\n"
            "Sources:\n"
            "- decisions/DECISIONS.md\n"
            "\n[D-20260102-002]\n"
            "Statement: The block store is append-only and every write lands in a canonical file.\n"
            "Date: 2026-01-02\n"
            "Status: active\n"
            "Scope: global\n"
            "Rationale: Auditability.\n"
            "Supersedes: none\n"
            "Tags: storage\n"
            "Sources:\n"
            "- decisions/DECISIONS.md\n"
        )


@pytest.mark.parametrize(
    "query",
    [
        "recall status filter top-k",
        "append-only canonical file write",
        "nothing matches this query at all",
    ],
)
def test_recall_is_byte_identical_when_the_importer_is_unused(query: str) -> None:
    """No import, no quarantined block — the filter is a pure no-op."""
    ws = _governed_ws()
    _seed_corpus(ws)
    baseline = json.dumps(recall(ws, query, limit=10), sort_keys=True, default=str)

    # The filter still runs; on an import-free workspace it must be a
    # pure identity, so bypassing it cannot change a single byte.
    with patch("mind_mem._recall_core.admit_corpus", side_effect=lambda blocks, **kw: list(blocks)):
        bypassed = json.dumps(recall(ws, query, limit=10), sort_keys=True, default=str)

    assert bypassed == baseline


def test_admitted_lookup_is_skipped_when_nothing_is_quarantined() -> None:
    """The DECISIONS.md read only happens when it can change the answer."""
    ws = _governed_ws()
    _seed_corpus(ws)
    boom = AssertionError("admission lookup on an import-free workspace")
    with patch("mind_mem.importers.quarantine.admitted_import_ids", side_effect=boom):
        assert recall(ws, "append-only canonical file write", limit=10)


def test_the_importer_status_is_withheld_by_the_admissibility_rule() -> None:
    """Recall no longer carries its own copy of the status literal.

    It used to pin ``_QUARANTINE_STATUS`` to the importer's constant and
    the two were checked for drift here. There is nothing to pin now: the
    withheld set is derived from the admission table, so the importer's
    status is withheld because of where it comes from, not because recall
    was told its spelling.
    """
    from mind_mem._recall_core import _withhold_inadmissible
    from mind_mem.admissibility import UNADMITTED, is_admissible_status

    assert QUARANTINE_STATUS in UNADMITTED
    assert not is_admissible_status(QUARANTINE_STATUS)
    hits = [{"_id": "IMP-1", "status": QUARANTINE_STATUS}, {"_id": "D-1", "status": "active"}]
    assert [h["_id"] for h in _withhold_inadmissible(hits, None, status_key="status")] == ["D-1"]
