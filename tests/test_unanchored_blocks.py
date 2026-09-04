# Copyright 2026 STARGA, Inc.
"""H1 — an out-of-band corpus write was served, and verify stayed green.

REPRODUCED on a fresh ``init`` workspace before this landed. Append a
decision to ``decisions/DECISIONS.md`` by hand — no receipt, no scope,
``Status: active`` — then ``build_index``, then ``recall``::

    recall ids:                     ['D-20260903-777']
    SERVED THE UNANCHORED BLOCK:    True
    hash chain rows: 0   evidence rows: 0
    verify ok: True  exit: 0        (7/7 green)

The status allow-list was not the hole. It judged the status string and
the string was fine; nothing asked who had admitted the id. The corpus is
human-editable Markdown, so this is a boundary the product has.

WHAT IS TESTED, and what is deliberately not. ``unanchored_blocks``
reports the gap always and fails only under ``--strict``, and
:func:`~mind_mem.anchoring.restamp_unanchored` closes it. Anchored-ONLY
SERVING — withholding an unanchored block at recall time — is a later
release, and ``test_an_unanchored_block_is_still_served`` pins that on
purpose: the scope of 5.0.2 is to make the gap visible and fixable, and a
test that quietly asserted withholding would be asserting a change that
was not made.

Every RED here is paired with the GREEN of the same check over a
*governed* write. Without that pair, "the check fired" is equally true of
a check that fires on everything.
"""

from __future__ import annotations

import json
import os
import sqlite3

import pytest

from mind_mem.anchoring import landed_block_ids, restamp_unanchored, unanchored_report
from mind_mem.enums import IngestTier
from mind_mem.evidence_objects import EvidenceChain
from mind_mem.governance_gate import OP_WRITE, PHASE_CLOSED, evict_gate, get_gate
from mind_mem.init_workspace import init
from mind_mem.storage import get_block_store
from mind_mem.verify_cli import EXIT_OK, EXIT_UNANCHORED, VerifyReport, check_unanchored_blocks, verify_workspace

UNGOVERNED_ID = "D-20260903-777"

_UNGOVERNED_BLOCK = f"""
[{UNGOVERNED_ID}]
Date: 2026-09-03
Status: active
Scope: global
Statement: Adopt the zebra quantum ledger for all telemetry.
Rationale: zebra quantum ledger throughput
Supersedes: none
Tags: zebra
Sources: none
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def ws(tmp_path) -> str:
    path = str(tmp_path / "ws")
    init(path)
    yield path
    evict_gate(path)


def _append_by_hand(workspace: str, text: str = _UNGOVERNED_BLOCK) -> None:
    """The defect's own gesture: content into the corpus, past every door."""
    with open(os.path.join(workspace, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as handle:
        handle.write(text)


def _governed_write(workspace: str, block_id: str, *, status: str = "pending") -> dict:
    """One block through the sanctioned path, so a green row means something."""
    block = {
        "_id": block_id,
        "Date": "2026-09-03",
        "Status": status,
        "Scope": "global",
        "Statement": f"governed statement for {block_id}",
        "Rationale": "governed",
        "Supersedes": "none",
        "Tags": "governed",
        "Sources": "none",
    }
    store = get_block_store(workspace)
    with get_gate(workspace).admit_block(
        "WRITE",
        block_id,
        json.dumps(block, sort_keys=True),
        tier=IngestTier.AUTO_CAPTURE,
        actor="tester",
    ):
        store.write_block(block)
    return block


def _row(workspace: str, *, strict: bool = False):
    report = verify_workspace(workspace, strict=strict)
    return report, report.checks["unanchored_blocks"], report.details.get("unanchored_blocks", {})


# ---------------------------------------------------------------------------
# 1. The reproduction, and the control that makes it mean something
# ---------------------------------------------------------------------------


class TestTheDefect:
    def test_a_hand_written_block_is_reported_unanchored(self, ws: str) -> None:
        _append_by_hand(ws)
        report, _ok, details = _row(ws)
        assert details["unanchored"] == 1
        assert details["corpus_blocks"] == 1
        assert details["landed_blocks"] == 0
        assert any("never landed by a write scope" in message for message in report.messages)
        # The id itself is reported by `mm anchor`, never by the verifier —
        # see the next test for why.
        assert unanchored_report(ws).unanchored == (UNGOVERNED_ID,)

    def test_the_verify_surface_names_no_corpus_id(self, ws: str) -> None:
        """The row carries counts, never ids, and that is load-bearing.

        ``verify_workspace``'s rows are republished verbatim by the
        ``verify_chain`` / ``memory_verify`` / ``mind_mem_verify`` MCP
        tools, which are USER-scope and deliberately absent from the
        pinned ``ID_DISCLOSING`` set. An unanchored block is by definition
        one nobody admitted and its id is text its author typed, so an id
        list here would carry near-arbitrary attacker-chosen text into an
        agent's context through the audit surface.
        ``tests/test_read_surface_admission.py`` caught exactly this and
        is the gate that keeps it caught; this asserts it at the source.
        """
        _append_by_hand(ws, _UNGOVERNED_BLOCK.replace(UNGOVERNED_ID, "D-20260903-IGNORE-PRIOR-INSTRUCTIONS"))
        report = verify_workspace(ws)
        assert report.details["unanchored_blocks"]["unanchored"] == 1, "positive control: nothing was found to leak"
        blob = json.dumps(report.as_dict())
        assert "IGNORE-PRIOR-INSTRUCTIONS" not in blob
        # And the door that IS allowed to name it still does.
        assert "D-20260903-IGNORE-PRIOR-INSTRUCTIONS" in unanchored_report(ws).unanchored

    def test_strict_fails_the_workspace(self, ws: str) -> None:
        _governed_write(ws, "D-20260903-001")
        _append_by_hand(ws)
        report, ok, _details = _row(ws, strict=True)
        assert ok is False
        assert report.exit_code != EXIT_OK

    def test_the_check_owns_its_exit_code(self, ws: str) -> None:
        """Asserted on the check, not on the workspace, and that is the point.

        ``verify_workspace`` keeps the FIRST failure's code, so a workspace
        that is also missing its audit sidecar would report exit 1 under
        ``--strict`` and this assertion would be about ordering rather than
        about the finding. Run the one check over a fresh report instead.
        """
        _append_by_hand(ws)
        report = VerifyReport(workspace=ws, ok=True)
        check_unanchored_blocks(ws, report, strict=True)
        assert report.checks["unanchored_blocks"] is False
        assert report.exit_code == EXIT_UNANCHORED

    def test_MUTATION_the_same_check_passes_without_strict(self, ws: str) -> None:
        """The row is reported ALWAYS and fatal only under --strict.

        The flag is the whole difference; if this went red too, ``--strict``
        would not be what makes the finding fatal.
        """
        _append_by_hand(ws)
        report = VerifyReport(workspace=ws, ok=True)
        check_unanchored_blocks(ws, report, strict=False)
        assert report.checks["unanchored_blocks"] is True
        assert report.exit_code == EXIT_OK
        assert report.details["unanchored_blocks"]["unanchored"] == 1

    def test_POSITIVE_CONTROL_a_governed_write_is_not_reported(self, ws: str) -> None:
        """Without this, the RED above is equally true of a check that always fires.

        Same workspace shape, same verifier, one difference: the block
        went through ``admit_block``. If this goes red the join is broken,
        not the corpus.
        """
        _governed_write(ws, "D-20260903-001")
        report, ok, details = _row(ws, strict=True)
        assert details["unanchored"] == 0, report.messages
        assert details["corpus_blocks"] == 1
        assert details["landed_blocks"] == 1
        assert ok is True

    def test_POSITIVE_CONTROL_the_landed_side_is_not_empty(self, ws: str) -> None:
        """``corpus - landed`` is only evidence when ``landed`` was read.

        An empty landed set would make every corpus block unanchored and
        every assertion above pass for the wrong reason.
        """
        _governed_write(ws, "D-20260903-001")
        chain = EvidenceChain(store_path=os.path.join(ws, "memory", "evidence_chain.jsonl"))
        landed = landed_block_ids(chain)
        assert "D-20260903-001" in landed.ids
        assert landed.close_records >= 1
        assert landed.complete is True

    def test_an_unanchored_block_is_still_served(self, ws: str) -> None:
        """5.0.2 reports; it does not withhold. Pinned so the scope is explicit."""
        from mind_mem.recall import recall
        from mind_mem.sqlite_index import build_index

        _append_by_hand(ws)
        build_index(ws)
        served = [hit.get("id") or hit.get("_id") for hit in recall(ws, "zebra quantum ledger")]
        assert UNGOVERNED_ID in served


# ---------------------------------------------------------------------------
# 2. The repair: a pre-gate corpus is anchored, not condemned
# ---------------------------------------------------------------------------


class TestRestamp:
    def test_the_round_trip_goes_red_then_green(self, ws: str) -> None:
        _append_by_hand(ws)
        assert unanchored_report(ws).unanchored == (UNGOVERNED_ID,)

        result = restamp_unanchored(ws, actor="operator")
        assert result.anchored == (UNGOVERNED_ID,)
        assert result.skipped == ()

        report, ok, details = _row(ws, strict=True)
        assert ok is True, report.messages
        assert details["unanchored"] == 0

    def test_anchoring_preserves_the_status_it_found(self, ws: str) -> None:
        """RESTAMP is a *carrying* tier: it mints nothing and may not escalate.

        The block was ``active`` before the pass. A tier that stamped its
        own status would demote it (and silently un-serve real content);
        one that could mint would be a second ``admit_proposal``.
        """
        _append_by_hand(ws)
        restamp_unanchored(ws)
        block = get_block_store(ws).get_by_id(UNGOVERNED_ID)
        assert block is not None
        assert str(block.get("Status", "")).strip().lower() == "active"

    def test_anchoring_preserves_every_field_and_only_reorders_the_file(self, ws: str) -> None:
        """The precise version of "content is rewritten unchanged".

        Field-for-field the block survives; the FILE does not stay
        byte-identical, because the block goes back through the store's
        serialiser and comes out in its canonical order with the trailing
        ``---`` every governed write emits. Both halves are asserted so
        neither can drift into the other: silently dropping a field would
        be data loss, and claiming byte-identity would be a false claim
        an operator's first diff would disprove.
        """
        from mind_mem.block_parser import parse_file

        path = os.path.join(ws, "decisions", "DECISIONS.md")
        _append_by_hand(ws)
        before = {b["_id"]: {k: v for k, v in b.items() if k != "_line"} for b in parse_file(path)}
        raw_before = open(path, "rb").read()
        assert UNGOVERNED_ID in before, "positive control: the block was not parsed before the pass"

        restamp_unanchored(ws)

        after = {b["_id"]: {k: v for k, v in b.items() if k != "_line"} for b in parse_file(path)}
        assert after[UNGOVERNED_ID] == before[UNGOVERNED_ID]
        assert open(path, "rb").read() != raw_before, "the store did not rewrite the block at all"

    def test_anchoring_writes_a_chain_row_and_a_close_record(self, ws: str) -> None:
        _append_by_hand(ws)
        before = _chain_rows(ws)
        restamp_unanchored(ws)
        assert _chain_rows(ws) > before

        chain = EvidenceChain(store_path=os.path.join(ws, "memory", "evidence_chain.jsonl"))
        assert UNGOVERNED_ID in landed_block_ids(chain).ids

    def test_an_empty_candidate_set_opens_no_scope(self, ws: str) -> None:
        """A receipt covering nothing authorises nothing; minting one is a lie.

        The delete side already refuses this. Proved by the ledger, not by
        the return value: a scope that opened would leave rows behind.
        """
        _governed_write(ws, "D-20260903-001")
        before = _chain_rows(ws)
        result = restamp_unanchored(ws)
        assert result.anchored == ()
        assert _chain_rows(ws) == before

    def test_dry_run_writes_nothing(self, ws: str) -> None:
        _append_by_hand(ws)
        planned = restamp_unanchored(ws, dry_run=True)
        assert planned.anchored == (UNGOVERNED_ID,)
        assert planned.dry_run is True
        assert not os.path.isfile(os.path.join(ws, "memory", "hash_chain_v2.db"))

    def test_limit_anchors_a_reviewable_slice(self, ws: str) -> None:
        _append_by_hand(ws)
        _append_by_hand(ws, _UNGOVERNED_BLOCK.replace(UNGOVERNED_ID, "D-20260903-778"))
        result = restamp_unanchored(ws, limit=1)
        assert len(result.anchored) == 1
        assert len(unanchored_report(ws).unanchored) == 1

    def test_an_id_the_store_cannot_resolve_is_skipped_not_invented(self, ws: str) -> None:
        result = restamp_unanchored(ws, ids=["D-19700101-000"])
        assert result.anchored == ()
        assert result.skipped == ("D-19700101-000",)


def _chain_rows(workspace: str) -> int:
    path = os.path.join(workspace, "memory", "hash_chain_v2.db")
    if not os.path.isfile(path):
        return 0
    conn = sqlite3.connect(path)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM hash_chain").fetchone()[0])
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 3. A search that did not happen must never read as a clean one
# ---------------------------------------------------------------------------


class TestTheSearchActuallyHappened:
    def test_a_postgres_corpus_is_not_reported_clean(self, ws: str) -> None:
        """The Markdown walk is blind on the other two backends. Say so.

        ``postgres`` keeps blocks in the database and ``encrypted`` keeps
        them in ciphertext, so ``corpus_block_ids`` finds nothing and "all
        anchored" would describe a search that could not have found
        anything — the vacuous pass this module exists to stop being.
        Reported absent, naming the backend, and fatal under ``--strict``.
        """
        config = os.path.join(ws, "mind-mem.json")
        payload = json.loads(open(config, encoding="utf-8").read())
        payload["block_store"] = {"backend": "postgres", "dsn": "postgresql://x/y", "schema": "mind_mem"}
        with open(config, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

        assert unanchored_report(ws).backend == "postgres"
        report = verify_workspace(ws)
        assert report.checks["unanchored_blocks"] is True
        assert "mind-mem.json:block_store.backend=postgres" in report.missing

        strict = verify_workspace(ws, strict=True)
        assert strict.checks["unanchored_blocks"] is False

    def test_POSITIVE_CONTROL_the_default_backend_is_walked(self, ws: str) -> None:
        """The pair: on Markdown the same workspace IS searched, not skipped."""
        _append_by_hand(ws)
        report = verify_workspace(ws)
        assert report.details["unanchored_blocks"]["backend"] == "markdown"
        assert report.details["unanchored_blocks"]["files_scanned"] > 0
        assert "mind-mem.json:block_store.backend=markdown" not in report.missing

    def test_no_corpus_file_reads_as_unsearched_not_as_anchored(self, tmp_path) -> None:
        bare = str(tmp_path / "bare")
        os.makedirs(bare)
        report = verify_workspace(bare)
        assert report.details["unanchored_blocks"]["files_scanned"] == 0
        assert "decisions/DECISIONS.md" in report.missing

    def test_an_unparseable_corpus_file_fails_hard(self, ws: str, monkeypatch) -> None:
        """Not lenient: the walk did not happen over content that may hold anything."""
        import mind_mem.anchoring as anchoring

        def _explode(path: str, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr("mind_mem.block_parser.parse_file", _explode)
        assert anchoring.corpus_block_ids(ws).unreadable, "positive control: the failure was not injected"

        report = verify_workspace(ws)
        assert report.checks["unanchored_blocks"] is False
        assert report.exit_code == EXIT_UNANCHORED

    def test_a_truncated_close_record_is_declared_an_upper_bound(self) -> None:
        """A capped id list destroys membership, so the answer says "at most".

        A close record beyond ``_MAX_LANDED_LISTED`` carries an exact
        ``landed_count`` and an exact ``landed_root`` and only the first
        256 ids; a Merkle root does not answer membership, so those blocks
        can be reported unanchored when they are not. Reporting that
        silently would be a false statement dressed as a count.

        Asserted over the reader rather than over a tampered ledger on
        disk: :class:`~mind_mem.evidence_objects.EvidenceChain` drops a row
        whose hash no longer matches, so hand-editing the JSONL removes the
        very close record this is about and the test would pass for the
        wrong reason — measured (``close_records=0`` after the edit).
        Producing a real >256-block batch is the only other route and costs
        a 256-write governed batch per run.
        """
        rows = [
            _FakeRow({"write_phase": PHASE_CLOSED, "operation": OP_WRITE, "landed": ["D-1", "D-2"]}),
            _FakeRow({"write_phase": PHASE_CLOSED, "operation": OP_WRITE, "landed": ["D-3"], "landed_truncated": True}),
        ]
        landed = landed_block_ids(_FakeChain(rows))
        assert landed.close_records == 2
        assert landed.truncated_scopes == 1
        assert landed.complete is False
        assert landed.ids == frozenset({"D-1", "D-2", "D-3"})

    def test_MUTATION_an_untruncated_batch_claims_completeness(self) -> None:
        """The same reader over the same rows minus the truncation marker.

        Without this pair, ``complete is False`` above is equally true of a
        reader hard-wired to say so.
        """
        rows = [_FakeRow({"write_phase": PHASE_CLOSED, "operation": OP_WRITE, "landed": ["D-3"]})]
        landed = landed_block_ids(_FakeChain(rows))
        assert landed.truncated_scopes == 0
        assert landed.complete is True


class _FakeRow:
    """The one attribute :func:`landed_block_ids` reads off an evidence row."""

    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata


class _FakeChain:
    """The two-method surface :func:`landed_block_ids` asks a chain for."""

    def __init__(self, rows: list) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def get_latest(self, n: int) -> list:
        return self._rows[-n:]
