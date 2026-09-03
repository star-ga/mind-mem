# Copyright 2026 STARGA, Inc.
"""The fourth door that killed content without a record: compaction.

``tests/test_governed_delete_mcp_tool.py`` closed the third of the three
doors that reached ``delete_block``. This one was found while fixing it,
and it never reached a store at all: ``compaction.compact_signals``
rewrites ``intelligence/SIGNALS.md`` itself, so the store-side gate could
not see it — the same shape as the MCP tool's Markdown leg, on a file
that IS in ``CORPUS_FILES`` and therefore was being served by recall.

Measured by live probe on a workspace built by ``mind-mem-init``, before
the fix::

    returned: ['Removed signal SIG-20240101-001 (resolved)',
               'Removed signal SIG-20240101-002 (rejected)']
    resolved gone after: True     rejected gone after: True
    chain rows after: 0           deleted_blocks.jsonl after: 0

Two blocks died, the evidence chain gained **zero** rows and the recovery
journal gained no entry.

A compaction removes many signals in one pass, so it is admitted as a
BATCH: one ``admit_delete_batch`` scope, one authorisation record, one
removal record carrying a Merkle root over everything that actually went
— the same reasoning as ``POST /clear``, and the reason
:class:`TestOneDecisionOneRecord` counts rows rather than just looking
for them.

``archive_completed_blocks`` is here too, under a different rule. It
MOVES blocks between files of record; the probe measured that
``BlockStore.get_by_id`` still resolves an archived block afterwards, so
nothing died and a delete record would be a false one. It gets a
carrying-tier ``admit_batch`` instead — a move, recorded as a move. See
:class:`TestArchiveIsRecordedAsAMove`.

Every negative assertion carries a positive control — the signal is shown
present before the sweep, and a PENDING signal is shown surviving it —
because ``assert gone`` passes just as well against a fixture that never
seeded anything. :class:`TestMutationTwin` removes the scope and the
removal report and shows these tests going red: a gate never observed
failing is not a gate.
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pytest
from _ledger_rows import authorisation_rows

from mind_mem.admission import UngatedDeleteError
from mind_mem.compaction import SIGNAL_SWEEP_RATIONALE, archive_completed_blocks, compact_signals
from mind_mem.governance_gate import PHASE_ADMITTED, PHASE_REMOVED, GovernanceBypassError, evict_gate

CORPUS = ("decisions", "tasks", "entities", "intelligence", "memory")

#: A date comfortably outside every retention window used here.
OLD = "2024-01-01"

#: Today, so a "too recent to sweep" fixture stays too recent forever.
#:
#: A literal would rot: the sweeps compare against ``now - days``, so a
#: hard-coded near-future date silently crosses the cutoff once enough
#: wall-clock passes and the test starts failing on a tree nobody
#: touched. Derived from the same clock the code under test reads.
RECENT = datetime.now().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[str]:
    """The zero-config default: blocks of record on the Markdown corpus."""
    ws = tmp_path / "ws"
    for sub in CORPUS:
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(json.dumps({"block_store": {"backend": "markdown"}}), encoding="utf-8")
    (ws / "intelligence" / "SIGNALS.md").write_text("# Captured Signals\n\n", encoding="utf-8")
    (ws / "tasks" / "TASKS.md").write_text("# Tasks\n\n", encoding="utf-8")
    (ws / "decisions" / "DECISIONS.md").write_text("# Decisions\n\n", encoding="utf-8")
    try:
        yield str(ws)
    finally:
        evict_gate(str(ws))


def _seed_signal(ws: str, bid: str, status: str, date: str = OLD) -> str:
    path = os.path.join(ws, "intelligence", "SIGNALS.md")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"[{bid}]\nDate: {date}\nStatus: {status}\nExcerpt: signal {bid}\n\n---\n")
    return bid


def _seed_task(ws: str, bid: str, status: str = "done", date: str = OLD) -> str:
    with open(os.path.join(ws, "tasks", "TASKS.md"), "a", encoding="utf-8") as handle:
        handle.write(f"[{bid}]\nTitle: task {bid}\nDate: {date}\nStatus: {status}\n\n---\n")
    return bid


def _seed_decision(ws: str, bid: str, status: str = "superseded", date: str = OLD) -> str:
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as handle:
        handle.write(f"[{bid}]\nStatement: decision {bid}\nDate: {date}\nStatus: {status}\n\n---\n")
    return bid


def _text(ws: str, *parts: str) -> str:
    path = Path(ws, *parts)
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _present(ws: str, bid: str, *parts: str) -> bool:
    """The positive control: is the block actually in the file of record?"""
    return f"[{bid}]" in _text(ws, *(parts or ("intelligence", "SIGNALS.md")))


def _records(ws: str) -> list[dict]:
    path = os.path.join(ws, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _authorisations(ws: str) -> list[dict]:
    """Evidence rows that authorised something — one per governed scope.

    A scope leaves two rows, the authorisation and the record of what it
    actually did (``write_phase=closed`` on the write side,
    ``delete_phase=removed`` on the delete side). ``tests/_ledger_rows``
    holds that convention once; :func:`_phase` below still reads the
    second half directly, which is the point of keeping both.
    """
    return authorisation_rows(_records(ws))


def _meta(record: dict) -> dict:
    return record.get("metadata") or {}


def _phase(ws: str, phase: str) -> list[dict]:
    return [r for r in _records(ws) if _meta(r).get("delete_phase") == phase]


def _journal(ws: str) -> list[dict]:
    path = os.path.join(ws, "memory", "deleted_blocks.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# A — the door: a sweep that removes content records the removal
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_signal_sweep_records_the_deaths_it_caused(workspace: str) -> None:
    """The measured hole: this sweep used to leave the chain empty."""
    doomed = _seed_signal(workspace, "SIG-20240101-001", "resolved")
    also = _seed_signal(workspace, "SIG-20240101-002", "rejected")
    kept = _seed_signal(workspace, "SIG-20240101-003", "pending")
    assert _present(workspace, doomed), "fixture never seeded the signal; every assertion below would be vacuous"
    assert _present(workspace, also)
    assert _records(workspace) == [], "the chain must start empty or the counts below prove nothing"

    actions = compact_signals(workspace, days=30)

    assert len(actions) == 2, actions
    assert not _present(workspace, doomed)
    assert not _present(workspace, also)
    assert _present(workspace, kept), "a PENDING signal is never removed — the function's own contract"

    admitted = _phase(workspace, PHASE_ADMITTED)
    removed = _phase(workspace, PHASE_REMOVED)
    assert len(admitted) == 1, f"expected one authorisation record, got {len(admitted)}"
    assert len(removed) == 1, f"expected one removal record, got {len(removed)}"
    assert _meta(removed[0])["removed_count"] == 2
    assert _meta(removed[0])["merkle_root"]
    assert _meta(removed[0])["scope_outcome"] == "ok"
    assert _meta(removed[0])["admission_entry_id"], "the removal record must name the authorisation it closes"


@pytest.mark.unit
def test_the_sweep_writes_the_recovery_journal(workspace: str) -> None:
    """``memory/deleted_blocks.jsonl`` is what makes the removal recoverable."""
    doomed = _seed_signal(workspace, "SIG-20240101-001", "resolved")
    assert _present(workspace, doomed)
    assert _journal(workspace) == []

    compact_signals(workspace, days=30)

    entries = _journal(workspace)
    assert [e["block_id"] for e in entries] == [doomed]
    assert "Excerpt: signal SIG-20240101-001" in entries[0]["content"], "the journal must carry what was removed, not just its id"
    assert entries[0]["deleted_at"]


@pytest.mark.unit
def test_the_authorisation_names_the_door_the_policy_and_the_covered_set(workspace: str) -> None:
    """A record that cannot say why content died is most of the way to none."""
    _seed_signal(workspace, "SIG-20240101-001", "resolved")
    _seed_signal(workspace, "SIG-20240101-002", "pending")

    compact_signals(workspace, days=30)

    meta = _meta(_phase(workspace, PHASE_ADMITTED)[0])
    assert meta["rationale"] == f"{SIGNAL_SWEEP_RATIONALE}: resolved/rejected signals older than 30d"
    assert meta["door"] == "compaction.compact_signals"
    assert meta["retention_days"] == 30
    assert meta["covers_count"] == 1, "the receipt must cover the eligible signal only, never the pending one"


@pytest.mark.unit
def test_a_dry_run_destroys_nothing_and_admits_nothing(workspace: str) -> None:
    """``memory_health`` calls this on every render; it must mint no decision."""
    doomed = _seed_signal(workspace, "SIG-20240101-001", "resolved")
    assert _present(workspace, doomed)

    actions = compact_signals(workspace, days=30, dry_run=True)

    assert actions and all("[dry-run]" in a for a in actions)
    assert _present(workspace, doomed), "a dry run must not remove anything"
    assert _records(workspace) == [], "a dry run destroys nothing, so it must put no delete decision in the chain"


@pytest.mark.unit
def test_a_sweep_with_nothing_to_remove_mints_no_authorisation(workspace: str) -> None:
    """A receipt covering nothing authorises nothing."""
    kept = _seed_signal(workspace, "SIG-20240101-001", "pending")
    recent = _seed_signal(workspace, "SIG-20990101-001", "resolved", date=RECENT)
    assert _present(workspace, kept) and _present(workspace, recent)

    assert compact_signals(workspace, days=30) == []

    assert _present(workspace, kept) and _present(workspace, recent)
    assert _records(workspace) == []


@pytest.mark.unit
def test_a_refused_scope_removes_nothing(workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed: a refused authorisation is never reported as a compaction."""
    doomed = _seed_signal(workspace, "SIG-20240101-001", "resolved")
    assert _present(workspace, doomed)

    def _refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise GovernanceBypassError("spec binding drifted")

    monkeypatch.setattr("mind_mem.governance_gate.GovernanceGate.admit_delete_batch", _refuse)

    with pytest.raises(GovernanceBypassError):
        compact_signals(workspace, days=30)

    assert _present(workspace, doomed), "the gate refused, so the signal must still be there"
    assert _journal(workspace) == []


@pytest.mark.unit
def test_an_ungated_delete_of_the_same_signal_is_still_refused(workspace: str) -> None:
    """The store seam is live for exactly the block this sweep removes.

    Wiring a door must not open a second, unguarded route to the same
    content: with no scope open, the store still refuses.
    """
    from mind_mem.storage import get_block_store

    doomed = _seed_signal(workspace, "SIG-20240101-001", "resolved")
    store = get_block_store(workspace)
    assert store.get_by_id(doomed) is not None, "positive control: the store can read the block it is about to refuse to delete"

    with pytest.raises(UngatedDeleteError):
        store.delete_block(doomed)

    assert _present(workspace, doomed)
    assert _records(workspace) == []


# ---------------------------------------------------------------------------
# B — one decision, one pair of records
# ---------------------------------------------------------------------------


class TestOneDecisionOneRecord:
    """A sweep of N signals is one decision, not N unlinked ones."""

    @pytest.mark.unit
    def test_five_signals_leave_one_authorisation_and_one_removal(self, workspace: str) -> None:
        ids = [_seed_signal(workspace, f"SIG-2024010{n}-001", "resolved") for n in range(1, 6)]
        assert all(_present(workspace, bid) for bid in ids), "positive control: all five seeded"

        actions = compact_signals(workspace, days=30)

        assert len(actions) == 5
        assert not any(_present(workspace, bid) for bid in ids)
        assert len(_phase(workspace, PHASE_ADMITTED)) == 1
        assert len(_phase(workspace, PHASE_REMOVED)) == 1
        assert _meta(_phase(workspace, PHASE_REMOVED)[0])["removed_count"] == 5
        assert len(_journal(workspace)) == 5


# ---------------------------------------------------------------------------
# C — the archive: a move, recorded as a move
# ---------------------------------------------------------------------------


class TestArchiveIsRecordedAsAMove:
    """The decision this file argues: relocation earns a record, not a death.

    The measured facts behind it are asserted here rather than described:
    an archived block is still resolvable through the block store (so it
    did not die), and it is no longer in ``CORPUS_FILES`` (so it did
    leave the served surface, which is why silence was not an option).
    """

    @pytest.mark.unit
    def test_the_archive_run_leaves_one_record_naming_every_block_moved(self, workspace: str) -> None:
        task = _seed_task(workspace, "T-20240101-001")
        decision = _seed_decision(workspace, "D-20240101-001")
        assert _present(workspace, task, "tasks", "TASKS.md")
        assert _present(workspace, decision, "decisions", "DECISIONS.md")

        actions = archive_completed_blocks(workspace, days=30)

        assert len(actions) == 2, actions
        assert not _present(workspace, task, "tasks", "TASKS.md")
        assert _present(workspace, task, "tasks", "TASKS_ARCHIVE.md")
        records = _authorisations(workspace)
        assert len(records) == 1, f"one archive run is one decision; got {len(records)} records"
        meta = _meta(records[0])
        assert meta["action_verb"] == "MIGRATE"
        assert meta["ingest_tier"] == "restamp", "an archive mints no status; it carries the one the block already had"
        assert meta["door"] == "compaction.archive_completed_blocks"
        assert meta["blocks"] == 2
        assert sorted(meta["files"]) == ["decisions/DECISIONS.md", "tasks/TASKS.md"]

    @pytest.mark.unit
    def test_an_archived_block_is_not_recorded_as_removed(self, workspace: str) -> None:
        """A delete record for a block that is still readable would be false."""
        from mind_mem.storage import get_block_store

        task = _seed_task(workspace, "T-20240101-001")
        assert get_block_store(workspace).get_by_id(task) is not None

        archive_completed_blocks(workspace, days=30)

        assert _phase(workspace, PHASE_ADMITTED) == []
        assert _phase(workspace, PHASE_REMOVED) == []
        assert _journal(workspace) == [], "nothing died, so nothing belongs in the deletion journal"
        # The fact the decision rests on: the store still resolves it. A
        # fresh store instance, because the old one caches its file list.
        assert get_block_store(workspace).get_by_id(task) is not None, "an archived block must remain readable, or it was a death"

    @pytest.mark.unit
    def test_an_archived_block_leaves_the_recall_surface(self, workspace: str) -> None:
        """The other half: why silence was not an acceptable answer either."""
        from mind_mem._recall_constants import CORPUS_FILES

        assert not any(v.endswith("_ARCHIVE.md") for v in CORPUS_FILES.values()), (
            "an archive file is not in CORPUS_FILES, so the indexer and the corpus walk stop seeing the block"
        )

    @pytest.mark.unit
    def test_an_archive_dry_run_admits_nothing(self, workspace: str) -> None:
        task = _seed_task(workspace, "T-20240101-001")
        actions = archive_completed_blocks(workspace, days=30, dry_run=True)

        assert actions and all("[dry-run]" in a for a in actions)
        assert _present(workspace, task, "tasks", "TASKS.md")
        assert _records(workspace) == []

    @pytest.mark.unit
    def test_an_archive_with_nothing_to_move_admits_nothing(self, workspace: str) -> None:
        recent = _seed_task(workspace, "T-20990101-001", date=RECENT)
        assert _present(workspace, recent, "tasks", "TASKS.md")

        assert archive_completed_blocks(workspace, days=30) == []

        assert _present(workspace, recent, "tasks", "TASKS.md")
        assert _records(workspace) == []


# ---------------------------------------------------------------------------
# C2 — the two sweeps that open no scope, and why that stays true
# ---------------------------------------------------------------------------


class TestTheUnscopedSweepsTouchNoBlockOfRecord:
    """``cleanup_snapshots`` and ``cleanup_daily_logs`` open no scope.

    The module docstring argues they need none: neither touches a block
    of record. That argument rests on three facts held in OTHER modules,
    which is exactly the shape that rots — the prose would still read
    correctly long after the registry it describes had moved. Pinned
    here so a change to either registry fails the build and forces the
    argument to be re-made rather than silently inherited.
    """

    @pytest.mark.unit
    def test_dated_logs_are_outside_the_store_read_surface(self) -> None:
        """``cleanup_daily_logs`` removes ``memory/<date>.md``."""
        from mind_mem.corpus_registry import CORPUS_DIRS

        assert "decisions" in CORPUS_DIRS, "positive control: the registry really is the dir list being checked"
        assert "memory" not in CORPUS_DIRS, (
            "memory/ became a store corpus dir, so MarkdownBlockStore._discover_files now globs the dated "
            "log files cleanup_daily_logs deletes — that sweep is a death door until it opens a scope"
        )

    @pytest.mark.unit
    def test_dated_logs_are_outside_the_recall_surface(self) -> None:
        """Every block-enumeration surface iterates ``CORPUS_FILES``."""
        import re as _re

        from mind_mem._recall_constants import CORPUS_FILES

        assert len(CORPUS_FILES) >= 12, "positive control: the registry is populated, so the check below is not vacuous"
        dated = [v for v in CORPUS_FILES.values() if _re.search(r"\d{4}-\d{2}-\d{2}\.md$", v)]
        assert not dated, f"a date-named file joined CORPUS_FILES ({dated}); cleanup_daily_logs now deletes recallable blocks"

    @pytest.mark.unit
    def test_snapshot_dirs_are_excluded_from_the_corpus(self) -> None:
        """``cleanup_snapshots`` rmtree's ``intelligence/applied/<ts>/``."""
        from mind_mem.corpus_registry import SNAPSHOT_EXCLUDE_DIRS

        assert "intelligence/applied" in SNAPSHOT_EXCLUDE_DIRS, (
            "intelligence/applied left the snapshot exclusion list; cleanup_snapshots may now be destroying "
            "content the apply engine relies on"
        )


# ---------------------------------------------------------------------------
# D — mutation twins: a gate never observed failing is not a gate
# ---------------------------------------------------------------------------


class TestMutationTwin:
    """Restore the pre-fix shape and watch the protective tests go red."""

    @pytest.mark.unit
    def test_without_the_scope_the_sweep_leaves_no_record(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """The 5.0.2 behaviour reproduced: removed, and nothing recorded.

        The twin bypasses the scope exactly as the old code did — splice
        the file with no admission open — and asserts the measured
        defect. If this ever fails, the splice acquired a second, hidden
        gate and the tests above are measuring something other than the
        scope.
        """

        class _NoReceipt:
            entry_id = "no-scope"

            def record_removal(self, *_args: Any, **_kwargs: Any) -> None:
                return None

        def _unscoped(*_args: Any, **_kwargs: Any) -> Any:
            return nullcontext(_NoReceipt())

        doomed = _seed_signal(workspace, "SIG-20240101-001", "resolved")
        assert _present(workspace, doomed)

        monkeypatch.setattr("mind_mem.governance_gate.GovernanceGate.admit_delete_batch", _unscoped)
        actions = compact_signals(workspace, days=30)

        assert len(actions) == 1, "the twin must reproduce a working sweep, not a broken one"
        assert not _present(workspace, doomed)
        assert _records(workspace) == [], "the twin did not actually bypass the gate"

    @pytest.mark.unit
    def test_dropping_record_removal_loses_the_removal_row(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """The half-fix: a scope that authorises and never reports.

        Wrapping the sweep without reporting the removals would leave an
        authorisation for deaths the chain never recorded. This twin
        proves the ``record_removal`` loop is what produces the second
        row.
        """
        doomed = _seed_signal(workspace, "SIG-20240101-001", "resolved")
        assert _present(workspace, doomed)

        monkeypatch.setattr("mind_mem.admission.AdmissionReceipt.record_removal", lambda self, block_id, content: None)
        actions = compact_signals(workspace, days=30)

        assert len(actions) == 1
        assert not _present(workspace, doomed)
        assert len(_phase(workspace, PHASE_ADMITTED)) == 1
        assert _phase(workspace, PHASE_REMOVED) == [], "the twin did not actually drop the removal report"

    @pytest.mark.unit
    def test_without_the_scope_the_archive_leaves_no_record(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same twin for the move: the pre-fix archive recorded nothing."""

        class _NoReceipt:
            entry_id = "no-scope"

        def _unscoped(*_args: Any, **_kwargs: Any) -> Any:
            return nullcontext(_NoReceipt())

        task = _seed_task(workspace, "T-20240101-001")
        assert _present(workspace, task, "tasks", "TASKS.md")

        monkeypatch.setattr("mind_mem.governance_gate.GovernanceGate.admit_batch", _unscoped)
        actions = archive_completed_blocks(workspace, days=30)

        assert len(actions) == 1, "the twin must reproduce a working archive, not a broken one"
        assert not _present(workspace, task, "tasks", "TASKS.md")
        assert _present(workspace, task, "tasks", "TASKS_ARCHIVE.md")
        assert _records(workspace) == [], "the twin did not actually bypass the gate"
