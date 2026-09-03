"""The served ledger under a SECOND WRITER — the two shapes production has.

A ledger of record forks in two ways, and this file covers both, because they
are one defect wearing two hats: a chain that can be restarted is not a chain.

**In parallel.** A workspace served by an MCP server while somebody runs
``mm recall``, or by an HTTP transport with a CLI beside it, has two
interpreters appending to one ``served.jsonl``. That was measured, not
imagined: three processes appending 300 rows each produced 700 rows of an
expected 900, 383 duplicate ``seq`` values, a chain red at row 1 and one worker
killed by a torn line ``_next_link`` could not parse. The guard was a
module-level ``threading.Lock``, which serialises the threads of one
interpreter and nothing else.

**In sequence.** The second writer is an editor, and it forks the chain by
leaving. Measured on the same code: five runs served, chain green;
``rm .mind-mem-ledger/served.jsonl``, and ``verify_served_chain`` names the
break — *"the ledger is empty but the recorded head is 7b4af0b3… — the rows
were removed"*. Then ONE more recall started a fresh sequence at ``seq 0`` and
re-sealed the head over it, and the verifier went **green**. The same held for
a truncated tail and for a deleted seal. A verifier whose verdict the next
ordinary recall erases proves nothing, so ``_next_link`` now refuses to start a
sequence where a seal says rows existed, and refuses to extend a tail the seal
does not agree with. Those tests live under THE RESTART GUARD below, each
paired with the state that must still be allowed.

Three tests for the parallel half, and the third is what makes the first two
worth anything:

* the ONE-process positive control, so "no duplicates" is a fact about the
  lock rather than a fact about a run where nothing overlapped;
* the two-process gate, which is the property itself;
* the MUTATION TWIN — the same two-process run with the cross-process lock
  swapped back for a per-process ``threading.Lock`` — which must go RED. A
  gate nobody has watched fail is a gate nobody has tested.

Workers are real subprocesses rather than ``multiprocessing`` children: the
defect is about separate interpreters, ``spawn`` and ``fork`` differ in what a
child inherits, and a subprocess behaves the same on every platform this ships
to. They rendezvous on a barrier directory before the first append, so the
overlap the twin needs is arranged rather than hoped for.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from collections import Counter
from typing import Any

import pytest

import mind_mem
from mind_mem.recall_digests import query_hash, served_set_digest
from mind_mem.served_ledger import (
    GENESIS_ROW_HASH,
    HEAD_RELPATH,
    LEDGER_ERROR_KEY,
    PROOF_RECORDED,
    PROOF_UNPROVEN,
    SERVED_PROOF_KEY,
    SERVED_ROW_HASH_KEY,
    SERVED_SEQ_KEY,
    ServedLedgerCorruptedError,
    append_served_run,
    attach_served_run,
    ledger_path,
    read_served_runs,
    row_hash,
    verify_served_chain,
)

#: Appends per worker. Enough contention that a per-process lock cannot get
#: through by luck — the twin below is asserted to fail, so a count too small
#: to break it fails this file rather than silently weakening it.
APPENDS = 120

#: Locking modes the worker understands. ``locked`` is the shipped code;
#: ``threadlock`` reinstates the pre-fix guard and nothing else.
LOCKED = "locked"
THREADLOCK = "threadlock"

_WORKER = """
import json, os, sys, threading, time

ws, tag, count, mode, barrier, nproc = sys.argv[1:7]
count, nproc = int(count), int(nproc)

import mind_mem.served_ledger as sl
from mind_mem.recall_digests import query_hash, served_set_digest

if mode == "threadlock":
    # THE MUTATION: exactly the guard this file exists to convict — one lock
    # per interpreter, which two interpreters do not share.
    _mutant = threading.Lock()
    sl._append_lock = lambda workspace: _mutant

# Rendezvous: nobody appends until every worker is loaded and ready, so the
# critical sections actually overlap.
os.makedirs(barrier, exist_ok=True)
open(os.path.join(barrier, tag), "w").close()
deadline = time.monotonic() + 60
while len(os.listdir(barrier)) < nproc:
    if time.monotonic() > deadline:
        raise SystemExit("barrier timeout")
    time.sleep(0.002)

for i in range(count):
    ids = ["%s-%d" % (tag, i)]
    sl.append_served_run(
        ws,
        query_hash=query_hash(ids[0]),
        served_digest=served_set_digest(ids),
        ids=ids,
        pipeline_hash="b" * 64,
        index_anchor="c" * 64,
        scoring_instant="2026-09-01",
    )
"""


def _workspace(tmp_path: pathlib.Path, name: str) -> str:
    ws = tmp_path / name
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(json.dumps({"served_ledger": {"enabled": True}}), encoding="utf-8")
    return str(ws)


def _run(tmp_path: pathlib.Path, name: str, *, nproc: int, mode: str) -> tuple[str, list[int]]:
    """Run *nproc* appending workers over one workspace; return it and their exits."""
    ws = _workspace(tmp_path, name)
    script = tmp_path / f"worker_{name}.py"
    script.write_text(_WORKER, encoding="utf-8")
    barrier = str(tmp_path / f"barrier_{name}")
    env = {**os.environ, "PYTHONPATH": str(pathlib.Path(mind_mem.__file__).resolve().parent.parent)}
    procs = [
        subprocess.Popen(
            [sys.executable, str(script), ws, f"P{k}", str(APPENDS), mode, barrier, str(nproc)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for k in range(nproc)
    ]
    codes = [proc.wait(timeout=300) for proc in procs]
    for proc in procs:
        proc.stdout.close() if proc.stdout else None
        proc.stderr.close() if proc.stderr else None
    return ws, codes


def _measure(ws: str) -> dict[str, Any]:
    """Rows, duplicate ``seq`` count and the chain verdict — never raising.

    A fork can leave a torn line that :func:`read_served_runs` cannot parse.
    That is the defect showing itself, not a broken measurement, so it is
    reported as a value rather than propagated as an error.
    """
    try:
        rows = read_served_runs(ws)
    except Exception as exc:  # noqa: BLE001 — an unparseable ledger is a result
        with open(ledger_path(ws), encoding="utf-8") as handle:
            lines = sum(1 for line in handle if line.strip())
        return {"rows": lines, "duplicate_seq": -1, "verify_ok": False, "unreadable": str(exc)}
    counts = Counter(row.seq for row in rows)
    return {
        "rows": len(rows),
        "duplicate_seq": sum(n - 1 for n in counts.values() if n > 1),
        "verify_ok": verify_served_chain(ws).ok,
        "unreadable": "",
    }


def test_one_process_appends_a_clean_chain(tmp_path: pathlib.Path) -> None:
    """POSITIVE CONTROL. Without it the two-process gate proves nothing.

    If a single appender could not produce ``APPENDS`` rows with no duplicate
    ``seq`` and a green chain, "the concurrent run was clean" would be a
    statement about the harness, not about the lock.
    """
    ws, codes = _run(tmp_path, "control", nproc=1, mode=LOCKED)
    assert codes == [0], f"the single worker did not finish cleanly: {codes}"
    assert _measure(ws) == {"rows": APPENDS, "duplicate_seq": 0, "verify_ok": True, "unreadable": ""}


def test_two_processes_append_one_unforked_chain(tmp_path: pathlib.Path) -> None:
    """THE GATE. Two interpreters, one ledger, every row present exactly once."""
    ws, codes = _run(tmp_path, "concurrent", nproc=2, mode=LOCKED)
    assert codes == [0, 0], f"a worker died — the appends were not all attempted: {codes}"
    assert _measure(ws) == {"rows": 2 * APPENDS, "duplicate_seq": 0, "verify_ok": True, "unreadable": ""}


def test_the_pre_fix_thread_lock_forks_the_chain(tmp_path: pathlib.Path) -> None:
    """THE MUTATION TWIN. Put the per-process lock back and this must go red.

    The assertion is deliberately a disjunction of every way the fork shows
    itself — rows lost, a ``seq`` written twice, a torn line, a worker killed
    by one, or a chain that no longer verifies — because which of them lands
    is a scheduling accident. What is not an accident is that at least one
    does; a run where none did would mean the gate above can pass without the
    lock, and this test says so by name.
    """
    ws, codes = _run(tmp_path, "mutant", nproc=2, mode=THREADLOCK)
    measured = _measure(ws)
    forked = (
        measured["rows"] != 2 * APPENDS
        or measured["duplicate_seq"] != 0
        or not measured["verify_ok"]
        or bool(measured["unreadable"])
        or codes != [0, 0]
    )
    assert forked, f"the per-process lock produced a clean chain — the gate above is not testing the lock: {measured} {codes}"


def test_the_shipped_lock_is_the_cross_process_one() -> None:
    """The twin swaps a symbol; this pins what that symbol normally is.

    Without it the twin could be swapping out something that was already a
    ``threading.Lock``, and the disjunction above would be measuring nothing.
    """
    import threading

    from mind_mem.mind_filelock import FileLock
    from mind_mem.served_ledger import _append_lock

    lock = _append_lock(os.getcwd())
    assert isinstance(lock, FileLock), f"the append guard is {type(lock).__name__}, not a cross-process FileLock"
    assert not isinstance(lock, type(threading.Lock()))


@pytest.mark.parametrize("mode", [LOCKED, THREADLOCK])
def test_the_worker_actually_appends(tmp_path: pathlib.Path, mode: str) -> None:
    """The control on the harness itself, in BOTH modes.

    A worker that failed to import, mis-parsed its argv or never reached the
    ledger would leave an empty file — which would satisfy the twin's "rows !=
    240" disjunct for entirely the wrong reason. So each mode must be shown to
    write something before its verdict means anything.
    """
    ws, _ = _run(tmp_path, f"harness_{mode}", nproc=1, mode=mode)
    assert _measure(ws)["rows"] == APPENDS, f"the {mode} worker did not append"


# ---------------------------------------------------------------------------
# THE RESTART GUARD — the second writer that forks the chain by leaving.
#
# Every test below is a PAIR: a tampering that must stay convicted, and the
# legitimate state next to it that must still be allowed. A guard tested only
# on the states it rejects is indistinguishable from a guard that rejects
# everything, and that one wedges every workspace on its first recall.
# ---------------------------------------------------------------------------


def _serve(ws: str, tag: str) -> Any:
    """One ledger row, through the shipped append path."""
    ids = [tag]
    return append_served_run(
        ws,
        query_hash=query_hash(tag),
        served_digest=served_set_digest(ids),
        ids=ids,
        pipeline_hash="b" * 64,
        index_anchor="c" * 64,
        scoring_instant="2026-09-01",
    )


def _served_ledger(tmp_path: pathlib.Path, name: str, rows: int = 5) -> str:
    """A workspace with *rows* rows and a chain that verifies."""
    ws = _workspace(tmp_path, name)
    for i in range(rows):
        _serve(ws, f"blk-{i}")
    assert verify_served_chain(ws).ok, "positive control: the untampered ledger verifies"
    return ws


def _refusal(ws: str) -> str:
    """The message the next append refuses with, or ``''`` if it appended."""
    try:
        _serve(ws, "next")
    except ServedLedgerCorruptedError as exc:
        return str(exc)
    return ""


def _laundered(ws: str, tamper: Any) -> tuple[bool, str]:
    """Tamper, then serve once more. ``(chain green again?, refusal)``."""
    tamper(ws)
    assert not verify_served_chain(ws).ok, "positive control: the tampering must break the chain first"
    refusal = _refusal(ws)
    return verify_served_chain(ws).ok, refusal


def test_a_removed_ledger_is_not_restarted_at_genesis(tmp_path: pathlib.Path) -> None:
    """``rm served.jsonl`` + one recall used to produce a clean chain.

    The seal is still on disk naming five rows, so "the ledger holds no row"
    is a deletion, not a fresh workspace. Restarting the sequence here writes
    a genesis row and re-seals the head over the removal — the deletion
    becomes unobservable, which is the whole claim inverted.
    """
    ws = _served_ledger(tmp_path, "restart_removed")
    green_again, refusal = _laundered(ws, lambda w: os.remove(ledger_path(w)))
    assert not green_again, "one recall re-sealed the head over a removed ledger and the chain went green"
    assert "rows were removed" in refusal, refusal
    assert read_served_runs(ws) == (), "the refusal must write no row at all, least of all a genesis one"


def test_a_truncated_tail_is_not_resealed_by_the_next_append(tmp_path: pathlib.Path) -> None:
    """Six rows cut off the end; the seal still names the row that is gone."""
    ws = _served_ledger(tmp_path, "restart_truncated", rows=10)
    sealed = row_hash(read_served_runs(ws)[-1])

    def _cut(w: str) -> None:
        kept = open(ledger_path(w), encoding="utf-8").read().splitlines()[:4]
        with open(ledger_path(w), "w", encoding="utf-8") as handle:
            handle.write("\n".join(kept) + "\n")

    green_again, refusal = _laundered(ws, _cut)
    assert not green_again, "one recall re-sealed the head over six removed rows and the chain went green"
    assert "neither that row nor its predecessor" in refusal, refusal
    assert (tmp_path / "restart_truncated" / HEAD_RELPATH).read_text(encoding="utf-8").strip() == sealed, (
        "the refusal rewrote the seal — the evidence of what the tail used to be is gone"
    )


def test_a_removed_seal_is_not_rewritten_by_the_next_append(tmp_path: pathlib.Path) -> None:
    """Delete the seal and the last row is unsealed; an append would re-seal it.

    This is the deletion the chain links cannot see on their own, which is why
    the seal exists — and why an append that silently recreates it hands an
    editor the last row for free.
    """
    ws = _served_ledger(tmp_path, "restart_unsealed")
    green_again, refusal = _laundered(ws, lambda w: os.remove(os.path.join(w, HEAD_RELPATH)))
    assert not green_again, "one recall rewrote the missing seal and the chain went green"
    assert "the seal was removed" in refusal, refusal
    assert not os.path.exists(os.path.join(ws, HEAD_RELPATH)), "the refusal minted a seal it had no basis for"


def test_a_torn_last_line_refuses_by_name(tmp_path: pathlib.Path) -> None:
    """The fork's other signature: a half-written row nothing can parse.

    Pre-fix this escaped as a bare ``JSONDecodeError`` from inside the append
    — the state that killed a worker outright in the measured three-process
    run. It is now the ledger's own exception, so a caller narrowing to
    ``OSError`` around ledger work catches it and
    :func:`attach_served_run` can name it on the record.
    """
    ws = _served_ledger(tmp_path, "restart_torn")
    with open(ledger_path(ws), "a", encoding="utf-8") as handle:
        handle.write('{"seq": 5, "prev_row_h')
    refusal = _refusal(ws)
    assert "not a readable row" in refusal, refusal
    assert issubclass(ServedLedgerCorruptedError, OSError)


def test_the_crash_window_still_heals(tmp_path: pathlib.Path) -> None:
    """RECOVERY MUST BE REACHABLE. A kill between the row and the seal is ordinary.

    :func:`_write_row` appends and then replaces the seal, so a process killed
    between the two leaves the seal one row behind a tail that names it. If
    the guard refused that, one ``SIGKILL`` would end recording in that
    workspace forever — a one-way door held open by nothing but luck. It is
    admitted because the last row cryptographically names the sealed head as
    its predecessor, which no removal can arrange without leaving a ``seq``
    gap or a broken link that the verifier still convicts.
    """
    ws = _served_ledger(tmp_path, "restart_crash")
    rows = read_served_runs(ws)
    with open(os.path.join(ws, HEAD_RELPATH), "w", encoding="utf-8") as handle:
        handle.write(row_hash(rows[-2]) + "\n")  # the seal as it stood one append ago
    assert not verify_served_chain(ws).ok, "positive control: a lagging seal is a break until it is caught up"

    row = _serve(ws, "after-crash")
    assert row is not None and row.seq == len(rows), "the append did not continue the existing sequence"
    assert verify_served_chain(ws).ok, "the crash window did not heal — the workspace can never record again"


def test_a_fresh_workspace_still_starts_at_genesis(tmp_path: pathlib.Path) -> None:
    """THE OTHER HALF. Neither a row nor a seal: the one state that may start a sequence."""
    ws = _workspace(tmp_path, "restart_fresh")
    row = _serve(ws, "first")
    assert row is not None and row.seq == 0 and row.prev_row_hash == GENESIS_ROW_HASH
    assert verify_served_chain(ws).ok


def _unguarded_next_link(workspace: Any) -> tuple[int, str]:
    """THE MUTANT — the pre-fix ``_next_link``, verbatim in behaviour.

    Reads the tail, and treats "no rows" as "a new ledger" without ever asking
    the seal whether rows used to exist.
    """
    import json as _json

    from mind_mem.served_ledger import ServedRun

    try:
        with open(ledger_path(workspace), encoding="utf-8") as handle:
            lines = [line for line in handle.read().splitlines() if line.strip()]
    except OSError:
        return 0, GENESIS_ROW_HASH
    if not lines:
        return 0, GENESIS_ROW_HASH
    last = ServedRun.from_row(_json.loads(lines[-1]))
    return last.seq + 1, row_hash(last)


def test_the_pre_fix_link_launders_a_removed_ledger(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """THE MUTATION TWIN for the guard. Put the old link back and this must launder.

    Without it, ``test_a_removed_ledger_is_not_restarted_at_genesis`` could be
    passing because of something else entirely — a verifier that happens to
    stay red, a tampering that happens not to be repairable. This pins the
    causation: with the unguarded link, and ONLY the link, changed, the
    deletion is erased by the next recall and the chain reports clean.
    """
    import mind_mem.served_ledger as sl

    ws = _served_ledger(tmp_path, "restart_mutant")
    os.remove(ledger_path(ws))
    assert not verify_served_chain(ws).ok, "positive control: the deletion is convicted before the mutation"

    monkeypatch.setattr(sl, "_next_link", _unguarded_next_link)
    row = _serve(ws, "laundering")

    assert row is not None and row.seq == 0, "the mutant did not restart the sequence — it is not the pre-fix link"
    assert verify_served_chain(ws).ok, "the guard is not what keeps a removed ledger convicted; this gate proves nothing"


# ---------------------------------------------------------------------------
# THE RECORD OF A RUN THAT COULD NOT BE RECORDED
# ---------------------------------------------------------------------------


def _record(ws: str, ids: list[str]) -> dict[str, Any]:
    """The attestation fields :func:`attach_served_run` reads, and nothing else."""
    return {
        "query_hash": query_hash(ids[0]),
        "results_digest": served_set_digest(ids),
        "config_hash": "b" * 64,
        "index_anchor": "c" * 64,
        "scoring_instant": "2026-09-01",
    }


def test_a_run_whose_row_was_refused_is_published_as_unproven(tmp_path: pathlib.Path) -> None:
    """A proof that omits its own failure is worse than no proof.

    The answer is still served — see :func:`attach_served_run` for the ruling
    — but the record says so in a word a consumer can branch on, names the
    reason, and carries no row id it does not have.
    """
    ws = _served_ledger(tmp_path, "unproven")
    ids = ["blk-0"]

    proven = attach_served_run(_record(ws, ids), ws, ids=ids)
    assert proven[SERVED_PROOF_KEY] == PROOF_RECORDED, "positive control: a healthy ledger must prove a run"
    assert isinstance(proven[SERVED_SEQ_KEY], int) and proven[LEDGER_ERROR_KEY] is None

    os.remove(ledger_path(ws))  # the row for the next run cannot be written
    unproven = attach_served_run(_record(ws, ids), ws, ids=ids)

    assert unproven[SERVED_PROOF_KEY] == PROOF_UNPROVEN
    assert unproven[SERVED_SEQ_KEY] is None and unproven[SERVED_ROW_HASH_KEY] is None
    assert "ServedLedgerCorruptedError" in str(unproven[LEDGER_ERROR_KEY])
    assert "rows were removed" in str(unproven[LEDGER_ERROR_KEY])
    assert read_served_runs(ws) == (), "the refused run left a row behind after all"


def test_the_proof_status_never_disagrees_with_the_row(tmp_path: pathlib.Path) -> None:
    """``served_proof`` is derived, so an older reader and a newer one agree.

    The forward-compatibility claim in the source is that a reader who knows
    only ``served_seq`` loses nothing, because the status is a function of it
    and never a second opinion. That is only true if it holds on every path a
    record can take — recorded, refused, and opted out.
    """
    ws = _served_ledger(tmp_path, "derived", rows=1)
    ids = ["blk-0"]
    records = [attach_served_run(_record(ws, ids), ws, ids=ids)]

    os.remove(ledger_path(ws))
    records.append(attach_served_run(_record(ws, ids), ws, ids=ids))

    off = _workspace(tmp_path, "derived_off")
    (pathlib.Path(off) / "mind-mem.json").write_text(json.dumps({"served_ledger": {"enabled": False}}), encoding="utf-8")
    records.append(attach_served_run(_record(off, ids), off, ids=ids))

    # And the fourth path: a ledger that cannot be written at all. A plain file
    # where the directory must be fails for every user on every platform, which
    # a permission bit does not (CI containers run as root).
    unwritable = _workspace(tmp_path, "derived_unwritable")
    pathlib.Path(unwritable, ".mind-mem-ledger").write_text("not a directory\n", encoding="utf-8")
    records.append(attach_served_run(_record(unwritable, ids), unwritable, ids=ids))
    assert records[-1][LEDGER_ERROR_KEY], "an unwritable ledger published no reason"

    seen = {record[SERVED_PROOF_KEY] for record in records}
    assert seen == {PROOF_RECORDED, PROOF_UNPROVEN}, f"a path published a status outside the vocabulary: {seen}"
    for record in records:
        expected = PROOF_RECORDED if record[SERVED_SEQ_KEY] is not None else PROOF_UNPROVEN
        assert record[SERVED_PROOF_KEY] == expected, f"status disagrees with the row it describes: {record}"


def test_the_link_still_comes_from_the_row_not_from_the_seal(tmp_path: pathlib.Path) -> None:
    """The invariant the guard must not cost: the seal never chooses the link.

    The seal holds exactly the value the next row's ``prev_row_hash`` needs,
    which makes reading it a tempting shortcut and a wrong one — appending
    from it would let one edited file re-anchor everything written afterwards
    and still verify. Proving it needs a state where the two disagree AND the
    append is allowed to proceed, and since the guard now refuses every other
    disagreement, that state is the crash window: the seal one row behind the
    tail. If the link came from the seal the new row would name the
    second-to-last row; it must name the last one.

    ``tests/test_served_ledger.py::test_t14_the_link_is_derived_from_the_file_not_from_the_seal``
    proved the same invariant by writing an arbitrary wrong seal and watching
    the append repair it. That state is now a refusal — it is how a truncated
    tail is laundered — so the invariant is re-proved here on a legal one.
    """
    ws = _served_ledger(tmp_path, "link_source", rows=2)
    first, second = read_served_runs(ws)
    with open(os.path.join(ws, HEAD_RELPATH), "w", encoding="utf-8") as handle:
        handle.write(row_hash(first) + "\n")

    third = _serve(ws, "third")
    assert third is not None, "the append refused the crash window it is required to heal"
    assert third.seq == 2
    assert third.prev_row_hash == row_hash(second), "the link came from the seal, not from the last row"
    assert verify_served_chain(ws).ok
