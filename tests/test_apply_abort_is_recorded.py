# Copyright 2026 STARGA, Inc.
"""A rolled-back apply must not close its admission scope claiming success.

The product claim is *prove what it stored*. The governance gate writes
one close record on both exits of a write scope and reads any exit that
is not an exception — a plain ``return`` included — as ``outcome=ok``.
``apply_engine.apply_proposal`` returned exactly that way from inside a
still-open scope after undoing its own work, so a failed, rolled-back
apply and a landed one closed with the same word. Measured on a fresh
workspace with ``execute_op`` forced to return ``(False, "boom")``::

    APPLY(P-…)                              op=write
    ROLLBACK(restore:20260902-185230)       op=write
    CLOSE(restore:20260902-185230)  outcome=ok    landed=0
    CLOSE(P-…)                      outcome=ok    landed=0

against the control run where the op succeeds::

    APPLY(P-…)                              op=write
    CLOSE(P-…)                      outcome=ok    landed=1

The two are told apart only by ``landed`` and by a sibling ``RESTORE``
row an auditor has to think to join. The close record itself — the one
row that names how the scope ended — said ``ok`` over work that was
withdrawn.

Three gates live here, each with a positive control (a negative
assertion over a method that cannot see the positive case proves
nothing) and each with a mutation twin that restores the old behaviour
in-process and asserts the gate goes red:

A  :class:`TestFailedApplyClosesAsError` — the close record for a
   rolled-back apply reads ``outcome=error``.
B  :class:`TestOneModeSource` — ``governance_mode`` has one reader and
   one file. Editing the unattested ``memory/intel-state.json`` cannot
   unblock an apply the bound ``mind-mem.json`` forbids.
C  :class:`TestRollbackKeepsTheLedgers` — the rollback's orphan sweep
   never deletes a ledger of record. On a workspace's first apply the
   gate's own ledgers postdate the pre-apply inventory, so the sweep
   used to delete the record of the rollback it was performing.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

import pytest

from mind_mem import apply_engine
from mind_mem.apply_engine import ApplyAborted, apply_proposal, compute_fingerprint
from mind_mem.corpus_registry import LEDGER_FILES
from mind_mem.governance_gate import (
    DETECT_ONLY_MODE,
    OUTCOME_ERROR,
    OUTCOME_OK,
    PHASE_CLOSED,
    config_path_for,
    evict_gate,
    get_gate,
    read_governance_mode,
)
from mind_mem.init_workspace import init
from mind_mem.spec_binding import SpecBindingManager

PROPOSAL_ID = "P-20260902-001"
TARGET_ID = "D-20260902-001"
DECISION_FILE = "decisions/DECISIONS.md"


# ---------------------------------------------------------------------------
# Workspace construction
# ---------------------------------------------------------------------------


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _set_config_mode(ws: str, mode: str, *, rebind: bool = True) -> None:
    """Set ``governance_mode`` in the bound config and re-attest it.

    The rebind is not decoration: editing the attested file drifts the
    spec binding, and the gate then refuses the apply with a ``DRIFT``
    row rather than running it. That refusal is the accountability the
    unattested ``intel-state.json`` never had, and re-attesting is the
    operator action (``mm bind --rebind``) that answers it.
    """
    path = config_path_for(ws)
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    config["governance_mode"] = mode
    _write(path, json.dumps(config, indent=2))
    if rebind:
        SpecBindingManager(path).rebind(path)


def _set_state_mode(ws: str, mode: str) -> None:
    """Set ``governance_mode`` in ``memory/intel-state.json``.

    The engine does not read this key. The function exists so the tests
    can prove that, and it deliberately does not rebind anything —
    nothing attests this file, which is the whole point.
    """
    path = os.path.join(ws, "memory", "intel-state.json")
    with open(path, encoding="utf-8") as handle:
        state = json.load(handle)
    state["governance_mode"] = mode
    _write(path, json.dumps(state, indent=2))


def _stage_proposal(ws: str) -> None:
    """A valid staged proposal plus the decision block it targets."""
    ops = [{"op": "set_status", "file": DECISION_FILE, "target": TARGET_ID, "status": "superseded"}]
    fingerprint = compute_fingerprint({"Type": "edit", "TargetBlock": TARGET_ID, "Ops": ops})
    _write(
        os.path.join(ws, "intelligence/proposed/DECISIONS_PROPOSED.md"),
        f"\n[{PROPOSAL_ID}]\n"
        f"ProposalId: {PROPOSAL_ID}\n"
        f"Type: edit\n"
        f"TargetBlock: {TARGET_ID}\n"
        f"Risk: low\n"
        f"Status: staged\n"
        f"Evidence:\n- the fixture stages this proposal\n"
        f"Rollback: restore the snapshot\n"
        f"Fingerprint: {fingerprint}\n"
        f"Ops:\n"
        f"- op: set_status\n"
        f"  file: {DECISION_FILE}\n"
        f"  target: {TARGET_ID}\n"
        f"  status: superseded\n",
    )
    _write(
        os.path.join(ws, DECISION_FILE),
        f"\n[{TARGET_ID}]\nId: {TARGET_ID}\nStatus: active\nTitle: fixture decision\n",
    )


@pytest.fixture
def workspace(tmp_path: Any) -> Iterator[str]:
    """An initialised workspace in ``propose`` mode with one staged proposal."""
    ws = str(tmp_path / "ws")
    init(ws)
    _set_config_mode(ws, "propose")
    _stage_proposal(ws)
    try:
        yield ws
    finally:
        evict_gate(ws)


@pytest.fixture
def preconditions_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the apply reach its ops.

    ``check_preconditions`` runs the full workspace validator, which a
    three-file fixture cannot satisfy and which is not what any test here
    is about. Patched to a fixed pass so both the pre-check and the
    post-check are out of the way; every assertion below is about the
    evidence chain, not about the validator.
    """
    monkeypatch.setattr(
        apply_engine,
        "check_preconditions",
        lambda ws: (True, ["validate: PASS (patched by the fixture)"]),
    )


# ---------------------------------------------------------------------------
# Reading the chain
# ---------------------------------------------------------------------------


def _rows(ws: str) -> list[Any]:
    chain = get_gate(ws).evidence
    return list(chain.get_latest(n=len(chain)))


def _close_record(ws: str, subject_id: str) -> dict:
    """The close record for *subject_id*'s write scope, as a metadata dict.

    Fails loudly when there is none: a test that silently accepts "no
    close record" would pass against a build that stopped writing them.
    """
    for row in _rows(ws):
        metadata = row.metadata or {}
        if metadata.get("write_phase") == PHASE_CLOSED and row.target_block_id == subject_id:
            return dict(metadata)
    raise AssertionError(f"no close record for {subject_id!r} in {[r.target_block_id for r in _rows(ws)]}")


def _fail_first_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apply_engine, "execute_op", lambda *a, **k: (False, "boom"))


# ---------------------------------------------------------------------------
# A — the close record tells the truth about a rolled-back apply
# ---------------------------------------------------------------------------


class TestFailedApplyClosesAsError:
    def test_a_landed_apply_closes_as_ok(self, workspace: str, preconditions_pass: None) -> None:
        """POSITIVE CONTROL for every assertion below.

        Proves the proposal really does open a write scope, that the
        scope really does close, that ``_close_record`` can find it, and
        that ``outcome=ok`` is a value this method observes — so the
        ``error`` assertion in the next test is a difference in the
        product, not an artefact of a reader that never worked.
        """
        ok, msg = apply_proposal(workspace, PROPOSAL_ID)
        assert ok, msg
        record = _close_record(workspace, PROPOSAL_ID)
        assert record["scope_outcome"] == OUTCOME_OK
        assert record["landed_count"] == 1

    def test_a_rolled_back_apply_closes_as_error(self, workspace: str, preconditions_pass: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """The defect, stated as the gate: the close record must say ``error``."""
        _fail_first_op(monkeypatch)
        ok, msg = apply_proposal(workspace, PROPOSAL_ID)
        assert ok is False
        assert msg == "Op 0 failed: boom"
        record = _close_record(workspace, PROPOSAL_ID)
        assert record["scope_outcome"] == OUTCOME_ERROR, "a rolled-back apply must not close as ok"
        assert record["landed_count"] == 0

    def test_the_restore_row_is_nested_inside_the_failed_scope(
        self, workspace: str, preconditions_pass: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Order matters: the rollback happens *inside* the scope it aborts.

        An auditor reading the chain forward sees the apply open, the
        restore run and close, and only then the apply close as an
        error. A restore recorded after the outer close would describe a
        withdrawal from a scope the chain already called finished.
        """
        _fail_first_op(monkeypatch)
        apply_proposal(workspace, PROPOSAL_ID)
        rows = _rows(workspace)
        subjects = [r.target_block_id for r in rows]
        restore = [s for s in subjects if s.startswith("restore:")]
        assert restore, f"no RESTORE row recorded: {subjects}"
        outer_close = max(
            i for i, r in enumerate(rows) if (r.metadata or {}).get("write_phase") == PHASE_CLOSED and r.target_block_id == PROPOSAL_ID
        )
        restore_close = max(
            i for i, r in enumerate(rows) if (r.metadata or {}).get("write_phase") == PHASE_CLOSED and r.target_block_id in restore
        )
        assert restore_close < outer_close

    def test_apply_aborted_never_escapes_to_the_caller(
        self, workspace: str, preconditions_pass: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exception is control flow, not an API change.

        ``apply_proposal`` still answers ``(False, message)``. If the
        ``except`` clause were ever moved or dropped, callers would start
        seeing an exception where they have always seen a tuple.
        """
        _fail_first_op(monkeypatch)
        try:
            result = apply_proposal(workspace, PROPOSAL_ID)
        except ApplyAborted as exc:  # pragma: no cover - the assertion below is the point
            raise AssertionError(f"ApplyAborted escaped apply_proposal: {exc}") from exc
        assert result == (False, "Op 0 failed: boom")

    def test_mutation_twin_a_plain_return_closes_the_scope_as_ok(self, workspace: str) -> None:
        """The defect, demonstrated directly against the gate.

        This is the shape ``apply_proposal`` had: leave an open admission
        scope by *returning*. The gate cannot see that the caller undid
        its work — a ``return`` is a normal exit — so it records ``ok``.
        Which is exactly why the fix has to be a raise, and why the close
        record above is worth reading.

        The source-level twin (putting ``return False`` back in
        ``apply_engine`` and watching
        ``test_a_rolled_back_apply_closes_as_error`` go red) is run at
        review time; this keeps the mechanism asserted in CI without a
        second copy of the engine.
        """
        gate = get_gate(workspace)

        def _returns_from_inside_the_scope() -> bool:
            with gate.admit_proposal("P-twin-return", "[]", actor="apply_engine", target_file=DECISION_FILE):
                return False

        assert _returns_from_inside_the_scope() is False
        assert _close_record(workspace, "P-twin-return")["scope_outcome"] == OUTCOME_OK

    def test_a_raise_closes_the_scope_as_error(self, workspace: str) -> None:
        """The other half of the twin: the same scope, left by raising."""
        gate = get_gate(workspace)
        with pytest.raises(ApplyAborted):
            with gate.admit_proposal("P-twin-raise", "[]", actor="apply_engine", target_file=DECISION_FILE):
                raise ApplyAborted(0, "boom")
        assert _close_record(workspace, "P-twin-raise")["scope_outcome"] == OUTCOME_ERROR


# ---------------------------------------------------------------------------
# B — one mode, one file, and it is the attested one
# ---------------------------------------------------------------------------


class TestOneModeSource:
    def test_the_bound_config_unblocks_an_apply(self, workspace: str, preconditions_pass: None) -> None:
        """POSITIVE CONTROL: something *can* unblock this apply.

        Without it, the refusals asserted below would be satisfied by an
        apply that is broken for some unrelated reason.
        """
        assert read_governance_mode(config_path_for(workspace)) == "propose"
        ok, msg = apply_proposal(workspace, PROPOSAL_ID)
        assert ok, msg

    @pytest.mark.parametrize("state_mode", ["propose", "enforce"])
    def test_intel_state_alone_cannot_unblock_an_apply(self, workspace: str, preconditions_pass: None, state_mode: str) -> None:
        """The bound config forbids; the unattested file says otherwise; the apply is refused."""
        _set_config_mode(workspace, DETECT_ONLY_MODE)
        _set_state_mode(workspace, state_mode)
        ok, msg = apply_proposal(workspace, PROPOSAL_ID)
        assert ok is False
        assert "detect_only" in msg

    def test_intel_state_alone_cannot_block_an_apply_either(self, workspace: str, preconditions_pass: None) -> None:
        """The mirror is inert in *both* directions.

        A key that can still block is a key that still decides, and an
        operator who edits it would go on believing it is the control.
        """
        _set_state_mode(workspace, DETECT_ONLY_MODE)
        ok, msg = apply_proposal(workspace, PROPOSAL_ID)
        assert ok, msg

    def test_an_unreadable_config_blocks_rather_than_enforces(self, tmp_path: Any) -> None:
        """Fail closed *for this decision*.

        The gate's strict answer to an unreadable config is ``enforce``,
        which refuses a drifted write — but an apply *proceeds* under
        ``enforce``. Sharing one "strict" value between the two readers
        would therefore have made an unreadable config unblock applies.
        """
        ws = str(tmp_path / "ws")
        init(ws)
        with open(config_path_for(ws), "w", encoding="utf-8") as handle:
            handle.write("{ not json")
        try:
            assert read_governance_mode(config_path_for(ws)) is None
            assert apply_engine._get_mode(ws) == DETECT_ONLY_MODE
        finally:
            evict_gate(ws)

    def test_mutation_twin_reading_intel_state_reopens_the_hole(
        self, workspace: str, preconditions_pass: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Restore the old ``_get_mode`` and watch the refusal above vanish."""

        def _old_get_mode(ws: str = ".") -> str:
            with open(os.path.join(ws, "memory/intel-state.json"), encoding="utf-8") as handle:
                state = json.load(handle)
            return str(state.get("governance_mode", "detect_only"))

        monkeypatch.setattr(apply_engine, "_get_mode", _old_get_mode)
        _set_config_mode(workspace, DETECT_ONLY_MODE)
        _set_state_mode(workspace, "propose")
        _ok, msg = apply_proposal(workspace, PROPOSAL_ID)
        # The refusal the fixed reader produces is gone: the unattested file
        # got the apply past the mode gate, which is the hole itself.
        assert "detect_only mode does not allow apply" not in msg, "mutation twin failed to restore the defect"


# ---------------------------------------------------------------------------
# C — a rollback never deletes the record of itself
# ---------------------------------------------------------------------------


class TestRollbackKeepsTheLedgers:
    def test_the_orphan_sweep_still_removes_a_non_ledger_file(self, tmp_path: Any) -> None:
        """POSITIVE CONTROL: the sweep runs and this method can see it work.

        Without this, "the ledgers survived" would be equally true of a
        build where ``_cleanup_orphan_files`` did nothing at all.
        """
        ws = str(tmp_path / "ws")
        init(ws)
        before = apply_engine._list_workspace_files(ws)
        _write(os.path.join(ws, "decisions/STRAY.md"), "created during the failed apply\n")
        for ledger in LEDGER_FILES:
            if ledger.startswith("memory/"):
                _write(os.path.join(ws, ledger), "")
        apply_engine._cleanup_orphan_files(ws, before)
        assert not os.path.exists(os.path.join(ws, "decisions/STRAY.md")), "the orphan sweep did not run"
        for ledger in LEDGER_FILES:
            if ledger.startswith("memory/"):
                assert os.path.exists(os.path.join(ws, ledger)), f"orphan sweep deleted the ledger {ledger}"

    def test_a_first_apply_that_fails_keeps_its_own_evidence(
        self, workspace: str, preconditions_pass: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End to end: nothing pre-seeds the ledgers, and they survive.

        This is the case the sweep got wrong — the gate is constructed
        after the pre-apply inventory is taken, so on a workspace's first
        apply its ledgers look brand new. Deleting them made the close
        record raise ``EvidenceChainCompromisedError`` out of
        ``apply_proposal`` instead of returning ``(False, msg)``.
        """
        chain_path = os.path.join(workspace, "memory/evidence_chain.jsonl")
        assert not os.path.exists(chain_path), "fixture must not pre-seed the evidence chain"
        _fail_first_op(monkeypatch)
        ok, msg = apply_proposal(workspace, PROPOSAL_ID)
        assert (ok, msg) == (False, "Op 0 failed: boom")
        assert os.path.exists(chain_path), "the rollback deleted the evidence chain"
        assert _close_record(workspace, PROPOSAL_ID)["scope_outcome"] == OUTCOME_ERROR

    def test_mutation_twin_sweeping_ledgers_destroys_the_record(
        self, workspace: str, preconditions_pass: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Put the unfiltered sweep back and watch the apply lose its chain."""

        def _old_cleanup(ws: str, pre_apply_files: set) -> None:
            for orphan in apply_engine._list_workspace_files(ws) - pre_apply_files:
                path = os.path.join(ws, orphan)
                if os.path.isfile(path):
                    os.remove(path)

        monkeypatch.setattr(apply_engine, "_cleanup_orphan_files", _old_cleanup)
        _fail_first_op(monkeypatch)
        ok, msg = apply_proposal(workspace, PROPOSAL_ID)
        assert (ok, msg) == (False, "Op 0 failed: boom")
        chain_path = os.path.join(workspace, "memory/evidence_chain.jsonl")
        assert not os.path.exists(chain_path), "mutation twin failed to restore the defect"
        # The close record could not be written at all: the chain it would
        # have linked to was deleted a few lines earlier by the rollback
        # that is supposed to be recording itself.
        with pytest.raises(AssertionError):
            _close_record(workspace, PROPOSAL_ID)
