# Copyright 2026 STARGA, Inc.
"""The one word that says how a write scope ended must be true.

The product claim is *prove what it served, stored and destroyed*. The
governance gate writes exactly one close record per write scope, and
``metadata["scope_outcome"]`` is the field that says how the scope
ended. A record that says ``ok`` over work that was undone does not make
the chain coarse — it makes it **false**, and sealed under a hash, so the
lie is tamper-evident rather than detectable.

``apply_engine`` has two branches that withdraw an apply's own work, and
they were fixed one release apart:

* **op failure** — an op returns ``(False, …)``, the snapshot is
  restored, the apply returns. Closed as ``ok`` until it was made to
  raise. That branch, its ledger survival and the ``governance_mode``
  single-source fix are held by
  ``tests/test_apply_abort_is_recorded.py``.
* **post-check failure** — every op lands, ``check_preconditions`` then
  fails, the snapshot is restored, the receipt and the proposal are
  marked ``rolled_back``, and the apply returns. Measured on a fresh
  workspace before the fix in this file's slice::

      control   → CLOSE P-20260902-001  outcome=ok    landed=1
      post-fail → CLOSE P-20260902-001  outcome=ok    landed=1
      DISTINGUISHABLE(control vs post-fail): False

  The two close records were identical in every field an auditor reads.
  The rollback was not invisible — the restore writes its own ``RESTORE``
  row — but the row that names how the *apply* ended said the wrong word,
  and only a reader who thought to join a sibling row would find out.

This file owns the property, not the branch: **a scope whose work was
withdrawn and a scope whose work stands must be distinguishable in the
chain**, for every branch that can withdraw, now and later. So it holds
three things:

A  :class:`TestFailedAndSuccessfulAreDistinguishable` — the comparison
   itself, run over both withdrawal branches, with a landed apply as the
   positive control.
B  :class:`TestTheScopeIsTheTransaction` — the by-construction guard.
   Every path that withdraws work leaves the admission scope by raising,
   enforced structurally over the source rather than by remembering:
   no ``return`` inside the scope, no rollback after it. A new branch
   added below cannot quietly reintroduce the defect.
C  :class:`TestOlderReadersStillParseTheRecord` — the outcome vocabulary
   stays two words and the close record adds no
   :class:`~mind_mem.evidence_objects.EvidenceAction` member, so a
   reader from the shipped release verifies rows it does not understand.

Measured against the real 5.0.1 reader (``HEAD:src/mind_mem/evidence_objects.py``
executed in memory, over a chain written by the fixed gate)::

    error close record action: VERIFY
    scope_error_type        : ApplyAborted
    HEAD EvidenceObject.from_dict -> action: <EvidenceAction.VERIFY: 'VERIFY'>
    HEAD chain.verify(record)  : True
    HEAD chain.verify_chain()  : True | errors: []

Every negative assertion here is paired with a positive control, and
every guard with a mutation twin that restores the old behaviour and
asserts the guard goes red — a test that cannot fail is not a test.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
from typing import Any, Iterator, Optional

import pytest

from mind_mem import apply_engine, governance_gate
from mind_mem.apply_engine import STAGE_POST_CHECK, ApplyAborted, apply_proposal, compute_fingerprint
from mind_mem.evidence_objects import EvidenceAction, UnknownAction
from mind_mem.governance_gate import (
    CLOSE_VERB,
    OUTCOME_ERROR,
    OUTCOME_OK,
    PHASE_CLOSED,
    config_path_for,
    evict_gate,
    get_gate,
)
from mind_mem.init_workspace import init
from mind_mem.spec_binding import SpecBindingManager

PROPOSAL_ID = "P-20260902-001"
TARGET_ID = "D-20260902-001"
DECISION_FILE = "decisions/DECISIONS.md"

#: The field an auditor reads to learn how a scope ended. Named so the
#: comparison below is about *that* field and not about incidental
#: differences (timestamps, entry ids) that would make any two records
#: "distinguishable" without saying anything true.
OUTCOME_KEY = "scope_outcome"


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _stage_proposal(ws: str) -> None:
    """One staged proposal and the decision block it edits."""
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
    """An initialised workspace in ``propose`` mode with one staged proposal.

    The mode is set in the *bound* config and re-attested, because that
    is the only file the apply engine reads it from.
    """
    ws = str(tmp_path / "ws")
    init(ws)
    path = config_path_for(ws)
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    config["governance_mode"] = "propose"
    _write(path, json.dumps(config, indent=2))
    SpecBindingManager(path).rebind(path)
    _stage_proposal(ws)
    try:
        yield ws
    finally:
        evict_gate(ws)


def _pass_preconditions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both the pre-check and the post-check pass.

    ``check_preconditions`` shells out to the full workspace validator,
    which a three-file fixture cannot satisfy and which nothing here is
    about. Every assertion below is about the evidence chain.
    """
    monkeypatch.setattr(
        apply_engine,
        "check_preconditions",
        lambda ws: (True, ["validate: PASS (patched by the fixture)"]),
    )


def _fail_post_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-check passes, the post-check fails — the branch under test.

    Counted rather than flagged: the apply calls the same function twice,
    and a patch that failed both would never reach an op, so the scope
    would never open and the test would pass for the wrong reason.
    """
    calls = {"n": 0}

    def _staged(ws: str) -> tuple[bool, list[str]]:
        calls["n"] += 1
        if calls["n"] == 1:
            return True, ["validate: PASS (pre)"]
        return False, ["validate: FAIL (post, patched by the fixture)"]

    monkeypatch.setattr(apply_engine, "check_preconditions", _staged)


def _fail_first_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apply_engine, "execute_op", lambda *a, **k: (False, "boom"))


# ---------------------------------------------------------------------------
# Reading the chain
# ---------------------------------------------------------------------------


def _rows(ws: str) -> list[Any]:
    chain = get_gate(ws).evidence
    return list(chain.get_latest(n=len(chain)))


def _close_record(ws: str, subject_id: str) -> dict:
    """The close record for *subject_id*'s write scope, as a metadata dict.

    Raises rather than returning ``None``: a comparison that silently
    accepted "no close record" would report two builds as different
    because neither wrote anything.
    """
    for row in _rows(ws):
        metadata = row.metadata or {}
        if metadata.get("write_phase") == PHASE_CLOSED and row.target_block_id == subject_id:
            return dict(metadata)
    raise AssertionError(f"no close record for {subject_id!r} in {[r.target_block_id for r in _rows(ws)]}")


# ---------------------------------------------------------------------------
# A — the comparison
# ---------------------------------------------------------------------------


class TestFailedAndSuccessfulAreDistinguishable:
    """A withdrawn apply and a landed one must not close the same way."""

    def test_a_landed_apply_closes_as_ok(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """POSITIVE CONTROL for every comparison below.

        Proves the apply reaches its ops, the scope opens and closes,
        ``_close_record`` finds it, and ``ok`` is a value this method
        observes. Without it, "the two differ" would be satisfied by a
        build where the failing runs are broken for unrelated reasons.
        """
        _pass_preconditions(monkeypatch)
        ok, msg = apply_proposal(workspace, PROPOSAL_ID)
        assert ok, msg
        record = _close_record(workspace, PROPOSAL_ID)
        assert record[OUTCOME_KEY] == OUTCOME_OK
        assert record["landed_count"] == 1
        assert "scope_error_type" not in record, "the ok record carries no error detail"

    def test_an_op_failure_closes_as_error(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        _pass_preconditions(monkeypatch)
        _fail_first_op(monkeypatch)
        ok, msg = apply_proposal(workspace, PROPOSAL_ID)
        assert (ok, msg) == (False, "Op 0 failed: boom")
        record = _close_record(workspace, PROPOSAL_ID)
        assert record[OUTCOME_KEY] == OUTCOME_ERROR
        assert record["scope_error_type"] == ApplyAborted.__name__

    def test_a_post_check_failure_closes_as_error(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """The branch this file's slice fixed.

        Every op landed — ``landed_count`` is 1, and saying otherwise
        would be a second lie — and then the whole apply was withdrawn.
        ``error`` over a non-zero ``landed`` is exactly the shape an
        auditor needs: content was authorised, it landed, the scope
        failed, and the nested ``RESTORE`` row says what went back.
        """
        _fail_post_checks(monkeypatch)
        ok, msg = apply_proposal(workspace, PROPOSAL_ID)
        assert (ok, msg) == (False, "Post-checks failed, rolled back")
        record = _close_record(workspace, PROPOSAL_ID)
        assert record[OUTCOME_KEY] == OUTCOME_ERROR, "a rolled-back apply must not close as ok"
        assert record["landed_count"] == 1, "the block did land inside the scope; the record must not deny it"
        assert record["scope_error_type"] == ApplyAborted.__name__

    def test_the_workspace_really_was_rolled_back(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """The premise of the comparison, asserted rather than assumed.

        If the post-check branch did not actually withdraw the work,
        ``error`` would be the false record and this whole file would be
        arguing for the wrong answer.
        """
        _fail_post_checks(monkeypatch)
        apply_proposal(workspace, PROPOSAL_ID)
        with open(os.path.join(workspace, DECISION_FILE), encoding="utf-8") as handle:
            restored = handle.read()
        assert "Status: active" in restored, "the snapshot was not restored"
        assert "superseded" not in restored
        with open(os.path.join(workspace, "intelligence/proposed/DECISIONS_PROPOSED.md"), encoding="utf-8") as handle:
            proposal = handle.read()
        assert "Status: rolled_back" in proposal

    @pytest.mark.parametrize("withdraw", ["op", "post_check"])
    def test_the_close_records_differ(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, withdraw: str) -> None:
        """THE TEST: read the chain of a landed apply and a withdrawn one and compare.

        Two workspaces, the same proposal, one difference in the
        product's behaviour — and the field that names how the scope
        ended has to be the field that differs. Anything else (entry
        ids, timestamps) differs between any two runs and proves nothing.
        """
        outcomes = {}
        for name, arrange in (("landed", _pass_preconditions), ("withdrawn", None)):
            ws = str(tmp_path / f"ws-{withdraw}-{name}")
            init(ws)
            path = config_path_for(ws)
            with open(path, encoding="utf-8") as handle:
                config = json.load(handle)
            config["governance_mode"] = "propose"
            _write(path, json.dumps(config, indent=2))
            SpecBindingManager(path).rebind(path)
            _stage_proposal(ws)
            with monkeypatch.context() as patch:
                if arrange is not None:
                    arrange(patch)
                elif withdraw == "op":
                    _pass_preconditions(patch)
                    _fail_first_op(patch)
                else:
                    _fail_post_checks(patch)
                apply_proposal(ws, PROPOSAL_ID)
            outcomes[name] = _close_record(ws, PROPOSAL_ID)[OUTCOME_KEY]
            evict_gate(ws)
        assert outcomes["landed"] == OUTCOME_OK
        assert outcomes["withdrawn"] == OUTCOME_ERROR
        assert outcomes["landed"] != outcomes["withdrawn"], f"indistinguishable in the chain: {outcomes}"

    def test_the_restore_row_is_nested_inside_the_failed_scope(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Order is part of the truth.

        Reading the chain forward, an auditor sees the apply open, the
        restore run and close, and only then the apply close as an error.
        A restore recorded *after* the outer close would describe a
        withdrawal from a scope the chain had already called finished —
        which is precisely the shape the post-check branch had.
        """
        _fail_post_checks(monkeypatch)
        apply_proposal(workspace, PROPOSAL_ID)
        rows = _rows(workspace)
        restore_subjects = {r.target_block_id for r in rows if str(r.target_block_id).startswith("restore:")}
        assert restore_subjects, f"no RESTORE row recorded: {[r.target_block_id for r in rows]}"
        closes = [(i, r) for i, r in enumerate(rows) if (r.metadata or {}).get("write_phase") == PHASE_CLOSED]
        outer = max(i for i, r in closes if r.target_block_id == PROPOSAL_ID)
        restore = max(i for i, r in closes if r.target_block_id in restore_subjects)
        assert restore < outer, "the rollback must be recorded inside the scope it aborts"

    def test_apply_aborted_never_escapes_to_the_caller(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """The raise is control flow, not an API change.

        Both withdrawal branches now leave the scope by raising; both
        must still answer ``(False, message)``. If the ``except`` clause
        were widened, moved or dropped, callers would start seeing an
        exception where they have always seen a tuple.
        """
        _fail_post_checks(monkeypatch)
        try:
            result = apply_proposal(workspace, PROPOSAL_ID)
        except ApplyAborted as exc:  # pragma: no cover - the assertion below is the point
            raise AssertionError(f"ApplyAborted escaped apply_proposal: {exc}") from exc
        assert result == (False, "Post-checks failed, rolled back")

    def test_mutation_twin_a_withdrawal_after_the_close_cannot_change_the_word(self, workspace: str) -> None:
        """The defect, demonstrated directly against the gate.

        This is the shape the post-check branch had: let the scope end
        normally, and only then withdraw the work. The gate has already
        written ``ok`` — nothing that happens afterwards can edit a
        sealed record, which is why the rollback had to move *inside*
        the scope rather than the gate being taught to look for it.

        The source-level twin (moving the post-check block back outside
        the ``with`` and watching
        ``test_a_post_check_failure_closes_as_error`` go red) is run at
        review time; this keeps the mechanism asserted in CI without a
        second copy of the engine.
        """
        gate = get_gate(workspace)
        with gate.admit_proposal("P-twin-postscope", "[]", actor="apply_engine", target_file=DECISION_FILE):
            pass  # the ops "succeeded"
        # …and only now do the post-checks fail and the work get withdrawn.
        record = _close_record(workspace, "P-twin-postscope")
        assert record[OUTCOME_KEY] == OUTCOME_OK, "mutation twin failed to restore the defect"
        assert "scope_error_type" not in record

    def test_a_raise_closes_the_same_scope_as_error(self, workspace: str) -> None:
        """The other half of the twin: the same scope, left by raising."""
        gate = get_gate(workspace)
        with pytest.raises(ApplyAborted):
            with gate.admit_proposal("P-twin-raise", "[]", actor="apply_engine", target_file=DECISION_FILE):
                raise ApplyAborted(None, "post-checks failed", stage=STAGE_POST_CHECK)
        record = _close_record(workspace, "P-twin-raise")
        assert record[OUTCOME_KEY] == OUTCOME_ERROR
        assert record["scope_error_type"] == ApplyAborted.__name__


# ---------------------------------------------------------------------------
# B — by construction: the scope IS the transaction
# ---------------------------------------------------------------------------

#: Calls that withdraw work already done. A call to one of these after
#: the admission scope has closed is the defect, whatever the branch
#: around it looks like, because the close record is already sealed.
WITHDRAWAL_CALLS = frozenset({"restore_snapshot"})

#: The scope whose body must never be left by ``return``.
SCOPE_OPENER = "admit_proposal"


def _function_named(source: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no function {name!r} in the parsed source")


def _admission_scope(fn: ast.FunctionDef) -> ast.With:
    """The ``with gate.admit_proposal(...)`` statement inside *fn*."""
    for node in ast.walk(fn):
        if isinstance(node, ast.With):
            for item in node.items:
                call = item.context_expr
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == SCOPE_OPENER:
                    return node
    raise AssertionError(f"no `with …{SCOPE_OPENER}(…)` in {fn.name!r} — the apply no longer opens a proposal scope")


def _returns_inside_the_scope(fn: ast.FunctionDef) -> list[int]:
    """Line numbers of ``return`` statements lexically inside the scope."""
    scope = _admission_scope(fn)
    return [node.lineno for node in ast.walk(scope) if isinstance(node, ast.Return)]


def _callee(node: ast.Call) -> Optional[str]:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _withdrawals_after_the_scope(fn: ast.FunctionDef) -> list[tuple[str, int]]:
    """``(callee, lineno)`` for every withdrawal call after the scope ends."""
    scope = _admission_scope(fn)
    end = scope.end_lineno or scope.lineno
    return [
        (str(_callee(node)), node.lineno)
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and _callee(node) in WITHDRAWAL_CALLS and node.lineno > end
    ]


#: A copy of the shape the engine had, small enough to read. The positive
#: controls run the analysers over this: an analyser that cannot see a
#: true positive proves nothing when it reports none.
DEFECTIVE_SOURCE = """
def _apply_proposal_locked(ws, proposal, proposal_id, source_file, lock):
    restore_snapshot(ws, snap_dir)          # the pre-scope branch — legitimate
    with gate.admit_proposal(proposal_id, "[]"):
        for i, op in enumerate(proposal["Ops"]):
            ok, msg = execute_op(ws, op)
            if not ok:
                restore_snapshot(ws, snap_dir)
                return False, "op failed"    # <- leaves the scope normally
    ok, post_report = check_preconditions(ws)
    if not ok:
        restore_snapshot(ws, snap_dir)       # <- withdraws after the close
        return False, "post-checks failed"
"""


class TestTheScopeIsTheTransaction:
    """The truthful outcome is structural, not a thing to remember."""

    def test_nothing_returns_from_inside_the_admission_scope(self) -> None:
        """A ``return`` is a normal exit, and a normal exit reads ``ok``.

        The gate cannot see that the body undid its work, so the only
        exits the scope may have are a raise (recorded as ``error``) and
        falling off the end (an apply that stands).
        """
        fn = _function_named(inspect.getsource(apply_engine), "_apply_proposal_locked")
        assert _returns_inside_the_scope(fn) == [], "a return inside the scope closes it as ok whatever it undid"

    def test_positive_control_the_scan_sees_a_return_inside_the_scope(self) -> None:
        fn = _function_named(DEFECTIVE_SOURCE, "_apply_proposal_locked")
        assert _returns_inside_the_scope(fn), "the scan cannot see a true positive"

    def test_nothing_withdraws_work_after_the_scope_has_closed(self) -> None:
        """A rollback after the close cannot change a sealed record.

        The one legitimate ``restore_snapshot`` outside the scope is the
        precondition branch, which runs *before* the gate is even
        constructed — there is no scope there to tell the truth about.
        """
        fn = _function_named(inspect.getsource(apply_engine), "_apply_proposal_locked")
        assert _withdrawals_after_the_scope(fn) == [], "these rollbacks happen after the close record is written"

    def test_positive_control_the_scan_sees_a_withdrawal_after_the_scope(self) -> None:
        fn = _function_named(DEFECTIVE_SOURCE, "_apply_proposal_locked")
        found = _withdrawals_after_the_scope(fn)
        assert [name for name, _ in found] == ["restore_snapshot"], f"the scan cannot see a true positive: {found}"

    def test_the_scope_is_still_the_thing_being_scanned(self) -> None:
        """Guards the guard: both scans resolve through ``_admission_scope``.

        If the apply stopped opening a proposal scope, or opened it under
        another name, every assertion above would pass over nothing.
        ``_admission_scope`` raises in that case rather than returning
        empty, and this asserts the raise is not the state we are in.
        """
        fn = _function_named(inspect.getsource(apply_engine), "_apply_proposal_locked")
        scope = _admission_scope(fn)
        assert scope.end_lineno and scope.end_lineno > scope.lineno
        body = ast.get_source_segment(inspect.getsource(apply_engine), scope) or ""
        assert "execute_op(" in body, "the ops must run inside the scope"
        assert "check_preconditions(" in body, "the post-checks must run inside the scope"


# ---------------------------------------------------------------------------
# C — an older reader still parses the record
# ---------------------------------------------------------------------------


class TestOlderReadersStillParseTheRecord:
    """Forward compatibility, held as a rule rather than a memory."""

    def test_the_outcome_vocabulary_is_exactly_two_words(self) -> None:
        """No third ``scope_outcome`` value, ever, without a decision.

        Readers *dispatch* on this field. A third word is a word an older
        reader cannot dispatch, so a new distinction belongs in an
        additive metadata key (``scope_error_type`` is one) and not here.
        This fails the moment someone adds ``OUTCOME_WITHDRAWN``, which
        is the conversation the failure exists to force.
        """
        declared = {value for name, value in vars(governance_gate).items() if name.startswith("OUTCOME_") and isinstance(value, str)}
        assert declared == {OUTCOME_OK, OUTCOME_ERROR}, f"a new outcome value needs an older-reader story: {declared}"

    def test_the_close_record_adds_no_evidence_action_member(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """``CLOSE`` maps onto ``VERIFY``, which 5.0.1 already had.

        Verified against the real shipped reader out of band
        (``HEAD:src/mind_mem/evidence_objects.py`` parsed a close record
        as ``<EvidenceAction.VERIFY: 'VERIFY'>`` and ``verify_chain()``
        returned ``True``); this keeps the property asserted in CI, where
        there is no guarantee of a git checkout to read HEAD from.
        """
        _fail_post_checks(monkeypatch)
        apply_proposal(workspace, PROPOSAL_ID)
        closes = [r for r in _rows(workspace) if (r.metadata or {}).get("write_phase") == PHASE_CLOSED]
        assert closes, "no close record to check"
        for row in closes:
            parsed = EvidenceAction.parse(str(getattr(row.action, "value", row.action)))
            assert isinstance(parsed, EvidenceAction), f"close records must use an existing action member, got {parsed!r}"
            assert not isinstance(parsed, UnknownAction)
            # Verify by raw string, dispatch by enum: the raw verb is kept
            # verbatim in metadata while the dispatched action is the
            # existing VERIFY member, so nothing has to learn a new word.
            assert row.metadata["action_verb"] == CLOSE_VERB
            assert row.metadata["scope_verb"] != CLOSE_VERB, "the scope's own verb is kept too, and it is not CLOSE"

    def test_positive_control_an_invented_verb_would_be_unknown(self) -> None:
        """Proves the check above can fail.

        If ``EvidenceAction.parse`` returned a member for everything, the
        assertion would be vacuous. It does not: an action string no
        release defines comes back as the ``UnknownAction`` sentinel.
        """
        parsed = EvidenceAction.parse("SCOPE_WITHDRAWN")
        assert isinstance(parsed, UnknownAction)
        assert not isinstance(parsed, EvidenceAction)

    def test_the_error_detail_is_additive_and_ignorable(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """A reader that knows only ``ok``/``error`` loses nothing.

        ``scope_error_type`` separates a deliberate, fully-handled
        withdrawal from a crash without a second outcome value. Dropping
        every key an older reader does not know must still leave a record
        that answers the question this file is about.
        """
        _fail_post_checks(monkeypatch)
        apply_proposal(workspace, PROPOSAL_ID)
        record = _close_record(workspace, PROPOSAL_ID)
        known_to_5_0_1 = {k: v for k, v in record.items() if k not in {"scope_error_type"}}
        assert known_to_5_0_1[OUTCOME_KEY] == OUTCOME_ERROR
        assert known_to_5_0_1["landed_count"] == 1
        assert record["scope_error_type"] == ApplyAborted.__name__
