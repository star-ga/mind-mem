#!/usr/bin/env python3
"""``skill_opt.validator.submit_to_governance`` stages a governed block (AUD-06).

It used to append a ``## SKILL-<mutation_id>`` markdown heading to
``intelligence/SIGNALS.md`` with ``open(..., "a")``. Two defects, and the
bypass was the smaller one: a ``##`` heading is not block syntax, so the id
the function returned — which ``mm skill-optimize`` records as the
mutation's ``governance_signal`` and stores against the mutation row —
resolved to nothing at all. "Submitted to governance" named a write
governance never saw and a handle no reader could follow.
"""

from __future__ import annotations

import os

import pytest
from _ledger_rows import count_chain_authorisations, count_evidence_authorisations

from mind_mem.enums import INITIAL_STATUS, IngestTier, is_servable
from mind_mem.init_workspace import init
from mind_mem.skill_opt._types import Mutation, ValidationResult
from mind_mem.skill_opt.validator import (
    SIGNAL_STATUS,
    SkillGovernanceError,
    submit_to_governance,
)


@pytest.fixture()
def ws(tmp_path) -> str:
    workspace = str(tmp_path / "wsp")
    init(workspace)
    return workspace


def _mutation(mutation_id: str = "m1", content: str = "improved skill body") -> Mutation:
    return Mutation(
        mutation_id=mutation_id,
        skill_id="skill-alpha",
        original_hash="0" * 64,
        proposed_content=content,
        rationale="tightened the failure-mode section",
    )


def _validation(improved: bool = True) -> ValidationResult:
    """``accepted`` is a computed property, so it is produced, not passed."""
    return ValidationResult(
        mutation_id="m1",
        skill_id="skill-alpha",
        score_before=0.5,
        score_after=0.75,
        improved=improved,
        critic_votes={"a": True, "b": True, "c": True},
    )


def _chain_admissions(ws: str) -> int:
    """Governed scopes that authorised something, per the hash chain.

    Counts authorisations rather than raw rows: a scope also appends a
    close record naming what landed. ``tests/_ledger_rows`` holds that
    convention once.
    """
    return count_chain_authorisations(ws)


def _evidence_admissions(ws: str) -> int:
    """The same count from the evidence chain."""
    return count_evidence_authorisations(ws)


def test_returned_id_resolves(ws: str) -> None:
    """The tracking handle names a block ``get_by_id`` returns."""
    from mind_mem.storage import get_block_store

    signal_id = submit_to_governance(_mutation(), _validation(), ws)
    assert signal_id.startswith("SIG-")

    block = get_block_store(ws).get_by_id(signal_id)
    assert block is not None, "the governance handle resolves to nothing — this is the pre-fix behaviour"
    assert block["Subject"] == "skill-alpha"
    assert block["Object"] == "m1"


def test_staged_signal_is_withheld_from_recall(ws: str) -> None:
    """Staged means PENDING, and pending means recall does not serve it."""
    from mind_mem.recall import recall
    from mind_mem.storage import iter_active_blocks

    signal_id = submit_to_governance(_mutation(), _validation(), ws)

    expected = INITIAL_STATUS[IngestTier.AUTO_CAPTURE]
    assert expected is not None and not is_servable(expected)
    assert SIGNAL_STATUS is expected

    assert signal_id not in {b.get("_id") for b in iter_active_blocks(ws)}
    assert signal_id not in {h.get("_id") for h in recall(ws, "skill-alpha mutation", limit=25)}


def test_submission_appends_to_both_ledgers(ws: str) -> None:
    before_chain, before_evidence = _chain_admissions(ws), _evidence_admissions(ws)
    submit_to_governance(_mutation(), _validation(), ws)
    assert _chain_admissions(ws) == before_chain + 1
    assert _evidence_admissions(ws) == before_evidence + 1


def test_no_prose_heading_is_written(ws: str) -> None:
    """The corpus file gains a BLOCK, not a ``##`` heading.

    Paired with a positive control: the block must be there, or "no ``##``"
    would also pass on a submission that silently wrote nothing.
    """
    signal_id = submit_to_governance(_mutation(), _validation(), ws)
    with open(os.path.join(ws, "intelligence", "SIGNALS.md"), "r", encoding="utf-8") as handle:
        text = handle.read()
    assert f"[{signal_id}]" in text, "positive control: the governed block is not in SIGNALS.md at all"
    assert "## SKILL-m1" not in text


def test_duplicate_submission_is_refused(ws: str) -> None:
    """An identical mutation cannot be staged twice.

    Same convention as ``LintAutofixError`` and ``ImportQuarantineError``:
    refuse loudly rather than mint a second staged copy of one decision.
    """
    submit_to_governance(_mutation(), _validation(), ws)
    with pytest.raises(SkillGovernanceError, match="already staged"):
        submit_to_governance(_mutation(), _validation(), ws)


def test_a_different_mutation_gets_the_next_id(ws: str) -> None:
    """Ids do not collide, and the counter is read from the store."""
    first = submit_to_governance(_mutation("m1", "body one"), _validation(), ws)
    second = submit_to_governance(_mutation("m2", "body two"), _validation(), ws)
    assert first != second
    assert int(second.rsplit("-", 1)[1]) == int(first.rsplit("-", 1)[1]) + 1


def test_write_is_refused_without_an_admission(ws: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION CONTROL: strip the gate scope and the write must fail."""
    import contextlib

    import mind_mem.governance_gate as gg

    class _Ungated:
        def admit_block(self, *args, **kwargs):
            return contextlib.nullcontext()

    monkeypatch.setattr(gg, "get_gate", lambda _ws: _Ungated())

    from mind_mem.admission import UngatedWriteError

    with pytest.raises(UngatedWriteError):
        submit_to_governance(_mutation(), _validation(), ws)
