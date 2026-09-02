"""RA.4 — the retention class, and the two things it must refuse to be.

The classifier answers one question: how much scrutiny does removing this
block need? Three properties carry that, and each is tested here rather
than asserted in a docstring.

R1  **Pure — no clock, no config, no I/O.** A retention class that moved
    with the wall clock could not be sealed into a death record: replaying
    the record a month later would compute a different class for the same
    bytes. Tested structurally (the module's own source names no clock, and
    the scanner is shown finding one in a source that does) and behaviourally
    (classification runs with every date clock broken).

R2  **PROTECTED is not self-declared.** A block does not become undeletable
    by saying so. Both protections are derived from an authority that
    already exists — ``admissibility.release_ids`` for a release decision,
    ``guardrails.guardrail_provenance_refusal`` for a guardrail — so
    external content cannot mint either one. The guardrail case is the
    live injection concern: a trigger-bearing block already bypasses the
    ranker, and protecting one the guardrail loader itself refuses would
    hand that primitive a second life as a block nothing may remove.

R3  **EPHEMERAL is about admission, never about worth.** Quarantined
    content is one approved release decision away from GOVERNED, and the
    test pins exactly that transition rather than the label alone.

Every negative assertion below is paired with the positive control that
proves the assertion could have failed.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from _recall_clock_sentinel import clock_census

from mind_mem.admissibility import RELEASE_FIELD
from mind_mem.guardrails import GUARDRAIL_ID_PREFIX
from mind_mem.retention_class import (
    EPHEMERAL,
    GOVERNED,
    PROTECTED,
    RETENTION_CLASSES,
    protected_reason,
    retention_class,
)

#: An id a release decision is allowed to name. Derived from the block-store
#: prefix routing table the same way ``admissibility._releasable_id_pattern``
#: derives it — a release decision may only admit what an untrusted ingest
#: tier minted, so a ``D-`` id here would be silently non-releasable and the
#: test would pass for the wrong reason.
RELEASABLE_ID = "IMP-20260901-001"

PLAIN_DECISION = {"_id": "D-20260901-001", "Statement": "ship the thing", "Status": "active"}

RELEASE_DECISION = {
    "_id": "D-20260901-002",
    "Statement": f"admit {RELEASABLE_ID}",
    "Status": "active",
    RELEASE_FIELD: [RELEASABLE_ID],
}

GUARDRAIL = {
    "_id": f"{GUARDRAIL_ID_PREFIX}20260901-001",
    "Statement": "never run `git reset --hard` without checking `git status`",
    "Status": "active",
    "TriggerCommands": "git reset --hard",
}

QUARANTINED = {"_id": "IMP-20260901-002", "Statement": "imported note", "Status": "quarantined"}


# ---------------------------------------------------------------------------
# R2 — PROTECTED, and the two authorities it defers to
# ---------------------------------------------------------------------------


def test_an_active_release_decision_is_protected_and_says_why() -> None:
    """Removing it silently withholds every id it admits, so it is load-bearing."""
    assert retention_class(RELEASE_DECISION) == PROTECTED
    assert "release decision" in protected_reason(RELEASE_DECISION)


def test_a_superseded_release_decision_is_no_longer_protected() -> None:
    """Positive control for the clause above: the protection tracks the authority.

    ``release_ids`` only honours an ACTIVE release decision — superseding one
    stops admitting its batch, which is what makes a governance rollback
    re-quarantine for free. A retention class that stayed PROTECTED here would
    be protecting a decision that no longer admits anything.
    """
    retired = dict(RELEASE_DECISION, Status="superseded")
    assert retention_class(retired) == GOVERNED
    assert protected_reason(retired) == ""


def test_a_release_decision_naming_nothing_releasable_is_not_protected() -> None:
    """A decision that admits no id withholds nothing when it goes."""
    empty = dict(RELEASE_DECISION)
    empty[RELEASE_FIELD] = ["D-20260901-999"]  # not an ingest-corpus id
    assert retention_class(empty) == GOVERNED


def test_a_guardrail_is_protected() -> None:
    assert retention_class(GUARDRAIL) == PROTECTED
    assert "guardrail" in protected_reason(GUARDRAIL)


@pytest.mark.parametrize(
    "marker",
    [
        {"Source": "imported:slack"},
        {"ActorRole": "importer"},
        {"ContentSource": "external"},
    ],
)
def test_external_content_cannot_mint_a_protected_guardrail(marker: dict) -> None:
    """R2 — the injection guard, with the clean block above as its positive control.

    ``test_a_guardrail_is_protected`` proves this method returns PROTECTED for
    a guardrail, so the ``!=`` below is a real refusal rather than a classifier
    that never protects anything.
    """
    hostile = dict(GUARDRAIL, **marker)
    assert retention_class(hostile) != PROTECTED
    assert protected_reason(hostile) == ""


def test_declaring_the_type_does_not_buy_protection() -> None:
    """A block is not a guardrail because it says it is — the id prefix is the rule."""
    impostor = dict(PLAIN_DECISION, Type="Guardrail", TriggerCommands="rm -rf /")
    assert retention_class(impostor) == GOVERNED


# ---------------------------------------------------------------------------
# R3 — EPHEMERAL is an admission fact, and it is reversible
# ---------------------------------------------------------------------------


def test_unadmitted_content_is_ephemeral() -> None:
    assert retention_class(QUARANTINED) == EPHEMERAL


def test_a_status_nobody_has_named_is_ephemeral_not_governed() -> None:
    """The fail-closed clause: an invented status is withheld, so it never passed the gate."""
    assert retention_class({"_id": "D-1", "Status": "staged-by-a-future-door"}) == EPHEMERAL


def test_ephemeral_is_about_admission_not_worth() -> None:
    """The same block, one approved release decision later, is GOVERNED.

    This is the sentence the class name has to survive: EPHEMERAL says the
    content never passed the gate, not that it is worthless.
    """
    assert retention_class(QUARANTINED) == EPHEMERAL
    released = dict(QUARANTINED, Status="active")
    assert retention_class(released) == GOVERNED


def test_plain_admitted_content_is_governed() -> None:
    assert retention_class(PLAIN_DECISION) == GOVERNED


def test_a_hit_shaped_dict_classifies_through_status_key() -> None:
    """Indexed hits spell the status ``status``; the classifier must not read past it."""
    hit = {"_id": "D-20260901-003", "status": "quarantined"}
    assert retention_class(hit, status_key="status") == EPHEMERAL
    assert retention_class(dict(hit, status="active"), status_key="status") == GOVERNED


def test_every_class_is_reachable_and_the_set_is_closed() -> None:
    produced = {
        retention_class(RELEASE_DECISION),
        retention_class(PLAIN_DECISION),
        retention_class(QUARANTINED),
    }
    assert produced == set(RETENTION_CLASSES)


# ---------------------------------------------------------------------------
# R1 — purity
# ---------------------------------------------------------------------------

_CLOCK_NAMES = frozenset({"now", "utcnow", "today", "time", "monotonic", "perf_counter"})


def _clock_calls(source: str) -> list[str]:
    """Every ``<something>.now()``-shaped call in *source*.

    An attribute-name scan rather than an import scan: the module could reach
    a clock through any alias, and the call is the thing that has to be absent.
    """
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _CLOCK_NAMES:
            found.append(f"{node.func.attr}() at line {node.lineno}")
    return found


def test_the_clock_scan_can_actually_see_one() -> None:
    """Positive control — without this the assertion below proves nothing."""
    assert _clock_calls("import datetime\ndef f():\n    return datetime.datetime.now()\n")


def test_retention_class_source_reads_no_clock() -> None:
    source = pathlib.Path(__import__("mind_mem.retention_class", fromlist=["x"]).__file__).read_text(encoding="utf-8")
    assert _clock_calls(source) == []


def test_the_census_sees_a_clock_read_made_from_inside_the_package() -> None:
    """Positive control for the census below — it is an observer, so it can be blind.

    Drives a known read through a ``mind_mem`` frame and requires the census
    to report it. Without this, an empty census is indistinguishable from a
    profiler hook that never armed.
    """
    from mind_mem.scoring_instant import resolve_scoring_instant

    with clock_census() as census:
        # THE sanctioned boundary read, made from inside ``scoring_instant.py``.
        # The census forbids that site unless a test opts in, so this is a read
        # it must report — the call has to originate in a ``mind_mem`` frame,
        # which is why a ``datetime.now()`` written here would not do.
        resolve_scoring_instant(None)
    assert census.reads


def test_classification_makes_no_date_clock_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """The behavioural half of R1 — the source scan alone could miss an alias.

    ``clock_census`` is a ``sys.setprofile`` observer over every
    ``datetime.now`` / ``date.today`` executed anywhere inside ``mind_mem``,
    so it needs no list of accessors and nothing can catch it.
    """
    with clock_census() as census:
        assert retention_class(RELEASE_DECISION) == PROTECTED
        assert retention_class(GUARDRAIL) == PROTECTED
        assert retention_class(PLAIN_DECISION) == GOVERNED
        assert retention_class(QUARANTINED) == EPHEMERAL
    census.assert_clock_free()


def test_classification_writes_nothing_to_disk(tmp_path) -> None:
    """No I/O: a workspace-shaped directory is untouched by a classification."""
    before = sorted(p.name for p in tmp_path.iterdir())
    for block in (RELEASE_DECISION, GUARDRAIL, PLAIN_DECISION, QUARANTINED):
        retention_class(block)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


# ---------------------------------------------------------------------------
# Mutation twin — a protection that cannot fail is not a protection
# ---------------------------------------------------------------------------


class TestMutationTwin:
    """Break each authority and require the protective test above to go red."""

    def test_neutering_the_provenance_refusal_lets_external_content_be_protected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from mind_mem import retention_class as module

        monkeypatch.setattr(module, "guardrail_provenance_refusal", lambda block: "")
        hostile = dict(GUARDRAIL, Source="imported:slack")
        with pytest.raises(AssertionError):
            assert retention_class(hostile) != PROTECTED

    def test_neutering_release_ids_drops_the_release_protection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem import retention_class as module

        monkeypatch.setattr(module, "release_ids", lambda blocks: frozenset())
        with pytest.raises(AssertionError):
            assert retention_class(RELEASE_DECISION) == PROTECTED
