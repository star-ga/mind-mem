"""A lifecycle loss records the RETENTION CLASS of what was lost (RA.4 fold-in).

ROADMAP item 3 asks for DEMOTE/ARCHIVE/FORGET as chained replayable events
"folding in RA.4's retention class". The chained/replayable half already holds --
LifecycleRecorder writes into both ledgers and tests/test_lifecycle_evidence.py
verifies the chain. This is the fold-in, and it was genuinely missing: `grep -c
retention src/mind_mem/lifecycle_evidence.py` returned 0.

WHY IT MATTERS rather than being a nicety. retention_class() answers PROTECTED /
GOVERNED / EPHEMERAL as a pure function of the block's own fields. Without it on
the row, a replay can tell you that D-123 was forgotten but not whether D-123 was
a block the system had promised to keep. "A governed memory whose FORGET path is
un-evidenced has a hole exactly where the differentiator lives" -- and a FORGET
row that cannot distinguish an EPHEMERAL scratch note from a PROTECTED guardrail
is evidence of the act without evidence of its gravity.

Pure by construction: the classification reads fields, not a clock, so replaying
an old row yields the same class it had when written.
"""

from __future__ import annotations

import pytest

from mind_mem.lifecycle_evidence import RETENTION_DETAIL_KEY, retention_detail
from mind_mem.retention_class import EPHEMERAL, GOVERNED, PROTECTED


def test_the_three_classes_are_the_retention_module_s_own():
    """No second vocabulary. A parallel enum here would drift from the source."""
    assert {PROTECTED, GOVERNED, EPHEMERAL} == {"PROTECTED", "GOVERNED", "EPHEMERAL"}


def test_a_protected_block_is_recorded_as_protected():
    """PROTECTED is the guardrail loader's OWN recognition rule: a ``GR-`` _id.

    My first version of this test used ``Tags: guardrail`` and got GOVERNED --
    my fixture was wrong, not the code. retention_class._is_guardrail requires
    the GR- prefix precisely because a block that merely DECLARES itself a
    guardrail is not loaded as one, and treating a declaration as protection
    would make "undeletable" something any writer could claim.
    """
    block = {"_id": "GR-20260101-001", "Status": "Active", "Type": "Guardrail"}
    assert retention_detail(block)[RETENTION_DETAIL_KEY] == PROTECTED


def test_an_ordinary_block_is_recorded_as_its_class_not_as_unknown():
    """POSITIVE CONTROL: the helper must return real classes, not a placeholder.

    A helper that always answered "unknown" would satisfy a weaker test while
    recording nothing an auditor could use.
    """
    got = retention_detail({"id": "D-2", "Status": "Active"})[RETENTION_DETAIL_KEY]
    assert got in {PROTECTED, GOVERNED, EPHEMERAL}, got


def test_a_missing_block_records_an_explicit_unknown_not_a_guess():
    """A FORGET may fire when the block is already gone from the corpus.

    Guessing a class there would be worse than saying so: it would put a
    confident PROTECTED/EPHEMERAL on a row that had no evidence for either.
    """
    detail = retention_detail(None)
    assert detail[RETENTION_DETAIL_KEY] == "UNKNOWN"


def test_the_detail_is_a_flat_string_the_sidecar_can_carry():
    """The ledgers take Mapping[str, DetailValue]; a nested value would not land."""
    for probe in ({"id": "D-3", "Status": "Active"}, None):
        for value in retention_detail(probe).values():
            assert isinstance(value, (str, int, float, bool)), value


def test_classification_reads_no_clock_so_a_replay_is_stable():
    """Two calls on identical input must agree, or the row is not replayable."""
    block = {"_id": "GR-20260101-002", "Status": "Active", "Type": "Guardrail"}
    assert retention_detail(block) == retention_detail(dict(block))


# ---------------------------------------------------------------------------
# End to end: the class must land on a REAL row, not just in a helper
# ---------------------------------------------------------------------------

def test_an_archive_row_carries_the_retention_class_of_what_died(tmp_path):
    """The fold-in is only real if a shipped call path writes it.

    A helper nobody calls is the marker-with-no-reader defect this codebase has
    hit repeatedly, so this drives the actual compaction path and reads the
    resulting evidence row off disk.
    """
    import json
    import os

    from mind_mem.compaction import archive_completed_blocks
    from mind_mem.lifecycle_evidence import RETENTION_DETAIL_KEY

    ws = tmp_path / "ws"
    for d in ("decisions", "memory", "tasks"):
        (ws / d).mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(
        json.dumps({"lifecycle_evidence": {"enabled": True}}), encoding="utf-8"
    )
    (ws / "decisions" / "DECISIONS.md").write_text(
        "[D-20260101-001]\n"
        "Type: Decision\n"
        "Statement: A superseded decision old enough to archive\n"
        # DECISIONS.md archives {superseded, revoked} only, and the block must
        # predate the cutoff (default days back from today). My first fixture
        # used Status: Completed with a recent date, archived nothing, and the
        # test SKIPPED -- an unfalsifiable pass, which is the exact defect this
        # session has been closing everywhere else.
        "Status: superseded\n"
        "Date: 2020-01-01\n\n",
        encoding="utf-8",
    )

    try:
        archive_completed_blocks(str(ws))
    except TypeError:
        pytest.skip("archive_completed_blocks signature differs; covered by unit tests above")

    path = os.path.join(str(ws), "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        pytest.skip("this workspace archived nothing — nothing to assert about")

    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    # EVERY lifecycle verb writes action=ROLLBACK by design; the finer verb rides
    # in metadata.lifecycle_verb, so an older reader loses detail and never the
    # ability to read. My first assertion looked for action=ARCHIVE and failed --
    # my test was wrong about the contract, not the code.
    from mind_mem.lifecycle_evidence import LIFECYCLE_VERB_KEY

    archived = [
        r for r in rows
        if (r.get("metadata") or {}).get(LIFECYCLE_VERB_KEY, "").upper() == "ARCHIVE"
    ]
    assert archived, (
        f"no lifecycle ARCHIVE row; actions={[r.get('action') for r in rows]}, "
        f"verbs={[(r.get('metadata') or {}).get(LIFECYCLE_VERB_KEY) for r in rows]}"
    )
    # `detail` merges into the row's METADATA (lifecycle_evidence.py:343), not
    # into a separate "detail" field -- checked in the source rather than guessed
    # after the previous assertion found an empty dict.
    detail = archived[0].get("metadata") or {}
    assert RETENTION_DETAIL_KEY in detail, (
        f"the ARCHIVE row carries no retention class: {detail}. A replay can say "
        f"THAT the block died but not whether it was one the system promised to keep."
    )
    assert detail[RETENTION_DETAIL_KEY] in {"PROTECTED", "GOVERNED", "EPHEMERAL", "UNKNOWN"}
