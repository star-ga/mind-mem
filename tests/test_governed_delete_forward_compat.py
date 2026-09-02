# Copyright 2026 STARGA, Inc.
"""A 5.0.1 reader must still parse what a 5.0.2 governed delete writes.

The delete scope put a record shape into the evidence chain that no
released version has ever seen: two rows per deletion carrying
``delete_phase``, ``operation``, ``removed_count``, ``merkle_root`` and
``admission_entry_id``. Anyone running the previous release against a
shared chain — a replica, a mirrored workspace, an auditor's copy — reads
those rows with the 5.0.1 parser. That parser is **strict** in one
specific place: ``EvidenceAction(d["action"])`` is a closed lookup, so a
new action value is not a field an old reader ignores, it is an exception
that stops the load. A frozen chain then refuses to append, which turns a
forward-compatibility slip into an outage rather than a warning.

The delete records were therefore built to reuse ``ROLLBACK`` and to put
every new fact inside ``metadata``, which the 5.0.1 reader carries
through as an opaque dict. That was a design decision with nothing
holding it in place. This file is what holds it: it re-implements the
5.0.1 reader from the shape released at the ``v5.0.1`` tag — raw JSON
line in, enum dispatch, the same required-field list — and runs it over
records produced by the live delete doors.

Every "it parsed" assertion here is paired with a demonstration that the
reader can refuse, because a parser that accepts everything proves
nothing about the records fed to it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import pytest

from mind_mem.evidence_objects import EvidenceAction
from mind_mem.governance_gate import PHASE_ADMITTED, PHASE_REMOVED, evict_gate
from mind_mem.http_transport import _handle_clear, _handle_delete_memory

# ---------------------------------------------------------------------------
# The 5.0.1 reader, reconstructed from the released record shape
# ---------------------------------------------------------------------------

#: ``EvidenceAction`` as released at the ``v5.0.1`` tag. A closed set:
#: ``EvidenceObject.from_dict`` calls ``EvidenceAction(d["action"])``, so
#: an action outside this vocabulary is a hard parse failure for every
#: reader at that version, not an ignorable unknown.
FROZEN_5_0_1_ACTIONS: frozenset[str] = frozenset({"PROPOSE", "APPLY", "ROLLBACK", "CONTRADICT", "DRIFT", "RESOLVE", "VERIFY"})

#: Keys ``EvidenceObject.from_dict`` indexes with ``[]`` at 5.0.1 — a
#: missing one is a ``KeyError``, so they are required, not optional.
FROZEN_5_0_1_REQUIRED: tuple[str, ...] = (
    "evidence_id",
    "timestamp",
    "action",
    "actor",
    "target_block_id",
    "target_file",
    "payload_hash",
    "previous_hash",
    "evidence_hash",
)

#: Every key 5.0.1's ``to_dict`` emitted. A new top-level key is not
#: fatal to that reader, but it is a fact it cannot see, so the delete
#: records must not put anything load-bearing up here.
FROZEN_5_0_1_TOP_LEVEL: frozenset[str] = frozenset(FROZEN_5_0_1_REQUIRED) | {"metadata", "confidence"}


class OldReaderRefused(Exception):
    """What the 5.0.1 parser does when it cannot read a row."""


def read_as_5_0_1(raw: str) -> dict[str, Any]:
    """Parse one raw chain line the way the 5.0.1 reader would.

    Deliberately re-implemented rather than imported: importing today's
    parser would test the current code against itself, which is the one
    thing this file must not do.
    """
    try:
        row = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - malformed line
        raise OldReaderRefused(f"not JSON: {exc}") from exc
    missing = [field for field in FROZEN_5_0_1_REQUIRED if field not in row]
    if missing:
        raise OldReaderRefused(f"missing required field(s): {missing}")
    if row["action"] not in FROZEN_5_0_1_ACTIONS:
        raise OldReaderRefused(f"unknown action {row['action']!r}; 5.0.1 knows {sorted(FROZEN_5_0_1_ACTIONS)}")
    return {
        "evidence_id": row["evidence_id"],
        "timestamp": row["timestamp"],
        "action": row["action"],
        "actor": row["actor"],
        "target_block_id": row["target_block_id"],
        "target_file": row["target_file"],
        "payload_hash": row["payload_hash"],
        "previous_hash": row["previous_hash"],
        "evidence_hash": row["evidence_hash"],
        "metadata": row.get("metadata", {}),
        "confidence": float(row.get("confidence", 1.0)),
    }


def links_as_5_0_1(rows: list[dict[str, Any]]) -> list[int]:
    """Indices where the hash chain does not link. What an old verifier checks."""
    breaks: list[int] = []
    for i in range(1, len(rows)):
        if rows[i]["previous_hash"] != rows[i - 1]["evidence_hash"]:
            breaks.append(i)
    return breaks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED_IDS = ("D-20260901-001", "D-20260901-002", "D-20260901-003")
CLEAR_BODY = {
    "rationale": "operator purge rehearsal for the release",
    "confirm": "yes-i-really-want-to-clear",
}


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[str]:
    ws = tmp_path / "ws"
    for sub in ("memory", "decisions", "tasks", "entities", "intelligence"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(
        json.dumps({"workspace_path": str(ws), "block_store": {"backend": "markdown"}}),
        encoding="utf-8",
    )
    for i, bid in enumerate(SEED_IDS, start=1):
        with open(ws / "decisions" / "DECISIONS.md", "a", encoding="utf-8") as handle:
            handle.write(f"[{bid}]\nStatement: forward-compat subject {i}\nDate: 2026-09-01\nStatus: active\n\n---\n\n")
    try:
        yield str(ws)
    finally:
        evict_gate(str(ws))


def _raw_lines(ws: str) -> list[str]:
    path = os.path.join(ws, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [line for line in handle if line.strip()]


@pytest.fixture
def deleted(workspace: str) -> str:
    """One single delete and one bulk clear — every delete record shape."""
    status, _body = _handle_delete_memory(workspace, SEED_IDS[0], actor="alice")
    assert status == 200, "the single-delete door failed, so its records are missing"
    status, body = _handle_clear(workspace, CLEAR_BODY, actor="alice")
    assert status == 200 and body["deleted"] == len(SEED_IDS) - 1, "the clear door failed, so its records are missing"
    return workspace


# ---------------------------------------------------------------------------
# The reader can refuse — established before it is trusted
# ---------------------------------------------------------------------------


def _a_valid_row() -> dict[str, Any]:
    return {
        "evidence_id": "e1",
        "timestamp": "2026-09-01T00:00:00+00:00",
        "action": "ROLLBACK",
        "actor": "alice",
        "target_block_id": "D-20260901-001",
        "target_file": "",
        "payload_hash": "0" * 64,
        "previous_hash": "0" * 64,
        "evidence_hash": "1" * 64,
        "metadata": {"delete_phase": "admitted"},
        "confidence": 1.0,
    }


def test_the_old_reader_accepts_a_well_formed_row() -> None:
    """Baseline: the refusals below are about the row, not the reader."""
    assert read_as_5_0_1(json.dumps(_a_valid_row()))["action"] == "ROLLBACK"


def test_the_old_reader_refuses_an_action_it_does_not_know() -> None:
    """The failure mode this file exists to prevent, demonstrated.

    A ``FORGET`` action would read as a clean new record to us and as an
    unparseable one to every 5.0.1 deployment.
    """
    row = _a_valid_row()
    row["action"] = "FORGET"
    with pytest.raises(OldReaderRefused, match="unknown action"):
        read_as_5_0_1(json.dumps(row))


@pytest.mark.parametrize("field", FROZEN_5_0_1_REQUIRED)
def test_the_old_reader_refuses_a_row_missing_a_field_it_requires(field: str) -> None:
    """Each required field is required — one at a time, so none is assumed."""
    row = _a_valid_row()
    del row[field]
    with pytest.raises(OldReaderRefused, match="missing required field"):
        read_as_5_0_1(json.dumps(row))


def test_the_link_check_can_report_a_break() -> None:
    """Positive control for :func:`links_as_5_0_1`."""
    a, b = _a_valid_row(), _a_valid_row()
    b["previous_hash"] = a["evidence_hash"]
    assert links_as_5_0_1([a, b]) == []
    b["previous_hash"] = "9" * 64
    assert links_as_5_0_1([a, b]) == [1]


# ---------------------------------------------------------------------------
# The records the delete doors actually write
# ---------------------------------------------------------------------------


def test_a_5_0_1_reader_parses_every_record_a_governed_delete_writes(deleted: str) -> None:
    """The whole point, asserted over real rows from both delete doors."""
    lines = _raw_lines(deleted)
    parsed = [read_as_5_0_1(line) for line in lines]

    # Positive control: the delete records are really in there, so this
    # is not a clean parse of an empty or write-only chain.
    phases = [p["metadata"].get("delete_phase") for p in parsed]
    assert phases.count(PHASE_ADMITTED) == 2, f"expected one authorisation per door; got phases {phases}"
    assert phases.count(PHASE_REMOVED) == 2, f"expected one removal record per door; got phases {phases}"


def test_every_delete_record_uses_an_action_5_0_1_already_knew(deleted: str) -> None:
    """No new enum member reaches the wire."""
    parsed = [read_as_5_0_1(line) for line in _raw_lines(deleted)]
    delete_rows = [p for p in parsed if p["metadata"].get("operation") == "delete"]
    assert delete_rows, "no delete records were written, so nothing was checked"
    assert {p["action"] for p in delete_rows} <= FROZEN_5_0_1_ACTIONS


def test_the_delete_records_add_no_new_top_level_field(deleted: str) -> None:
    """New facts live in ``metadata``, which an old reader carries opaquely."""
    for line in _raw_lines(deleted):
        row = json.loads(line)
        unknown = set(row) - FROZEN_5_0_1_TOP_LEVEL
        assert unknown == set(), f"a 5.0.1 reader cannot see top-level key(s) {sorted(unknown)}"


def test_the_new_delete_facts_survive_the_old_reader_intact(deleted: str) -> None:
    """An old reader loses none of it — it just does not interpret it."""
    parsed = [read_as_5_0_1(line) for line in _raw_lines(deleted)]
    removed = [p for p in parsed if p["metadata"].get("delete_phase") == PHASE_REMOVED]
    assert removed, "no removal record to read"
    for row in removed:
        assert row["metadata"]["operation"] == "delete"
        assert row["metadata"]["removed_count"] >= 1
        assert row["metadata"]["merkle_root"]
        assert row["metadata"]["admission_entry_id"]
        assert row["target_block_id"], "the record does not name what died"
        assert row["actor"] == "alice", "the record does not name who asked"


def test_the_chain_still_links_across_the_new_records(deleted: str) -> None:
    """An old verifier walking the file finds no break at a delete row."""
    parsed = [read_as_5_0_1(line) for line in _raw_lines(deleted)]
    assert len(parsed) >= 4, "too few records for the linkage to mean anything"
    assert links_as_5_0_1(parsed) == []


def test_the_frozen_vocabulary_is_still_the_live_one() -> None:
    """A renamed or removed member breaks old readers and old chains alike.

    Additive only: this asserts the 5.0.1 names still resolve, not that
    nothing was ever added — but anything added must first satisfy
    :func:`test_every_delete_record_uses_an_action_5_0_1_already_knew`
    before it can reach the wire from a delete.
    """
    live = {member.value for member in EvidenceAction}
    assert FROZEN_5_0_1_ACTIONS <= live, f"5.0.1 action(s) gone: {sorted(FROZEN_5_0_1_ACTIONS - live)}"
    for name in FROZEN_5_0_1_ACTIONS:
        assert EvidenceAction(name).value == name
