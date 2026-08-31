"""An unreadable ``Status`` must be withheld on the stale-index path too.

``is_admissible_status`` separates two non-string cases on purpose
(``admissibility.py``, the ``isinstance(status, str)`` branch):

* an EMPTY collection is the third on-disk spelling of *unstated*, and
  unstated is servable;
* a POPULATED non-string is a status this code cannot read, and is withheld.

``_parse_statuses`` used to collapse both to ``""`` on its way into the
live-status map. ``""`` is the *unstated* spelling, so a block whose ``Status``
parsed as ``["quarantined"]`` — withheld when recall reads the corpus directly
— was SERVED the moment a stale index routed it through ``live_statuses`` /
``with_live_statuses``. That is fail-open on the exact path those two functions
exist to make fail-closed.
"""

from __future__ import annotations

import os
from typing import Any

from mind_mem.admissibility import (
    RECOGNISED_STATUSES,
    UNREADABLE_STATUS,
    _parse_statuses,
    admit_leg,
    is_admissible_status,
    with_live_statuses,
)

BID = "D-20260829-001"


def _corpus(tmp_path: Any, status_lines: str) -> str:
    """One decisions file holding one block with the given Status lines."""
    path = os.path.join(str(tmp_path), "DECISIONS.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"[{BID}]\nStatement: A governed thing\n{status_lines}Date: 2026-08-29\n\n---\n\n")
    return path


def _served(overrides: dict[str, str]) -> list[dict]:
    """Run a stale-index hit through the live-status overlay and the allow-list.

    The hit carries ``active`` the way an index that has not caught up with a
    quarantine would, so anything that survives here was served on the strength
    of the overlay alone.
    """
    hits = [{"_id": BID, "status": "active", "score": 1.0}]
    refreshed = with_live_statuses(hits, overrides)
    return admit_leg(refreshed, status_key="status", leg="test")


def test_a_list_valued_status_parses_as_a_populated_non_string(tmp_path: Any) -> None:
    """The premise: this really is how the parser renders it (not a synthetic dict)."""
    from mind_mem.block_parser import parse_file

    blocks = parse_file(_corpus(tmp_path, "Status:\n- quarantined\n"))
    assert blocks[0]["Status"] == ["quarantined"]
    assert is_admissible_status(blocks[0]["Status"]) is False


def test_populated_non_string_status_maps_to_the_withheld_sentinel(tmp_path: Any) -> None:
    statuses = _parse_statuses([_corpus(tmp_path, "Status:\n- quarantined\n")])
    assert statuses[BID] == UNREADABLE_STATUS
    assert is_admissible_status(statuses[BID]) is False


def test_stale_index_does_not_serve_a_block_the_direct_path_withholds(tmp_path: Any) -> None:
    """The defect itself: same block, two paths, one verdict."""
    statuses = _parse_statuses([_corpus(tmp_path, "Status:\n- quarantined\n")])
    assert _served(statuses) == []


def test_empty_collection_status_stays_servable(tmp_path: Any) -> None:
    """The other half of the distinction must not regress into fail-closed.

    A bare ``Status:`` line with nothing after it is the parser's rendering of
    *unstated*, and unstated is servable — withholding it would punish a block
    for how its author spaced a line.
    """
    statuses = _parse_statuses([_corpus(tmp_path, "Status:\n")])
    assert statuses[BID] == ""
    assert [h["_id"] for h in _served(statuses)] == [BID]


def test_a_readable_string_status_is_passed_through_verbatim(tmp_path: Any) -> None:
    statuses = _parse_statuses([_corpus(tmp_path, "Status: quarantined\n")])
    assert statuses[BID] == "quarantined"
    assert _served(statuses) == []

    statuses = _parse_statuses([_corpus(tmp_path, "Status: active\n")])
    assert statuses[BID] == "active"
    assert [h["_id"] for h in _served(statuses)] == [BID]


def test_the_sentinel_cannot_be_confused_for_a_lifecycle_status() -> None:
    """If the sentinel ever entered the recognised set it would serve silently."""
    assert UNREADABLE_STATUS.strip().lower() not in RECOGNISED_STATUSES
    assert is_admissible_status(UNREADABLE_STATUS) is False


def test_the_allow_widening_cannot_readmit_the_sentinel() -> None:
    """``include_pending`` widens by naming a lifecycle status, not by opening a door."""
    hits = [{"_id": BID, "status": UNREADABLE_STATUS}]
    assert admit_leg(hits, status_key="status", allow=frozenset({"pending"}), leg="test") == []
