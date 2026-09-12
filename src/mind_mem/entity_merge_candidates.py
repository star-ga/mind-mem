"""Record entity MERGE CANDIDATES at mint time. Never merge.

ROADMAP, the remaining halves of two partials:
  * "Blocking + LLM-arbitration hybrid" -- "wiring blocking into the ``capture.py`` merge
    path";
  * "Description-grounded entity resolution" -- "wiring the guard into the ``capture.py``
    merge path".

BOTH POINTERS ARE WRONG, recorded here rather than quietly worked around: ``capture.py``
has no merge path -- no merge function, no entity resolution, no ``EntityRegistry``
reference. The real door is :meth:`EntityRegistry.resolve`, which canonicalises a surface
form and CREATES the entity when it is new. That create is the exact moment two spellings
of one person become two entities, and it is the only place a candidate can be noticed.

NOTHING IS MERGED, BY CONSTRUCTION, and that ordering is the safety property rather than a
policy. ``resolve`` keeps its behaviour exactly: a new surface still gets its own entity id
and every existing caller sees the same return value. An auto-merge on a blocking hit would
fuse "Ada Lovelace" and "Alan Lovelace" on a shared token -- and unmerging is the operation
this store cannot offer, which is why the whole entity-resolution design refuses to act
without a human.

Each candidate carries the ``merge_guard`` verdict and reason, so a reviewer sees WHY the
pair was proposed and whether the guard objected. Without the verdict the queue is a list
of coincidences. The guard never authorises a merge either, so the queue is advisory at
both ends.

Flag-gated OFF (``v4.merge_candidates``) and probed silently: this WRITES ROWS, and a
feature that starts writing to an operator's database because they upgraded is not
additive. With the flag off the table is never even created.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Any

__all__ = [
    "MERGE_CANDIDATES_FLAG",
    "SCHEMA",
    "list_candidates",
    "note_candidates_for",
]

MERGE_CANDIDATES_FLAG = "merge_candidates"

#: Append-only review queue. ``PRIMARY KEY (name, candidate)`` is what makes recording
#: idempotent: one pair is ONE fact, and a second row would double-count it in any review
#: view -- a queue that shows the same pair twice trains a reviewer to skim.
SCHEMA = """
CREATE TABLE IF NOT EXISTS entity_merge_candidates (
    name       TEXT NOT NULL,
    candidate  TEXT NOT NULL,
    verdict    TEXT NOT NULL,
    reason     TEXT NOT NULL,
    PRIMARY KEY (name, candidate)
);
"""


def _flag_on() -> bool:
    """Silent probe. Never ``is_enabled`` -- it warns on a malformed config, and a probe
    that logs on an OFF path makes the flag-off build observably different."""
    try:
        from .v4.feature_flags import is_enabled_quiet

        return bool(is_enabled_quiet("merge_candidates"))
    except Exception:  # pragma: no cover — a probe must never raise
        return False


def note_candidates_for(
    conn: sqlite3.Connection,
    lock: threading.RLock,
    name: str,
    existing: list[str],
) -> list[str]:
    """Record a candidate row for each existing name that blocks with *name*.

    Returns the candidate names recorded (possibly empty). Writes only to
    ``entity_merge_candidates`` -- never to ``entities`` or ``aliases``, which is asserted
    by a test over this module's source because "it does not merge" has to be a property,
    not a promise.
    """
    from .entity_blocking import block_for, build_blocks
    from .merge_guard import assess_merge

    target = str(name or "").strip()
    if not target:
        return []

    # Blocking decides WHO is even worth asking about -- the point of the cheap layer is
    # that no expensive judgement is ever asked whether "Ada Lovelace" and "ada lovelace"
    # are the same person.
    # build_blocks groups the whole name set; block_for then reads the block containing
    # this name out of that mapping. Two calls, because the grouping is over the SET and
    # the lookup is per name -- passing a name list straight to block_for was my first
    # attempt and it takes the mapping, not the list.
    blocks = build_blocks([*existing, target])
    block = [n for n in block_for(blocks, target) if n != target]
    if not block:
        return []

    recorded: list[str] = []
    with lock:
        conn.executescript(SCHEMA)
        for candidate in block:
            # The descriptions are the NAMES here: at mint time there is no per-entity
            # description yet, so the guard judges what is available and its verdict is
            # recorded as such rather than dressed up as more than it is.
            assessment = assess_merge(target, candidate)
            conn.execute(
                "INSERT OR IGNORE INTO entity_merge_candidates"
                "(name, candidate, verdict, reason) VALUES (?, ?, ?, ?)",
                (target, candidate, assessment.verdict.value, assessment.reason),
            )
            recorded.append(candidate)
        conn.commit()
    return recorded


def list_candidates(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every recorded candidate, or an empty list when the table does not exist.

    A missing table is "nothing recorded", not an error: with the flag off it is never
    created, and a reader asking an off feature what it queued is a normal question.
    """
    try:
        rows = conn.execute(
            "SELECT name, candidate, verdict, reason FROM entity_merge_candidates "
            "ORDER BY name, candidate"
        ).fetchall()
    except sqlite3.Error:
        return []
    return [
        {"name": r[0], "candidate": r[1], "verdict": r[2], "reason": r[3]} for r in rows
    ]
