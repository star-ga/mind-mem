# Copyright 2026 STARGA, Inc.
"""RA.1 — the served-set ledger: an append-only record of what was served.

Retrieval quality is measured everywhere. What is not: whether the memory
actually *helped*. Answering that needs an exact join — this run served these
blocks, in this order, under this pipeline — so a later outcome can be credited
to a **run** rather than correlated against a query string. This module is the
left-hand side of that join.

Three rules give it its shape, and each one is a refusal:

**One digest, one owner.** The served set is committed to by
:func:`~mind_mem.recall_digests.served_set_digest`, imported from its owner, a
leaf module. A second encoding of "the served set" is the drift a canonical
form exists to prevent, so this module mints none — not even a convenience
wrapper.

**The ledger is not the block store.** It is a separate append-only file with
its own row chain, anchored per row to the block chain head (``index_anchor``).
It is written *after* ``recall()`` has returned, and it writes no block, so
"nothing reaches the store without a gate receipt" is untouched.

**Nothing on the scoring path may read it.** Frequency-of-serving is derivable
from any served-set ledger; that is harmless only while it cannot flow backward
into a ranking. The rule is structural, not conventional: this module is
imported by the MCP recall *handler*, never by ``_recall_core`` or
``recall_attestation``, and ``tests/test_recall_attestation_v2.py`` fails the
build on an import edge in that direction.

Two amendments to RA.1 as originally worded, both deliberate:

* ``run_id`` is keyed on the **ordered** digest. A ledger that stores rank
  order but keys on the unordered set holds two distinct answers under one
  key, which an append-only ledger cannot be consistent with.
* ``run_id`` **excludes the scoring instant** — and, contra ``ROADMAP.md:94``,
  the index anchor with it. It names THE ANSWER, stably across days, so "has
  this exact answer been served before?" is answerable. The anchor advances on
  every store write, so binding it would make ``run_id`` per-write rather than
  per-answer. Both still travel on the row; only the key is narrowed. Repeated
  ``run_id`` rows are legitimate — ``seq`` is the ledger's unique key.

THE RULING ON ``credited`` — recorded here because this is where the counters
would live, and it is the one place a future author will look.

``ROADMAP.md:94`` lets ``served`` "buy attention tiers" and ``credited``
(distinct-actor successes) write ``block_tier_meta.confirmations`` and "buy
trust tiers". That is automatic tier promotion driven by agent-reported
outcomes, which is the do-not-build item — *"memory learns to retrieve from
rewards, automatically"* — arriving in a smaller hat. A bounded factor on a
score is fine; an unreviewed STATE TRANSITION from an agent signal is
actuation.

**``credited`` writes ``confirmations`` only through a proposal, or the
promotion is a ``plan_consolidation`` output that ``approve_apply`` executes.
Never a direct write. Never an automatic promotion off a counter.** Absence of
credit must never demote, either: silence-is-deletion is hostile to a governed
store, whatever it is dressed as.

Default OFF, opt-in through ``mind-mem.json``::

    {"served_ledger": {"enabled": true}}
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional

from .preimage import preimage
from .recall_digests import served_set_digest

#: Domain tag for the run identity. Its own tag, so a run id can never be
#: substituted for a row hash or an attestation hash of the same shape.
RUN_TAG = "MM_RUN_v1"

#: Domain tag for the row-chain link.
ROW_TAG = "MM_LEDGER_ROW_v1"

#: ``prev_row_hash`` of the first row. SHA-256 width — deliberately NOT the
#: 128-char ``hash_chain_v2.GENESIS_HASH`` (SHA3-512) nor
#: ``recall_attestation.GENESIS_ANCHOR``, which means "no block chain yet".
#: Three sentinels, three meanings; borrowing one would blur them.
GENESIS_ROW_HASH = "0" * 64

#: The ledger file, relative to the workspace. Its own directory, not the
#: block store's and not the audit chain's.
LEDGER_RELPATH = os.path.join(".mind-mem-ledger", "served.jsonl")

#: Sidecar holding the current chain head. The last row has no successor to
#: bind it, so without this its ``index_anchor`` / ``scoring_instant`` would be
#: editable until the next append. Not a row field — the schema is nine.
HEAD_RELPATH = os.path.join(".mind-mem-ledger", "served.head")

#: ``mind-mem.json`` section. Absent means off; only ``true`` means on.
CONFIG_SECTION = "served_ledger"

_HEX = frozenset("0123456789abcdef")

# In-process serialisation of read-tail-then-append. Cross-process appends are
# deferred: a race writes a duplicate ``seq`` or a stale ``prev_row_hash``,
# which :func:`verify_served_chain` reports and names rather than hiding.
# deferred: would take an OS file lock (fcntl / LockFileEx), stubbed because a
# portable lock is its own module — upgrade path: wrap _append_line() in one.
_append_lock = threading.Lock()


def _hex64(name: str, value: str) -> str:
    """Reject anything that is not a lowercase 64-char hex digest.

    ``run_id`` concatenates its three inputs with no separator between them.
    That is unambiguous only because each is fixed-width, so the width is a
    *contract*, enforced here rather than assumed.
    """
    text = str(value)
    if len(text) != 64 or not set(text) <= _HEX:
        raise ValueError(f"{name} must be a 64-character lowercase hex digest, got {value!r}")
    return text


def run_id(*, query_hash: str, served_digest: str, pipeline_hash: str) -> str:
    """``SHA256("MM_RUN_v1\\0" || query_hash || served_digest || pipeline_hash)``.

    Content-derived: no clock, no randomness, no sequence number. Two runs that
    answered the same question with the same blocks in the same order under the
    same pipeline share an id, on any host, on any day — which is exactly the
    question the ledger exists to answer.

    There is deliberately no ``scoring_instant`` parameter. Not "we chose not
    to pass it": excluding it is what makes the id name an answer rather than
    an occurrence, and a parameter would invite the opposite.
    """
    body = _hex64("query_hash", query_hash) + _hex64("served_digest", served_digest) + _hex64("pipeline_hash", pipeline_hash)
    return hashlib.sha256(RUN_TAG.encode("ascii") + b"\x00" + body.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class ServedRun:
    """One ledger row. Nine fields, and the ninth is the last one.

    What is **not** here is the point. No attestation, no ``degraded`` marker,
    no leg list, no per-item score, no verdict — a stored judgement about an
    answer is the thing a later ranking learns to read back, and the rail that
    keeps scoring away from this file is worth nothing if the file carries the
    judgement anyway. ``ids`` is the ranking; ``served_digest`` commits to it.
    """

    seq: int
    prev_row_hash: str
    run_id: str
    query_hash: str
    served_digest: str
    ids: tuple[str, ...]
    pipeline_hash: str
    index_anchor: str
    scoring_instant: str

    def to_row(self) -> dict[str, Any]:
        """Serialise to the on-disk shape — exactly the declared fields."""
        row = {f.name: getattr(self, f.name) for f in fields(self)}
        row["ids"] = list(self.ids)
        return row

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ServedRun:
        """Rebuild from a persisted row, refusing an unknown or missing key."""
        expected = {f.name for f in fields(cls)}
        if set(row) != expected:
            raise ValueError(f"row keys {sorted(row)} do not match the schema {sorted(expected)}")
        return cls(
            seq=int(row["seq"]),
            prev_row_hash=str(row["prev_row_hash"]),
            run_id=str(row["run_id"]),
            query_hash=str(row["query_hash"]),
            served_digest=str(row["served_digest"]),
            ids=tuple(str(i) for i in row["ids"]),
            pipeline_hash=str(row["pipeline_hash"]),
            index_anchor=str(row["index_anchor"]),
            scoring_instant=str(row["scoring_instant"]),
        )


def row_hash(row: ServedRun) -> str:
    """The chain link, **derived** rather than stored.

    A stored row hash can be rewritten alongside the row it covers; a derived
    one cannot. ``ids`` enters through ``served_digest``, whose agreement with
    the ids is checked separately by :func:`verify_served_chain` — so editing
    the ids alone fails there, and editing them consistently fails here.
    """
    return hashlib.sha256(
        preimage(
            ROW_TAG,
            row.seq,
            row.prev_row_hash,
            row.run_id,
            row.query_hash,
            row.served_digest,
            row.pipeline_hash,
            row.index_anchor,
            row.scoring_instant,
        )
    ).hexdigest()


@dataclass(frozen=True)
class ChainVerdict:
    """Outcome of :func:`verify_served_chain`, naming the row that failed."""

    ok: bool
    rows_checked: int
    bad_seq: Optional[int]
    reason: str
    head: str


def ledger_path(workspace: str | Path) -> str:
    """Absolute path of the ledger file for *workspace* (may not exist)."""
    return os.path.join(str(workspace), LEDGER_RELPATH)


def _head_path(workspace: str | Path) -> str:
    return os.path.join(str(workspace), HEAD_RELPATH)


def ledger_enabled(workspace: str | Path) -> bool:
    """True only when ``served_ledger.enabled`` is literally ``true``.

    Default OFF, and absent is off. Reads the config file directly rather than
    through the MCP config helper: this module must stay importable without
    pulling the server layer in behind it.
    """
    path = os.path.join(str(workspace), "mind-mem.json")
    try:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    section = config.get(CONFIG_SECTION) if isinstance(config, dict) else None
    return isinstance(section, dict) and section.get("enabled") is True


def read_served_runs(workspace: str | Path) -> tuple[ServedRun, ...]:
    """Every row on disk, in file order. Empty when the ledger is absent."""
    try:
        with open(ledger_path(workspace), encoding="utf-8") as handle:
            lines = [line for line in handle.read().splitlines() if line.strip()]
    except OSError:
        return ()
    return tuple(ServedRun.from_row(json.loads(line)) for line in lines)


def _next_link(workspace: str | Path) -> tuple[int, str]:
    """``(seq, prev_row_hash)`` for the row about to be appended."""
    rows = read_served_runs(workspace)
    if not rows:
        return 0, GENESIS_ROW_HASH
    return rows[-1].seq + 1, row_hash(rows[-1])


def append_served_run(
    workspace: str | Path,
    *,
    query_hash: str,
    served_digest: str,
    ids: Sequence[str],
    pipeline_hash: str,
    index_anchor: str,
    scoring_instant: str,
) -> Optional[ServedRun]:
    """Append one row. Returns ``None`` — writing nothing — when disabled.

    Reads no clock: ``scoring_instant`` is the value the run *already scored
    with*, passed through. A ledger that stamped its own "now" would record an
    instant the run never used, and the row would attest to a different
    scoring than the one that happened.

    ``served_digest`` is passed in rather than recomputed so the row commits to
    the digest the run's attestation already published; it is cross-checked
    against ``ids`` here, and again on every verification.
    """
    if not ledger_enabled(workspace):
        return None
    served = tuple(str(i) for i in ids)
    if served_set_digest(served) != served_digest:
        raise ValueError("served_digest does not match ids — refusing to record an inconsistent row")
    with _append_lock:
        seq, prev = _next_link(workspace)
        row = ServedRun(
            seq=seq,
            prev_row_hash=prev,
            run_id=run_id(query_hash=query_hash, served_digest=served_digest, pipeline_hash=pipeline_hash),
            query_hash=_hex64("query_hash", query_hash),
            served_digest=served_digest,
            ids=served,
            pipeline_hash=_hex64("pipeline_hash", pipeline_hash),
            index_anchor=_hex64("index_anchor", index_anchor),
            scoring_instant=str(scoring_instant),
        )
        _write_row(workspace, row)
    return row


def _write_row(workspace: str | Path, row: ServedRun) -> None:
    """Append the row, then re-anchor the head sidecar."""
    path = Path(ledger_path(workspace))
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row.to_row(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    with open(_head_path(workspace), "w", encoding="utf-8") as handle:
        handle.write(row_hash(row) + "\n")


def _read_head(workspace: str | Path) -> str:
    try:
        with open(_head_path(workspace), encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _check_row(row: ServedRun, index: int) -> str:
    """Failure reason for one row's INTERNAL invariants, or ``""``.

    These three are self-checking: a row carries enough to convict itself, so
    they name the tampered row directly rather than the row after it.
    """
    if row.seq != index:
        return f"row {index}: seq {row.seq} out of order"
    if served_set_digest(row.ids) != row.served_digest:
        return f"row {index}: served_digest does not match ids"
    expected = run_id(
        query_hash=row.query_hash,
        served_digest=row.served_digest,
        pipeline_hash=row.pipeline_hash,
    )
    if row.run_id != expected:
        return f"row {index}: run_id is not derived from query_hash + served_digest + pipeline_hash"
    return ""


def _link_breaks(rows: Sequence[ServedRun]) -> list[int]:
    """Positions whose ``prev_row_hash`` disagrees with their predecessor.

    Checked independently per position — never propagated forward — so the
    breaks a single edit produces are a *pattern*, and the pattern is what
    :func:`_locate_break` reads to name the row that was actually edited.
    """
    out: list[int] = []
    for index, row in enumerate(rows):
        expected = GENESIS_ROW_HASH if index == 0 else row_hash(rows[index - 1])
        if row.prev_row_hash != expected:
            out.append(index)
    return out


def _locate_break(rows: Sequence[ServedRun], breaks: list[int], stored_head: str) -> int:
    """Name the edited row from the break pattern.

    A chain link seals the row BEFORE it, so a naive report always accuses the
    successor of its predecessor's edit. The pattern disambiguates:

    * editing row *j*'s sealed content changes only ``row_hash(j)``, so exactly
      one link breaks — at *j+1*. The culprit is *j*.
    * editing row *j*'s own ``prev_row_hash`` breaks the link at *j* AND
      changes ``row_hash(j)``, breaking *j+1* too. Two consecutive breaks; the
      culprit is *j*.
    * on the last row the successor does not exist, so the head sidecar plays
      its part: a head that still matches the row means the row is intact and
      the edit is behind it.
    """
    first = breaks[0]
    if first + 1 in breaks:
        return first
    if first == len(rows) - 1 and stored_head and stored_head != row_hash(rows[first]):
        return first
    return max(0, first - 1)


def verify_served_chain(workspace: str | Path) -> ChainVerdict:
    """Walk the chain and name the FIRST row that cannot be trusted.

    Reads no clock, so a verification run is reproducible on any host on any
    day. Two passes: the self-checking invariants first (position, digest vs
    ids, ``run_id`` vs its three inputs), then the links — which seal the
    fields no invariant covers, ``index_anchor`` and ``scoring_instant``.
    """
    try:
        rows = read_served_runs(workspace)
    except (ValueError, json.JSONDecodeError) as exc:
        return ChainVerdict(ok=False, rows_checked=0, bad_seq=None, reason=f"unreadable row: {exc}", head="")

    for index, row in enumerate(rows):
        reason = _check_row(row, index)
        if reason:
            return ChainVerdict(ok=False, rows_checked=index, bad_seq=index, reason=reason, head="")

    stored_head = _read_head(workspace)
    breaks = _link_breaks(rows)
    if breaks:
        culprit = _locate_break(rows, breaks, stored_head)
        reason = f"row {culprit}: chain break — the link at row {breaks[0]} does not seal it"
        return ChainVerdict(ok=False, rows_checked=culprit, bad_seq=culprit, reason=reason, head="")

    head = row_hash(rows[-1]) if rows else GENESIS_ROW_HASH
    if rows and stored_head and stored_head != head:
        last = len(rows) - 1
        return ChainVerdict(
            ok=False,
            rows_checked=last,
            bad_seq=last,
            reason=f"row {last}: chain head does not match the recorded head",
            head=head,
        )
    return ChainVerdict(ok=True, rows_checked=len(rows), bad_seq=None, reason="", head=head)


__all__ = [
    "CONFIG_SECTION",
    "GENESIS_ROW_HASH",
    "HEAD_RELPATH",
    "LEDGER_RELPATH",
    "ROW_TAG",
    "RUN_TAG",
    "ChainVerdict",
    "ServedRun",
    "append_served_run",
    "ledger_enabled",
    "ledger_path",
    "read_served_runs",
    "row_hash",
    "run_id",
    "verify_served_chain",
]
