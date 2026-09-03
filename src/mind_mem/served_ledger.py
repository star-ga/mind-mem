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

What the chain proves, stated narrowly: no row was **altered or removed** after
it was written — a truncated tail fails against the head sidecar, an edited row
fails against its successor's link, and an emptied ledger or a removed sidecar
fails as well, because the head comparison is unconditional rather than skipped
whenever one of the two files is gone.

And it keeps failing. A verifier that convicts a ledger is worth nothing if the
next ordinary recall quietly repairs it, which is what happened: an append that
found no rows started a fresh sequence at ``seq 0`` and re-sealed the head over
the deletion, so ``rm served.jsonl`` plus one recall turned a named break into
a clean chain. :func:`_next_link` now refuses — it may start a new sequence
only where neither a row nor a seal has ever existed, and may extend only a
tail the seal agrees with — so a break stays visible until an operator decides
what to do about it.

It does not prove **completeness**. An append that never happened leaves no gap
to detect, so a run whose write failed is absent rather than visibly missing —
but the RECORD of that run is not silent about it: every attestation carries
``served_proof`` (``recorded`` / ``unproven``) beside ``ledger_error``, so a
caller can always tell "this run was never recorded" from "somebody removed the
row". The answer is still served; see :func:`attach_served_run` for why that is
the ruling and what the alternative would cost. "Proof of what was served" is
therefore a claim about every row present, and about every record saying
whether it has one — not a claim that every run has one.

Nor is it a claim that the ledger itself survived. Rows and seal both live in
``.mind-mem-ledger``, so deleting the directory outright is indistinguishable
from a workspace where the ledger never ran — and, for the same reason, an
editor who rewrites BOTH files consistently (drop the last rows, re-seal to the
new last row) leaves a shorter history that verifies, which the guard above
does not and cannot change: it stops the ledger from performing that repair
*for* an editor who did only half of it. Closing the other half needs an anchor
kept somewhere this module does not write, and inventing one here would be a
second place to keep the truth.

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

Default **ON** since 5.0.2 — an opt-in proof is not a proof. Opt out, per
workspace, with the one literal that means it::

    {"served_ledger": {"enabled": false}}
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

from .mind_filelock import FileLock
from .preimage import preimage

# ``RUN_TAG`` and ``run_id`` are re-exported, not defined here. This module
# *stores* the run identity; since RA.1's residual closed it no longer
# *defines* it, because :mod:`mind_mem.recall_attestation` must publish the
# same id on the recall envelope and may not reach a ledger module on any
# import path (``tests/test_recall_attestation_v2.py`` fails the build on that
# edge). One object, one encoding, one owner — the rule that already governs
# ``served_set_digest``. Both names stay importable from here and stay in
# ``__all__``: a name that moved silently is a break for every caller that
# already spells it this way.
from .recall_digests import RUN_TAG, hex64, run_id, served_set_digest

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
#: Once a row exists the sidecar is REQUIRED, not merely consulted when
#: present: a seal whose absence is tolerated is the seal an editor removes
#: first, and tolerating it turns "delete one file" into a clean pass.
HEAD_RELPATH = os.path.join(".mind-mem-ledger", "served.head")

#: ``mind-mem.json`` section. Absent means ON; only a literal ``false`` opts
#: out. See :func:`ledger_enabled` for why the default inverted in 5.0.2.
CONFIG_SECTION = "served_ledger"

#: The three keys this module contributes to a run's attestation, and the ONE
#: place their names are written down. A serving surface gets them by calling
#: :func:`attach_served_run`, never by spelling them itself, so a fourth
#: surface cannot publish two of the three or misname one.
#:
#: They are carried BESIDE the attestation preimage, and that is a deliberate
#: limit rather than an oversight. Binding them would mean a new
#: ``RECALL_ATTEST`` tag — the attestation's own rule for a layout change —
#: and it would buy nothing: what makes the row trustworthy is the ledger's
#: row chain, not a second hash of a pointer to it. So the narrow claim is the
#: only one made here: ``served_seq`` and ``served_row_hash`` say WHICH ROW to
#: look up, and :func:`verify_served_chain` is what says whether that row is
#: intact. What they close is the other direction — a caller can no longer
#: hold an attestation that is silent about a row which was never written.
SERVED_SEQ_KEY = "served_seq"
SERVED_ROW_HASH_KEY = "served_row_hash"
LEDGER_ERROR_KEY = "ledger_error"

#: The one key a consumer DISPATCHES on, and the reason it exists beside the
#: three above rather than instead of them. ``served_seq: null`` already says
#: "no row" — but it says it by absence, and inferring "this answer is not
#: proven" from a null is the same silent-omission shape the missing keys were,
#: one step removed: a consumer that forgets the null-check reads an unproven
#: record as a proven one and nothing in the record objects. So the status is
#: stated, in a word, on every record.
SERVED_PROOF_KEY = "served_proof"

#: The CLOSED vocabulary of :data:`SERVED_PROOF_KEY`. Exactly two members, and
#: it stays two: every *reason* a row is missing goes in ``ledger_error``,
#: which is an open string. That split is what keeps a consumer's dispatch
#: total forever — ``proof == PROOF_RECORDED`` else "unproven" can never be
#: made wrong by a reason nobody has invented yet.
#:
#: FORWARD COMPATIBILITY, stated rather than assumed. This key is DERIVED:
#: ``PROOF_RECORDED`` iff ``served_seq is not None``, always, with no third
#: state and no new fact of its own. A reader written before it existed reads
#: ``served_seq`` / ``served_row_hash`` / ``ledger_error`` and loses nothing;
#: a reader written after it verifies against the raw ``ledger_error`` string
#: and dispatches on this enum. Both see one truth because there is only one,
#: and :func:`_ledger_fields` is the single place it is spelled.
PROOF_RECORDED = "recorded"
PROOF_UNPROVEN = "unproven"

#: Every attesting surface publishes all four, always. A key that is present
#: only sometimes is a key consumers stop checking, so "no row" is spelled
#: ``served_seq: null`` with a status and a reason beside it rather than by
#: omission.
LEDGER_ATTESTATION_KEYS: tuple[str, ...] = (
    SERVED_SEQ_KEY,
    SERVED_ROW_HASH_KEY,
    LEDGER_ERROR_KEY,
    SERVED_PROOF_KEY,
)

#: ``ledger_error`` for the one non-failure that still writes no row. Distinct
#: from an exception string so an operator can tell "this workspace opted out"
#: from "this workspace is broken" without parsing prose.
LEDGER_DISABLED = "disabled"

#: The width contract on every hex field a row seals, under its historical
#: private name so the four call sites below — and any reader who learned the
#: name here — keep working after the encoding moved to the leaf.
_hex64 = hex64

#: How long an append waits for the ledger lock before giving up, matching
#: ``evidence_objects._APPEND_LOCK_TIMEOUT_SECONDS``. A serving process that
#: cannot take the lock inside this window raises rather than writing, and
#: :func:`attach_served_run` turns that into a ``ledger_error`` on the
#: attestation — never into a silently missing row.
_APPEND_LOCK_TIMEOUT_SECONDS = 30.0

#: How long a loser sleeps between attempts. Ten times tighter than
#: ``FileLock``'s 0.05 s default, and the reason is the call site: the evidence
#: chain's lock guards a governed write, this one sits on the SERVING path,
#: where a 50 ms nap on every contended recall would be a latency regression
#: paid by a reader. The critical section is a tail read plus two small writes,
#: so a loser is normally through on its first or second poll.
_APPEND_LOCK_POLL_SECONDS = 0.005


def _append_lock(workspace: str | Path) -> FileLock:
    """The CROSS-PROCESS lock guarding read-tail-then-append for *workspace*.

    This used to be a module-level ``threading.Lock``, which serialises the
    threads of one interpreter and nothing else. Measured, three processes
    appending 300 rows each into one workspace: 700 rows of an expected 900,
    383 duplicate ``seq`` values, ``verify_served_chain`` red at row 1, and one
    worker killed outright by a torn line that ``_next_link`` could not parse —
    against a single-process control of 300 rows, zero duplicates, chain ok. A
    ledger that forks the moment an MCP server and a ``mm recall`` share a
    workspace is not a ledger of record, and that pairing is the ordinary case
    rather than an exotic one.

    So it takes the same :class:`~mind_mem.mind_filelock.FileLock` the evidence
    chain takes on its store — one class, one set of platform caveats, one
    stale-holder arbitration — rather than a second locking scheme with its own
    bugs. The lock file is ``served.jsonl.lock``, beside the ledger.

    Not reentrant: ``FileLock``'s per-path thread lock is a plain
    ``threading.Lock``, so nothing inside the critical section may take it
    again.
    """
    return FileLock(
        ledger_path(workspace),
        timeout=_APPEND_LOCK_TIMEOUT_SECONDS,
        poll_interval=_APPEND_LOCK_POLL_SECONDS,
    )


def _identity(value: Any) -> Any:
    return value


def _as_list(value: Any) -> list[Any]:
    return list(value)


def _as_ids(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in value)


#: Per-field coercions for the two directions. Every field not named here is
#: ``str`` on the way in and itself on the way out — so declaring a field is
#: all it takes to thread it, and forgetting to extend a table cannot leave a
#: field half-persisted.
_COERCE: Mapping[str, Callable[[Any], Any]] = MappingProxyType({"seq": int, "ids": _as_ids})
_JSONABLE: Mapping[str, Callable[[Any], Any]] = MappingProxyType({"ids": _as_list})


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
        """Serialise to the on-disk shape — exactly the declared fields.

        Derived from the schema rather than listed, so a field added to the
        dataclass cannot reach disk through a writer that knows about it while
        the reader and the hash do not.
        """
        return {f.name: _JSONABLE.get(f.name, _identity)(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ServedRun:
        """Rebuild from a persisted row, refusing an unknown or missing key.

        Also derived from the schema. A hand-written argument list silently
        drops a declared field the author forgot to thread — the value is on
        disk, absent from the object, and therefore absent from every check
        that reads the object. ``str`` is the default coercion, so a new field
        is threaded by declaring it, not by remembering to.
        """
        expected = {f.name for f in fields(cls)}
        if set(row) != expected:
            raise ValueError(f"row keys {sorted(row)} do not match the schema {sorted(expected)}")
        return cls(**{name: _COERCE.get(name, str)(row[name]) for name in expected})


#: The ONE field :func:`row_hash` does not name directly. ``ids`` enters
#: through ``served_digest``, whose agreement with the ids is re-derived on
#: every verification, so editing the ids alone fails there and editing them
#: consistently fails the link. Everything else is covered BY DERIVATION from
#: the schema: hand-listing the hashed fields meant a tenth field could be
#: declared, written to disk and read back while contributing nothing to any
#: hash — an unsealed field inside a tamper-evident row.
_HASH_EXCLUDED = frozenset({"ids"})


def _hashed_values(row: ServedRun) -> tuple[Any, ...]:
    """Every field the row chain seals, in declaration order."""
    return tuple(getattr(row, f.name) for f in fields(row) if f.name not in _HASH_EXCLUDED)


def row_hash(row: ServedRun) -> str:
    """The chain link, **derived** rather than stored — over the whole schema.

    A stored row hash can be rewritten alongside the row it covers; a derived
    one cannot. The field list is derived too, for the same reason one level
    up: a hash whose coverage is a literal is only as current as the last
    author who remembered to extend it.
    """
    return hashlib.sha256(preimage(ROW_TAG, *_hashed_values(row))).hexdigest()


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
    """True unless ``served_ledger.enabled`` is literally ``false``.

    **Default ON since 5.0.2, and this is the whole point.** The ledger shipped
    opt-in, and an opt-in proof is not a proof: "mind-mem can prove what it
    served" was true of a workspace that had turned a flag on and false of
    every other one, which makes it a property of a configuration rather than
    of the product. Nothing here is a judgement about an answer and nothing
    here can reach the scoring path (see this module's header), so the reason
    to leave it off was never safety — it was caution about a file appearing in
    a workspace, and that is the wrong trade against a claim the product makes.

    Exactly one value opts out: ``{"served_ledger": {"enabled": false}}``. An
    absent section, an absent key, a non-dict section and any other value all
    mean ON, so a workspace cannot end up unrecorded by omission.

    A workspace with **no readable ``mind-mem.json``** is the one OFF case, and
    it is not the flag being absent — it is the workspace being absent. There
    is no configured store to serve from, no directory this module has been
    told it may write into, and `verify_cli` reports such a workspace's ledger
    as ``missing`` rather than broken. Named here rather than left implicit:
    a corrupt config therefore *also* reads as off, which is a real (if narrow)
    way to silence the ledger without saying ``false``. Closing it means
    deciding that an unparseable config is a hard error on the recall path,
    which is a bigger ruling than this one and belongs with the config loader.

    Reads the config file directly rather than through the MCP config helper:
    this module must stay importable without pulling the server layer in behind
    it. Now that the default is on, that read happens on every served recall,
    so it was measured rather than argued about: **0.0156 ms per call against
    0.0016 ms for a bare ``os.stat`` of the same file** — 0.014 ms of avoidable
    work on a recall that takes milliseconds. An mtime-keyed cache would buy
    that back and pay for it with a window in which an operator's ``false`` is
    not yet honoured, which is the wrong trade for a number that small. Revisit
    if the recall path ever gets cheap enough for it to matter.
    """
    path = os.path.join(str(workspace), "mind-mem.json")
    try:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(config, dict):
        return False
    section = config.get(CONFIG_SECTION)
    return not (isinstance(section, dict) and section.get("enabled") is False)


def read_served_runs(workspace: str | Path) -> tuple[ServedRun, ...]:
    """Every row on disk, in file order. Empty when the ledger is absent."""
    try:
        with open(ledger_path(workspace), encoding="utf-8") as handle:
            lines = [line for line in handle.read().splitlines() if line.strip()]
    except OSError:
        return ()
    return tuple(ServedRun.from_row(json.loads(line)) for line in lines)


class ServedLedgerCorruptedError(OSError):
    """The ledger cannot be extended without writing over evidence.

    Named after :class:`~mind_mem.audit_chain.AuditChainCorruptedError` and
    subclassing :class:`OSError` for the same two reasons: the condition is a
    statement about the files on disk, and a caller that already narrows to
    ``OSError`` around ledger work keeps catching it instead of a fresh class
    slipping past its handler.

    Raised by :func:`_next_link`, and therefore by
    :func:`append_served_run`. :func:`attach_served_run` turns it into
    ``ledger_error`` + ``served_proof: unproven`` on the attestation, so a
    refusal is never a silently missing row — every recall served over a
    broken ledger says so, on its own record, for as long as the break lasts.

    It is a REFUSAL, not a repair. Rewriting the seal, dropping the orphaned
    rows or renumbering the tail would each destroy the evidence
    :func:`verify_served_chain` needs to name what happened, and repairing
    recorded history is an operator's decision rather than an append's.
    """


def _last_row(workspace: str | Path) -> Optional[ServedRun]:
    """The final row on disk, or ``None`` when the ledger genuinely holds none.

    ``None`` means exactly one thing: no file, or a file with no non-blank
    line. A tail that EXISTS but cannot be read or parsed raises
    :class:`ServedLedgerCorruptedError` instead — collapsing the two is what
    lets an append restart at ``seq 0`` on top of rows it could not read, and
    the sibling ledger's :class:`~mind_mem.audit_chain.AuditChainCorruptedError`
    exists for the same reason.

    Only the last line is parsed. This used to rebuild every previous row
    through :meth:`ServedRun.from_row` to read two fields off the final one,
    which made the cost of one append grow with the length of the ledger —
    tolerable while the feature was opt-in, quadratic over a workspace's
    lifetime now that it is on by default. The bytes are still read (a jsonl
    has no index); the N-1 JSON parses are not.

    Validation is unchanged where it belongs: :func:`verify_served_chain`
    still rebuilds and checks every row. What moved is only *when* an earlier
    corrupt row is noticed — the verifier convicts it, instead of an append
    silently failing on it and quietly recording nothing further.
    """
    path = ledger_path(workspace)
    try:
        with open(path, encoding="utf-8") as handle:
            lines = [line for line in handle.read().splitlines() if line.strip()]
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise ServedLedgerCorruptedError(f"cannot read the served ledger {path}: {exc}") from exc
    if not lines:
        return None
    try:
        return ServedRun.from_row(json.loads(lines[-1]))
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ServedLedgerCorruptedError(
            f"the last line of the served ledger {path} is not a readable row ({exc}); refusing to "
            "append on top of a broken link — run verify_served_chain() to locate the damage"
        ) from exc


def _next_link(workspace: str | Path) -> tuple[int, str]:
    """``(seq, prev_row_hash)`` for the row about to be appended, or REFUSE.

    The whole guard, and the only way to obtain a link — there is deliberately
    no unguarded twin a future caller could reach for instead.

    THE DEFECT THIS CLOSES, measured on the shipped code. Five runs served,
    ``verify_served_chain`` green. ``rm .mind-mem-ledger/served.jsonl`` — the
    verifier convicts it by name: *"the ledger is empty but the recorded head
    is 7b4af0b3… — the rows were removed"*. Then ONE more recall: the append
    found no rows, started a fresh sequence at ``seq 0`` with the genesis
    link, and :func:`_write_row` re-sealed the head over it. Verify: **ok, no
    reason**. The same held for a truncated tail (10 rows, 6 removed, RED,
    one recall, GREEN) and for a deleted seal (rows intact, seal removed, RED,
    one recall, GREEN). A ledger whose tampering is erased by the next
    ordinary recall does not prove what was served; it proves what survived.

    So an append may start a NEW sequence in exactly one state — a workspace
    where neither a row nor a seal has ever existed — and may EXTEND only a
    tail the seal agrees with. Every other state raises.

    The link itself is still derived from the last row **on disk**, never from
    the sidecar: the sidecar is a seal to be checked against the file, and
    appending from it would let an edited seal choose the link for every row
    that follows.

    THE ONE ALLOWED DISAGREEMENT, and the proof it launders nothing. A row is
    appended and *then* the seal is replaced (:func:`_write_row`), so a process
    killed between the two leaves a tail one row ahead of its seal. Refusing
    that would make an ordinary ``SIGKILL`` a permanent one-way door: the
    workspace could never record again, and a recovery path nobody can reach
    is the defect the round trip exists to prevent. It is admitted only when
    the last row NAMES the sealed head as its predecessor — and that state
    cannot be manufactured by a removal. A removal that leaves the file
    otherwise intact (contiguous ``seq`` from 0, every link sound) must leave
    the seal naming a row that is *gone*, which fails both branches below; and
    a removal that leaves the seal one behind the surviving tail must have
    taken a row from earlier in the file, which leaves a ``seq`` gap or a
    broken link that :func:`verify_served_chain` still convicts — advancing
    the head repairs neither. So the branch admits the crash and nothing else.

    What it costs, measured rather than argued: the guard adds one read of the
    ~65-byte seal per append — 0.0112 ms against the append's own 0.6915 ms,
    or 1.6%, on a 550-row ledger. It is also off the answer path entirely,
    since the row is written after ``recall()`` has returned.

    Raises:
        ServedLedgerCorruptedError: the tail is unreadable, or the tail and
            the seal describe two different ledgers.
    """
    last = _last_row(workspace)
    head = _read_head(workspace)
    if last is None:
        if head is None:
            return 0, GENESIS_ROW_HASH
        raise ServedLedgerCorruptedError(
            f"the served ledger holds no row but {HEAD_RELPATH} still seals "
            f"{head or 'a blank value'}: the rows were removed. Refusing to restart the sequence "
            "at 0 — that append would re-seal the head over the deletion and leave "
            "verify_served_chain reporting a clean chain"
        )
    tail = row_hash(last)
    if head == tail:
        return last.seq + 1, tail
    if head and last.prev_row_hash == head:
        # The crash window: the seal lags the tail by exactly one row, and that
        # row names the seal as its predecessor. Extending re-seals a tail
        # nothing was taken from — see this function's docstring for why no
        # removal can reach this state undetected.
        return last.seq + 1, tail
    if head is None:
        raise ServedLedgerCorruptedError(
            f"the served ledger ends at row {last.seq} but {HEAD_RELPATH} is gone: the seal was "
            "removed. Refusing to append — that append would write a fresh seal over an unsealed "
            "tail and turn 'the last row is unsealed' into a clean chain"
        )
    raise ServedLedgerCorruptedError(
        f"the served ledger ends at row {last.seq} ({tail[:12]}…) but {HEAD_RELPATH} seals "
        f"{(head or 'a blank value')[:12]}…, which is neither that row nor its predecessor: rows "
        "were removed or edited. Refusing to append — run verify_served_chain() to locate the break"
    )


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

    ``index_anchor`` is likewise the attestation's value verbatim, so this row
    is only as meaningful as that anchor is. It was not meaningful: until the
    resolver was repointed, it read the field-audit sidecar, which no governed
    write or delete touches, and every row here recorded
    :data:`~mind_mem.recall_attestation.GENESIS_ANCHOR` forever. It now reads
    the head of ``memory/hash_chain_v2.db`` — the ledger the governance gate
    appends to on every mint — normalised to this field's 64-hex width, which
    is why :func:`_hex64` below still holds. That width is a live coupling
    rather than a formality: an anchor of any other width raises here, and the
    recall path's ledger call swallows the exception, so widening the anchor
    without widening this check would silently stop the ledger recording
    anything. See ``recall_attestation._resolve_index_anchor``'s WIDTH note.
    """
    if not ledger_enabled(workspace):
        return None
    served = tuple(str(i) for i in ids)
    if served_set_digest(served) != served_digest:
        raise ValueError("served_digest does not match ids — refusing to record an inconsistent row")
    # The lock file lives beside the ledger, so the directory has to exist
    # before the lock can be taken rather than at the first write. Reaching
    # here means the ledger is enabled, so this creates nothing a disabled
    # workspace would not already have refused above.
    Path(ledger_path(workspace)).parent.mkdir(parents=True, exist_ok=True)
    with _append_lock(workspace):
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


def _ledger_log() -> Any:
    """The serving logger, imported lazily.

    Function-local so this module's EAGER import closure stays free of the
    observability layer — ``tests/test_recall_attestation_v2.py`` walks that
    closure — and so the happy path pays for no import at all: it is reached
    only when there is a failure to report.
    """
    from .observability import get_logger

    return get_logger("served_ledger")


def _ledger_fields(row: Optional[ServedRun], error: Optional[str]) -> dict[str, Any]:
    """The four attestation keys for *row*, or for its absence.

    The ONE place the status is decided, so ``served_proof`` cannot disagree
    with ``served_seq``: both are read off the same ``row``, in one expression,
    and no caller is offered a way to set one without the other.
    """
    return {
        SERVED_SEQ_KEY: None if row is None else row.seq,
        SERVED_ROW_HASH_KEY: None if row is None else row_hash(row),
        LEDGER_ERROR_KEY: error,
        SERVED_PROOF_KEY: PROOF_UNPROVEN if row is None else PROOF_RECORDED,
    }


def attach_served_run(record: Mapping[str, Any], workspace: str | Path, *, ids: Sequence[str]) -> dict[str, Any]:
    """Append this run's row and return *record* plus the keys naming it.

    **The only function a serving surface should call.** Two defects made it
    the entry point rather than :func:`append_served_run`:

    *The row was invisible to the caller.* A recall whose append raised got its
    answer, its attestation and a log line nobody reads — measured: with the
    ledger directory replaced by a plain file, ``recall()`` returned one hit
    and an attestation while ``read_served_runs`` returned nothing, and no key
    in the attestation named a row. A verifier holding that record cannot tell
    "this run was never recorded" from "somebody removed the row". Now it can:
    ``served_proof`` is ``unproven``, ``served_seq`` is ``None``, and
    ``ledger_error`` says why.

    *The six fields were threaded by hand at every call site.* Both surfaces
    copied the same attestation-field-to-ledger-field mapping, so a third one
    could thread ``config_hash`` into ``index_anchor`` and produce a row that
    verifies against itself while naming the wrong pipeline. The mapping now
    exists once, here, and reads the values out of the record the run already
    published — which is also what makes the row and the record incapable of
    being two opinions of one run.

    Returns a NEW dict; *record* is never mutated.

    THE RULING ON A FAILED LEDGER WRITE — decided here because this is the
    only place the choice exists, and a future author will look for it here.

    A run whose row could not be written is served, and its record is marked
    ``served_proof: unproven`` with the failure verbatim in ``ledger_error``.
    It does **not** fail the recall. Three reasons, in the order that decided
    it:

    1. *The ledger is downstream of the answer.* It is appended after
       ``recall()`` has returned, from values the run already published.
       Raising here would destroy an answer that was correctly computed and
       correctly gated, on account of a record ABOUT it — the tail wagging the
       store.
    2. *Fail-closed here is a denial-of-service surface, and a cheap one.*
       Anyone who can make one file unwritable — ``chmod`` the ledger
       directory, fill the disk, leave a stale lock — would take the memory
       offline for every reader. Serving is the property a memory must not
       lose; the recorder failing is a fact to publish, not a reason to stop
       answering. This is the argument that settled it.
    3. *Nothing is hidden by continuing.* The refusal is on the record the
       caller is about to publish, in a two-member vocabulary, on every
       surface, for as long as the break lasts — and since
       :func:`_next_link` now refuses to append over a tampered ledger, a
       break marks EVERY subsequent recall rather than one. A consumer that
       must not act on an unproven answer implements that in one line
       (``record[SERVED_PROOF_KEY] != PROOF_RECORDED``), which is the caller's
       ruling to make, and it can only make it because the field is there.

    The rejected alternative, named so it is not re-litigated silently:
    raising would buy a stronger-sounding claim ("every served answer has a
    row") and pay for it by making the claim false in the only way that
    matters — a workspace that cannot record would serve nothing at all, so
    there would be no answers to prove. What is claimed instead is exactly
    what is true: every row present is intact, and every record says whether
    it has one.
    """
    row: Optional[ServedRun] = None
    error: Optional[str] = None
    try:
        row = append_served_run(
            workspace,
            query_hash=record["query_hash"],
            served_digest=record["results_digest"],
            ids=ids,
            pipeline_hash=record["config_hash"],
            index_anchor=record["index_anchor"],
            scoring_instant=record["scoring_instant"],
        )
        if row is None:
            error = LEDGER_DISABLED
    except Exception as exc:  # noqa: BLE001 — every failure becomes a reported field
        error = f"{type(exc).__name__}: {exc}"
        _ledger_log().warning("served_ledger_append_failed", error=error)
    return {**record, **_ledger_fields(row, error)}


def _write_row(workspace: str | Path, row: ServedRun) -> None:
    """Append the row, then re-anchor the head sidecar. Caller holds the lock."""
    path = Path(ledger_path(workspace))
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row.to_row(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    _write_head(workspace, row_hash(row))


def _write_head(workspace: str | Path, head: str) -> None:
    """Replace the head sidecar in one step — temp file, then ``os.replace``.

    ``open(path, "w")`` truncates first and writes second, so between those two
    the seal on disk is a zero-length file. Any reader in that window sees a
    blank head, and a blank head is not a neutral value here: `_read_head`
    deliberately reports it as an OVERWRITTEN seal, so
    :func:`verify_served_chain` convicts an untouched ledger of tampering. The
    rename is atomic, so no reader ever observes a partial seal.

    What this does NOT close, stated rather than implied: durability. Neither
    the row append nor this replace is ``fsync``-ed, so a power loss can still
    leave the pair inconsistent in either direction — and an ``fsync`` per
    served recall is a cost the serving path should not pay for a record whose
    verifier already reports the inconsistency by name instead of hiding it.

    The temp name carries the pid so a file left behind by a crashed writer is
    never the one this process is renaming.
    """
    final = Path(_head_path(workspace))
    tmp = final.with_name(f"{final.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(head + "\n")
        os.replace(tmp, final)
    finally:
        # A failed write leaves no debris to be mistaken for a seal.
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _read_head(workspace: str | Path) -> Optional[str]:
    """The recorded head, or ``None`` when the sidecar is **absent**.

    Absent and empty are different facts and must stay distinguishable.
    Collapsing both to ``""`` is what let a deleted sidecar read as "no head
    to check against": the caller's truthiness test then skipped the head
    comparison entirely, so removing the seal removed the check with it. A
    present-but-blank file is not a missing seal, it is an overwritten one,
    and it fails the comparison like any other wrong value.
    """
    try:
        with open(_head_path(workspace), encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return None


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


def _locate_break(rows: Sequence[ServedRun], breaks: list[int], stored_head: Optional[str]) -> int:
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
      the edit is behind it. A head that is missing or blank carries no such
      information — falsy here means "cannot disambiguate", never "nothing to
      answer for"; :func:`verify_served_chain` convicts the missing seal on
      its own account.
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
    day. Three passes: the self-checking invariants first (position, digest vs
    ids, ``run_id`` vs its three inputs), then the links, then the head sidecar
    — which seals the fields no invariant and no successor covers, the last
    row's ``index_anchor`` and ``scoring_instant``.

    The sidecar comparison is **unconditional**, and both halves of that matter:

    * it runs when the ledger holds NO rows. A recorded head with nothing left
      to hash means every row was removed, which is the one deletion the row
      chain cannot see — an emptied file has no seq gap and no broken link.
    * it runs when the sidecar is ABSENT. Skipping the check because the seal
      is gone makes deleting the seal the way to unseal the tail, which is
      exactly what the seal exists to prevent.

    The residual hole is named rather than papered over: removing the ledger
    AND the sidecar together leaves a directory indistinguishable from one
    where the ledger never ran, because both records live inside it. Detecting
    that needs an anchor kept somewhere else, which this module does not own.
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
    last = len(rows) - 1
    if stored_head is None:
        if rows:
            return ChainVerdict(
                ok=False,
                rows_checked=last,
                bad_seq=last,
                reason=f"row {last}: the head sidecar is missing — the last row is unsealed",
                head=head,
            )
        return ChainVerdict(ok=True, rows_checked=0, bad_seq=None, reason="", head=head)
    if stored_head != head:
        if not rows:
            recorded = stored_head or "blank"
            return ChainVerdict(
                ok=False,
                rows_checked=0,
                bad_seq=None,
                reason=f"the ledger is empty but the recorded head is {recorded} — the rows were removed",
                head=head,
            )
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
    "LEDGER_ATTESTATION_KEYS",
    "LEDGER_DISABLED",
    "LEDGER_ERROR_KEY",
    "LEDGER_RELPATH",
    "PROOF_RECORDED",
    "PROOF_UNPROVEN",
    "ROW_TAG",
    "RUN_TAG",
    "SERVED_PROOF_KEY",
    "SERVED_ROW_HASH_KEY",
    "SERVED_SEQ_KEY",
    "ChainVerdict",
    "ServedLedgerCorruptedError",
    "ServedRun",
    "append_served_run",
    "attach_served_run",
    "ledger_enabled",
    "ledger_path",
    "read_served_runs",
    "row_hash",
    "run_id",
    "verify_served_chain",
]
