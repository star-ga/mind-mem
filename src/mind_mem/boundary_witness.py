# Copyright 2026 STARGA, Inc.
"""The recovery boundary is fixed by a governed witness, not by a wall clock.

WHY THIS EXISTS. The first attempt bracketed the boundary against the anchor's
seal time, on the reasoning that companion timestamps are hash-bound and so
cannot be edited after the fact. That is true and insufficient. Hash-binding
gives INTEGRITY -- no retroactive edit -- not ACCURACY: a writer supplies its own
timestamp at append time. A single row appended after the seal but stamped before
it does not merely evade the bracket, it INVERTS it: the true boundary is refused
as "too small" and N+1 is accepted. The check then certifies the post-recovery
admission it existed to exclude.

So the clock is demoted to evidence and a witness becomes the authority.

WHAT A WITNESS IS. An independently established, explicitly operator-authorised
document whose CONTENT carries the boundary -- anchor identity, exact prefix count
and tail, authority, and an explicit trust assumption. The count and tail are read
OUT of it; they are never supplied alongside it.

WHERE THE TRUST ACTUALLY SITS. Not in the digest the caller hands us -- anyone can
digest a claim they authored. The trust root is a digest established OUT OF BAND
under governance (in production, the pre-recovery receipt). The attestation
supplies content; that content must reproduce the governed digest. Changing the
count to overclaim changes the digest, so an arbitrary count is unusable. This is
the difference between "a digest is present" and "the digest binds".

WHAT IT DOES NOT DO. It fixes a boundary under a stated operator trust assumption.
It restores no historic trust, rehabilitates no sealed history, and asserts nothing
about whether the archived rows were ever continuous.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Optional

from .hash_chain_v2 import GENESIS_HASH, HashChainV2
from .preimage import preimage

#: Metadata key under which an attestation carries the witness CONTENT. The
#: content is a claim to be checked; the governed digest it must reproduce comes
#: from governance, never from the record itself.
WITNESS_CONTENT_KEY = "boundary_witness"

#: Version tag for the witness preimage class. A distinct tag keeps a witness
#: digest from ever colliding with an evidence or audit preimage.
WITNESS_TAG = "WITNESS_v1"

#: Authorities a witness may claim. Deliberately a closed set: "whatever the
#: metadata says" is not an authority, it is an assertion.
#:
#: WHAT THIS IS NOT. Matching this string does not AUTHENTICATE anybody. A string
#: in a document proves nothing about who wrote it. What authenticates the
#: witness is the governed digest, which is pinned out of band under operator
#: control; the authority field records, under that operator's trust assumption,
#: WHO the pin is understood to speak for. Treating the string itself as proof of
#: caller identity would be exactly the mistake this module exists to avoid.
RECOGNISED_AUTHORITIES = frozenset({"operator"})

#: Timestamp verdicts. Evidence only -- none of these can establish or overturn
#: a boundary, and none of them may raise.
TS_CORROBORATES = "corroborates"
TS_CONTRADICTS = "contradicts"
TS_NO_EVIDENCE = "no_evidence"

#: Every one of these must be present. ``trust_assumption`` is required
#: DELIBERATELY: the whole point of the witness is that the assumption is stated
#: rather than implied, so a witness that omits it is not a witness.
_REQUIRED = (
    "anchor_id",
    "anchor_hash",
    "prefix_count",
    "prefix_tail",
    "authority",
    "trust_assumption",
)


@dataclass(frozen=True)
class BoundaryWitness:
    """A boundary fixed under an explicit, named authority."""

    anchor_id: str
    anchor_hash: str
    prefix_count: int
    prefix_tail: str
    authority: str
    trust_assumption: str


def witness_digest(content: object) -> str:
    """Canonical digest of a witness's content.

    Uses the shared NUL-separated preimage builder rather than an ad-hoc join:
    it renders bool distinctly from int and cannot be collided by a field value
    containing the separator. A malformed content object has no digest and
    returns "" -- callers compare against a governed digest, and "" matches none.
    """
    w = parse_witness(content)
    if w is None:
        return ""
    pre = preimage(
        WITNESS_TAG,
        w.anchor_id,
        w.anchor_hash,
        w.prefix_count,
        w.prefix_tail,
        w.authority,
        w.trust_assumption,
    )
    return hashlib.sha3_512(pre).hexdigest()


def parse_witness(content: object) -> Optional[BoundaryWitness]:
    """Strictly typed parse. Anything malformed is None, never a partial object."""
    if not isinstance(content, dict):
        return None
    for key in _REQUIRED:
        if key not in content:
            return None
    count = content.get("prefix_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        return None
    for key in ("anchor_id", "anchor_hash", "prefix_tail", "authority"):
        value = content.get(key)
        if not isinstance(value, str) or not value.strip():
            return None
    assumption = content.get("trust_assumption")
    if not isinstance(assumption, str) or not assumption.strip():
        return None
    return BoundaryWitness(
        anchor_id=str(content["anchor_id"]),
        anchor_hash=str(content["anchor_hash"]),
        prefix_count=count,
        prefix_tail=str(content["prefix_tail"]),
        authority=str(content["authority"]),
        trust_assumption=assumption,
    )


#: companion_state outcomes. ABSENT and CORRUPT are DIFFERENT facts: a chain that
#: was never created is not a chain that will not open. Collapsing them is what
#: lets a corrupted database read as "nothing here yet".
COMPANION_ABSENT = "absent"
COMPANION_PRESENT = "present"
COMPANION_CORRUPT = "corrupt"


def companion_status(workspace: str) -> tuple[str, int, str]:
    """(status, length, tail) for the companion chain beside *workspace*.

    Returns COMPANION_ABSENT, COMPANION_PRESENT or COMPANION_CORRUPT. An earlier
    version returned the same (False, 0, GENESIS) tuple for absent AND for an
    unreadable database while its own comment claimed unreadable was not absent
    -- the docstring said one thing and the code did the other, so no caller
    could have told them apart even if it wanted to.
    """
    db = os.path.join(os.path.dirname(os.path.abspath(workspace)), "hash_chain_v2.db")
    if not os.path.isfile(db):
        return (COMPANION_ABSENT, 0, GENESIS_HASH)
    try:
        chain = HashChainV2.open_readonly(db)
        length = chain.length
        if length <= 0:
            return (COMPANION_PRESENT, 0, GENESIS_HASH)
        rows = chain.get_latest(1)
        return (COMPANION_PRESENT, length, rows[-1].entry_hash if rows else GENESIS_HASH)
    except Exception:
        return (COMPANION_CORRUPT, 0, GENESIS_HASH)


def companion_state(workspace: str) -> tuple[bool, int, str]:
    """Back-compatible (present, length, tail).

    Kept so existing callers keep working, but it CANNOT distinguish absent from
    corrupt -- both are False. New code calls :func:`companion_status`.
    """
    status, length, tail = companion_status(workspace)
    return (status == COMPANION_PRESENT, length, tail)


def _prefix_tail(workspace: str, count: int) -> Optional[str]:
    """The tail of the chain's first *count* rows, or None if unavailable."""
    present, length, _ = companion_state(workspace)
    if not present or count > length:
        return None
    if count == 0:
        return GENESIS_HASH
    db = os.path.join(os.path.dirname(os.path.abspath(workspace)), "hash_chain_v2.db")
    try:
        chain = HashChainV2.open_readonly(db)
        rows = chain.get_latest(length - count + 1)
    except Exception:
        return None
    return rows[0].entry_hash if rows else None


def timestamp_consistency(workspace: str, count: int, seal_timestamp: object) -> str:
    """Does the clock CORROBORATE the boundary? Evidence only, and never raises.

    Naive-versus-aware comparison raises TypeError, which is how the previous
    bracket crashed out of a public verifier instead of refusing. Every malformed,
    naive, mixed or absent case resolves to TS_NO_EVIDENCE here. A contradiction
    is reported for operators to look at; it does not and must not move a
    witness-fixed boundary.
    """
    from datetime import datetime, timezone

    def _parse(value: object) -> Optional["datetime"]:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
        # Normalise to aware so a naive/aware pair can never raise on compare.
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(timezone.utc)

    sealed_at = _parse(seal_timestamp)
    if sealed_at is None:
        return TS_NO_EVIDENCE

    present, length, _ = companion_state(workspace)
    if not present or count <= 0 or count > length:
        return TS_NO_EVIDENCE
    db = os.path.join(os.path.dirname(os.path.abspath(workspace)), "hash_chain_v2.db")
    try:
        chain = HashChainV2.open_readonly(db)
        rows = chain.get_latest(length - count + 1)
    except Exception:
        return TS_NO_EVIDENCE
    if not rows:
        return TS_NO_EVIDENCE

    at_boundary = _parse(getattr(rows[0], "timestamp", None))
    if at_boundary is None:
        return TS_NO_EVIDENCE
    if at_boundary > sealed_at:
        return TS_CONTRADICTS
    if len(rows) > 1:
        after = _parse(getattr(rows[1], "timestamp", None))
        if after is not None and after <= sealed_at:
            return TS_CONTRADICTS
    return TS_CORROBORATES


def verify_witness(
    workspace: str,
    anchor_id: str,
    anchor_hash: str,
    content: object,
    governed_digest: str,
    *,
    authorities: frozenset = RECOGNISED_AUTHORITIES,
) -> tuple[Optional[BoundaryWitness], str]:
    """The witness, or the reason it does not fix a boundary.

    *governed_digest* is the out-of-band trust root. It is NOT taken from
    *content*; that is the whole point. Every clause fails closed.
    """
    if not isinstance(governed_digest, str) or not governed_digest.strip():
        return None, "no governed witness digest is configured, so no boundary is fixed"

    witness = parse_witness(content)
    if witness is None:
        return None, "the boundary witness is malformed"

    # The binding, first: content that cannot reproduce the governed digest is
    # not this witness, whatever else it says.
    if not _constant_time_equal(witness_digest(content), governed_digest):
        return None, "the boundary witness does not match its governed digest"

    if witness.anchor_id != anchor_id:
        return None, "the boundary witness fixes a different anchor id"
    if witness.anchor_hash != anchor_hash:
        return None, "the boundary witness names this anchor with the wrong evidence hash"

    # EXACT match, no strip() and no case-folding. Normalising "operator " into
    # "operator" would silently rewrite a claim into a different one that
    # verifies -- the same mistake as normalising a traversal path away. The
    # digest binds the exact string, so an authority that differs by whitespace
    # is a different witness and must be refused, not repaired.
    if witness.authority not in authorities:
        return None, "the boundary witness carries no recognised authority"

    status, length, _ = companion_status(workspace)
    if status == COMPANION_CORRUPT:
        return None, "the companion hash chain will not open, so no prefix can be bound"
    if status == COMPANION_ABSENT:
        return None, "the companion hash chain is absent, so no prefix can be bound"
    if witness.prefix_count > length:
        return None, (f"the witness binds {witness.prefix_count} companion entries but the chain holds {length}")
    actual = _prefix_tail(workspace, witness.prefix_count)
    if actual is None:
        return None, "the bound companion prefix could not be read"
    if actual != witness.prefix_tail:
        return None, "the bound companion prefix does not end in the witnessed tail"

    return witness, ""


def _constant_time_equal(a: str, b: str) -> bool:
    """Compare digests without leaking position through timing."""
    import hmac

    return hmac.compare_digest(a or "", b or "")


# ---------------------------------------------------------------------------
# The out-of-band pin
# ---------------------------------------------------------------------------

#: Config section holding governed witness pins, in the workspace's
#: ``mind-mem.json`` -- the SAME configuration mechanism the ordinary verifier
#: already reads for its backend and alert settings. Deliberately not a new
#: parallel authority: one config file, one governance surface.
#:
#:     {"recovery": {"boundary_witness_pins": {"<anchor id>": "<digest>"}}}
#:
#: Keyed BY ANCHOR ID so a pin issued for one anchor can never bind another, and
#: so a workspace can carry more than one without them being interchangeable.
CONFIG_SECTION = "recovery"
CONFIG_PINS_KEY = "boundary_witness_pins"


def governed_witness_pin(workspace: str, anchor_id: str) -> str:
    """The operator-pinned witness digest for *anchor_id*, or "".

    Read straight out of ``mind-mem.json`` with :mod:`json`, matching
    :func:`mind_mem.anchoring.configured_backend`: the verifier is stdlib-only
    and must not construct a store to answer a question about a config file.

    An unreadable or malformed config answers "" -- which REFUSES, because a
    boundary with no pin is a boundary with no authority. Absent config is not
    permissive here; it is fail-closed by construction.

    *workspace* is the workspace DIRECTORY, or a path inside it (the evidence
    ledger path is what callers usually hold).
    """
    import json  # noqa: PLC0415 -- stdlib, kept off this module's import closure

    if not isinstance(anchor_id, str) or not anchor_id.strip():
        return ""
    # ONE authority. An earlier version searched the ledger's own directory
    # FIRST and then the parent, and fell through to the next candidate on a
    # parse error -- so a mind-mem.json dropped beside the ledger silently
    # outranked the workspace config, and corrupting the canonical file
    # promoted the rogue one. The workspace config is the only authority, and a
    # malformed one refuses rather than deferring.
    directory = workspace if os.path.isdir(workspace) else os.path.dirname(os.path.abspath(workspace))
    # The ledger lives in <workspace>/memory/; the config belongs to the workspace.
    if os.path.basename(directory) == "memory":
        directory = os.path.dirname(directory)
    try:
        with open(os.path.join(directory, "mind-mem.json"), encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ""
    if not isinstance(config, dict):
        return ""
    section = config.get(CONFIG_SECTION)
    if not isinstance(section, dict):
        return ""
    pins = section.get(CONFIG_PINS_KEY)
    if not isinstance(pins, dict):
        return ""
    pin = pins.get(anchor_id)
    return pin if isinstance(pin, str) and pin.strip() else ""
