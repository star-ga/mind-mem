# Copyright 2026 STARGA, Inc.
"""Route redaction events into the tamper-evident ledger.

A redaction nobody can audit is not a governance feature — it is a silent
edit. The moment the pass rewrites (or refuses) a document, the hash
chain gets an entry saying so, and that entry is the only record of the
event that survives.

**What the entry may carry, and why the obvious choice is wrong.** The
chain's ``reason`` and ``fields_changed`` are stored verbatim, so the
temptation is to put the matched values there for a useful audit trail.
That would move every secret the pass just removed into an append-only
file that by design is never rewritten — the redaction would *create*
the durable copy it exists to prevent. So the entry carries:

* ``operation="update_field"`` — a redaction is a content mutation, and
  the ledger already knows that shape;
* ``fields_changed`` — the detector names that fired, never the values;
* ``reason`` — a count-and-kind summary plus the leading bytes of the
  before/after digests, which identifies the document without describing
  its content;
* ``payload`` — the full finding record, which the chain hashes and
  discards, so the entry is a *commitment* to the offsets without
  storing them.

``tests/test_compliance_redaction.py`` pins that: the canary secret is
absent from the ledger file, against a positive control proving the same
search finds the entry that redacted it.

One residual, stated rather than hidden: the digest in ``reason`` is over
the *whole document*, so for a document that is nothing but one secret it
degenerates into a digest of that secret — still only useful for
confirming a guess, never for producing one, but a caller redacting a
bare credential should hand this pass the containing document rather than
the credential on its own.

The chain call is unconditional once redaction is on. There is no
"record only when something was found" shortcut, because "the pass ran
and found nothing" is exactly the claim an auditor needs evidence for.

Copyright STARGA, Inc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..observability import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..audit_chain import AuditEntry
    from .redaction import RedactionResult

__all__ = ["REDACTION_OPERATION", "record_redaction"]

#: The ledger operation a redaction is filed under. Reusing an existing
#: verb rather than minting one keeps every ≤5.0.1 verifier able to read
#: the entry: an unknown operation is refused by ``AuditChain.append``
#: and would be an unreadable row to an older reader.
REDACTION_OPERATION = "update_field"

_log = get_logger("compliance.audit")


def record_redaction(
    workspace: str,
    result: "RedactionResult",
    *,
    target: str,
    agent: str = "",
) -> Optional["AuditEntry"]:
    """Append one redaction event for *target* to the workspace ledger.

    Returns the entry, or ``None`` when the pass did not run (mode
    ``off``) — an OFF path writes nothing, so the disabled build leaves
    the ledger byte-identical to the build that never had the feature.

    Failures propagate. A redaction whose audit record could not be
    written is not a redaction that quietly succeeded: swallowing the
    error would leave a rewritten document with no evidence that
    anything happened to it.
    """
    from ..audit_chain import AuditChain
    from .redaction import MODE_OFF

    if result.mode == MODE_OFF:
        return None

    entry = AuditChain(workspace).append(
        REDACTION_OPERATION,
        target,
        agent=agent,
        reason=result.summary(),
        payload={"schema": "mind-mem/redaction/1", **result.to_dict()},
        fields_changed=sorted(result.counts) or None,
    )
    _log.info(
        "redaction_recorded",
        seq=entry.seq,
        target=entry.target,
        mode=result.mode,
        findings=len(result.findings),
    )
    return entry
