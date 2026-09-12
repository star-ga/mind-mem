"""Canary blocks: detect corpus poisoning by planting what must NOT move.

ROADMAP (Adversarial / poisoning defense) lists two mechanisms: "per-actor anomaly
detection + canary blocks not yet shipped." Canaries are the half this product can
have WITHOUT breaking its own wedge. Per-actor anomaly detection means learned
per-actor scoring, and the roadmap already rules that out for trust scores -- "No
per-actor learned or anomaly scoring (determinism wedge)". A canary is the
opposite: a fixed, known-good block whose exact fingerprint is recorded once, so a
later deviation is a DETERMINISTIC signal rather than a statistical guess.

WHAT A CANARY DETECTS:

* its content changed  -> an unauthorised edit reached the corpus
* its status changed   -> something demoted or quarantined a known-good block
* it VANISHED          -> the cheapest attack on any tripwire is removing it, so
                          absence is a failure, never a clean sheet

WHAT IT DOES NOT DETECT, stated because a defense believed to do more than it does
is worse than none: an injected block that adds a false claim without touching any
canary. A canary is a tripwire, not a filter. It tells you the corpus moved; it
cannot tell you everything that arrived.

WHY A CANARY MUST DECLARE ITSELF. Recognition is by TAG, not by id prefix. If
``CANARY-`` in an id were sufficient, an attacker could plant ``CANARY-999`` and
its own clean verdict would launder the corpus -- the tripwire would be under the
attacker's control. The tag has to be written by whoever planted it.

Pure: the fingerprint is a hash over recorded fields with no clock, so a check run
tomorrow compares like with like. A fingerprint that drifted with time would make
every canary look tampered-with eventually.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

#: Tag a block must carry to be treated as a canary. Declared, never inferred.
CANARY_TAG = "canary"

#: Fields the fingerprint covers. Deliberately NARROW: a fingerprint over every
#: field would flip on any metadata touch (an access count, a last-read stamp) and
#: drown the real signal in noise until the check got ignored. These four are the
#: ones whose change means the corpus moved.
FINGERPRINT_FIELDS = ("_id", "Statement", "Status", "Tags")

#: Domain separator, so a canary fingerprint can never collide with another
#: hash in this system that happens to cover the same bytes.
_TAG = b"MM_CANARY_v1\x00"


@dataclass(frozen=True)
class CanaryVerdict:
    """Outcome of a canary sweep.

    ``vacuous`` exists because "no canary deviated" reads identically whether the
    corpus is clean or no canary was ever planted. Every other gate in this
    codebase that conflated those two shipped a silent pass, so the distinction is
    a field rather than a convention.
    """

    ok: bool
    changed: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    checked: int = 0
    vacuous: bool = False
    reason: str = ""


def is_canary(block: Mapping[str, Any] | None) -> bool:
    """True when *block* DECLARES itself a canary via its tags."""
    if not block:
        return False
    tags = str(block.get("Tags") or "").lower()
    return CANARY_TAG in [t.strip() for t in tags.replace(";", ",").split(",")]


def canary_fingerprint(block: Mapping[str, Any] | None) -> str:
    """Stable digest over :data:`FINGERPRINT_FIELDS`.

    Length-prefixed rather than joined: with a bare separator, a value containing
    the separator could forge another field's content, which would let an editor
    change the statement while holding the fingerprint fixed -- exactly the
    tamper this function exists to catch.
    """
    h = hashlib.sha256(_TAG)
    for name in FINGERPRINT_FIELDS:
        raw = str((block or {}).get(name) or "").encode("utf-8")
        h.update(str(len(raw)).encode("ascii") + b":" + raw)
    return h.hexdigest()


def check_canaries(
    blocks: Iterable[Mapping[str, Any]],
    baseline: Mapping[str, str],
) -> CanaryVerdict:
    """Compare every canary in *blocks* against its recorded *baseline*.

    *baseline* maps block id -> fingerprint, as produced when the canary was
    planted. A canary present in the baseline but absent from *blocks* is a
    FAILURE: removing the tripwire is the cheapest way past it.

    A canary present but absent from the baseline is NOT a failure -- planting a
    new canary must not require re-baselining before it is legal, or nobody will
    plant one.
    """
    found = {
        str(b.get("_id") or b.get("id") or ""): b for b in (blocks or ()) if is_canary(b)
    }
    changed = tuple(
        sorted(
            bid
            for bid, fp in baseline.items()
            if bid in found and canary_fingerprint(found[bid]) != fp
        )
    )
    missing = tuple(sorted(bid for bid in baseline if bid not in found))
    checked = len(found)

    if not baseline and not checked:
        return CanaryVerdict(
            ok=True,
            checked=0,
            vacuous=True,
            reason=(
                "no canaries planted and no baseline recorded: this check verified "
                "NOTHING. Reported as vacuous rather than clean, because 'no canary "
                "deviated' otherwise reads the same whether the corpus is intact or "
                "the tripwire was never installed."
            ),
        )

    if changed or missing:
        bits = []
        if changed:
            bits.append(f"{len(changed)} canary block(s) changed: {list(changed)}")
        if missing:
            bits.append(
                f"{len(missing)} canary block(s) VANISHED: {list(missing)} — removing "
                f"a tripwire is the cheapest way past it, so absence is a failure"
            )
        return CanaryVerdict(
            ok=False, changed=changed, missing=missing, checked=checked,
            vacuous=False, reason="; ".join(bits),
        )

    return CanaryVerdict(ok=True, checked=checked, vacuous=False)


__all__ = [
    "CANARY_TAG",
    "FINGERPRINT_FIELDS",
    "CanaryVerdict",
    "canary_fingerprint",
    "check_canaries",
    "is_canary",
]
