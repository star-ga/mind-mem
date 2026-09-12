"""Fail-closed capability flags: an unenforced property cannot report success (M5).

ROADMAP M5's close condition is a MECHANISM, not a document: "Each audit finding
should close by installing a flag that FAILS CLOSED while unimplemented, so an
unenforced property is a function returning 0 that gates the path -- not a doc
saying the property is aspirational. A caller intending to rely on it must check
and refuse."

The pattern is internal precedent. ``512-mind`` ships
``drift.semantic_mutation_scan_supported() -> u8 { 0 }`` with the reasoning written
into the source: *"An undefined/empty mutation list must NEVER make `equivalent`
true -- that was the forgery-by-absence path this fix closes."*

FORGERY BY ABSENCE is the one failure this module exists to prevent: a check that
could not run returning the same value as a check that passed. :func:`verdict_for`
therefore makes "unsupported" a THIRD state that no supported path can produce, so
"we never verified this" can never be read as "this verified clean".

WHY A FLAG AND NOT A DOC. A doc saying a property is aspirational is read once, by
whoever wrote it. A function returning False gates the path every time, and a
caller who forgets to check gets an exception from :func:`require` rather than a
convenient default. That asymmetry is the whole design.

HOW TO ADD ONE. Add the member, set its flag honestly, and write the reason beside
it. Setting a flag True is a claim that code enforces the property -- if the
enforcement is a prompt asking a model to behave, the honest value is False.

Pure: no clock, no I/O, no subprocess. A capability answer must not depend on when
it was asked.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Capability(str, Enum):
    """Governance properties a caller might want to RELY on.

    A member exists here because someone could reasonably assume the property
    holds. Its flag says whether code actually enforces it.
    """

    #: Pre-write redaction of detected PII/secrets on the governed door.
    PREWRITE_REDACTION = "prewrite_redaction"
    #: Provenance fields can be made mandatory on a governed write.
    PROVENANCE_REQUIRED = "provenance_required"
    #: Export policies genuinely remove sensitive values.
    EXPORT_REDACTION = "export_redaction"
    #: Corpus tamper detection via planted canary blocks.
    CANARY_TRIPWIRE = "canary_tripwire"
    #: Lifecycle losses (DEMOTE/ARCHIVE/FORGET) are chained and replayable.
    LIFECYCLE_EVIDENCE = "lifecycle_evidence"
    #: Contradiction prevented structurally by closed-set slot keys.
    SLOT_UPSERT = "slot_upsert"
    #: Per-actor anomaly detection over write behaviour.
    ACTOR_ANOMALY_DETECTION = "actor_anomaly_detection"
    #: Semantic (paraphrase-tolerant) duplicate detection.
    SEMANTIC_DUPLICATE_DETECTION = "semantic_duplicate_detection"
    #: Signed release manifests covering the corpus, not just release artifacts.
    SIGNED_CORPUS_MANIFEST = "signed_corpus_manifest"


#: capability -> (supported, why). The "why" is not decoration: a False with no
#: stated reason becomes a mystery nobody dares flip, and a True with no stated
#: mechanism is the prompt-shaped claim M5 exists to find.
_FLAGS: dict[Capability, tuple[bool, str]] = {
    Capability.PREWRITE_REDACTION: (
        True,
        "compliance/prewrite.py:143 calls redact() with the detector chain, and "
        "governance.py:345 calls screen() on the governed door — code, not a prompt",
    ),
    Capability.PROVENANCE_REQUIRED: (
        True,
        "v4.provenance.policy=required refuses a write missing provenance "
        "(measured: error='provenance_required'); a malformed policy fails closed",
    ),
    Capability.EXPORT_REDACTION: (
        True,
        "export policies measurably differ: 'full' carries an email through, "
        "'redacted' and 'metadata-only' do not, and all three still export the row",
    ),
    Capability.CANARY_TRIPWIRE: (
        True,
        "canary.check_canaries reports edited, status-changed and VANISHED canaries, "
        "and marks a check over nothing vacuous rather than clean",
    ),
    Capability.LIFECYCLE_EVIDENCE: (
        True,
        "lifecycle_evidence.LifecycleRecorder writes DEMOTE/ARCHIVE/FORGET into both "
        "ledgers and is reached from memory_tiers.demote and compaction",
    ),
    Capability.SLOT_UPSERT: (
        True,
        "upsert_slots refuses a slug outside the closed Slot enum and returns a "
        "supersession PLAN the governed door routes, never a silent overwrite",
    ),
    Capability.ACTOR_ANOMALY_DETECTION: (
        False,
        "NOT IMPLEMENTED, and deliberately: it requires learned per-actor scoring, "
        "which the determinism wedge rules out ('No per-actor learned or anomaly "
        "scoring'). A caller wanting behavioural anomaly detection must refuse, not "
        "assume the canary tripwire covers it — the canary detects a moved corpus, "
        "not a hostile actor",
    ),
    Capability.SEMANTIC_DUPLICATE_DETECTION: (
        False,
        "NOT IMPLEMENTED. gist.py collapses statements differing only in a slot "
        "value, which is LEXICAL; two facts on one topic phrased with entirely "
        "different vocabulary are still seen as distinct. A caller relying on "
        "paraphrase-tolerant dedup must refuse",
    ),
    Capability.SIGNED_CORPUS_MANIFEST: (
        False,
        "NOT IMPLEMENTED for the corpus. Sigstore signing covers RELEASE ARTIFACTS "
        "only, so a verified release says nothing about whether a workspace's blocks "
        "are the ones that were published",
    ),
}


class CapabilityUnsupported(RuntimeError):
    """A caller required a property nothing in this build enforces.

    Raised rather than returning a falsy value, so a caller who forgets to check
    fails loudly instead of proceeding on an assumption.
    """


@dataclass(frozen=True)
class CapabilityVerdict:
    """An observation, qualified by whether anything enforces it.

    ``ok`` is False whenever ``supported`` is False, WHATEVER was observed. That
    single rule is the forgery-by-absence fix: an unenforced property cannot
    report success by being unobservable.
    """

    capability: str
    supported: bool
    ok: bool
    reason: str


def supported(capability: Capability) -> bool:
    """Whether code (not a prompt, not a doc) enforces *capability*.

    Unknown capabilities answer False. A member added to the enum without a flag
    must fail closed rather than inherit a passing default -- that omission is
    exactly the gap M5 is about.
    """
    return _FLAGS.get(capability, (False, ""))[0]


def reason_for(capability: Capability) -> str:
    """Why the flag has the value it has."""
    return _FLAGS.get(capability, (False, "no flag recorded for this capability"))[1]


def require(capability: Capability) -> None:
    """Raise unless *capability* is enforced by code.

    Raises:
        CapabilityUnsupported: always, when unsupported. The message names the
            capability and the reason, so a caller learns what to do instead of
            what merely failed.
    """
    if not supported(capability):
        raise CapabilityUnsupported(
            f"capability {capability.value!r} is NOT enforced by code in this build: "
            f"{reason_for(capability)}. Refusing rather than proceeding — a caller "
            f"relying on an unenforced governance property is the failure this flag "
            f"exists to prevent."
        )


def verdict_for(capability: Capability, *, observed_ok: bool) -> CapabilityVerdict:
    """Qualify *observed_ok* by whether anything enforces *capability*.

    When unsupported, ``ok`` is False regardless of the observation: a check that
    could not run must never return what a passing check returns.
    """
    is_supported = supported(capability)
    return CapabilityVerdict(
        capability=capability.value,
        supported=is_supported,
        ok=bool(observed_ok) and is_supported,
        reason=reason_for(capability),
    )


__all__ = [
    "Capability",
    "CapabilityUnsupported",
    "CapabilityVerdict",
    "reason_for",
    "require",
    "supported",
    "verdict_for",
]
