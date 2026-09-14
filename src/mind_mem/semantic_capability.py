"""Explicit capability boundary for semantic answer verification.

Citation membership proves that an answer points at served evidence.  It does
not prove that the prose follows from that evidence.  This module keeps that
distinction in one place until a reviewed semantic verifier exists.
"""

from __future__ import annotations

SEMANTIC_VERIFICATION_NOT_ESTABLISHED = "not_established"


def semantic_entailment_verification_available() -> bool:
    """Return whether semantic entailment verification is actually available.

    No entailment verifier is shipped in the current runtime.  Keeping this
    predicate false makes callers choose an explicit abstention path instead
    of treating citation membership as semantic proof.
    """

    return False


__all__ = [
    "SEMANTIC_VERIFICATION_NOT_ESTABLISHED",
    "semantic_entailment_verification_available",
]
