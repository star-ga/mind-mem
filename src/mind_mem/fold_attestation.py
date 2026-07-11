#!/usr/bin/env python3
"""Evidence-anchored fold-equivalence ("folded==ran") attestation. Zero new deps.

``recompaction.recompact_cluster`` collapses a cluster of related blocks to a
byte *fixed point*: it re-reads and re-compresses until a pass returns the same
bytes it was given. That fixed point is only meaningful because the compressor
is pure w.r.t. its inputs (``temperature=0`` + a pinned ``seed``), so the whole
fold is a deterministic function of ``(input cluster, compressor)``.

This module turns that determinism into a *third-party-checkable claim*. It
anchors the fold's digests into the tamper-evident audit hash chain as a
``fold_attest`` entry, and provides a verifier that, given the attestation plus
the original cluster, **re-runs one deterministic recompaction and diffs** —
proving the recorded ``output_digest`` is exactly what re-running the fold
produces (*folded == ran*). It verifies; it never searches for a compressor or
config that would make the digests match.

Two bindings make the claim precise:

1. **Compressor identity + version.** The preimage binds *which* compressor
   produced the fold (``compressor_id``) and the exact behaviour fingerprint
   (``compressor_version`` — the pinned knobs + post-processing pipeline
   version). A re-verify is therefore against the exact function that produced
   it; a verifier holding a different compressor is refused, not silently
   diffed against the wrong ground truth.

2. **No wall-clock, no randomness.** The attestation preimage contains only the
   fold's own content-derived digests, iteration count, and compressor
   identity. It is a pure function of the fold — building it twice from the same
   ``RecompactionResult`` yields byte-identical bytes. (The anchoring audit
   entry carries a timestamp, but that lives in the *chain* record, never in the
   attestation preimage.)

HARD RAIL — the chain is **tamper-evident, not signed.** The attestation hash is
a plain SHA-256 over the preimage: it detects an internally inconsistent record,
and the audit chain's ``prev_hash`` linkage detects any post-hoc edit of an
anchored entry. Neither is a cryptographic signature — anyone can recompute a
SHA-256. Authenticated signing (Ed25519 / ML-DSA) is separate, deferred work.

    from mind_mem.audit_chain import AuditChain
    from mind_mem.recompaction import recompact_cluster
    from mind_mem.fold_attestation import attest_fold, verify_fold

    result = recompact_cluster(cluster, compressor=compressor)
    att = attest_fold(AuditChain(workspace), result, compressor, agent="dream_cycle")
    # ... later, a third party ...
    outcome = verify_fold(att, cluster, compressor)
    assert outcome.ok   # folded == ran
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Any

from .audit_chain import AuditChain
from .observability import get_logger, metrics
from .preimage import preimage
from .recompaction import (
    RecompactionConfig,
    RecompactionResult,
    cluster_digest,
    recompact_cluster,
)

_log = get_logger("fold_attestation")

# Version tag for the fold-equivalence preimage class. Distinct from AUDIT_v1 /
# EV_v1 so an attestation preimage can never collide with an audit-entry or
# evidence-object preimage even when their bodies coincide.
FOLD_ATTEST_TAG = "FOLD_ATTEST_v1"

# Audit-chain operation name for an anchored attestation. Registered in
# audit_chain.VALID_OPERATIONS.
FOLD_ATTEST_OP = "fold_attest"


def _seq_digest(items: tuple[str, ...]) -> str:
    """Unambiguous SHA-256 over an *ordered* sequence of strings.

    Length-prefixed, then each item folded in as its fixed-width (32-byte)
    SHA-256 digest, so neither element boundaries nor ordering can be forged: a
    trajectory ``("a", "bc")`` and ``("ab", "c")`` hash differently, and
    reordering the sequence changes the result (trajectory order is
    load-bearing — it is the *path* the fold took to its fixed point).
    """
    h = hashlib.sha256()
    h.update(str(len(items)).encode("ascii"))
    h.update(b"\x00")
    for it in items:
        h.update(hashlib.sha256(it.encode("utf-8")).digest())
    return h.hexdigest()


def compressor_identity(compressor: Any) -> tuple[str, str]:
    """Resolve a compressor's canonical ``(id, version)`` for attestation binding.

    The compressor is injected into ``recompact_cluster`` as a bare ``Callable``,
    so identity is discovered through an explicit ``fold_identity`` hook rather
    than by type-sniffing: a compressor exposes either a ``fold_identity()``
    method or a ``fold_identity`` ``(id, version)`` attribute. The built-in
    ``OllamaCompressor`` and ``EchoCompressor`` both carry it.

    A compressor with no hook raises ``ValueError`` — refusing to attest a fold
    whose producing function cannot be named is the honest failure: an
    unlabelled attestation could never be re-verified against "the exact
    compressor that produced it".
    """
    hook = getattr(compressor, "fold_identity", None)
    if hook is None:
        name = getattr(compressor, "__qualname__", None) or type(compressor).__name__
        raise ValueError(
            f"compressor {name!r} exposes no fold_identity — cannot bind it into a "
            "fold-equivalence attestation. Add a fold_identity() method or a "
            "(id, version) fold_identity attribute so a verifier can re-run "
            "against the exact compressor that produced the fold."
        )
    ident = hook() if callable(hook) else hook
    try:
        cid, cver = ident
    except (TypeError, ValueError) as exc:  # not a 2-tuple
        raise ValueError(f"compressor fold_identity must be an (id, version) pair, got {ident!r}") from exc
    return str(cid), str(cver)


def _attestation_preimage(
    *,
    compressor_id: str,
    compressor_version: str,
    input_digest: str,
    output_digest: str,
    iterations: int,
    trajectory_digest: str,
    source_digest: str,
) -> bytes:
    """Build the tagged, NUL-separated fold-equivalence preimage.

    Deterministic by construction — every field is a content-derived digest, a
    count, or the compressor identity. No timestamp, no randomness, so the same
    fold always yields the same preimage (and hence the same attestation hash).
    """
    return preimage(
        FOLD_ATTEST_TAG,
        compressor_id,
        compressor_version,
        input_digest,
        output_digest,
        iterations,
        trajectory_digest,
        source_digest,
    )


@dataclass(frozen=True)
class FoldAttestation:
    """A hash-anchored *folded==ran* claim over one recompaction fixed point.

    Every field is bound into ``attestation_hash`` via the ``FOLD_ATTEST_v1``
    preimage, so mutating any one of them (``output_digest`` in particular)
    without recomputing the hash makes the record internally inconsistent —
    detectable with :meth:`is_internally_consistent`. Tamper-evidence of the
    *anchored* record additionally comes from the audit chain's ``prev_hash``
    linkage; this record-level check is the first, cheaper line.

    ``source_ids`` is carried verbatim for readability; the preimage binds their
    order-sensitive ``source_digest`` so the readable list cannot be swapped
    without invalidating the hash.
    """

    compressor_id: str
    compressor_version: str
    input_digest: str
    output_digest: str
    iterations: int
    trajectory_digest: str
    source_digest: str
    source_ids: tuple[str, ...]
    attestation_hash: str
    schema: str = field(default=FOLD_ATTEST_TAG)

    def recompute_hash(self) -> str:
        """Recompute the attestation hash from the bound fields (no I/O)."""
        return hashlib.sha256(
            _attestation_preimage(
                compressor_id=self.compressor_id,
                compressor_version=self.compressor_version,
                input_digest=self.input_digest,
                output_digest=self.output_digest,
                iterations=self.iterations,
                trajectory_digest=self.trajectory_digest,
                source_digest=self.source_digest,
            )
        ).hexdigest()

    def is_internally_consistent(self) -> bool:
        """True iff the stored hash matches its own preimage (constant-time compare)."""
        return hmac.compare_digest(self.recompute_hash(), self.attestation_hash)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the audit payload / JSON export (sorted, stable)."""
        return {
            "schema": self.schema,
            "compressor_id": self.compressor_id,
            "compressor_version": self.compressor_version,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "iterations": self.iterations,
            "trajectory_digest": self.trajectory_digest,
            "source_digest": self.source_digest,
            "source_ids": list(self.source_ids),
            "attestation_hash": self.attestation_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FoldAttestation:
        """Reconstruct from a serialized dict (e.g. an anchored audit payload)."""
        return cls(
            compressor_id=d["compressor_id"],
            compressor_version=d["compressor_version"],
            input_digest=d["input_digest"],
            output_digest=d["output_digest"],
            iterations=int(d["iterations"]),
            trajectory_digest=d["trajectory_digest"],
            source_digest=d["source_digest"],
            source_ids=tuple(d.get("source_ids", ())),
            attestation_hash=d["attestation_hash"],
            schema=d.get("schema", FOLD_ATTEST_TAG),
        )


def build_attestation(result: RecompactionResult, *, compressor_id: str, compressor_version: str) -> FoldAttestation:
    """Build (but do not anchor) a :class:`FoldAttestation` from a fold result.

    Pure: no I/O, no clock, no randomness. Building twice from the same
    ``result`` + identity yields an equal attestation with the same hash.
    """
    trajectory_digest = _seq_digest(result.trajectory)
    source_digest = _seq_digest(result.source_ids)
    attestation_hash = hashlib.sha256(
        _attestation_preimage(
            compressor_id=compressor_id,
            compressor_version=compressor_version,
            input_digest=result.input_digest,
            output_digest=result.output_digest,
            iterations=result.iterations,
            trajectory_digest=trajectory_digest,
            source_digest=source_digest,
        )
    ).hexdigest()
    return FoldAttestation(
        compressor_id=compressor_id,
        compressor_version=compressor_version,
        input_digest=result.input_digest,
        output_digest=result.output_digest,
        iterations=result.iterations,
        trajectory_digest=trajectory_digest,
        source_digest=source_digest,
        source_ids=result.source_ids,
        attestation_hash=attestation_hash,
    )


def attest_fold(
    chain: AuditChain,
    result: RecompactionResult,
    compressor: Any,
    *,
    agent: str = "",
    reason: str = "",
    target: str | None = None,
) -> FoldAttestation:
    """Anchor a fold-equivalence attestation into the tamper-evident audit chain.

    Derives the compressor identity from the same ``compressor`` object that
    produced ``result`` (so the attest and verify sides agree by construction),
    builds the attestation, and appends it as a ``fold_attest`` entry. The audit
    chain hashes the full attestation payload into the entry and chains the entry
    into the ledger — that linkage, not the attestation hash alone, is what makes
    a later edit of the record detectable.

    Args:
        chain: The workspace audit chain to append to.
        result: A converged :class:`RecompactionResult`.
        compressor: The compressor that produced ``result``; must expose
            ``fold_identity`` (see :func:`compressor_identity`).
        agent: Identity of who ran the fold (recorded on the audit entry).
        reason: Governance justification (recorded on the audit entry).
        target: Audit target path. Defaults to ``recompaction/<input_digest>``.

    Returns:
        The anchored :class:`FoldAttestation`.
    """
    compressor_id, compressor_version = compressor_identity(compressor)
    attestation = build_attestation(result, compressor_id=compressor_id, compressor_version=compressor_version)
    tgt = target if target is not None else f"recompaction/{result.input_digest}"
    chain.append(
        FOLD_ATTEST_OP,
        tgt,
        agent=agent,
        reason=reason or "folded==ran fold-equivalence attestation",
        payload=attestation.to_dict(),
    )
    metrics.inc("fold_attestations_anchored")
    _log.info(
        "fold_attest_anchored",
        cluster=result.input_digest[:12],
        compressor=compressor_id,
        iterations=result.iterations,
    )
    return attestation


@dataclass(frozen=True)
class FoldVerification:
    """Outcome of a re-run-and-diff fold-equivalence check.

    ``ok`` is the single verdict; ``reason`` names the first failing gate (or
    ``"folded == ran"`` on success). ``recomputed_output_digest`` is populated
    once the re-run actually happens, so a caller can see what the fold produced
    on this machine even when it fails to match.
    """

    ok: bool
    reason: str
    recomputed_output_digest: str | None = None


def verify_fold(
    attestation: FoldAttestation,
    blocks: list[dict[str, Any]],
    compressor: Any,
    *,
    config: RecompactionConfig | None = None,
) -> FoldVerification:
    """Re-run one deterministic recompaction and assert *folded == ran*.

    This is a verifier, not a search: it re-runs the fold exactly once with the
    supplied compressor and diffs the resulting ``output_digest`` against the
    attested one. It never tries alternative compressors or configs to *make*
    the digests agree.

    Gates, in order (first failure short-circuits):

    1. **Record consistency** — the attestation hash matches its own preimage.
       Catches a tampered bound field (``output_digest``, ``compressor_version``,
       …) without running the compressor at all.
    2. **Compressor identity** — the verifier's compressor must carry the exact
       ``(id, version)`` the attestation bound. Re-running a different function
       is refused rather than diffed.
    3. **Input binding** — the supplied cluster must hash to the attested
       ``input_digest``. Folding a different cluster is refused.
    4. **Re-run and diff** — recompact the cluster and require the fresh
       ``output_digest`` to equal the attested one. Equality is *folded == ran*.

    Determinism is load-bearing for gate 4: the compressor must be pure (see
    ``recompaction`` / ``compressors`` docstrings), or a stateful re-run will not
    reproduce the fixed point.
    """
    if not attestation.is_internally_consistent():
        metrics.inc("fold_verifications_failed")
        return FoldVerification(False, "attestation_hash does not match its bound preimage (record tampered)")

    got_id, got_version = compressor_identity(compressor)
    if not (hmac.compare_digest(got_id, attestation.compressor_id) and hmac.compare_digest(got_version, attestation.compressor_version)):
        metrics.inc("fold_verifications_failed")
        return FoldVerification(
            False,
            f"compressor identity mismatch: attested {attestation.compressor_id}@{attestation.compressor_version}, "
            f"verifier holds {got_id}@{got_version}",
        )

    got_input = cluster_digest(blocks)
    if not hmac.compare_digest(got_input, attestation.input_digest):
        metrics.inc("fold_verifications_failed")
        return FoldVerification(False, "input cluster digest does not match the attested input_digest")

    result = recompact_cluster(blocks, compressor=compressor, config=config)
    if not hmac.compare_digest(result.output_digest, attestation.output_digest):
        metrics.inc("fold_verifications_failed")
        return FoldVerification(
            False,
            "re-run output_digest does not match the attested output_digest (folded != ran)",
            result.output_digest,
        )

    metrics.inc("fold_verifications_ok")
    _log.info("fold_verify_ok", cluster=attestation.input_digest[:12], iterations=result.iterations)
    return FoldVerification(True, "folded == ran", result.output_digest)


__all__ = [
    "FOLD_ATTEST_OP",
    "FOLD_ATTEST_TAG",
    "FoldAttestation",
    "FoldVerification",
    "attest_fold",
    "build_attestation",
    "compressor_identity",
    "verify_fold",
]
