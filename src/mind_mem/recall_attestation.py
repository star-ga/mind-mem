#!/usr/bin/env python3
"""Per-run recall attestation — runtime evidence of *how* an answer was produced.

A recall returns ranked blocks, but nothing in that list attests *which path*
produced them: was it the two-leg BM25+vector fusion the ``hybrid`` label
implies, or BM25-only because the vector leg was unavailable? The
``.degraded`` marker (419bee5) was the first member of this class — a runtime
signal about *how* an answer was produced. This module generalises it into a
full **recall attestation**: a per-run runtime artifact recording which legs
actually ran, the effective config hash, any degradation, and the index/state
anchor the run observed.

LOAD-BEARING WEDGE — three rails, each with a test in
``tests/test_recall_attestation.py``:

1. **DERIVABLE, never self-declared.** Every field is recomputed from a
   *recorded* run signal — the ``RecallResults.degraded`` marker, per-hit
   provenance flags (``_retrieval_source``, ``_graph_hop``), the backend's own
   ``vector_enabled`` / ``vector_available`` flags, the pipeline-probe config
   SHA (:func:`mind_mem.pipeline_hash.current_pipeline_hash`, *reused* — this
   module never invents a config hash), and the audit-chain head. None of it is
   a string a producer typed. We do **not** copy an external "trust tier"
   granted by prefixing a text file; a verdict here is derived from the artifact
   the run left behind or it does not exist.

2. **NEVER PERSISTED.** The attestation is returned in the recall response /
   envelope and is *never* written back to the block store or anchored into the
   audit chain. This is the deliberate asymmetry with
   :mod:`mind_mem.fold_attestation`, which *anchors* a fold digest: a fold is a
   durable content fact; a recall verdict is a runtime observation about one
   query, and persisting it would be storing a credibility score — exactly what
   the spec refuses. There is intentionally **no** ``attest_*`` / ``anchor_*`` /
   ``append`` / ``write`` function in this module. Even reading the index anchor
   is done through a read-only file peek (:func:`_resolve_index_anchor`) that
   creates nothing.

3. **DETERMINISTIC.** The ``RECALL_ATTEST_v1`` preimage carries only
   content-derived digests, the config hash, the anchor, and a count — no
   wall-clock, no randomness. Building an attestation twice from the same run
   state yields byte-identical bytes (and hence an identical ``attestation_hash``).

HARD RAIL — like every hash in this codebase, ``attestation_hash`` is a plain
SHA-256 over the preimage: it detects an internally inconsistent record, not a
forged one (anyone can recompute a SHA-256). It is **tamper-evident, not
signed**; authenticated signing (Ed25519 / ML-DSA) is separate, deferred work.

    from mind_mem.recall_attestation import derive_recall_attestation

    att = derive_recall_attestation(
        results,                       # the RecallResults from HybridBackend.search
        vector_requested=hb.vector_enabled,
        vector_available=hb.vector_available,
        config_hash=current_pipeline_hash(workspace),
        index_anchor=_resolve_index_anchor(workspace),
    )
    envelope["attestation"] = att.to_dict()   # surfaced, never stored
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from typing import Any

from .observability import get_logger, metrics
from .preimage import preimage

_log = get_logger("recall_attestation")

# Version tag for the recall-attestation preimage class. Distinct from
# FOLD_ATTEST_v1 / AUDIT_v1 / EV_v1 so a recall-attestation preimage can never
# collide with a fold-attestation, audit-entry, or evidence-object preimage even
# when their bodies coincide.
RECALL_ATTEST_TAG = "RECALL_ATTEST_v1"

# Canonical leg names. A recall may run some subset of these.
LEG_BM25 = "bm25"  # lexical base leg — runs on every recall path
LEG_VECTOR = "vector"  # dense embedding leg — only the hybrid backend runs it
LEG_GRAPH = "graph"  # multi-hop graph expansion — leaves ``_graph_hop`` on hits
LEG_HYBRID = "hybrid"  # the two-leg fusion mode — present iff bm25 AND vector ran

# Sentinel anchor when no audit chain exists yet. Mirrors audit_chain._GENESIS_HASH
# (SHA-256 width of zeros) so an absent chain is a stable, recomputable value
# rather than an empty string that could be confused with "unresolved".
GENESIS_ANCHOR = "0" * 64

# Provenance of an attestation's leg/config values (rail 1 hardening, Finding 3).
# ``derived`` — legs were recomputed from the recorded ``RecallResults`` run
# state by :func:`derive_recall_attestation` (the sanctioned path); the value is
# a function of the artifact the run left behind, not a caller's assertion.
# ``asserted`` — the values were supplied verbatim by a caller to
# :func:`build_recall_attestation` (the raw builder). The marker is bound into
# the preimage, so a caller cannot mint a self-consistent attestation over
# fabricated (leg, config) and pass it off as ``derived``: flipping the field
# breaks the hash. Consumers that require provenance-guaranteed legs check
# ``derivation == DERIVATION_DERIVED``.
DERIVATION_DERIVED = "derived"
DERIVATION_ASSERTED = "asserted"


# ---------------------------------------------------------------------------
# Derivation helpers — every one reads a *recorded* run signal, never a claim.
# ---------------------------------------------------------------------------


def _seq_digest(items: tuple[str, ...]) -> str:
    """Unambiguous SHA-256 over an *ordered* sequence of strings.

    Length-prefixed, then each item folded in as its fixed-width (32-byte)
    SHA-256 digest, so neither element boundaries nor ordering can be forged.
    Mirrors ``fold_attestation._seq_digest`` for the same reason: a leg set
    ``("bm25", "vector")`` must hash distinctly from ``("bm25vector",)``.
    """
    h = hashlib.sha256()
    h.update(str(len(items)).encode("ascii"))
    h.update(b"\x00")
    for it in items:
        h.update(hashlib.sha256(it.encode("utf-8")).digest())
    return h.hexdigest()


def _marker_digest(marker: dict[str, str] | None) -> str:
    """Deterministic SHA-256 over a ``.degraded`` marker dict (``""`` when None).

    The marker ``{leg, reason, [variants_degraded, variants_total]}`` is
    serialised with ``sort_keys`` so key order cannot change the digest, then
    hashed. Binding this digest into the preimage means the readable
    ``degraded`` field carried on the attestation cannot be swapped without
    invalidating the hash.
    """
    if not marker:
        return ""
    canonical = json.dumps(marker, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _has_provenance(results: Any, key: str, *, equals: str | None = None) -> bool:
    """True iff any hit dict carries ``key`` (optionally equal to ``equals``).

    Reads per-hit provenance stamped by the recall pipeline itself
    (``_retrieval_source`` for the pgvector BM25-fallback label, ``_graph_hop``
    for graph-walked hits) — recorded signals, never a caller's assertion.
    """
    try:
        it = iter(results)
    except TypeError:  # pragma: no cover — defensive
        return False
    for r in it:
        if not isinstance(r, dict):
            continue
        if key not in r:
            continue
        if equals is None:
            if r.get(key):
                return True
        elif r.get(key) == equals:
            return True
    return False


def derive_legs(
    results: Any,
    *,
    vector_requested: bool,
    vector_available: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Derive ``(legs_ran, legs_degraded)`` from the *actual* recall run state.

    Reads only recorded signals — never a passed-in leg string:

    * ``bm25`` — the lexical base leg. It executes on every recall path
      (hybrid, sqlite-FTS, full-scan all resolve lexical matches), so it is
      always in ``legs_ran``.
    * ``vector`` — only the hybrid backend runs a dense leg, and only when the
      operator requested it (``vector_requested``). It *ran* iff it was
      requested, the backend was available, the ``.degraded`` marker does not
      name it, and no hit carries the pgvector ``bm25_fallback`` provenance.
      Otherwise, if it was requested, it is reported *degraded* (requested but
      not served) — the ``hybrid`` label without a vector leg is the exact
      silent lie ``.degraded`` was introduced to expose.
    * ``graph`` — ran iff any hit was graph-walked (carries ``_graph_hop``).
    * ``hybrid`` — the fusion *mode*; present iff both ``bm25`` and ``vector``
      actually contributed (a genuine two-leg fusion).

    Any leg named in the recorded ``.degraded`` marker (``leg`` may be
    comma-joined across query variants) is folded into ``legs_degraded`` too, so
    the multi-query union marker (31e8af4) is honoured rather than re-derived.
    """
    degraded_marker = getattr(results, "degraded", None)
    ran: set[str] = {LEG_BM25}
    degraded: set[str] = set()

    # Fold in whatever the recorded marker already names (comma-joined legs
    # from the multi-query union path are split back out).
    if isinstance(degraded_marker, dict):
        for leg in str(degraded_marker.get("leg", "")).split(","):
            leg = leg.strip()
            if leg:
                degraded.add(leg)

    # Vector leg: derived from recorded backend flags + recorded provenance.
    marker_names_vector = LEG_VECTOR in degraded
    pg_fallback = _has_provenance(results, "_retrieval_source", equals="bm25_fallback")
    if vector_requested:
        vector_served = vector_available and not marker_names_vector and not pg_fallback
        if vector_served:
            ran.add(LEG_VECTOR)
        else:
            degraded.add(LEG_VECTOR)
    # A vector leg that was never requested must not appear as degraded even if
    # a stray marker leaked one in — degradation is only meaningful relative to
    # what the run asked for.
    elif not vector_requested and LEG_VECTOR in degraded and not (marker_names_vector or pg_fallback):
        degraded.discard(LEG_VECTOR)

    # Graph leg: recorded per-hit provenance.
    if _has_provenance(results, "_graph_hop"):
        ran.add(LEG_GRAPH)

    # Hybrid is the fusion mode, not a leg you request: it happened iff both
    # base legs contributed.
    if LEG_BM25 in ran and LEG_VECTOR in ran:
        ran.add(LEG_HYBRID)

    return tuple(sorted(ran)), tuple(sorted(degraded))


def _resolve_index_anchor(workspace: str) -> str:
    """Read the audit-chain head hash for *workspace* — read-only, creates nothing.

    Deliberately does **not** instantiate :class:`~mind_mem.audit_chain.AuditChain`
    (whose ``__init__`` ``makedirs`` the audit dir), because deriving an
    attestation must have zero side effects on the store (rail 2). Peeks the
    last non-empty line of ``chain.jsonl`` directly; returns
    :data:`GENESIS_ANCHOR` when the chain is absent, empty, or unreadable.
    """
    chain_path = os.path.join(os.path.abspath(workspace), ".mind-mem-audit", "chain.jsonl")
    if not os.path.isfile(chain_path):
        return GENESIS_ANCHOR
    last_line = ""
    try:
        with open(chain_path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    last_line = stripped
    except OSError:
        return GENESIS_ANCHOR
    if not last_line:
        return GENESIS_ANCHOR
    try:
        entry = json.loads(last_line)
        head = entry.get("entry_hash") if isinstance(entry, dict) else None
    except (json.JSONDecodeError, ValueError):
        return GENESIS_ANCHOR
    return str(head) if head else GENESIS_ANCHOR


# ---------------------------------------------------------------------------
# The attestation record
# ---------------------------------------------------------------------------


def _attestation_preimage(
    *,
    legs_ran_digest: str,
    legs_degraded_digest: str,
    config_hash: str,
    degraded_digest: str,
    index_anchor: str,
    result_count: int,
    derivation: str,
) -> bytes:
    """Build the tagged, NUL-separated recall-attestation preimage.

    Deterministic by construction — every field is a content-derived digest,
    the reused config hash, the index anchor, a count, or the derivation-
    provenance marker. No timestamp, no randomness, so the same run state always
    yields the same preimage (and hence the same attestation hash). Binding
    ``derivation`` here means a caller-``asserted`` record cannot be relabelled
    ``derived`` without invalidating the hash (Finding 3).
    """
    return preimage(
        RECALL_ATTEST_TAG,
        legs_ran_digest,
        legs_degraded_digest,
        config_hash,
        degraded_digest,
        index_anchor,
        result_count,
        derivation,
    )


@dataclass(frozen=True)
class RecallAttestation:
    """A per-run, runtime-only record of *how* a recall produced its answer.

    Every field is bound into ``attestation_hash`` via the ``RECALL_ATTEST_v1``
    preimage, so mutating any one of them without recomputing the hash makes the
    record internally inconsistent — detectable with
    :meth:`is_internally_consistent`. This record is **never persisted** (rail
    2): it lives in the recall response and is discarded when the response is.

    ``degraded`` carries the existing ``.degraded`` ``{leg, reason}`` marker
    verbatim for readability; the preimage binds its order-independent
    ``_marker_digest`` so the readable dict cannot be swapped without
    invalidating the hash.
    """

    legs_ran: tuple[str, ...]
    legs_degraded: tuple[str, ...]
    config_hash: str
    degraded: dict[str, str] | None
    index_anchor: str
    result_count: int
    attestation_hash: str
    schema: str = field(default=RECALL_ATTEST_TAG)
    #: Whether the legs/config were derived from recorded run state
    #: (:data:`DERIVATION_DERIVED`) or asserted by a caller
    #: (:data:`DERIVATION_ASSERTED`). Bound into ``attestation_hash``.
    derivation: str = field(default=DERIVATION_ASSERTED)

    def recompute_hash(self) -> str:
        """Recompute the attestation hash from the bound fields (no I/O)."""
        return hashlib.sha256(
            _attestation_preimage(
                legs_ran_digest=_seq_digest(self.legs_ran),
                legs_degraded_digest=_seq_digest(self.legs_degraded),
                config_hash=self.config_hash,
                degraded_digest=_marker_digest(self.degraded),
                index_anchor=self.index_anchor,
                result_count=self.result_count,
                derivation=self.derivation,
            )
        ).hexdigest()

    def is_internally_consistent(self) -> bool:
        """True iff the stored hash matches its own preimage (constant-time compare)."""
        return hmac.compare_digest(self.recompute_hash(), self.attestation_hash)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the recall envelope / JSON response (stable field order)."""
        return {
            "schema": self.schema,
            "legs_ran": list(self.legs_ran),
            "legs_degraded": list(self.legs_degraded),
            "config_hash": self.config_hash,
            "degraded": self.degraded,
            "index_anchor": self.index_anchor,
            "result_count": self.result_count,
            "derivation": self.derivation,
            "attestation_hash": self.attestation_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RecallAttestation:
        """Reconstruct from a serialized dict (e.g. an envelope round-trip)."""
        return cls(
            legs_ran=tuple(d.get("legs_ran", ())),
            legs_degraded=tuple(d.get("legs_degraded", ())),
            config_hash=d["config_hash"],
            degraded=d.get("degraded"),
            index_anchor=d["index_anchor"],
            result_count=int(d["result_count"]),
            attestation_hash=d["attestation_hash"],
            schema=d.get("schema", RECALL_ATTEST_TAG),
            derivation=d.get("derivation", DERIVATION_ASSERTED),
        )


def build_recall_attestation(
    *,
    legs_ran: tuple[str, ...],
    legs_degraded: tuple[str, ...],
    config_hash: str,
    degraded: dict[str, str] | None,
    index_anchor: str,
    result_count: int,
    derivation: str = DERIVATION_ASSERTED,
) -> RecallAttestation:
    """Build a :class:`RecallAttestation` from already-derived recorded values.

    Pure: no I/O, no clock, no randomness. Building twice from the same values
    yields an equal attestation with the same hash. Leg tuples are normalised
    (deduped + sorted) so the digest is order-stable regardless of caller input
    order.

    TRUST BOUNDARY (Finding 3): the ``legs_ran`` / ``legs_degraded`` /
    ``config_hash`` values are taken **verbatim from the caller** — this builder
    does not derive them from run state and cannot vouch for them. It therefore
    stamps ``derivation=DERIVATION_ASSERTED`` by default, marking the record as
    caller-asserted. Only :func:`derive_recall_attestation`, which recomputes the
    legs from a recorded ``RecallResults``, may pass
    ``derivation=DERIVATION_DERIVED``. The marker is hash-bound, so an asserted
    record cannot be relabelled derived without breaking internal consistency.
    Callers minting an attestation from untrusted input must not pass
    ``DERIVATION_DERIVED``.
    """
    legs_ran_n = tuple(sorted(set(legs_ran)))
    legs_degraded_n = tuple(sorted(set(legs_degraded)))
    if result_count < 0:
        raise ValueError("result_count must be >= 0")
    if derivation not in (DERIVATION_DERIVED, DERIVATION_ASSERTED):
        raise ValueError(f"derivation must be {DERIVATION_DERIVED!r} or {DERIVATION_ASSERTED!r}, got {derivation!r}")
    attestation_hash = hashlib.sha256(
        _attestation_preimage(
            legs_ran_digest=_seq_digest(legs_ran_n),
            legs_degraded_digest=_seq_digest(legs_degraded_n),
            config_hash=config_hash,
            degraded_digest=_marker_digest(degraded),
            index_anchor=index_anchor,
            result_count=result_count,
            derivation=derivation,
        )
    ).hexdigest()
    return RecallAttestation(
        legs_ran=legs_ran_n,
        legs_degraded=legs_degraded_n,
        config_hash=config_hash,
        degraded=degraded,
        index_anchor=index_anchor,
        result_count=result_count,
        attestation_hash=attestation_hash,
        derivation=derivation,
    )


def derive_recall_attestation(
    results: Any,
    *,
    vector_requested: bool,
    vector_available: bool,
    config_hash: str,
    index_anchor: str = GENESIS_ANCHOR,
) -> RecallAttestation:
    """Derive a :class:`RecallAttestation` from a completed recall's run state.

    This is the primary entry point. It reads the recorded signals off
    ``results`` (the ``.degraded`` marker + per-hit provenance) and the recorded
    backend flags, derives the legs, folds in the degraded marker, and binds the
    *reused* ``config_hash`` (from the pipeline probe — this function never
    invents one) and ``index_anchor``. Nothing is written anywhere.

    Because the legs are recomputed from the recorded run state here (not taken
    from a caller), the produced record is stamped
    ``derivation=DERIVATION_DERIVED`` (Finding 3). The one remaining
    caller-supplied value is ``config_hash``: the sanctioned callers
    (:func:`derive_recall_attestation_for_workspace`, the MCP recall path) resolve
    it from :func:`mind_mem.pipeline_hash.current_pipeline_hash`, never from
    untrusted input. A caller that cannot vouch for ``config_hash`` should use
    :func:`build_recall_attestation` (which stamps ``asserted``) rather than this
    function.

    Args:
        results: The recall result list — a ``RecallResults`` (carrying
            ``.degraded`` + per-hit flags) on the hybrid path, or a plain list
            on the BM25-only path.
        vector_requested: The backend's recorded ``vector_enabled`` flag.
        vector_available: The backend's recorded ``vector_available`` flag.
        config_hash: The pipeline-probe config SHA
            (:func:`mind_mem.pipeline_hash.current_pipeline_hash`), reused.
        index_anchor: The audit-chain head / index snapshot hash
            (:func:`_resolve_index_anchor`), or :data:`GENESIS_ANCHOR`.

    Returns:
        A runtime :class:`RecallAttestation` (never persisted).
    """
    legs_ran, legs_degraded = derive_legs(
        results,
        vector_requested=vector_requested,
        vector_available=vector_available,
    )
    degraded_marker = getattr(results, "degraded", None)
    degraded = degraded_marker if isinstance(degraded_marker, dict) else None
    try:
        result_count = len(results)
    except TypeError:  # pragma: no cover — defensive
        result_count = 0
    att = build_recall_attestation(
        legs_ran=legs_ran,
        legs_degraded=legs_degraded,
        config_hash=config_hash,
        degraded=degraded,
        index_anchor=index_anchor,
        result_count=result_count,
        # Sanctioned path: legs were recomputed from the recorded run state
        # above, so this record is provenance-guaranteed, not caller-asserted.
        derivation=DERIVATION_DERIVED,
    )
    metrics.inc("recall_attestations_derived")
    if legs_degraded:
        metrics.inc("recall_attestations_degraded")
    _log.info(
        "recall_attestation_derived",
        legs_ran=",".join(att.legs_ran),
        legs_degraded=",".join(att.legs_degraded),
        config_hash=att.config_hash[:16],
        result_count=att.result_count,
    )
    return att


def derive_recall_attestation_for_workspace(
    results: Any,
    workspace: str,
    *,
    vector_requested: bool,
    vector_available: bool,
) -> RecallAttestation:
    """Convenience wrapper: resolve ``config_hash`` + ``index_anchor`` from *workspace*.

    Reuses :func:`mind_mem.pipeline_hash.current_pipeline_hash` for the config
    SHA and :func:`_resolve_index_anchor` for the chain head. Both are
    read-only; this wrapper writes nothing (rail 2). On any failure resolving
    the config hash it falls back to :data:`GENESIS_ANCHOR`-style empty binding
    rather than raising — an attestation with an unresolved config hash is
    honest, a crashed recall is not.
    """
    try:
        from .pipeline_hash import current_pipeline_hash

        config_hash = current_pipeline_hash(workspace)
        if not isinstance(config_hash, str):  # pragma: no cover — overload guard
            config_hash = ""
    except Exception as exc:  # pragma: no cover — defensive; recall must not fail on attestation
        _log.warning("recall_attestation_config_hash_failed", error=str(exc))
        config_hash = ""
    index_anchor = _resolve_index_anchor(workspace)
    return derive_recall_attestation(
        results,
        vector_requested=vector_requested,
        vector_available=vector_available,
        config_hash=config_hash,
        index_anchor=index_anchor,
    )


__all__ = [
    "DERIVATION_ASSERTED",
    "DERIVATION_DERIVED",
    "GENESIS_ANCHOR",
    "LEG_BM25",
    "LEG_GRAPH",
    "LEG_HYBRID",
    "LEG_VECTOR",
    "RECALL_ATTEST_TAG",
    "RecallAttestation",
    "build_recall_attestation",
    "derive_legs",
    "derive_recall_attestation",
    "derive_recall_attestation_for_workspace",
]
