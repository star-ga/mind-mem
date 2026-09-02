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

3. **DETERMINISTIC — and complete enough to mean it.** The
   ``RECALL_ATTEST_v2`` preimage carries content-derived digests, the config
   hash, the anchor, the query digest, the served ids in rank order, the
   derivation marker, and the run's ``scoring_instant``. No randomness, and no
   *hidden* clock read: the instant is recorded, not sampled behind the
   caller's back. Building an attestation twice from the same run state yields
   byte-identical bytes (and hence an identical ``attestation_hash``).

   This rail used to read "no wall-clock in the preimage", which was true of the
   record and false of the run it attested, and it described the answer only by
   a ``result_count``. Both halves of that were wrong in the same direction.
   The scoring path read the clock — ``date_score``'s ramp, the calibration
   window, the temporal hard filter — so two runs of the same corpus and query
   that ranked differently produced one identical hash; and because only the
   cardinality was bound, so did two runs that served *entirely different
   blocks*. A record that cannot distinguish two different served answers
   asserts a reproducibility it does not have, which is worse than not
   attesting at all.

   Two bindings close it. ``scoring_instant`` — the last hidden *input* — makes
   the guarantee replayable: an attested run is reproduced by passing its date
   back to ``recall()``. ``results_digest`` — the served ids in rank order,
   through the same order-sensitive :func:`seq_digest` as the leg tuples —
   makes it complete against the *output*: any two runs that served a different
   set, or the same set in a different order, hash differently whatever the
   cause, including a hidden input nobody has found yet. The ids are a fact of
   serving rather than a verdict, and nothing is written, so rail 2 holds.

   The v2 bump closes what those two left, and it closes the same class of
   hole a third and fourth time. A fingerprint of a pure function has to bind
   every input, and the **query** was not one: two different questions
   answered with the same ranked list produced one identical hash, so the
   record could not say what had been asked. And the **schema** was a sibling
   field, bound nowhere — the preimage's domain separator was a module
   constant, so a holder could relabel a record, including *downward* to the
   weaker layout, and it stayed internally consistent. Both now sit in the
   preimage: :func:`query_hash` as its own slot, ``schema`` as the tag itself.

   Both went in **under a new tag**, which is the only honest way to do it. A
   value carried beside the preimage is unbound by definition — anyone holding
   the attestation can swap it, so it attests nothing; and editing the layout
   under the old name would leave two incompatible records answering to one
   version string. The tag is a domain separator whose entire job is to make a
   layout change visible, so a layout change moves it.
   :func:`verify_recall_attestation` accepts this tag and no other.

   What the preimage must NEVER bind is the other side of the run: how many
   blocks were withheld, which ones, or any per-item score. Those are
   judgments about content rather than facts about the answer, and a judgment
   that reaches an emitted value is a credibility score leaking out of the
   store — the thing rail 2 exists to prevent. The served ids in rank order
   are the output itself, and a commitment that does not bind its output
   commits to nothing.

   KNOWN LIMITS, stated rather than implied. (a) The digest binds the served
   ids and their order, **not the scores**: two runs that serve the same blocks
   in the same order with different score *values* still collide. (b) The
   corpus is bound only through ``index_anchor``, which is
   :data:`GENESIS_ANCHOR` on a workspace with no audit chain — so what the
   record pins is the answer, not the store it came from. (c) ``result_count``
   and ``served_ids`` are supplied separately by
   :func:`build_recall_attestation`'s callers and are not cross-checked, so a
   hand-built record may claim a count its digest does not carry;
   :func:`derive_recall_attestation` always derives both from the same hit
   list. None of these is a hidden clock; all are stated scope.

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
        query=query,                   # bound as a digest, never as text
    )
    envelope["attestation"] = att.to_dict()   # surfaced, never stored

The serialized form carries one derived key, ``query_id``: the run identity a
client passes back to ``report_outcome(query_id=…)``. It is
:func:`mind_mem.recall_digests.run_id` over three fields the record already
binds, so publishing it mints no second identity and creates no import edge to
a ledger — the encoding lives in the leaf both sides depend on. That is the
whole of RA.1's residual: the right-hand side of the join already accepted an
id, and until now nothing put one in the caller's hands.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .observability import get_logger, metrics
from .preimage import preimage
from .recall_digests import marker_digest, query_hash, run_id, seq_digest, served_set_digest
from .scoring_instant import format_scoring_instant, resolve_scoring_instant

_log = get_logger("recall_attestation")

# Version tag for the recall-attestation preimage class. Distinct from
# FOLD_ATTEST_v1 / AUDIT_v1 / EV_v1 so a recall-attestation preimage can never
# collide with a fold-attestation, audit-entry, or evidence-object preimage even
# when their bodies coincide.
#
# v2 (this bump) added two bindings and changed the served-set encoding, so it
# is a DIFFERENT preimage class and takes a different name. The retired tag is
# not defined anywhere in this package — not as a constant, not as a fallback,
# not as an accepted value: a tag a producer can still reach is a downgrade
# target, and "verify accepts either" would make the version stamp decorative.
# It survives only in ``tests/test_recall_attestation_v2.py``, which asserts it
# is never emitted and that :func:`verify_recall_attestation` refuses it.
RECALL_ATTEST_TAG = "RECALL_ATTEST_v2"

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
    # A vector leg that was never requested must not appear as degraded just
    # because a stray marker leaked one in — degradation is only meaningful
    # relative to what the run asked for. ``pg_fallback`` is the one signal
    # that survives an unrequested leg: it is recorded per-hit provenance
    # saying a dense leg really was attempted server-side and served BM25.
    #
    # The condition used to also require ``not marker_names_vector``, which is
    # by construction true here (``marker_names_vector`` IS ``LEG_VECTOR in
    # degraded`` at this point, and nothing outside the ``vector_requested``
    # branch adds to ``degraded``), so the discard could never run and the
    # comment described behaviour the code did not have.
    elif LEG_VECTOR in degraded and not pg_fallback:
        degraded.discard(LEG_VECTOR)

    # Graph leg: recorded per-hit provenance.
    if _has_provenance(results, "_graph_hop"):
        ran.add(LEG_GRAPH)

    # Hybrid is the fusion mode, not a leg you request: it happened iff both
    # base legs contributed.
    if LEG_BM25 in ran and LEG_VECTOR in ran:
        ran.add(LEG_HYBRID)

    return tuple(sorted(ran)), tuple(sorted(degraded))


def _served_ids(results: Any) -> tuple[str, ...]:
    """The block ids a run served, **in the order it served them**.

    A recorded run output, not a caller's claim (rail 1): it reads ``_id`` off
    the very hit dicts the response carries. Order is preserved and duplicates
    are kept, because the thing being attested is the ranking — collapsing it
    to a set would put the collision straight back.

    A hit with no ``_id`` contributes ``""`` rather than being dropped, so the
    positions of its neighbours are still bound.
    """
    try:
        it = iter(results)
    except TypeError:  # pragma: no cover — defensive
        return ()
    return tuple(str(r.get("_id", "") or "") if isinstance(r, dict) else "" for r in it)


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
    tag: str,
    legs_ran_digest: str,
    legs_degraded_digest: str,
    config_hash: str,
    degraded_digest: str,
    index_anchor: str,
    query_hash: str,
    result_count: int,
    results_digest: str,
    derivation: str,
    scoring_instant: str,
) -> bytes:
    """Build the tagged, NUL-separated recall-attestation preimage.

    Deterministic by construction — every field is a content-derived digest,
    the reused config hash, the index anchor, the served answer, the
    derivation-provenance marker, or the run's scoring instant. No randomness
    and no *hidden* clock read, so the same run state always yields the same
    preimage (and hence the same attestation hash). Binding ``derivation`` here
    means a caller-``asserted`` record cannot be relabelled ``derived`` without
    invalidating the hash (Finding 3).

    Two of these fields close the completeness hole that made the pre-seam
    record dishonest, and they close different halves of it:

    * ``results_digest`` describes the **answer** — the served block ids in
      rank order, through the same order-sensitive :func:`seq_digest` the leg
      tuples use, so ``(A, B)`` and ``(B, A)`` are distinct. It sits next to
      ``result_count``, which it subsumes for hashing purposes and which is
      kept only because it is the readable form. Without it the preimage
      described the served set by cardinality alone: two runs that served no
      block in common, or the same two blocks in opposite order, produced one
      identical hash.
    * ``scoring_instant`` describes the **input** that decides every recency
      term, in exactly the class of ``config_hash`` and ``index_anchor``.
      Without it a run was not replayable: "today" moved, the ranking moved,
      and nothing in the record said so.

    Neither is a verdict and neither is written anywhere, so binding them does
    not weaken rail 2. ``scoring_instant`` is serialized as a bare
    ``YYYY-MM-DD`` UTC date: ten fixed ASCII bytes, no time component and no
    offset suffix, so a run's hash is stable for the whole day and the envelope
    value round-trips for replay. A second-precision timestamp here would churn
    the hash on every single call.
    """
    return preimage(
        tag,
        legs_ran_digest,
        legs_degraded_digest,
        config_hash,
        degraded_digest,
        index_anchor,
        query_hash,
        result_count,
        results_digest,
        derivation,
        scoring_instant,
    )


@dataclass(frozen=True)
class RecallAttestation:
    """A per-run, runtime-only record of *how* a recall produced its answer.

    Every field is bound into ``attestation_hash`` via the ``RECALL_ATTEST_v2``
    preimage — ``schema`` included, as the preimage tag — so mutating any one of
    them without recomputing the hash makes the record internally inconsistent,
    detectable with :meth:`is_internally_consistent`. There is no field on this
    record that the hash does not cover; a sibling value would be forgeable and
    would therefore attest nothing. :attr:`query_id` is not a counterexample and
    not an exception to that rule: it is a ``@property``, stored nowhere and
    recomputed from three hashed fields on every access, so there is no value
    for a forger to move. This record is **never persisted** (rail 2): it lives
    in the recall response and is discarded when the response is.

    ``degraded`` carries the existing ``.degraded`` ``{leg, reason}`` marker
    verbatim for readability; the preimage binds its order-independent
    ``marker_digest`` so the readable dict cannot be swapped without
    invalidating the hash.
    """

    legs_ran: tuple[str, ...]
    legs_degraded: tuple[str, ...]
    config_hash: str
    degraded: dict[str, str] | None
    index_anchor: str
    result_count: int
    #: :func:`~mind_mem.recall_digests.served_set_digest` of the served block
    #: ids, in rank order — order-sensitive and duplicate-preserving. This is
    #: what lets the record distinguish two different served *answers*, rather
    #: than only two answers of different length.
    #:
    #: It is deliberately NOT :func:`seq_digest`, which owns the leg tuples.
    #: This docstring said ``seq_digest`` while the code called
    #: ``served_set_digest``, and a ledger author reading it would have
    #: concluded the served-set ledger needed a different encoding and minted a
    #: second one. It is the SAME value ``served_ledger`` stores as
    #: ``served_digest``: one object, one encoding, one owner.
    results_digest: str
    #: :func:`query_hash` of the question this run answered.
    query_hash: str
    #: The UTC date the run's recency layer scored against, ``YYYY-MM-DD``.
    #: Hash-bound, so replaying a run means passing this value back to
    #: ``recall(scoring_instant=...)``.
    scoring_instant: str
    attestation_hash: str
    schema: str = field(default=RECALL_ATTEST_TAG)
    #: Whether the legs/config were derived from recorded run state
    #: (:data:`DERIVATION_DERIVED`) or asserted by a caller
    #: (:data:`DERIVATION_ASSERTED`). Bound into ``attestation_hash``.
    derivation: str = field(default=DERIVATION_ASSERTED)

    def recompute_hash(self) -> str:
        """Recompute the attestation hash from the bound fields (no I/O).

        ``schema`` is passed as the preimage *tag*, which is what makes the
        version stamp a bound field rather than a sibling: relabel the record
        and the domain separator moves with it, so the hash no longer matches.
        """
        return hashlib.sha256(
            _attestation_preimage(
                tag=self.schema,
                legs_ran_digest=seq_digest(self.legs_ran),
                legs_degraded_digest=seq_digest(self.legs_degraded),
                config_hash=self.config_hash,
                degraded_digest=marker_digest(self.degraded),
                index_anchor=self.index_anchor,
                query_hash=self.query_hash,
                result_count=self.result_count,
                results_digest=self.results_digest,
                derivation=self.derivation,
                scoring_instant=self.scoring_instant,
            )
        ).hexdigest()

    @property
    def query_id(self) -> str:
        """The run identity a client reports outcomes against (RA.1's join key).

        ``SHA256(MM_RUN_v1\\0 ‖ query_hash ‖ results_digest ‖ config_hash)``,
        computed by :func:`mind_mem.recall_digests.run_id` — the same encoding
        and the same owner the served-set ledger stores under ``run_id``, so a
        client holding a recall envelope and the ledger holding a row for that
        run land on one value without either reaching the other.

        **Derived, and that is why it may sit on the record at all.** Every
        stored field here is bound into ``attestation_hash``, deliberately:
        an unbound sibling would be forgeable and would therefore attest
        nothing. This is not a sibling. It is a pure function of three bound
        fields, recomputed on every access and never stored, so forging it
        means moving ``query_hash``, ``results_digest`` or ``config_hash`` —
        which breaks the hash. :meth:`from_dict` re-derives it and refuses a
        serialized dict whose value disagrees, so it cannot be edited in
        transit either.

        ``""`` — never a guess — when any of the three inputs is not a
        64-character hex digest. That happens on a real degraded path:
        ``derive_recall_attestation_for_workspace`` binds ``config_hash=""``
        when the pipeline probe fails, and an attestation with an unresolved
        config hash cannot name a run. An id we cannot mint is absent, and the
        consumer (``accountability_views.run_precision``) already reports an
        unjoinable credit row by name rather than dropping it.
        """
        try:
            return run_id(
                query_hash=self.query_hash,
                served_digest=self.results_digest,
                pipeline_hash=self.config_hash,
            )
        except (ValueError, TypeError):
            return ""

    def is_internally_consistent(self) -> bool:
        """True iff the stored hash matches its own preimage (constant-time compare).

        Total by construction. Now that ``schema`` is the preimage tag, a
        record carrying an empty or non-ascii tag makes the preimage builder
        refuse it — and a *predicate* that raises on a malformed record is
        unusable at a trust boundary, which is the only place it is called.
        Unbuildable is not consistent, so it answers False.
        """
        try:
            recomputed = self.recompute_hash()
        except (ValueError, TypeError, UnicodeEncodeError):
            return False
        return hmac.compare_digest(recomputed, self.attestation_hash)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the recall envelope / JSON response (stable field order).

        ``query_id`` is the one key here that is not a stored field. It is
        emitted because this dict IS the recall envelope's ``attestation``
        object, and RA.1's residual was precisely that a client holding an
        envelope had no id to pass to ``report_outcome(query_id=…)`` — the
        right-hand side of the join already accepted one. Publishing the
        derived value closes that without minting a second identity: see
        :attr:`query_id`.
        """
        return {
            "schema": self.schema,
            "legs_ran": list(self.legs_ran),
            "legs_degraded": list(self.legs_degraded),
            "config_hash": self.config_hash,
            "degraded": self.degraded,
            "index_anchor": self.index_anchor,
            "result_count": self.result_count,
            "results_digest": self.results_digest,
            "query_hash": self.query_hash,
            "query_id": self.query_id,
            "derivation": self.derivation,
            "scoring_instant": self.scoring_instant,
            "attestation_hash": self.attestation_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RecallAttestation:
        """Reconstruct from a serialized dict (e.g. an envelope round-trip).

        Raises:
            ValueError: the ``schema`` is not this preimage version — an older
                tag, a newer one, or none at all. There is deliberately no
                dual-tag path: a parser that accepts two layouts is a
                downgrade target, and a record whose tag disagrees with its
                fields is not a record of this class.
            ValueError: the dict carries the right tag but is missing
                ``scoring_instant`` (the recency input), ``results_digest``
                (the served answer) or ``query_hash`` (the question).
                Reviving it with a guessed value would yield a record that
                merely reports ``is_internally_consistent() is False`` with no
                explanation, so the boundary refuses it by name instead.
            ValueError: the dict carries a ``query_id`` that disagrees with
                the one its own bound fields derive. ``query_id`` is emitted
                by :meth:`to_dict` and is a pure function of three hashed
                fields, so a disagreement means the value was edited in
                transit. Silently recomputing it would honour the record
                while discarding the evidence that someone rewrote the join
                key; a record whose published identity is not its identity is
                refused. A dict with no ``query_id`` at all is accepted — that
                is an envelope from before RA.1's residual closed, and it
                claims nothing to disagree with.
        """
        schema = d.get("schema")
        if schema != RECALL_ATTEST_TAG:
            raise ValueError(
                f"attestation schema {schema!r} is not {RECALL_ATTEST_TAG!r}: this preimage class "
                "accepts exactly one layout, and every hash emitted under another tag is "
                "unrecomputable here by design"
            )
        missing = [k for k in ("scoring_instant", "results_digest", "query_hash") if k not in d]
        if missing:
            raise ValueError(
                f"attestation dict has no {' and no '.join(repr(k) for k in missing)}: it predates "
                "the determinism seam, and every hash emitted before it is unrecomputable by design"
            )
        record = cls(
            legs_ran=tuple(d.get("legs_ran", ())),
            legs_degraded=tuple(d.get("legs_degraded", ())),
            config_hash=d["config_hash"],
            degraded=d.get("degraded"),
            index_anchor=d["index_anchor"],
            result_count=int(d["result_count"]),
            results_digest=str(d["results_digest"]),
            query_hash=str(d["query_hash"]),
            scoring_instant=str(d["scoring_instant"]),
            attestation_hash=d["attestation_hash"],
            schema=RECALL_ATTEST_TAG,
            derivation=d.get("derivation", DERIVATION_ASSERTED),
        )
        if "query_id" in d and str(d["query_id"]) != record.query_id:
            raise ValueError(
                "attestation query_id does not derive from its own query_hash / results_digest / "
                "config_hash: the published run identity was edited in transit"
            )
        return record


def build_recall_attestation(
    *,
    legs_ran: tuple[str, ...],
    legs_degraded: tuple[str, ...],
    config_hash: str,
    degraded: dict[str, str] | None,
    index_anchor: str,
    result_count: int,
    query: str,
    served_ids: tuple[str, ...] = (),
    derivation: str = DERIVATION_ASSERTED,
    scoring_instant: date | str | None = None,
) -> RecallAttestation:
    """Build a :class:`RecallAttestation` from already-derived recorded values.

    Pure given its arguments: no I/O, no randomness. Building twice from the
    same values yields an equal attestation with the same hash. Leg tuples are
    normalised (deduped + sorted) so the digest is order-stable regardless of
    caller input order.

    ``served_ids`` are the block ids the run served, **in rank order**. Unlike
    the leg tuples they are deliberately *not* sorted or deduped: their order
    is the thing being attested, so ``(A, B)`` must hash differently from
    ``(B, A)``. It is what makes the record able to tell two different served
    answers apart at all.

    ``query`` is the question the run answered. It is **required**, with no
    default, because a fingerprint of a pure function has to bind every input
    and a defaulted one binds a constant — a caller who forgot would silently
    mint exactly the record this version exists to retire. Only its
    :func:`query_hash` is stored; the text never enters the record.

    ``scoring_instant`` should be the instant the run *actually scored with*,
    so the record can distinguish two differently-ranked runs and replay either.
    Omitting it resolves today-in-UTC, matching ``recall()``'s own default —
    correct for a run that also took the default, and the one place a clock is
    read here.

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
    instant = format_scoring_instant(resolve_scoring_instant(scoring_instant))
    results_digest = served_set_digest(served_ids)
    q_digest = query_hash(query)
    attestation_hash = hashlib.sha256(
        _attestation_preimage(
            tag=RECALL_ATTEST_TAG,
            legs_ran_digest=seq_digest(legs_ran_n),
            legs_degraded_digest=seq_digest(legs_degraded_n),
            config_hash=config_hash,
            degraded_digest=marker_digest(degraded),
            index_anchor=index_anchor,
            query_hash=q_digest,
            result_count=result_count,
            results_digest=results_digest,
            derivation=derivation,
            scoring_instant=instant,
        )
    ).hexdigest()
    return RecallAttestation(
        legs_ran=legs_ran_n,
        legs_degraded=legs_degraded_n,
        config_hash=config_hash,
        degraded=degraded,
        index_anchor=index_anchor,
        result_count=result_count,
        results_digest=results_digest,
        query_hash=q_digest,
        scoring_instant=instant,
        attestation_hash=attestation_hash,
        derivation=derivation,
    )


def derive_recall_attestation(
    results: Any,
    *,
    vector_requested: bool,
    vector_available: bool,
    config_hash: str,
    query: str,
    index_anchor: str = GENESIS_ANCHOR,
    scoring_instant: date | str | None = None,
) -> RecallAttestation:
    """Derive a :class:`RecallAttestation` from a completed recall's run state.

    This is the primary entry point. It reads the recorded signals off
    ``results`` (the ``.degraded`` marker + per-hit provenance), derives the
    legs and the served-id sequence, folds in the degraded marker, and binds
    the *reused* ``config_hash`` (from the pipeline probe — this function never
    invents one) and ``index_anchor``. Nothing is written anywhere.

    The served ids come from :func:`_served_ids`, which reads them off the hit
    dicts in rank order — the recorded output of the run, not a caller's
    summary of it. That is what makes the resulting hash change whenever the
    served answer changes, whatever moved it.

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
        query: The query text this run answered, bound as :func:`query_hash`.
            Required: the ranking is a function of the question, so a record
            that omits it cannot distinguish two runs that happened to serve
            the same list for different reasons, and cannot be replayed.
        index_anchor: The audit-chain head / index snapshot hash
            (:func:`_resolve_index_anchor`), or :data:`GENESIS_ANCHOR`.
        scoring_instant: The UTC date the run's recency layer scored against —
            the value ``recall()`` resolved, passed through so the record binds
            the instant that was actually used rather than re-resolving one.
            Re-resolving here would let the record disagree with the run it
            attests, which is the same class of staleness Finding 2 fixed for
            ``config_hash`` across the cache boundary.

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
    served_ids = _served_ids(results)
    att = build_recall_attestation(
        legs_ran=legs_ran,
        legs_degraded=legs_degraded,
        config_hash=config_hash,
        degraded=degraded,
        index_anchor=index_anchor,
        result_count=result_count,
        served_ids=served_ids,
        query=query,
        scoring_instant=scoring_instant,
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
        results_digest=att.results_digest[:16],
        scoring_instant=att.scoring_instant,
    )
    return att


def derive_recall_attestation_for_workspace(
    results: Any,
    workspace: str,
    *,
    vector_requested: bool,
    vector_available: bool,
    query: str,
    scoring_instant: date | str | None = None,
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
        query=query,
        scoring_instant=scoring_instant,
    )


def verify_recall_attestation(record: RecallAttestation | dict[str, Any]) -> bool:
    """True iff *record* is a well-formed, internally consistent v2 attestation.

    The one verification surface, and it accepts **exactly one tag**. Dual-tag
    support would be a downgrade target: an attacker holding a record under the
    weaker layout could present it unchanged and have it honoured, so the
    version stamp would buy nothing. A record under any other tag is rejected
    here, not migrated — the older layout bound fewer inputs, and there is no
    honest way to synthesise what it never committed to.

    Total on hostile input. A verifier's whole job is to be handed malformed
    values, so it answers False rather than raising: a wrong type, a missing
    field, an absent or foreign tag, a broken hash all mean the same thing to
    the caller. Note that this checks *tamper-evidence*, not authenticity —
    ``attestation_hash`` is a plain SHA-256 that anyone can recompute over a
    forged body. Signing is separate, deferred work.
    """
    if isinstance(record, RecallAttestation):
        return record.schema == RECALL_ATTEST_TAG and record.is_internally_consistent()
    if not isinstance(record, dict):
        return False
    try:
        return RecallAttestation.from_dict(record).is_internally_consistent()
    except (ValueError, KeyError, TypeError):
        return False


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
    # ``query_hash`` / ``served_set_digest`` are deliberately NOT re-exported:
    # they are owned by :mod:`mind_mem.recall_digests`, and two import paths to
    # one canonical encoding is the first step toward two encodings.
    "verify_recall_attestation",
]
