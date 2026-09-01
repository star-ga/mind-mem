"""The v4 kind-index build pass — ``mm kinds backfill``'s engine.

Four restored v4 surfaces only ever do work in one order, and this module is
that order, written once:

1. :mod:`~mind_mem.v4.block_kinds` classifies every **admitted** corpus block
   and writes both kind surfaces (``blocks.kind`` + ``block_kind_tags``).
2. :mod:`~mind_mem.v4.kind_summaries` rebuilds one summary per kind, reading
   the ``blocks`` rows step 1 just wrote.
3. :mod:`~mind_mem.v4.embedding_pipeline` derives an embedding per block from
   the same rows.
4. :mod:`~mind_mem.v4.hnsw_kind_index` registers those embeddings under the
   block's primary kind, so ``knn_by_kind`` has a partition to scan.

Each step is independently flag-gated and every step after the first is
skipped unless step 1 ran, so a workspace with ``v4.block_kinds`` OFF pays
exactly ONE config read for the whole pass and writes nothing.

**Admission is not optional here.** The pass enumerates the corpus with
``iter_blocks(active_only=False)`` and then filters it through
:func:`mind_mem.admissibility.admit_corpus` — the shared gate, called, never
re-implemented. A hand-rolled status check here is exactly the defect that
leaked quarantined blocks once already.

The two filters are not interchangeable and the weaker-looking one is the
governance one. ``active_only`` is a LIFECYCLE filter: it keeps exactly
``Status: active``, so it silently drops an open task and a superseded
decision — both of which recall serves — while saying nothing about
``quarantined`` or ``pending`` as a category. ``admit_corpus`` is the
SERVABILITY filter: it admits every recognised status (including ``open``),
honours a release decision, and withholds the withheld. Enumerating without
``active_only`` and admitting afterwards therefore indexes MORE of the corpus
and withholds MORE of what must not be served — and it makes the admission
call load-bearing rather than decorative, which is the only way a test can
prove it is there.

**What it writes.** Only ``<workspace>/index.db``, the v4 side store. No
corpus file, no ``BlockStore``, no hash-chain entry — nothing here mints a
block, so nothing here needs (or may claim) a governance admission of its
own. It is a derived index over blocks that already passed the gate, in the
same class as the FTS index.

**Determinism.** Blocks are processed in sorted id order and every input is
a pure function of the corpus, so two runs over an unchanged corpus write
identical rows. Nothing here touches the scored recall path.

Copyright STARGA, Inc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..observability import get_logger
from .feature_flags import is_enabled

_log = get_logger("v4.kind_backfill")

__all__ = ["BackfillResult", "backfill"]

#: Bookkeeping keys the corpus parser stamps on every block. They carry no
#: content signal, so they are left out of the text the summariser and the
#: embedder see.
_SKIP_PREFIX = "_"

#: Fields never worth embedding or summarising even without the underscore.
_SKIP_FIELDS = frozenset({"Status", "status"})


@dataclass(frozen=True)
class BackfillResult:
    """What one pass did, per step. Every count is a real write."""

    blocks_scanned: int = 0
    blocks_admitted: int = 0
    kinds_written: int = 0
    summaries_refreshed: int = 0
    embeddings_derived: int = 0
    embeddings_registered: int = 0
    rows_pruned: int = 0
    kind_counts: dict[str, int] = field(default_factory=dict)
    steps_enabled: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "blocks_scanned": self.blocks_scanned,
            "blocks_admitted": self.blocks_admitted,
            "kinds_written": self.kinds_written,
            "summaries_refreshed": self.summaries_refreshed,
            "embeddings_derived": self.embeddings_derived,
            "embeddings_registered": self.embeddings_registered,
            "rows_pruned": self.rows_pruned,
            "kind_counts": dict(self.kind_counts),
            "steps_enabled": dict(self.steps_enabled),
        }


def _block_text(block: dict[str, Any]) -> str:
    """Flatten a block into the text the summariser and embedder see.

    Field order follows the parsed block (insertion order), so the same block
    always renders the same string — the whole pass has to be replayable.
    """
    parts: list[str] = []
    for key, value in block.items():
        if key.startswith(_SKIP_PREFIX) or key in _SKIP_FIELDS:
            continue
        if isinstance(value, str):
            if value.strip():
                parts.append(value)
        elif isinstance(value, (list, tuple)):
            parts.extend(str(v) for v in value if isinstance(v, (str, int, float)))
        elif isinstance(value, (int, float)):
            parts.append(str(value))
    return "\n".join(parts)


def _admitted_blocks(workspace: str) -> tuple[int, list[dict[str, Any]]]:
    """``(scanned, admitted)`` — the corpus through BOTH filters, sorted.

    See the module docstring for why ``admit_corpus`` is not optional.
    """
    from ..admissibility import admit_corpus
    from ..storage import iter_blocks

    raw = iter_blocks(workspace, active_only=False)
    admitted = admit_corpus(raw)
    admitted.sort(key=lambda b: str(b.get("_id", "")))
    return len(raw), [b for b in admitted if str(b.get("_id", "")).strip()]


def _install_recall_vector_embedder(workspace: str) -> str:
    """Point ``embedding_pipeline`` at the product's real embedding provider.

    ``embedding_pipeline``'s built-in embedder is hashed character 3-grams —
    deterministic and dependency-free, but it is a stand-in, not a semantic
    embedder. When ``recall_vector``'s provider chain is usable, install it;
    otherwise leave the stdlib default in place and say which one ran, so a
    deployment can never mistake trigram buckets for real embeddings.

    Returns the name of the embedder actually installed.
    """
    from .embedding_pipeline import set_embedder

    try:
        import json as _json

        from ..recall_vector import VectorBackend

        cfg: dict[str, Any] = {}
        try:
            with open(Path(workspace) / "mind-mem.json", encoding="utf-8") as fh:
                cfg = _json.load(fh).get("recall", {}) or {}
        except (OSError, ValueError):
            cfg = {}
        backend = VectorBackend(cfg)
        # The chained, circuit-broken provider path when it exists, the plain
        # sentence-transformers one otherwise. One text per call because that
        # is ``embedding_pipeline``'s Embedder contract; this is an operator
        # batch command, not a per-recall path.
        chain = getattr(backend, "_embed_for_provider", None) or backend.embed

        def _recall_vector_embedder(text: str, dim: int = 128) -> list[float]:
            vecs = chain([text])
            return [float(x) for x in vecs[0]] if vecs else []

        # Prove the chain actually answers before installing it: a provider
        # that raises on first use would otherwise turn every derived
        # embedding into an exception halfway through the pass.
        if not _recall_vector_embedder("mind-mem embedding probe"):
            raise RuntimeError("recall_vector provider returned an empty vector")
        set_embedder(_recall_vector_embedder)
        return "recall_vector"
    except Exception as exc:  # noqa: BLE001 - any provider failure is a fallback, not a crash
        _log.info("kind_backfill_embedder_fallback", error=str(exc), embedder="hashed_trigram")
        return "hashed_trigram"


def backfill(workspace: str | Path) -> BackfillResult:
    """Run the kind-index build pass over *workspace*.

    Raises :class:`~mind_mem.v4.feature_flags.FeatureDisabledError` when
    ``v4.block_kinds`` is OFF — the caller is an operator who typed
    ``mm kinds backfill``, so a clear refusal beats a silent no-op.
    """
    from .block_kinds import (
        classify_block,
        ensure_block_kind_column,
        ensure_block_kind_tags_table,
        primary_kind,
        prune_kind_index,
        set_block_kind,
        set_block_kinds,
    )

    ws = str(workspace)

    # Step 1 — kinds. `require_enabled` fires inside these; the operator gets
    # the actionable "enable v4.block_kinds" message rather than a no-op.
    ensure_block_kind_column(ws)
    ensure_block_kind_tags_table(ws)

    scanned, blocks = _admitted_blocks(ws)
    kind_counts: dict[str, int] = {}
    primaries: dict[str, str] = {}
    written = 0
    for block in blocks:
        bid = str(block["_id"])
        tags = classify_block(block)
        primary = primary_kind(block)
        set_block_kinds(ws, bid, tags)
        set_block_kind(ws, bid, primary, content=_block_text(block))
        primaries[bid] = primary.value
        kind_counts[primary.value] = kind_counts.get(primary.value, 0) + 1
        written += 1

    # Step 1b — CONVERGE. A block admitted at the last run and quarantined
    # since still has its row, its tags and its stored text; without this the
    # index only ever grows, and it grows fail-open. Re-running the backfill
    # has to be able to take something away.
    pruned = prune_kind_index(ws, primaries)

    # Steps 2-4 are optional. Each flag is read ONCE, here, outside every
    # loop — a per-block `is_enabled` would re-parse mind-mem.json per block.
    do_summaries = is_enabled("kind_summaries")
    do_embeddings = is_enabled("embedding_pipeline")
    do_register = is_enabled("hnsw_kind_index")

    # Step 2 — summaries, over the rows step 1 just wrote.
    summaries = 0
    if do_summaries:
        from .kind_summaries import refresh_summary

        for kind in sorted(kind_counts):
            if refresh_summary(ws, kind) is not None:
                summaries += 1

    # Step 3 — embeddings.
    #
    # deferred: with ``hnsw_kind_index`` OFF the vectors are derived, counted
    # and then dropped, because step 4 is the only consumer that exists. That
    # is the flag's own meaning ("derive embeddings"), and the count is a real
    # answer -- how many admitted blocks have embeddable content -- but it is
    # wasted work on a large corpus. Upgrade path: a second consumer (a
    # kind-aware recall leg, or a persisted embedding cache) or a documented
    # rule that embedding_pipeline implies hnsw_kind_index.
    embeddings: dict[str, list[float]] = {}
    embedder = ""
    if do_embeddings:
        from .embedding_pipeline import derive_embeddings

        embedder = _install_recall_vector_embedder(ws)
        embeddings = derive_embeddings(ws, sorted(primaries))

    # Step 4 — register them under each block's primary kind.
    registered = 0
    if do_register:
        from .hnsw_kind_index import ensure_hnsw_schema, prune_embeddings, register_block_embedding

        ensure_hnsw_schema(ws)
        for bid in sorted(embeddings):
            vec = embeddings[bid]
            if not vec:
                continue
            register_block_embedding(ws, bid, primaries.get(bid, ""), vec)
            registered += 1
        # Same convergence rule as step 1b, for the vector partition. Runs
        # even when nothing was derived this pass: "no embeddings now" is
        # exactly when a stale one from last time is most dangerous.
        pruned += prune_embeddings(ws, sorted(primaries))

    result = BackfillResult(
        blocks_scanned=scanned,
        blocks_admitted=len(blocks),
        kinds_written=written,
        summaries_refreshed=summaries,
        embeddings_derived=len(embeddings),
        embeddings_registered=registered,
        rows_pruned=pruned,
        kind_counts=kind_counts,
        steps_enabled={
            "block_kinds": True,
            "kind_summaries": do_summaries,
            "embedding_pipeline": do_embeddings,
            "hnsw_kind_index": do_register,
        },
    )
    _log.info(
        "v4_kind_backfill",
        scanned=scanned,
        admitted=len(blocks),
        written=written,
        summaries=summaries,
        registered=registered,
        pruned=pruned,
        embedder=embedder or "none",
    )
    return result
