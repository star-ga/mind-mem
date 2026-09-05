# Copyright 2026 STARGA, Inc.
"""Stage masks for the SQLite recall ranking stack — measurement, not tuning.

``sqlite_index.query_index`` does not serve FTS5's ``bm25()`` ordering. It
serves that ordering after a stack of modifiers, each hand-tuned for a
governed Markdown corpus rather than for a session-document haystack:

===================  ==============================================================
stage                what it multiplies / changes
===================  ==============================================================
``columns``          BM25F per-field weights (statement 3.0 … context 0.5)
``recency``          ``score *= 1 - rw + rw * date_score(block)``, ``rw`` = 0.3 (0.6 temporal)
``date_boost``       ``score *= 2.0`` for a dated block on a temporal query
``status``           ``score *= 1.2`` active / ``1.1`` todo|doing
``priority``         ``score *= 1.1`` for ``P0``/``P1``
``calibration``      ``score *=`` the per-block calibration-feedback weight
``rerank``           ``rerank_hits`` v7 re-scores the top 200 before the cut
``expansion``        ``expand_query`` widens the FTS5 ``MATCH`` token set
===================  ==============================================================

A mask names the stages to DISABLE. Nothing here edits the product: every
stage is switched off by rebinding a value the product already reads, in the
benchmark's own child process, after import and before the first query. With
no mask the module is never imported and not one product symbol is touched —
which is the property :func:`selftest` and the unmasked control run exist to
prove, since an ablation harness that perturbs the control measures itself.

Two stages have no constant to rebind: ``status`` and ``priority`` are inline
literals in ``query_index``'s scoring loop. They are cancelled instead, by the
one per-row multiplier that IS a seam — the calibration weight — carrying the
exact reciprocal of the boost each row is about to receive. That composes with
the real calibration weight rather than replacing it, so ``status`` can be
ablated without silently ablating ``calibration`` too.

Every mask that will be run is listed in :data:`MASKS` and committed BEFORE
any result is looked at. Running masks until one wins and reporting that one
is the failure this file is shaped to prevent.
"""

from __future__ import annotations

from typing import Any

#: Every stage a mask may name. A mask naming anything else is a typo, and a
#: typo that silently ablates nothing would be reported as "no effect".
STAGES: frozenset[str] = frozenset(
    {
        "columns",
        "recency",
        "date_boost",
        "status",
        "priority",
        "calibration",
        "rerank",
        "expansion",
        "facts",
    }
)

#: The pre-committed run list: mask name -> stages disabled.
#:
#: ``control`` is the unmasked product default and is run FIRST, as a positive
#: control: it must reproduce the committed 2026-09-03 rep1 artifact question
#: for question, or the harness — not the ranking — is what is being measured.
#:
#: Singles isolate each modifier. The three composites are the pre-declared
#: hypotheses: strip the multiplier stack, strip the multipliers and the
#: reranker (leaving FTS5 BM25F alone), and strip everything down to plain
#: BM25 over the same text, which is the structural analogue of the zero-dep
#: in-memory baseline this stack loses to.
MASKS: dict[str, tuple[str, ...]] = {
    "control": (),
    "no_recency": ("recency",),
    "no_date_boost": ("date_boost",),
    "no_status": ("status",),
    "no_priority": ("priority",),
    "no_calibration": ("calibration",),
    "no_rerank": ("rerank",),
    "flat_columns": ("columns",),
    "no_expansion": ("expansion",),
    "no_multipliers": ("recency", "date_boost", "status", "priority", "calibration"),
    "bm25f_only": ("recency", "date_boost", "status", "priority", "calibration", "rerank"),
    "flat_bm25": ("recency", "date_boost", "status", "priority", "calibration", "rerank", "columns"),
    "plain_bm25": ("recency", "date_boost", "status", "priority", "calibration", "rerank", "columns", "expansion"),
}

#: Masks added AFTER the battery above was committed and BEFORE any of its
#: results were read, from a direct probe of the eval workspace rather than
#: from a result. They are kept in a separate dict, and reported in a separate
#: section, so the pre-registered list stays exactly what was pre-registered.
#:
#: What the probe measured: the FTS corpus a 53-session haystack produces holds
#: **373** documents, not 53. ``index_block`` mints a FACT sub-block per
#: extracted card, so 320 short cards (median 65 characters) share the
#: ``bm25()`` statistics surface with 53 long session parents (mean 1355
#: characters of ``statement``). IDF and the average document length -- both
#: corpus-wide, neither narrowed by any WHERE clause -- are therefore computed
#: over a population the zero-dep floor never sees, and BM25's length
#: normalisation scores the parents against an average dragged down by the
#: cards. This is an INGEST-time difference, so it sits outside the modifier
#: stack F5 was ruled to ablate; it is measured here rather than asserted.
POST_HOC_MASKS: dict[str, tuple[str, ...]] = {
    "no_facts": ("facts",),
    "no_facts_plain": ("facts", "recency", "date_boost", "status", "priority", "calibration", "rerank", "columns", "expansion"),
}


def parse_mask(spec: str) -> frozenset[str]:
    """``"no_rerank"`` or ``"rerank+recency"`` -> the stage set to disable.

    Raises:
        ValueError: An unknown mask name or an unknown stage. Never silently
            ignored: a mask that ablates nothing reports "no effect", which
            reads exactly like a measured null.
    """
    spec = (spec or "").strip()
    if not spec:
        return frozenset()
    if spec in MASKS:
        return frozenset(MASKS[spec])
    if spec in POST_HOC_MASKS:
        return frozenset(POST_HOC_MASKS[spec])
    stages = frozenset(part.strip() for part in spec.split("+") if part.strip())
    unknown = stages - STAGES
    if unknown:
        raise ValueError(f"unknown ablation stage(s): {sorted(unknown)}; known: {sorted(STAGES)}")
    return stages


def _patch_query_type_params(stages: frozenset[str]) -> None:
    """Rebind the per-query-type ranking parameters, in place.

    ``sqlite_index`` binds this dict by name at import, so the sub-dicts are
    shared objects: mutating them here is what the FTS5 leg will read.
    """
    from mind_mem import _recall_detection

    for params in _recall_detection._QUERY_TYPE_PARAMS.values():
        if "recency" in stages:
            params["recency_weight"] = 0.0
        if "date_boost" in stages:
            params["date_boost"] = 1.0
        if "expansion" in stages:
            params["expand_query"] = False


def _patch_columns() -> None:
    """Flatten the BM25F field weights to a uniform 1.0.

    ``_bm25_weights()`` reads ``FTS5_COLUMNS`` off the module at call time, so
    rebinding the module attribute is enough; the column ORDER is preserved
    because the weight string is positional.
    """
    from mind_mem import sqlite_index

    sqlite_index.FTS5_COLUMNS = [(name, 1.0) for name, _ in sqlite_index.FTS5_COLUMNS]


def _patch_rerank() -> None:
    """Make ``rerank_hits`` the identity.

    Passing ``rerank=False`` would work for the ``recall()`` leg but not for
    any other caller, and would also skip the candidate-cap slice. Neutering
    the function keeps the surrounding control flow — the cap, the slice, the
    logging — exactly as the unmasked run executes it, so the only thing this
    stage removes is the v7 re-scoring itself.
    """
    from mind_mem import sqlite_index

    sqlite_index.rerank_hits = lambda query, hits, debug=False: hits


def _patch_facts() -> None:
    """Stop the indexer minting FACT sub-blocks.

    ``index_block`` imports ``extract_facts`` at module scope and calls it on
    every statement longer than 15 characters, so rebinding the module
    attribute empties the fact layer at INGEST time -- which is the only place
    it can be emptied, because the cards are rows in ``blocks_fts`` and it is
    their presence in that table, not their score, that moves ``bm25()``'s IDF
    and length average. With no cards minted, ``_aggregate_facts_to_parents``
    has nothing to fold and is inert on its own.
    """
    from mind_mem import sqlite_index

    sqlite_index.extract_facts = lambda *args, **kwargs: []


def _row_corrections(workspace: str, block_ids: list[str], stages: frozenset[str]) -> dict[str, float]:
    """Reciprocals of the inline status / priority boosts, per block id.

    Read from the same ``blocks`` table ``query_index`` selects from, so a
    heterogeneous corpus is cancelled row by row rather than by assuming the
    boost is uniform. A row that cannot be read is left uncorrected (1.0),
    which under-ablates rather than inventing a factor.
    """
    import json as _json

    from mind_mem.sqlite_index import TaskStatus, _get_conn_manager

    out: dict[str, float] = {}
    if not block_ids:
        return out
    conn = _get_conn_manager(workspace).get_read_connection()
    placeholders = ",".join("?" for _ in block_ids)
    rows = conn.execute(
        f"SELECT id, status, json_blob FROM blocks WHERE id IN ({placeholders})",  # noqa: S608 - ids are bound
        block_ids,
    ).fetchall()
    for row in rows:
        bid, status, blob = row[0], row[1], row[2]
        factor = 1.0
        if "status" in stages:
            if status == "active":
                factor /= 1.2
            elif status in {TaskStatus.TODO, TaskStatus.DOING}:
                factor /= 1.1
        if "priority" in stages:
            try:
                data = _json.loads(blob) if blob else {}
            except (TypeError, ValueError):
                data = {}
            if isinstance(data, dict) and data.get("Priority", "") in ("P0", "P1"):
                factor /= 1.1
        if factor != 1.0:
            out[str(bid)] = factor
    return out


def _patch_calibration(stages: frozenset[str]) -> None:
    """Own the one per-row multiplicative seam ``query_index`` exposes.

    ``query_index`` imports ``batch_calibration_weights`` from
    ``_recall_core`` inside its own body, so rebinding the attribute on
    ``_recall_core`` is what the next query resolves.
    """
    from mind_mem import _recall_core

    original = _recall_core.batch_calibration_weights
    drop_calibration = "calibration" in stages
    cancel = bool({"status", "priority"} & stages)

    def _weights(cal_mgr: Any, block_ids: list[str], *, now: Any) -> dict[str, float]:
        base: dict[str, float] = {} if drop_calibration else dict(original(cal_mgr, block_ids, now=now))
        if not cancel:
            return base
        workspace = getattr(cal_mgr, "_workspace", None)
        if not workspace:
            return base
        for bid, factor in _row_corrections(str(workspace), list(block_ids), stages).items():
            base[bid] = base.get(bid, 1.0) * factor
        return base

    _recall_core.batch_calibration_weights = _weights


def apply_mask(stages: frozenset[str]) -> frozenset[str]:
    """Disable *stages* in this process. Returns what was disabled.

    A no-op for an empty set — and the caller must not even reach this
    function on the unmasked path, so that a control run cannot import this
    module at all.
    """
    if not stages:
        return stages
    unknown = stages - STAGES
    if unknown:
        raise ValueError(f"unknown ablation stage(s): {sorted(unknown)}")
    if {"recency", "date_boost", "expansion"} & stages:
        _patch_query_type_params(stages)
    if "columns" in stages:
        _patch_columns()
    if "rerank" in stages:
        _patch_rerank()
    if "facts" in stages:
        _patch_facts()
    if {"calibration", "status", "priority"} & stages:
        _patch_calibration(stages)
    return stages


def selftest() -> dict[str, Any]:
    """Prove each patch reaches the value the product actually reads.

    A mask that patched a stale copy would ablate nothing and be reported as
    a measured null, which is indistinguishable from the finding. Run before
    the battery; the result is committed beside it.
    """
    from mind_mem import _recall_core, _recall_detection, sqlite_index

    before = {
        "weights": sqlite_index._bm25_weights(),
        "recency": {k: v.get("recency_weight") for k, v in _recall_detection._QUERY_TYPE_PARAMS.items()},
        "date_boost": {k: v.get("date_boost") for k, v in _recall_detection._QUERY_TYPE_PARAMS.items()},
        "expand": {k: v.get("expand_query") for k, v in _recall_detection._QUERY_TYPE_PARAMS.items()},
        "rerank_hits": sqlite_index.rerank_hits.__name__,
        "cal": _recall_core.batch_calibration_weights.__name__,
    }
    apply_mask(frozenset(MASKS["plain_bm25"]))
    after = {
        "weights": sqlite_index._bm25_weights(),
        "recency": {k: v.get("recency_weight") for k, v in _recall_detection._QUERY_TYPE_PARAMS.items()},
        "date_boost": {k: v.get("date_boost") for k, v in _recall_detection._QUERY_TYPE_PARAMS.items()},
        "expand": {k: v.get("expand_query") for k, v in _recall_detection._QUERY_TYPE_PARAMS.items()},
        "rerank_hits": sqlite_index.rerank_hits.__name__,
        "cal": _recall_core.batch_calibration_weights.__name__,
    }
    return {"before": before, "after": after}


if __name__ == "__main__":  # pragma: no cover - operator probe
    import json as _j

    print(_j.dumps(selftest(), indent=2, sort_keys=True))
