# Copyright 2026 STARGA, Inc.
"""Capture the served ``(id, score)`` list of the DEFAULT fused recall path.

This is the runnable form of the 5.0.2 F1 claim. F1 made ``rrf_fuse`` write
``score = rrf_score``; before it, fusion wrote only ``rrf_score`` and left
``score`` holding whichever leg's raw value survived the dict copy -- an
unbounded BM25F number on some hits and a ``[0, 1]`` cosine on others, in one
column. Two questions follow, and both are measurements:

1. Does the product default move? Run this module under a pre-F1 tree and
   under the current one and compare with :mod:`benchmarks.ranking_identity`.
2. How broken was the column it replaced? Count, in the pre-F1 artifact, how
   many served lists were not non-increasing in ``score``.

Neither question is answerable from a report, which is why this is a
committed module and not a paragraph. It is also why the artifact records
**which tree produced it**: the resolved ``mind_mem`` path and a SHA-256 of
``hybrid_recall.py``. A before/after pair whose two digests match measured one
tree twice, and the comparator refuses it.

Both retrieval legs are stubbed with a fixed synthetic ranking, so the two
trees see byte-identical input and any output difference is the patch.
Everything downstream of fusion runs for real and at its default setting --
dedup, session boost, temporal decay, truth scoring -- because those are the
stages that read ``score``.

Selecting the tree
------------------
``F1_PROBE_SRC`` names the ``src`` directory to import ``mind_mem`` from. It
is prepended to ``sys.path``; the repo's own ``src`` is used when it is unset.
The resolved path travels in the artifact either way, so the choice is on the
record rather than in a shell history.

No clock, no network, no unseeded randomness: every score below is a closed
form of its rank.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.environ.get("F1_PROBE_SRC") or os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import mind_mem.hybrid_recall as hr  # noqa: E402
from mind_mem.hybrid_recall import HybridBackend  # noqa: E402

#: Leg depth. Both stubbed legs can serve this many blocks.
POOL = 60

#: Distinct ``Type`` values, so the dedup type cap (3 per type) does not
#: collapse every served list to three hits and blunt the comparison.
TYPE_COUNT = 20

#: The two batteries below are the 45 served lists the 5.0.2 CHANGELOG counts.
FUSED_QUERIES = ("retrieval fusion scoring", "how does fusion relate to scoring and why")
FUSED_LIMITS = (3, 10, 25)
FUSED_CONFIGS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("default", {}),
    ("weights_2_1", {"bm25_weight": 2.0, "vector_weight": 1.0}),
    ("rrf_k_10", {"rrf_k": 10}),
)

STAGE_QUERY = "retrieval fusion scoring"
STAGE_LIMITS = (5, 10, 25)
STAGE_SHAPES: tuple[tuple[str, dict[str, bool]], ...] = (
    ("plain", {"sessions": False, "chunks": False}),
    ("session_boost", {"sessions": True, "chunks": False}),
    ("chunked_dedup", {"sessions": False, "chunks": True}),
)
STAGE_CONFIGS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("default", {}),
    ("temporal_decay", {"retrieval": {"temporal_decay_hot_path": True}}),
    ("trust_scores", {"retrieval": {"trust_scores": {"enabled": True}}}),
)

#: The deep battery. The two batteries above are the historical 45, and every
#: one of their served lists is three hits long: their blocks set ``Type`` but
#: not ``type``, and the dedup type cap reads ``type`` and falls back to an
#: ``_id`` prefix, so all sixty land in one bucket and the cap trims to three.
#: An identity claim over three-hit lists is a thin claim. These cases carry
#: the fields the caps actually read (``type``, ``file``) and lexically
#: distinct excerpts, so the served list reaches the requested limit and a
#: reordering has room to show up.
DEEP_SEED = 20260904
DEEP_QUERIES = 40
DEEP_CORPUS = 200
DEEP_LIMITS = (10, 25)
DEEP_VOCAB = tuple(f"w{i:04d}" for i in range(2000))

#: Case names whose served order is the PRODUCT DEFAULT: default config, no
#: session metadata, no chunk ids. These are the lists the patch release
#: promises not to move, and the comparator asserts identity over exactly
#: these. The rest are the stages that re-sort on ``score`` and therefore
#: read a different number after F1 -- they are measured, not asserted.
DEFAULT_PATH_PREFIXES = ("fused|default|", "stage|plain|default|", "deep|")

#: Keys of the historical battery the 5.0.2 monotonicity count is taken over.
LEGACY_PREFIXES = ("fused|", "stage|")


def _fused_block(index: int, score: float) -> dict[str, Any]:
    """One block as the *fusion* battery builds it."""
    return {
        "_id": f"BLK-{index:03d}",
        "score": score,
        "Type": f"Kind{index % TYPE_COUNT:02d}",
        "statement": f"block {index} about retrieval fusion scoring",
        "excerpt": f"block {index} about retrieval fusion scoring",
        "Date": f"2026-0{(index % 9) + 1}-1{index % 10}",
        "_source": f"src{index % 6}",
    }


def _stage_block(index: int, score: float, *, sessions: bool, chunks: bool) -> dict[str, Any]:
    """One block as the *stage* battery builds it.

    ``chunks`` collapses sixty ids onto twenty base ids with a ``.n`` suffix,
    which is what makes dedup layer 1 -- keep the highest-``score`` chunk per
    base id -- a score-dependent decision rather than a no-op.
    """
    bid = f"BLK-{index % 20:03d}.{index // 20}" if chunks else f"BLK-{index:03d}"
    block: dict[str, Any] = {
        "_id": bid,
        "score": score,
        "Type": f"Kind{index % TYPE_COUNT:02d}",
        "statement": f"block {index} retrieval fusion scoring",
        "excerpt": f"block {index} retrieval fusion scoring",
        "Date": f"2026-0{(index % 9) + 1}-1{index % 10}",
        "_source": f"src{index % 6}",
    }
    if sessions:
        block["SessionId"] = f"S{index % 4}"
    return block


def _legs(builder: Any) -> tuple[Any, Any]:
    """A forward BM25F-scale leg and a reversed cosine-scale leg.

    The order difference is the whole point: it is what makes the fused item
    a copy of one leg or the other, which is how the mixed-scale ``score``
    column arose. The scales differ by two orders of magnitude, which is what
    made the mixture visible.
    """

    def bm25(_self: Any, _query: str, _workspace: str, limit: int = 10, **_kw: Any) -> list[dict]:
        return [builder(i, 14.0 - rank * 0.11) for rank, i in enumerate(range(min(limit, POOL)))]

    def vector(_self: Any, _query: str, _workspace: str, limit: int = 10, **_kw: Any) -> list[dict]:
        ordered = list(range(POOL - 1, -1, -1))[: min(limit, POOL)]
        return [builder(i, 0.94 - rank * 0.004) for rank, i in enumerate(ordered)]

    return bm25, vector


def _base_config(extra: dict[str, Any]) -> dict[str, Any]:
    """The cross-encoder-OFF configuration every case runs under.

    Multi-query expansion is off because its variant generation reaches the
    network, which this module must not do. It is untouched by F1 and its
    fusion is covered by the ``rrf_k_10`` case.
    """
    return {
        "vector_enabled": True,
        "cross_encoder": {"enabled": False, "auto_enable": False},
        "query_expansion": {"enabled": False, "auto_enable": False},
        **extra,
    }


def _served(config: dict[str, Any], query: str, limit: int) -> list[list[Any]]:
    backend = HybridBackend(config=config)
    backend._vector_available = True
    hits = backend.search(query, "/ws", limit=limit, retrieve_wide_k=POOL)
    return [[hit.get("_id"), float(hit.get("score", 0.0) or 0.0)] for hit in hits]


def _deep_blocks(index: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One deep case's two legs: top-``POOL`` of ``DEEP_CORPUS``, no sessions.

    No ``SessionId`` anywhere, so ``_has_session_info`` says no and the
    session-boost gate never opens -- these cases are the product default with
    nothing re-sorting behind fusion.
    """
    rng = random.Random(DEEP_SEED * 7919 + index)
    blocks: list[dict[str, Any]] = []
    for slot in range(DEEP_CORPUS):
        shared = rng.random()
        eps_bm25 = rng.random()
        eps_cos = rng.random()
        text = " ".join(rng.sample(DEEP_VOCAB, 10))
        blocks.append(
            {
                "_id": f"D{index:03d}-B{slot:03d}",
                "type": f"kind{slot % 20:02d}",
                "file": f"doc{slot % 17}.md",
                "statement": text,
                "excerpt": text,
                "Date": f"2026-0{(slot % 9) + 1}-1{slot % 10}",
                "_source": f"src{slot % 17}",
                "_bm25": 4.0 + 10.0 * (0.6 * shared + 0.4 * eps_bm25),
                "_cos": min(1.0, 0.35 + 0.45 * (0.6 * shared + 0.4 * eps_cos)),
            }
        )

    def leg(key: str) -> list[dict[str, Any]]:
        ordered = sorted(blocks, key=lambda b: (-b[key], b["_id"]))[:POOL]
        return [{k: v for k, v in b.items() if not k.startswith("_") or k == "_id" or k == "_source"} | {"score": b[key]} for b in ordered]

    return leg("_bm25"), leg("_cos")


def run_battery() -> dict[str, list[list[Any]]]:
    """Every case, in a stable order. Keys are ``battery|config|query|limit``."""
    hr.live_statuses = lambda _ws: {}
    HybridBackend._admit = lambda _self, hits, _ws, **_kw: hits

    captured: dict[str, list[list[Any]]] = {}

    bm25, vector = _legs(lambda i, s: _fused_block(i, s))
    HybridBackend._bm25_search = bm25
    HybridBackend._vector_search = vector
    for label, extra in FUSED_CONFIGS:
        config = _base_config(extra)
        for query in FUSED_QUERIES:
            for limit in FUSED_LIMITS:
                captured[f"fused|{label}|{query}|{limit}"] = _served(config, query, limit)

    for shape, shape_kwargs in STAGE_SHAPES:
        builder = (lambda kw: lambda i, s: _stage_block(i, s, **kw))(shape_kwargs)
        bm25, vector = _legs(builder)
        HybridBackend._bm25_search = bm25
        HybridBackend._vector_search = vector
        for label, extra in STAGE_CONFIGS:
            config = _base_config(extra)
            for limit in STAGE_LIMITS:
                captured[f"stage|{shape}|{label}|{limit}"] = _served(config, STAGE_QUERY, limit)

    config = _base_config({})
    for index in range(DEEP_QUERIES):
        bm25_hits, vector_hits = _deep_blocks(index)
        HybridBackend._bm25_search = lambda _s, _q, _w, limit=10, _h=bm25_hits, **_kw: [dict(h) for h in _h[: min(limit, POOL)]]
        HybridBackend._vector_search = lambda _s, _q, _w, limit=10, _h=vector_hits, **_kw: [dict(h) for h in _h[: min(limit, POOL)]]
        for limit in DEEP_LIMITS:
            captured[f"deep|default|q{index:03d}|{limit}"] = _served(config, f"deep query {index}", limit)

    return captured


def provenance() -> dict[str, str]:
    """Which tree ran. Two artifacts sharing this measured one tree twice."""
    module_path = os.path.abspath(hr.__file__)
    with open(module_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return {"hybrid_recall_path": module_path, "hybrid_recall_sha256": digest}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: f1_score_contract_probe.py OUT.json", file=sys.stderr)
        return 2
    cases = run_battery()
    payload = {"provenance": provenance(), "cases": cases}
    with open(argv[1], "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {argv[1]} cases={len(cases)} hits={sum(len(v) for v in cases.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
