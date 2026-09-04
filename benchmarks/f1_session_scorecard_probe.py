# Copyright 2026 STARGA, Inc.
"""A session-shaped recall run, emitted as the per-question NDJSON a scorecard eats.

:mod:`benchmarks.f1_score_contract_probe` answers "does the default order
move?". It cannot answer "and where it moves, is the move an improvement?" --
that needs gold labels, one row per question, and a paired test. This module
produces the rows; :mod:`benchmarks.paired_scorecard` runs the test.

Why a session-shaped corpus specifically. ``apply_session_boost`` sits behind
``FeatureGate(implicit_section=True, auto_enable_default=True,
auto_detector=_has_session_info)``: it fires when a hit in the head of the
result set carries ``SessionId`` / ``session_id`` / ``Session``. On a corpus
without those fields the stage never runs and there is nothing to measure. So
every block here carries one, and the gold blocks cluster in one session --
which is the premise the stage was built on, not a thumb on the scale: the
generator assigns gold before either leg is scored and never consults the
ranking.

The corpus is generated, not sampled, for two reasons. It is seeded, so the
two arms see byte-identical input and the difference is the patch. And it is
lexically spread on purpose: an earlier synthetic corpus reused one sentence
across every block, and the cosine dedup layer then collapsed a twenty-hit
list to one. That measured the deduplicator, not the ranking. Each block here
draws distinct vocabulary, distinct ``Type`` and distinct ``_source`` so the
dedup layers pass the list through and the served order is the thing under
test.

Metrics per question, against the served list:

* ``recall_any_at_k`` -- 1 if any gold block is in the top k.
* ``recall_all_at_k``  -- 1 if every gold block is.
* ``reciprocal_rank``  -- 1/rank of the first gold block, 0 if none is served.

No clock, no network, and one committed seed.
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

#: Committed corpus seed. It travels in the artifact; a generated corpus whose
#: seed is not recorded is not a corpus anyone else can re-derive.
CORPUS_SEED = 20260904

#: Questions per run. Large enough that a real one-in-twenty effect clears the
#: exact-McNemar floor rather than dying underpowered.
QUESTIONS = 400

#: Blocks in the corpus each question is asked against.
CORPUS = 200

#: Blocks each leg offers per question -- its own top ``POOL`` of ``CORPUS``.
#: Deliberately smaller than the corpus: fusing two *complete* lists is rank
#: averaging, not fusion, and a block only one leg retrieved is exactly the
#: case the pre-F1 mixed column mishandled -- a vector-only hit carried a
#: ``[0, 1]`` cosine into a column whose other entries were unbounded BM25F
#: numbers, so it sorted below every BM25-sourced hit whatever its merit.
POOL = 60

#: Sessions the corpus is partitioned over.
SESSIONS = 6

#: Gold blocks per question, all inside one session.
GOLD_PER_QUESTION = 3

#: Cut-offs the artifact reports. The scorecard tests one of them.
K_VALUES = (1, 3, 5, 10)

#: Response size every question is served at.
LIMIT = 10

#: Vocabulary the excerpts are drawn from. Wide enough that two blocks rarely
#: share enough tokens to trip the 0.85 cosine dedup threshold.
VOCAB = tuple(f"w{i:04d}" for i in range(2000))

#: Words per excerpt.
EXCERPT_WORDS = 10

#: How much of each leg's noise is SHARED between the two legs.
#:
#: This is the one generator parameter the verdict is sensitive to, so it is
#: named and swept rather than chosen. Two rankers reading the same text do
#: not err independently -- a lexical and a dense scorer agree far more than
#: chance -- and at ``0.0`` they do, which starves rank fusion of the very
#: signal it exists to exploit and hands the win to whichever single leg is
#: strongest. Fixing the value by picking the one that flatters the patch
#: would be the same mistake the 85.3 retraction was made under, so every
#: setting in :data:`SHARED_NOISE_SWEEP` is run and every result committed.
DEFAULT_SHARED_NOISE = 0.6

#: The settings the artifact is produced at: independent legs, correlated
#: legs, near-agreeing legs.
SHARED_NOISE_SWEEP = (0.0, 0.6, 0.85)


_Leg = list[dict[str, Any]]


def _question_corpus(index: int, shared_noise: float = DEFAULT_SHARED_NOISE) -> tuple[str, _Leg, _Leg, set[str]]:
    """One question's two legs and its gold set.

    The two legs rank the same blocks by two independently drawn scores on two
    different scales -- an unbounded BM25F-like number and a ``[0, 1]``
    cosine -- so the fused item is a copy of one leg or the other. That is the
    condition under which the pre-F1 ``score`` column was a mixture, and it is
    the condition this measurement has to reproduce.
    """
    rng = random.Random(CORPUS_SEED * 100003 + index)
    gold_session = rng.randrange(SESSIONS)
    gold_slots = set(rng.sample(range(CORPUS), GOLD_PER_QUESTION))

    blocks: list[dict[str, Any]] = []
    gold_ids: set[str] = set()
    for slot in range(CORPUS):
        is_gold = slot in gold_slots
        relevance = 1.0 if is_gold else 0.0
        session = gold_session if is_gold else rng.randrange(SESSIONS)
        block_id = f"Q{index:04d}-B{slot:03d}"
        # One latent both legs read, plus one private draw each. Drawn in a
        # fixed order so the corpus is a pure function of the seed.
        shared = rng.random()
        eps_bm25 = rng.random()
        eps_cos = rng.random()
        words = rng.sample(VOCAB, EXCERPT_WORDS)
        text = " ".join(words)
        blocks.append(
            {
                "_id": block_id,
                # Lowercase ``type`` on purpose: the dedup type cap reads
                # ``type`` and falls back to an ``_id`` prefix, so a corpus
                # that only sets ``Type`` lands every block in one bucket and
                # the cap trims a ten-hit list to three. That would measure
                # the cap, not the ranking.
                "type": f"kind{slot % 20:02d}",
                "SessionId": f"S{session}",
                "statement": text,
                "excerpt": text,
                "Date": f"2026-0{(slot % 9) + 1}-1{slot % 10}",
                "_source": f"src{slot % 17}",
                # ``file`` on purpose too: the per-source chunk cap reads
                # ``file`` and defaults every block that lacks one to "?",
                # which caps the served list at five however many sources the
                # corpus really has.
                "file": f"doc{slot % 17}.md",
                "_gold": is_gold,
                "_bm25": 4.0 + 10.0 * (shared_noise * shared + (1.0 - shared_noise) * eps_bm25) + 3.0 * relevance,
                "_cos": min(1.0, 0.35 + 0.45 * (shared_noise * shared + (1.0 - shared_noise) * eps_cos) + 0.15 * relevance),
            }
        )
        if is_gold:
            gold_ids.add(block_id)

    def leg(score_key: str, out_scale: str) -> list[dict[str, Any]]:
        ordered = sorted(blocks, key=lambda b: (-b[score_key], b["_id"]))[:POOL]
        out = []
        for block in ordered:
            hit = {k: v for k, v in block.items() if not k.startswith("_") or k == "_id" or k == "_source"}
            hit["score"] = block[score_key]
            hit["_leg"] = out_scale
            out.append(hit)
        return out

    query = f"session {gold_session} question {index}"
    return query, leg("_bm25", "bm25"), leg("_cos", "vector"), gold_ids


def _metrics(served_ids: list[str], gold_ids: set[str]) -> dict[str, Any]:
    any_at: dict[str, int] = {}
    all_at: dict[str, int] = {}
    for k in K_VALUES:
        head = set(served_ids[:k])
        any_at[str(k)] = 1 if head & gold_ids else 0
        all_at[str(k)] = 1 if gold_ids <= head else 0
    rank = 0
    for position, block_id in enumerate(served_ids, 1):
        if block_id in gold_ids:
            rank = position
            break
    return {
        "recall_any_at_k": any_at,
        "recall_all_at_k": all_at,
        "reciprocal_rank": 0.0 if rank == 0 else 1.0 / rank,
        "first_gold_rank": rank,
        "n_served": len(served_ids),
    }


def run(questions: int = QUESTIONS, shared_noise: float = DEFAULT_SHARED_NOISE) -> list[dict[str, Any]]:
    """Serve every question through the default fused path and score it."""
    hr.live_statuses = lambda _ws: {}
    HybridBackend._admit = lambda _self, hits, _ws, **_kw: hits

    config = {
        "vector_enabled": True,
        "cross_encoder": {"enabled": False, "auto_enable": False},
        "query_expansion": {"enabled": False, "auto_enable": False},
    }

    rows: list[dict[str, Any]] = []
    for index in range(questions):
        query, bm25_hits, vector_hits, gold_ids = _question_corpus(index, shared_noise)
        HybridBackend._bm25_search = lambda _s, _q, _w, limit=10, _h=bm25_hits, **_kw: [dict(h) for h in _h[: min(limit, POOL)]]
        HybridBackend._vector_search = lambda _s, _q, _w, limit=10, _h=vector_hits, **_kw: [dict(h) for h in _h[: min(limit, POOL)]]
        backend = HybridBackend(config=config)
        backend._vector_available = True
        served = backend.search(query, "/ws", limit=LIMIT, retrieve_wide_k=POOL)
        served_ids = [str(hit.get("_id")) for hit in served]
        row: dict[str, Any] = {
            "question_id": f"Q{index:04d}",
            "unit_status": "ok",
            "n_gold": len(gold_ids),
        }
        row.update(_metrics(served_ids, gold_ids))
        rows.append(row)
    return rows


def resolve_shared_noise() -> float:
    """The sweep point this run uses, from ``F1_SHARED_NOISE`` or the default.

    Refuses a value outside ``[0, 1]`` rather than clamping: a typo that
    silently became a legal setting would put an unlabelled corpus behind a
    labelled artifact.
    """
    raw = os.environ.get("F1_SHARED_NOISE")
    if raw is None:
        return DEFAULT_SHARED_NOISE
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"F1_SHARED_NOISE must be in [0, 1], got {value!r}")
    return value


def provenance(shared_noise: float) -> dict[str, Any]:
    """Which tree ran, and the corpus parameters it ran over."""
    module_path = os.path.abspath(hr.__file__)
    with open(module_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return {
        "hybrid_recall_path": module_path,
        "hybrid_recall_sha256": digest,
        "corpus_seed": CORPUS_SEED,
        "questions": QUESTIONS,
        "corpus": CORPUS,
        "pool": POOL,
        "sessions": SESSIONS,
        "gold_per_question": GOLD_PER_QUESTION,
        "limit": LIMIT,
        "shared_noise": shared_noise,
    }


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print("usage: f1_session_scorecard_probe.py OUT.ndjson [PROVENANCE.json]", file=sys.stderr)
        return 2
    shared_noise = resolve_shared_noise()
    rows = run(shared_noise=shared_noise)
    with open(argv[1], "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if len(argv) == 3:
        with open(argv[2], "w", encoding="utf-8") as handle:
            json.dump(provenance(shared_noise), handle, indent=2, sort_keys=True)
            handle.write("\n")
    boosted = sum(1 for r in rows if r["n_served"] > 0)
    mean_rr = sum(r["reciprocal_rank"] for r in rows) / len(rows)
    print(f"wrote {argv[1]} rows={len(rows)} shared_noise={shared_noise} served>0={boosted} mean_rr={mean_rr:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
