# Copyright 2026 STARGA, Inc.
"""Detect a vector leg that carries no signal, and stop pretending it does.

Measured 2026-08-31 on all-MiniLM-L6-v2 (``benchmarks/embed_augmentation_ab.py``):

    prose corpus (symptom query vs resolution text)   spread 0.108   top1 80%
    near-duplicate boilerplate                        spread 0.002   top1 12%

12.5% top1 over 8 blocks is EXACTLY random, and the mean margin is negative --
the right block scores *below* the best distractor. An inter-block cosine
spread of 0.002 says why: the vectors are indistinguishable. When blocks share
most of their tokens, the embedding is dominated by the shared boilerplate and
the discriminating information (dates, hosts, identifiers) is precisely what a
384-dimension sentence vector compresses away. This is arithmetic, not tuning:
no better prefix string fixes it, which is why the augmentation A/B came out
flat.

The consequence is worse than a weak leg. RRF still fuses the vector ranking in
at full weight, so noise is blended with BM25's real signal and the result is
labelled "hybrid". The label lies, and nothing in the system says so.

So: measure the spread from the vectors ALREADY CACHED in the index (no model
load, no re-embedding), and when it falls below the floor, drop the vector leg's
fusion weight to zero and SAY SO in the degraded marker and in
``retrieval_diagnostics``. A governed store that can prove its recall claims
should refuse to fake a hybrid.

This is a read-side honesty gauge, not a fix. The fix is an embed-vs-store split
(embed a gist, keep slot values as filterable siblings) so near-duplicate
records stop colliding in the first place.
"""

from __future__ import annotations

import math
import os
import sqlite3
import statistics
import struct
from dataclasses import dataclass

from .observability import get_logger

_log = get_logger("vector_inertness")

#: Below this inter-block cosine standard deviation the vector ranking is
#: treated as carrying no usable signal. Chosen from the measurement above:
#: the inert corpus sits at 0.002 and a healthy prose corpus at 0.108, so this
#: floor is an order of magnitude clear of BOTH. It is deliberately not tuned
#: to sit near either -- a threshold close to observed data is a threshold that
#: flips on noise.
DEFAULT_SPREAD_FLOOR = 0.02

#: Below this many vectors the spread is not a meaningful statistic and the
#: gauge abstains rather than guessing. Refusing to answer is the honest
#: outcome for a sample too small to support one.
MIN_SAMPLE = 8

#: Cap on vectors read for the measurement. The spread of a random sample
#: estimates the population spread; reading an entire large index to compute
#: one standard deviation would make recall pay for the diagnosis.
MAX_SAMPLE = 256


@dataclass(frozen=True)
class InertnessReport:
    """What the gauge concluded, and enough to argue with it."""

    inert: bool
    spread: float | None
    sampled: int
    floor: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "inert": self.inert,
            "spread": None if self.spread is None else round(self.spread, 6),
            "sampled": self.sampled,
            "floor": self.floor,
            "reason": self.reason,
        }


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity, 0.0 for a degenerate or mismatched pair."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def spread_of(vectors: list[list[float]]) -> float:
    """Population standard deviation of all pairwise cosine similarities.

    The discriminating statistic. A high MEAN similarity is fine on its own --
    a corpus can be topically tight and still rank correctly. What kills
    ranking is low VARIANCE: when every pair is equally similar, the ordering
    is arbitrary.
    """
    sims = [cosine(vectors[i], vectors[j]) for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
    return statistics.pstdev(sims) if len(sims) > 1 else 0.0


def _db_path(workspace: str) -> str:
    return os.path.join(workspace, ".mind-mem-index", "recall.db")


def sample_cached_vectors(workspace: str, *, limit: int = MAX_SAMPLE) -> list[list[float]]:
    """Read up to *limit* embeddings already cached for this workspace.

    Plain sqlite3: ``embedding_cache`` is an ordinary table, so this needs
    neither the sqlite-vec extension nor an embedding model. The gauge must be
    cheap enough to run on the recall path; anything that loads a model there
    would cost more than the leg it is judging.
    """
    path = _db_path(workspace)
    if not os.path.isfile(path):
        return []
    conn = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT embedding, dimension FROM embedding_cache ORDER BY block_id LIMIT ?",
            (int(limit),),
        ).fetchall()
    except sqlite3.Error:
        # No cache table yet, or an unreadable index. Not an error condition:
        # the caller gets an abstention, never a fabricated verdict.
        return []
    finally:
        if conn is not None:
            conn.close()

    out: list[list[float]] = []
    for blob, dim in rows:
        try:
            out.append(list(struct.unpack(f"{int(dim)}f", blob)))
        except (struct.error, TypeError):
            continue
    return out


def measure(workspace: str, *, floor: float = DEFAULT_SPREAD_FLOOR) -> InertnessReport:
    """Is this workspace's vector leg carrying signal?

    Abstains (``inert=False``) whenever it cannot tell. A gauge that guesses
    "inert" on thin evidence would silently disable a working retrieval leg,
    which is a worse failure than the one it exists to catch.
    """
    vectors = sample_cached_vectors(workspace)
    if len(vectors) < MIN_SAMPLE:
        return InertnessReport(
            inert=False,
            spread=None,
            sampled=len(vectors),
            floor=floor,
            reason=(
                f"not enough cached vectors to judge ({len(vectors)} < {MIN_SAMPLE}); "
                "gauge abstains rather than disabling a leg on thin evidence"
            ),
        )
    spread = spread_of(vectors)
    if spread < floor:
        return InertnessReport(
            inert=True,
            spread=spread,
            sampled=len(vectors),
            floor=floor,
            reason=(
                f"inter-block cosine spread {spread:.4f} < floor {floor}: the vectors "
                "are effectively indistinguishable, so the vector ranking is noise. "
                "Vector leg dropped from fusion; results are BM25-only and say so."
            ),
        )
    return InertnessReport(
        inert=False,
        spread=spread,
        sampled=len(vectors),
        floor=floor,
        reason=f"inter-block cosine spread {spread:.4f} >= floor {floor}: vector leg carries signal",
    )


# --- cached view for the recall path ---------------------------------------
#
# `measure` reads up to 256 vectors and does O(n^2) cosines. That is cheap in
# absolute terms but not free, and recall must not pay it per query. The cache
# is keyed on the index file's (size, mtime_ns), so a rebuilt or appended index
# re-measures automatically while a quiet index answers from memory. No TTL:
# a clock would make the gauge time-dependent, and this store keeps clocks off
# the deterministic paths.
_CACHE: dict[str, tuple[tuple[int, int], InertnessReport]] = {}
_CACHE_MAX = 32


def _index_stamp(workspace: str) -> tuple[int, int] | None:
    try:
        st = os.stat(_db_path(workspace))
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns)


def inertness_for(workspace: str, *, floor: float = DEFAULT_SPREAD_FLOOR) -> InertnessReport:
    """Cached :func:`measure`, invalidated by any change to the index file."""
    stamp = _index_stamp(workspace)
    if stamp is None:
        return measure(workspace, floor=floor)

    cached = _CACHE.get(workspace)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    report = measure(workspace, floor=floor)
    if len(_CACHE) >= _CACHE_MAX and workspace not in _CACHE:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[workspace] = (stamp, report)
    if report.inert:
        _log.warning(
            "vector_leg_inert",
            workspace=workspace,
            spread=report.spread,
            floor=report.floor,
            sampled=report.sampled,
        )
    return report


def reset_cache() -> None:
    """Drop the memoised reports (tests, and after a deliberate reindex)."""
    _CACHE.clear()
