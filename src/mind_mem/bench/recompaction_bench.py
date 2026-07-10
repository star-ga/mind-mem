#!/usr/bin/env python3
"""Recompaction benchmark — the scalar metric downstream tooling depends on.

Answers one question with a number: does iterative re-compression
(``mind_mem.recompaction.recompact_cluster``) tighten a cluster of related
blocks *without losing facts*? The final line printed by :func:`main` is
machine-greppable::

    recompaction_score: <float>

``recompaction_score = fact_retention * convergence_rate`` — a compressor
that never converges scores 0; one that converges but drops facts scores low.

Fact retention is a **regex-derived fact-retention proxy, not an LLM judge**:
probes (numbers, quoted identifiers, capitalized entities, dates) are
extracted from the source blocks via regex, then checked for token-level
presence in the rewrite. What this misses: a semantically faithful paraphrase
of a fact (e.g. "1,469 blocks" -> "roughly fifteen hundred blocks") counts as
a *loss* under this proxy. That is a known conservative bias — for a
data-loss check, conservative is the right direction; it can only
under-credit a compressor, never over-credit one.

Clustering is deterministic kNN over the ``vec_blocks`` sqlite-vec table:
blocks are visited in sorted-id order, each seeds a cluster from its k
nearest neighbors, and every block is assigned to at most one cluster — no
RNG, no reliance on wall-clock or insertion order, so runs are comparable.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Any

from ..compressors import CompressorError, EchoCompressor, OllamaCompressor
from ..observability import get_logger, metrics
from ..recompaction import (
    Compressor,
    NonConvergenceError,
    RecompactionConfig,
    recompact_cluster,
)

_log = get_logger("recompaction_bench")

_DEFAULT_K = 5
_DEFAULT_MIN_SIZE = 2
_SIMILARITY_FLOOR = 0.55  # cosine similarity; below this, neighbors don't cluster


# --- fact-retention proxy ----------------------------------------------------

_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_QUOTED_RE = re.compile(r'"([^"]{2,80})"')
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9]*(?:-[A-Za-z0-9]+)*|[A-Z]{2,}(?:-[A-Za-z0-9]+)*)\b")

# Common capitalized function words that are not salient facts on their own —
# excluded so the probe set isn't dominated by sentence-initial "The", "A".
_ENTITY_STOPWORDS = frozenset({"The", "A", "An", "This", "That", "It", "In", "On", "At", "For", "With", "Is", "Was", "Are"})


def extract_probes(text: str) -> set[str]:
    """Deterministically extract salient facts from *text* via regex.

    Returns numbers, quoted identifiers, ISO dates, and capitalized entities
    (including hyphenated block-ID-shaped tokens like ``PRJ-mind``). This is
    the fact-retention proxy described in the module docstring — see there
    for what it misses.
    """
    probes: set[str] = set()
    probes.update(_NUMBER_RE.findall(text))
    probes.update(_QUOTED_RE.findall(text))
    probes.update(_DATE_RE.findall(text))
    for m in _ENTITY_RE.findall(text):
        if m not in _ENTITY_STOPWORDS and len(m) > 1:
            probes.add(m)
    return probes


def probes_present(probes: set[str], text: str) -> float:
    """Fraction of *probes* that still appear as substrings of *text*.

    Vacuously ``1.0`` when there are no probes to check (nothing to lose).
    """
    if not probes:
        return 1.0
    hits = sum(1 for p in probes if p in text)
    return hits / len(probes)


# --- deterministic clustering -------------------------------------------------


def _open_readonly(db_path: str) -> sqlite3.Connection:
    """Open *db_path* strictly read-only. Raises if the file does not exist.

    ``mode=ro`` refuses to create a new file, which is what turns a missing
    path into a clear error instead of a silently-empty benchmark run.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    # Cheap existence probe: mode=ro on a genuinely absent file raises
    # OperationalError only once a statement is executed, not on connect.
    conn.execute("SELECT 1")
    return conn


def _vec_table_has_extension(conn: sqlite3.Connection) -> bool:
    """Try to load the sqlite-vec extension and confirm `vec_blocks` is queryable.

    Returns False (never raises) on any failure — a missing/unloadable
    extension, or a `vec_blocks` table that doesn't exist, is a legitimate
    reason to fall back to lexical clustering, not a benchmark crash.
    """
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("SELECT block_id FROM vec_blocks LIMIT 1")
        return True
    except Exception:  # noqa: BLE001 - any failure means "no vector path", by design
        return False


def _block_dict(conn: sqlite3.Connection, block_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT json_blob FROM blocks WHERE id = ?", (block_id,)).fetchone()
    if row is None or not row[0]:
        return None
    try:
        blob = json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return None
    return blob if isinstance(blob, dict) else None


def _cluster_via_vectors(conn: sqlite3.Connection, k: int, min_size: int) -> list[list[dict[str, Any]]]:
    """Deterministic greedy kNN sweep over the `vec_blocks` sqlite-vec table.

    Block IDs are visited in sorted order; each unassigned block seeds a
    cluster from its *k* nearest neighbors above :data:`_SIMILARITY_FLOOR`
    cosine similarity that are also unassigned. Every block belongs to at
    most one cluster.
    """
    all_ids = [r[0] for r in conn.execute("SELECT block_id FROM vec_blocks ORDER BY block_id")]
    embeddings: dict[str, bytes] = dict(conn.execute("SELECT block_id, embedding FROM vec_blocks"))

    assigned: set[str] = set()
    id_clusters: list[list[str]] = []
    for anchor in all_ids:
        if anchor in assigned:
            continue
        neighbor_rows = conn.execute(
            "SELECT block_id, distance FROM vec_blocks WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (embeddings[anchor], k + 1),
        ).fetchall()
        # sqlite-vec only allows a single `ORDER BY distance` clause — break
        # ties on block_id in Python so the result stays fully deterministic
        # even when two neighbors are equidistant.
        neighbor_rows.sort(key=lambda row: (row[1], row[0]))
        member_ids = [anchor]
        for bid, distance in neighbor_rows:
            if bid == anchor or bid in assigned:
                continue
            similarity = 1.0 - distance / 2.0
            if similarity >= _SIMILARITY_FLOOR:
                member_ids.append(bid)
        if len(member_ids) >= min_size:
            id_clusters.append(member_ids)
            assigned.update(member_ids)

    return _materialize(conn, id_clusters, min_size)


_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_STOP_TOKENS = frozenset({"the", "and", "for", "with", "this", "that", "was", "were", "are", "not"})


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP_TOKENS)


def _cluster_via_lexical_overlap(conn: sqlite3.Connection, k: int, min_size: int) -> list[list[dict[str, Any]]]:
    """Deterministic fallback clustering when the vector path is unavailable.

    # deferred: this is a lexical (Jaccard token-overlap) stand-in for the
    # real semantic kNN path in `_cluster_via_vectors`, used only when the
    # sqlite-vec extension cannot be loaded. Upgrade path: none needed once
    # sqlite-vec is loadable — always prefer `_cluster_via_vectors` when the
    # extension check in `_vec_table_has_extension` succeeds.
    #
    # Same greedy-sweep shape as the vector path (sorted anchors, at-most-one
    # cluster per block, no RNG) so the two are comparable, just scored by
    # Jaccard token overlap of each block's `json_blob` text instead of
    # cosine similarity of an embedding.
    """
    rows = conn.execute("SELECT id, json_blob FROM blocks ORDER BY id").fetchall()
    tokens: dict[str, frozenset[str]] = {}
    ids: list[str] = []
    for block_id, json_blob in rows:
        if not json_blob:
            continue
        try:
            blob = json.loads(json_blob)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(blob, dict):
            continue
        # Exclude private `_`-prefixed keys (`_id`, `_source_file`, ...) —
        # same convention as `_block_scan_text` in dream_cycle.py. Otherwise
        # every block sharing the same source file would spuriously overlap.
        text = " ".join(str(v) for k, v in blob.items() if not k.startswith("_") and isinstance(v, (str, int, float)))
        tok = _tokenize(text)
        if tok:
            tokens[block_id] = tok
            ids.append(block_id)

    assigned: set[str] = set()
    clusters: list[list[str]] = []
    for anchor in ids:
        if anchor in assigned or not tokens[anchor]:
            continue
        scored: list[tuple[float, str]] = []
        for other in ids:
            if other == anchor or other in assigned:
                continue
            # `union` is never empty here: both `tokens[anchor]` and
            # `tokens[other]` are only populated (see the `ids` build loop
            # above) when their token set is non-empty.
            union = tokens[anchor] | tokens[other]
            jaccard = len(tokens[anchor] & tokens[other]) / len(union)
            if jaccard > 0.0:
                scored.append((jaccard, other))
        scored.sort(key=lambda t: (-t[0], t[1]))
        member_ids = [anchor] + [bid for _, bid in scored[:k]]
        if len(member_ids) >= min_size:
            clusters.append(member_ids)
            assigned.update(member_ids)

    return _materialize(conn, clusters, min_size)


def _materialize(conn: sqlite3.Connection, id_clusters: list[list[str]], min_size: int) -> list[list[dict[str, Any]]]:
    """Resolve block-id clusters to sorted block-dict clusters, dropping undersized ones."""
    clusters: list[list[dict[str, Any]]] = []
    for member_ids in id_clusters:
        members: list[dict[str, Any]] = []
        for bid in sorted(set(member_ids)):
            block = _block_dict(conn, bid)
            if block is not None:
                members.append(block)
        if len(members) >= min_size:
            clusters.append(members)
    clusters.sort(key=lambda c: tuple(sorted(str(b.get("_id", "")) for b in c)))
    return clusters


def load_clusters(db_path: str, k: int, min_size: int = _DEFAULT_MIN_SIZE) -> list[list[dict[str, Any]]]:
    """Load deterministic clusters of related blocks from a recall.db-shaped sqlite file.

    Opens *db_path* strictly read-only (never mutates the source corpus).
    Prefers semantic kNN clustering over the ``vec_blocks`` sqlite-vec
    virtual table (see :func:`_cluster_via_vectors`). If the sqlite-vec
    extension cannot be loaded, or the corpus has no vector index built,
    falls back to deterministic lexical (Jaccard token-overlap) clustering
    over ``blocks.json_blob`` (see :func:`_cluster_via_lexical_overlap`) —
    this is a **known-degraded** fallback: it groups blocks that share
    surface vocabulary, not blocks that are semantically related but
    phrased differently. Either path is deterministic (sorted traversal, no
    RNG), so results are comparable across repeated runs of the same corpus.

    Raises:
        sqlite3.OperationalError: if *db_path* does not exist or is not a
            valid sqlite database (surfaced by the read-only open probe).
    """
    conn = _open_readonly(db_path)
    try:
        if _vec_table_has_extension(conn):
            clusters = _cluster_via_vectors(conn, k, min_size)
            if clusters:
                return clusters
            _log.info("recompaction_bench_vector_clusters_empty", fallback="lexical")
        else:
            _log.info("recompaction_bench_sqlite_vec_unavailable", fallback="lexical")
        return _cluster_via_lexical_overlap(conn, k, min_size)
    finally:
        conn.close()


# --- evaluation ---------------------------------------------------------


@dataclass(frozen=True)
class ClusterRecord:
    """Per-cluster outcome of a benchmark run."""

    source_ids: tuple[str, ...]
    converged: bool
    iterations: int
    changed: bool
    fact_retention: float
    compression_ratio: float
    failure_reason: str | None = None


@dataclass(frozen=True)
class BenchResult:
    """Aggregate benchmark outcome over a set of clusters. Immutable."""

    n_clusters: int
    n_failures: int
    convergence_rate: float
    mean_iterations: float
    fact_retention: float
    compression_ratio: float
    records: tuple[ClusterRecord, ...] = field(default=())

    @property
    def recompaction_score(self) -> float:
        """``fact_retention * convergence_rate`` — the single reported number."""
        return self.fact_retention * self.convergence_rate


def _concat_source_text(blocks: list[dict[str, Any]]) -> str:
    from ..recompaction import _block_body

    return "\n\n".join(_block_body(b) for b in blocks)


def _evaluate_one_cluster(
    blocks: list[dict[str, Any]],
    compressor: Compressor,
    config: RecompactionConfig,
) -> ClusterRecord:
    source_ids = tuple(str(b.get("_id", "")) for b in blocks)
    source_text = _concat_source_text(blocks)
    probes = extract_probes(source_text)

    try:
        result = recompact_cluster(blocks, compressor=compressor, config=config)
    except NonConvergenceError as exc:
        return ClusterRecord(
            source_ids=source_ids,
            converged=False,
            iterations=exc.iterations,
            changed=False,
            fact_retention=0.0,
            compression_ratio=0.0,
            failure_reason=f"non_convergence: {exc}",
        )
    except ValueError as exc:
        return ClusterRecord(
            source_ids=source_ids,
            converged=False,
            iterations=0,
            changed=False,
            fact_retention=0.0,
            compression_ratio=0.0,
            failure_reason=f"retention_floor: {exc}",
        )
    except CompressorError as exc:
        # An infrastructure fault (timeout, non-200, malformed response) is NOT
        # a verdict on the model's ability to converge. Recording it as a plain
        # cluster failure would let a slow GPU masquerade as "this compressor
        # cannot reach a fixed point" — a category error that would send us
        # retraining a model that was merely timing out. Kept as its own
        # `compressor_error:` reason so it is visible in the per-cluster records
        # and countable separately, and it does NOT abort the remaining clusters.
        metrics.inc("recompaction_bench_compressor_errors")
        _log.warning("recompaction_bench_compressor_error", error=str(exc))
        return ClusterRecord(
            source_ids=source_ids,
            converged=False,
            iterations=0,
            changed=False,
            fact_retention=0.0,
            compression_ratio=0.0,
            failure_reason=f"compressor_error: {exc}",
        )

    retention = probes_present(probes, result.text)
    ratio = (len(result.text) / len(source_text)) if source_text else 1.0
    return ClusterRecord(
        source_ids=source_ids,
        converged=result.converged,
        iterations=result.iterations,
        changed=result.changed,
        fact_retention=retention,
        compression_ratio=ratio,
        failure_reason=None,
    )


def evaluate(
    clusters: list[list[dict[str, Any]]],
    compressor: Compressor,
    config: RecompactionConfig,
) -> BenchResult:
    """Run ``recompact_cluster`` over every cluster and aggregate the outcome.

    A cluster that raises :class:`NonConvergenceError` or the retention-floor
    ``ValueError`` is recorded as a failure (``converged=False``,
    ``fact_retention=0.0``) rather than aborting the run — one bad cluster
    must not blank out the signal from the rest. Never mutates *clusters* or
    their block dicts.
    """
    if not clusters:
        return BenchResult(
            n_clusters=0,
            n_failures=0,
            convergence_rate=0.0,
            mean_iterations=0.0,
            fact_retention=0.0,
            compression_ratio=0.0,
            records=(),
        )

    records = [_evaluate_one_cluster([dict(b) for b in cluster], compressor, config) for cluster in clusters]

    n = len(records)
    n_failures = sum(1 for r in records if not r.converged)
    converged_records = [r for r in records if r.converged]

    result = BenchResult(
        n_clusters=n,
        n_failures=n_failures,
        convergence_rate=(n - n_failures) / n,
        # Averaged over CONVERGED clusters only. A retention-floor rejection
        # records iterations=0, and a non-convergence records the bound — mixing
        # either into the mean makes it a number about failures, not about how
        # many passes a fixed point actually costs. Worse, it deflates in the
        # reassuring direction: a compressor that fails every cluster would
        # report a low, healthy-looking iteration count. 0.0 here means "nothing
        # converged", which convergence_rate already says plainly.
        mean_iterations=(sum(r.iterations for r in converged_records) / len(converged_records))
        if converged_records
        else 0.0,
        fact_retention=(sum(r.fact_retention for r in converged_records) / len(converged_records)) if converged_records else 0.0,
        compression_ratio=(sum(r.compression_ratio for r in converged_records) / len(converged_records)) if converged_records else 0.0,
        records=tuple(records),
    )
    metrics.inc("recompaction_bench_runs")
    metrics.inc("recompaction_bench_failures", n_failures)
    _log.info(
        "recompaction_bench_complete",
        n_clusters=n,
        n_failures=n_failures,
        convergence_rate=result.convergence_rate,
        fact_retention=result.fact_retention,
        score=result.recompaction_score,
    )
    return result


# --- compressor selection + CLI ---------------------------------------------


def _build_compressor(model: str, host: str, seed: int) -> Compressor:
    if model == "echo":
        return EchoCompressor()
    if not model:
        raise ValueError("--model must be non-empty ('echo' or an ollama model tag)")
    return OllamaCompressor(model=model, host=host, seed=seed)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mind-mem recompaction benchmark")
    parser.add_argument("--db", required=True, help="Path to a recall.db-shaped sqlite file (opened read-only)")
    parser.add_argument("--model", required=True, help="'echo' for the control, else an ollama model tag")
    parser.add_argument("--host", default="http://localhost:11434", help="ollama host")
    parser.add_argument("--clusters", type=int, default=30, help="max number of clusters to evaluate")
    parser.add_argument("--k", type=int, default=_DEFAULT_K, help="kNN neighbor count per cluster seed")
    parser.add_argument("--min-size", type=int, default=_DEFAULT_MIN_SIZE, help="minimum blocks per cluster")
    parser.add_argument("--seed", type=int, default=0, help="fixed ollama sampling seed")
    parser.add_argument("--max-iterations", type=int, default=6, help="recompaction iteration bound")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point. Prints the machine-greppable `recompaction_score:` line."""
    args = _parse_args(argv)

    compressor = _build_compressor(args.model, args.host, args.seed)
    clusters = load_clusters(args.db, k=args.k, min_size=args.min_size)[: args.clusters]
    config = RecompactionConfig(max_iterations=args.max_iterations)

    result = evaluate(clusters, compressor, config)

    print(f"n_clusters: {result.n_clusters}")
    print(f"n_failures: {result.n_failures}")
    # Infrastructure faults are surfaced separately from model verdicts. A run
    # with n_compressor_errors > 0 measured the GPU, not the compressor — treat
    # its score as void, not as evidence the model cannot converge.
    n_errors = sum(1 for r in result.records if (r.failure_reason or "").startswith("compressor_error:"))
    print(f"n_compressor_errors: {n_errors}")
    if n_errors:
        print(
            f"WARNING: {n_errors}/{result.n_clusters} clusters hit a compressor "
            "infrastructure fault (timeout/HTTP). This score reflects the runtime, "
            "not the model's ability to reach a fixed point. Do not compare it.",
            file=sys.stderr,
        )
    print(f"convergence_rate: {result.convergence_rate}")
    print(f"fact_retention: {result.fact_retention}")
    print(f"mean_iterations: {result.mean_iterations}")
    print(f"compression_ratio: {result.compression_ratio}")
    print(f"recompaction_score: {result.recompaction_score}")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])


__all__ = [
    "BenchResult",
    "ClusterRecord",
    "evaluate",
    "extract_probes",
    "load_clusters",
    "main",
    "probes_present",
]
