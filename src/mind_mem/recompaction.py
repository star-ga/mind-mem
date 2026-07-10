#!/usr/bin/env python3
"""mind-mem Iterative Re-Compression ("sleep") Engine. Zero external deps.

A block written mid-session, rushed, or from partial understanding is frozen
at the fidelity of the single pass that produced it. This engine takes a
*cluster* of semantically-related blocks, re-reads them together, and
re-compresses to a tighter block — then repeats until the rewrite stops
changing. The output of that fixed point is a *proposal*; it never mutates the
source of truth (that stays HITL-gated through ``propose_update``).

Two properties make this safe to run unattended, and both are load-bearing:

1. **Fixed point, not fixed count.** The loop stops when a pass returns bytes
   identical to its input, or raises :class:`NonConvergenceError` at a bound.
   A "4 sleep loops" hyperparameter would hide whether the rewrite actually
   settled — the same discipline as the mic@1 self-host fixed-point gate.

2. **Injected compressor.** The language-model call is a ``Callable`` the
   caller supplies. The loop, convergence detection, order-independent cluster
   digest, and retention floor are therefore testable deterministically with
   zero API calls — which is what ``tests/test_recompaction.py`` does.

The compressor signature is ``(current_text: str, blocks: list[dict]) -> str``:
it receives the current best summary plus every sibling block in the cluster,
and returns a new summary. It must be pure w.r.t. its inputs for the fixed
point to mean anything (no clock, no RNG) — same rule as the evidence-chain
preimages.

Usage (with a real model injected by the caller):

    from mind_mem.recompaction import recompact_cluster
    res = recompact_cluster(cluster_blocks, compressor=my_llm_compress)
    if res.changed:
        propose_update(..., body=res.text, supersedes=res.source_ids)
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .observability import get_logger, metrics

_log = get_logger("recompaction")

# A compressor re-reads (current_summary, all_sibling_blocks) -> new_summary.
Compressor = Callable[[str, "list[dict[str, Any]]"], str]

# Type alias documenting that a cluster digest is a hex sha256 string.
ClusterDigest = str

_DIGEST_LEN = 64  # sha256 hex


def _block_body(block: dict[str, Any]) -> str:
    """Extract the scannable text body from a store block dict.

    Prefers an explicit ``body`` field, else joins the public (non ``_``)
    field values — the same convention dream_cycle uses to reconstruct
    scannable text from a structured block.
    """
    if "body" in block and isinstance(block["body"], str):
        return block["body"]
    parts: list[str] = []
    for key, val in block.items():
        if key.startswith("_"):
            continue
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, (list, tuple)):
            parts.extend(str(v) for v in val)
    return "\n".join(parts)


def cluster_digest(blocks: list[dict[str, Any]]) -> ClusterDigest:
    """Order-independent sha256 over a cluster's block bodies.

    A cluster is a *set* of blocks; its identity must not depend on the order
    they were retrieved in, or the same cluster would re-compress on every run
    and never register as stable. We hash the sorted per-block body digests so
    permutation is invisible but any content change is not.
    """
    per_block = sorted(hashlib.sha256(_block_body(b).encode("utf-8")).hexdigest() for b in blocks)
    h = hashlib.sha256()
    for d in per_block:
        h.update(d.encode("ascii"))
    return h.hexdigest()


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class NonConvergenceError(RuntimeError):
    """Raised when re-compression does not reach a fixed point within the bound.

    Failing loudly here is the whole safety argument: a silent stop at the
    iteration cap would return a non-converged rewrite that is byte-shaped
    exactly like a converged one, and the caller would propose it as if it had
    settled.
    """

    def __init__(self, iterations: int):
        self.iterations = iterations
        super().__init__(
            f"re-compression did not converge within {iterations} iterations; "
            "refusing to emit a non-fixed-point rewrite"
        )


@dataclass(frozen=True)
class RecompactionConfig:
    """Bounds and floors for a re-compression run.

    Args:
        max_iterations: Hard cap on passes. Reaching it raises
            :class:`NonConvergenceError` rather than returning a partial result.
        min_retention_ratio: The converged text must be at least this fraction
            of the *largest single source block* body length. A rewrite that
            collapses below the floor is treated as data loss and rejected —
            re-compression tightens, it does not delete.
    """

    max_iterations: int = 6
    min_retention_ratio: float = 0.25

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if not 0.0 <= self.min_retention_ratio <= 1.0:
            raise ValueError("min_retention_ratio must be in [0.0, 1.0]")


@dataclass(frozen=True)
class RecompactionResult:
    """The outcome of re-compressing one cluster to a fixed point.

    ``text`` is a *proposal*: applying it to the source of truth is a separate,
    HITL-gated step. ``source_ids`` + ``input_digest`` let the approval layer
    prove exactly which cluster state this rewrite was derived from.
    """

    text: str
    converged: bool
    iterations: int
    changed: bool
    source_ids: tuple[str, ...]
    input_digest: ClusterDigest
    output_digest: str
    trajectory: tuple[str, ...] = field(default=())


def recompact_cluster(
    blocks: list[dict[str, Any]],
    *,
    compressor: Compressor,
    config: RecompactionConfig | None = None,
    seed_text: str | None = None,
) -> RecompactionResult:
    """Iteratively re-compress a block cluster to a fixed point.

    The input ``blocks`` list and its dicts are never mutated — the source of
    truth is untouched; the result is a proposal for the HITL gate.

    Args:
        blocks: The semantically-related cluster (typically from ``find_similar``).
        compressor: ``(current_text, blocks) -> new_text``. Injected so the loop
            is testable with no model. Must be pure w.r.t. its inputs.
        config: Bounds/floors. Defaults to :class:`RecompactionConfig`.
        seed_text: Starting summary. Defaults to the concatenated block bodies,
            i.e. "read everything, then compress".

    Returns:
        A frozen :class:`RecompactionResult`.

    Raises:
        NonConvergenceError: if no fixed point is reached within
            ``config.max_iterations``.
        ValueError: if the converged text falls below the retention floor.
    """
    cfg = config or RecompactionConfig()
    source_ids = tuple(str(b.get("_id", "")) for b in blocks)
    input_digest = cluster_digest(blocks)

    # Safety gate: nothing to consolidate with fewer than two blocks. A single
    # block has no sibling context, so a rewrite adds no information — skip it
    # rather than burn a model call.
    if len(blocks) < 2:
        text = seed_text if seed_text is not None else (_block_body(blocks[0]) if blocks else "")
        return RecompactionResult(
            text=text,
            converged=True,
            iterations=0,
            changed=False,
            source_ids=source_ids,
            input_digest=input_digest,
            output_digest=_text_digest(text),
        )

    current = seed_text if seed_text is not None else "\n\n".join(_block_body(b) for b in blocks)
    original = current
    trajectory: list[str] = []

    converged = False
    iterations = 0
    for _ in range(cfg.max_iterations):
        iterations += 1
        rewritten = compressor(current, blocks)
        trajectory.append(rewritten)
        if rewritten == current:
            # Fixed point: re-reading the settled text returned the same bytes.
            converged = True
            break
        current = rewritten

    if not converged:
        metrics.inc("recompaction_non_convergence")
        _log.warning("recompaction_non_convergence", iterations=iterations, cluster=input_digest[:12])
        raise NonConvergenceError(iterations)

    # Retention floor: a converged rewrite must not collapse the cluster's
    # information. Compare against the largest single source body — tightening
    # many blocks into one below a quarter of the biggest is data loss.
    largest = max((len(_block_body(b)) for b in blocks), default=0)
    if largest > 0 and len(current) < cfg.min_retention_ratio * largest:
        metrics.inc("recompaction_retention_floor_reject")
        _log.warning(
            "recompaction_retention_floor",
            cluster=input_digest[:12],
            out_len=len(current),
            floor=cfg.min_retention_ratio * largest,
        )
        raise ValueError(
            f"re-compression fell below retention floor "
            f"({len(current)} < {cfg.min_retention_ratio:.2f} * {largest}); "
            "treating as data loss, not a win"
        )

    metrics.inc("recompaction_converged")
    _log.info(
        "recompaction_converged",
        cluster=input_digest[:12],
        iterations=iterations,
        changed=current != original,
    )
    return RecompactionResult(
        text=current,
        converged=True,
        iterations=iterations,
        changed=current != original,
        source_ids=source_ids,
        input_digest=input_digest,
        output_digest=_text_digest(current),
        trajectory=tuple(trajectory),
    )


__all__ = [
    "ClusterDigest",
    "Compressor",
    "NonConvergenceError",
    "RecompactionConfig",
    "RecompactionResult",
    "cluster_digest",
    "recompact_cluster",
]
