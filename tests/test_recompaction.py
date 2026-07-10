#!/usr/bin/env python3
"""Tests for recompaction.py — iterative re-compression to a fixed point.

The compressor is injected, so every convergence property below is proven
deterministically with zero API calls. That is the point: the loop, the
fixed-point detection, and the safety gates are the artifact under test —
not the quality of any particular language model.
"""

from __future__ import annotations

import pytest

from mind_mem.recompaction import (
    NonConvergenceError,
    RecompactionConfig,
    cluster_digest,
    recompact_cluster,
)

# --- helpers: deterministic stand-in compressors ---------------------------


def _idempotent(text: str, blocks: list[dict]) -> str:
    """Converges on iteration 1 — returns its input unchanged."""
    return text


def _shrink_to(target: str, steps: int):
    """A compressor that takes exactly `steps` passes to reach `target`."""
    state = {"n": 0}

    def _c(text: str, blocks: list[dict]) -> str:
        state["n"] += 1
        return target if state["n"] >= steps else f"{text}."

    return _c


def _oscillating(text: str, blocks: list[dict]) -> str:
    """Never converges: flips between two states forever."""
    return "B" if text.startswith("A") else "A"


def _blocks(*bodies: str) -> list[dict]:
    return [{"_id": f"DEC-{i:03d}", "body": b} for i, b in enumerate(bodies, 1)]


# --- cluster_digest --------------------------------------------------------


def test_digest_is_order_independent():
    """Same blocks in any order produce the same digest.

    Load-bearing: a cluster is a *set*. If digest depended on ordering, the
    same cluster would re-compress on every run and never be seen as stable.
    """
    a = _blocks("alpha", "beta", "gamma")
    b = [a[2], a[0], a[1]]
    assert cluster_digest(a) == cluster_digest(b)


def test_digest_changes_when_content_changes():
    assert cluster_digest(_blocks("alpha")) != cluster_digest(_blocks("alphb"))


def test_digest_is_stable_across_calls():
    a = _blocks("x", "y")
    assert cluster_digest(a) == cluster_digest(a)


# --- convergence -----------------------------------------------------------


def test_idempotent_compressor_converges_on_first_pass():
    """A compressor that returns its input is a fixed point immediately."""
    res = recompact_cluster(_blocks("one", "two"), compressor=_idempotent)
    assert res.converged is True
    assert res.iterations == 1
    assert res.changed is False


def test_converges_at_the_iteration_the_output_stops_changing():
    """3 shrink steps, then stable -> loop stops at the *4th* pass.

    Pass 4 is the one that proves the fixed point: it re-reads the settled
    text and gets the same bytes back. A fixed *count* of 3 would have
    stopped without ever proving convergence.
    """
    res = recompact_cluster(
        _blocks("seed", "seed2"),
        compressor=_shrink_to("settled", steps=3),
        config=RecompactionConfig(max_iterations=10, min_retention_ratio=0.0),
    )
    assert res.converged is True
    assert res.iterations == 4
    assert res.text == "settled"
    assert res.changed is True


def test_non_convergent_compressor_raises_rather_than_silently_truncating():
    """An oscillating compressor must fail loudly at the bound.

    A silent `break` at max_iterations would hand back a non-converged result
    that *looks* identical to a converged one. That is the exact failure mode
    a fixed iteration count hides.
    """
    with pytest.raises(NonConvergenceError) as exc:
        recompact_cluster(
            _blocks("A", "A2"),
            compressor=_oscillating,
            config=RecompactionConfig(max_iterations=5, min_retention_ratio=0.0),
        )
    assert exc.value.iterations == 5
    assert "did not converge" in str(exc.value).lower()


def test_max_iterations_must_be_positive():
    with pytest.raises(ValueError):
        RecompactionConfig(max_iterations=0)


# --- safety gates ----------------------------------------------------------


def test_empty_cluster_is_a_no_op():
    res = recompact_cluster([], compressor=_idempotent)
    assert res.converged is True
    assert res.iterations == 0
    assert res.changed is False
    assert res.text == ""


def test_single_block_cluster_is_a_no_op():
    """Re-compressing one block against itself adds no sibling context.

    Guard against burning a model call to rewrite a block with no new
    information available to it.
    """
    res = recompact_cluster(_blocks("solo"), compressor=_idempotent)
    assert res.iterations == 0
    assert res.changed is False


def test_compressor_that_collapses_content_is_rejected():
    """A rewrite below the retention floor is a data-loss bug, not a win."""

    def _destroy(text: str, blocks: list[dict]) -> str:
        return "ok"

    with pytest.raises(ValueError, match="retention floor"):
        recompact_cluster(
            _blocks("a substantial block body", "another substantial block body"),
            compressor=_destroy,
            config=RecompactionConfig(min_retention_ratio=0.25),
        )


def test_result_carries_the_source_block_ids_for_provenance():
    res = recompact_cluster(_blocks("one", "two"), compressor=_idempotent)
    assert res.source_ids == ("DEC-001", "DEC-002")


def test_result_carries_input_and_output_digests():
    """A proposal must be able to prove which cluster state it was derived from."""
    blocks = _blocks("one", "two")
    res = recompact_cluster(blocks, compressor=_idempotent)
    assert res.input_digest == cluster_digest(blocks)
    assert len(res.output_digest) == 64  # sha256 hex


def test_recompaction_never_mutates_the_input_blocks():
    """Immutability: the source of truth is untouched. Approval is a separate step."""
    blocks = _blocks("one", "two")
    before = [dict(b) for b in blocks]
    recompact_cluster(blocks, compressor=_shrink_to("tight", steps=2))
    assert blocks == before


def test_compressor_receives_all_sibling_blocks_each_pass():
    """The whole premise: each pass re-reads with full cluster context."""
    seen: list[int] = []

    def _c(text: str, blocks: list[dict]) -> str:
        seen.append(len(blocks))
        return text

    recompact_cluster(_blocks("a", "b", "c"), compressor=_c)
    assert seen == [3]


def test_digest_of_a_converged_result_is_reproducible():
    """Re-running a converged cluster is a no-op — the loop is idempotent."""
    blocks = _blocks("one", "two")
    first = recompact_cluster(blocks, compressor=_shrink_to("final text here", steps=2))
    again = recompact_cluster(blocks, compressor=_shrink_to("final text here", steps=2))
    assert first.output_digest == again.output_digest


# --- ClusterDigest type ----------------------------------------------------


def test_cluster_digest_returns_hex_sha256():
    d = cluster_digest(_blocks("x"))
    assert isinstance(d, str) and len(d) == 64
    int(d, 16)  # parses as hex


def test_digest_handles_blocks_without_a_body_field():
    """Real store blocks carry named + list fields, not a single `body`.

    The digest must reconstruct scannable text from those fields (skipping
    private `_`-prefixed keys) so a Postgres-backed cluster hashes correctly.
    """
    structured = [
        {"_id": "DEC-001", "_source_file": "x.md", "summary": "chose Q16.16", "tags": ["a", "b"]},
        {"_id": "DEC-002", "summary": "chose append-only", "tags": ["c"]},
    ]
    d = cluster_digest(structured)
    assert len(d) == 64
    # _source_file (private) must not affect the digest
    same = [dict(structured[0], _source_file="other.md"), structured[1]]
    assert cluster_digest(same) == d


def test_recompaction_result_is_frozen():
    res = recompact_cluster(_blocks("a", "b"), compressor=_idempotent)
    with pytest.raises(Exception):
        res.text = "mutated"  # type: ignore[misc]
