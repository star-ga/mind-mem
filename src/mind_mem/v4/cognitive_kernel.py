"""v4 Cognitive Mind Kernel — composable retrieval strategies (Group A).

v3.x ships one fixed retrieval strategy: BM25 + vector + RRF fusion. Good
default. But different reasoning tasks want different memory routing:

    surprise_weighted    Returns blocks that semantically *contradict* the
                         rolling recall context. Useful when an agent needs
                         to catch its own wrong assumptions.

    lineage_first        Walks the v3.11 typed-edge graph
                         (cites/implements/refines/contradicts) over the
                         candidate set. Useful for tracing decision chains
                         and 'why is this the way it is'.

    recent_first         Temporal recency boost. Useful for 'what changed
                         lately' queries.

    contradicts_first    Surfaces lineage-contradiction edges before
                         consensus. Useful for hypothesis testing and
                         governance review.

    graph_walk           Multi-hop BFS retrieval over the lineage graph
                         starting from a seed match. Useful for following
                         dependency chains.

Default callers stay on v3 ``recall(query)`` — no kernel parameter, no
behaviour change. Power callers opt in:

    from mind_mem.v4.cognitive_kernel import mind_recall, KernelKind
    hits = mind_recall(workspace, query, kernel=KernelKind.LINEAGE_FIRST)

Kernels are **registered**, not hardcoded. The registry is module-level
and additive: third parties can call :func:`register_kernel` to add a
new strategy without forking. Built-in kernels register at import time.

This file ships the **registry, type surface, and a no-op default
kernel** that delegates to ``recall(query)`` unchanged. The five named
kernels above land as separate strategies in subsequent v4 commits;
registering them here without an implementation would mislead callers
into thinking the routing is live.

Feature-flag gated under ``v4.cognitive_kernel``. v3.x callers see no
behaviour change.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from .feature_flags import require_enabled

__all__ = [
    "FLAG",
    "KernelKind",
    "KernelHit",
    "KernelResult",
    "KernelStrategy",
    "register_kernel",
    "available_kernels",
    "mind_recall",
    "DEFAULT_KERNEL",
]


#: Feature-flag key in ``mind-mem.json: v4: {...}``.
FLAG: str = "cognitive_kernel"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class KernelKind(str, Enum):
    """Named retrieval strategies the kernel API can route through.

    The value strings match the ``kernel=`` argument users pass into
    :func:`mind_recall`. ``DEFAULT`` is the v3-compatible delegate that
    runs the existing recall path unchanged.
    """

    DEFAULT = "default"
    SURPRISE_WEIGHTED = "surprise_weighted"
    LINEAGE_FIRST = "lineage_first"
    RECENT_FIRST = "recent_first"
    CONTRADICTS_FIRST = "contradicts_first"
    GRAPH_WALK = "graph_walk"


@dataclass(frozen=True)
class KernelHit:
    """One ranked block returned by a kernel.

    Mirrors the shape of v3's recall hit — ``block_id`` + ``score`` —
    plus a ``reason`` tag the kernel attaches so callers can audit
    *why* a block surfaced under a specific strategy. Reasons are
    free-form strings; convention is ``"<kernel>:<short-tag>"``
    (e.g. ``"surprise_weighted:high_distance"``).
    """

    block_id: str
    score: float
    reason: str = ""


@dataclass(frozen=True)
class KernelResult:
    """Full kernel-recall result.

    Carries the ranked hits plus the kernel name and any per-call
    metadata the strategy chose to surface (e.g. surprise threshold,
    lineage hop count). Metadata is read-only and mostly intended for
    audit / debugging.
    """

    kernel: KernelKind
    hits: list[KernelHit] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)


#: A kernel strategy callable. Receives the workspace, the query, and
#: an arbitrary ``**kwargs`` bag (so future strategies can take
#: per-call tunables). Returns a :class:`KernelResult`.
KernelStrategy = Callable[..., KernelResult]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registry: dict[KernelKind, KernelStrategy] = {}
_registry_lock = threading.Lock()


def register_kernel(kind: KernelKind | str, strategy: KernelStrategy) -> None:
    """Register a kernel strategy under a kind.

    Replaces any previously-registered strategy for the same kind —
    useful for testing and for swap-in research kernels. Thread-safe.

    Raises :class:`FeatureDisabledError` if the flag is OFF.
    """
    require_enabled(FLAG)
    if isinstance(kind, str):
        kind = KernelKind(kind)
    with _registry_lock:
        _registry[kind] = strategy


def available_kernels() -> list[KernelKind]:
    """Return every currently-registered kernel.

    Order matches insertion. Includes the built-in :data:`DEFAULT_KERNEL`
    if it has been registered (always true after this module imports).

    Raises :class:`FeatureDisabledError` if the flag is OFF.
    """
    require_enabled(FLAG)
    with _registry_lock:
        return list(_registry.keys())


# ---------------------------------------------------------------------------
# Default delegate kernel
# ---------------------------------------------------------------------------


def _default_kernel(workspace: str, query: str, **_: Any) -> KernelResult:
    """Pass-through to v3 recall.

    Imports lazily so the v4 module doesn't pull the v3 recall stack at
    import time (keeps the v4 surface importable in a fresh checkout
    where the v3 recall path may not be initialised yet — useful for
    schema-only test runs).

    The returned :class:`KernelResult` carries no metadata and uses
    each hit's RRF score as :attr:`KernelHit.score`. Reason tags are
    empty since the default kernel adds no semantic routing.
    """
    require_enabled(FLAG)

    try:
        from mind_mem._recall_core import recall as _v3_recall
    except ImportError:
        # v3 recall not available in this build — return empty rather
        # than crash, so callers can detect the no-op case.
        return KernelResult(kernel=KernelKind.DEFAULT, hits=[], metadata={"degraded": True})

    raw = _v3_recall(workspace, query)
    hits: list[KernelHit] = []
    for h in raw or []:
        if isinstance(h, dict):
            bid = str(h.get("_id") or h.get("block_id") or "")
            score = float(h.get("rrf_score") or h.get("score") or 0.0)
        else:
            bid = getattr(h, "block_id", "") or getattr(h, "_id", "")
            score = float(getattr(h, "score", 0.0))
        if bid:
            hits.append(KernelHit(block_id=bid, score=score, reason=""))
    return KernelResult(kernel=KernelKind.DEFAULT, hits=hits, metadata={})


#: The pass-through kernel. Registered automatically at import time
#: under :attr:`KernelKind.DEFAULT`.
DEFAULT_KERNEL: KernelStrategy = _default_kernel


# Bootstrap the registry without going through ``register_kernel``
# (which requires the flag to be ON). The registry being populated at
# import time is independent of whether the flag is set; what the flag
# gates is *use* of the kernel API.
_registry[KernelKind.DEFAULT] = DEFAULT_KERNEL


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def mind_recall(
    workspace: str,
    query: str,
    *,
    kernel: KernelKind | str = KernelKind.DEFAULT,
    **kwargs: Any,
) -> KernelResult:
    """Route a recall through the named kernel strategy.

    The default kernel delegates to v3 ``recall(query)`` unchanged, so
    a flag-on workspace that calls ``mind_recall(...)`` without a
    kernel argument behaves identically to the v3 API.

    Unknown kernel names raise :class:`KeyError` with the list of
    registered kernels embedded in the message — fail-loud so callers
    catch typos early.

    Raises :class:`FeatureDisabledError` if the flag is OFF.
    """
    require_enabled(FLAG)
    if isinstance(kernel, str):
        kernel = KernelKind(kernel)
    with _registry_lock:
        strategy = _registry.get(kernel)
    if strategy is None:
        registered = sorted(k.value for k in _registry.keys())
        raise KeyError(f"no kernel registered under {kernel.value!r}; available: {registered}")
    return strategy(workspace, query, **kwargs)
