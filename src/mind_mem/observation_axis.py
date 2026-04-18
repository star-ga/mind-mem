# Copyright 2026 STARGA, Inc.
"""Observer-Dependent Cognition (ODC) — axis-aware retrieval primitives.

Every recall in mind-mem has always used multiple retrieval signals (BM25,
vector similarity, recency, cross-encoder rerank, entity-graph expansion).
ODC formalises this: callers *declare* which axes are in play, results
carry per-axis confidence metadata, and the system can *rotate* to
orthogonal axes when initial confidence is low.

Six axes are defined:

    LEXICAL        — BM25F term matching with Porter stemming
    SEMANTIC       — vector similarity over embeddings
    TEMPORAL       — recency decay plus date-range filters
    ENTITY_GRAPH   — typed-edge traversal from entity mentions
    CONTRADICTION  — retrieval biased toward opposing / superseded blocks
    ADVERSARIAL    — deliberate counter-axis query to surface conflicts

The module intentionally contains no I/O or retrieval logic. It exposes
value objects and pure helpers so the fixed-point retrieval pipeline
(``hybrid_recall``) and the MCP surface (``recall_with_axis``) can
consume a stable API without circular dependencies.

Zero external deps — dataclasses, enum, math (all stdlib).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Optional

# ---------------------------------------------------------------------------
# ObservationAxis enum
# ---------------------------------------------------------------------------


class ObservationAxis(str, Enum):
    """Enumerates the retrieval axes a caller can select or weight.

    Inherits from ``str`` so axis names serialise cleanly into JSON (MCP
    responses, evidence metadata) without a custom encoder.
    """

    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    ENTITY_GRAPH = "entity_graph"
    CONTRADICTION = "contradiction"
    ADVERSARIAL = "adversarial"

    @classmethod
    def from_str(cls, name: str) -> "ObservationAxis":
        """Case-insensitive, hyphen-tolerant lookup by name.

        Callers receive axis names from user-controlled inputs (MCP tool
        args, config JSON) so the parser accepts common spellings:
        ``entity-graph``, ``entity_graph``, ``ENTITY_GRAPH`` all resolve
        to :attr:`ENTITY_GRAPH`.
        """
        normalised = name.strip().lower().replace("-", "_")
        for axis in cls:
            if axis.value == normalised:
                return axis
        valid = ", ".join(a.value for a in cls)
        raise ValueError(f"Unknown observation axis: {name!r}. Valid: {valid}")


# ---------------------------------------------------------------------------
# AxisWeights — per-axis weight vector for hybrid fusion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisWeights:
    """Immutable weight vector over the observation axes.

    Weights do not need to sum to 1.0 — the fusion layer will normalise
    when combining. A zero weight disables an axis entirely.
    """

    lexical: float = 1.0
    semantic: float = 1.0
    temporal: float = 0.0
    entity_graph: float = 0.0
    contradiction: float = 0.0
    adversarial: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "lexical",
            "semantic",
            "temporal",
            "entity_graph",
            "contradiction",
            "adversarial",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value):
                raise ValueError(f"Axis weight {field_name!r} must be finite, got {value!r}")
            if value < 0:
                raise ValueError(f"Axis weight {field_name!r} must be non-negative, got {value!r}")

    def as_dict(self) -> dict[str, float]:
        """Serialise to a JSON-friendly dict keyed by axis name."""
        return {
            ObservationAxis.LEXICAL.value: self.lexical,
            ObservationAxis.SEMANTIC.value: self.semantic,
            ObservationAxis.TEMPORAL.value: self.temporal,
            ObservationAxis.ENTITY_GRAPH.value: self.entity_graph,
            ObservationAxis.CONTRADICTION.value: self.contradiction,
            ObservationAxis.ADVERSARIAL.value: self.adversarial,
        }

    def active_axes(self) -> tuple[ObservationAxis, ...]:
        """Return the set of axes with non-zero weight, in axis order."""
        mapping: list[tuple[ObservationAxis, float]] = [
            (ObservationAxis.LEXICAL, self.lexical),
            (ObservationAxis.SEMANTIC, self.semantic),
            (ObservationAxis.TEMPORAL, self.temporal),
            (ObservationAxis.ENTITY_GRAPH, self.entity_graph),
            (ObservationAxis.CONTRADICTION, self.contradiction),
            (ObservationAxis.ADVERSARIAL, self.adversarial),
        ]
        return tuple(ax for ax, w in mapping if w > 0.0)

    def normalised(self) -> "AxisWeights":
        """Return a copy where weights sum to 1.0 (or all-zero stays all-zero)."""
        total = self.lexical + self.semantic + self.temporal + self.entity_graph + self.contradiction + self.adversarial
        if total <= 0:
            return self
        return AxisWeights(
            lexical=self.lexical / total,
            semantic=self.semantic / total,
            temporal=self.temporal / total,
            entity_graph=self.entity_graph / total,
            contradiction=self.contradiction / total,
            adversarial=self.adversarial / total,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float]) -> "AxisWeights":
        """Construct from a dict keyed by axis name.

        Keys are resolved via :meth:`ObservationAxis.from_str` so the same
        hyphen-tolerant parsing applies. Missing axes default to 0.0.
        """
        kwargs: dict[str, float] = {}
        for raw_key, value in mapping.items():
            axis = ObservationAxis.from_str(raw_key)
            kwargs[axis.value] = float(value)
        return cls(
            lexical=kwargs.get(ObservationAxis.LEXICAL.value, 0.0),
            semantic=kwargs.get(ObservationAxis.SEMANTIC.value, 0.0),
            temporal=kwargs.get(ObservationAxis.TEMPORAL.value, 0.0),
            entity_graph=kwargs.get(ObservationAxis.ENTITY_GRAPH.value, 0.0),
            contradiction=kwargs.get(ObservationAxis.CONTRADICTION.value, 0.0),
            adversarial=kwargs.get(ObservationAxis.ADVERSARIAL.value, 0.0),
        )

    @classmethod
    def uniform(cls, axes: Iterable[ObservationAxis]) -> "AxisWeights":
        """Uniform weighting over the given axes (1.0 each, rest 0.0)."""
        kwargs = {a.value: 1.0 for a in axes}
        return cls(
            lexical=kwargs.get(ObservationAxis.LEXICAL.value, 0.0),
            semantic=kwargs.get(ObservationAxis.SEMANTIC.value, 0.0),
            temporal=kwargs.get(ObservationAxis.TEMPORAL.value, 0.0),
            entity_graph=kwargs.get(ObservationAxis.ENTITY_GRAPH.value, 0.0),
            contradiction=kwargs.get(ObservationAxis.CONTRADICTION.value, 0.0),
            adversarial=kwargs.get(ObservationAxis.ADVERSARIAL.value, 0.0),
        )


# Default profile mirrors the pre-ODC retrieval behaviour: lexical + semantic
# carry the bulk of the signal, everything else is off unless the caller asks.
DEFAULT_WEIGHTS: AxisWeights = AxisWeights(lexical=1.0, semantic=1.0)


# ---------------------------------------------------------------------------
# AxisScore — per-axis contribution for a single result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisScore:
    """Per-axis contribution recorded for a single result.

    ``confidence`` is the axis's own normalised score for the result in
    [0, 1]. ``rank`` (1-based) is optional — some axes (like RRF inputs)
    track rank rather than raw score, so we keep both available.
    """

    axis: ObservationAxis
    confidence: float
    rank: Optional[int] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence):
            raise ValueError("AxisScore.confidence must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"AxisScore.confidence must be in [0, 1], got {self.confidence!r}")
        if self.rank is not None and self.rank < 1:
            raise ValueError(f"AxisScore.rank must be >=1, got {self.rank!r}")

    def as_dict(self) -> dict[str, object]:
        """Serialise to a JSON-friendly dict."""
        out: dict[str, object] = {
            "axis": self.axis.value,
            "confidence": round(self.confidence, 6),
        }
        if self.rank is not None:
            out["rank"] = self.rank
        return out


# ---------------------------------------------------------------------------
# Observation — collection of axis scores attached to a result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """The ODC "observation basis" for a single result.

    ``axes`` records every axis that produced the block (in contribution
    order). ``rotated`` is True when the result only appeared after the
    pipeline rotated to a secondary axis because initial confidence was
    below :data:`DEFAULT_ROTATION_THRESHOLD`.
    """

    axes: tuple[AxisScore, ...]
    rotated: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "axes": [a.as_dict() for a in self.axes],
            "rotated": self.rotated,
            "notes": list(self.notes),
        }

    def diversity(self) -> int:
        """Return the count of distinct axes that produced this result."""
        return len({a.axis for a in self.axes})

    def top_axis(self) -> Optional[AxisScore]:
        """Return the highest-confidence AxisScore, or None when empty."""
        if not self.axes:
            return None
        return max(self.axes, key=lambda a: a.confidence)


# ---------------------------------------------------------------------------
# Axis diversity metric
# ---------------------------------------------------------------------------


def axis_diversity(results: Iterable[Mapping[str, object]]) -> int:
    """Count distinct axes that contributed across a result set.

    Expects each result dict to carry an ``axes`` list (from
    :meth:`Observation.as_dict`). Missing or malformed entries are
    skipped silently so legacy BM25-only results don't poison the count.
    """
    seen: set[str] = set()
    for res in results:
        obs = res.get("observation") if isinstance(res, Mapping) else None
        if not isinstance(obs, Mapping):
            continue
        axes = obs.get("axes")
        if not isinstance(axes, list):
            continue
        for entry in axes:
            if isinstance(entry, Mapping):
                axis_name = entry.get("axis")
                if isinstance(axis_name, str):
                    seen.add(axis_name)
    return len(seen)


# ---------------------------------------------------------------------------
# Axis rotation
# ---------------------------------------------------------------------------


# Tuned conservatively: below 0.35 we assume the initial axes did not
# meaningfully disambiguate the query and a rotation is worth the extra
# retrieval cost. Callers can override per-request.
DEFAULT_ROTATION_THRESHOLD: float = 0.35


# Orthogonality map — for a given pivot axis, these are the axes considered
# sufficiently independent that rotating onto them is likely to surface
# different results. Lexical and semantic are complements (token vs
# embedding space). Temporal is orthogonal to everything. Entity-graph is
# orthogonal to lexical/semantic ranking. Contradiction and adversarial are
# deliberately separated because they answer different questions: "who
# disagrees with this?" vs "what would disprove this?".
_ORTHOGONAL_AXES: dict[ObservationAxis, tuple[ObservationAxis, ...]] = {
    ObservationAxis.LEXICAL: (
        ObservationAxis.SEMANTIC,
        ObservationAxis.ENTITY_GRAPH,
        ObservationAxis.TEMPORAL,
    ),
    ObservationAxis.SEMANTIC: (
        ObservationAxis.LEXICAL,
        ObservationAxis.ENTITY_GRAPH,
        ObservationAxis.TEMPORAL,
    ),
    ObservationAxis.TEMPORAL: (
        ObservationAxis.LEXICAL,
        ObservationAxis.SEMANTIC,
        ObservationAxis.ENTITY_GRAPH,
    ),
    ObservationAxis.ENTITY_GRAPH: (
        ObservationAxis.LEXICAL,
        ObservationAxis.SEMANTIC,
        ObservationAxis.CONTRADICTION,
    ),
    ObservationAxis.CONTRADICTION: (
        ObservationAxis.ADVERSARIAL,
        ObservationAxis.ENTITY_GRAPH,
    ),
    ObservationAxis.ADVERSARIAL: (
        ObservationAxis.CONTRADICTION,
        ObservationAxis.LEXICAL,
    ),
}


def rotate_axes(
    current: AxisWeights,
    *,
    already_tried: Optional[Iterable[ObservationAxis]] = None,
    max_new: int = 2,
) -> AxisWeights:
    """Suggest an orthogonal set of axes to rotate onto.

    Returns a new :class:`AxisWeights` with weight ``1.0`` on up to
    ``max_new`` axes that are orthogonal to the currently-active ones and
    have not yet been tried. When no new axes are available, returns the
    input weights unchanged so callers don't loop forever.
    """
    if max_new < 1:
        return current
    tried: set[ObservationAxis] = set(already_tried or ())
    tried.update(current.active_axes())

    candidates: list[ObservationAxis] = []
    for pivot in current.active_axes():
        for candidate in _ORTHOGONAL_AXES.get(pivot, ()):
            if candidate in tried or candidate in candidates:
                continue
            candidates.append(candidate)
            if len(candidates) >= max_new:
                break
        if len(candidates) >= max_new:
            break

    if not candidates:
        return current

    return AxisWeights.uniform(candidates)


def should_rotate(
    top_confidence: float,
    threshold: float = DEFAULT_ROTATION_THRESHOLD,
) -> bool:
    """Return True when the top-ranked result's confidence is below threshold.

    Centralised so both the retrieval pipeline and tests agree on the
    rotation trigger (fuzz the threshold in exactly one place).
    """
    if not math.isfinite(top_confidence):
        return True
    return top_confidence < threshold


# ---------------------------------------------------------------------------
# Adversarial axis pairing
# ---------------------------------------------------------------------------


# Adversarial retrieval runs the same query under an *opposing* basis so
# that surfaces like "evidence against X" show up alongside "evidence for
# X". The mapping tells the pipeline which axis to pair against for any
# given primary axis.
_ADVERSARIAL_PAIRS: dict[ObservationAxis, ObservationAxis] = {
    ObservationAxis.LEXICAL: ObservationAxis.CONTRADICTION,
    ObservationAxis.SEMANTIC: ObservationAxis.CONTRADICTION,
    ObservationAxis.TEMPORAL: ObservationAxis.CONTRADICTION,
    ObservationAxis.ENTITY_GRAPH: ObservationAxis.CONTRADICTION,
    ObservationAxis.CONTRADICTION: ObservationAxis.ADVERSARIAL,
}


def adversarial_pair(axis: ObservationAxis) -> ObservationAxis:
    """Return the opposing axis used for adversarial probing."""
    return _ADVERSARIAL_PAIRS.get(axis, ObservationAxis.ADVERSARIAL)


__all__ = [
    "ObservationAxis",
    "AxisWeights",
    "AxisScore",
    "Observation",
    "DEFAULT_WEIGHTS",
    "DEFAULT_ROTATION_THRESHOLD",
    "axis_diversity",
    "rotate_axes",
    "should_rotate",
    "adversarial_pair",
]
