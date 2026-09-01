"""Shared config-resolver for retrieval features (architect audit item #6).

Every v3.3.0 retrieval feature shipped with the same pattern:

    def is_X_enabled(config) -> bool: ...
    def resolve_X_config(config) -> dict: ...

Five near-identical implementations drifted in small ways (default
auto_enable, validation of bounds, fallback paths). This module
collapses the pattern into :class:`FeatureGate` so new features land
as a single declaration and existing ones can migrate incrementally.

Backward-compat: each feature's original ``is_*_enabled`` and
``resolve_*_config`` are preserved as thin wrappers around a
FeatureGate instance. Existing callers keep working; new features
use FeatureGate directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

_AutoDetector = Callable[[str | None, list[dict] | None], bool]


@dataclass
class FieldSpec:
    """Bounds + coercion for a single knob in a feature's config."""

    default: Any
    coerce: Callable[[Any], Any] | None = None
    validate: Callable[[Any], bool] | None = None

    def resolve(self, raw: Any) -> Any:
        """Return ``raw`` coerced + validated, else ``default``."""
        if raw is None:
            return self.default
        value: Any = raw
        if self.coerce is not None:
            try:
                value = self.coerce(raw)
            except (TypeError, ValueError):
                return self.default
        if self.validate is not None and not self.validate(value):
            return self.default
        return value


@dataclass
class FeatureGate:
    """Declarative config gate for a retrieval feature.

    Example::

        GRAPH_EXPAND = FeatureGate(
            name="multi_hop",
            auto_detector=lambda query, results: (
                query is not None
                and detect_query_type(query) == "multi-hop"
            ),
            fields={
                "max_hops": FieldSpec(
                    default=2,
                    coerce=int,
                    validate=lambda v: 1 <= v <= 3,
                ),
                "decay": FieldSpec(
                    default=0.5,
                    coerce=float,
                    validate=lambda v: 0 < v <= 1,
                ),
            },
        )

        GRAPH_EXPAND.is_enabled(config, query="...", results=None)
        GRAPH_EXPAND.resolve(config)  # → {"max_hops": 2, "decay": 0.5}

    Every gate supports ``enabled`` (hard on), ``auto_enable``
    (default True) + an optional ``auto_detector`` for query-type /
    result-shape inference. The config lookup path is
    ``retrieval.<name>``.
    """

    name: str
    fields: dict[str, FieldSpec] = field(default_factory=dict)
    auto_detector: _AutoDetector | None = None
    auto_enable_default: bool = True

    def _section(self, config: dict[str, Any] | None) -> dict[str, Any] | None:
        """Return the ``retrieval.<name>`` dict or None when absent.

        An empty dict (``{}``) is a valid section — it means "use
        defaults" — and is distinct from a missing section.
        """
        if not config or not isinstance(config, dict):
            return None
        retrieval = config.get("retrieval", {})
        if not isinstance(retrieval, dict):
            return None
        if self.name not in retrieval:
            return None
        section = retrieval[self.name]
        return section if isinstance(section, dict) else None

    def is_enabled(
        self,
        config: dict[str, Any] | None,
        *,
        query: str | None = None,
        results: list[dict] | None = None,
    ) -> bool:
        section = self._section(config)
        if section is None:
            return False
        if section.get("enabled", False):
            return True
        if not section.get("auto_enable", self.auto_enable_default):
            return False
        if self.auto_detector is None:
            return False
        try:
            return bool(self.auto_detector(query, results))
        except Exception:
            return False

    def resolve(self, config: dict[str, Any] | None) -> dict[str, Any]:
        section = self._section(config) or {}
        return {name: spec.resolve(section.get(name)) for name, spec in self.fields.items()}


# ---------------------------------------------------------------------------
# Pre-baked detectors — reused across gates that condition on query type.
# ---------------------------------------------------------------------------


def multi_hop_detector(query: str | None, results: list[dict] | None) -> bool:
    if not query:
        return False
    try:
        from ._recall_detection import detect_query_type

        return detect_query_type(query) == "multi-hop"
    except Exception:
        return False


def multi_hop_or_temporal_detector(query: str | None, results: list[dict] | None) -> bool:
    if not query:
        return False
    try:
        from ._recall_detection import detect_query_type

        return detect_query_type(query) in ("multi-hop", "temporal")
    except Exception:
        return False


def has_capitalised_token_detector(query: str | None, results: list[dict] | None) -> bool:
    if not query:
        return False
    return bool(re.search(r"\b[A-Z][a-zA-Z]{2,}\b", query))


__all__ = [
    "FeatureGate",
    "FieldSpec",
    "has_capitalised_token_detector",
    "multi_hop_detector",
    "multi_hop_or_temporal_detector",
]
