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

Migrated (each declares one module-level gate, and its public pair is
now a delegation rather than a second implementation):

* ``graph_recall.GRAPH_EXPAND_GATE`` — ``retrieval.multi_hop``
* ``entity_prefetch.ENTITY_PREFETCH_GATE`` — ``retrieval.entity_prefetch``
* ``session_boost.SESSION_BOOST_GATE`` — ``retrieval.session_boost``
* ``kg_fusion.KG_FUSION_GATE`` — ``retrieval.kg_fusion``
* ``truth_score.TRUTH_SCORE_GATE`` — ``retrieval.truth_score``

Not migrated: ``trust_scores.resolve_trust_config`` — its contract is
``Mapping`` rather than ``dict`` and three of its knobs are read as
``bool(section.get(k, default))``, which disagrees with
:meth:`FieldSpec.resolve` on an explicit ``null``. The reasons, and the
upgrade path, are recorded at the function itself.

The two shapes these five had drifted into are both preserved, not
harmonised — see ``implicit_section``. Deduplicating a gate must not
change any answer it gives; ``tests/test_feature_gate.py`` differences
every migrated function against its pre-migration body over a config
matrix to keep that honest.
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


def strict_int(value: Any) -> int:
    """Coerce to ``int`` only when the raw value already *is* one.

    Every hand-rolled resolver this module replaces guarded its integer
    knobs with ``isinstance(raw, int)`` and only then called ``int(raw)``
    — the string ``"5"`` was rejected, not parsed. A plain ``coerce=int``
    would parse it, so the shared coercer keeps the stricter contract.
    ``bool`` passes, exactly as ``isinstance(True, int)`` reports it; that
    quirk is inherited deliberately, not overlooked — fixing it here would
    change a migrated feature's answer for ``max_hops: true``.
    """
    if not isinstance(value, int):
        raise TypeError(f"expected int, got {type(value).__name__}")
    return int(value)


def strict_number(value: Any) -> float:
    """Coerce to ``float`` only when the raw value is already int|float.

    The ``isinstance(raw, (int, float))`` half of the same guard; a
    numeric-looking string is rejected rather than parsed.
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"expected int|float, got {type(value).__name__}")
    return float(value)


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

    ``implicit_section`` selects which of the two shapes the migrated
    features actually shipped. A gate whose ancestor read
    ``retrieval.get(<name>, {})`` treated an *absent* section as an empty
    one, so ``auto_enable`` still ran; one that returned early on a
    missing key did not. Both are preserved verbatim — the flag is how a
    migration stays behaviour-identical instead of quietly picking a
    winner.
    """

    name: str
    fields: dict[str, FieldSpec] = field(default_factory=dict)
    auto_detector: _AutoDetector | None = None
    auto_enable_default: bool = True
    implicit_section: bool = False

    def _section(self, config: dict[str, Any] | None) -> dict[str, Any] | None:
        """Return the ``retrieval.<name>`` dict or None when absent.

        An empty dict (``{}``) is a valid section — it means "use
        defaults" — and is distinct from a missing section, unless
        ``implicit_section`` collapses the two.

        A section that is present but of the wrong type is a hard OFF
        either way: a malformed section is never read as "defaults".
        """
        if not config or not isinstance(config, dict):
            return None
        retrieval = config.get("retrieval", {})
        if not isinstance(retrieval, dict):
            return None
        if self.name not in retrieval:
            return {} if self.implicit_section else None
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


def always_detector(query: str | None, results: list[dict] | None) -> bool:
    """Auto-enable with no further condition — ``auto_enable`` alone decides.

    The degenerate detector, for a feature whose ancestor ended its gate
    on ``return bool(section.get("auto_enable", True))``. Spelled out as
    a detector rather than as "no detector", because ``auto_detector is
    None`` already means the opposite (auto-enable can never fire).
    """
    return True


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
    "always_detector",
    "has_capitalised_token_detector",
    "multi_hop_detector",
    "multi_hop_or_temporal_detector",
    "strict_int",
    "strict_number",
]
