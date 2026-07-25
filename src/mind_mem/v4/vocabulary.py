"""v4 vocabulary-bound fields — per-workspace controlled vocabularies.

ROADMAP Group E surface (``v4.vocabulary``): a workspace can declare the
allowed values for a named field (e.g. ``block_kind``, ``category``) and
ingest/validation then rejects — or merely flags — out-of-vocabulary
values. Backward compatible by construction: no vocabulary declared for
a field means no restriction on that field, and with the feature flag
OFF nothing in the v3.x paths changes.

Declarations are config-driven. Two sources, merged per-field with the
workspace file winning:

    1. ``<workspace>/mind-mem.json`` under a top-level ``vocabularies``
       key.
    2. ``<workspace>/vocabularies.json`` — a standalone workspace file
       holding just the declarations mapping (same shape as the
       ``vocabularies`` key). Overrides mind-mem.json field-by-field.

Declaration shapes (both accepted)::

    {
        "vocabularies": {
            "block_kind": ["decision", "fact", "reference"],
            "category": {
                "values": ["project", "user", "ops"],
                "mode": "flag",
                "case_sensitive": false
            }
        }
    }

The list shorthand means ``mode="reject"``, ``case_sensitive=true``.
``mode`` is one of ``"reject"`` (violation blocks the write) or
``"flag"`` (violation is reported but the write proceeds).

API:
    load_vocabularies(workspace, strict=False) -> dict[str, FieldVocabulary]
    check_fields(fields, vocabularies) -> list[VocabularyViolation]
    validate_workspace_fields(workspace, fields) -> list[VocabularyViolation]
    rejections(violations) / flagged(violations)  — mode split helpers

Wiring: :func:`mind_mem.v4.block_metadata.validate_block` runs the
vocabulary check when given a ``workspace`` and the flag is on, and
:func:`mind_mem.v4.block_metadata.set_block_metadata` gates tag writes
the same way (reject-mode violations raise
:class:`OutOfVocabularyError`; flag-mode violations log a warning).

Feature-flag gated under ``v4.vocabulary``.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..observability import get_logger
from .feature_flags import require_enabled

__all__ = [
    "FLAG",
    "MODES",
    "WORKSPACE_FILE",
    "FieldVocabulary",
    "OutOfVocabularyError",
    "VocabularyConfigError",
    "VocabularyViolation",
    "check_fields",
    "flagged",
    "load_vocabularies",
    "rejections",
    "validate_workspace_fields",
]

_log = get_logger("v4.vocabulary")

FLAG: str = "vocabulary"

#: Standalone workspace declarations file (optional, overrides
#: ``mind-mem.json`` field-by-field).
WORKSPACE_FILE: str = "vocabularies.json"

#: Valid enforcement modes.
MODES: tuple[str, ...] = ("reject", "flag")


class VocabularyConfigError(ValueError):
    """Raised (in strict mode) when a vocabulary declaration is malformed."""


class OutOfVocabularyError(ValueError):
    """Raised when a write carries a reject-mode out-of-vocabulary value."""

    def __init__(self, violations: list[VocabularyViolation]):
        self.violations = violations
        super().__init__("; ".join(v.message() for v in violations))


@dataclass(frozen=True)
class FieldVocabulary:
    """Controlled vocabulary for one named field.

    ``values`` preserves declaration order (shown in error messages);
    matching is set-based. With ``case_sensitive=False`` comparison is
    casefolded on both sides.
    """

    field: str
    values: tuple[str, ...]
    mode: str = "reject"
    case_sensitive: bool = True

    def __post_init__(self) -> None:
        if not self.field:
            raise VocabularyConfigError("FieldVocabulary.field must be a non-empty string")
        if not self.values:
            raise VocabularyConfigError(f"vocabulary for {self.field!r} declares no values")
        if self.mode not in MODES:
            raise VocabularyConfigError(f"vocabulary for {self.field!r} has invalid mode {self.mode!r} (expected one of {MODES})")

    def allows(self, value: str) -> bool:
        """True when ``value`` is inside the vocabulary."""
        if self.case_sensitive:
            return value in self.values
        folded = value.casefold()
        return any(folded == v.casefold() for v in self.values)


@dataclass(frozen=True)
class VocabularyViolation:
    """One out-of-vocabulary value found by :func:`check_fields`."""

    field: str
    value: str
    mode: str
    allowed: tuple[str, ...]

    def message(self) -> str:
        return f"field {self.field!r} value {self.value!r} not in vocabulary {list(self.allowed)!r} (mode={self.mode})"


# ---------------------------------------------------------------------------
# Declaration loading
# ---------------------------------------------------------------------------


def _parse_one(field: str, raw: Any) -> FieldVocabulary:
    """Parse a single declaration (list shorthand or dict form).

    Raises :class:`VocabularyConfigError` on any malformed shape.
    """
    if not isinstance(field, str) or not field:
        raise VocabularyConfigError(f"vocabulary field name must be a non-empty string, got {field!r}")
    if isinstance(raw, (list, tuple)):
        raw = {"values": list(raw)}
    if not isinstance(raw, Mapping):
        raise VocabularyConfigError(f"vocabulary for {field!r} must be a list of values or a mapping, got {type(raw).__name__}")
    values_raw = raw.get("values")
    if not isinstance(values_raw, (list, tuple)) or not values_raw:
        raise VocabularyConfigError(f"vocabulary for {field!r} needs a non-empty 'values' list")
    values: list[str] = []
    for v in values_raw:
        if not isinstance(v, str) or not v:
            raise VocabularyConfigError(f"vocabulary for {field!r} has a non-string value: {v!r}")
        if v not in values:  # dedupe, preserve order
            values.append(v)
    mode = raw.get("mode", "reject")
    case_sensitive = raw.get("case_sensitive", True)
    if not isinstance(case_sensitive, bool):
        raise VocabularyConfigError(f"vocabulary for {field!r} 'case_sensitive' must be a bool, got {case_sensitive!r}")
    if not isinstance(mode, str):
        raise VocabularyConfigError(f"vocabulary for {field!r} 'mode' must be a string, got {mode!r}")
    return FieldVocabulary(field=field, values=tuple(values), mode=mode, case_sensitive=case_sensitive)


def _parse_declarations(
    raw: Any,
    *,
    source: str,
    strict: bool,
) -> dict[str, FieldVocabulary]:
    """Parse a declarations mapping. Malformed entries are skipped with a
    warning (``strict=False``) or raise (``strict=True``)."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        if strict:
            raise VocabularyConfigError(f"{source}: vocabularies must be a mapping, got {type(raw).__name__}")
        _log.warning("vocabulary_config_invalid", source=source, reason="not a mapping")
        return {}
    out: dict[str, FieldVocabulary] = {}
    for field, decl in raw.items():
        try:
            out[str(field)] = _parse_one(str(field), decl)
        except VocabularyConfigError as exc:
            if strict:
                raise
            _log.warning("vocabulary_declaration_skipped", source=source, field=str(field), reason=str(exc))
    return out


def _read_json(path: Path, *, strict: bool) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        if strict:
            raise VocabularyConfigError(f"could not read {path.name}: {exc}") from exc
        _log.warning("vocabulary_config_unreadable", path=str(path), reason=str(exc))
        return None


def load_vocabularies(
    workspace: str | Path,
    *,
    strict: bool = False,
) -> dict[str, FieldVocabulary]:
    """Load per-workspace vocabulary declarations.

    Merges ``mind-mem.json``'s top-level ``vocabularies`` key with the
    standalone ``vocabularies.json`` workspace file; the workspace file
    wins field-by-field. Missing files or absent keys mean an empty
    result — no vocabulary, no restriction.

    With ``strict=False`` (default) malformed declarations are skipped
    with a logged warning so a config typo can't take ingest down; with
    ``strict=True`` they raise :class:`VocabularyConfigError`.
    """
    require_enabled(FLAG)
    ws = Path(workspace)
    merged: dict[str, FieldVocabulary] = {}

    config = _read_json(ws / "mind-mem.json", strict=strict)
    if isinstance(config, Mapping):
        merged.update(_parse_declarations(config.get("vocabularies"), source="mind-mem.json", strict=strict))

    ws_file = _read_json(ws / WORKSPACE_FILE, strict=strict)
    if ws_file is not None:
        merged.update(_parse_declarations(ws_file, source=WORKSPACE_FILE, strict=strict))

    return merged


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _iter_values(value: Any) -> Iterable[str]:
    """Yield the comparable string form(s) of a field value.

    ``None`` yields nothing (absence is never a violation — required-ness
    is the ontology/schema-validator layer's job, not the vocabulary's).
    List-like values are checked element-wise so tag-list fields work.
    Non-string scalars are coerced with ``str()``.
    """
    if value is None:
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            if item is not None:
                yield item if isinstance(item, str) else str(item)
        return
    yield value if isinstance(value, str) else str(value)


def check_fields(
    fields: Mapping[str, Any],
    vocabularies: Mapping[str, FieldVocabulary],
) -> list[VocabularyViolation]:
    """Check ``fields`` against ``vocabularies``.

    Fields without a declared vocabulary pass unconditionally (backward
    compatible). Returns one :class:`VocabularyViolation` per offending
    value; an empty list means everything is in-vocabulary.
    """
    require_enabled(FLAG)
    violations: list[VocabularyViolation] = []
    for field, value in fields.items():
        vocab = vocabularies.get(field)
        if vocab is None:
            continue
        for candidate in _iter_values(value):
            if not vocab.allows(candidate):
                violations.append(
                    VocabularyViolation(
                        field=field,
                        value=candidate,
                        mode=vocab.mode,
                        allowed=vocab.values,
                    )
                )
    return violations


def validate_workspace_fields(
    workspace: str | Path,
    fields: Mapping[str, Any],
) -> list[VocabularyViolation]:
    """Convenience: :func:`load_vocabularies` + :func:`check_fields`."""
    require_enabled(FLAG)
    vocabularies = load_vocabularies(workspace)
    if not vocabularies:
        return []
    return check_fields(fields, vocabularies)


def rejections(violations: Iterable[VocabularyViolation]) -> list[VocabularyViolation]:
    """The subset of violations that must block the write."""
    return [v for v in violations if v.mode == "reject"]


def flagged(violations: Iterable[VocabularyViolation]) -> list[VocabularyViolation]:
    """The subset of violations that only warn."""
    return [v for v in violations if v.mode == "flag"]
