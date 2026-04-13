# Copyright 2026 STARGA, Inc.
"""Ontology & schema validation (v2.5.0).

A lightweight OWL-lite layer on top of the block store. Entity types
declare required and optional properties, plus a type hierarchy for
subclass matching. Blocks are validated against the active ontology
version on write; a validation error becomes a governance Evidence
Object rather than a silent drop.

Schema evolution is deliberately simple: each ontology is tagged with
a semver string, and blocks record the version they were validated
against. Re-validating an older block under a newer ontology is an
explicit caller decision, not an automatic background process.

Pure-Python stdlib only.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityType:
    """An OWL-lite entity type declaration.

    Attributes
    ----------
    name:
        Canonical type identifier (e.g. ``PERSON``).
    required:
        Property names that MUST be present on any instance.
    optional:
        Property names that MAY be present. Properties outside both
        sets are rejected during validation so schemas stay tight.
    parent:
        Optional parent type. A block satisfying the parent's
        properties also satisfies the child if the child adds none
        of its own.
    property_types:
        Mapping of property name → a str label for the expected Python
        type (``"str"``, ``"int"``, ``"float"``, ``"bool"``, ``"list"``,
        ``"dict"``). Enforced on write.
    """

    name: str
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    parent: Optional[str] = None
    property_types: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not re.fullmatch(r"[A-Z][A-Z0-9_]*", self.name):
            raise ValueError(
                f"EntityType.name must be UPPER_SNAKE_CASE, got {self.name!r}"
            )
        conflict = set(self.required) & set(self.optional)
        if conflict:
            raise ValueError(
                f"property cannot be both required and optional: {sorted(conflict)!r}"
            )

    def all_properties(self) -> set[str]:
        return set(self.required) | set(self.optional)


# Python-type labels the validator understands.
_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "str": (str,),
    "int": (int,),
    "float": (float, int),
    "bool": (bool,),
    "list": (list, tuple),
    "dict": (dict,),
}


class ValidationError(Exception):
    """Raised when a block fails ontology validation."""


# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ontology:
    """A versioned collection of :class:`EntityType` declarations."""

    version: str
    types: dict[str, EntityType]

    def __post_init__(self) -> None:
        if not self.version or not isinstance(self.version, str):
            raise ValueError("Ontology.version must be a non-empty string")
        for name, et in self.types.items():
            if et.name != name:
                raise ValueError(
                    f"Ontology key {name!r} does not match EntityType.name={et.name!r}"
                )
            if et.parent and et.parent not in self.types:
                raise ValueError(
                    f"EntityType {name!r} references unknown parent {et.parent!r}"
                )

    def type_names(self) -> list[str]:
        return sorted(self.types.keys())

    def has(self, type_name: str) -> bool:
        return type_name in self.types

    def effective_required(self, type_name: str) -> set[str]:
        """Required properties including inherited ones."""
        if type_name not in self.types:
            raise KeyError(type_name)
        et = self.types[type_name]
        required = set(et.required)
        if et.parent:
            required |= self.effective_required(et.parent)
        return required

    def effective_allowed(self, type_name: str) -> set[str]:
        """Required + optional, merged across the parent chain."""
        if type_name not in self.types:
            raise KeyError(type_name)
        et = self.types[type_name]
        allowed = et.all_properties()
        if et.parent:
            allowed |= self.effective_allowed(et.parent)
        return allowed

    def effective_property_types(self, type_name: str) -> dict[str, str]:
        """Merge ``property_types`` across the parent chain."""
        if type_name not in self.types:
            raise KeyError(type_name)
        et = self.types[type_name]
        if et.parent:
            merged = self.effective_property_types(et.parent)
        else:
            merged = {}
        merged.update(et.property_types)
        return merged

    def validate(
        self,
        type_name: str,
        block: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> list[str]:
        """Return the list of validation errors for *block* against *type_name*.

        Empty list means the block is valid. When ``strict=True`` (the
        default) extra properties outside the effective allowed set are
        an error; ``strict=False`` permits additional properties but
        still catches missing required and type mismatches.
        """
        if type_name not in self.types:
            return [f"unknown type: {type_name!r}"]
        errors: list[str] = []
        required = self.effective_required(type_name)
        allowed = self.effective_allowed(type_name)
        prop_types = self.effective_property_types(type_name)

        for prop in required:
            if prop not in block or block.get(prop) in (None, ""):
                errors.append(f"missing required property: {prop!r}")

        if strict:
            for prop in block.keys():
                if prop.startswith("_"):
                    continue  # framework-private fields (_id, _score, …)
                if prop not in allowed:
                    errors.append(f"unexpected property: {prop!r}")

        for prop, label in prop_types.items():
            if prop not in block:
                continue
            expected = _TYPE_MAP.get(label)
            if expected is None:
                errors.append(f"unknown type label for {prop!r}: {label!r}")
                continue
            value = block[prop]
            if not isinstance(value, expected):
                errors.append(
                    f"type mismatch for {prop!r}: expected {label}, "
                    f"got {type(value).__name__}"
                )
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "types": {
                name: {
                    "name": et.name,
                    "required": list(et.required),
                    "optional": list(et.optional),
                    "parent": et.parent,
                    "property_types": dict(et.property_types),
                }
                for name, et in self.types.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Ontology":
        types: dict[str, EntityType] = {}
        for name, raw in data.get("types", {}).items():
            types[name] = EntityType(
                name=str(raw["name"]),
                required=tuple(raw.get("required", ())),
                optional=tuple(raw.get("optional", ())),
                parent=raw.get("parent"),
                property_types=dict(raw.get("property_types", {})),
            )
        return cls(version=str(data["version"]), types=types)


# ---------------------------------------------------------------------------
# Registry — load / lookup active ontology per-workspace
# ---------------------------------------------------------------------------


class OntologyRegistry:
    """Process-local registry of ontologies keyed by version."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_version: dict[str, Ontology] = {}
        self._active: Optional[str] = None

    def load(self, ontology: Ontology, *, make_active: bool = False) -> None:
        with self._lock:
            self._by_version[ontology.version] = ontology
            if make_active or self._active is None:
                self._active = ontology.version

    def get(self, version: str) -> Optional[Ontology]:
        with self._lock:
            return self._by_version.get(version)

    def active(self) -> Optional[Ontology]:
        with self._lock:
            if self._active is None:
                return None
            return self._by_version.get(self._active)

    def versions(self) -> list[str]:
        with self._lock:
            return sorted(self._by_version.keys())

    def set_active(self, version: str) -> None:
        with self._lock:
            if version not in self._by_version:
                raise KeyError(f"ontology version {version!r} not loaded")
            self._active = version


# ---------------------------------------------------------------------------
# Domain ontology library — a few common profiles shipped in-box
# ---------------------------------------------------------------------------


def software_engineering_ontology() -> Ontology:
    """A small OWL-lite profile for software-engineering knowledge."""
    return Ontology(
        version="se-1.0",
        types={
            "ENTITY": EntityType(
                name="ENTITY",
                required=("name",),
                optional=("description", "tags"),
                property_types={"name": "str", "description": "str", "tags": "list"},
            ),
            "PERSON": EntityType(
                name="PERSON",
                parent="ENTITY",
                required=("role",),
                optional=("email", "github"),
                property_types={"role": "str", "email": "str", "github": "str"},
            ),
            "PROJECT": EntityType(
                name="PROJECT",
                parent="ENTITY",
                required=("status",),
                optional=("repo", "owner"),
                property_types={"status": "str", "repo": "str", "owner": "str"},
            ),
            "DECISION": EntityType(
                name="DECISION",
                parent="ENTITY",
                required=("statement",),
                optional=("rationale", "confidence", "date"),
                property_types={
                    "statement": "str",
                    "rationale": "str",
                    "confidence": "float",
                    "date": "str",
                },
            ),
            "TASK": EntityType(
                name="TASK",
                parent="ENTITY",
                required=("title",),
                optional=("assignee", "priority", "due"),
                property_types={
                    "title": "str",
                    "assignee": "str",
                    "priority": "str",
                    "due": "str",
                },
            ),
        },
    )


__all__ = [
    "EntityType",
    "Ontology",
    "OntologyRegistry",
    "ValidationError",
    "software_engineering_ontology",
]
