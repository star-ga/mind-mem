"""Persona-aware recall projection (v3.9.0 candidate).

Reshapes recall result lists for different consumers without touching
the underlying index. Three named personas:

* ``brief``     — id + 1-line subject + score; for routing layers,
                  Slack snippets, status panels.
* ``detailed``  — full block (current ``recall`` default); for agents
                  that need the actual content.
* ``technical`` — full block + axis scores + governance state +
                  provenance hash chain; for audit / governance checks.

Implemented as a pure function over the dicts that ``recall`` and
``hybrid_search`` already return — zero index cost, no schema change.
Inspired by the Understand-Anything (MIT) detail-level UX; concept
only, no shared code.
"""

from __future__ import annotations

import logging
from typing import Any, Final, Literal

__all__ = [
    "PERSONAS",
    "Persona",
    "PersonaError",
    "apply_persona",
    "project_block",
]

_log = logging.getLogger("mind_mem.personas")

# Tightly enumerated — adding a persona is a deliberate decision, not
# a free-for-all. Caller-supplied unknown personas raise PersonaError.
Persona = Literal["brief", "detailed", "technical"]

PERSONAS: Final[tuple[str, ...]] = ("brief", "detailed", "technical")
DEFAULT_PERSONA: Final[Persona] = "detailed"


class PersonaError(ValueError):
    """Raised when an unknown persona is requested."""


# ---------------------------------------------------------------------------
# Per-block projection
# ---------------------------------------------------------------------------


def _one_line_subject(block: dict[str, Any], cap: int = 120) -> str:
    """Return a single short line summarising a block. Used by ``brief``."""
    candidates = (
        block.get("Subject"),
        block.get("subject"),
        block.get("Statement"),
        block.get("statement"),
        block.get("content"),
    )
    for c in candidates:
        if isinstance(c, str) and c.strip():
            line = c.split("\n", 1)[0].strip()
            if len(line) > cap:
                line = line[: cap - 1] + "…"
            return line
    return "(no subject)"


def _block_id(block: dict[str, Any]) -> str | None:
    bid = block.get("_id") or block.get("id") or block.get("block_id")
    return str(bid) if bid else None


def project_block(block: dict[str, Any], persona: Persona) -> dict[str, Any]:
    """Project a single block into the *persona*-shaped dict.

    Raises:
        PersonaError: *persona* is not one of :data:`PERSONAS`.
    """
    if persona not in PERSONAS:
        raise PersonaError(f"unknown persona {persona!r}; must be one of {PERSONAS}")

    if persona == "brief":
        return {
            "id": _block_id(block),
            "score": block.get("score") or block.get("_score"),
            "subject": _one_line_subject(block),
        }

    if persona == "detailed":
        # Identity projection — the existing recall format already is
        # the "detailed" view. Strip nothing, add nothing.
        return dict(block)

    # persona == "technical"
    out = dict(block)
    # Promote rarely-surfaced governance / provenance fields to the
    # top-level so audit consumers don't have to fish them out of
    # nested keys.
    for surface_key, source_keys in (
        ("axis_scores", ("axis_scores", "AxisScores", "_axis_scores")),
        ("governance_state", ("governance_state", "GovernanceState", "Status")),
        ("provenance_hash", ("provenance_hash", "ProvenanceHash", "audit_hash", "AuditHash")),
        ("source_span", ("source_span", "SourceSpan")),
        ("transform_hash", ("transform_hash", "TransformHash")),
    ):
        if surface_key in out:
            continue
        for src in source_keys:
            if src in block and block[src] is not None:
                out[surface_key] = block[src]
                break
    return out


# ---------------------------------------------------------------------------
# List-level projection
# ---------------------------------------------------------------------------


def apply_persona(blocks: list[dict[str, Any]], persona: str | None) -> list[dict[str, Any]]:
    """Apply *persona* projection to every block in *blocks*.

    Args:
        blocks: Result list from ``recall`` / ``hybrid_search``.
        persona: One of :data:`PERSONAS`, or ``None`` for the default
                 (``detailed`` — identity).

    Returns:
        New list of projected dicts. The input list is not mutated.

    Raises:
        PersonaError: *persona* is not recognised.
    """
    if persona is None or persona == "":
        return [project_block(b, DEFAULT_PERSONA) for b in blocks]
    if persona not in PERSONAS:
        raise PersonaError(f"unknown persona {persona!r}; must be one of {PERSONAS}")
    # mypy: persona has been narrowed to one of PERSONAS literal values.
    persona_typed: Persona = persona  # type: ignore[assignment]
    return [project_block(b, persona_typed) for b in blocks]
