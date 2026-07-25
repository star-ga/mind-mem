"""Provenance-rich blocks (roadmap Group E) — optional actor/session/tool metadata.

Five schema-ADDITIVE, fully optional provenance fields that record *who*
wrote a block, *in what role*, *from which session*, *via which tool*,
and *why*:

    ============  ==============  =========================================
    caller param  block field     meaning
    ============  ==============  =========================================
    actor_id      ``ActorId``     stable identifier of the writing agent
    actor_role    ``ActorRole``   role the actor acted under (e.g. planner)
    session_id    ``SessionId``   conversation / run the write came from
    tool_id       ``ToolId``      tool or pipeline that produced the write
    purpose       ``Purpose``     free-text intent for the write
    ============  ==============  =========================================

Backward compatible by construction: every field is optional, blocks
without them parse / render / recall exactly as before, and nothing in
the pipeline requires their presence. The canonical PascalCase field
names follow the existing block-field convention (``EventId``,
``ContentHash``, ``DiaID``).

Values are single-line by contract — :func:`sanitize_provenance_value`
flattens CR/LF so a crafted value can never start a new ``[ID]`` block
header or a ``Key:`` governance line inside the Markdown corpus (same
threat model as ``apply_engine._sanitize_reason_for_markdown``).
"""

from __future__ import annotations

from typing import Any, Optional

# caller-facing snake_case parameter name -> canonical block field name.
# Insertion order is the canonical emission order.
PROVENANCE_FIELDS: dict[str, str] = {
    "actor_id": "ActorId",
    "actor_role": "ActorRole",
    "session_id": "SessionId",
    "tool_id": "ToolId",
    "purpose": "Purpose",
}

# Canonical block field names, in emission order.
PROVENANCE_FIELD_NAMES: tuple[str, ...] = tuple(PROVENANCE_FIELDS.values())

# Hard cap per value — provenance is metadata, not content. Mirrors the
# tag/rationale bounding in ``propose_update`` (issue #512 / T-003).
MAX_PROVENANCE_VALUE_LEN = 256


def sanitize_provenance_value(value: str) -> str:
    """Return *value* as a single line, stripped and length-capped.

    CR/LF are flattened to spaces so the value can never terminate a
    Markdown block early or inject a new ``[ID]`` header / ``Key:``
    line when rendered into the corpus.
    """
    flat = value.replace("\r", " ").replace("\n", " ").strip()
    return flat[:MAX_PROVENANCE_VALUE_LEN]


def _clean(name: str, value: Optional[str]) -> Optional[str]:
    """Validate + sanitize one provenance value; None for absent/blank."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"provenance field {name!r} must be a str, got {type(value).__name__}")
    cleaned = sanitize_provenance_value(value)
    return cleaned or None


def attach_provenance(
    block: dict[str, Any],
    *,
    actor_id: Optional[str] = None,
    actor_role: Optional[str] = None,
    session_id: Optional[str] = None,
    tool_id: Optional[str] = None,
    purpose: Optional[str] = None,
) -> dict[str, Any]:
    """Return a NEW block dict with the given provenance fields attached.

    The input *block* is never mutated (immutability convention). Fields
    passed as ``None`` or blank strings are omitted; existing provenance
    fields on the block are overwritten only when a replacement value is
    supplied.

    Raises:
        TypeError: a provenance value is not a ``str``.
    """
    values = {
        "actor_id": actor_id,
        "actor_role": actor_role,
        "session_id": session_id,
        "tool_id": tool_id,
        "purpose": purpose,
    }
    out = dict(block)
    for param, field in PROVENANCE_FIELDS.items():
        cleaned = _clean(param, values[param])
        if cleaned is not None:
            out[field] = cleaned
    return out


def extract_provenance(block: dict[str, Any]) -> dict[str, str]:
    """Return the provenance present on *block* as a snake_case dict.

    Only fields that are present and non-blank are included; a block
    with no provenance yields ``{}``. Non-string stored values are
    coerced via ``str`` so a hand-edited corpus can't crash recall.
    """
    out: dict[str, str] = {}
    for param, field in PROVENANCE_FIELDS.items():
        raw = block.get(field)
        if raw is None:
            continue
        value = sanitize_provenance_value(str(raw))
        if value:
            out[param] = value
    return out
