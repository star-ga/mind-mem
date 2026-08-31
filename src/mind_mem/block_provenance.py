"""Provenance-rich blocks — optional actor/session/tool/source metadata.

Schema-ADDITIVE, fully optional provenance fields. Five (roadmap Group E)
record *who* wrote a block, *in what role*, *from which session*, *via
which tool*, and *why*; a sixth (roadmap T-001) records *what class of
source the content came from*:

    ==============  =================  =====================================
    caller param    block field        meaning
    ==============  =================  =====================================
    actor_id        ``ActorId``        stable id of the writing agent
    actor_role      ``ActorRole``      role the actor acted under (planner)
    session_id      ``SessionId``      conversation / run the write came from
    tool_id         ``ToolId``         tool or pipeline that produced it
    purpose         ``Purpose``        free-text intent for the write
    content_source  ``ContentSource``  class of source the CONTENT came from
    ==============  =================  =====================================

``ContentSource`` (roadmap T-001) is a second, orthogonal axis. The five
Group E fields answer *who wrote the block*; ``ContentSource`` answers
*where the text itself came from* — ``agent`` (the model composed it),
``user`` (a human typed it), or ``external`` (it was pulled in from a
system outside the governed store). The two can disagree, and the
disagreement is the point: an agent faithfully recording a scraped web
page is ``actor_role: planner`` **and** ``content_source: external``.

Unlike the free-text Group E fields it is **vocabulary-bound**, and it is
read as a security control, so three rules govern it:

* **No default.** An omitted tag stays absent. Defaulting to ``agent``
  would silently mint a trusted-looking claim for every legacy and
  lazy caller; absent-and-explicit beats silently-assumed, and absence
  is already the corpus's neutral, never-promoting state.
* **Loud rejection.** A value outside the vocabulary raises on every
  write path (:func:`normalize_content_source`) rather than being
  coerced to the nearest legal token. Coercion in a trust field turns a
  typo into a trust decision.
* **Fail-closed read.** :func:`read_content_source` never raises: a
  hand-edited corpus value outside the vocabulary reads back as
  ``None`` (unknown, therefore not trusted), never as a trusted class.

The tag **demotes only** — see :mod:`mind_mem.provenance_class`.

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

from collections.abc import Mapping
from typing import Any, Optional

# caller-facing snake_case parameter name -> canonical block field name.
# Insertion order is the canonical emission order.
PROVENANCE_FIELDS: dict[str, str] = {
    "actor_id": "ActorId",
    "actor_role": "ActorRole",
    "session_id": "SessionId",
    "tool_id": "ToolId",
    "purpose": "Purpose",
    "content_source": "ContentSource",
}

# Canonical block field names, in emission order.
PROVENANCE_FIELD_NAMES: tuple[str, ...] = tuple(PROVENANCE_FIELDS.values())

# --- content-provenance axis (roadmap T-001) -------------------------------

#: The model composed this text itself.
CONTENT_SOURCE_AGENT = "agent"
#: A human typed this text.
CONTENT_SOURCE_USER = "user"
#: Pulled in from a system outside the governed store (import, scrape, feed).
CONTENT_SOURCE_EXTERNAL = "external"

#: Caller-facing parameter name and canonical block field for the axis.
CONTENT_SOURCE_PARAM = "content_source"
CONTENT_SOURCE_FIELD = PROVENANCE_FIELDS[CONTENT_SOURCE_PARAM]

#: The closed vocabulary. Deliberately NOT named ``Source``: that field is
#: already taken by the importer's origin token (``Source: imported:slack``,
#: see ``importers.engine``) which ``guardrails`` and ``provenance_class``
#: match by prefix. Overloading it would make one field mean two things and
#: silently change how those two modules read every imported block.
CONTENT_SOURCES: tuple[str, ...] = (
    CONTENT_SOURCE_AGENT,
    CONTENT_SOURCE_USER,
    CONTENT_SOURCE_EXTERNAL,
)

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


def normalize_content_source(value: Any) -> Optional[str]:
    """Return the canonical ``ContentSource`` token, or ``None`` if absent.

    The strict, **write-path** validator: a value outside
    :data:`CONTENT_SOURCES` raises instead of being coerced. There is no
    fuzzy matching, so a typo can never resolve to the nearest legal token
    and quietly pick a trust class.

    Case and surrounding whitespace are folded (``"  External "`` ->
    ``"external"``): that normalises the *same* value to its canonical
    spelling, which is not the same thing as substituting a different one.

    Absent (``None``) and blank both return ``None`` — the field is then
    simply not written. There is deliberately **no default**: silently
    stamping ``agent`` on every untagged write would manufacture a
    trusted-looking claim nobody made.

    Raises:
        TypeError: *value* is neither ``None`` nor a ``str``.
        ValueError: *value* is a non-blank string outside the vocabulary.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"provenance field {CONTENT_SOURCE_PARAM!r} must be a str, got {type(value).__name__}")
    token = sanitize_provenance_value(value).lower()
    if not token:
        return None
    if token not in CONTENT_SOURCES:
        raise ValueError(f"provenance field {CONTENT_SOURCE_PARAM!r} must be one of {', '.join(CONTENT_SOURCES)}; got {value!r}")
    return token


def read_content_source(block: Mapping[str, Any]) -> Optional[str]:
    """The declared content source of *block*, or ``None`` when unknown.

    The lenient, **read-path** counterpart of
    :func:`normalize_content_source`, and the single reader every consumer
    should use. It never raises: the corpus is hand-editable Markdown, so
    recall must survive whatever is sitting in the file.

    Fail-closed by construction — an out-of-vocabulary value (``operator``,
    ``trusted-internal``, a list, an integer) reads back as ``None``, i.e.
    *unknown*, never as a recognised class. An attacker who can write the
    field can therefore only ever reach "unknown" or a real token; there is
    no spelling that yields a class the vocabulary does not contain.
    """
    raw = block.get(CONTENT_SOURCE_FIELD)
    if raw is None:
        return None
    try:
        return normalize_content_source(str(raw))
    except ValueError:
        return None


def clean_provenance_value(param: str, value: Any) -> Optional[str]:
    """Validate + sanitize one provenance value; ``None`` for absent/blank.

    The single choke point every write path goes through, so the
    vocabulary-bound fields cannot be validated in one writer and skipped
    in the next. Free-text fields are sanitized; ``content_source`` is
    routed to :func:`normalize_content_source` and rejected loudly.

    Raises:
        TypeError: the value is not a ``str``.
        ValueError: a vocabulary-bound value is outside its vocabulary.
    """
    if param == CONTENT_SOURCE_PARAM:
        return normalize_content_source(value)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"provenance field {param!r} must be a str, got {type(value).__name__}")
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
    content_source: Optional[str] = None,
) -> dict[str, Any]:
    """Return a NEW block dict with the given provenance fields attached.

    The input *block* is never mutated (immutability convention). Fields
    passed as ``None`` or blank strings are omitted; existing provenance
    fields on the block are overwritten only when a replacement value is
    supplied. An omitted *content_source* stays omitted — no default.

    Raises:
        TypeError: a provenance value is not a ``str``.
        ValueError: *content_source* is outside :data:`CONTENT_SOURCES`.
            Raised before anything is written, so a refused tag never
            leaves a half-tagged block behind.
    """
    values = {
        "actor_id": actor_id,
        "actor_role": actor_role,
        "session_id": session_id,
        "tool_id": tool_id,
        "purpose": purpose,
        CONTENT_SOURCE_PARAM: content_source,
    }
    cleaned_values = {param: clean_provenance_value(param, values[param]) for param in PROVENANCE_FIELDS}
    out = dict(block)
    for param, field in PROVENANCE_FIELDS.items():
        cleaned = cleaned_values[param]
        if cleaned is not None:
            out[field] = cleaned
    return out


def extract_provenance(block: dict[str, Any]) -> dict[str, str]:
    """Return the provenance present on *block* as a snake_case dict.

    Only fields that are present and non-blank are included; a block
    with no provenance yields ``{}``. Non-string stored values are
    coerced via ``str`` so a hand-edited corpus can't crash recall, and a
    ``content_source`` outside the vocabulary is dropped rather than
    surfaced (see :func:`read_content_source` — unknown, not trusted).
    """
    out: dict[str, str] = {}
    for param, field in PROVENANCE_FIELDS.items():
        if param == CONTENT_SOURCE_PARAM:
            token = read_content_source(block)
            if token:
                out[param] = token
            continue
        raw = block.get(field)
        if raw is None:
            continue
        value = sanitize_provenance_value(str(raw))
        if value:
            out[param] = value
    return out
