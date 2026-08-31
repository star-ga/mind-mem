# Copyright 2026 STARGA, Inc.
"""GUARDRAIL blocks — deterministic, trigger-fired prohibitions.

Similarity ranking is the wrong retrieval model for a prohibition.  A rule
like *"never run ``git reset --hard`` without checking ``git status``"* must
fire at the moment the agent is about to run ``git reset`` — **not** when a
query happens to land cosine-near it.  Ranked recall gets this backwards:
the rule competes for top-k against whatever else the query resembles, and
loses exactly when the query is about something else (which is most of the
time a destructive action is attempted).

A ``[GR-...]`` block therefore carries **declarative trigger conditions**
instead of relying on lexical/semantic proximity:

    [GR-20260827-001]
    Type: Guardrail
    Statement: Never run `git reset --hard` without checking `git status` first.
    Severity: critical
    TriggerTools: Bash
    TriggerCommands: git reset --hard, git clean -fd
    TriggerPaths: src/**/*.py
    Status: active

Semantics
---------
* **AND across declared dimensions, OR within a dimension.**  The block
  above fires only for a ``Bash`` call whose command contains one of the
  declared command patterns.  A trigger that declares nothing never fires
  (fail-closed) — an always-on "guardrail" is just noise.
* **Deterministic.**  Matching is literal/glob only: no model call, no
  embedding, no clock, no randomness.  Same context + same corpus ⇒ same
  guardrails in the same order, on every machine.
* **Ranker bypass.**  Matching guardrails are surfaced by
  :mod:`mind_mem.guardrail_surface` ahead of ranked hits regardless of
  their similarity score — see that module for the bounded-displacement
  contract.
* **Read-only.**  Nothing here writes to the store.  Guardrail blocks are
  operator-authored in ``guardrails/GUARDRAILS.md`` and only ever read
  back — ``propose_update`` mints ``decision`` / ``task`` blocks into
  ``SIGNALS.md`` and cannot author a ``[GR-...]`` block (see
  ``docs/guardrails.md`` §Authoring).
* **Provenance-restricted.**  A block carrying external-ingest / imported
  provenance is never recognised as a guardrail, whatever its content or
  metadata declares — see :func:`guardrail_provenance_refusal`.  A
  guardrail bypasses the ranker and is surfaced unconditionally, so
  trigger-minting is an injection primitive: untrusted content must not
  reach it.

Every dimension shares one matcher — the glob grammar, normalisation and
per-pattern bounds live in :mod:`mind_mem.guardrail_patterns`.

# deferred: trigger dimensions are the four an agent harness can supply
# without inference (tool / command / intent class / path). A semantic
# "topic" dimension is intentionally NOT implemented - upgrade path: add a
# `topics` field on GuardrailTrigger fed by an INJECTED classifier callable
# (the Compressor pattern), never a hard model dependency, so the default
# path stays model-free and deterministic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .block_parser import parse_file
from .block_provenance import extract_provenance
from .guardrail_patterns import (
    MAX_PATTERN_LEN,
    MAX_PATTERNS_PER_DIMENSION,
    GuardrailSpecError,
    coerce_patterns,
    exact_or_glob,
    path_match,
    substring_or_glob,
)
from .observability import get_logger
from .provenance_class import (
    EXTERNAL_INGEST,
    EXTERNAL_ROLES,
    EXTERNAL_TOKEN_PREFIXES,
    classify_provenance,
)

__all__ = [
    "DEFAULT_GUARDRAIL_FILES",
    "DEFAULT_MAX_SURFACED",
    "GUARDRAIL_ID_PREFIX",
    "MAX_PATTERNS_PER_DIMENSION",
    "MAX_PATTERN_LEN",
    "MAX_SURFACED_HARD_CAP",
    "SEVERITY_RANK",
    "UNTRUSTED_BLOCK_TYPES",
    "UNTRUSTED_PROVENANCE_CLASSES",
    "Guardrail",
    "GuardrailContext",
    "GuardrailPolicy",
    "GuardrailProvenanceError",
    "GuardrailSpecError",
    "GuardrailTrigger",
    "guardrail_provenance_refusal",
    "load_guardrails",
    "match_guardrails",
    "normalise_status",
    "parse_guardrail_block",
]

_log = get_logger("guardrails")

#: Block-ID prefix that marks a block as a guardrail.
GUARDRAIL_ID_PREFIX = "GR-"

#: Workspace-relative files scanned for guardrail blocks by default.
DEFAULT_GUARDRAIL_FILES: tuple[str, ...] = ("guardrails/GUARDRAILS.md",)

#: Default cap on how many guardrails may be force-surfaced per recall.
DEFAULT_MAX_SURFACED = 3

#: Absolute ceiling on ``max_surfaced`` — a misconfigured policy can never
#: flood a result set with constraints.
MAX_SURFACED_HARD_CAP = 10

#: Deterministic severity ordering; unknown severities sort last-but-one.
SEVERITY_RANK: Mapping[str, int] = MappingProxyType({"critical": 0, "high": 1, "medium": 2, "low": 3})

_DEFAULT_SEVERITY = "medium"
_UNKNOWN_SEVERITY_RANK = len(SEVERITY_RANK)

#: Field names carrying each trigger dimension, in match-report order.
_TRIGGER_FIELDS: tuple[tuple[str, str], ...] = (
    ("tool", "TriggerTools"),
    ("command", "TriggerCommands"),
    ("intent", "TriggerIntents"),
    ("path", "TriggerPaths"),
)

#: Statuses that keep a guardrail live.  Anything else (deprecated,
#: archived, superseded) is parsed but never fires.
_LIVE_STATUSES = frozenset({"", "active", "wip"})

#: Separator used to render a multi-valued ``Status`` field back to text.
#: A multi-valued status names no single state, so the joined form is
#: deliberately *not* in :data:`_LIVE_STATUSES` — it fails closed.
_STATUS_JOIN = ", "

#: Provenance classes that may never mint a guardrail.  Only affirmative
#: evidence of *external* origin disqualifies a block: ``unknown`` (a
#: corpus that predates provenance fields) stays eligible, exactly as
#: :mod:`mind_mem.provenance_class` treats absence as neutral everywhere
#: else.  Agent-authored guardrails are still admitted — the threat model
#: here is untrusted *content*, not an authenticated agent.
UNTRUSTED_PROVENANCE_CLASSES: frozenset[str] = frozenset({EXTERNAL_INGEST})

#: Block ``Type`` values stamped by an ingest pipeline.  Mirrors
#: ``importers.engine.IMPORT_BLOCK_TYPE``, kept as a literal so the
#: recall-hot guardrail path never imports the importer package;
#: ``tests/test_guardrail_blocks.py`` asserts the two stay in sync.
UNTRUSTED_BLOCK_TYPES: frozenset[str] = frozenset({"importedmemory"})


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardrailContext:
    """What the agent is about to do.  Supplied by the caller, never inferred."""

    tool: str = ""
    command: str = ""
    intent: str = ""
    paths: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: "Mapping[str, Any] | GuardrailContext | None") -> "GuardrailContext | None":
        """Build a context from a plain mapping (MCP / JSON boundary).

        Returns ``None`` for ``None`` input so callers can pass a missing
        context straight through.  Unknown keys are rejected loudly rather
        than silently ignored — a typo'd ``tools`` key must not read as
        "no tool declared".
        """
        if data is None:
            return None
        if isinstance(data, GuardrailContext):
            return data
        if not isinstance(data, Mapping):
            raise GuardrailSpecError(f"guardrail context must be a mapping, got {type(data).__name__}")
        allowed = {"tool", "command", "intent", "paths"}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise GuardrailSpecError(f"unknown guardrail context key(s): {', '.join(unknown)}")
        raw_paths = data.get("paths") or ()
        if isinstance(raw_paths, str):
            raw_paths = (raw_paths,)
        if not isinstance(raw_paths, (list, tuple)):
            raise GuardrailSpecError(f"guardrail context 'paths' must be a string or list, got {type(raw_paths).__name__}")
        paths = tuple(str(p) for p in raw_paths if str(p).strip())
        return cls(
            tool=str(data.get("tool") or ""),
            command=str(data.get("command") or ""),
            intent=str(data.get("intent") or ""),
            paths=paths,
        )

    def is_empty(self) -> bool:
        return not (self.tool or self.command or self.intent or self.paths)


@dataclass(frozen=True)
class GuardrailTrigger:
    """Declarative fire conditions.  AND across dimensions, OR within one."""

    tools: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()
    intents: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        return not (self.tools or self.commands or self.intents or self.paths)

    def match(self, context: GuardrailContext) -> tuple[str, ...]:
        """Return the dimensions that matched, or ``()`` for no match.

        Fail-closed: an empty trigger never matches.  Every declared
        dimension must match (AND); the return value names them in
        :data:`_TRIGGER_FIELDS` order so reports are deterministic.
        """
        if self.is_empty():
            return ()
        matched: list[str] = []
        if self.tools:
            if not exact_or_glob(self.tools, context.tool):
                return ()
            matched.append("tool")
        if self.commands:
            if not substring_or_glob(self.commands, context.command):
                return ()
            matched.append("command")
        if self.intents:
            if not exact_or_glob(self.intents, context.intent):
                return ()
            matched.append("intent")
        if self.paths:
            if not path_match(self.paths, context.paths):
                return ()
            matched.append("path")
        return tuple(matched)


@dataclass(frozen=True)
class Guardrail:
    """A parsed ``[GR-...]`` block plus its compiled trigger."""

    block_id: str
    statement: str
    severity: str
    trigger: GuardrailTrigger
    source_file: str
    line: int
    status: str
    block: Mapping[str, Any]

    @property
    def severity_rank(self) -> int:
        return SEVERITY_RANK.get(self.severity, _UNKNOWN_SEVERITY_RANK)

    def sort_key(self) -> tuple[int, str]:
        """Total order: severity first, then block ID.  No clocks, no scores."""
        return (self.severity_rank, self.block_id)

    def is_live(self) -> bool:
        return self.status.strip().casefold() in _LIVE_STATUSES


@dataclass(frozen=True)
class GuardrailPolicy:
    """Bounds + kill switch for guardrail surfacing."""

    enabled: bool = True
    max_surfaced: int = DEFAULT_MAX_SURFACED
    sources: tuple[str, ...] = DEFAULT_GUARDRAIL_FILES

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "GuardrailPolicy":
        """Read ``recall.guardrails`` from a workspace config mapping.

        Invalid values fall back to the defaults with a warning rather than
        failing recall — a typo in ``mind-mem.json`` must not take retrieval
        down, but it must not silently disable the constraints either.
        """
        section: Any = {}
        if isinstance(config, Mapping):
            recall_cfg = config.get("recall")
            if isinstance(recall_cfg, Mapping):
                section = recall_cfg.get("guardrails", {})
        if not isinstance(section, Mapping):
            _log.warning("guardrail_config_ignored", reason="recall.guardrails is not an object")
            return cls()

        enabled = section.get("enabled", True)
        max_surfaced = section.get("max_surfaced", DEFAULT_MAX_SURFACED)
        try:
            bounded = max(0, min(int(max_surfaced), MAX_SURFACED_HARD_CAP))
        except (TypeError, ValueError):
            _log.warning("guardrail_config_ignored", reason="max_surfaced is not an integer")
            bounded = DEFAULT_MAX_SURFACED

        raw_sources = section.get("sources", DEFAULT_GUARDRAIL_FILES)
        if isinstance(raw_sources, str):
            raw_sources = (raw_sources,)
        if not isinstance(raw_sources, (list, tuple)):
            _log.warning("guardrail_config_ignored", reason="sources is not a list")
            raw_sources = DEFAULT_GUARDRAIL_FILES
        sources = tuple(str(s).strip() for s in raw_sources if str(s).strip())

        return cls(enabled=bool(enabled), max_surfaced=bounded, sources=sources or DEFAULT_GUARDRAIL_FILES)


class GuardrailProvenanceError(GuardrailSpecError):
    """Raised when a block's provenance disqualifies it from being a guardrail.

    A subclass of :class:`GuardrailSpecError` so every existing caller keeps
    failing closed on it, while :func:`load_guardrails` can log the security
    refusal under its own event.
    """


def normalise_status(raw: Any) -> str:
    """Canonical text of a block ``Status`` field.

    The block parser has a list-field convention: a field written as a
    bare ``Status:`` line followed by ``- active`` renders as the *list*
    ``["active"]``, while ``Status: active`` renders as the string
    ``"active"``.  Both spell one state, so both must read as one state
    here — stringifying the list gives ``"['active']"``, which matches no
    known status and would silently drop a live guardrail over how its
    author spaced a line.

    A bare ``Status:`` with nothing under it parses as ``[]`` and means
    *unstated*, exactly like an absent field — the same reading
    :func:`mind_mem.admissibility.is_admissible_status` gives it.

    A multi-valued status is joined rather than picked from: it names no
    single state this code can read, so it falls through to fail-closed
    (not live) instead of guessing which entry wins.

    Args:
        raw: The block's ``Status`` value as the parser produced it.

    Returns:
        The status as plain text (``""`` when unstated).
    """
    if raw is None:
        return ""
    if isinstance(raw, (list, tuple)):
        parts = [str(item).strip() for item in raw if str(item).strip()]
        return parts[0] if len(parts) == 1 else _STATUS_JOIN.join(parts)
    return str(raw).strip()


def _norm_marker(value: Any) -> str:
    """Lower-cased, stripped ``str`` view of a corpus value (``""`` for None)."""
    if value is None:
        return ""
    return str(value).strip().lower()


def guardrail_provenance_refusal(block: Mapping[str, Any]) -> str:
    """Why *block* may not mint a guardrail — ``""`` when it may.

    A guardrail bypasses the ranker and is surfaced unconditionally, so a
    trigger-bearing block is an injection primitive: content that arrived
    from outside the governed store must never be able to declare one.

    Un-launderable by construction.  The external markers are read straight
    off the block **before** any role-based promotion, so a crafted
    ``ActorRole: operator`` sitting next to ``Source: imported:slack``
    is still refused — unlike
    :func:`~mind_mem.provenance_class.classify_provenance`, whose
    operator rule deliberately wins for *scoring*.  That classifier is
    then consulted as a second, table-driven pass, so any external role
    or ingest token added there applies here for free.

    Pure and deterministic: reads only *block*, no I/O, no clock, no model.

    Args:
        block: A parsed block dict.

    Returns:
        A short human-readable reason, or ``""`` when the block's
        provenance permits guardrail recognition.
    """
    provenance = extract_provenance(dict(block))

    role = _norm_marker(provenance.get("actor_role"))
    if role in EXTERNAL_ROLES:
        return f"external actor role {role!r}"

    for field, raw in (("ToolId", provenance.get("tool_id")), ("Source", block.get("Source"))):
        token = _norm_marker(raw)
        if any(token.startswith(prefix) for prefix in EXTERNAL_TOKEN_PREFIXES):
            return f"{field} carries ingest token {token!r}"

    block_type = _norm_marker(block.get("Type"))
    if block_type in UNTRUSTED_BLOCK_TYPES:
        return f"ingest-authored block type {block_type!r}"

    provenance_class = classify_provenance(block)
    if provenance_class in UNTRUSTED_PROVENANCE_CLASSES:
        return f"provenance class {provenance_class!r}"

    return ""


# ---------------------------------------------------------------------------
# Parsing + loading
# ---------------------------------------------------------------------------


def parse_guardrail_block(block: Mapping[str, Any]) -> Guardrail:
    """Read one parsed block dict as a :class:`Guardrail`.

    Raises:
        GuardrailProvenanceError: the block carries external-ingest /
            imported provenance, so it may not mint a guardrail whatever
            its content declares.
        GuardrailSpecError: the block is not a ``GR-`` block, or a trigger
            field is malformed.
    """
    block_id = str(block.get("_id", ""))
    if not block_id.startswith(GUARDRAIL_ID_PREFIX):
        raise GuardrailSpecError(f"not a guardrail block: {block_id!r}")

    # Provenance is checked FIRST, before a single trigger field is read:
    # an untrusted block must be refused as a guardrail on origin alone,
    # never on how well-formed its declaration happens to be.
    refusal = guardrail_provenance_refusal(block)
    if refusal:
        raise GuardrailProvenanceError(f"{block_id}: untrusted provenance ({refusal}) — cannot declare a guardrail")

    statement = block.get("Statement") or block.get("Rule") or block.get("Summary") or ""
    if isinstance(statement, (list, tuple)):
        statement = " ".join(str(s) for s in statement)
    statement = str(statement).strip()
    if not statement:
        raise GuardrailSpecError(f"{block_id}: guardrail has no Statement")

    severity_raw = block.get("Severity", _DEFAULT_SEVERITY)
    severity = str(severity_raw).strip().casefold() or _DEFAULT_SEVERITY

    trigger = GuardrailTrigger(
        tools=coerce_patterns(block.get("TriggerTools"), field="TriggerTools", block_id=block_id, as_path=False),
        commands=coerce_patterns(block.get("TriggerCommands"), field="TriggerCommands", block_id=block_id, as_path=False),
        intents=coerce_patterns(block.get("TriggerIntents"), field="TriggerIntents", block_id=block_id, as_path=False),
        paths=coerce_patterns(block.get("TriggerPaths"), field="TriggerPaths", block_id=block_id, as_path=True),
    )
    if trigger.is_empty():
        raise GuardrailSpecError(
            f"{block_id}: guardrail declares no trigger ({', '.join(field for _, field in _TRIGGER_FIELDS)}) — it would never fire"
        )

    line_raw = block.get("_line", 0)
    try:
        line = int(line_raw)
    except (TypeError, ValueError):
        line = 0

    return Guardrail(
        block_id=block_id,
        statement=statement,
        severity=severity,
        trigger=trigger,
        source_file=str(block.get("_source_file", "")),
        line=line,
        status=normalise_status(block.get("Status")),
        block=MappingProxyType(dict(block)),
    )


def load_guardrails(workspace: str, policy: GuardrailPolicy | None = None) -> tuple[Guardrail, ...]:
    """Parse every live guardrail declared under *workspace*.

    Sources come from *policy* (default: :data:`DEFAULT_GUARDRAIL_FILES`).
    A source that escapes the workspace root is refused; a malformed
    guardrail is skipped with a loud warning so one bad block cannot take
    the whole constraint set (or recall) down.  A block whose provenance
    is external-ingest / imported is refused outright (logged as
    ``guardrail_provenance_refused``) even when its declaration is
    perfectly well-formed — see :func:`guardrail_provenance_refusal`.

    Returns:
        Guardrails in deterministic order (severity, then block ID), with
        duplicate IDs resolved first-source-wins.
    """
    policy = policy or GuardrailPolicy()
    workspace_real = os.path.realpath(workspace)
    prefix = workspace_real + os.sep

    found: dict[str, Guardrail] = {}
    for rel_path in policy.sources:
        candidate = os.path.realpath(os.path.join(workspace_real, rel_path))
        if not (candidate == workspace_real or candidate.startswith(prefix)):
            _log.warning("guardrail_source_escaped_workspace", source=rel_path)
            continue
        if not os.path.isfile(candidate):
            # A source the operator configured but that is not there is
            # invisible otherwise: every prohibition it declares is simply
            # absent from every recall, byte-identical to a workspace that
            # declares none.  The shipped default is exempt — no workspace
            # initialiser creates ``guardrails/GUARDRAILS.md``, so its
            # absence is the normal state and warning on it would be noise.
            if rel_path not in DEFAULT_GUARDRAIL_FILES:
                _log.warning("guardrail_source_missing", source=rel_path, resolved=candidate)
            continue
        try:
            blocks = parse_file(candidate)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            _log.warning("guardrail_source_parse_failed", source=rel_path, error=str(exc))
            continue
        for raw in blocks:
            if not str(raw.get("_id", "")).startswith(GUARDRAIL_ID_PREFIX):
                continue
            enriched = dict(raw)
            enriched.setdefault("_source_file", rel_path)
            try:
                guardrail = parse_guardrail_block(enriched)
            except GuardrailProvenanceError as exc:
                _log.warning("guardrail_provenance_refused", source=rel_path, error=str(exc))
                continue
            except GuardrailSpecError as exc:
                _log.warning("guardrail_block_rejected", source=rel_path, error=str(exc))
                continue
            if not guardrail.is_live():
                # Expected for a retired guardrail, so this is debug rather
                # than a warning — but it is never silent: a status the
                # parser rendered in a shape this code cannot read looks
                # exactly like a deliberate retirement from the outside.
                _log.debug(
                    "guardrail_not_live",
                    source=rel_path,
                    block_id=guardrail.block_id,
                    status=guardrail.status,
                )
                continue
            found.setdefault(guardrail.block_id, guardrail)

    return tuple(sorted(found.values(), key=Guardrail.sort_key))


def match_guardrails(
    guardrails: Sequence[Guardrail],
    context: GuardrailContext,
) -> tuple[tuple[Guardrail, tuple[str, ...]], ...]:
    """Return ``(guardrail, matched_dimensions)`` for every firing guardrail.

    Deterministic and model-free: ordered by ``(severity, block_id)``.
    An empty context matches nothing.
    """
    if context.is_empty():
        return ()
    hits: list[tuple[Guardrail, tuple[str, ...]]] = []
    for guardrail in guardrails:
        matched = guardrail.trigger.match(context)
        if matched:
            hits.append((guardrail, matched))
    hits.sort(key=lambda pair: pair[0].sort_key())
    return tuple(hits)
