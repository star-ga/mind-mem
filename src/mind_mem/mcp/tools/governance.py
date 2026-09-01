"""Governance MCP tools — propose / apply / rollback / scan / contradictions / memory_evolution.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, governance domain). Six tools that cover the
"memory is never modified except by governance" invariant:

* ``propose_update`` — stage a new decision/task as a SIGNAL.
* ``approve_apply`` — apply a staged proposal (dry-run by default).
* ``rollback_proposal`` — restore workspace from pre-apply snapshot.
* ``scan`` — integrity scan (contradictions / drift / pending).
* ``list_contradictions`` — enriched contradiction listing.
* ``memory_evolution`` — A-MEM metadata for a block.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re as _re_mod
import sqlite3
from typing import Any

from mind_mem.block_parser import get_active, parse_blocks, parse_file
from mind_mem.event_fanout import (
    EVENT_CONTRADICTION_DETECTED,
    EVENT_PROPOSAL_APPLIED,
    EVENT_ROLLBACK_EXECUTED,
    emit_event,
)
from mind_mem.storage import iter_active_blocks
from mind_mem.v4.block_metadata import FLAG as _V4_METADATA_FLAG
from mind_mem.v4.feature_flags import FeatureDisabledError as _V4FeatureDisabledError
from mind_mem.v4.feature_flags import is_enabled_quiet as _v4_enabled_quiet

from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import _is_db_locked, _sqlite_busy_error, mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import get_logger, metrics
from ._helpers import traced as _traced

_log = get_logger("mcp_server")

# Backends whose blocks of record live on the local Markdown corpus
# (decisions/DECISIONS.md, intelligence/CONTRADICTIONS.md, …). For these
# ``scan`` keeps the exact legacy file-based behaviour so the default,
# zero-config SQLite path stays byte-for-byte unchanged. Every other
# backend (e.g. ``postgres``) keeps its blocks in the store, so ``scan``
# must enumerate via :func:`mind_mem.storage.iter_active_blocks` instead
# of ``parse_file`` over on-disk Markdown that is the empty init template
# on those backends (audit bugs #3 / #10). Kept in sync with
# ``mind_mem.storage._MARKDOWN_BACKENDS``.
_MARKDOWN_BACKENDS: frozenset[str] = frozenset({"markdown", "encrypted"})


# --------------------------------------------------------------------------
# Quality-gate near-duplicate window
# --------------------------------------------------------------------------
#
# ``quality_gate`` rule 6 (near_duplicate) is the only rule that needs more
# than the candidate text, and it stayed dead in the product because no
# caller ever built the window to compare against: this module called
# ``validate_block(statement, strict=...)`` and the preview tool called
# ``validate_block(text, strict=..., force=...)``, so ``recent`` was always
# ``None`` and the rule never executed. Building the window here — at the
# enforcement point — is what turns it on; ``mcp.tools.quality`` imports the
# same function so the preview and the enforcer compare against the same
# prior proposals rather than two different ideas of "recent".
#
# Bounds so this is cheap enough for the propose_update hot path: SIGNALS.md
# is append-only, so anything inside the 24h window is in the tail.
#
# Measured on an 8.5 MB / 18k-block SIGNALS.md: ~21 ms to build the window,
# and *flat* in file size (1 MB measured the same) because of the tail cap,
# plus ~26 ms worst case for the 400-way SequenceMatcher sweep when nothing
# matches. For scale, ``append_signals`` already reads the whole file and
# regexes it on every proposal — 16 ms on the same corpus, and that one grows
# linearly. So this adds a bounded cost beside an unbounded one.
_RECENT_TAIL_BYTES = 256 * 1024
_RECENT_MAX_BLOCKS = 400
_RECENT_LOOKBACK_DAYS = 2


def _recent_statements(ws: str, *, now: _dt.datetime | None = None) -> list[tuple[str, _dt.datetime]]:
    """Prior proposal texts from ``intelligence/SIGNALS.md``, oldest first.

    Feeds ``quality_gate.validate_block(recent=...)``. Each pair is
    ``(text, timestamp)``; the caller's 24h cutoff does the final filtering.

    Two properties are deliberate and bound what the rule can see:

    * **Timestamp granularity.** ``capture.append_signals`` now records a
      ``Captured:`` field holding the full UTC instant, and this reader
      prefers it. Blocks written before that field existed carry only
      ``Date: YYYY-MM-DD``, and those still fall back to the *start* of
      their day -- which reads as up to 24 h older than the write, making
      the effective retention ``24h - time-of-day`` rather than the
      documented 24 h. That is understated by calling it "near a day
      boundary": a block written at 22:00 is already 22 h old to this
      reader the moment it lands. The direction is conservative (the rule
      under-fires, never rejects something genuinely out of window), but
      only legacy blocks are affected now.
    * **Excerpt, not the raw statement.** ``capture.append_signals`` stores
      the statement truncated to 500 chars with newlines collapsed, which is
      exactly what a re-run of the same proposal would store again — so an
      identical re-proposal matches at ratio 1.0. A candidate longer than
      500 chars is compared against a truncated prior and scores lower, i.e.
      the rule under-reports for over-long statements. Again a false
      negative, which is where the rule already was.

    This complements rather than duplicates the exact-``ContentHash``
    dedupe already in ``append_signals``: that one catches byte-identical
    re-runs, this one catches the near misses it lets through.

    Never raises — a missing, unreadable or malformed SIGNALS.md yields an
    empty window, and an empty window is reported by the gate as a *skipped*
    rule, not a passed one.
    """
    path = os.path.join(ws, "intelligence", "SIGNALS.md")
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            truncated = size > _RECENT_TAIL_BYTES
            if truncated:
                fh.seek(size - _RECENT_TAIL_BYTES)
            raw = fh.read().decode("utf-8", errors="replace")
        if truncated:
            # A byte-offset seek can land mid-block; drop everything before
            # the first block header so no half-parsed block enters the
            # window. Losing the partial block is fine — it is the oldest
            # one in the tail, and the tail is already an over-approximation
            # of the 24h window.
            cut = raw.find("\n[")
            raw = raw[cut + 1 :] if cut != -1 else ""
        blocks = parse_blocks(raw)
    except FileNotFoundError:
        # The ordinary case: a workspace that has never staged a signal.
        return []
    except OSError as exc:
        _log.debug("quality_gate_recent_window_unavailable", path=path, error=str(exc))
        return []
    except Exception as exc:  # noqa: BLE001 — see below
        # This runs on the write path, ahead of the gate. If building the
        # window can raise, a malformed SIGNALS.md stops every proposal —
        # strictly worse than the missing rule this function exists to
        # restore. So: report, degrade to an empty window, and let the
        # verdict say near_duplicate was *skipped*. Loud, because unlike a
        # missing file this one is a bug worth reading.
        _log.warning(
            "quality_gate_recent_window_failed",
            path=path,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return []

    cutoff_day = (now or _dt.datetime.now(_dt.timezone.utc)).date() - _dt.timedelta(days=_RECENT_LOOKBACK_DAYS)
    window: list[tuple[str, _dt.datetime]] = []
    for block in blocks:
        text = block.get("Excerpt") or block.get("Statement")
        date_raw = block.get("Date")
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(date_raw, str):
            continue
        # Prefer the full instant when the writer recorded one. `Date:` alone
        # places the block at 00:00 of its day, so it reads as up to 24h older
        # than it is and the effective retention becomes (24h - time-of-day of
        # the write) rather than the documented 24h. `Captured:` (added in
        # capture.append_signals) removes the guesswork; blocks written before
        # that field existed still parse via the day fallback below.
        stamp: _dt.datetime | None = None
        captured_raw = block.get("Captured")
        if isinstance(captured_raw, str) and captured_raw.strip():
            try:
                parsed = _dt.datetime.fromisoformat(captured_raw.strip())
            except ValueError:
                parsed = None
            if parsed is not None:
                # A naive value means an unknown offset; read it as UTC, which
                # is what the writer emits.
                stamp = parsed if parsed.tzinfo else parsed.replace(tzinfo=_dt.timezone.utc)
        if stamp is None:
            try:
                stamp = _dt.datetime.strptime(date_raw.strip(), "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc)
            except ValueError:
                continue
        if stamp.date() < cutoff_day:
            continue
        window.append((text, stamp))
    return window[-_RECENT_MAX_BLOCKS:]


@mcp_tool_observe
@_traced("propose_update")
def propose_update(
    block_type: str,
    statement: str,
    rationale: str = "",
    tags: str = "",
    confidence: str = "medium",
    actor_id: str = "",
    actor_role: str = "",
    session_id: str = "",
    tool_id: str = "",
    purpose: str = "",
    content_source: str = "",
) -> str:
    """Propose a new decision or task. Writes to SIGNALS.md for human review.

    v3.6.1: ``rationale`` is required for ``block_type="decision"`` (tasks
    stay permissive). Forcing a written reason on every decision proposal
    means the audit trail answers "why" three months later without having
    to dig through Slack. Must be at least 8 non-whitespace characters
    so callers can't bypass the gate with a trivial string.

    Provenance (roadmap Group E, all optional): ``actor_id`` /
    ``actor_role`` / ``session_id`` / ``tool_id`` / ``purpose`` record
    who proposed the block, in what role, from which session, via which
    tool, and why. When provided they are written into the SIGNALS.md
    block as ``ActorId:`` / ``ActorRole:`` / ``SessionId:`` / ``ToolId:``
    / ``Purpose:`` fields and travel with the block from then on.
    Omitting them keeps the exact pre-Group-E behaviour.

    Content provenance (roadmap T-001, optional): ``content_source``
    declares what class of source the *statement text itself* came from —
    ``agent`` (composed by the model), ``user`` (typed by a human), or
    ``external`` (pulled in from outside the governed store). It is
    written as ``ContentSource:`` and is read downstream as a trust
    signal, so it is validated strictly: an unrecognised value is refused
    with an error envelope and **nothing is written**. Omitting it leaves
    the field absent — there is no default, because stamping ``agent`` on
    an untagged proposal would invent a claim the caller never made.
    """
    ws = _workspace()

    from mind_mem.block_provenance import (
        CONTENT_SOURCE_PARAM,
        CONTENT_SOURCES,
        MAX_PROVENANCE_VALUE_LEN,
        PROVENANCE_FIELDS,
        clean_provenance_value,
    )

    provenance_in = {
        "actor_id": actor_id,
        "actor_role": actor_role,
        "session_id": session_id,
        "tool_id": tool_id,
        "purpose": purpose,
        "content_source": content_source,
    }
    provenance: dict[str, str] = {}
    for prov_param in PROVENANCE_FIELDS:
        raw_val = provenance_in[prov_param]
        if not raw_val:
            continue
        if len(raw_val) > MAX_PROVENANCE_VALUE_LEN:
            return json.dumps(
                {
                    "error": (f"{prov_param} exceeds {MAX_PROVENANCE_VALUE_LEN} chars (provenance values are metadata, not content)"),
                    "field": prov_param,
                    "length": len(raw_val),
                }
            )
        # Vocabulary-bound fields raise rather than coerce. Refuse the whole
        # proposal here, before the SIGNALS.md append: a rejected trust tag
        # must not leave the statement written with the tag quietly dropped.
        try:
            cleaned = clean_provenance_value(prov_param, raw_val)
        except ValueError as exc:
            refusal: dict[str, object] = {"error": str(exc), "field": prov_param}
            if prov_param == CONTENT_SOURCE_PARAM:
                refusal["allowed"] = list(CONTENT_SOURCES)
            return json.dumps(refusal)
        if cleaned:
            provenance[prov_param] = cleaned

    if block_type not in ("decision", "task"):
        return json.dumps({"error": f"block_type must be 'decision' or 'task', got '{block_type}'"})

    if block_type == "decision" and len(rationale.strip()) < 8:
        return json.dumps(
            {
                "error": (
                    "rationale is required for decision proposals and must be at least "
                    "8 non-whitespace characters. Decisions without written reasons leave "
                    "no audit trail. Tasks may still omit rationale."
                ),
                "block_type": block_type,
                "rationale_length": len(rationale.strip()),
            }
        )

    # Issue #512 / T-003: bound rationale + tags length and sanitize
    # markdown injection vectors before they land in SIGNALS.md.
    if len(rationale) > 2000:
        return json.dumps(
            {
                "error": "rationale exceeds 2000 chars (issue #512 / T-003)",
                "rationale_length": len(rationale),
            }
        )
    raw_tags = [t.strip() for t in tags.split(",") if t.strip()]
    if len(raw_tags) > 16:
        return json.dumps({"error": "too many tags (max 16, issue #512 / T-003)", "tag_count": len(raw_tags)})
    for t in raw_tags:
        if len(t) > 64:
            return json.dumps({"error": "tag exceeds 64 chars (issue #512 / T-003)", "tag": t[:32] + "..."})

    from mind_mem.apply_engine import _sanitize_reason_for_markdown

    rationale = _sanitize_reason_for_markdown(rationale.strip()) if rationale else ""
    raw_tags = [_sanitize_reason_for_markdown(t) for t in raw_tags]

    # Quality gate pre-write check (v3.12.0 Theme B).
    from mind_mem.mcp.infra.config import _get_quality_gate_mode
    from mind_mem.quality_gate import validate_block

    _qg_mode = _get_quality_gate_mode(ws)
    if _qg_mode != "off":
        _qg_is_strict = _qg_mode == "strict"
        # `recent` is what makes rule 6 (near_duplicate) run at all. Without
        # it the rule reports as skipped and a re-proposal that is 97%+
        # identical to one staged an hour ago sails through.
        _qg_verdict = validate_block(statement, strict=_qg_is_strict, recent=_recent_statements(ws))
        if not _qg_verdict.accept:
            # Increment aggregate rejection counter.
            metrics.inc("quality_gate_rejections")
            # Increment per-rule counters for observability.
            for _qg_reason in _qg_verdict.reasons:
                _qg_rule = _qg_reason.split(":")[0].strip()
                metrics.inc(f"quality_gate_rejections_{_qg_rule}")
            _log.warning(
                "quality_gate_reject",
                mode=_qg_mode,
                reasons=_qg_verdict.reasons,
                block_type=block_type,
            )
            return json.dumps(
                {
                    "error": "quality_gate_rejection",
                    "mode": _qg_mode,
                    "reasons": _qg_verdict.reasons,
                    "advisory": _qg_verdict.advisory,
                    "hint": (
                        "Statement did not pass the quality gate. "
                        'Revise and resubmit, or set quality_gate.mode="advisory" '
                        "in mind-mem.json to downgrade to advisory-only."
                    ),
                },
                indent=2,
            )
        if _qg_verdict.advisory:
            # Advisory mode: log warnings but do not block.
            metrics.inc("quality_gate_rejections")
            for _qg_adv in _qg_verdict.advisory:
                _qg_rule = _qg_adv.split(":")[0].strip()
                metrics.inc(f"quality_gate_rejections_{_qg_rule}")
            _log.warning(
                "quality_gate_advisory",
                mode=_qg_mode,
                advisory=_qg_verdict.advisory,
                block_type=block_type,
            )

    # v4 schema-validation hooks + vocabulary-bound fields
    # (``v4.block_metadata`` / ``v4.vocabulary``, both default OFF).
    #
    # This is the door: block_type, tags and every provenance value arrive
    # from outside the store, and the quality gate above judges the STATEMENT
    # TEXT only. Nothing judged the fields. A workspace that wants
    # "ActorRole is one of these four" or "only these tags exist" had no place
    # to say it, and a per-kind invariant ("a task proposal must carry a
    # purpose") had nowhere to live either. Both now land here, before the
    # SIGNALS.md append, so a refused proposal leaves nothing written.
    #
    # The probe runs ONCE per proposal, at the outermost point and outside
    # every loop. With the flag off it is one stat-cached lookup that logs
    # nothing, touches no database and reads no JSON, so a default deployment
    # is indistinguishable from one that never had the surface.
    if _v4_enabled_quiet(_V4_METADATA_FLAG):
        from mind_mem.v4.block_metadata import validate_block as _v4_validate_block

        _v4_fields: dict[str, Any] = {
            "statement": statement,
            "confidence": confidence,
            "tags": raw_tags,
            **provenance,
        }
        try:
            # ``block_kind`` is left implicit: validate_block defaults it to
            # the kind argument, which is the one place that mapping is
            # defined. Spelling it again here would be a second definition
            # free to drift from the first.
            _v4_verdict = _v4_validate_block(block_type, _v4_fields, workspace=ws)
        except _V4FeatureDisabledError:
            # The quiet probe stats the config; validate_block re-reads it.
            # A config removed between the two answers "off" on the second
            # read. Skipping the leg is the correct response to "the surface
            # is off" — it must never turn a flag race into a failed write.
            _log.warning("v4_block_metadata_flag_race", block_type=block_type)
        else:
            if not _v4_verdict.ok:
                metrics.inc("v4_schema_validation_rejections")
                _log.warning(
                    "v4_schema_validation_reject",
                    block_type=block_type,
                    reason=_v4_verdict.reason,
                )
                return json.dumps(
                    {
                        "error": "schema_validation_rejection",
                        "reason": _v4_verdict.reason,
                        "block_type": block_type,
                        "hint": (
                            "The proposal was refused by this workspace's v4 field rules — a "
                            "registered schema validator for this block kind, or a controlled "
                            "vocabulary declared under mind-mem.json 'vocabularies' / "
                            "vocabularies.json. Nothing was written to SIGNALS.md."
                        ),
                    },
                    indent=2,
                )
            if _v4_verdict.reason.startswith("vocabulary_flagged:"):
                # ``flag`` mode: reported, not enforced. The write proceeds.
                metrics.inc("v4_vocabulary_flagged")
                _log.warning(
                    "v4_vocabulary_flagged",
                    block_type=block_type,
                    reason=_v4_verdict.reason,
                )

    from datetime import datetime

    from mind_mem.capture import CONFIDENCE_TO_PRIORITY, append_signals

    today = datetime.now().strftime("%Y-%m-%d")
    priority = CONFIDENCE_TO_PRIORITY.get(confidence, "P2")

    statement = statement[:500]

    signal = {
        "line": 0,
        "type": block_type,
        "text": statement,
        "pattern": "mcp_propose_update",
        "confidence": confidence,
        "priority": priority,
        "structure": {
            "subject": " ".join(statement.split()[:3]) if statement else "",
            "tags": raw_tags,
        },
    }
    if rationale:
        signal["structure"]["rationale"] = rationale  # type: ignore[index]
    if provenance:
        signal["provenance"] = provenance

    written = append_signals(ws, [signal], today)

    metrics.inc("mcp_proposals")
    _log.info("mcp_propose", block_type=block_type, confidence=confidence, written=written)

    # v3.2.1: invalidate the recall cache so the next query doesn't
    # serve a pre-proposal envelope that omits the new signal.
    _invalidate_recall_cache()

    response: dict[str, Any] = {
        "_schema_version": MCP_SCHEMA_VERSION,
        "status": "proposed",
        "written": written,
        "location": "intelligence/SIGNALS.md",
        "next_step": ("Run /apply or `python3 maintenance/apply_engine.py` to review and promote to source of truth."),
        "safety": "This signal is in SIGNALS.md only. It has NOT been written to DECISIONS.md or TASKS.md.",
    }
    if provenance:
        response["provenance_attached"] = sorted(provenance)
    return json.dumps(response, indent=2)


def _invalidate_recall_cache() -> None:
    """Flush the recall cache after a governance event.

    Namespace-wide invalidation — targeted per-block invalidation
    would require tracking which queries touched which blocks, which
    is more complexity than the typical workspace needs. Best-effort
    (swallows errors so governance operations never fail because the
    cache backend is unavailable).
    """
    try:
        from mind_mem.recall_cache import invalidate

        invalidate()
    except Exception as exc:  # pragma: no cover — best-effort
        _log.debug("recall_cache_invalidate_failed", error=str(exc))


def _resolve_backend(ws: str) -> str:
    """Return the configured ``block_store.backend`` for *ws*.

    Routes through ``mind_mem.storage._backend_name`` — the single source
    of truth for backend detection — via a lazy import so this module
    stays import-cheap and free of import cycles. Degrades to
    ``"markdown"`` on any failure, so the default SQLite / Markdown path
    is never disturbed by a config read error.
    """
    try:
        from ...storage import _backend_name

        return _backend_name(ws)
    except Exception:  # pragma: no cover - defensive: storage/config failure
        return "markdown"


def _is_decision_block(block: dict[str, Any]) -> bool:
    """True when *block* is a decision (the unit ``scan`` counts).

    A block enumerated from the store may carry its origin in several
    shapes: the ``_source_label`` tag the Markdown enumerator sets, the
    ``_source_file`` path the store backends populate, a ``Type`` field,
    or — for the canonical ``D-YYYYMMDD-NNN`` ids — the ``D-`` id prefix.
    Any one is sufficient. This mirrors how the legacy Markdown path
    scoped the decision count to ``decisions/DECISIONS.md``.
    """
    label = str(block.get("_source_label", "")).lower()
    if label == "decisions":
        return True
    source = str(block.get("_source_file", "")).lower()
    if "decisions/decisions.md" in source.replace("\\", "/"):
        return True
    btype = str(block.get("Type", "")).lower()
    if btype == "decision":
        return True
    return str(block.get("_id", "")).startswith("D-")


# Fields scanned for a block's human-readable assertion, most-specific
# first. ``Statement`` is the canonical mind-mem decision/task field;
# the rest are fallbacks for richer block shapes.
_STATEMENT_FIELDS: tuple[str, ...] = (
    "Statement",
    "Decision",
    "Title",
    "Summary",
    "Description",
    "Content",
    "Action",
    "Details",
)

# Antonym pairs that, when one appears on each side of an otherwise
# topically-similar block pair, signal a direct contradiction. Lower-case,
# matched on word boundaries.
_ANTONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("enable", "disable"),
    ("allow", "deny"),
    ("accept", "reject"),
    ("add", "remove"),
    ("increase", "decrease"),
    ("true", "false"),
    ("on", "off"),
    ("always", "never"),
    ("required", "forbidden"),
)

# Negation cues — a block that carries one while its topical twin does not
# is asserting the opposite of the same subject.
_NEGATION_RE = _re_mod.compile(
    r"\b(?:not|never|no|won't|will not|cannot|can't|don't|doesn't|"
    r"shouldn't|without|avoid|reject|deny|disable|forbid)\b"
)

# Minimum topical overlap (Jaccard over content words) before two blocks
# are considered to be about the same subject. Below this they are simply
# unrelated, not contradictory.
_TOPIC_OVERLAP_THRESHOLD = 0.25

# ``<subject> is/are/= <value>`` assignment pattern. Two blocks that
# assign *different* values to the *same* subject are a value conflict
# (e.g. "default backend is SQLite" vs "default backend is Postgres").
_ASSIGNMENT_RE = _re_mod.compile(r"\b([a-z][a-z0-9 ]{2,40}?)\s+(?:is|are|=|:|should be|must be)\s+([a-z0-9][a-z0-9._-]*)")


def _statement_text(block: dict[str, Any]) -> str:
    """Return the block's human-readable assertion text (lower-cased)."""
    parts: list[str] = []
    for field in _STATEMENT_FIELDS:
        val = block.get(field)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return " ".join(parts).lower()


def _content_tokens(text: str) -> set[str]:
    """Topic tokens for *text* — drop short stop-ish words and negations."""
    tokens = set(_re_mod.findall(r"\b[a-z0-9]{3,}\b", text))
    return {t for t in tokens if not _NEGATION_RE.fullmatch(t)}


def _detect_statement_contradictions(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find pairwise statement-level contradictions among *blocks*.

    Backend-agnostic and self-contained: operates on already-enumerated
    active block dicts (from :func:`mind_mem.storage.iter_active_blocks`),
    so it behaves identically whether the blocks came from the Markdown
    corpus or a Postgres store. A pair is flagged when the two blocks are
    about the same subject (Jaccard topic overlap above
    :data:`_TOPIC_OVERLAP_THRESHOLD`) **and** one of:

    * an antonym pair appears split across the two blocks
      (e.g. ``enable`` here, ``disable`` there), or
    * exactly one side carries a negation cue (``not`` / ``never`` /
      ``will not`` …) — the same subject asserted in the affirmative on
      one side and the negative on the other.

    Deterministic, zero-dependency, and owned entirely by this module
    (no dependence on another component's tunable thresholds). Returns a
    list of ``{"block_a", "block_b", "reason"}`` dicts, deduplicated by
    unordered id pair.
    """
    entries: list[tuple[str, str, set[str]]] = []
    for b in blocks:
        bid = b.get("_id", "")
        if not bid:
            continue
        text = _statement_text(b)
        if not text.strip():
            continue
        entries.append((bid, text, _content_tokens(text)))

    contradictions: list[dict[str, Any]] = []
    for i in range(len(entries)):
        id_a, text_a, tok_a = entries[i]
        for j in range(i + 1, len(entries)):
            id_b, text_b, tok_b = entries[j]
            if not tok_a or not tok_b:
                continue
            overlap = len(tok_a & tok_b) / len(tok_a | tok_b)
            if overlap < _TOPIC_OVERLAP_THRESHOLD:
                continue

            reason: str | None = None
            for word_a, word_b in _ANTONYM_PAIRS:
                a_here = _word_in(word_a, text_a)
                b_there = _word_in(word_b, text_b)
                a_there = _word_in(word_a, text_b)
                b_here = _word_in(word_b, text_a)
                if (a_here and b_there) or (b_here and a_there):
                    reason = f"antonym conflict: {word_a} vs {word_b}"
                    break

            if reason is None:
                reason = _value_conflict_reason(text_a, text_b)

            if reason is None:
                neg_a = bool(_NEGATION_RE.search(text_a))
                neg_b = bool(_NEGATION_RE.search(text_b))
                if neg_a != neg_b:
                    reason = "affirmation vs negation on a shared subject"

            if reason is not None:
                contradictions.append({"block_a": id_a, "block_b": id_b, "reason": reason})
    return contradictions


def _word_in(word: str, text: str) -> bool:
    """Whole-word membership test (text is already lower-cased)."""
    return bool(_re_mod.search(r"\b" + _re_mod.escape(word) + r"\b", text))


def _assignments(text: str) -> dict[str, set[str]]:
    """Map ``subject -> {values}`` for every ``X is/= Y`` clause in *text*."""
    out: dict[str, set[str]] = {}
    for subj, val in _ASSIGNMENT_RE.findall(text):
        out.setdefault(subj.strip(), set()).add(val.strip())
    return out


def _value_conflict_reason(text_a: str, text_b: str) -> str | None:
    """Reason string when the two texts assign different values to one subject.

    e.g. ``"default backend is SQLite"`` vs ``"default backend is
    Postgres"`` — same subject (``default backend``), disjoint values
    (``sqlite`` / ``postgres``). Returns ``None`` when there is no shared
    subject with conflicting values.
    """
    a = _assignments(text_a)
    b = _assignments(text_b)
    for subject, vals_a in a.items():
        vals_b = b.get(subject)
        if vals_b and vals_a.isdisjoint(vals_b):
            va = sorted(vals_a)[0]
            vb = sorted(vals_b)[0]
            return f"value conflict on '{subject}': {va} vs {vb}"
    return None


def _log_digest(text: str) -> str:
    """Short SHA-256 of engine output — a correlation handle, never the text.

    An event payload may carry a HASH of the apply log so a subscriber can tie
    its notification to the receipt. It must never carry the log itself: the
    apply log quotes block content, and ``LoggingPublisher`` writes payloads
    verbatim into the log while ``RedisStreamPublisher`` puts them on a stream.
    """
    import hashlib

    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _world_staleness_enabled(ws: str) -> bool:
    """True when the ``v4.world_staleness`` flag is ON for *ws*.

    Lazy import so the default (flag-off) ``scan`` path never even loads
    the checker, and a broken feature module can never break ``scan``.
    """
    try:
        from mind_mem.world_staleness import is_world_staleness_enabled

        return is_world_staleness_enabled(ws)
    except Exception as exc:  # pragma: no cover - defensive: import/config failure
        _log.warning("world_staleness_flag_check_failed", error=str(exc))
        return False


def _world_staleness_summary(ws: str) -> dict[str, Any]:
    """External-anchor liveness summary for *ws* — only called when flag is ON.

    Deterministic + local-only: filesystem existence, a per-language
    definition grep, and ``git rev-parse`` / ``merge-base``. No network,
    no model. A failure degrades to a zeroed summary carrying the error
    rather than taking the whole scan down.
    """
    try:
        from mind_mem.world_staleness import world_staleness_summary

        return world_staleness_summary(ws)
    except Exception as exc:
        _log.warning("world_staleness_check_failed", error=str(exc))
        return {
            "blocks_scanned": 0,
            "blocks_with_anchors": 0,
            "anchors_checked": 0,
            "stale_blocks": [],
            "dead_anchor_count": 0,
            "dead_anchors": [],
            "error": str(exc),
        }


@mcp_tool_observe
@_traced("scan")
def scan() -> str:
    """Run integrity scan — contradictions, drift, dead decisions, impact graph.

    Backend-aware (audit bugs #3 / #10): the legacy implementation read
    only the local Markdown corpus via ``parse_file`` over
    ``decisions/DECISIONS.md`` + ``intelligence/*.md``. On a Postgres
    (or any non-Markdown) backend those files are the empty init
    templates, so the contradiction / drift / decision counts were a
    silent no-op even when the store held contradictory blocks. The
    block enumeration now routes through
    :func:`mind_mem.storage.iter_active_blocks`, so governance sees the
    configured backend's blocks. The Markdown / SQLite default path keeps
    its exact legacy file-based behaviour.
    """
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    backend = _resolve_backend(ws)
    is_markdown = backend in _MARKDOWN_BACKENDS

    checks: dict[str, Any] = {}

    if is_markdown:
        # Default (SQLite / Markdown) path — byte-for-byte unchanged: read
        # decisions/DECISIONS.md directly so the ``total`` count still
        # includes archived blocks (which iter_active_blocks excludes).
        decisions_path = os.path.join(ws, "decisions", "DECISIONS.md")
        if os.path.isfile(decisions_path):
            blocks = parse_file(decisions_path)
            active = get_active(blocks)
            checks["decisions"] = {
                "total": len(blocks),
                "active": len(active),
            }
        else:
            checks["decisions"] = {"total": 0, "active": 0}
    else:
        # Non-Markdown backend (e.g. postgres) — the blocks of record live
        # in the store; enumerate via the shared backend-aware helper.
        active_blocks = iter_active_blocks(ws)
        decision_blocks = [b for b in active_blocks if _is_decision_block(b)]
        checks["decisions"] = {
            "total": len(decision_blocks),
            "active": len(decision_blocks),
        }

    if is_markdown:
        # Legacy file-based contradiction surface (CONTRADICTIONS.md +
        # conflict_resolver) — unchanged so the default path stays green.
        contra_path = os.path.join(ws, "intelligence", "CONTRADICTIONS.md")
        raw_count = 0
        if os.path.isfile(contra_path):
            raw_count = len(parse_file(contra_path))
        try:
            from mind_mem.conflict_resolver import resolve_contradictions

            resolutions = resolve_contradictions(ws)
            checks["contradictions"] = {
                "raw": raw_count,
                "resolvable": len(resolutions),
            }
        except (ImportError, OSError, ValueError) as exc:
            _log.warning("scan_contradiction_check_failed", error=str(exc))
            checks["contradictions"] = {"raw": raw_count, "resolvable": 0}
    else:
        # Non-Markdown backend — detect statement-level contradictions over
        # the store-resident active blocks. ``resolvable`` is 0 here:
        # auto-resolution still flows through the Markdown supersede-proposal
        # pipeline, but ``raw`` now correctly reflects the store's contents
        # instead of silently reporting 0 (audit bugs #3 / #10).
        try:
            store_contradictions = _detect_statement_contradictions(active_blocks)
            checks["contradictions"] = {
                "raw": len(store_contradictions),
                "resolvable": 0,
            }
        except Exception as exc:  # pragma: no cover - defensive
            _log.warning("scan_store_contradiction_check_failed", error=str(exc))
            checks["contradictions"] = {"raw": 0, "resolvable": 0}

    drift_path = os.path.join(ws, "intelligence", "DRIFT.md")
    if os.path.isfile(drift_path):
        drifts = parse_file(drift_path)
        checks["drift_items"] = len(drifts)
    else:
        checks["drift_items"] = 0

    signals_path = os.path.join(ws, "intelligence", "SIGNALS.md")
    if os.path.isfile(signals_path):
        signals = parse_file(signals_path)
        checks["pending_signals"] = len(signals)
    else:
        checks["pending_signals"] = 0

    # External grounding (v4 ``world_staleness`` flag, default OFF). When
    # the flag is off this branch is not taken and no key is added, so the
    # scan payload is byte-identical to the pre-feature output.
    if _world_staleness_enabled(ws):
        checks["world_staleness"] = _world_staleness_summary(ws)

    result: dict[str, Any] = {
        "_schema_version": MCP_SCHEMA_VERSION,
        "backend": backend,
        "checks": checks,
    }
    metrics.inc("mcp_scans")
    _log.info("mcp_scan", backend=backend, checks=checks)

    return json.dumps(result, indent=2)


@mcp_tool_observe
def list_contradictions() -> str:
    """List detected contradictions with resolution analysis."""
    ws = _workspace()

    from mind_mem.conflict_resolver import resolve_contradictions

    resolutions = resolve_contradictions(ws)
    if not resolutions:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "status": "clean",
                "contradictions": 0,
                "message": "No contradictions found.",
            }
        )

    enriched: list[dict] = []
    try:
        from mind_mem.auto_resolver import AutoResolver

        suggestions = AutoResolver(ws).suggest_resolutions()
        by_id = {s.contradiction_id: s for s in suggestions}
        for res in resolutions:
            sug = by_id.get(str(res.get("contradiction_id", "")))
            merged = dict(res)
            if sug is not None:
                merged["confidence_score"] = sug.confidence_score
                merged["side_effects"] = list(sug.side_effects)
                merged["preference_boost_applied"] = True
            enriched.append(merged)
    except Exception as exc:  # pragma: no cover — best-effort
        _log.warning("auto_resolver_enrich_failed", error=str(exc))
        enriched = list(resolutions)

    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "contradictions_found",
            "contradictions": len(enriched),
            "resolutions": enriched,
        },
        indent=2,
        default=str,
    )


@mcp_tool_observe
def approve_apply(proposal_id: str, dry_run: bool = True) -> str:
    """Apply a staged proposal from intelligence/proposed/."""
    ws = _workspace()

    import re

    if not re.match(r"^P-\d{8}-\d{3}$", proposal_id):
        return json.dumps({"error": f"Invalid proposal ID format: {proposal_id}. Expected P-YYYYMMDD-NNN."})

    import contextlib
    import io

    from mind_mem.apply_engine import apply_proposal, find_proposal
    from mind_mem.contradiction_detector import check_proposal_contradictions

    contra_report = None
    try:
        proposal, _source = find_proposal(ws, proposal_id)
        if proposal:
            contra_report = check_proposal_contradictions(ws, proposal)
    except Exception as e:
        _log.warning("contradiction_check_failed", error=str(e))

    # Governance event: a detection, so it fires on dry runs too — a subscriber
    # watching for conflicts wants to hear about them before the apply, not
    # only when one gets through. Counts only; the conflicting STATEMENTS stay
    # in the response envelope, which is inside the ACL.
    if contra_report and contra_report.get("has_contradictions"):
        emit_event(
            ws,
            EVENT_CONTRADICTION_DETECTED,
            lambda: {
                "proposal_id": proposal_id,
                "dry_run": dry_run,
                "contradiction_count": int(contra_report.get("contradiction_count") or 0),
                "conflict_count": int(contra_report.get("total_conflicts") or 0),
            },
        )

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        success, message = apply_proposal(ws, proposal_id, dry_run=dry_run)

    log_output = capture.getvalue()

    metrics.inc("mcp_apply_calls")
    _log.info("mcp_approve_apply", proposal_id=proposal_id, dry_run=dry_run, success=success)

    # v3.2.1: invalidate recall cache only on a real (non-dry-run) apply.
    if success and not dry_run:
        _invalidate_recall_cache()
        # 5.0.1: one event per REAL apply. Deliberately not on a dry run —
        # "applied" must mean the corpus changed, or a subscriber that acts on
        # the event acts on a change that never happened. Emission is the last
        # thing this branch does and cannot raise, so a dead subscriber leaves
        # the apply applied.
        emit_event(
            ws,
            EVENT_PROPOSAL_APPLIED,
            lambda: {
                "proposal_id": proposal_id,
                "success": True,
                "log_digest": _log_digest(log_output),
            },
        )

    blocked_by_contradictions = not success and message == "Blocked: contradictions detected"

    result: dict[str, Any] = {
        "_schema_version": MCP_SCHEMA_VERSION,
        "status": (
            "blocked_contradictions"
            if blocked_by_contradictions
            else "applied"
            if success and not dry_run
            else "dry_run_passed"
            if success
            else "failed"
        ),
        "proposal_id": proposal_id,
        "dry_run": dry_run,
        "success": success,
        "message": message,
        "log": log_output[-2000:] if len(log_output) > 2000 else log_output,
        "next_step": (
            "Resolve contradictions or set contradiction.block_on_detect=false in mind-mem.json."
            if blocked_by_contradictions
            else "Call again with dry_run=False to apply."
            if success and dry_run
            else None
        ),
    }

    if contra_report:
        result["contradictions"] = {
            "summary": contra_report["summary"],
            "has_contradictions": contra_report["has_contradictions"],
            "contradiction_count": contra_report["contradiction_count"],
            "total_conflicts": contra_report["total_conflicts"],
            "conflicts": contra_report["conflicts"],
        }

    return json.dumps(result, indent=2)


@mcp_tool_observe
def reject_proposal(proposal_id: str, reason: str) -> str:
    """Reject a staged proposal explicitly, preserving the rationale.

    v3.6.1: fills the "no explicit rejection tool" gap — previously
    rejection happened implicitly by letting proposals expire. Now
    operators can reject with a mandatory written reason (≥ 8
    non-whitespace characters) which gets appended as a ``Reason:``
    line inside the proposal block. The audit chain answers "why did
    we reject P-20260412-007?" months later with the rationale in the
    file, not in Slack.

    Args:
        proposal_id: The proposal's ID (e.g. ``P-20260412-007``).
        reason: Human-written rationale. Required, ≥ 8 non-whitespace
            characters. Multi-line reasons are preserved verbatim.
    """
    ws = _workspace()

    if not proposal_id or not proposal_id.strip():
        return json.dumps({"error": "proposal_id is required"})

    if len(reason.strip()) < 8:
        return json.dumps(
            {
                "error": (
                    "reason is required and must be at least 8 non-whitespace characters. "
                    "Rejections without a written reason leave no audit trail."
                ),
                "proposal_id": proposal_id,
                "reason_length": len(reason.strip()),
            }
        )

    from mind_mem.apply_engine import _mark_proposal_status, find_proposal

    proposal, source_file = find_proposal(ws, proposal_id)
    if not proposal or not source_file:
        return json.dumps({"error": f"proposal not found: {proposal_id}"})

    current_status = proposal.get("Status", "").strip().lower()
    if current_status in ("applied", "rolled_back"):
        return json.dumps(
            {
                "error": (f"cannot reject proposal in status '{current_status}'. Use rollback_proposal for applied proposals."),
                "proposal_id": proposal_id,
                "current_status": current_status,
            }
        )

    ok = _mark_proposal_status(source_file, proposal_id, "rejected", reason=reason)
    if not ok:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": (
                    "rejection failed: could not persist the new status + rationale to the "
                    "proposal file (lock contention or I/O error). Check stderr for details "
                    "and retry. No state change was committed."
                ),
                "proposal_id": proposal_id,
                "source_file": source_file,
                "status": "unchanged",
            },
            indent=2,
        )

    metrics.inc("mcp_rejections")
    _log.info(
        "mcp_reject",
        proposal_id=proposal_id,
        reason_length=len(reason.strip()),
    )

    _invalidate_recall_cache()

    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "rejected",
            "proposal_id": proposal_id,
            "source_file": source_file,
            "reason_preserved": True,
        },
        indent=2,
    )


@mcp_tool_observe
def rollback_proposal(receipt_ts: str, reason: str = "") -> str:
    """Rollback an applied proposal using its receipt timestamp.

    v3.6.1: ``reason`` is required (≥ 8 non-whitespace characters). The
    rationale is appended to the APPLY_RECEIPT.md as a ``Reason: <text>``
    line so the audit chain preserves why the rollback was initiated.
    This closes the "recurring churn is invisible" gap — a rejection
    rationale three months ago shows up next to the receipt, not in
    chat scrollback.
    """
    ws = _workspace()

    import re

    if not re.match(r"^\d{8}-\d{6}$", receipt_ts):
        return json.dumps({"error": f"Invalid receipt timestamp format: {receipt_ts}. Expected YYYYMMDD-HHMMSS."})

    if len(reason.strip()) < 8:
        return json.dumps(
            {
                "error": (
                    "reason is required and must be at least 8 non-whitespace characters. "
                    "Rollbacks without a written reason leave no audit trail for why the "
                    "revert happened."
                ),
                "receipt_ts": receipt_ts,
                "reason_length": len(reason.strip()),
            }
        )

    import contextlib
    import io

    from mind_mem.apply_engine import rollback as engine_rollback

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        success = engine_rollback(ws, receipt_ts, reason=reason)

    log_output = capture.getvalue()

    metrics.inc("mcp_rollbacks")
    _log.info("mcp_rollback", receipt_ts=receipt_ts, success=success, has_reason=True)

    # v3.2.1: post-rollback cache flush so recall sees the restored state.
    if success:
        _invalidate_recall_cache()
        # ``reason`` is operator free text and never leaves in a payload — its
        # LENGTH is the auditable part, and the text itself is already on the
        # receipt where the audit chain covers it.
        emit_event(
            ws,
            EVENT_ROLLBACK_EXECUTED,
            lambda: {
                "receipt_ts": receipt_ts,
                "success": True,
                "reason_length": len(reason.strip()),
                "log_digest": _log_digest(log_output),
            },
        )

    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "rolled_back" if success else "rollback_failed",
            "receipt_ts": receipt_ts,
            "success": success,
            "log": log_output[-2000:] if len(log_output) > 2000 else log_output,
        },
        indent=2,
    )


@mcp_tool_observe
def memory_evolution(block_id: str, action: str = "get") -> str:
    """A-MEM metadata for a block — importance, access patterns, keywords."""
    if not _re_mod.match(r"^[A-Z]+-[a-zA-Z0-9_.-]+$", block_id):
        return json.dumps({"error": f"Invalid block_id format: {block_id}"})
    ws = _workspace()
    db_path = os.path.join(ws, "memory", "block_meta.db")

    try:
        from mind_mem.block_metadata import BlockMetadataManager

        mgr = BlockMetadataManager(db_path)

        if action == "update":
            importance = mgr.update_importance(block_id)
            metrics.inc("mcp_evolution_updates")
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "block_id": block_id,
                    "action": "updated",
                    "importance": round(importance, 4),
                },
                indent=2,
            )
        else:
            importance = mgr.get_importance_boost(block_id)
            co_blocks = mgr.get_co_occurring_blocks(block_id)
            metrics.inc("mcp_evolution_reads")
            payload: dict[str, Any] = {
                "_schema_version": MCP_SCHEMA_VERSION,
                "block_id": block_id,
                "importance": round(importance, 4),
                "co_occurring_blocks": co_blocks,
            }
            # Provenance (Group E) — only included when recorded.
            prov = mgr.get_provenance(block_id)
            if prov:
                payload["provenance"] = prov
            return json.dumps(payload, indent=2)

    except ImportError:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "memory_evolution requires block_metadata module",
                "block_id": block_id,
            },
            indent=2,
        )
    except sqlite3.OperationalError as exc:
        if _is_db_locked(exc):
            return _sqlite_busy_error()
        raise
    except (OSError, ValueError, KeyError) as exc:
        _log.warning("memory_evolution_failed", block_id=block_id, error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "Memory evolution lookup failed. Access history may not be initialized.",
                "block_id": block_id,
            },
            indent=2,
        )


def register(mcp) -> None:
    """Wire the governance tools onto *mcp*."""
    mcp.tool(propose_update)
    mcp.tool(scan)
    mcp.tool(list_contradictions)
    mcp.tool(approve_apply)
    mcp.tool(reject_proposal)
    mcp.tool(rollback_proposal)
    mcp.tool(memory_evolution)
