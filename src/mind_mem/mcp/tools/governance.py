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

import json
import os
import re as _re_mod
import sqlite3
from typing import Any

from mind_mem.block_parser import get_active, parse_file
from mind_mem.storage import iter_active_blocks

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


@mcp_tool_observe
@_traced("propose_update")
def propose_update(
    block_type: str,
    statement: str,
    rationale: str = "",
    tags: str = "",
    confidence: str = "medium",
) -> str:
    """Propose a new decision or task. Writes to SIGNALS.md for human review.

    v3.6.1: ``rationale`` is required for ``block_type="decision"`` (tasks
    stay permissive). Forcing a written reason on every decision proposal
    means the audit trail answers "why" three months later without having
    to dig through Slack. Must be at least 8 non-whitespace characters
    so callers can't bypass the gate with a trivial string.
    """
    ws = _workspace()

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
        _qg_verdict = validate_block(statement, strict=_qg_is_strict)
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

    written = append_signals(ws, [signal], today)

    metrics.inc("mcp_proposals")
    _log.info("mcp_propose", block_type=block_type, confidence=confidence, written=written)

    # v3.2.1: invalidate the recall cache so the next query doesn't
    # serve a pre-proposal envelope that omits the new signal.
    _invalidate_recall_cache()

    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "proposed",
            "written": written,
            "location": "intelligence/SIGNALS.md",
            "next_step": ("Run /apply or `python3 maintenance/apply_engine.py` to review and promote to source of truth."),
            "safety": "This signal is in SIGNALS.md only. It has NOT been written to DECISIONS.md or TASKS.md.",
        },
        indent=2,
    )


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

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        success, message = apply_proposal(ws, proposal_id, dry_run=dry_run)

    log_output = capture.getvalue()

    metrics.inc("mcp_apply_calls")
    _log.info("mcp_approve_apply", proposal_id=proposal_id, dry_run=dry_run, success=success)

    # v3.2.1: invalidate recall cache only on a real (non-dry-run) apply.
    if success and not dry_run:
        _invalidate_recall_cache()

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
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "block_id": block_id,
                    "importance": round(importance, 4),
                    "co_occurring_blocks": co_blocks,
                },
                indent=2,
            )

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
