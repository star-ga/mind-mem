"""Session-boundary preservation for recall (v3.3.0 Tier 2 #5).

LoCoMo inputs are multi-session dialogues. A question like "What did
Alice say about the outage?" is most likely answered within a single
session of the conversation — the one where that topic was actively
discussed. Naive BM25 / vector retrieval scatters results across
sessions and gives equal weight to each.

This scorer identifies which session(s) the top-ranked results come
from, then boosts other candidates from the same sessions. Effect:
conversational co-reference ("she said earlier") gets surfaced
alongside the verbatim hit.

Session identity is derived from the ``dia_id`` field (LoCoMo dialog
turn IDs like ``DIA-D1-3`` or ``D1:3``) or the ``SessionId`` field
when ingestion added one. Blocks without either are untouched.

Opt-in via:

    {
      "retrieval": {
        "session_boost": {
          "enabled": false,
          "auto_enable": true,
          "top_seed_count": 3,
          "boost": 0.3
        }
      }
    }

Auto-enable fires when any result in the top-N carries session info,
so non-LoCoMo workspaces see zero effect.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .feature_gate import FeatureGate, FieldSpec, strict_int, strict_number
from .observability import get_logger

_log = get_logger("session_boost")


# LoCoMo dialog-turn patterns. DIA-D1-3 (canonical) or D1:3 (legacy
# raw field format). Session token = everything before the turn
# separator.
_SESSION_FROM_DIA_RE = re.compile(r"(?:DIA-)?([A-Za-z]+\d+)[-:]\d+")


def _session_of(block: dict) -> str | None:
    """Derive a canonical session ID from a block's metadata."""
    if not isinstance(block, dict):
        return None
    # Preferred: explicit SessionId / session_id field.
    for key in ("SessionId", "session_id", "Session"):
        val = block.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # LoCoMo dialog turn id.
    for key in ("dia_id", "DiaId", "DIA"):
        val = block.get(key)
        if isinstance(val, str):
            m = _SESSION_FROM_DIA_RE.match(val)
            if m:
                return m.group(1)
    # Fall back to block-ID prefix for DIA-* blocks.
    bid = block.get("_id") or ""
    if isinstance(bid, str) and bid.startswith("DIA-"):
        m = _SESSION_FROM_DIA_RE.match(bid)
        if m:
            return m.group(1)
    return None


def apply_session_boost(
    results: list[dict],
    *,
    top_seed_count: int = 3,
    boost: float = 0.3,
    score_field: str = "score",
) -> list[dict]:
    """Boost results that share a session with the top seeds.

    Args:
        results: Ranked recall results (highest score first).
        top_seed_count: How many top results define the "active"
            session(s). Ties broken by existing rank.
        boost: Multiplicative bump applied to score for same-session
            blocks below the seed set. ``score *= (1 + boost)``.
        score_field: Which field carries the numeric score.

    Returns:
        Re-ranked copy of ``results`` with boosted scores where
        applicable. Blocks carry ``_session_boost`` when they received
        the bump so downstream callers can audit.
    """
    if not results or top_seed_count <= 0 or boost <= 0:
        return results

    # Identify the active session(s) from the top seeds.
    seed_sessions: Counter[str] = Counter()
    for b in results[:top_seed_count]:
        sid = _session_of(b)
        if sid:
            seed_sessions[sid] += 1

    if not seed_sessions:
        return results

    active = {sid for sid, _ in seed_sessions.most_common()}
    boosted = 0
    out: list[dict] = []
    for b in results:
        sid = _session_of(b)
        new = dict(b)
        if sid and sid in active:
            current = float(new.get(score_field, 0.0) or 0.0)
            new[score_field] = current * (1 + boost)
            new["_session_boost"] = round(boost, 3)
            boosted += 1
        out.append(new)

    # Re-sort by updated scores (stable to preserve original tie order).
    out.sort(key=lambda x: float(x.get(score_field, 0.0) or 0.0), reverse=True)
    if boosted:
        _log.info(
            "session_boost_applied",
            active_sessions=len(active),
            boosted=boosted,
            boost=boost,
        )
    return out


def _has_session_info(query: str | None, results: list[dict] | None) -> bool:
    """Auto-enable detector: does the head of the result set carry sessions?

    Reads at most the top 10 hits and nothing else — no clock, no disk, no
    log line. A probe that decides a feature is OFF must leave no trace
    that it ran, or "off" stops meaning "as if unwired".
    """
    if not results:
        return False
    return any(_session_of(b) for b in results[:10])


#: The ``retrieval.session_boost`` gate, declared once.
SESSION_BOOST_GATE = FeatureGate(
    name="session_boost",
    fields={
        "top_seed_count": FieldSpec(
            default=3,
            coerce=strict_int,
            validate=lambda v: v > 0,
        ),
        "boost": FieldSpec(
            default=0.3,
            coerce=strict_number,
            validate=lambda v: 0 < v <= 5,
        ),
    },
    auto_detector=_has_session_info,
    implicit_section=True,
)


def is_session_boost_enabled(config: dict[str, Any] | None, results: list[dict] | None = None) -> bool:
    """Decide whether session_boost should fire.

    Auto-enables when the config section exists (``auto_enable`` is
    True by default) and at least one result carries session info —
    avoids wasting cycles on non-LoCoMo workspaces.
    """
    return SESSION_BOOST_GATE.is_enabled(config, results=results)


def resolve_session_boost_config(config: dict[str, Any] | None) -> dict[str, Any]:
    return SESSION_BOOST_GATE.resolve(config)


__all__ = [
    "SESSION_BOOST_GATE",
    "apply_session_boost",
    "is_session_boost_enabled",
    "resolve_session_boost_config",
]
