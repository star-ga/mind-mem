"""Temporal metadata injection for retrieved blocks (v3.4.0).

LoCoMo temporal questions collapse on the full bench (regression to
~39 across all convs) because the answerer LLM has no absolute
chronological anchor — retrieved blocks have dates in metadata, but
the answerer sees only the excerpt text.

This module prepends a compact ``[Stored N days ago • 2026-04-15]``
tag to each block's excerpt before evidence packing, giving the
answerer an explicit timeline reference for every piece of evidence.

The anchor date (``now``) defaults to the system clock but can be
overridden for deterministic benches.

Public entry: :func:`annotate_with_temporal_metadata`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .observability import get_logger

_log = get_logger("temporal_metadata")

_DATE_KEYS = ("created_at", "date", "timestamp", "last_seen", "updated_at")


def _parse_dt(raw: Any) -> datetime | None:
    """Parse one of several date shapes into a timezone-aware datetime."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, (int, float)):
        # Assume unix seconds; reject obviously-wrong values.
        if raw < 0 or raw > 4102444800:  # 2100-01-01
            return None
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        # Normalise trailing Z → +00:00 for fromisoformat on Py < 3.11.
        s_norm = s.replace("Z", "+00:00") if s.endswith("Z") else s
        try:
            dt = datetime.fromisoformat(s_norm)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
        # Accept bare YYYY-MM-DD.
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _extract_block_date(block: dict[str, Any]) -> datetime | None:
    """Pull a date from any of the common metadata keys, first hit wins."""
    for key in _DATE_KEYS:
        if key in block:
            dt = _parse_dt(block[key])
            if dt is not None:
                return dt
    # Dates are sometimes nested under ``metadata`` or ``meta``.
    for nest in ("metadata", "meta"):
        nested = block.get(nest)
        if isinstance(nested, dict):
            for key in _DATE_KEYS:
                if key in nested:
                    dt = _parse_dt(nested[key])
                    if dt is not None:
                        return dt
    return None


def annotate_with_temporal_metadata(
    blocks: list[dict[str, Any]],
    now: datetime | None = None,
    excerpt_key: str = "excerpt",
    max_days: int = 3650,  # ~10 years — tight default per security audit
) -> list[dict[str, Any]]:
    """Return copies of ``blocks`` with ``[Stored N days ago • YYYY-MM-DD]``
    prefixed to each excerpt. Blocks without a parseable date are left
    unchanged.

    Args:
        blocks: retrieval hits.
        now: reference "now" datetime (UTC preferred). Defaults to
            ``datetime.now(timezone.utc)``.
        excerpt_key: field name holding the text to prefix. Also tries
            ``Statement`` as a legacy fallback.
        max_days: clamp extreme ages (corrupt metadata) to prevent the
            tag from dominating the excerpt.

    Returns:
        A new list of dict copies. The original ``blocks`` are not
        mutated.
    """
    if not blocks:
        return []
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)

    out: list[dict[str, Any]] = []
    annotated = 0
    for b in blocks:
        dt = _extract_block_date(b)
        if dt is None:
            out.append(dict(b))
            continue
        delta = (ref - dt).days
        if delta < 0 or delta > max_days:
            # Out-of-bounds date — strip the field in the returned copy
            # so downstream consumers can't act on a tampered anchor.
            cp = dict(b)
            for k in _DATE_KEYS:
                cp.pop(k, None)
            if isinstance(cp.get("metadata"), dict):
                cp["metadata"] = {k: v for k, v in cp["metadata"].items() if k not in _DATE_KEYS}
            if isinstance(cp.get("meta"), dict):
                cp["meta"] = {k: v for k, v in cp["meta"].items() if k not in _DATE_KEYS}
            cp["_temporal_date_rejected"] = True
            out.append(cp)
            continue
        tag = f"[Stored {delta} days ago • {dt.date().isoformat()}] "
        cp = dict(b)
        # Target the existing text field rather than creating a new
        # ``excerpt`` when only ``Statement`` is present — prevents
        # schema drift for downstream consumers (Gemini audit 2026-04-22).
        if cp.get(excerpt_key):
            target_key = excerpt_key
        elif cp.get("Statement"):
            target_key = "Statement"
        else:
            target_key = None
        if target_key is not None:
            cp[target_key] = tag + str(cp[target_key])
            cp.setdefault("_temporal_delta_days", delta)
            annotated += 1
        out.append(cp)

    _log.info("temporal_annotate", n_blocks=len(blocks), annotated=annotated)
    return out


__all__ = ["annotate_with_temporal_metadata"]
