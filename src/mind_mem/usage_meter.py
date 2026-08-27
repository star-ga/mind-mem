# Copyright 2026 STARGA, Inc.
"""Local token counter for mind-mem's only real variable cost: model calls.

mind-mem is self-hosted and single-operator. Retrieval, indexing and the
governance gate run on the operator's own machine, so counting and pricing
them is theatre. The one thing that actually costs money is a call out to a
model — today that is the injected compressor behind recompaction, and the
optional extraction backend. This module counts **tokens** for those calls,
per UTC day, and offers one optional **daily token cap**.

What it deliberately is not: no currency, no rate card, no spending alerts,
no quota subsystem. A token count and a ceiling.

Hard guarantee: **nothing leaves the host.** No socket, no HTTP client, no
exporter, and no import of one anywhere in this module. The ledger is a
plain local JSON file; the cap comes from ``mind-mem.json`` or the CLI.
Egress-free operation is asserted by ``tests/test_usage_meter.py``.

Determinism: the ledger is keyed by UTC day, and every entry point takes an
explicit ``day`` so a caller (and every test) can pin it. Nothing here feeds
retrieval or scoring.

Usage::

    from mind_mem import usage_meter
    from mind_mem.recompaction import recompact_cluster

    compressor = usage_meter.metered_compressor(my_compressor, workspace)
    recompact_cluster(blocks, compressor=compressor)   # counts as it goes

    print(usage_meter.format_report(usage_meter.report(workspace)))
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Optional

from .cognitive_forget import estimate_tokens
from .observability import get_logger

_log = get_logger("usage_meter")

LEDGER_REL_PATH = os.path.join(".mind-mem-index", "usage.json")
LEDGER_VERSION = 2

#: Exit status used when the daily token cap is reached or exceeded.
CAP_EXIT_CODE = 3

#: Days retained in the ledger. Older days are pruned on write so the file
#: stays a few kilobytes forever.
RETENTION_DAYS = 90

#: Operation tag for the recompaction compressor path.
OP_RECOMPACTION = "recompaction"

_DAY_FORMAT = "%Y-%m-%d"
_MAX_OPERATION_LEN = 64
_LOCK = threading.Lock()


class DailyTokenCapExceeded(RuntimeError):
    """A metered model call was refused: the day's token cap is used up."""


# ---------------------------------------------------------------------------
# Boundary validation
# ---------------------------------------------------------------------------


def _require_workspace(workspace: str) -> str:
    if not isinstance(workspace, str) or not workspace.strip():
        raise ValueError("workspace must be a non-empty path string")
    return os.path.realpath(workspace)


def _require_day(day: Optional[str]) -> str:
    """Validate ``YYYY-MM-DD``; ``None`` means 'today, UTC'."""
    if day is None:
        return datetime.now(timezone.utc).strftime(_DAY_FORMAT)
    if not isinstance(day, str):
        raise ValueError("day must be a YYYY-MM-DD string")
    try:
        datetime.strptime(day, _DAY_FORMAT)
    except ValueError as exc:
        raise ValueError(f"day must be YYYY-MM-DD, got {day!r}") from exc
    return day


def _require_tokens(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _require_cap(cap: Optional[int]) -> Optional[int]:
    if cap is None:
        return None
    return _require_tokens(cap, "daily_token_cap")


def _require_operation(operation: str) -> str:
    if not isinstance(operation, str) or not operation.strip():
        raise ValueError("operation must be a non-empty string")
    text = operation.strip()
    if len(text) > _MAX_OPERATION_LEN:
        raise ValueError(f"operation must be at most {_MAX_OPERATION_LEN} characters")
    return text


def _clean_int_map(raw: Any) -> dict[str, int]:
    """Keep only ``str -> non-negative int`` pairs from untrusted JSON."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or isinstance(value, bool):
            continue
        if not isinstance(value, int) or value < 0:
            continue
        out[key] = value
    return out


# ---------------------------------------------------------------------------
# Value objects (immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DayUsage:
    """One UTC day of counted model-call tokens."""

    day: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    operations: Mapping[str, int]

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "operations": dict(self.operations),
        }


@dataclass(frozen=True)
class TokenReport:
    """Immutable view of one workspace's token ledger for a given day."""

    workspace: str
    day: str
    days: Mapping[str, DayUsage]
    today_tokens: int
    today_calls: int
    total_tokens: int
    daily_cap: Optional[int] = None
    cap_exceeded: bool = False
    ledger_error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace,
            "day": self.day,
            "days": {d: u.as_dict() for d, u in sorted(self.days.items())},
            "today_tokens": self.today_tokens,
            "today_calls": self.today_calls,
            "total_tokens": self.total_tokens,
            "daily_token_cap": self.daily_cap,
            "cap_exceeded": self.cap_exceeded,
            "ledger_error": self.ledger_error,
            "egress": "none",
        }


# ---------------------------------------------------------------------------
# Ledger I/O (local file only)
# ---------------------------------------------------------------------------


def ledger_path(workspace: str) -> str:
    """Absolute path of the workspace token ledger."""
    return os.path.join(_require_workspace(workspace), LEDGER_REL_PATH)


def _parse_days(raw: Any) -> dict[str, DayUsage]:
    if not isinstance(raw, dict):
        return {}
    days: dict[str, DayUsage] = {}
    for day, entry in raw.items():
        if not isinstance(day, str) or not isinstance(entry, dict):
            continue
        try:
            datetime.strptime(day, _DAY_FORMAT)
        except ValueError:
            continue
        counts = _clean_int_map(entry)
        days[day] = DayUsage(
            day=day,
            calls=counts.get("calls", 0),
            prompt_tokens=counts.get("prompt_tokens", 0),
            completion_tokens=counts.get("completion_tokens", 0),
            operations=MappingProxyType(_clean_int_map(entry.get("operations"))),
        )
    return days


def load_ledger(workspace: str) -> tuple[dict[str, DayUsage], Optional[str]]:
    """Read the ledger. Returns ``(days, error)``.

    A missing, corrupt or foreign-version ledger yields an empty result plus a
    short error tag rather than raising — a read-only token report must never
    be the thing that breaks a session.
    """
    path = ledger_path(workspace)
    if not os.path.isfile(path):
        return {}, None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        _log.warning("usage_ledger_unreadable", error=type(exc).__name__)
        return {}, type(exc).__name__
    if not isinstance(data, dict):
        return {}, "MalformedLedger"
    if data.get("version") != LEDGER_VERSION:
        return {}, "UnsupportedLedgerVersion"
    return _parse_days(data.get("days")), None


def _write_ledger(workspace: str, days: Mapping[str, DayUsage]) -> str:
    """Atomically persist the ledger, pruned to :data:`RETENTION_DAYS`."""
    path = ledger_path(workspace)
    kept = dict(sorted(days.items())[-RETENTION_DAYS:])
    payload = {"version": LEDGER_VERSION, "days": {d: u.as_dict() for d, u in kept.items()}}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".usage-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


# ---------------------------------------------------------------------------
# Cap configuration
# ---------------------------------------------------------------------------


def load_daily_cap(workspace: str) -> Optional[int]:
    """Read ``mind-mem.json`` -> ``{"usage": {"daily_token_cap": N}}``.

    Returns ``None`` when unset or unreadable: no cap is the default.
    """
    cfg_path = os.path.join(_require_workspace(workspace), "mind-mem.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError) as exc:
        _log.warning("usage_config_unreadable", error=type(exc).__name__)
        return None
    section = cfg.get("usage") if isinstance(cfg, dict) else None
    raw = section.get("daily_token_cap") if isinstance(section, dict) else None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return None
    return raw


# ---------------------------------------------------------------------------
# Recording + reporting
# ---------------------------------------------------------------------------


def _build_report(
    workspace: str,
    days: Mapping[str, DayUsage],
    *,
    day: str,
    daily_cap: Optional[int],
    ledger_error: Optional[str],
) -> TokenReport:
    today = days.get(day)
    today_tokens = today.total_tokens if today else 0
    return TokenReport(
        workspace=workspace,
        day=day,
        days=MappingProxyType(dict(sorted(days.items()))),
        today_tokens=today_tokens,
        today_calls=today.calls if today else 0,
        total_tokens=sum(u.total_tokens for u in days.values()),
        daily_cap=daily_cap,
        cap_exceeded=daily_cap is not None and today_tokens >= daily_cap,
        ledger_error=ledger_error,
    )


def record_call(
    workspace: str,
    *,
    operation: str,
    prompt_tokens: int,
    completion_tokens: int,
    day: Optional[str] = None,
) -> TokenReport:
    """Count one model call into the workspace ledger and return the report."""
    ws = _require_workspace(workspace)
    op = _require_operation(operation)
    prompt = _require_tokens(prompt_tokens, "prompt_tokens")
    completion = _require_tokens(completion_tokens, "completion_tokens")
    key = _require_day(day)

    with _LOCK:
        days, err = load_ledger(ws)
        prev = days.get(key)
        operations = dict(prev.operations) if prev else {}
        operations[op] = operations.get(op, 0) + prompt + completion
        days = {
            **days,
            key: DayUsage(
                day=key,
                calls=(prev.calls if prev else 0) + 1,
                prompt_tokens=(prev.prompt_tokens if prev else 0) + prompt,
                completion_tokens=(prev.completion_tokens if prev else 0) + completion,
                operations=MappingProxyType(operations),
            ),
        }
        _write_ledger(ws, days)

    return _build_report(ws, days, day=key, daily_cap=load_daily_cap(ws), ledger_error=err)


def report(workspace: str, *, daily_cap: Optional[int] = None, day: Optional[str] = None) -> TokenReport:
    """Read-only token report. ``daily_cap=None`` falls back to the config cap."""
    ws = _require_workspace(workspace)
    key = _require_day(day)
    cap = load_daily_cap(ws) if daily_cap is None else _require_cap(daily_cap)
    days, err = load_ledger(ws)
    return _build_report(ws, days, day=key, daily_cap=cap, ledger_error=err)


def reset(workspace: str, *, day: Optional[str] = None) -> TokenReport:
    """Clear the ledger; returns the report as it stood before clearing."""
    ws = _require_workspace(workspace)
    key = _require_day(day)
    with _LOCK:
        days, err = load_ledger(ws)
        before = _build_report(ws, days, day=key, daily_cap=None, ledger_error=err)
        _write_ledger(ws, {})
    return before


def cap_line(r: TokenReport) -> str:
    """One-line, machine-greppable daily-cap report (stderr form)."""
    return f"DAILY TOKEN CAP: workspace={r.workspace} day={r.day} tokens={r.today_tokens} cap={r.daily_cap} calls={r.today_calls}"


def format_report(r: TokenReport) -> str:
    """Human-readable token report (stdout form of ``mm usage``)."""
    lines = [
        f"mind-mem model-call tokens — {r.workspace}",
        "  egress : none (counts are local; model calls are the only metered cost)",
        "",
        f"  {'day':<14}{'calls':>8}{'prompt':>12}{'completion':>14}{'total':>12}",
        f"  {'-' * 60}",
    ]
    for day, usage in r.days.items():
        lines.append(f"  {day:<14}{usage.calls:>8}{usage.prompt_tokens:>12}{usage.completion_tokens:>14}{usage.total_tokens:>12}")
    lines.append(f"  {'-' * 60}")
    lines.append(f"  {'TOTAL':<14}{'':>8}{'':>12}{'':>14}{r.total_tokens:>12}")
    cap_text = "none" if r.daily_cap is None else str(r.daily_cap)
    lines.append(f"  today ({r.day}) : {r.today_tokens} tokens in {r.today_calls} calls    cap: {cap_text}")
    if r.cap_exceeded:
        lines.append("  daily token cap reached — metered model calls are refused for the rest of the day")
    if r.ledger_error:
        lines.append(f"  note: ledger unusable ({r.ledger_error}) — reported as empty")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metered model call (opt-in by construction — nothing wraps itself)
# ---------------------------------------------------------------------------


def metered_compressor(
    compressor: Callable[[str, list[dict[str, Any]]], str],
    workspace: str,
    *,
    operation: str = OP_RECOMPACTION,
    daily_cap: Optional[int] = None,
    day: Optional[str] = None,
) -> Callable[[str, list[dict[str, Any]]], str]:
    """Wrap an injected ``Compressor`` so its token cost is counted locally.

    The wrapper is transparent: it returns exactly the bytes the wrapped
    compressor returned, so a recompaction fixed point is unchanged by
    metering. Nothing meters itself — a caller that never wraps writes no
    ledger, which is the default-OFF proof in ``tests/test_usage_meter.py``.

    With a cap in force (argument, else ``mind-mem.json``), a call made once
    the day's counted tokens have reached it raises
    :class:`DailyTokenCapExceeded` **before** the model is called.

    Token counts are estimates from the text handed to and returned by the
    compressor (~4 chars/token, the same estimator the context packer uses).
    # deferred: provider-reported token counts are not available through the
    # injected-callable contract - upgrade path: let a compressor optionally
    # return (text, prompt_tokens, completion_tokens) and prefer those.
    """
    if not callable(compressor):
        raise ValueError("compressor must be callable")
    ws = _require_workspace(workspace)
    op = _require_operation(operation)
    cap = _require_cap(daily_cap)
    pinned_day = day if day is None else _require_day(day)

    from .recompaction import _block_body  # reuse: the exact text the compressor is handed

    def _metered(current_text: str, blocks: list[dict[str, Any]]) -> str:
        key = _require_day(pinned_day)
        effective_cap = load_daily_cap(ws) if cap is None else cap
        before = report(ws, daily_cap=effective_cap, day=key)
        if before.cap_exceeded:
            raise DailyTokenCapExceeded(cap_line(before))

        result = compressor(current_text, blocks)

        prompt = estimate_tokens(current_text) + sum(estimate_tokens(_block_body(b)) for b in blocks)
        completion = estimate_tokens(result) if isinstance(result, str) else 0
        record_call(ws, operation=op, prompt_tokens=prompt, completion_tokens=completion, day=key)
        return result

    return _metered


__all__ = [
    "CAP_EXIT_CODE",
    "DailyTokenCapExceeded",
    "DayUsage",
    "TokenReport",
    "cap_line",
    "format_report",
    "ledger_path",
    "load_daily_cap",
    "load_ledger",
    "metered_compressor",
    "record_call",
    "report",
    "reset",
]
