# Copyright 2026 STARGA, Inc.
"""Per-workspace usage + cost rollup over the counters mind-mem already keeps.

This module adds **no** telemetry of its own. It reads the existing
in-process counters from :mod:`mind_mem.observability` (``metrics``),
folds them into a small per-workspace JSON ledger under
``<workspace>/.mind-mem-index/usage.json``, and prices the result against
a local, operator-editable rate card.

Hard guarantee: **nothing leaves the host.** There is no socket, no HTTP
client, no exporter and no import of one anywhere in this module or its
call graph. The ledger is a plain local file; the rate card is either the
in-repo default or ``mind-mem.json``. Egress-free operation is part of
the contract and is asserted by ``tests/test_usage_meter.py``.

Usage::

    from mind_mem import usage_meter

    usage_meter.record(workspace)                    # fold this process in
    r = usage_meter.rollup(workspace, quota_usd=1.0) # price + check quota
    print(usage_meter.format_report(r))
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Optional

from .observability import get_logger, metrics

_log = get_logger("usage_meter")

LEDGER_REL_PATH = os.path.join(".mind-mem-index", "usage.json")
LEDGER_VERSION = 1

#: Exit status used when a ``--quota`` threshold is breached.
QUOTA_EXIT_CODE = 3

#: Opt-in switch for the ``mm`` end-of-command flush. Default OFF: with the
#: variable unset mind-mem writes no ledger and behaves byte-identically to
#: a build without this module.
ENV_ENABLE = "MIND_MEM_USAGE_METER"

#: Local rate card: USD per counted operation. These are nominal
#: compute-equivalent rates, not a measured price list — override them per
#: workspace via ``mind-mem.json`` -> ``{"usage": {"unit_costs": {...}}}``.
# deferred: rates are nominal, not measured — upgrade path: derive them from
# the observability latency observations (``*_ms``) times a machine-hour rate.
DEFAULT_UNIT_COSTS: Mapping[str, float] = MappingProxyType(
    {
        "recall_queries": 0.000020,
        "vector_searches": 0.000050,
        "embeddings_generated": 0.000010,
        "index_builds": 0.000500,
        "index_blocks_indexed": 0.000002,
        "index_queries": 0.000005,
        "contradiction_checks": 0.000030,
        "signals_written": 0.000005,
        "summaries_written": 0.000200,
        "dream_cycle_runs": 0.002000,
        "mcp_proposals": 0.000100,
        "mcp_apply_calls": 0.000100,
        "mcp_recall_queries": 0.000020,
    }
)

_LOCK = threading.Lock()
# High-water mark of counters already folded into a ledger by THIS process.
# ``metrics`` counters are cumulative, so record() must post the delta or a
# second call would double-count.
_RECORDED: Mapping[str, float] = MappingProxyType({})


# ---------------------------------------------------------------------------
# Boundary validation
# ---------------------------------------------------------------------------


def _require_workspace(workspace: str) -> str:
    if not isinstance(workspace, str) or not workspace.strip():
        raise ValueError("workspace must be a non-empty path string")
    return os.path.realpath(workspace)


def _require_quota(quota_usd: Optional[float]) -> Optional[float]:
    if quota_usd is None:
        return None
    if isinstance(quota_usd, bool) or not isinstance(quota_usd, (int, float)):
        raise ValueError("quota_usd must be a number")
    value = float(quota_usd)
    if not math.isfinite(value) or value < 0:
        raise ValueError("quota_usd must be a finite, non-negative number")
    return value


def _clean_counters(raw: Any) -> dict[str, float]:
    """Keep only ``str -> finite non-negative number`` pairs."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        if value < 0:
            continue
        out[key] = float(value)
    return out


# ---------------------------------------------------------------------------
# Rollup value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UsageRollup:
    """Immutable priced view of one workspace's usage ledger."""

    workspace: str
    counters: Mapping[str, float]
    costs: Mapping[str, float]
    total_operations: float
    total_cost_usd: float
    sessions: int
    first_recorded: Optional[str]
    last_recorded: Optional[str]
    quota_usd: Optional[float] = None
    quota_breached: bool = False
    ledger_error: Optional[str] = None
    generated_at: str = field(default_factory=lambda: _utc_now())

    def as_dict(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace,
            "counters": dict(self.counters),
            "costs_usd": dict(self.costs),
            "total_operations": self.total_operations,
            "total_cost_usd": self.total_cost_usd,
            "sessions": self.sessions,
            "first_recorded": self.first_recorded,
            "last_recorded": self.last_recorded,
            "quota_usd": self.quota_usd,
            "quota_breached": self.quota_breached,
            "ledger_error": self.ledger_error,
            "generated_at": self.generated_at,
            "egress": "none",
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Ledger I/O (local file only)
# ---------------------------------------------------------------------------


def ledger_path(workspace: str) -> str:
    """Absolute path of the workspace usage ledger."""
    return os.path.join(_require_workspace(workspace), LEDGER_REL_PATH)


def load_ledger(workspace: str) -> tuple[dict[str, Any], Optional[str]]:
    """Read the ledger. Returns ``(ledger, error)``; a corrupt or unreadable
    ledger yields an empty ledger plus a short error tag rather than raising —
    a read-only usage report must never be the thing that breaks a session."""
    path = ledger_path(workspace)
    empty: dict[str, Any] = {"version": LEDGER_VERSION, "counters": {}, "sessions": 0}
    if not os.path.isfile(path):
        return empty, None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        _log.warning("usage_ledger_unreadable", error=type(exc).__name__)
        return empty, type(exc).__name__
    if not isinstance(data, dict):
        return empty, "MalformedLedger"
    sessions = data.get("sessions")
    return (
        {
            "version": LEDGER_VERSION,
            "counters": _clean_counters(data.get("counters")),
            "sessions": int(sessions) if isinstance(sessions, int) and sessions >= 0 else 0,
            "first_recorded": data.get("first_recorded"),
            "last_recorded": data.get("last_recorded"),
        },
        None,
    )


def _write_ledger(workspace: str, ledger: Mapping[str, Any]) -> str:
    path = ledger_path(workspace)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".usage-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(dict(ledger), fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def snapshot_process_counters(source: Any = None) -> dict[str, float]:
    """Snapshot the existing in-process counters (``observability.metrics``)."""
    src = metrics if source is None else source
    summary = src.summary()
    return _clean_counters(summary.get("counters") if isinstance(summary, dict) else None)


def reset_process_high_water() -> None:
    """Forget what this process has already folded into a ledger (tests)."""
    global _RECORDED
    with _LOCK:
        _RECORDED = MappingProxyType({})


def record(workspace: str, counters: Optional[Mapping[str, float]] = None, *, source: Any = None) -> UsageRollup:
    """Fold counters into the workspace ledger and return the new rollup.

    ``counters`` defaults to the delta of the in-process ``metrics`` counters
    since this process last recorded, so repeated calls never double-count.
    """
    global _RECORDED
    ws = _require_workspace(workspace)

    with _LOCK:
        if counters is None:
            current = snapshot_process_counters(source)
            delta = {k: v - _RECORDED.get(k, 0.0) for k, v in current.items() if v - _RECORDED.get(k, 0.0) > 0}
            new_high_water: Mapping[str, float] = MappingProxyType(dict(current))
        else:
            delta = _clean_counters(dict(counters))
            new_high_water = _RECORDED

        ledger, err = load_ledger(ws)
        merged = dict(ledger["counters"])
        for key, value in delta.items():
            merged[key] = merged.get(key, 0.0) + value

        now = _utc_now()
        updated = {
            "version": LEDGER_VERSION,
            "counters": merged,
            "sessions": int(ledger["sessions"]) + (1 if delta else 0),
            "first_recorded": ledger.get("first_recorded") or now,
            "last_recorded": now if delta else ledger.get("last_recorded"),
        }
        # Only touch the file when there is something new, or on first run —
        # a read-only command with an enabled meter must not churn the ledger.
        if delta or not os.path.isfile(ledger_path(ws)):
            _write_ledger(ws, updated)
        _RECORDED = new_high_water

    return _build_rollup(ws, updated, quota_usd=None, unit_costs=load_unit_costs(ws), ledger_error=err)


def reset(workspace: str) -> UsageRollup:
    """Clear the ledger; returns the rollup as it stood before clearing."""
    ws = _require_workspace(workspace)
    with _LOCK:
        ledger, err = load_ledger(ws)
        before = _build_rollup(ws, ledger, quota_usd=None, unit_costs=load_unit_costs(ws), ledger_error=err)
        _write_ledger(ws, {"version": LEDGER_VERSION, "counters": {}, "sessions": 0, "first_recorded": None, "last_recorded": None})
    return before


# ---------------------------------------------------------------------------
# Pricing + rollup
# ---------------------------------------------------------------------------


def load_unit_costs(workspace: str) -> Mapping[str, float]:
    """Default rate card overlaid with ``mind-mem.json`` -> ``usage.unit_costs``."""
    ws = _require_workspace(workspace)
    rates = dict(DEFAULT_UNIT_COSTS)
    cfg_path = os.path.join(ws, "mind-mem.json")
    if not os.path.isfile(cfg_path):
        return MappingProxyType(rates)
    try:
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError) as exc:
        _log.warning("usage_rate_card_unreadable", error=type(exc).__name__)
        return MappingProxyType(rates)
    section = cfg.get("usage") if isinstance(cfg, dict) else None
    override = section.get("unit_costs") if isinstance(section, dict) else None
    rates.update(_clean_counters(override))
    return MappingProxyType(rates)


def price(counters: Mapping[str, float], unit_costs: Mapping[str, float]) -> tuple[float, dict[str, float]]:
    """Return ``(total_usd, per_counter_usd)``. Unpriced counters cost 0."""
    per: dict[str, float] = {}
    for key, count in _clean_counters(dict(counters)).items():
        rate = unit_costs.get(key)
        if rate is None:
            continue
        per[key] = round(count * float(rate), 8)
    return round(sum(per.values()), 8), per


def _build_rollup(
    workspace: str,
    ledger: Mapping[str, Any],
    *,
    quota_usd: Optional[float],
    unit_costs: Mapping[str, float],
    ledger_error: Optional[str],
) -> UsageRollup:
    counters = _clean_counters(dict(ledger.get("counters") or {}))
    total_cost, costs = price(counters, unit_costs)
    return UsageRollup(
        workspace=workspace,
        counters=MappingProxyType(dict(sorted(counters.items()))),
        costs=MappingProxyType(dict(sorted(costs.items()))),
        total_operations=round(sum(counters.values()), 6),
        total_cost_usd=total_cost,
        sessions=int(ledger.get("sessions") or 0),
        first_recorded=ledger.get("first_recorded"),
        last_recorded=ledger.get("last_recorded"),
        quota_usd=quota_usd,
        quota_breached=quota_usd is not None and total_cost > quota_usd,
        ledger_error=ledger_error,
    )


def rollup(
    workspace: str,
    *,
    quota_usd: Optional[float] = None,
    unit_costs: Optional[Mapping[str, float]] = None,
    include_process: bool = False,
) -> UsageRollup:
    """Price the workspace ledger, optionally against a ``quota_usd`` threshold.

    ``include_process=True`` adds this process's not-yet-recorded counters to
    the view without writing anything.
    """
    ws = _require_workspace(workspace)
    quota = _require_quota(quota_usd)
    rates = load_unit_costs(ws) if unit_costs is None else MappingProxyType(_clean_counters(dict(unit_costs)))

    ledger, err = load_ledger(ws)
    if include_process:
        merged = dict(ledger["counters"])
        for key, value in snapshot_process_counters().items():
            pending = value - _RECORDED.get(key, 0.0)
            if pending > 0:
                merged[key] = merged.get(key, 0.0) + pending
        ledger = {**ledger, "counters": merged}
    return _build_rollup(ws, ledger, quota_usd=quota, unit_costs=rates, ledger_error=err)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def quota_alert_line(r: UsageRollup) -> str:
    """One-line, machine-greppable quota breach alert."""
    over = round(r.total_cost_usd - float(r.quota_usd or 0.0), 8)
    return (
        f"QUOTA BREACH: workspace={r.workspace} cost=${r.total_cost_usd:.6f} "
        f"quota=${float(r.quota_usd or 0.0):.6f} over=${over:.6f} "
        f"operations={r.total_operations:g}"
    )


def format_report(r: UsageRollup) -> str:
    """Human-readable rollup (stdout form of ``mm usage``)."""
    lines = [
        f"mind-mem usage — {r.workspace}",
        f"  sessions : {r.sessions}    window : {r.first_recorded or '-'} → {r.last_recorded or '-'}",
        "  egress   : none (all counters are local)",
        "",
        f"  {'operation':<34}{'count':>12}{'cost (USD)':>16}",
        f"  {'-' * 62}",
    ]
    for name, count in r.counters.items():
        cost = r.costs.get(name)
        cost_text = f"{cost:.6f}" if cost is not None else "unpriced"
        lines.append(f"  {name:<34}{count:>12g}{cost_text:>16}")
    lines.append(f"  {'-' * 62}")
    lines.append(f"  {'TOTAL':<34}{r.total_operations:>12g}{r.total_cost_usd:>16.6f}")
    if r.quota_usd is not None:
        lines.append(f"  {'QUOTA':<34}{'':>12}{r.quota_usd:>16.6f}")
    if r.ledger_error:
        lines.append(f"  note: ledger unreadable ({r.ledger_error}) — reported as empty")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Opt-in end-of-command flush (default OFF)
# ---------------------------------------------------------------------------


def meter_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """True only when ``MIND_MEM_USAGE_METER`` is explicitly truthy."""
    source = os.environ if env is None else env
    return str(source.get(ENV_ENABLE, "")).strip().lower() in {"1", "true", "yes", "on"}


def flush_if_enabled(workspace: str) -> bool:
    """Fold this process's counters into the ledger iff the meter is enabled.

    Best-effort: a metering failure must never change the exit status of the
    command the operator actually ran.
    """
    if not meter_enabled():
        return False
    try:
        record(workspace)
        return True
    except Exception as exc:  # pragma: no cover — defensive
        _log.warning("usage_flush_failed", error=type(exc).__name__)
        return False
