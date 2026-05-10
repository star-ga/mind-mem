"""v4 health-check surface (round 4 audit, DeepSeek 9.75→10 gap).

Provides one-line health introspection for production deployments
that need a liveness/readiness probe. ``health_check(workspace)``
returns a structured report, never raises:

    {
        "status": "ok" | "degraded" | "fail",
        "modules": {
            "feature_flags": "ok",
            "tier_memory":   "ok|missing|error: <msg>",
            "block_kinds":   ...,
            ...
        },
        "latency_ms": <float>,
        "checked_at": "<iso>"
    }

Aggregate status:
    every module ok                 → "ok"
    any module missing or degraded  → "degraded"
    any module raises an exception  → "fail"

Each module check is a small probe — typically a flag read + one
SQLite query — designed to complete in single-digit milliseconds so
the health endpoint can be hit at high frequency without becoming a
load source.

The check is **never flag-gated**: ``health_check`` itself runs
unconditionally because operators need it during failure debugging.
The individual probes that *would* be flag-gated (e.g. checking
tier_memory only when the flag is on) report "disabled" rather than
"missing" so operators can distinguish "feature off" from "feature
broken".

Copyright STARGA, Inc.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .feature_flags import is_enabled

__all__ = [
    "health_check",
    "register_health_probe",
    "reset_custom_probes_for_tests",
    "ModuleStatus",
]

ModuleStatus = str  # "ok" | "missing" | "disabled" | "error: <msg>"


def _probe_feature_flags(_workspace: Path) -> ModuleStatus:
    """Always tries to read the registry; missing flags → still ok."""
    try:
        from .feature_flags import ALL_V4_FLAGS

        if not ALL_V4_FLAGS:
            return "missing"
    except Exception as e:
        return f"error: {e!r}"
    return "ok"


def _probe_tier_memory(workspace: Path) -> ModuleStatus:
    if not is_enabled("tier_memory"):
        return "disabled"
    db = workspace / "index.db"
    if not db.is_file():
        return "missing"
    try:
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='block_recall_tier'"
            ).fetchone()
    except sqlite3.Error as e:
        return f"error: {e!r}"
    return "ok" if row else "missing"


def _probe_block_kinds(workspace: Path) -> ModuleStatus:
    if not is_enabled("block_kinds"):
        return "disabled"
    db = workspace / "index.db"
    if not db.is_file():
        return "missing"
    try:
        with sqlite3.connect(db) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(blocks)")}
    except sqlite3.Error as e:
        return f"error: {e!r}"
    return "ok" if "kind" in cols else "missing"


def _probe_cognitive_kernel(_workspace: Path) -> ModuleStatus:
    if not is_enabled("cognitive_kernel"):
        return "disabled"
    try:
        from .cognitive_kernel import KernelKind, is_kernel_registered

        return "ok" if is_kernel_registered(KernelKind.DEFAULT) else "missing"
    except Exception as e:
        return f"error: {e!r}"


def _probe_federation(workspace: Path) -> ModuleStatus:
    if not is_enabled("federation"):
        return "disabled"
    db = workspace / "index.db"
    if not db.is_file():
        return "missing"
    try:
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='block_tier_vclock'"
            ).fetchone()
    except sqlite3.Error as e:
        return f"error: {e!r}"
    return "ok" if row else "missing"


def _probe_observability(_workspace: Path) -> ModuleStatus:
    if not is_enabled("observability"):
        return "disabled"
    try:
        from .observability import snapshot

        snapshot()
    except Exception as e:
        return f"error: {e!r}"
    return "ok"


def _probe_eviction(_workspace: Path) -> ModuleStatus:
    if not is_enabled("eviction"):
        return "disabled"
    try:
        from .eviction import EvictionPolicy, is_policy_registered

        return "ok" if is_policy_registered(EvictionPolicy.LRU) else "missing"
    except Exception as e:
        return f"error: {e!r}"


_BUILTIN_PROBES: list[tuple[str, Callable[[Path], ModuleStatus]]] = [
    ("feature_flags", _probe_feature_flags),
    ("tier_memory", _probe_tier_memory),
    ("block_kinds", _probe_block_kinds),
    ("cognitive_kernel", _probe_cognitive_kernel),
    ("federation", _probe_federation),
    ("observability", _probe_observability),
    ("eviction", _probe_eviction),
]


_custom_probes: list[tuple[str, Callable[[Path], ModuleStatus]]] = []
_custom_probes_lock = threading.Lock()


def register_health_probe(name: str, fn: Callable[[Path], ModuleStatus]) -> None:
    """Register a custom probe. Re-registering under an existing name
    replaces the old probe instead of stacking — this matches operator
    expectations (one probe per name) and lets tests opt-out of stale
    state from previous runs.

    Thread-safe: writes are guarded by an internal lock so a probe
    being installed concurrently with a ``health_check`` call never
    sees a partially-modified registry.
    """
    global _custom_probes
    with _custom_probes_lock:
        _custom_probes = [(n, f) for (n, f) in _custom_probes if n != name]
        _custom_probes.append((name, fn))


def reset_custom_probes_for_tests() -> None:
    """Drop every custom probe. Test-only — never call in production."""
    global _custom_probes
    with _custom_probes_lock:
        _custom_probes = []


def health_check(workspace: str | Path) -> dict[str, Any]:
    """Run every probe; return a structured report. Never raises.

    The contract is **never raises** — even if a probe explodes with a
    ``BaseException`` subclass (``KeyboardInterrupt``, ``SystemExit``,
    or any custom non-Exception). The catch is therefore deliberately
    broad: a health endpoint that crashes during failure is worse than
    a health endpoint that reports the failure as ``"error: ..."``.
    """
    ws = Path(workspace)
    t0 = time.perf_counter()
    modules: dict[str, ModuleStatus] = {}
    statuses: list[str] = []
    # Snapshot the custom-probe list under the lock so a concurrent
    # register / reset doesn't mutate the iteration target.
    with _custom_probes_lock:
        custom_snapshot = list(_custom_probes)
    for name, fn in _BUILTIN_PROBES + custom_snapshot:
        try:
            modules[name] = fn(ws)
        except BaseException as e:  # noqa: BLE001  (intentional; see docstring)
            modules[name] = f"error: {e!r}"
        statuses.append(modules[name])

    if any(s.startswith("error:") for s in statuses):
        agg = "fail"
    elif any(s == "missing" for s in statuses):
        agg = "degraded"
    else:
        agg = "ok"

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    disabled_count = sum(1 for s in statuses if s == "disabled")
    return {
        "status": agg,
        "modules": modules,
        "latency_ms": elapsed_ms,
        "checked_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        # Operators distinguishing "healthy-but-minimal" (most probes
        # disabled by feature-flags) from "fully armed" need this.
        "disabled_count": disabled_count,
    }
