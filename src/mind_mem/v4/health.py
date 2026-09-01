"""v4 health-check surface (round 4 audit, DeepSeek 9.75→10 gap).

Provides one-line health introspection for production deployments
that need a liveness/readiness probe. ``health_check(workspace)``
returns a structured report, never raises:

    {
        "status": "ok" | "degraded" | "fail",
        "modules": {
            "feature_flags": "ok",
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
block_kinds only when the flag is on) report "disabled" rather than
"missing" so operators can distinguish "feature off" from "feature
broken".

Copyright STARGA, Inc.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
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
    """Check the registry **and** the config file it is read from.

    Importing the registry proves nothing on its own: ``ALL_V4_FLAGS`` is a
    module-level literal, so an import-plus-emptiness test can only ever say
    "ok". The failure operators actually hit is a ``mind-mem.json`` that does
    not parse — the flag loader swallows that and returns an empty block, so
    every v4 surface goes silently OFF while this probe reports a healthy
    deployment. Resolve the active config and re-read it instead:

        no config file on disk        → "ok"  (plain v3.x deployment)
        unreadable file / invalid JSON → "error: ..."
        root or ``v4`` not an object   → "error: ..."
    """
    try:
        from .feature_flags import ALL_V4_FLAGS, _config_path

        if not ALL_V4_FLAGS:
            return "missing"
        path = _config_path()
        if not path.is_file():
            # Absent config is a supported deployment (all v4 flags OFF),
            # not a fault — the loader's fallback is correct here.
            return "ok"
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as e:
            return f"error: config unreadable at {path}: {e!r}"
        try:
            data = json.loads(raw)
        except ValueError as e:
            return f"error: config is not valid JSON at {path}: {e!r}"
        if not isinstance(data, dict):
            return f"error: config root at {path} is {type(data).__name__}, expected object"
        v4 = data.get("v4")
        if v4 is not None and not isinstance(v4, dict):
            return f"error: config 'v4' block at {path} is {type(v4).__name__}, expected object"
    except Exception as e:
        return f"error: {e!r}"
    return "ok"


def _probe_block_kinds(workspace: Path) -> ModuleStatus:
    if not is_enabled("block_kinds"):
        return "disabled"
    db = workspace / "index.db"
    if not db.is_file():
        return "missing"
    try:
        # closing(), not ``with sqlite3.connect(...)``: the connection context
        # manager commits or rolls back and then leaves the handle open, and the
        # connection's prepared-statement cache references it back, so
        # refcounting never reclaims it. A liveness probe is polled — a handle
        # per poll is an unbounded descriptor leak in exactly the process an
        # operator is watching. Read-only, so there is nothing to commit.
        with contextlib.closing(sqlite3.connect(db, timeout=30)) as conn:
            # The set comprehension drains the cursor inside the block.
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
        # Read-only, and closed rather than merely committed — see
        # _probe_block_kinds for why the bare connection context manager is
        # not enough.
        with contextlib.closing(sqlite3.connect(db, timeout=30)) as conn:
            row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='block_tier_vclock'").fetchone()
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


_BUILTIN_PROBES: list[tuple[str, Callable[[Path], ModuleStatus]]] = [
    ("feature_flags", _probe_feature_flags),
    ("block_kinds", _probe_block_kinds),
    ("cognitive_kernel", _probe_cognitive_kernel),
    ("federation", _probe_federation),
    ("observability", _probe_observability),
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

    Rejects a non-string name or a non-callable probe at registration
    time: ``health_check`` must not be the place a misconfigured probe
    is discovered, because by then it is already inside the endpoint
    that promises never to raise.
    """
    global _custom_probes
    if not isinstance(name, str) or not name:
        raise TypeError(f"health probe name must be a non-empty str, got {type(name).__name__}")
    if not callable(fn):
        raise TypeError(f"health probe {name!r} must be callable, got {type(fn).__name__}")
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
            status = fn(ws)
        except BaseException as e:  # noqa: BLE001  (intentional; see docstring)
            status = f"error: {e!r}"
        if not isinstance(status, str):
            # A probe is only contractually a ``ModuleStatus`` — an alias for
            # ``str`` with no runtime enforcement. Coerce anything else here:
            # the aggregation below calls ``str.startswith``, and letting that
            # raise would break the never-raises contract on exactly the
            # custom-probe misconfiguration this endpoint exists to report.
            status = f"error: probe {name!r} returned {type(status).__name__}, expected a status str"
        modules[name] = status
        statuses.append(status)

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
