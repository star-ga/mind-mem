"""v4 backpressure controller (round 4 audit, DeepSeek 9.75→10 gap).

When the embedding pipeline or consolidation worker can't keep up
with incoming work, callers need an explicit signal that says "back
off". Without it the queue grows unbounded and OOM kills the process.

This module ships a thread-safe :class:`BackpressureController` with:

    queue depth tracking      callers tell the controller how deep
                              their backlog is via ``set_depth(n)``.
    threshold gates           ``high_watermark`` triggers overload;
                              ``low_watermark`` triggers recovery.
                              Hysteresis prevents flapping.
    adaptive sleep            when overloaded, ``recommended_pause()``
                              returns an exponential-backoff hint
                              (capped at ``max_pause_seconds``).
    is_overloaded()           one-line caller check before submitting
                              work. Always safe to call (no flag
                              required for the read).

Feature-flag gated under ``v4.backpressure``.

Producer wiring (5.1.0)
-----------------------
A controller nothing reports into is a thermometer in a drawer. The
bottom half of this module is the seam the real producer loops use:

    :func:`report_depth`      producer says how deep its backlog is and
                              learns, in the same call, whether it is
                              overloaded.
    :func:`producer_overloaded` / :func:`any_overloaded`
                              read-only checks for a loop deciding
                              whether to add more work.
    :func:`batch_limit`       per-tick burst cap while overloaded.
    :func:`snapshot`          per-producer state for ``stream_status``.

Each producer gets its OWN controller, keyed by name — a single shared
``set_depth`` is last-writer-wins, so the inbox backlog would silently
overwrite the change-stream queue depth and both signals would be
noise. :func:`any_overloaded` is the aggregate for a loop that only
wants to know whether the *process* is behind.

**This module opens no door.** It writes nothing, reads no block, and
touches no store; it only paces loops that already have their own
governed write path. Backpressure sheds RATE, never DATA: every wired
loop defers work to a later tick and no input is dropped, so nothing
here can move content past
:meth:`~mind_mem.governance_gate.GovernanceGate.admit_block`.

Every seam function probes the flag with
:func:`~mind_mem.v4.feature_flags.is_enabled_quiet` and returns the
inert answer when it is off — no controller is constructed, nothing is
logged, no config warning is emitted. A probe that decides whether a
feature runs must not be observable when the answer is no (slice 1).

Copyright STARGA, Inc.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .feature_flags import flag_config, is_enabled_quiet, require_enabled

__all__ = [
    "FLAG",
    "BackpressureController",
    "DEFAULT_HIGH_WATERMARK",
    "DEFAULT_LOW_WATERMARK",
    "DEFAULT_MAX_PAUSE_S",
    "PRODUCER_CHANGE_STREAM",
    "PRODUCER_DAEMON",
    "PRODUCER_INBOX",
    "PRODUCER_WEBHOOK",
    "any_overloaded",
    "batch_limit",
    "controller",
    "producer_controller",
    "producer_overloaded",
    "report_depth",
    "reset_for_tests",
    "snapshot",
    "wiring_enabled",
]


FLAG: str = "backpressure"

DEFAULT_HIGH_WATERMARK: int = 1000
DEFAULT_LOW_WATERMARK: int = 200
DEFAULT_MAX_PAUSE_S: float = 5.0

#: Canonical producer names. A loop reports under its own name so two
#: unrelated backlogs cannot overwrite each other's depth.
PRODUCER_CHANGE_STREAM: str = "change_stream"
PRODUCER_INBOX: str = "inbox"
PRODUCER_DAEMON: str = "daemon"
#: Reserved for the ingestion webhook drain, which does not exist yet —
#: see the wiring note in ``docs/v4-release.md``. Named here so the drain
#: author wires one line instead of inventing a fourth spelling.
PRODUCER_WEBHOOK: str = "webhook"


@dataclass(eq=False)
class BackpressureController:
    """Hysteresis-gated overload signal with exponential-backoff hint.

    Two watermarks, ``high_watermark`` and ``low_watermark``, gate
    state transitions:

        depth >= high_watermark    →  enter overloaded state
        depth <= low_watermark     →  exit overloaded state
        in between                 →  state unchanged (hysteresis)

    The hysteresis prevents flapping at the boundary. While
    overloaded, ``recommended_pause()`` returns an exponentially-
    growing pause hint (doubles each call up to ``max_pause_seconds``).
    Once depth recovers below ``low_watermark``, the pause hint resets
    to zero.

    Defaults: ``high=1000``, ``low=200``, ``max_pause=5.0``. Override
    via ``mind-mem.json``:

        "v4": {"backpressure": {"enabled": true,
                                "high_watermark": 5000,
                                "low_watermark": 500}}
    """

    high_watermark: int = DEFAULT_HIGH_WATERMARK
    low_watermark: int = DEFAULT_LOW_WATERMARK
    max_pause_seconds: float = DEFAULT_MAX_PAUSE_S
    _depth: int = 0
    _overloaded: bool = False
    _consecutive_overload: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.low_watermark > self.high_watermark:
            raise ValueError(f"low_watermark must be <= high_watermark (got low={self.low_watermark}, high={self.high_watermark})")

    def set_depth(self, depth: int) -> None:
        """Update queue depth. Triggers hysteresis-gated state change."""
        with self._lock:
            self._depth = max(0, int(depth))
            if not self._overloaded and self._depth >= self.high_watermark:
                self._overloaded = True
                self._consecutive_overload = 0
            elif self._overloaded and self._depth <= self.low_watermark:
                self._overloaded = False
                self._consecutive_overload = 0

    def is_overloaded(self) -> bool:
        """Read-only overload check. Safe in any caller."""
        with self._lock:
            return self._overloaded

    def depth(self) -> int:
        """Current queue depth as last reported."""
        with self._lock:
            return self._depth

    def recommended_pause(self) -> float:
        """Return seconds the caller should pause AND advance the
        backoff counter.

        Zero when not overloaded. Exponential backoff while overloaded
        (1× → 2× → 4× → 8× × ``max_pause_seconds`` cap).

        **Side-effect note:** every call advances the internal backoff
        tick. Callers that want to *peek* the current pause without
        advancing should use :meth:`current_pause` instead and call
        :meth:`record_overload_tick` explicitly after sleeping.
        """
        with self._lock:
            if not self._overloaded:
                return 0.0
            self._consecutive_overload = min(self._consecutive_overload + 1, 16)
            base: float = 0.05  # 50ms base
            pause: float = base * float(2 ** (self._consecutive_overload - 1))
            return min(pause, self.max_pause_seconds)

    def current_pause(self) -> float:
        """Pure read — what the next pause WOULD be, without advancing.

        Returns 0.0 when not overloaded. Useful for logging /
        observability dashboards that want to surface the controller
        state without distorting it.
        """
        with self._lock:
            if not self._overloaded:
                return 0.0
            tick = max(self._consecutive_overload, 1)
            base: float = 0.05
            pause: float = base * float(2 ** (tick - 1))
            return min(pause, self.max_pause_seconds)

    def record_overload_tick(self) -> None:
        """Manually advance the backoff counter without returning a
        pause hint. Pair with :meth:`current_pause` when the caller
        wants to read-then-tick under explicit control."""
        with self._lock:
            if self._overloaded:
                self._consecutive_overload = min(self._consecutive_overload + 1, 16)

    def wait_until_clear(self, *, timeout: float = 30.0, poll: float = 0.1) -> bool:
        """Block until ``is_overloaded()`` becomes False or ``timeout``.

        Returns True if cleared, False if timed out. Useful for
        synchronous producer threads that want a simple "wait for
        capacity" call. Async callers should use ``recommended_pause()``
        in a non-blocking loop.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if not self.is_overloaded():
                return True
            time.sleep(poll)
        return not self.is_overloaded()


# ---------------------------------------------------------------------------
# Module-level singleton (config-driven)
# ---------------------------------------------------------------------------


_singleton: BackpressureController | None = None
_singleton_lock = threading.Lock()


def controller() -> BackpressureController:
    """Get-or-create the workspace-level controller. Lazy + thread-safe.

    Configuration is read from ``mind-mem.json`` at first call;
    subsequent calls return the same instance. Calling this when the
    flag is OFF raises :class:`FeatureDisabledError`.
    """
    require_enabled(FLAG)
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            cfg = _load_config()
            _singleton = BackpressureController(
                high_watermark=cfg["high_watermark"],
                low_watermark=cfg["low_watermark"],
                max_pause_seconds=cfg["max_pause_seconds"],
            )
        return _singleton


def reset_for_tests() -> None:
    """Reset the module singleton AND the keyed producer registry.

    Both, always. A test that flipped the flag and reset only the
    singleton would keep a stale producer controller — carrying the
    previous test's depth and watermarks — and the next assertion would
    be reading the wrong object.
    """
    global _singleton
    with _singleton_lock:
        _singleton = None
    with _producers_lock:
        _producers.clear()


def _load_config(producer: str | None = None) -> dict[str, Any]:
    """Watermarks for *producer*, or the workspace defaults when None.

    Per-producer overrides nest under ``producers``; anything absent
    falls back to the top-level value, and that to the module default::

        "v4": {"backpressure": {"enabled": true,
                                "high_watermark": 5000,
                                "producers": {"inbox": {"high_watermark": 50,
                                                        "low_watermark": 10}}}}

    A change-stream queue holding 5000 events and an inbox holding 50
    files are not the same kind of "deep", so one pair of watermarks for
    every producer would mean the signal fires for at most one of them.
    """
    raw = flag_config(FLAG, quiet=True)
    if not isinstance(raw, dict):
        raw = {}
    if producer:
        per = raw.get("producers")
        override = per.get(producer) if isinstance(per, dict) else None
        if isinstance(override, dict):
            raw = {**raw, **override}
    fields = {
        "high_watermark": (int, DEFAULT_HIGH_WATERMARK),
        "low_watermark": (int, DEFAULT_LOW_WATERMARK),
        "max_pause_seconds": (float, DEFAULT_MAX_PAUSE_S),
    }
    out: dict[str, Any] = {}
    for key, (caster, default) in fields.items():
        v = raw.get(key, default)
        try:
            out[key] = caster(v)
        except (TypeError, ValueError):
            out[key] = default
    if out["low_watermark"] > out["high_watermark"]:
        out["low_watermark"] = out["high_watermark"]
    return out


# ---------------------------------------------------------------------------
# Producer wiring seam
# ---------------------------------------------------------------------------
#
# The half that makes the controller reachable. Everything below is
# FLAG-GATED AND SILENT WHEN OFF: the probe is is_enabled_quiet, so a
# flag-off process constructs no controller, logs no line, and emits no
# v4_config_unreadable warning on a malformed config. See the slice-1
# finding recorded in feature_flags.is_enabled_quiet.
#
# Nothing here writes, reads or admits a block. Backpressure paces loops
# that already own their governed write path; it never becomes one.

#: Keyed per-producer controllers. Distinct from the ``controller()``
#: singleton, which stays the workspace-level default for callers that
#: predate the wiring.
_producers: dict[str, BackpressureController] = {}
_producers_lock = threading.Lock()


def wiring_enabled() -> bool:
    """Is the producer wiring armed? Silent either way.

    Uses :func:`~mind_mem.v4.feature_flags.is_enabled_quiet`, never
    ``is_enabled``: this is the probe that decides whether an OFF-by-
    default surface runs at all, and a probe that logs makes the
    flag-off build observably different from a build that never had the
    feature.
    """
    return is_enabled_quiet(FLAG)


def producer_controller(name: str) -> BackpressureController:
    """Get-or-create the controller for producer *name*.

    Lazy, thread-safe, config-driven. Callers must have checked
    :func:`wiring_enabled` first — this constructs state, so calling it
    on an OFF path is exactly the observability leak the flag exists to
    prevent.
    """
    with _producers_lock:
        ctrl = _producers.get(name)
        if ctrl is None:
            cfg = _load_config(name)
            ctrl = BackpressureController(
                high_watermark=cfg["high_watermark"],
                low_watermark=cfg["low_watermark"],
                max_pause_seconds=cfg["max_pause_seconds"],
            )
            _producers[name] = ctrl
        return ctrl


def report_depth(name: str, depth: int) -> bool:
    """Report *depth* for producer *name*; return whether it is overloaded.

    The one call a producer loop needs: it hands over its backlog size
    and learns, from the same call, whether to stop adding work. Returns
    ``False`` and does nothing at all when the flag is off.
    """
    if not wiring_enabled():
        return False
    ctrl = producer_controller(name)
    ctrl.set_depth(depth)
    return ctrl.is_overloaded()


def producer_overloaded(name: str) -> bool:
    """Read-only overload check for one producer. ``False`` when off.

    Never constructs a controller: a *reader* that creates the thing it
    reads would report a fresh, empty controller as "not overloaded" and
    quietly reset nothing — but it would also make an OFF-path caller
    allocate state. Absence of a controller means nothing has reported,
    which is not overloaded.
    """
    if not wiring_enabled():
        return False
    with _producers_lock:
        ctrl = _producers.get(name)
    return bool(ctrl is not None and ctrl.is_overloaded())


def any_overloaded() -> bool:
    """True when ANY producer that has reported is overloaded.

    The aggregate for a loop that only wants to know whether the
    *process* is behind (the daemon tick), rather than which queue.
    ``False`` when the flag is off.
    """
    if not wiring_enabled():
        return False
    with _producers_lock:
        controllers = list(_producers.values())
    return any(c.is_overloaded() for c in controllers)


def batch_limit(name: str, depth: int) -> int | None:
    """Report *depth* and return a per-tick burst cap, or ``None``.

    ``None`` means "no cap" — the flag is off, or the producer is
    keeping up. While overloaded the cap is the producer's
    ``low_watermark`` (at least 1), so a tick makes guaranteed forward
    progress and then yields.

    This is the shape backpressure has to take on a SELF-DRAINING
    backlog. Telling the inbox drain to stop while its own queue is what
    is deep would livelock it: the depth can only fall if the drain
    runs. Capping the burst relieves the downstream without ever
    stalling the thing that clears the backlog, and the deferred files
    stay exactly where they are — rate is shed, data never is.
    """
    if not report_depth(name, depth):
        return None
    return max(1, producer_controller(name).low_watermark)


def snapshot() -> dict[str, dict[str, Any]]:
    """Per-producer state for operator surfaces. ``{}`` when off.

    Uses :meth:`BackpressureController.current_pause`, the peek, never
    ``recommended_pause`` — an observability read that advanced the
    backoff counter would make watching the system change it.
    """
    if not wiring_enabled():
        return {}
    with _producers_lock:
        items = sorted(_producers.items())
    return {
        name: {
            "depth": ctrl.depth(),
            "overloaded": ctrl.is_overloaded(),
            "high_watermark": ctrl.high_watermark,
            "low_watermark": ctrl.low_watermark,
            "pause_seconds": ctrl.current_pause(),
        }
        for name, ctrl in items
    }
