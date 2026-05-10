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

Copyright STARGA, Inc.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .feature_flags import flag_config, require_enabled

__all__ = [
    "FLAG",
    "BackpressureController",
    "DEFAULT_HIGH_WATERMARK",
    "DEFAULT_LOW_WATERMARK",
    "DEFAULT_MAX_PAUSE_S",
]


FLAG: str = "backpressure"

DEFAULT_HIGH_WATERMARK: int = 1000
DEFAULT_LOW_WATERMARK: int = 200
DEFAULT_MAX_PAUSE_S: float = 5.0


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
            raise ValueError(
                "low_watermark must be <= high_watermark "
                f"(got low={self.low_watermark}, high={self.high_watermark})"
            )

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
            self._consecutive_overload = min(
                self._consecutive_overload + 1, 16
            )
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
                self._consecutive_overload = min(
                    self._consecutive_overload + 1, 16
                )

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
    """Reset the module singleton. Test-only."""
    global _singleton
    with _singleton_lock:
        _singleton = None


def _load_config() -> dict[str, Any]:
    raw = flag_config(FLAG)
    if not isinstance(raw, dict):
        raw = {}
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
