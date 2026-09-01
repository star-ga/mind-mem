"""v4 circuit breaker (round 5 audit, Mistral + GLM 9.9→10 gap).

Both round-5 reviewers flagged the same single missing primitive: a
circuit breaker for external embedders. Per-call fallback handling (the
``FallbackPolicy`` that lived in ``v4.surprise_retrieval``, removed as
unreachable in 5.0.0) covered *individual* embedding failures, but a slow
or timing-out embedder still gets called on every recall, dragging the
whole pipeline. A circuit breaker prevents this
cascading-failure scenario by *short-circuiting* calls to a known-bad
dependency until it recovers.

Three-state machine:

    CLOSED       calls pass through to the wrapped function. Failures
                 increment a counter; after ``failure_threshold``
                 consecutive failures, the breaker trips OPEN.

                 With the optional ``failure_window_s`` set, the counter
                 is instead a *rolling window*: only failures observed in
                 the last ``failure_window_s`` seconds count toward the
                 threshold (older ones age out), so ``failure_threshold``
                 failures *within the window* trip OPEN. Left unset
                 (default), the pure consecutive-failure behaviour above
                 is preserved unchanged.

    OPEN         every call short-circuits with
                 :class:`CircuitOpenError` for the next
                 ``recovery_timeout`` seconds. No load on the failing
                 dependency.

    HALF_OPEN    after ``recovery_timeout``, the next call is a
                 *probe*. Success closes the breaker; failure
                 re-opens it for another full timeout window.

All transitions are atomic under an internal lock. Multiple threads
calling ``call(fn)`` concurrently see consistent state.

Public API:

    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
    try:
        result = breaker.call(my_embedder, text)
    except CircuitOpenError:
        # Fall back to cached embedding / cheaper model / etc.
        ...

Or as a decorator::

    @circuit_breaker(failure_threshold=5, recovery_timeout=30.0)
    def embed(text: str) -> list[float]:
        return external_service.embed(text)

State is observable via :meth:`state`, :meth:`failure_count`, and
:meth:`time_until_retry` for dashboards.

Feature-flag gated under ``v4.circuit_breaker``.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from .feature_flags import flag_config, require_enabled

__all__ = [
    "FLAG",
    "CircuitState",
    "CircuitBreaker",
    "CircuitOpenError",
    "circuit_breaker",
    "DEFAULT_FAILURE_THRESHOLD",
    "DEFAULT_RECOVERY_TIMEOUT_S",
    "DEFAULT_HALF_OPEN_PROBES",
]


FLAG: str = "circuit_breaker"

DEFAULT_FAILURE_THRESHOLD: int = 5
DEFAULT_RECOVERY_TIMEOUT_S: float = 30.0
#: Number of probe successes required in HALF_OPEN before closing.
#: Default 1: a single success closes the breaker. Higher values
#: require sustained recovery before trusting the dependency again.
DEFAULT_HALF_OPEN_PROBES: int = 1


T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised by :meth:`CircuitBreaker.call` while the breaker is OPEN.

    Carries ``retry_after`` (seconds until the next probe is allowed)
    so callers can schedule a retry instead of polling.
    """

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"circuit breaker OPEN; retry_after={retry_after:.2f}s")
        self.retry_after = retry_after


@dataclass(eq=False)
class CircuitBreaker:
    """Three-state breaker around a wrapped callable.

    Defaults are chosen for typical embedding-service profiles:
    ``failure_threshold=5`` (one bad batch is forgiven; persistent
    badness trips), ``recovery_timeout=30.0s`` (long enough for a
    rolling restart, short enough to not block tier-decay loops).

    Override per-instance::

        breaker = CircuitBreaker(failure_threshold=10,
                                 recovery_timeout=60.0,
                                 half_open_probes=3)

    ``failure_window_s`` (optional) switches the CLOSED-state failure
    counter from *consecutive* to a *rolling time window*: only failures
    in the last ``failure_window_s`` seconds count toward the threshold.
    Left ``None`` (default), the consecutive-failure behaviour is
    preserved. This lets one breaker subsume ad-hoc windowed breakers
    (e.g. "trip after N failures within W seconds")::

        breaker = CircuitBreaker(failure_threshold=3,
                                 recovery_timeout=60.0,
                                 failure_window_s=60.0)

    Or globally via ``mind-mem.json``::

        "v4": {"circuit_breaker": {"enabled": true,
                                   "failure_threshold": 5,
                                   "recovery_timeout_s": 30.0,
                                   "half_open_probes": 1,
                                   "failure_window_s": null}}
    """

    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT_S
    half_open_probes: int = DEFAULT_HALF_OPEN_PROBES
    #: Optional rolling-window width (seconds). ``None`` => consecutive
    #: counting (the original behaviour); a positive value => only
    #: failures in the last window count toward ``failure_threshold``.
    failure_window_s: float | None = None

    _state: CircuitState = CircuitState.CLOSED
    _failure_count: int = 0
    _success_count_in_half_open: int = 0
    _opened_at: float = 0.0
    #: Monotonic timestamps of in-window CLOSED-state failures. Only
    #: populated / consulted when ``failure_window_s`` is set.
    _failure_times: list[float] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError(f"failure_threshold must be >= 1 (got {self.failure_threshold})")
        if self.recovery_timeout < 0:
            raise ValueError(f"recovery_timeout must be >= 0 (got {self.recovery_timeout})")
        if self.half_open_probes < 1:
            raise ValueError(f"half_open_probes must be >= 1 (got {self.half_open_probes})")
        if self.failure_window_s is not None and self.failure_window_s <= 0:
            raise ValueError(f"failure_window_s must be > 0 when set (got {self.failure_window_s})")

    # -----------------------------------------------------------------
    # Read-only state inspection
    # -----------------------------------------------------------------

    def state(self) -> CircuitState:
        """Current state — may transition OPEN→HALF_OPEN if the
        recovery window has elapsed since this read."""
        with self._lock:
            self._maybe_half_open_locked()
            return self._state

    def failure_count(self) -> int:
        """CLOSED-state failures counted toward the threshold.

        In consecutive mode (``failure_window_s`` unset) this is the run
        of failures since the last success / transition. In windowed mode
        it is the number of failures observed in the last
        ``failure_window_s`` seconds (stale ones are pruned on read).
        Reset to zero on every success and on every state transition;
        does not include failures observed in HALF_OPEN.
        """
        with self._lock:
            if self.failure_window_s is not None:
                self._prune_failures_locked(time.monotonic())
                self._failure_count = len(self._failure_times)
            return self._failure_count

    def time_until_retry(self) -> float:
        """Seconds remaining until the next probe is allowed.

        Returns 0.0 when CLOSED or HALF_OPEN. When OPEN, returns the
        clamped non-negative remainder of the recovery window.
        """
        with self._lock:
            if self._state is not CircuitState.OPEN:
                return 0.0
            elapsed = time.monotonic() - self._opened_at
            return max(0.0, self.recovery_timeout - elapsed)

    # -----------------------------------------------------------------
    # Manual transitions (for tests + operator overrides)
    # -----------------------------------------------------------------

    def reset(self) -> None:
        """Force the breaker back to CLOSED. Operator override —
        useful when an operator knows the dependency is healthy
        before the recovery window elapses."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._failure_times.clear()
            self._success_count_in_half_open = 0
            self._opened_at = 0.0

    def trip(self) -> None:
        """Force the breaker OPEN. Operator override — useful when
        external monitoring sees the dependency is sick before the
        breaker reaches its failure threshold (fail-fast)."""
        with self._lock:
            now = time.monotonic()
            self._state = CircuitState.OPEN
            self._opened_at = now
            self._failure_count = self.failure_threshold
            if self.failure_window_s is not None:
                self._failure_times = [now] * self.failure_threshold
            self._success_count_in_half_open = 0

    # -----------------------------------------------------------------
    # The call path
    # -----------------------------------------------------------------

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call ``fn`` through the breaker. Returns its result on
        success, or raises :class:`CircuitOpenError` while OPEN.

        Exceptions raised by ``fn`` are recorded as failures and
        re-raised (callers see the original exception, not a wrapper)
        unless the breaker is OPEN, in which case ``fn`` is never
        called and ``CircuitOpenError`` is raised pre-emptively.

        This is the v4-surface entry point: it requires the
        ``circuit_breaker`` feature flag. Internal always-on callers that
        must be protected regardless of the flag use :meth:`guarded_call`.
        """
        require_enabled(FLAG)
        return self.guarded_call(fn, *args, **kwargs)

    def guarded_call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Breaker-guarded call WITHOUT the v4 feature-flag gate.

        Identical semantics to :meth:`call` (state check → invoke →
        record success/failure → re-raise original exception; raise
        :class:`CircuitOpenError` while OPEN) but never consults
        :func:`require_enabled`. Intended for internal, always-on
        breakers — e.g. the embedder fallback chain — that must
        short-circuit a dead dependency even when the optional
        ``circuit_breaker`` surface is disabled.
        """
        # Decide state under the lock; release before calling.
        with self._lock:
            self._maybe_half_open_locked()
            if self._state is CircuitState.OPEN:
                elapsed = time.monotonic() - self._opened_at
                retry_after = max(0.0, self.recovery_timeout - elapsed)
                raise CircuitOpenError(retry_after=retry_after)

        try:
            result = fn(*args, **kwargs)
        except BaseException:
            self._record_failure()
            raise
        else:
            self._record_success()
            return result

    # -----------------------------------------------------------------
    # Internals — all assume caller holds the lock or is about to
    # take it.
    # -----------------------------------------------------------------

    def _maybe_half_open_locked(self) -> None:
        """Transition OPEN → HALF_OPEN if the recovery window elapsed."""
        if self._state is not CircuitState.OPEN:
            return
        if (time.monotonic() - self._opened_at) >= self.recovery_timeout:
            self._state = CircuitState.HALF_OPEN
            self._success_count_in_half_open = 0

    def _prune_failures_locked(self, now: float) -> None:
        """Drop failure timestamps older than the rolling window.

        No-op in consecutive mode (``failure_window_s`` unset). Caller
        holds the lock.
        """
        if self.failure_window_s is None:
            return
        cutoff = now - self.failure_window_s
        if self._failure_times and self._failure_times[0] <= cutoff:
            self._failure_times = [t for t in self._failure_times if t > cutoff]

    def _record_success(self) -> None:
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._success_count_in_half_open += 1
                if self._success_count_in_half_open >= self.half_open_probes:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._failure_times.clear()
                    self._success_count_in_half_open = 0
            else:
                # Reset the failure run/window on every CLOSED success —
                # only failures *without an intervening success* should
                # trip the breaker.
                self._failure_count = 0
                self._failure_times.clear()

    def _record_failure(self) -> None:
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                # A failed probe re-opens for another full window.
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._success_count_in_half_open = 0
                return
            if self.failure_window_s is None:
                self._failure_count += 1
            else:
                now = time.monotonic()
                self._prune_failures_locked(now)
                self._failure_times.append(now)
                self._failure_count = len(self._failure_times)
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()


# ---------------------------------------------------------------------------
# Decorator wrapper
# ---------------------------------------------------------------------------


def circuit_breaker(
    *,
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
    recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT_S,
    half_open_probes: int = DEFAULT_HALF_OPEN_PROBES,
    failure_window_s: float | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator factory — wraps a function in a per-function breaker.

    Each decorated function gets its own :class:`CircuitBreaker`
    instance; failures of one wrapped function do not trip the breaker
    of another. Use :meth:`CircuitBreaker.reset` on the ``.breaker``
    attribute attached to the wrapper for tests / manual recovery.

    ::

        @circuit_breaker(failure_threshold=3, recovery_timeout=10.0)
        def embed(text: str) -> list[float]:
            return slow_external_service.embed(text)

        # Inspect / control the underlying breaker
        embed.breaker.state()
        embed.breaker.reset()
    """
    breaker = CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        half_open_probes=half_open_probes,
        failure_window_s=failure_window_s,
    )

    def _decorate(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def _inner(*args: Any, **kwargs: Any) -> T:
            return breaker.call(fn, *args, **kwargs)

        # Expose the breaker for inspection / reset.
        _inner.breaker = breaker  # type: ignore[attr-defined]
        return _inner

    return _decorate


# ---------------------------------------------------------------------------
# Config-driven singleton (optional convenience)
# ---------------------------------------------------------------------------


_default_breaker: CircuitBreaker | None = None
_default_lock = threading.Lock()


def default_breaker() -> CircuitBreaker:
    """Get-or-create a process-wide default breaker. Lazy + thread-safe.

    Useful for callers that want one breaker shared across every
    embedder call, without each module instantiating its own. Reads
    config from ``mind-mem.json`` at first call.
    """
    require_enabled(FLAG)
    global _default_breaker
    with _default_lock:
        if _default_breaker is None:
            cfg = _load_config()
            _default_breaker = CircuitBreaker(
                failure_threshold=cfg["failure_threshold"],
                recovery_timeout=cfg["recovery_timeout_s"],
                half_open_probes=cfg["half_open_probes"],
                failure_window_s=cfg["failure_window_s"],
            )
        return _default_breaker


def reset_for_tests() -> None:
    """Reset the module singleton. Test-only — never call in production."""
    global _default_breaker
    with _default_lock:
        _default_breaker = None


def _load_config() -> dict[str, Any]:
    raw = flag_config(FLAG)
    if not isinstance(raw, dict):
        raw = {}
    fields = {
        "failure_threshold": (int, DEFAULT_FAILURE_THRESHOLD),
        "recovery_timeout_s": (float, DEFAULT_RECOVERY_TIMEOUT_S),
        "half_open_probes": (int, DEFAULT_HALF_OPEN_PROBES),
    }
    out: dict[str, Any] = {}
    for key, (caster, default) in fields.items():
        v = raw.get(key, default)
        try:
            out[key] = caster(v)
        except (TypeError, ValueError):
            out[key] = default
    # Optional rolling window: absent / null => None (consecutive mode).
    window = raw.get("failure_window_s", None)
    if window is None:
        out["failure_window_s"] = None
    else:
        try:
            parsed = float(window)
            out["failure_window_s"] = parsed if parsed > 0 else None
        except (TypeError, ValueError):
            out["failure_window_s"] = None
    return out
