"""v4 circuit breaker (round 5 audit, Mistral + GLM 9.9→10 gap).

Both round-5 reviewers flagged the same single missing primitive: a
circuit breaker for external embedders. The :class:`FallbackPolicy` in
:mod:`mind_mem.v4.surprise_retrieval` handles *individual* embedding
failures, but a slow / timing-out embedder still gets called on every
recall, dragging the whole pipeline. A circuit breaker prevents this
cascading-failure scenario by *short-circuiting* calls to a known-bad
dependency until it recovers.

Three-state machine:

    CLOSED       calls pass through to the wrapped function. Failures
                 increment a counter; after ``failure_threshold``
                 consecutive failures, the breaker trips OPEN.

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
        super().__init__(
            f"circuit breaker OPEN; retry_after={retry_after:.2f}s"
        )
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

    Or globally via ``mind-mem.json``::

        "v4": {"circuit_breaker": {"enabled": true,
                                   "failure_threshold": 5,
                                   "recovery_timeout_s": 30.0,
                                   "half_open_probes": 1}}
    """

    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT_S
    half_open_probes: int = DEFAULT_HALF_OPEN_PROBES

    _state: CircuitState = CircuitState.CLOSED
    _failure_count: int = 0
    _success_count_in_half_open: int = 0
    _opened_at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError(
                f"failure_threshold must be >= 1 (got {self.failure_threshold})"
            )
        if self.recovery_timeout < 0:
            raise ValueError(
                f"recovery_timeout must be >= 0 (got {self.recovery_timeout})"
            )
        if self.half_open_probes < 1:
            raise ValueError(
                f"half_open_probes must be >= 1 (got {self.half_open_probes})"
            )

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
        """Consecutive failures observed in CLOSED state.

        Reset to zero on every success and on every state transition.
        Does not include failures observed in HALF_OPEN.
        """
        with self._lock:
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
            self._success_count_in_half_open = 0
            self._opened_at = 0.0

    def trip(self) -> None:
        """Force the breaker OPEN. Operator override — useful when
        external monitoring sees the dependency is sick before the
        breaker reaches its failure threshold (fail-fast)."""
        with self._lock:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            self._failure_count = self.failure_threshold
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
        """
        require_enabled(FLAG)
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

    def _record_success(self) -> None:
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._success_count_in_half_open += 1
                if self._success_count_in_half_open >= self.half_open_probes:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count_in_half_open = 0
            else:
                # Reset failure run on every CLOSED success — only
                # *consecutive* failures should trip the breaker.
                self._failure_count = 0

    def _record_failure(self) -> None:
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                # A failed probe re-opens for another full window.
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._success_count_in_half_open = 0
                return
            self._failure_count += 1
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
    return out
