"""Per-client sliding-window rate limiter for the MCP surface.

Extracted from ``mcp_server.py`` in the v3.2.0 §1.2 decomposition
(see docs/v3.2.0-mcp-decomposition-plan.md PR-1). Provides:

* :class:`SlidingWindowRateLimiter` — the primitive.
* :func:`_get_client_rate_limiter` — per-client cache with an LRU
  eviction cap so a token-rotation attacker can't grow the dict
  without bound.
* :func:`_get_client_id` — stable client identifier pulled from
  the active FastMCP access token, with ``"default"`` fallback.
* :func:`_init_rate_limiter` — factory consulted by the cache; it
  resolves the ``rate_limit_calls_per_minute`` limit from config
  via a lazy import of ``mcp_server._get_limits`` (that helper
  moves to ``infra/config.py`` in a later step of this same PR-1).

Behavior is bit-for-bit identical to the pre-move version — the
cap (1024), default calls-per-minute (120), and 60-second window
are all preserved.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

from fastmcp.server.dependencies import get_access_token


class SlidingWindowRateLimiter:
    """In-memory sliding-window rate limiter."""

    def __init__(self, max_calls: int = 120, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window = window_seconds
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    def allow(self) -> tuple[bool, float]:
        """Check if a call is allowed.

        Returns (allowed, retry_after_seconds).  retry_after is 0.0 when allowed.
        """
        now = time.monotonic()
        with self._lock:
            cutoff = now - self.window
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            if len(self._timestamps) >= self.max_calls:
                retry_after = self._timestamps[0] - cutoff
                return False, max(retry_after, 0.1)
            self._timestamps.append(now)
            return True, 0.0


def _init_rate_limiter() -> SlidingWindowRateLimiter:
    """Create a rate limiter from config limits (used as per-client factory)."""
    try:
        # Late import: ``_get_limits`` still lives in mcp_server.py and will
        # move to ``infra/config.py`` in a later step of this PR-1. The
        # deferred lookup avoids the import cycle that a top-level import
        # would create (mcp_server → infra.rate_limit → mcp_server).
        from mind_mem.mcp_server import _get_limits

        limits = _get_limits()
        max_calls = limits["rate_limit_calls_per_minute"]
    except Exception:
        max_calls = 120
    return SlidingWindowRateLimiter(max_calls=max_calls, window_seconds=60)


# Per-client rate limiters keyed by client_id — prevents one client from
# exhausting the global budget and blocking all other clients. An
# attacker rotating Authorization tokens on every request would
# otherwise grow this dict unbounded, so we LRU-evict the least-
# recently-used entry when the cap is reached.
_RATE_LIMITER_MAX: int = 1024
_rate_limiters: "OrderedDict[str, SlidingWindowRateLimiter]" = OrderedDict()
_rate_limiters_lock = threading.Lock()


def _get_client_rate_limiter(client_id: str) -> SlidingWindowRateLimiter:
    """Return (creating if needed) the per-client SlidingWindowRateLimiter."""
    with _rate_limiters_lock:
        existing = _rate_limiters.get(client_id)
        if existing is not None:
            _rate_limiters.move_to_end(client_id, last=True)
            return existing
        limiter = _init_rate_limiter()
        _rate_limiters[client_id] = limiter
        while len(_rate_limiters) > _RATE_LIMITER_MAX:
            _rate_limiters.popitem(last=False)
        return limiter


def _get_client_id() -> str:
    """Return a stable client identifier for the current request."""
    try:
        token = get_access_token()
        if token is not None and token.client_id:
            return token.client_id
    except Exception:
        pass
    return "default"
