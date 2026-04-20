"""Observability + DB-busy helpers for the MCP surface.

Extracted from ``mcp_server.py`` in the v3.2.0 §1.2 decomposition
(see docs/v3.2.0-mcp-decomposition-plan.md PR-1). Provides:

* :func:`mcp_tool_observe` — the cross-cutting decorator applied
  to every ``@mcp.tool`` body. Enforces the rate limiter + ACL
  gates and emits structured JSON logs + counters per call.
* :func:`_sqlite_busy_error` — canonical JSON payload returned
  when SQLite reports ``database is locked``.
* :func:`_is_db_locked` — predicate for ``sqlite3.OperationalError``.

Behavior is bit-for-bit identical to the pre-move version — the
log category ``mcp_server`` is preserved, the metric names
(``mcp_tool_duration_ms``, ``mcp_tool_success``,
``mcp_tool_failure``) are unchanged, and the rate-limit + ACL
ordering is identical.
"""

from __future__ import annotations

import functools
import json
import os
import sqlite3
import time

from mind_mem.observability import get_logger, metrics

from .acl import ADMIN_TOOLS, USER_TOOLS, _get_request_scope, check_tool_acl
from .constants import MCP_SCHEMA_VERSION
from .rate_limit import _get_client_id, _get_client_rate_limiter

_log = get_logger("mcp_server")


def _sqlite_busy_error() -> str:
    """Return structured JSON error for SQLite database locked (#29)."""
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": "database_busy",
            "message": "Database is temporarily locked by another process",
            "retry_after_seconds": 1,
        }
    )


def _is_db_locked(exc: sqlite3.OperationalError) -> bool:
    """Check if a sqlite3.OperationalError is a database-locked error."""
    return "database is locked" in str(exc).lower()


def mcp_tool_observe(fn):
    """Decorator that wraps MCP tool calls with observability logging (#31).

    Logs structured JSON for every call: tool_name, duration_ms, success,
    error_type, result_size.  Also increments success/failure counters.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        tool_name = fn.__name__

        # Rate limit enforcement (#475): per-client sliding window
        client_id = _get_client_id()
        limiter = _get_client_rate_limiter(client_id)
        allowed, retry_after = limiter.allow()
        if not allowed:
            _log.warning("rate_limit_exceeded", tool=tool_name, client=client_id, retry_after=retry_after)
            return json.dumps(
                {
                    "error": "Rate limit exceeded. Try again later.",
                    "retry_after_seconds": round(retry_after, 1),
                    "_schema_version": MCP_SCHEMA_VERSION,
                }
            )

        # ACL enforcement: when MIND_MEM_ADMIN_TOKEN is configured,
        # admin tools require MIND_MEM_SCOPE=admin (stdio) or a valid
        # Authorization header (http).  When no admin token is set the
        # check is skipped so single-user local setups are unaffected.
        admin_token = os.environ.get("MIND_MEM_ADMIN_TOKEN")
        request_scope = _get_request_scope()
        acl_scope = request_scope or os.environ.get("MIND_MEM_SCOPE", "user")
        acl_active = admin_token is not None or request_scope is not None
        if acl_active and tool_name in ADMIN_TOOLS:
            scope = acl_scope
            acl_error = check_tool_acl(tool_name, scope)
            if acl_error:
                _log.warning("acl_blocked", tool=tool_name, scope=scope)
                return acl_error
        elif acl_active and tool_name not in USER_TOOLS:
            _log.warning("acl_unknown_tool", tool=tool_name)
            return json.dumps({"error": f"Tool '{tool_name}' is not in ACL policy", "_schema_version": "1.0"})

        start = time.monotonic()
        error_type = None
        success = True
        result = ""
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as exc:
            success = False
            error_type = type(exc).__name__
            raise
        finally:
            duration_ms = round((time.monotonic() - start) * 1000, 2)
            result_size = len(result) if isinstance(result, str) else 0
            _log.info(
                "mcp_tool_call",
                tool_name=tool_name,
                duration_ms=duration_ms,
                success=success,
                error_type=error_type,
                result_size=result_size,
            )
            metrics.observe("mcp_tool_duration_ms", duration_ms)
            if success:
                metrics.inc("mcp_tool_success")
            else:
                metrics.inc("mcp_tool_failure")

    return wrapper
