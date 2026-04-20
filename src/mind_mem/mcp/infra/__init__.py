"""Cross-cutting infra helpers extracted from mcp_server.py (v3.2.0 §1.2 PR-1).

Re-exports the public helpers from each submodule so callers can
``from mind_mem.mcp.infra import _workspace, _check_workspace`` in
one go. Each helper keeps its original leading-underscore name for
source compatibility with the callers still inside mcp_server.py.
"""

from __future__ import annotations

from .acl import (
    _ADMIN_SCOPES,
    ADMIN_TOOLS,
    USER_TOOLS,
    _get_request_scope,
    check_tool_acl,
)
from .config import (
    _DEFAULT_LIMITS,
    QUERY_TIMEOUT_SECONDS,
    _get_limits,
    _load_config,
    _load_extra_categories,
)
from .rate_limit import (
    _RATE_LIMITER_MAX,
    SlidingWindowRateLimiter,
    _get_client_id,
    _get_client_rate_limiter,
    _init_rate_limiter,
    _rate_limiters,
    _rate_limiters_lock,
)
from .workspace import _check_workspace, _read_file, _validate_path, _workspace

__all__ = [
    "_workspace",
    "_check_workspace",
    "_validate_path",
    "_read_file",
    "ADMIN_TOOLS",
    "USER_TOOLS",
    "_ADMIN_SCOPES",
    "check_tool_acl",
    "_get_request_scope",
    "SlidingWindowRateLimiter",
    "_init_rate_limiter",
    "_get_client_rate_limiter",
    "_get_client_id",
    "_RATE_LIMITER_MAX",
    "_rate_limiters",
    "_rate_limiters_lock",
    "_DEFAULT_LIMITS",
    "_get_limits",
    "_load_config",
    "_load_extra_categories",
    "QUERY_TIMEOUT_SECONDS",
]
