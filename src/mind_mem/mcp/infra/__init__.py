"""MCP infrastructure compatibility exports, resolved only when requested.

Core library paths read configuration and schema constants from this package.
Importing those modules must not require optional transport/auth dependencies.
Explicit tool, ACL and HTTP imports retain their existing public names.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "_ADMIN_SCOPES": ("acl", "_ADMIN_SCOPES"),
    "ADMIN_TOOLS": ("acl", "ADMIN_TOOLS"),
    "USER_TOOLS": ("acl", "USER_TOOLS"),
    "AuthSnapshot": ("acl", "AuthSnapshot"),
    "_get_request_scope": ("acl", "_get_request_scope"),
    "bind_auth_snapshot": ("acl", "bind_auth_snapshot"),
    "check_tool_acl": ("acl", "check_tool_acl"),
    "current_auth_snapshot": ("acl", "current_auth_snapshot"),
    "_DEFAULT_LIMITS": ("config", "_DEFAULT_LIMITS"),
    "QUERY_TIMEOUT_SECONDS": ("config", "QUERY_TIMEOUT_SECONDS"),
    "_get_limits": ("config", "_get_limits"),
    "_load_config": ("config", "_load_config"),
    "_load_extra_categories": ("config", "_load_extra_categories"),
    "MCP_SCHEMA_VERSION": ("constants", "MCP_SCHEMA_VERSION"),
    "_build_http_auth_tokens": ("http_auth", "_build_http_auth_tokens"),
    "_check_token": ("http_auth", "_check_token"),
    "verify_token": ("http_auth", "verify_token"),
    "_is_db_locked": ("observability", "_is_db_locked"),
    "_sqlite_busy_error": ("observability", "_sqlite_busy_error"),
    "mcp_tool_observe": ("observability", "mcp_tool_observe"),
    "_RATE_LIMITER_MAX": ("rate_limit", "_RATE_LIMITER_MAX"),
    "SlidingWindowRateLimiter": ("rate_limit", "SlidingWindowRateLimiter"),
    "_get_client_id": ("rate_limit", "_get_client_id"),
    "_get_client_rate_limiter": ("rate_limit", "_get_client_rate_limiter"),
    "_init_rate_limiter": ("rate_limit", "_init_rate_limiter"),
    "_rate_limiters": ("rate_limit", "_rate_limiters"),
    "_rate_limiters_lock": ("rate_limit", "_rate_limiters_lock"),
    "_check_workspace": ("workspace", "_check_workspace"),
    "_read_file": ("workspace", "_read_file"),
    "_validate_path": ("workspace", "_validate_path"),
    "_workspace": ("workspace", "_workspace"),
}

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
    "AuthSnapshot",
    "bind_auth_snapshot",
    "current_auth_snapshot",
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
    "MCP_SCHEMA_VERSION",
    "mcp_tool_observe",
    "_sqlite_busy_error",
    "_is_db_locked",
    "_check_token",
    "verify_token",
    "_build_http_auth_tokens",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
