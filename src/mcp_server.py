"""Compatibility wrapper for the packaged Mind-Mem MCP server."""

from __future__ import annotations

from importlib import import_module

_IMPL = import_module("mind_mem.mcp_server")

globals().update(
    {name: getattr(_IMPL, name) for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))}
)


def __getattr__(name: str):
    return getattr(_IMPL, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_IMPL)))


if __name__ == "__main__":
    raise SystemExit(_IMPL.main())
