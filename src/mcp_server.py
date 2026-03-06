"""Compatibility wrapper for the packaged Mind-Mem MCP server."""

from __future__ import annotations

import argparse
import sys
from importlib import import_module


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mind-Mem MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "http"], default="stdio", help="Transport protocol (default: stdio)"
    )
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (only used with --transport http)")
    parser.add_argument("--token", default=None, help="Bearer token for HTTP auth (or set MIND_MEM_TOKEN env var)")
    parser.add_argument("--watch", action="store_true", help="Auto-reindex when workspace .md files change")
    parser.add_argument(
        "--watch-interval", type=float, default=5.0, help="File watch polling interval in seconds (default: 5.0)"
    )
    return parser


def _load_impl():
    if {"-h", "--help"} & set(sys.argv[1:]):
        _build_parser().print_help()
        raise SystemExit(0)
    try:
        return import_module("mind_mem.mcp_server")
    except ModuleNotFoundError as exc:
        if exc.name == "fastmcp":
            print(
                "Error: fastmcp is required to run the Mind-Mem MCP server. "
                "Install the 'mcp' extra or `pip install fastmcp==2.14.5`.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
        raise


_IMPL = _load_impl()

globals().update(
    {name: getattr(_IMPL, name) for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))}
)


def __getattr__(name: str):
    return getattr(_IMPL, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_IMPL)))


if __name__ == "__main__":
    raise SystemExit(_IMPL.main())
