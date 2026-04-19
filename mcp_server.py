#!/usr/bin/env python3
"""Source-checkout entrypoint for the packaged Mind-Mem MCP server.

Loads the real implementation via `exec(compile(_SOURCE, ...))` so
that every symbol — including the private `_`-prefixed helpers and
the ``fts_query`` free function — lives in *this* module's namespace.

Tests (`tests/test_mcp_v140.py` etc.) patch attributes on this
module and expect the patches to propagate to the tool functions.
An `import`-and-re-export shim would break those patches because
the tool functions would continue resolving via their own
originating module. The exec path preserves the expected behaviour.

In the installed wheel, `mind-mem-mcp` (console script →
`mind_mem.mcp_entry:main`) is the canonical entry point and this
file is not shipped.

STARGA, Inc. — Apache-2.0.
"""

from __future__ import annotations

import argparse
import os
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mind-Mem MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="HTTP port (only used with --transport http)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Bearer token for HTTP auth (or set MIND_MEM_TOKEN env var)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Auto-reindex when workspace .md files change",
    )
    parser.add_argument(
        "--watch-interval",
        type=float,
        default=5.0,
        help="File watch polling interval in seconds (default: 5.0)",
    )
    return parser


def _missing_fastmcp() -> int:
    print(
        "Error: fastmcp is required to run the Mind-Mem MCP server. "
        "Install the 'mcp' extra or `pip install fastmcp==2.14.5`.",
        file=sys.stderr,
    )
    return 1


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_IMPL_PATH = os.path.join(_SRC_DIR, "mind_mem", "mcp_server.py")
with open(_IMPL_PATH, "r", encoding="utf-8") as f:
    _SOURCE = f.read()

if {"-h", "--help"} & set(sys.argv[1:]):
    _build_parser().print_help()
    raise SystemExit(0)

try:
    exec(compile(_SOURCE, _IMPL_PATH, "exec"), globals(), globals())
except ModuleNotFoundError as exc:
    if exc.name == "fastmcp":
        raise SystemExit(_missing_fastmcp()) from exc
    raise
