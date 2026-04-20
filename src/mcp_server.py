"""Wheel-level compatibility module for `mind_mem.mcp_server`.

Registered as a top-level `mcp_server` module via
`[tool.setuptools] py-modules = ["mcp_server"]` in pyproject.toml so
that clients importing `mcp_server` from anywhere in the MCP
ecosystem resolve to the packaged implementation.

The real implementation lives in `mind_mem.mcp_server`; this file
re-exports its public surface and nothing else.

STARGA, Inc. — Apache-2.0.
"""

from __future__ import annotations

import sys

try:
    from mind_mem.mcp_server import *  # noqa: F401,F403 — re-export surface
    from mind_mem.mcp_server import main  # noqa: F401 — exposed for console script fallback
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name == "fastmcp":
        sys.stderr.write(
            "Error: fastmcp is required to run the Mind-Mem MCP server. Install the 'mcp' extra or `pip install fastmcp==2.14.5`.\n"
        )
        raise SystemExit(1) from exc
    raise


if __name__ == "__main__":
    main()
    raise SystemExit(0)
