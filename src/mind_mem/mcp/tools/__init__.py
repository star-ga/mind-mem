"""Per-domain ``@mcp.tool`` modules (v3.2.0 §1.2 PR-3+).

Each submodule owns one cohesive slice of the 57-tool MCP surface
and exposes a ``register(mcp)`` function that wires its tools onto
a ``FastMCP`` instance. Module-level function definitions keep the
names resolvable as ``mcp_server.<tool>`` via re-export shims, so
existing callers + tests don't need to learn the new paths.
"""

from __future__ import annotations
