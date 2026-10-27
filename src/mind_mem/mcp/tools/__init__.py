"""Per-domain ``@mcp.tool`` modules (v3.2.0 §1.2 PR-3+).

Each submodule owns one cohesive slice of the MCP tool surface and
exposes a ``register(mcp)`` function that wires its tools onto a
``FastMCP`` instance. The surface size is not restated here: the
registry is the authority (``scripts/count_mcp_tools.py`` counts it and
CI pins the number). Module-level function definitions keep every
registered name resolvable as ``mind_mem.mcp_server.<tool>`` via the
re-export shim, so existing callers + tests don't need to learn the
new paths.
"""

from __future__ import annotations
