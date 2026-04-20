"""MCP-surface-wide constants shared by the infra submodules.

Extracted from ``mcp_server.py`` in the v3.2.0 §1.2 decomposition
(see docs/v3.2.0-mcp-decomposition-plan.md PR-1). A single home
for tiny constants that would otherwise create a cycle if put in
any individual submodule (observability → mcp_server, etc.).
"""

from __future__ import annotations

MCP_SCHEMA_VERSION = "1.0"
