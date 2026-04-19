"""v3.2.0 §1.2 decomposition namespace — subpackage for MCP server modules.

The full mcp_server.py monolith (4,604 LOC, 57 tools, 8 resources)
is being decomposed per docs/v3.2.0-mcp-decomposition-plan.md
across a sequence of per-PR increments. PR-1 extracts the cross-
cutting infra helpers (workspace, ACL, rate-limit, observability,
config, http_auth) into ``mind_mem.mcp.infra.*`` while leaving the
public ``mcp_server`` surface untouched. Subsequent PRs move the
``@mcp.tool`` declarations into ``mind_mem.mcp.tools.*`` (one
module per domain) and the ``@mcp.resource`` declarations into
``mind_mem.mcp.resources.*``.

Imports from ``mind_mem.mcp_server`` stay unchanged throughout;
the old file re-exports every moved symbol so downstream callers
never see the migration. See the plan for the 16-PR staging.
"""

from __future__ import annotations
