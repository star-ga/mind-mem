# Copyright 2026 STARGA, Inc.
"""Declarative registry of every AI client mind-mem integrates with.

Each :class:`AgentSpec` says where the client's instruction file and MCP
config live, which dialect each uses, and how to detect the client. The
installer (:mod:`mind_mem.hook_installer`, :mod:`mind_mem.client_mcp`) is
generic over this data: adding a client is adding an entry here.

Integration tiers:

* automatic MCP -- ``mcp_fmt`` set; the entry is merged into the config file
* guarded MCP   -- ``mcp_guard="dir"``: written only when the client's config
  directory already exists (path or key not fully documented upstream)
* manual        -- ``manual`` set, no ``mcp_fmt``: the client has MCP but no
  per-user file to merge into (UI-only, per-deployment, HTTP-only); the
  installer prints the exact steps and snippet
* instructions only -- neither: no native MCP, so only the instruction file
"""

from __future__ import annotations

from mind_mem.client_registry_data import CORE_AGENTS, MANUAL_AGENTS, MORE_AGENTS
from mind_mem.client_registry_schema import (
    _MM_MARKER,
    MEMORY_PROTOCOL_SNIPPET,
    AgentSpec,
    _client,  # noqa: F401
    _manual,  # noqa: F401
)

AGENT_REGISTRY: dict[str, AgentSpec] = {spec.name: spec for spec in (*CORE_AGENTS, *MORE_AGENTS, *MANUAL_AGENTS)}


__all__ = ["AGENT_REGISTRY", "AgentSpec", "MEMORY_PROTOCOL_SNIPPET", "_MM_MARKER"]
