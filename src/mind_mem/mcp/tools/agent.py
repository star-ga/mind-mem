"""Agent-bridge + vault MCP tools.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, agent domain). Four tools:

* ``agent_inject`` — render a recall result in the target agent's
  expected snippet format (claude-code / codex / gemini / cursor /
  windsurf / aider / generic).
* ``vault_scan`` / ``vault_sync`` — Obsidian-style vault bridge;
  both gated by the ``MIND_MEM_VAULT_ALLOWLIST`` env var.
* ``stream_status`` — publish / delivery / drop counters from the
  process-wide :class:`ChangeStream` singleton.

``agent_inject`` late-imports ``mcp_server._recall_impl`` because
recall is extracted in a later PR step; the deferred lookup keeps
both sides of that extraction independently committable.
"""

from __future__ import annotations

import json
import os

from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import _change_stream


def _vault_allowlist() -> list[str]:
    """Return the configured vault-root allowlist.

    Set ``MIND_MEM_VAULT_ALLOWLIST`` to a ``:``-separated list of
    absolute directories. When set, every vault MCP tool refuses
    requests targeting paths outside the list. Empty/unset = allow
    any path (legacy behaviour; recommended only for local dev).
    """
    raw = os.environ.get("MIND_MEM_VAULT_ALLOWLIST", "").strip()
    if not raw:
        return []
    sep = ";" if ";" in raw else ":"
    return [os.path.realpath(p.strip()) for p in raw.split(sep) if p.strip()]


def _vault_root_allowed(vault_root: str) -> tuple[bool, str]:
    """Check vault_root against the allowlist. (ok, reason)."""
    allow = _vault_allowlist()
    if not allow:
        return True, ""
    target = os.path.realpath(vault_root.strip())
    for root in allow:
        try:
            common = os.path.commonpath([target, root])
        except ValueError:
            continue
        if common == root:
            return True, ""
    return False, (f"vault_root {vault_root!r} is outside MIND_MEM_VAULT_ALLOWLIST")


@mcp_tool_observe
def agent_inject(query: str, agent: str = "generic", limit: int = 10) -> str:
    """Render a context snippet in the target agent's expected format."""
    from mind_mem.agent_bridge import KNOWN_AGENTS, AgentFormatter, UnknownAgentError

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(query, str) or not query.strip():
        return json.dumps({"error": "query must be a non-empty string"})
    if agent not in KNOWN_AGENTS:
        return json.dumps(
            {
                "error": f"unknown agent: {agent!r}",
                "valid": list(KNOWN_AGENTS),
            }
        )
    if not (1 <= limit <= 100):
        return json.dumps({"error": "limit must be in [1, 100]"})

    # Late import — recall is extracted in a later PR step; the deferred
    # lookup keeps the two extractions independently committable.
    from mind_mem.mcp_server import _recall_impl

    raw = json.loads(_recall_impl(query, limit=limit))
    if isinstance(raw, dict):
        results = raw.get("results", []) or []
    elif isinstance(raw, list):
        results = raw
    else:
        results = []

    fmt = AgentFormatter(max_blocks=limit)
    try:
        text = fmt.inject(agent, query, results)
    except UnknownAgentError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(
        {"agent": agent, "query": query, "snippet": text, "_schema_version": "1.0"},
        indent=2,
    )


@mcp_tool_observe
def vault_scan(vault_root: str, sync_dirs: str = "") -> str:
    """Walk an Obsidian-style vault and return parsed VaultBlocks (JSON)."""
    from mind_mem.agent_bridge import VaultBridge

    if not isinstance(vault_root, str) or not vault_root.strip():
        return json.dumps({"error": "vault_root must be a non-empty string"})
    ok, reason = _vault_root_allowed(vault_root)
    if not ok:
        return json.dumps({"error": reason})
    dirs = [d.strip() for d in sync_dirs.split(",") if d.strip()] or None
    try:
        bridge = VaultBridge(vault_root=vault_root.strip())
        blocks = bridge.scan(sync_dirs=dirs)
    except (FileNotFoundError, ValueError) as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(
        {
            "vault_root": vault_root,
            "blocks": [b.as_dict() for b in blocks],
            "_schema_version": "1.0",
        },
        indent=2,
    )


@mcp_tool_observe
def vault_sync(
    vault_root: str,
    block_id: str,
    relative_path: str,
    body: str,
    block_type: str = "note",
    title: str = "",
    overwrite: bool = False,
    include_links: bool = False,
) -> str:
    """Write a single block back into a vault at a relative path.

    When *include_links* is ``True`` and the workspace contains a
    KnowledgeGraph database, outgoing edges from *block_id* are appended
    as an Obsidian ``## Links`` section so Obsidian's graph view can
    visualise memory relationships without manual link authoring.
    """
    from mind_mem.agent_bridge import VaultBlock, VaultBridge

    for arg, label in (
        (vault_root, "vault_root"),
        (block_id, "block_id"),
        (relative_path, "relative_path"),
    ):
        if not isinstance(arg, str) or not arg.strip():
            return json.dumps({"error": f"{label} must be a non-empty string"})
    ok, reason = _vault_root_allowed(vault_root)
    if not ok:
        return json.dumps({"error": reason})

    # Resolve KG path only when include_links is requested.
    kg_path: str | None = None
    if include_links:
        ws = _workspace()
        if ws:
            candidate = os.path.join(ws, "knowledge_graph.db")
            if os.path.isfile(candidate):
                kg_path = candidate

    try:
        bridge = VaultBridge(vault_root=vault_root.strip())
        target = bridge.write(
            VaultBlock(
                relative_path=relative_path.strip(),
                block_id=block_id.strip(),
                block_type=block_type.strip() or "note",
                title=title.strip() or block_id.strip(),
                body=body,
            ),
            overwrite=bool(overwrite),
            kg_path=kg_path,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(
        {"written": target, "links_included": kg_path is not None, "_schema_version": "1.0"},
        indent=2,
    )


@mcp_tool_observe
def stream_status() -> str:
    """Current change-stream publish / delivery / drop counters."""
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err
    return json.dumps(
        {**_change_stream().stats().as_dict(), "_schema_version": "1.0"},
        indent=2,
    )


def register(mcp) -> None:
    """Wire the agent tools onto *mcp*."""
    mcp.tool(agent_inject)
    mcp.tool(vault_scan)
    mcp.tool(vault_sync)
    mcp.tool(stream_status)
