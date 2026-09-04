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
from ._helpers import _change_stream, _kg_path


def _backpressure_snapshot() -> dict:
    """Per-producer backpressure state, or ``{}`` when the flag is off.

    Late import so the tool module keeps its current import cost, and so
    a v4 module is not pulled in by a server that never turns it on.
    """
    from mind_mem.v4.backpressure import snapshot

    return snapshot()


def _ingest_door_snapshot() -> dict:
    """Streaming ingest-door counters, or ``{}`` when no door is open.

    Late import for the same reason as the backpressure probe above: a
    server that never turns ``streaming.enabled`` on should not pay for the
    module. ``build_stream_door`` registers the door; with the flag off
    nothing is registered and this returns ``{}``.
    """
    from mind_mem.streaming import stream_door_snapshot

    return stream_door_snapshot() or {}


def _vault_allowlist() -> list[str]:
    """Return the configured vault-root allowlist.

    Set ``MIND_MEM_VAULT_ALLOWLIST`` to a ``:``-separated list of
    absolute directories. ``vault_scan`` / ``vault_sync`` reject every
    request when the allowlist is empty (issue #509 / T-006: prevents
    arbitrary host markdown exfil via ``vault_root=/etc``). Operators
    who need legacy open behaviour set
    ``MIND_MEM_VAULT_ALLOW_ANY=true`` (not recommended; documented in
    SECURITY.md).
    """
    raw = os.environ.get("MIND_MEM_VAULT_ALLOWLIST", "").strip()
    if not raw:
        return []
    # Split on os.pathsep plus ";".
    #
    # The old rule was `";" if ";" in raw else ":"`, which silently destroys
    # every Windows path: `C:\\Users\\me\\vault` has no semicolon, so it split on
    # the DRIVE-LETTER colon into ["C", "\\Users\\me\\vault"] and the allowlist
    # then matched nothing -- every vault call on Windows was refused as
    # "outside MIND_MEM_VAULT_ALLOWLIST", including one pointing at the
    # allowlisted directory itself.
    #
    # os.pathsep is the platform's own answer (";" on Windows, ":" on POSIX),
    # so drive letters are safe there. ";" stays accepted everywhere so a POSIX
    # operator who already writes the list that way is not broken by the fix.
    seps = {os.pathsep, ";"}
    parts: list[str] = []
    current: list[str] = []
    for ch in raw:
        if ch in seps:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [os.path.realpath(part.strip()) for part in parts if part.strip()]


def _vault_allow_any() -> bool:
    return os.environ.get("MIND_MEM_VAULT_ALLOW_ANY", "").lower() in ("1", "true", "yes")


def _vault_root_allowed(vault_root: str) -> tuple[bool, str]:
    """Check vault_root against the allowlist. (ok, reason)."""
    allow = _vault_allowlist()
    if not allow:
        if _vault_allow_any():
            return True, ""
        return False, (
            "vault tools refuse when MIND_MEM_VAULT_ALLOWLIST is empty "
            "(issue #509 / T-006). Set MIND_MEM_VAULT_ALLOWLIST=/path/to/vault, "
            "or MIND_MEM_VAULT_ALLOW_ANY=true for the legacy open behaviour."
        )
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
def agent_inject(query: str, agent: str = "generic", limit: int = 10, scoring_instant: str = "") -> str:
    """Render a context snippet in the target agent's expected format.

    ``scoring_instant`` is an ISO-8601 UTC date pinning the recency layer of the
    recall underneath; empty means today in UTC. Two agents handed the same
    instant get the same snippet.

    The envelope carries the recall's own ``attestation`` — the
    ``RECALL_ATTEST_v2`` record committing to the run that produced this
    snippet. It is the same record ``recall`` publishes, forwarded rather than
    re-derived, so the two surfaces cannot disagree about what was served.
    Without it this tool was the one door that handed an agent block content
    with no receipt: the snippet went into a system prompt and nothing
    downstream could say which recall, at which index anchor, produced it.
    ``None`` when the recall underneath produced no record (an error envelope,
    or an anticipation-cache hit, which deliberately does not fabricate one) —
    absent evidence is reported as absent, never invented.
    """
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

    raw = json.loads(_recall_impl(query, limit=limit, scoring_instant=scoring_instant or None))
    attestation = None
    if isinstance(raw, dict):
        results = raw.get("results", []) or []
        attestation = raw.get("attestation")
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
        {
            "agent": agent,
            "query": query,
            "snippet": text,
            "attestation": attestation,
            "_schema_version": "1.0",
        },
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

    # Resolve KG path only when include_links is requested. The DB lives
    # at <ws>/memory/knowledge_graph.db -- use the shared helper every
    # graph tool uses rather than re-deriving the path here, which is how
    # this looked for <ws>/knowledge_graph.db (a location the product
    # never writes) and silently produced link-free notes.
    kg_path: str | None = None
    if include_links:
        ws = _workspace()
        if ws:
            candidate = _kg_path(ws)
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
    """Current change-stream publish / delivery / drop counters.

    With ``v4.backpressure`` enabled the payload also carries a
    ``backpressure`` object: per-producer depth, watermarks, and whether
    the producer is currently overloaded. The key is ABSENT when the flag
    is off rather than present-and-empty, so a client can tell "nothing
    is measuring" from "measuring, and fine".

    With ``streaming.enabled`` the payload also carries an ``ingest_door``
    object: queue depth and capacity, the per-client 429 count, and how
    many streamed events have been admitted to the store QUARANTINED. Same
    absent-when-off rule as ``backpressure``, so with both flags off -- the
    default -- this payload is byte-identical to the pre-5.0.1 one.

    No new tool, and no new ACL row: both objects are queue telemetry about
    buses this tool already reports on. They carry counters and watermarks
    only -- never block ids, never block content -- so they stay exactly as
    sensitive as the counters beside them.
    """
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err
    stream = _change_stream()
    payload: dict = {**stream.stats().as_dict(), "_schema_version": "1.0"}
    bp = _backpressure_snapshot()
    if bp:
        payload["backpressure"] = bp
    door = _ingest_door_snapshot()
    if door:
        payload["ingest_door"] = door
    return json.dumps(payload, indent=2)


def register(mcp) -> None:
    """Wire the agent tools onto *mcp*."""
    mcp.tool(agent_inject)
    mcp.tool(vault_scan)
    mcp.tool(vault_sync)
    mcp.tool(stream_status)
