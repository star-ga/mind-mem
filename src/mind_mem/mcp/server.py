"""FastMCP instance + ``main()`` entry point for the Mind-Mem MCP server.

This is the final seat of the v3.2.0 §1.2 decomposition
(docs/v3.2.0-mcp-decomposition-plan.md PR-final). Before this
step, both concerns lived inside the ``mcp_server.py`` monolith:

* construction of the ``FastMCP`` instance
* wiring every ``@mcp.resource`` + ``@mcp.tool`` onto it
* the CLI entry point (transport parsing, watcher, HTTP auth)

All three now live here; ``mcp_server.py`` is a deprecation-soft
re-export shim that keeps every historical import path working
while directing new code at ``mind_mem.mcp.*`` submodules.

Registration order matters only for resources vs tools (resources
are read-only, tools are mutating) — within each category the
order is cosmetic. We register resources first so tool logs
surface below them in trace output.
"""

from __future__ import annotations

from fastmcp import FastMCP

from mind_mem.mcp import resources as _resources
from mind_mem.mcp.infra.http_auth import _build_http_auth_tokens, _check_token, check_token_strength
from mind_mem.mcp.infra.workspace import _workspace
from mind_mem.mcp.tools import (
    agent as _tools_agent,
)
from mind_mem.mcp.tools import (
    audit as _tools_audit,
)
from mind_mem.mcp.tools import (
    benchmark as _tools_benchmark,
)
from mind_mem.mcp.tools import (
    calibration as _tools_calibration,
)
from mind_mem.mcp.tools import (
    consolidation as _tools_consolidation,
)
from mind_mem.mcp.tools import (
    core as _tools_core,
)
from mind_mem.mcp.tools import (
    encryption as _tools_encryption,
)
from mind_mem.mcp.tools import (
    governance as _tools_governance,
)
from mind_mem.mcp.tools import (
    graph as _tools_graph,
)
from mind_mem.mcp.tools import (
    kernels as _tools_kernels,
)
from mind_mem.mcp.tools import (
    memory_ops as _tools_memory_ops,
)
from mind_mem.mcp.tools import (
    ontology as _tools_ontology,
)
from mind_mem.mcp.tools import (
    recall as _tools_recall,
)
from mind_mem.mcp.tools import (
    signal as _tools_signal,
)
from mind_mem.observability import get_logger

_log = get_logger("mcp_server")


mcp = FastMCP(
    name="mind-mem",
    instructions=(
        "Mind-Mem: persistent, auditable, contradiction-safe memory for coding agents. "
        "Use recall to search memory. Use propose_update to suggest changes (never writes "
        "directly to source of truth). All proposals go through human review."
    ),
)


# Resources first, tools second — keeps trace output readable.
_resources.register(mcp)

_tools_recall.register(mcp)
_tools_audit.register(mcp)
_tools_core.register(mcp)
_tools_consolidation.register(mcp)
_tools_ontology.register(mcp)
_tools_agent.register(mcp)
_tools_graph.register(mcp)
_tools_signal.register(mcp)
_tools_governance.register(mcp)
_tools_encryption.register(mcp)
_tools_benchmark.register(mcp)
_tools_memory_ops.register(mcp)
_tools_kernels.register(mcp)
_tools_calibration.register(mcp)


def main() -> None:
    """Entry point for the MCP server (used by console_scripts and __main__)."""
    import argparse
    import os
    import warnings

    parser = argparse.ArgumentParser(description="Mind-Mem MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio", help="Transport protocol (default: stdio)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (only used with --transport http)")
    parser.add_argument("--token", default=None, help="Bearer token for HTTP auth (or set MIND_MEM_TOKEN env var)")
    parser.add_argument("--watch", action="store_true", help="Auto-reindex when workspace .md files change")
    parser.add_argument(
        "--watch-interval",
        type=float,
        default=5.0,
        help="File watch polling interval in seconds (default: 5.0)",
    )
    args = parser.parse_args()

    if args.token and not os.environ.get("MIND_MEM_TOKEN"):
        warnings.warn(
            "Passing --token on the command line exposes it in /proc/cmdline. "
            "Use MIND_MEM_TOKEN environment variable instead.",
            stacklevel=2,
        )
        os.environ["MIND_MEM_TOKEN"] = args.token

    if args.transport == "http" and _check_token() and not os.environ.get("MIND_MEM_ADMIN_TOKEN"):
        warnings.warn(
            "HTTP transport without MIND_MEM_ADMIN_TOKEN: authenticated clients only receive user scope. "
            "Set MIND_MEM_ADMIN_TOKEN to enable admin operations over HTTP.",
            stacklevel=2,
        )

    token = _check_token()
    for _warn in check_token_strength():
        _log.warning("weak_token_config", detail=_warn)
    _log.info(
        "mcp_server_start",
        transport=args.transport,
        workspace=_workspace(),
        auth="token" if token else "none",
    )

    if args.watch:
        from mind_mem.sqlite_index import build_index
        from mind_mem.watcher import FileWatcher

        ws = _workspace()

        def _on_changes(changed_files: set[str]) -> None:
            try:
                result = build_index(ws, incremental=True)
                _log.info(
                    "watch_reindex_complete",
                    blocks_new=result.get("blocks_new", 0),
                    blocks_modified=result.get("blocks_modified", 0),
                )
            except Exception as e:
                _log.warning("watch_reindex_failed", error=str(e))

        watcher = FileWatcher(ws, callback=_on_changes, interval=args.watch_interval)
        watcher.start()
        _log.info("file_watcher_enabled", interval=args.watch_interval)

    if args.transport == "http":
        auth_tokens = _build_http_auth_tokens()
        if not auth_tokens:
            _log.warning(
                "mcp_http_no_auth",
                hint="HTTP transport running without token auth. Set MIND_MEM_TOKEN or MIND_MEM_ADMIN_TOKEN for security.",
            )
        else:
            from fastmcp.server.auth import StaticTokenVerifier

            mcp.auth = StaticTokenVerifier(tokens=auth_tokens)
            _log.info("mcp_auth_enforced", mode="static_token", token_count=len(auth_tokens))
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
