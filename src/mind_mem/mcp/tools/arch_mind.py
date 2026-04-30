"""arch-mind MCP tools — wraps the ``arch-mind`` binary as 7 MCP tools.

This module is the mind-mem side of arch-mind v0.1 Phase F. It
exposes the seven arch-mind subcommands the spec lists as MCP
tools so any client mounting mind-mem's MCP surface (Claude Code,
Codex CLI, Cursor, Windsurf, etc.) can call them.

Tool surface:

* ``arch_baseline`` — initialise the arch-mind store with a baseline.
* ``arch_delta`` — compute (current scan) - (baseline scores).
* ``arch_history`` — list events in the arch-mind store.
* ``arch_check_rules`` — apply a rules.mind to a fresh scan.
* ``arch_session_start`` — open a session evidence node.
* ``arch_session_end`` — close the session, write delta evidence.
* ``arch_metric_explain`` — per-metric breakdown for a fixture.

Every tool shells out to the ``arch-mind`` binary; the wrapper does
no metric arithmetic of its own. arch-mind itself enforces the
canonical AST schema and Q16.16 determinism contract; mind-mem
trusts and re-exposes the binary.

The binary is located via the ``ARCH_MIND_BIN`` environment variable
(falls back to looking up ``arch-mind`` on ``PATH``). Set this in
the mind-mem deployment env when arch-mind ships from a non-standard
location.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any

from ..infra.observability import mcp_tool_observe

# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------


class ArchMindError(Exception):
    """Raised when the arch-mind binary is missing or returns non-zero."""


def _resolve_binary() -> str:
    candidate = os.environ.get("ARCH_MIND_BIN")
    if candidate:
        return candidate
    found = shutil.which("arch-mind")
    if found:
        return found
    raise ArchMindError(
        "arch-mind binary not found. Set ARCH_MIND_BIN or place it on PATH."
    )


def _run(args: list[str], *, timeout: float = 60.0) -> dict[str, Any]:
    """Invoke ``arch-mind <args>`` and return a structured result.

    Output convention:
        {
          "ok":   bool,                # process exit code == 0
          "code": int,                 # exit code
          "stdout": str,
          "stderr": str,
          "json": dict | list | None,  # parsed stdout if it looked like JSON
        }
    """
    binary = _resolve_binary()
    cmd = [binary, *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise ArchMindError(f"arch-mind timed out after {timeout}s: {' '.join(cmd)}")
    parsed: Any = None
    out = proc.stdout.strip()
    if out and (out.startswith("{") or out.startswith("[")):
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            parsed = None
    return {
        "ok": proc.returncode == 0,
        "code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "json": parsed,
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp_tool_observe
def arch_baseline(repo: str, fixture: str) -> dict[str, Any]:
    """Initialise the arch-mind store at *repo* with a baseline event.

    Args:
        repo: absolute path to the repository whose architectural
            baseline should be captured.
        fixture: absolute path to a v0.0.1-format ``_aggregated_for_phase_a``
            fixture (typically the output of ``arch-mind sidecar-scan``).

    Returns the wrapped subprocess result; ``json`` carries the
    arch-mind binary's stdout when it is parseable JSON.
    """
    return _run(["baseline", "--fixture", fixture, "--out", f"{repo}/.arch-mind/baseline.json"])


@mcp_tool_observe
def arch_delta(repo: str, before: str, after: str) -> dict[str, Any]:
    """Compute the per-metric delta between two arch-mind baselines.

    Args:
        repo: absolute path to the repository (currently unused by
            ``arch-mind delta`` but kept for symmetry with the other
            tools and for the eventual store-aware delta lookup).
        before: path to the baseline JSON captured before the change.
        after: path to the baseline JSON captured after the change.
    """
    del repo  # explicitly ignored; reserved for v0.1.1 store-aware deltas
    return _run(["delta", "--before", before, "--after", after])


@mcp_tool_observe
def arch_history(repo: str, days: int = 0, verify: bool = False) -> dict[str, Any]:
    """List events in the arch-mind store at *repo*.

    Args:
        repo: absolute path to the repository.
        days: 0 (default) returns the full history; ``> 0`` filters to
            events whose ``wall_clock_iso`` falls within the last N days.
        verify: if true, recompute every MAC and report tamper.
    """
    args = ["history", "--repo", repo]
    if days > 0:
        args += ["--days", str(int(days))]
    if verify:
        args.append("--verify")
    return _run(args)


@mcp_tool_observe
def arch_check_rules(
    repo: str,
    fixture: str,
    rules: str | None = None,
    mode: str = "report",
) -> dict[str, Any]:
    """Apply a ``rules.mind`` to a fresh arch-mind scan.

    Args:
        repo: absolute path to the repository.
        fixture: path to an arch-mind fixture (output of
            ``arch_baseline`` or a fresh ``sidecar-scan``).
        rules: path to ``.arch-mind/rules.mind``; defaults to
            ``<repo>/.arch-mind/rules.mind``.
        mode: ``"enforce"`` (exit 1 on any violation) or ``"report"``
            (always exit 0, return violations in the result).
    """
    args = ["check-rules", "--repo", repo, "--fixture", fixture, "--mode", mode]
    if rules:
        args += ["--rules", rules]
    return _run(args)


@mcp_tool_observe
def arch_session_start(
    repo: str,
    fixture: str,
    agent_id: str,
    commit_sha: str,
) -> dict[str, Any]:
    """Open an arch-mind session evidence node.

    Returns the binary's stdout including the new ``baseline_id``
    (the truncated HMAC of the session_start event). Pair every
    successful call with exactly one ``arch_session_end`` to close
    the chain.
    """
    return _run([
        "session-start",
        "--repo", repo,
        "--fixture", fixture,
        "--agent", agent_id,
        "--commit", commit_sha,
    ])


@mcp_tool_observe
def arch_session_end(repo: str, fixture: str) -> dict[str, Any]:
    """Close the open arch-mind session at *repo*.

    Writes a ``session_end`` evidence node chained to the most recent
    ``session_start``. Returns the per-metric delta and the
    ``any_regression`` boolean.
    """
    return _run([
        "session-end",
        "--repo", repo,
        "--fixture", fixture,
    ])


@mcp_tool_observe
def arch_metric_explain(metric: str, fixture: str) -> dict[str, Any]:
    """Per-metric human-readable breakdown for a given fixture.

    Args:
        metric: one of the nine arch-mind kernel names
            (``modularity_q16``, ``acyclicity_q16``, ``depth_q16``,
            ``equality_q16``, ``redundancy_q16``,
            ``q16_determinism_purity``, ``evidence_chain_density``,
            ``mcp_tool_isolation``, ``governance_kernel_coverage``).
        fixture: path to an arch-mind fixture.
    """
    # arch-mind's CLI version of metric_explain is "explain --scan",
    # which prints all 9 metrics. The MCP tool filters to the one
    # the caller asked about so the response is small and focused.
    result = _run(["explain", "--scan", fixture])
    if not result["ok"]:
        return result

    # Extract the requested metric's line from explain's pretty output.
    lines = result["stdout"].splitlines()
    selected = [line for line in lines if metric in line]
    return {
        **result,
        "metric": metric,
        "metric_lines": selected,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(mcp) -> None:
    """Wire the seven arch-mind tools onto *mcp* (a ``FastMCP`` instance)."""
    mcp.tool(arch_baseline)
    mcp.tool(arch_delta)
    mcp.tool(arch_history)
    mcp.tool(arch_check_rules)
    mcp.tool(arch_session_start)
    mcp.tool(arch_session_end)
    mcp.tool(arch_metric_explain)
