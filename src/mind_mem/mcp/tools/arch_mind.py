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
import re
import shutil
import subprocess
from typing import Any

from ..infra.observability import mcp_tool_observe

# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------


class ArchMindError(Exception):
    """Raised when the arch-mind binary is missing or returns non-zero."""


# Audit S-6: every caller-supplied argument that lands in a subprocess
# argv must be validated. The MCP threat model treats tool callers as
# untrusted (a poisoned agent or compromised client could supply
# arguments crafted to flag-inject the arch-mind binary, or to escape
# the repository sandbox). Paths get a structural check (no flag-style
# prefix, no shell metacharacters, bounded length, no NUL byte) and
# enumerated string arguments get an allowlist or regex.

_ARCH_MODE_ALLOWLIST = frozenset({"enforce", "report"})
_ARCH_METRIC_ALLOWLIST = frozenset(
    {
        "modularity_q16",
        "acyclicity_q16",
        "depth_q16",
        "equality_q16",
        "redundancy_q16",
        "q16_determinism_purity",
        "evidence_chain_density",
        "mcp_tool_isolation",
        "governance_kernel_coverage",
    }
)
_ARCH_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")
_ARCH_PATH_MAX = 4096
_ARCH_HISTORY_DAYS_MAX = 36500  # ~100 years, generous upper bound


def _validate_arch_path(name: str, value: str) -> str:
    """Validate an arch-mind path argument before it reaches subprocess.

    The arch-mind binary takes absolute repository / fixture / rules /
    baseline paths. The wrapper does not interpret them relative to a
    workspace (they may legitimately point outside the mind-mem
    workspace) so we only enforce structural guards:

    * non-empty, str-typed, bounded length;
    * no NUL byte (defeats C-string truncation tricks);
    * does not start with ``-`` (otherwise the caller could craft an
      argument that arch-mind would parse as a flag).

    These guards keep subprocess-with-list-argv safe even if the caller
    is hostile. We deliberately do NOT realpath / stat here — that is
    arch-mind's job and depends on its own filesystem view.
    """
    if not isinstance(value, str):
        raise ArchMindError(f"arch-mind {name}: expected str, got {type(value).__name__}")
    if not value:
        raise ArchMindError(f"arch-mind {name}: empty path is not allowed")
    if len(value) > _ARCH_PATH_MAX:
        raise ArchMindError(f"arch-mind {name}: path exceeds {_ARCH_PATH_MAX} chars")
    if "\x00" in value:
        raise ArchMindError(f"arch-mind {name}: NUL byte in path is not allowed")
    if value.startswith("-"):
        raise ArchMindError(f"arch-mind {name}: path may not start with '-' (flag-injection guard)")
    return value


def _validate_arch_id(name: str, value: str) -> str:
    """Validate an opaque identifier (agent_id, commit_sha) for subprocess.

    Allowed: ``^[A-Za-z0-9_.\\-]{1,128}$``. Any character outside that
    class is rejected so a hostile caller cannot smuggle whitespace,
    quote, or flag-prefix bytes into the arch-mind argv.
    """
    if not isinstance(value, str):
        raise ArchMindError(f"arch-mind {name}: expected str, got {type(value).__name__}")
    if not _ARCH_ID_RE.match(value):
        raise ArchMindError(f"arch-mind {name}: must match ^[A-Za-z0-9_.\\-]{{1,128}}$")
    return value


def _validate_arch_mode(value: str) -> str:
    if value not in _ARCH_MODE_ALLOWLIST:
        raise ArchMindError(f"arch-mind mode: must be one of {sorted(_ARCH_MODE_ALLOWLIST)}, got {value!r}")
    return value


def _validate_arch_metric(value: str) -> str:
    if value not in _ARCH_METRIC_ALLOWLIST:
        raise ArchMindError(f"arch-mind metric: must be one of {sorted(_ARCH_METRIC_ALLOWLIST)}, got {value!r}")
    return value


def _validate_arch_days(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArchMindError(f"arch-mind days: expected int, got {type(value).__name__}")
    if value < 0 or value > _ARCH_HISTORY_DAYS_MAX:
        raise ArchMindError(f"arch-mind days: must be in [0, {_ARCH_HISTORY_DAYS_MAX}], got {value}")
    return value


def _resolve_binary() -> str:
    candidate = os.environ.get("ARCH_MIND_BIN")
    if candidate:
        return candidate
    found = shutil.which("arch-mind")
    if found:
        return found
    raise ArchMindError("arch-mind binary not found. Set ARCH_MIND_BIN or place it on PATH.")


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
    repo = _validate_arch_path("repo", repo)
    fixture = _validate_arch_path("fixture", fixture)
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
    # repo is still validated even though it is unused, so a poisoned
    # caller cannot stash flag-injection payloads in the "unused" arg
    # and silently bypass the audit S-6 contract once v0.1.1 wires it in.
    _validate_arch_path("repo", repo)
    before = _validate_arch_path("before", before)
    after = _validate_arch_path("after", after)
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
    repo = _validate_arch_path("repo", repo)
    days = _validate_arch_days(days)
    args = ["history", "--repo", repo]
    if days > 0:
        args += ["--days", str(days)]
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
    repo = _validate_arch_path("repo", repo)
    fixture = _validate_arch_path("fixture", fixture)
    mode = _validate_arch_mode(mode)
    args = ["check-rules", "--repo", repo, "--fixture", fixture, "--mode", mode]
    if rules is not None:
        rules = _validate_arch_path("rules", rules)
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
    repo = _validate_arch_path("repo", repo)
    fixture = _validate_arch_path("fixture", fixture)
    agent_id = _validate_arch_id("agent_id", agent_id)
    commit_sha = _validate_arch_id("commit_sha", commit_sha)
    return _run(
        [
            "session-start",
            "--repo",
            repo,
            "--fixture",
            fixture,
            "--agent",
            agent_id,
            "--commit",
            commit_sha,
        ]
    )


@mcp_tool_observe
def arch_session_end(repo: str, fixture: str) -> dict[str, Any]:
    """Close the open arch-mind session at *repo*.

    Writes a ``session_end`` evidence node chained to the most recent
    ``session_start``. Returns the per-metric delta and the
    ``any_regression`` boolean.
    """
    repo = _validate_arch_path("repo", repo)
    fixture = _validate_arch_path("fixture", fixture)
    return _run(
        [
            "session-end",
            "--repo",
            repo,
            "--fixture",
            fixture,
        ]
    )


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
    metric = _validate_arch_metric(metric)
    fixture = _validate_arch_path("fixture", fixture)
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
