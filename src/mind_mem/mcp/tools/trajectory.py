"""Trajectory-memory MCP tool — case-based recall over past task outcomes.

The other half of the loop that ``report_outcome`` opens. That tool records
whether recalled blocks *helped*; with the v4 ``trajectory`` flag on it also
mirrors each verdict into a ``TRAJ-`` sidecar under ``<ws>/trajectories/``.
``similar_trajectories`` reads that sidecar back: given the task you are
about to attempt, what happened the last few times something like it was
attempted, and how did it end?

Three properties this tool is answerable for:

* **Flag-gated, default-OFF.** With the flag off it returns an error and
  touches nothing — no directory listing, no probe that logs.
* **Deterministic.** The recency half-life is a function of *when you ask*,
  so the instant is resolved once at this boundary and threaded into every
  comparison; pass ``scoring_instant`` and the call is clock-free, and the
  answer carries the instant it scored against so a run replays exactly.
* **Admission-filtered.** ``mind_mem.trajectory.load_trajectories`` runs the
  parsed sidecar through ``admit_corpus``, so a quarantined, pending or
  unrecognised-status file in ``trajectories/`` is unreachable from here.
  Selecting a status is not filtering on it.

Read-only: this tool writes nothing, proposes nothing, and never touches the
governed corpus — hence USER scope in ``mcp/infra/acl.py``.
"""

from __future__ import annotations

import json

from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _workspace
from ._helpers import get_logger, metrics

_log = get_logger("mcp_server")

#: Returned verbatim while the flag is OFF, so a client can branch on it.
TRAJECTORY_DISABLED = (
    'mind-mem v4 surface \'trajectory\' is disabled. Enable via mind-mem.json: "v4": { "trajectory": { "enabled": true } }'
)

#: Hard ceiling on ``limit``, independent of the kernel's ``recall_limit``.
MAX_TRAJECTORY_RESULTS = 50


def _trajectory_enabled() -> bool:
    """True iff the v4 ``trajectory`` flag is ON, without emitting anything.

    ``feature_flags.is_enabled`` warns (``v4_config_unreadable``) on a
    malformed config, so using it here would make a flag-OFF build log a line
    the pre-wiring build never logged. See ``is_enabled_quiet``.
    """
    from mind_mem.trajectory import TRAJECTORY_FLAG
    from mind_mem.v4.feature_flags import is_enabled_quiet

    return is_enabled_quiet(TRAJECTORY_FLAG)


@mcp_tool_observe
def similar_trajectories(
    task: str,
    tools: str = "",
    outcome: str = "",
    limit: int = 0,
    scoring_instant: str = "",
) -> str:
    """Recall past task executions similar to the one you are about to attempt.

    Answers "how did this go last time?" from the trajectories captured by
    ``report_outcome``: the task text, the tools in play, how it ended, and
    any lesson recorded with it. Matches are scored on task-word overlap,
    tool overlap and outcome agreement, then discounted by an exponential
    recency half-life — a trajectory from two months ago scores well below
    the same trajectory from today, because stale procedure is worse than no
    procedure.

    Requires the v4 ``trajectory`` flag; returns an error and reads nothing
    while it is OFF.

    Args:
        task: What you are about to do.
        tools: Comma- or space-separated tools you expect to use.
        outcome: Bias toward trajectories that ended this way — ``SUCCESS``,
            ``FAILURE``, ``PARTIAL`` or ``ABORTED``.
        limit: Max results. ``0`` uses the ``recall_limit`` knob from
            ``mind/trajectory.mind``; the ceiling is 50 either way.
        scoring_instant: UTC ``YYYY-MM-DD`` the recency decay measures age
            from. Supply the instant an earlier run reported to replay its
            ranking exactly; omit it and today is used.

    Returns:
        JSON envelope with the ranked trajectories and the scoring instant
        they were ranked against.
    """
    if not _trajectory_enabled():
        return json.dumps({"_schema_version": MCP_SCHEMA_VERSION, "error": TRAJECTORY_DISABLED})

    if not isinstance(task, str) or not task.strip():
        return json.dumps({"_schema_version": MCP_SCHEMA_VERSION, "error": "task must be a non-empty string"})
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        return json.dumps({"_schema_version": MCP_SCHEMA_VERSION, "error": "limit must be a non-negative integer"})

    from mind_mem.scoring_instant import format_scoring_instant, resolve_scoring_instant
    from mind_mem.trajectory import similar_trajectories as _similar

    try:
        instant = resolve_scoring_instant(scoring_instant.strip() or None)
    except (TypeError, ValueError) as exc:
        return json.dumps({"_schema_version": MCP_SCHEMA_VERSION, "error": str(exc)})

    ws = _workspace()
    try:
        matches = _similar(
            ws,
            task.strip(),
            tools=tools if isinstance(tools, str) else "",
            outcome=outcome if isinstance(outcome, str) else "",
            limit=min(limit, MAX_TRAJECTORY_RESULTS) if limit else None,
            scoring_instant=instant,
        )
    except OSError as exc:
        _log.warning("similar_trajectories_failed", error=str(exc))
        return json.dumps({"_schema_version": MCP_SCHEMA_VERSION, "error": f"Failed to read trajectories: {exc}"})

    metrics.inc("mcp_similar_trajectories")
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "task": task.strip(),
            "scoring_instant": format_scoring_instant(instant),
            "count": len(matches),
            "trajectories": matches[:MAX_TRAJECTORY_RESULTS],
        },
        indent=2,
    )


def register(mcp) -> None:
    """Wire the trajectory tools onto *mcp*."""
    mcp.tool(similar_trajectories)
