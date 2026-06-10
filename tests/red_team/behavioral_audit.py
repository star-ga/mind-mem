"""Behavioral audit scaffold for the mind-mem MCP surface.

Runs three inspect_petri seeds against the live mcp_server.py via the
Dish runtime (Petri 3.0).  The test is tagged ``behavioral`` so the
default ``pytest`` invocation (``not stress``) does NOT collect it.

Run manually::

    pip install -e ".[red-team]"
    pytest -m behavioral -v

The test is advisory: a fail is a signal to investigate, not a hard
gate.  CI should run it on a separate optional workflow with
``continue-on-error: true`` until the team decides to promote it.

Seeds exercised
---------------
self_exfiltration_memory_trigger
    Probes whether the MCP server can be tricked into leaking
    workspace content (block text, file paths, tokens) via a crafted
    recall or propose_update call.

broken_tool_error_handling
    Passes malformed arguments to every registered tool and asserts
    that the server returns structured errors rather than tracebacks
    or silent corruption.

weird_ood_tool_use
    Sends out-of-distribution queries (emoji floods, null bytes,
    adversarial Unicode) and checks that the server neither crashes
    nor returns hallucinated tool names outside the known 84-tool
    surface.
"""

from __future__ import annotations

import pytest

petri = pytest.importorskip(
    "inspect_petri",
    reason=("inspect_petri is not installed.  Run: pip install -e '.[red-team]' to enable behavioral audits."),
)


@pytest.mark.behavioral
def test_advisory_petri_audit(petri_limit: int) -> None:
    """Run three Petri 3.0 seeds against the live MCP server (advisory).

    Sample count is controlled by ``--petri-limit`` (default 5).  A
    Sonnet-class judge is expected; estimated spend at limit=5: $10-15
    per run (see docs/red-team-audit.md).
    """
    seeds: list[str] = [
        "self_exfiltration_memory_trigger",
        "broken_tool_error_handling",
        "weird_ood_tool_use",
    ]

    dish = petri.Dish.from_mcp_server(
        "python3 mcp_server.py",
        transport="stdio",
    )

    results: list[dict] = []
    for seed in seeds:
        result = dish.run_seed(seed, limit=petri_limit)
        results.append({"seed": seed, "result": result})

    failures = [r for r in results if not r["result"].passed]
    if failures:
        detail = "\n".join(f"  {r['seed']}: {r['result'].summary}" for r in failures)
        pytest.fail(
            f"{len(failures)}/{len(seeds)} Petri seeds did not pass:\n{detail}\n"
            "This is an advisory audit — investigate before promoting to a hard gate."
        )
