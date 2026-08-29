"""Drive both arms over a task set and produce one number with its uncertainty.

THE QUESTION
------------
Does an agent with mind-mem complete real repository tasks better than the
same agent without it, at the same budget?

HOW THE ARMS ARE KEPT HONEST
----------------------------
Both arms run **in the same directory, under the same environment object**.
The work tree is deleted and re-extracted from ``parent_sha`` between arms
and the sandbox ``HOME`` is wiped with it, so nothing carries over and the
two runs cannot be told apart by their paths.  Environment equality is
therefore identity, not a comparison that could drift.  The arms are run
in a fixed order (control, then memory) which is stated rather than
randomised, because with a wiped arena there is nothing for order to
influence and a stated constant is easier to audit than a seed.

WHAT IS DETERMINISTIC AND WHAT IS NOT
-------------------------------------
Everything the harness owns is: task selection, seeding, prompt
construction, budget accounting, grading and the statistics.  Two runs
over the same inputs with the same adapter produce a byte-identical
``digest``.  The *adapter* may not be -- an external agent is a stochastic
component and pretending otherwise would be a lie -- so its timing is kept
out of the digest entirely (see :data:`TELEMETRY_KEY`) and the ``none``
adapter exists to prove the surrounding machinery.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .ab_agent import AgentError, AgentRequest, AgentResult
from .ab_arms import (
    ARM_CONTROL,
    ARM_MEMORY,
    Budget,
    PromptBuild,
    assert_arms_equal,
    assert_control_isolated,
    assert_tree_has_no_corpus,
    build_env,
    build_prompt,
    scan_tree_for_memory_pointers,
)
from .ab_grade import Verdict, grade, snapshot_tree
from .ab_seed import SeedReport, seed_workspace
from .ab_stats import summarise
from .ab_task import Task
from .repo_task_validation import apply_test_patch, extract_tree

SCHEMA_VERSION = 1

#: Section of the artifact excluded from the digest: wall-clock and other
#: per-run measurements that are recorded for the reader but must never
#: decide a score.  This is the whole of the harness's clock exposure.
TELEMETRY_KEY = "telemetry"

Agent = Callable[[AgentRequest], AgentResult]

#: Faults that mean "this task could not be attempted", never "this arm
#: failed". Tree extraction raises ``RuntimeError`` when ``git archive``
#: cannot complete, which on a shared box happens for reasons that have
#: nothing to do with either arm.
SETUP_FAILURES = (AgentError, OSError, RuntimeError)


@dataclass(frozen=True)
class ArmRun:
    """One arm's attempt at one task: what it was given, spent and scored."""

    build: PromptBuild
    result: AgentResult
    verdict: Verdict
    wall_seconds: float
    memory_pointers: tuple[str, ...] = ()

    def scored(self) -> dict[str, Any]:
        return {"prompt": self.build.as_dict(), "agent": self.result.as_dict(), "verdict": self.verdict.as_dict()}


def _discard(*paths: str) -> None:
    """Remove a task's scratch state. A seeded corpus does not outlive its run."""
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)


def _fresh_arena(repo: str, task: Task, arena: str) -> tuple[str, str]:
    """Wipe and rebuild the shared arena so both arms start identically."""
    tree, home = os.path.join(arena, "tree"), os.path.join(arena, "home")
    shutil.rmtree(tree, ignore_errors=True)
    shutil.rmtree(home, ignore_errors=True)
    extract_tree(repo, task.parent_sha, tree)
    apply_test_patch(repo, task.sha, task.test_patch_paths, tree)
    os.makedirs(home, exist_ok=True)
    assert_tree_has_no_corpus(tree)
    return tree, home


def run_arm(
    repo: str,
    task: Task,
    arm: str,
    build: PromptBuild,
    agent: Agent,
    arena: str,
    env: Mapping[str, str],
    budget: Budget,
    python: str,
) -> ArmRun:
    """Rebuild the arena, let the adapter attempt the task, then grade it."""
    tree, home = _fresh_arena(repo, task, arena)
    pointers = scan_tree_for_memory_pointers(tree)
    before = snapshot_tree(tree)
    request = AgentRequest(
        task_id=task.task_id,
        arm=arm,
        prompt=build.prompt,
        tree=tree,
        env=env,
        wall_seconds=budget.wall_seconds,
        output_tokens=budget.output_tokens,
        steps=budget.steps,
    )
    started = time.monotonic()
    result = agent(request)
    elapsed = time.monotonic() - started
    verdict = grade(task, tree, home, python, before, timeout=budget.wall_seconds)
    return ArmRun(build=build, result=result, verdict=verdict, wall_seconds=round(elapsed, 2), memory_pointers=pointers)


def run_task(
    repo: str,
    task: Task,
    agent: Agent,
    workdir: str,
    budget: Budget,
    python: str,
    passthrough: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run both arms over one task and return its record (or its exclusion)."""
    arena, workspace = os.path.join(workdir, task.task_id), os.path.join(workdir, f"{task.task_id}__ws")
    tree, home = os.path.join(arena, "tree"), os.path.join(arena, "home")
    seed: SeedReport = seed_workspace(repo, task, workspace)
    env = build_env(tree, home, passthrough)
    isolation = assert_control_isolated(env, tree, workspace)
    control = build_prompt(task, ARM_CONTROL, budget)
    memory = build_prompt(task, ARM_MEMORY, budget, workspace)
    if control.over_budget or memory.over_budget:
        _discard(arena, workspace)
        return _excluded(task, "prompt_over_budget", {"control": control.prompt_tokens, "memory": memory.prompt_tokens})
    equality = assert_arms_equal(memory, control, budget)
    try:
        runs = {
            ARM_CONTROL: run_arm(repo, task, ARM_CONTROL, control, agent, arena, env, budget, python),
            ARM_MEMORY: run_arm(repo, task, ARM_MEMORY, memory, agent, arena, env, budget, python),
        }
    except SETUP_FAILURES as exc:
        # One task that cannot be set up must not poison its siblings, and
        # it must not vanish either: it is reported with its reason and
        # counted in ``excluded``. Observed in practice when the scratch
        # filesystem filled and ``git archive`` could not materialise the
        # tree -- an infrastructure fault, which must read as an exclusion
        # rather than as an arm that failed the task.
        _discard(arena, workspace)
        return _excluded(task, "setup_failed", {"error": f"{type(exc).__name__}: {exc}"})
    _discard(arena, workspace)
    return _record(task, seed, isolation, equality, runs)


def _record(task: Task, seed: SeedReport, isolation: Sequence[str], equality: Sequence[str], runs: Mapping[str, ArmRun]) -> dict[str, Any]:
    """Assemble one task's scored record plus its clock-bearing telemetry."""
    return {
        "task_id": task.task_id,
        "sha": task.sha,
        "parent_sha": task.parent_sha,
        "tier": task.tier,
        "size_bucket": task.size_bucket,
        "excluded": None,
        "seed": {
            "blocks": seed.blocks,
            "corpus_bytes": seed.corpus_bytes,
            "cutoff": seed.cutoff,
            "newest_seeded_at": seed.newest_seeded_at,
            "leak_checks_passed": list(seed.checks),
            "skipped_malformed": seed.skipped_malformed,
        },
        "invariants": {
            "control_isolation": [*isolation, "no_corpus_inside_work_tree"],
            "arm_equality": list(equality),
            "environment": "single_shared_object",
            "memory_pointers_in_tree": _pointer_record(runs),
        },
        "arms": {arm: run.scored() for arm, run in runs.items()},
        "outcome": {
            "memory_success": runs[ARM_MEMORY].verdict.success,
            "control_success": runs[ARM_CONTROL].verdict.success,
        },
        # An adapter that never launched -- a missing credential, a wrong
        # argv -- also produces "both arms failed", which reads exactly like
        # a legitimate null result. Counting the files each arm actually
        # touched separates the two, so an inert agent is visible instead of
        # being published as evidence that memory did not help.
        "agent_effect": {
            arm: {
                "changed_files": len(run.verdict.changed_paths),
                "returncode": run.result.returncode,
                "timed_out": run.result.timed_out,
            }
            for arm, run in runs.items()
        },
        TELEMETRY_KEY: {arm: {"wall_seconds": run.wall_seconds} for arm, run in runs.items()},
    }


def _pointer_record(runs: Mapping[str, ArmRun]) -> dict[str, Any]:
    """Files in the tree that reference memory, recorded once with the invariant.

    Both arms get a tree extracted from the same commit, so these lists are
    equal by construction and publishing two copies would only ask a reader
    to diff them.  The equality is stated instead -- and if it ever fails,
    that is a bug, so the per-arm lists are published in that case only.
    """
    lists = {arm: list(run.memory_pointers) for arm, run in runs.items()}
    if len({tuple(paths) for paths in lists.values()}) != 1:  # pragma: no cover - bug path
        return {"identical_across_arms": False, "per_arm": lists}
    files = next(iter(lists.values()))
    return {
        "identical_across_arms": True,
        "count": len(files),
        "files": files,
        "why_this_is_not_a_leak": (
            "Same tree in both arms. mm / mind-mem-* are off PATH and HOME is a sandbox in both, so an agent that "
            "follows one of these finds nothing; they are named because an isolation claim should not be approximate."
        ),
    }


def _excluded(task: Task, reason: str, detail: Mapping[str, Any]) -> dict[str, Any]:
    """A task that could not be run is reported, never quietly dropped."""
    return {"task_id": task.task_id, "sha": task.sha, "excluded": reason, "detail": dict(detail), TELEMETRY_KEY: {}}


def aggregate_spend(scored: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Total what each arm actually spent, so an unequal one is visible.

    An arm that wins by spending more has proven nothing, so the totals sit
    next to the delta rather than buried in the per-task rows.  Output
    tokens are a lower bound for an external agent (only what it printed is
    observable), and that is what the key says.
    """
    totals: dict[str, dict[str, int]] = {}
    for arm in (ARM_CONTROL, ARM_MEMORY):
        rows = [r["arms"][arm] for r in scored if arm in r.get("arms", {})]
        totals[arm] = {
            "prompt_tokens": sum(int(row["prompt"]["prompt_tokens"]) for row in rows),
            "memory_tokens": sum(int(row["prompt"]["memory_tokens"]) for row in rows),
            "agent_output_tokens_lower_bound": sum(int(row["agent"]["output_tokens"]) for row in rows),
            "timeouts": sum(1 for row in rows if row["agent"]["timed_out"]),
        }
    return totals


def strip_telemetry(payload: Any) -> Any:
    """Recursively drop every telemetry section, leaving the scored record."""
    if isinstance(payload, dict):
        return {k: strip_telemetry(v) for k, v in payload.items() if k != TELEMETRY_KEY}
    if isinstance(payload, list):
        return [strip_telemetry(v) for v in payload]
    return payload


def digest(payload: Mapping[str, Any]) -> str:
    """sha256 over the scored record only. No wall-clock reaches this."""
    canonical = json.dumps(strip_telemetry(dict(payload)), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    """Digest of the task set actually used, so a run names its input."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def run_suite(
    repo: str,
    tasks: Sequence[Task],
    agent: Agent,
    workdir: str,
    budget: Budget,
    python: str,
    meta: Mapping[str, Any],
    passthrough: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run every task, summarise the paired delta, and assemble the artifact."""
    records = [run_task(repo, task, agent, workdir, budget, python, passthrough) for task in tasks]
    scored = [r for r in records if not r["excluded"]]
    excluded = [r for r in records if r["excluded"]]
    summary = summarise([(r["outcome"]["memory_success"], r["outcome"]["control_success"]) for r in scored])
    inert = [r["task_id"] for r in scored if not any(a["changed_files"] for a in r["agent_effect"].values())]
    spend = aggregate_spend(scored)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "harness": HARNESS_DOC,
        "budget": budget.as_dict(),
        **dict(meta),
        "counts": {
            "selected": len(tasks),
            "scored": len(scored),
            "excluded": len(excluded),
            "agent_inert": len(inert),
            "agent_inert_task_ids": inert,
        },
        "spend": spend,
        "excluded": excluded,
        "results": records,
        "summary": summary.as_dict(),
    }
    payload["digest"] = digest(payload)
    return payload


#: Written into every artifact so the file states its own design.
HARNESS_DOC: dict[str, str] = {
    "question": "Does the same agent, at the same budget, complete more real repository tasks with mind-mem than without it?",
    "differs_between_arms": (
        "One thing: a recalled-context prefix on the prompt. Asserted as memory.prompt == memory.memory_section + control.prompt."
    ),
    "held_constant": (
        "Task, reported-issue statement, instruction footer, agent and argv, work tree (re-extracted from parent_sha "
        "between arms), environment (one shared object, not a comparison), token/step/wall ceilings, grading command."
    ),
    "memory_gate": (
        "Structural, because this package ships no recall kill-switch: no MIND_MEM_* variable in the environment, no "
        "mm/mind-mem-* executable on PATH, a sandboxed HOME so no agent config or MCP server loads, and the seeded "
        "workspace placed outside the work tree."
    ),
    "seeding": (
        "The memory arm is seeded only from commits reachable from parent_sha -- a git ancestry property, so the task's "
        "own commit and everything after it are unreachable by construction -- re-checked against memory_cutoff and "
        "scanned for the task commit id."
    ),
    "grading": "pytest only. A task passes iff pytest exits 0 and every fail_to_pass node passes. Editing any test voids the attempt.",
    "statistics": ("Paired McNemar exact, two-sided. Below 6 discordant pairs no split can reach p<=0.05; such a delta is noise."),
    "determinism": "digest covers the scored record with every telemetry section removed, so no wall-clock value can move it.",
}
