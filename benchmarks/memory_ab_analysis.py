#!/usr/bin/env python3
"""Reduce repeated memory-A/B runs to one paired number, plus the counter-metric.

Three things the per-run scorecard cannot do, all of which change how a
number should be read:

**Reps.** One attempt per cell is one sample of a stochastic agent. A single
flipped task looks identical whether memory caused it or the agent was
having a different day. This reduces R repetitions per task to one verdict
per arm (``majority`` / ``all`` / ``any``) *before* the paired test sees
anything, and reports how often the reps disagreed with each other -- which
is the honest measure of whether R was large enough.

**Context poisoning.** A "memory helps" delta is only half the ledger. The
other half is how often recalled context made a task that the agent could
otherwise do *fail*::

    context_poisoning_rate = |passed without memory AND failed with memory|
                             ---------------------------------------------
                                      |passed without memory|

Its mirror, the rescue rate, is reported beside it. Both come from the same
paired cells the McNemar test uses, so neither can be quietly dropped.

**Disclosure.** ``agent_cutoff`` is recorded here because nothing else can
know it: if the agent's training data postdates the commits the tasks were
mined from, the agent may recall the fix regardless of the memory arm, and
that possibility has to travel with the number rather than being discovered
later.

Nothing here re-implements a statistic. The pooled test is
:func:`mind_mem.bench.ab_stats.summarise` -- the harness's own exact
McNemar -- handed one reduced pair per task.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mind_mem.bench.ab_stats import summarise  # noqa: E402

#: How R attempts in one cell become one verdict.
#: ``majority`` -- passed more often than not (ties are a FAIL: a coin-flip
#: task is not a pass). ``all`` -- every rep passed (pass^R, the strict
#: reading). ``any`` -- some rep passed (pass@R, the generous one).
REDUCERS = ("majority", "all", "any")


def reduce_cell(results: list[bool], how: str) -> bool:
    """Collapse one arm's repetitions for one task into a single verdict."""
    if how not in REDUCERS:
        raise ValueError(f"unknown reducer {how!r}; known: {REDUCERS}")
    if not results:
        raise ValueError("cannot reduce an empty cell")
    if how == "all":
        return all(results)
    if how == "any":
        return any(results)
    return sum(results) * 2 > len(results)


def load_runs(paths: list[str]) -> dict[str, dict[str, list[bool]]]:
    """Read per-task run artifacts into ``task_id -> arm -> [success, ...]``.

    Artifacts whose task was excluded, or whose agent made no edit at all,
    are dropped with the reason recorded by the caller: an inert agent
    measures the harness, not the memory.
    """
    cells: dict[str, dict[str, list[bool]]] = defaultdict(lambda: {"memory": [], "control": []})
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        for record in payload.get("results", []):
            if record.get("excluded"):
                continue
            outcome = record.get("outcome") or {}
            if "memory_success" not in outcome or "control_success" not in outcome:
                continue
            task_id = str(record["task_id"])
            cells[task_id]["memory"].append(bool(outcome["memory_success"]))
            cells[task_id]["control"].append(bool(outcome["control_success"]))
    return dict(cells)


def rep_instability(cells: dict[str, dict[str, list[bool]]]) -> dict[str, Any]:
    """How often repetitions of the SAME cell disagreed with each other.

    Zero disagreement with R>1 means the agent is deterministic enough that
    one rep would have done. Any disagreement means a single-rep run was
    reading noise, and says by how much.
    """
    total = unstable = 0
    reps_seen: set[int] = set()
    for arms in cells.values():
        for results in arms.values():
            if not results:
                continue
            reps_seen.add(len(results))
            if len(results) > 1:
                total += 1
                if len(set(results)) > 1:
                    unstable += 1
    return {
        "cells_with_multiple_reps": total,
        "cells_that_disagreed_across_reps": unstable,
        "unstable_fraction": round(unstable / total, 4) if total else None,
        "reps_per_cell_observed": sorted(reps_seen),
    }


def poisoning(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    """Context poisoning and its mirror, from the same paired cells."""
    control_passed = [(m, c) for m, c in pairs if c]
    control_failed = [(m, c) for m, c in pairs if not c]
    poisoned = sum(1 for m, _ in control_passed if not m)
    rescued = sum(1 for m, _ in control_failed if m)
    return {
        "definition": "poisoned / passed_without_memory; a task the agent could do WITHOUT memory and could not do WITH it",
        "passed_without_memory": len(control_passed),
        "poisoned": poisoned,
        "context_poisoning_rate": round(poisoned / len(control_passed), 4) if control_passed else None,
        "failed_without_memory": len(control_failed),
        "rescued": rescued,
        "memory_rescue_rate": round(rescued / len(control_failed), 4) if control_failed else None,
    }


#: The files that decide what the memory arm's prompt LOOKS like. A change to
#: any of them changes the treatment, so runs made on either side of such a
#: change are not the same experiment and must not be pooled. (Concretely:
#: ``data_marking`` + ``agent_bridge`` added ``<evidence>`` framing and a
#: preamble to the recalled section, which lengthens the memory arm's prompt.)
PROMPT_SHAPING_FILES = (
    "src/mind_mem/bench/ab_arms.py",
    "src/mind_mem/agent_bridge.py",
    "src/mind_mem/data_marking.py",
)


def harness_fingerprint(repo_root: str, artifact_paths: list[str]) -> dict[str, Any]:
    """Fingerprint the prompt-shaping code, and catch a mid-run change.

    A promise that "all these runs used the same harness" is worth nothing
    once several seats share a tree. This checks it: if any prompt-shaping
    file was modified AFTER the earliest artifact in the set was written,
    the set spans a harness change and the flag says so rather than leaving
    it to be discovered in the numbers.
    """
    files: dict[str, Any] = {}
    newest_mtime = 0.0
    for rel in PROMPT_SHAPING_FILES:
        path = os.path.join(repo_root, rel)
        if not os.path.isfile(path):
            files[rel] = {"present": False}
            continue
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()[:16]
        mtime = os.path.getmtime(path)
        newest_mtime = max(newest_mtime, mtime)
        files[rel] = {"present": True, "sha256_16": digest, "mtime": mtime}
    earliest_artifact = min((os.path.getmtime(p) for p in artifact_paths), default=0.0)
    changed_mid_run = bool(artifact_paths) and newest_mtime > earliest_artifact
    return {
        "files": files,
        "earliest_artifact_mtime": earliest_artifact,
        "newest_prompt_shaping_mtime": newest_mtime,
        "harness_changed_after_first_artifact": changed_mid_run,
        "note": (
            "True means this artifact set spans a change to the memory arm's rendering; "
            "the runs are not one experiment and must be re-run, not blended."
        ),
    }


def prompt_lengths(paths: list[str]) -> dict[str, Any]:
    """The memory arm's token surcharge — what a placebo would have to match.

    The memory arm's prompt is longer than the control's by construction, and
    the framing preamble widened that gap. So a positive delta is consistent
    with two stories: memory helped, or MORE CONTEXT helped. Only a
    length-matched placebo arm (same token count, corpus drawn from a
    DIFFERENT task) separates them. This does not separate them; it measures
    exactly how much length a placebo has to match, so the confound is a
    stated quantity rather than an unstated one.
    """
    memory: list[int] = []
    control: list[int] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        for record in payload.get("results", []):
            arms = record.get("arms") or {}
            try:
                memory.append(int(arms["memory"]["prompt"]["prompt_tokens"]))
                control.append(int(arms["control"]["prompt"]["prompt_tokens"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not memory:
        return {"n": 0, "placebo_required": True}
    gaps = sorted(m - c for m, c in zip(memory, control))
    mid = len(gaps) // 2
    return {
        "n": len(gaps),
        "mean_memory_prompt_tokens": round(sum(memory) / len(memory), 1),
        "mean_control_prompt_tokens": round(sum(control) / len(control), 1),
        "mean_gap_tokens": round(sum(gaps) / len(gaps), 1),
        "median_gap_tokens": gaps[mid] if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2,
        "min_gap_tokens": gaps[0],
        "max_gap_tokens": gaps[-1],
        "placebo_required": True,
        "placebo_note": (
            "A length-matched placebo arm (this many tokens, seeded from a DIFFERENT task's corpus) "
            "is required before any positive delta can be attributed to memory rather than to context "
            "length. It is NOT present in this artifact."
        ),
    }


def build(paths: list[str], *, reduce: str = "majority", agent_cutoff: str = "") -> dict[str, Any]:
    """Pool run artifacts into one reduced, disclosed, paired result."""
    cells = load_runs(paths)
    usable = {tid: arms for tid, arms in cells.items() if arms["memory"] and arms["control"]}
    pairs = [(reduce_cell(a["memory"], reduce), reduce_cell(a["control"], reduce)) for a in usable.values()]
    summary = summarise(pairs)
    return {
        "schema_version": 2,
        "inputs": {"n_artifacts": len(paths), "artifacts": sorted(os.path.basename(p) for p in paths)},
        "harness": harness_fingerprint(_REPO_ROOT, paths),
        "prompt_lengths": prompt_lengths(paths),
        "reduction": {"reducer": reduce, "known_reducers": list(REDUCERS)},
        "disclosure": {
            "agent_cutoff": agent_cutoff or "UNSTATED",
            "agent_cutoff_note": (
                "The agent's training cutoff. If it postdates the commits these tasks were mined from, "
                "the agent may reproduce a fix from memorised training data in BOTH arms, which suppresses "
                "any measurable memory effect. Recorded as UNSTATED rather than guessed."
            ),
        },
        "n_tasks": len(usable),
        "summary": summary.as_dict(),
        "rep_stability": rep_instability(usable),
        "context_poisoning": poisoning(pairs),
        "per_task": {
            tid: {"memory": arms["memory"], "control": arms["control"]} for tid, arms in sorted(usable.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reduce repeated memory-A/B runs to one paired number")
    ap.add_argument("--runs-glob", default="benchmarks/.cache/ab_runs/*.json")
    ap.add_argument("--reduce", default="majority", choices=REDUCERS)
    ap.add_argument("--agent-cutoff", default="", help="the agent model's training cutoff, as a disclosure")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    paths = sorted(glob.glob(a.runs_glob))
    if not paths:
        print(f"no run artifacts matched {a.runs_glob!r}", file=sys.stderr)
        return 2
    payload = build(paths, reduce=a.reduce, agent_cutoff=a.agent_cutoff)
    print(json.dumps(payload, indent=2))
    s = payload["summary"]
    print(f"memory_ab_reduced_tasks: {payload['n_tasks']}")
    print(f"memory_ab_reduced_reducer: {a.reduce}")
    print(f"memory_ab_reduced_memory_successes: {s['memory_successes']}")
    print(f"memory_ab_reduced_control_successes: {s['control_successes']}")
    print(f"memory_ab_reduced_delta: {s['delta_successes']}")
    print(f"memory_ab_reduced_discordant: {s['n_discordant']}")
    print(f"memory_ab_reduced_min_discordant_for_significance: {s['min_discordant_for_significance']}")
    print(f"memory_ab_reduced_p_value: {s['p_value']}")
    print(f"memory_ab_reduced_verdict: {s['verdict']}")
    print(f"memory_ab_context_poisoning_rate: {payload['context_poisoning']['context_poisoning_rate']}")
    print(f"memory_ab_memory_rescue_rate: {payload['context_poisoning']['memory_rescue_rate']}")
    print(f"memory_ab_agent_cutoff: {payload['disclosure']['agent_cutoff']}")
    print(f"memory_ab_harness_changed_mid_run: {payload['harness']['harness_changed_after_first_artifact']}")
    print(f"memory_ab_mean_prompt_gap_tokens: {payload['prompt_lengths'].get('mean_gap_tokens')}")
    print(f"memory_ab_placebo_arm_present: False")
    if a.out:
        out = a.out if os.path.isabs(a.out) else os.path.join(os.getcwd(), a.out)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"memory_ab_reduced_artifact: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
