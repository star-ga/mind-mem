#!/usr/bin/env python3
"""Run a memory A/B stratum one task at a time, resumably.

``mind-mem-bench-ab run`` writes its artifact only when the whole selection
finishes, so a stratum killed at task 10 of 29 leaves nothing at all. On a
shared box that is not a hypothetical: it happened, and two hours of agent
work went with it.

This driver runs the same harness with ``--select task:<id>`` once per task
per rep, so every completed task lands its own artifact immediately. A rerun
skips artifacts already on disk, which makes the stratum resumable and makes
``--reps`` cheap: rep 2 is the same loop with a different output suffix.

It decides nothing about the measurement. The harness owns the arms, the
budget, the isolation and the grading; ``benchmarks/memory_ab_analysis.py``
owns the reduction across reps and the pooled statistic. This file owns the
loop and nothing else.

    python3 benchmarks/memory_ab_stratum.py --bucket single_file --reps 1 \\
        --wall-seconds 120 --agent-env OPENAI_API_KEY \\
        -- /path/to/agent -y -p '{prompt}'
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess  # nosec B404 - fixed argv, shell=False
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mind_mem.bench.ab_task import load_task_set, select_tasks  # noqa: E402

DEFAULT_TASK_SET = "benchmarks/tasks/real_repo_tasks.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Resumable per-task driver for the memory A/B stratum")
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--task-set", default=DEFAULT_TASK_SET)
    ap.add_argument("--bucket", required=True, help="size_bucket to run (single_file | small | medium | large)")
    ap.add_argument("--reps", type=int, default=1, help="repetitions per task; each rep writes its own artifact")
    ap.add_argument("--wall-seconds", type=int, default=120)
    ap.add_argument("--prompt-tokens", type=int, default=8000)
    ap.add_argument("--memory-tokens", type=int, default=1500)
    ap.add_argument("--out-dir", default="benchmarks/.cache/ab_runs")
    ap.add_argument("--agent", default="command")
    ap.add_argument("--agent-env", action="append", default=[])
    ap.add_argument("--label", default="")
    ap.add_argument("agent_argv", nargs="*")
    a = ap.parse_args(argv)

    task_set_path = a.task_set if os.path.isabs(a.task_set) else os.path.join(a.repo, a.task_set)
    tasks = select_tasks(load_task_set(task_set_path), f"bucket:{a.bucket}")
    out_dir = a.out_dir if os.path.isabs(a.out_dir) else os.path.join(a.repo, a.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"stratum={a.bucket} tasks={len(tasks)} reps={a.reps} wall_seconds={a.wall_seconds}", flush=True)
    t0 = time.time()
    ran = skipped = failed = 0
    for rep in range(1, a.reps + 1):
        for i, task in enumerate(tasks, start=1):
            out = os.path.join(out_dir, f"{a.bucket}__{task.task_id}__rep{rep}.json")
            if os.path.isfile(out):
                skipped += 1
                continue
            cmd = [
                sys.executable,
                os.path.join(a.repo, "benchmarks", "memory_ab_bench.py"),
                "run",
                "--repo",
                a.repo,
                "--task-set",
                a.task_set,
                "--select",
                f"task:{task.task_id}",
                "--agent",
                a.agent,
                "--wall-seconds",
                str(a.wall_seconds),
                "--prompt-tokens",
                str(a.prompt_tokens),
                "--memory-tokens",
                str(a.memory_tokens),
                "--out",
                out,
                "--label",
                a.label or f"{a.bucket} rep{rep}",
            ]
            for name in a.agent_env:
                cmd += ["--agent-env", name]
            if a.agent_argv:
                cmd += ["--", *a.agent_argv]
            proc = subprocess.run(cmd, cwd=a.repo, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)  # nosec B603
            if proc.returncode != 0 or not os.path.isfile(out):
                failed += 1
                print(f"[rep{rep} {i}/{len(tasks)}] {task.task_id} FAILED rc={proc.returncode} :: {proc.stdout[-300:]}", flush=True)
                continue
            ran += 1
            with open(out, encoding="utf-8") as handle:
                payload = json.load(handle)
            s = payload["summary"]
            print(
                f"[rep{rep} {i}/{len(tasks)}] {task.task_id} mem={s['memory_successes']} ctl={s['control_successes']} "
                f"({time.time() - t0:.0f}s elapsed)",
                flush=True,
            )
    print(f"ran={ran} skipped={skipped} failed={failed} elapsed={time.time() - t0:.0f}s out_dir={out_dir}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
