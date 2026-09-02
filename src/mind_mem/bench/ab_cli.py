"""``mind-mem-bench-ab`` -- run the with-memory versus without-memory comparison.

Two subcommands::

    mind-mem-bench-ab run --select bucket:single_file:1 --agent none
    mind-mem-bench-ab run --select task:mm-7571daeac51c --agent command \\
        --agent-env API_KEY -- /path/to/agent --prompt-file -

    mind-mem-bench-ab selfcheck --select bucket:single_file:1
    mind-mem-bench-ab report --artifact a.json --artifact b.json

``run`` is the measurement.  ``report`` pools several ``run`` artifacts --
the suite is run one stated stratum at a time on a shared box -- into the
single paired number, with the per-tier and per-stratum breakdowns beside
it.  ``selfcheck`` is the positive control: it
proves the grader registers a pass when the reference fix is applied and a
failure when it is not.  Without that, "both arms failed" would be
indistinguishable from a grader that never returns success, so the
selfcheck is not optional decoration -- it is what licenses a null result.

Credentials named with ``--agent-env`` are read from the caller's
environment and forwarded to **both** arms.  Only their names are ever
written to the artifact.

Scratch: each arm materialises a full work tree, so ``--workdir`` needs
room for one copy of the repository at a time (the arena is wiped between
arms and deleted after each task).  A run that cannot extract a tree
records the task as ``setup_failed`` and keeps going rather than reporting
an infrastructure fault as an arm that failed the task.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Any

from .ab_agent import AgentRequest, get_agent, make_reference_fix_agent
from .ab_arms import Budget
from .ab_grade import grade, snapshot_tree
from .ab_harness import file_sha256, run_suite
from .ab_task import Task, load_task_set, select_tasks
from .repo_task_validation import apply_test_patch, extract_tree

DEFAULT_TASK_SET = "benchmarks/tasks/real_repo_tasks.json"


def _budget(args: argparse.Namespace) -> Budget:
    return Budget(
        prompt_tokens=args.prompt_tokens,
        memory_tokens=args.memory_tokens,
        output_tokens=args.output_tokens,
        wall_seconds=args.wall_seconds,
        steps=args.steps,
    )


def _passthrough(names: list[str]) -> dict[str, str]:
    """Read the named variables from the caller's environment, by name only."""
    missing = [name for name in names if name not in os.environ]
    if missing:
        raise SystemExit(f"--agent-env named variable(s) not present in this environment: {', '.join(missing)}")
    return {name: os.environ[name] for name in names}


def _load(args: argparse.Namespace) -> tuple[str, tuple[Task, ...]]:
    path = args.task_set if os.path.isabs(args.task_set) else os.path.join(args.repo, args.task_set)
    return path, select_tasks(load_task_set(path), args.select)


def _write(payload: dict[str, Any], out: str, repo: str) -> str:
    path = out if os.path.isabs(out) else os.path.join(repo, out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def cmd_run(args: argparse.Namespace) -> int:
    """Run both arms over the selected tasks and write the artifact."""
    path, tasks = _load(args)
    passthrough = _passthrough(args.agent_env)
    agent = get_agent(args.agent, args.agent_argv, for_arm=True)
    meta = {
        "agent": {"name": args.agent, "argv": list(args.agent_argv), "env_passthrough_keys": sorted(passthrough)},
        "task_set": {
            # POSIX separators, matching ``ab_arms``/``ab_grade``: this key is
            # published in the run artifact and is covered by its digest, so an
            # os-native separator would make the same run hash differently on
            # Windows than on Linux.
            "path": os.path.relpath(path, args.repo).replace(os.sep, "/"),
            "sha256": file_sha256(path),
            "selection": args.select,
            "n_selected": len(tasks),
        },
        "run_label": args.label,
    }
    workdir = args.workdir or tempfile.mkdtemp(prefix="mm_ab_")
    os.makedirs(workdir, exist_ok=True)
    payload = run_suite(args.repo, tasks, agent, workdir, _budget(args), args.python, meta, passthrough)
    out = _write(payload, args.out, args.repo)
    _print_scorecard(payload, out)
    return 0


def _print_scorecard(payload: dict[str, Any], out: str) -> None:
    """Machine-greppable summary. Every load-bearing number on its own line."""
    summary = payload["summary"]
    print(json.dumps({"counts": payload["counts"], "spend": payload["spend"], "summary": summary}, indent=2))
    print(f"memory_ab_tasks: {summary['n_tasks']}")
    print(f"memory_ab_memory_successes: {summary['memory_successes']}")
    print(f"memory_ab_control_successes: {summary['control_successes']}")
    print(f"memory_ab_delta: {summary['delta_successes']}")
    print(f"memory_ab_discordant: {summary['n_discordant']}")
    print(f"memory_ab_agent_inert: {payload['counts']['agent_inert']}")
    print(f"memory_ab_prompt_tokens_memory: {payload['spend']['memory']['prompt_tokens']}")
    print(f"memory_ab_prompt_tokens_control: {payload['spend']['control']['prompt_tokens']}")
    print(f"memory_ab_p_value: {summary['p_value']}")
    print(f"memory_ab_verdict: {summary['verdict']}")
    print(f"memory_ab_digest: {payload['digest']}")
    print(f"memory_ab_artifact: {out}")


def _selfcheck_task(repo: str, task: Task, workdir: str, python: str, timeout: int) -> dict[str, Any]:
    """Grade the parent tree, then the reference fix, in the same arena."""
    tree, home = os.path.join(workdir, "tree"), os.path.join(workdir, "home")
    extract_tree(repo, task.parent_sha, tree)
    apply_test_patch(repo, task.sha, task.test_patch_paths, tree)
    os.makedirs(home, exist_ok=True)
    before = snapshot_tree(tree)
    untouched = grade(task, tree, home, python, before, timeout)
    fix_paths = commit_fix_paths(repo, task.sha)
    fixer = make_reference_fix_agent(repo, task.sha, fix_paths)
    fixer(AgentRequest(task.task_id, "selfcheck", "", tree, {}, timeout, 0, 0))
    # Diff against the ORIGINAL snapshot so the record names the files the
    # positive control wrote, rather than reporting an empty change set.
    fixed = grade(task, tree, home, python, before, timeout)
    return {
        "task_id": task.task_id,
        "reference_paths_replayed": [f"{status} {path}" for status, path in fix_paths],
        "untouched_parent": untouched.as_dict(),
        "reference_fix": fixed.as_dict(),
        "grader_detects_failure": not untouched.success,
        "grader_detects_success": fixed.success,
    }


def commit_fix_paths(repo: str, sha: str) -> list[tuple[str, str]]:
    """The commit's whole non-test delta, as ``(status, path)`` pairs.

    Every status is kept, including ``D``: a commit that fixes a defect by
    deleting a module is as much a reference fix as one that adds a
    function, and a replay restricted to added/modified files under
    ``src/`` could reproduce neither that nor a fix whose substance lives
    outside ``src/``.  It then reported "the grader cannot see a success"
    when the grader was fine and the replay was incomplete -- a positive
    control that fails for its own reasons is worse than none.  Test-side
    paths are dropped: they are already applied as the task's test patch,
    and rewriting one would trip the tamper check and void the attempt.
    """
    from .repo_task_mining import git, is_test_infra_path

    raw = git(repo, "show", "--name-status", "--no-renames", "--format=", sha)
    paths: list[tuple[str, str]] = []
    for line in raw.splitlines():
        if "\t" not in line:
            continue
        status, path = line.split("\t", 1)
        path = path.strip()
        if path and not is_test_infra_path(path):
            paths.append((status.strip()[:1], path))
    return sorted(paths, key=lambda row: row[1])


def _print_pooled(payload: dict[str, Any]) -> None:
    """Machine-greppable pooled summary, same key shape as a single run."""
    summary = payload["summary"]
    print(json.dumps({k: payload[k] for k in ("n_pairs", "summary", "by_tier", "by_size_bucket", "spend", "excluded")}, indent=2))
    print(f"memory_ab_pooled_tasks: {summary['n_tasks']}")
    print(f"memory_ab_pooled_memory_successes: {summary['memory_successes']}")
    print(f"memory_ab_pooled_control_successes: {summary['control_successes']}")
    print(f"memory_ab_pooled_delta: {summary['delta_successes']}")
    print(f"memory_ab_pooled_discordant: {summary['n_discordant']}")
    print(f"memory_ab_pooled_min_discordant_for_significance: {summary['min_discordant_for_significance']}")
    print(f"memory_ab_pooled_excluded: {payload['excluded']['total']}")
    print(f"memory_ab_pooled_agent_inert: {len(payload['agent_inert_task_ids'])}")
    print(f"memory_ab_pooled_p_value: {summary['p_value']}")
    print(f"memory_ab_pooled_verdict: {summary['verdict']}")
    print(f"memory_ab_pooled_digest: {payload['digest']}")


def cmd_report(args: argparse.Namespace) -> int:
    """Pool stratum artifacts into the single paired number."""
    from .ab_report import build_report

    payload = build_report(args.artifact)
    _print_pooled(payload)
    if args.out:
        print(f"memory_ab_pooled_artifact: {_write(payload, args.out, args.repo)}")
    return 0


def cmd_selfcheck(args: argparse.Namespace) -> int:
    """Prove the grader can see both a failure and a success."""
    _, tasks = _load(args)
    rows = []
    for task in tasks:
        workdir = tempfile.mkdtemp(prefix=f"mm_ab_sc_{task.task_id}_")
        try:
            rows.append(_selfcheck_task(args.repo, task, workdir, args.python, args.wall_seconds))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    ok = all(row["grader_detects_failure"] and row["grader_detects_success"] for row in rows)
    print(json.dumps(rows, indent=2))
    print(f"ab_selfcheck_tasks: {len(rows)}")
    print(f"ab_selfcheck: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", default=os.getcwd(), help="Repository the tasks were mined from (default: cwd).")
    parser.add_argument("--task-set", default=DEFAULT_TASK_SET, help="Generated task set to read.")
    parser.add_argument("--select", default="all", help="all | first:N | bucket:NAME[:N] | task:ID")
    parser.add_argument("--python", default=sys.executable, help="Interpreter used for the sandboxed pytest runs.")
    parser.add_argument("--prompt-tokens", type=int, default=8000, help="Input-token ceiling, identical in both arms.")
    parser.add_argument("--memory-tokens", type=int, default=1500, help="Recalled-context sub-budget inside the input ceiling.")
    parser.add_argument("--output-tokens", type=int, default=4000, help="Output-token ceiling handed to the adapter.")
    parser.add_argument("--wall-seconds", type=int, default=900, help="Per-arm wall-clock ceiling, identical in both arms.")
    parser.add_argument("--steps", type=int, default=40, help="Step ceiling handed to the adapter.")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mind-mem-bench-ab", description="With-memory versus without-memory, on real repository tasks.")
    subs = parser.add_subparsers(dest="command", required=True)
    run = subs.add_parser("run", help="Run both arms and write the artifact.")
    _add_common(run)
    run.add_argument("--agent", default="none", help="Adapter: none | command.")
    run.add_argument(
        "--agent-env", action="append", default=[], help="Forward this environment variable to BOTH arms (name only is recorded)."
    )
    run.add_argument("--workdir", default="", help="Scratch directory (default: a fresh temporary one).")
    run.add_argument("--out", default="benchmarks/memory_ab_results.json", help="Artifact path.")
    run.add_argument("--label", default="", help="Free-text label recorded in the artifact.")
    run.add_argument(
        "agent_argv",
        nargs="*",
        help="Agent argv, after --. '{prompt}' / '{output_tokens}' / '{steps}' are substituted; without '{prompt}' it goes on stdin.",
    )
    run.set_defaults(func=cmd_run)
    check = subs.add_parser("selfcheck", help="Prove the grader sees a failure and a success.")
    _add_common(check)
    check.set_defaults(func=cmd_selfcheck)
    report = subs.add_parser("report", help="Pool run artifacts into one paired number.")
    report.add_argument("--repo", default=os.getcwd(), help="Repository the runs were made against (default: cwd).")
    report.add_argument("--artifact", action="append", default=[], required=True, help="A run artifact to pool. Repeatable.")
    report.add_argument("--out", default="", help="Optional path for the pooled report.")
    report.set_defaults(func=cmd_report)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    exit_code: int = args.func(args)
    return exit_code


if __name__ == "__main__":  # pragma: no cover - module entry
    raise SystemExit(main())
