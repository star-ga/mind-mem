"""Agent adapters: the one component the harness does not own.

The harness owns the task, the arms, the budget and the grading.  What
attempts the task is pluggable, because the measurement is about memory,
not about any particular agent -- and because naming one would make the
number un-reproducible for anyone without it.  An adapter receives a
prompt and a work tree and is expected to edit the tree.

Three adapters ship:

``none``
    Edits nothing.  Deterministic, free, and the only way to prove the
    harness itself is deterministic: two runs of the same task must
    produce byte-identical scored records.  It also fixes the floor --
    with this adapter both arms must fail, and a run where one "passes"
    means the grader is broken.

``command``
    Runs an external agent, argv given at run time, prompt on stdin, in
    the sandboxed environment both arms share.  This is the real
    measurement path.

``reference-fix``
    Applies the source side of the task's own commit.  It exists to prove
    the grader can register a *pass* -- without a positive control,
    "both arms failed" is indistinguishable from "the grader never
    returns success".  It reads material from after the cutoff, so it is
    barred from arms by :data:`ARM_ELIGIBLE` and reachable only through
    the ``selfcheck`` entry point.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from ..cognitive_forget import estimate_tokens


@dataclass(frozen=True)
class AgentRequest:
    """Everything an attempt is given. Identical across arms but the prompt."""

    task_id: str
    arm: str
    prompt: str
    tree: str
    env: Mapping[str, str]
    wall_seconds: int
    output_tokens: int
    steps: int


@dataclass(frozen=True)
class AgentResult:
    """What the attempt spent. No verdict here -- grading is a separate step.

    ``steps_observed`` distinguishes "took zero steps" from "this adapter
    cannot see steps".  An external agent reports neither its step count
    nor its own token usage, so ``output_tokens`` is measured from what it
    wrote to stdout -- a lower bound on its spend, recorded as such rather
    than presented as the true figure.
    """

    returncode: int
    timed_out: bool
    output_tokens: int
    steps: int
    tail: str
    steps_observed: bool = True
    output_tokens_are_lower_bound: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "output_tokens": self.output_tokens,
            "output_tokens_are_lower_bound": self.output_tokens_are_lower_bound,
            "steps": self.steps,
            "steps_observed": self.steps_observed,
            "tail": self.tail,
        }


class AgentError(RuntimeError):
    """The adapter could not be constructed or could not run."""


def _no_edit(request: AgentRequest) -> AgentResult:
    """Attempt nothing. The floor, and the determinism fixture."""
    return AgentResult(returncode=0, timed_out=False, output_tokens=0, steps=0, tail="no-edit adapter made no change")


def _tail(text: str, lines: int = 12) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _substitute(argv: Sequence[str], request: AgentRequest) -> list[str]:
    """Fill the placeholders an operator may use in the agent's argv."""
    return [
        arg.replace("{prompt}", request.prompt)
        .replace("{output_tokens}", str(request.output_tokens))
        .replace("{steps}", str(request.steps))
        for arg in argv
    ]


def _timed_out(request: AgentRequest) -> AgentResult:
    """The arm hit its wall-clock ceiling: a budget event, recorded as one."""
    return AgentResult(
        returncode=-1,
        timed_out=True,
        output_tokens=request.output_tokens,
        steps=request.steps,
        tail="TIMEOUT",
        steps_observed=False,
        output_tokens_are_lower_bound=True,
    )


def _from_process(proc: "subprocess.CompletedProcess[str]") -> AgentResult:
    """Read spend off a finished process, marking what is unobservable."""
    return AgentResult(
        returncode=proc.returncode,
        timed_out=False,
        output_tokens=estimate_tokens(proc.stdout),
        steps=0,
        tail=_tail(proc.stdout + proc.stderr),
        steps_observed=False,
        output_tokens_are_lower_bound=True,
    )


def make_command_agent(argv: Sequence[str], nice_level: int = 15) -> Callable[[AgentRequest], AgentResult]:
    """Build an adapter that runs ``argv`` in the work tree.

    ``{prompt}`` in any argument is substituted with the prompt; if no
    argument contains it the prompt is written to stdin instead.
    ``{output_tokens}`` and ``{steps}`` are substituted with the arm's
    ceilings, so an operator can wire the budget into whatever flag their
    agent uses without this module naming any particular one.  The process
    is niced (this is a shared box) and hard-capped by the arm's
    wall-clock budget, which is the same number for both arms.
    """
    if not argv:
        raise AgentError("command agent requires a non-empty argv")
    uses_stdin = not any("{prompt}" in arg for arg in argv)

    def run(request: AgentRequest) -> AgentResult:
        cmd = ["nice", "-n", str(nice_level), *_substitute(argv, request)]
        try:
            # Fixed argv supplied by the operator at run time; shell=False.
            proc = subprocess.run(  # nosec B603
                cmd,
                cwd=request.tree,
                env=dict(request.env),
                input=request.prompt if uses_stdin else None,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=request.wall_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _timed_out(request)
        except OSError as exc:
            raise AgentError(f"could not launch the agent command: {exc}") from exc
        return _from_process(proc)

    return run


def make_reference_fix_agent(repo: str, sha: str, src_paths: Sequence[str]) -> Callable[[AgentRequest], AgentResult]:
    """Harness positive control: write the commit's own source files.

    Not an agent and never an arm.  It answers one question -- can this
    pipeline observe a success at all? -- and a benchmark that cannot
    answer that has no business reporting a failure.
    """
    from .repo_task_mining import git

    def run(request: AgentRequest) -> AgentResult:
        written = 0
        for path in src_paths:
            target = os.path.join(request.tree, path)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(git(repo, "show", f"{sha}:{path}"))
            written += 1
        return AgentResult(returncode=0, timed_out=False, output_tokens=0, steps=written, tail=f"applied {written} source file(s)")

    return run


#: Adapters an arm may use.  ``reference-fix`` is deliberately absent.
ARM_ELIGIBLE: frozenset[str] = frozenset({"none", "command"})


def get_agent(name: str, argv: Sequence[str] = (), *, for_arm: bool = True) -> Callable[[AgentRequest], AgentResult]:
    """Resolve an adapter by name, refusing one that may not run in an arm."""
    if for_arm and name not in ARM_ELIGIBLE:
        raise AgentError(f"agent {name!r} may not run in an arm; eligible: {sorted(ARM_ELIGIBLE)}")
    if name == "none":
        return _no_edit
    if name == "command":
        return make_command_agent(argv)
    raise AgentError(f"unknown agent {name!r}")
