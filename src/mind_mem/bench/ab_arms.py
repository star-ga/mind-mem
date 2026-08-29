"""The two arms, and the invariants that make "memory is the only variable" true.

WHAT DIFFERS BETWEEN THE ARMS -- exactly one thing
--------------------------------------------------
A prefix on the prompt.  The ``memory`` arm's prompt is the ``control``
arm's prompt with a recalled-context section glued on the front; the
control section is the empty string.  :func:`assert_arms_equal` checks
that literally -- ``memory.prompt == memory.memory_section + control.prompt``
-- so the claim is enforced by string equality rather than asserted in a
docstring.

WHAT IS HELD CONSTANT
---------------------
The task, the reported-issue statement, the instruction footer, the agent
and its argv, the work tree (both arms start from a tree extracted at the
same ``parent_sha`` with the same test patch applied), the environment
**byte for byte**, the token ceilings, the step ceiling and the wall-clock
ceiling.  Because the environment is shared and identical, the control
arm's isolation is not a separate configuration that could drift out of
sync: there is one environment, it is checked once, and both arms run in it.

THE MEMORY GATE IS STRUCTURAL, NOT A FLAG
-----------------------------------------
This package ships no recall kill-switch, so "memory off" cannot be a flag
flip.  :func:`assert_control_isolated` enforces it structurally and refuses
to run otherwise:

1. no ``MIND_MEM_*`` workspace or DSN variable reaches the agent;
2. no ``mm`` or ``mind-mem-*`` executable is on the agent's ``PATH``;
3. ``HOME`` is a sandbox directory, so no user-level agent configuration
   and no MCP server registration (mind-mem's included) is loaded;
4. the seeded workspace lives outside the work tree, so an agent that
   reads every file it can reach still cannot read the corpus.

Point 4 matters more here than it looks: the corpus is this repository's
own history and the agent works inside this repository.  Without it the
control arm would quietly re-acquire memory by reading files, the delta
would collapse, and we would publish a null result caused by our own
harness.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..cognitive_forget import estimate_tokens
from .ab_task import Task

ARM_MEMORY = "memory"
ARM_CONTROL = "control"
ARMS: tuple[str, ...] = (ARM_MEMORY, ARM_CONTROL)

#: Executables whose presence on PATH would hand the control arm a memory.
_MEMORY_EXECUTABLES = ("mm",)
_MEMORY_EXECUTABLE_PREFIX = "mind-mem"

#: How many hits recall is asked for before the token budget trims them.
RECALL_LIMIT = 20

#: Our own memory CLI and workspace variable, as an instruction file in a
#: checked-in repository would spell them. A work tree can legitimately
#: contain a file telling an agent to consult memory before answering --
#: this repository's does -- and pretending otherwise would make the
#: isolation claim inaccurate. Such files are identical in both arms and
#: point at a CLI that is absent from PATH in both, so they cannot bias the
#: delta; they are detected and named in the artifact rather than removed,
#: because deleting files from the task's starting tree would change the
#: task itself.
_MEMORY_POINTER = re.compile(r"\bmm +(?:inject|recall|context|resume)\b|MIND_MEM_WORKSPACE|\bmind-mem-recall\b")

#: Trees that are the subject of the task, not instructions to an agent.
_POINTER_SCAN_SKIP = ("src", "tests", "benchmarks", "docs", "templates", "train", "examples")

#: Read cap per scanned file; an instruction file is small by nature.
MAX_POINTER_FILE_BYTES = 262_144

#: Environment variables that must never reach either agent process.
_FORBIDDEN_ENV_PREFIX = "MIND_MEM_"
_ALLOWED_MIND_MEM_ENV = frozenset({"MIND_MEM_DISABLE_TELEMETRY"})

#: Identical in both arms.  It states the grading contract in the same
#: words the generator used, so neither arm is told anything the other is not.
INSTRUCTIONS = """\
How you are graded
------------------
The named test files are run exactly as written. You pass only if pytest
exits zero and every test that is currently failing passes. Nothing else
is scored; no one reads your explanation.

Constraints
-----------
- Edit files under src/ only. Any change under tests/ or to conftest.py
  voids the attempt.
- Work from the files in this directory. It has no version-control
  history, so the project's past is not readable from here.
- Stop when the tests pass.
"""


class ArmMismatch(AssertionError):
    """The two arms differ in something other than memory availability."""


class ControlLeak(AssertionError):
    """The control arm could reach memory."""


@dataclass(frozen=True)
class Budget:
    """The ceiling each arm is held to. Equal by construction, and stated."""

    prompt_tokens: int = 8000
    memory_tokens: int = 1500
    output_tokens: int = 4000
    wall_seconds: int = 900
    steps: int = 40

    def __post_init__(self) -> None:
        if self.memory_tokens >= self.prompt_tokens:
            raise ValueError("memory_tokens must leave room for the task statement inside prompt_tokens")
        if min(self.prompt_tokens, self.memory_tokens, self.output_tokens, self.wall_seconds, self.steps) <= 0:
            raise ValueError("every budget component must be positive")

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "memory_tokens": self.memory_tokens,
            "output_tokens": self.output_tokens,
            "wall_seconds": self.wall_seconds,
            "steps": self.steps,
        }


@dataclass(frozen=True)
class PromptBuild:
    """One arm's prompt, with the accounting that proves it stayed in budget."""

    arm: str
    memory_section: str
    prompt: str
    prompt_tokens: int
    memory_tokens: int
    memory_blocks: tuple[str, ...] = ()
    memory_dropped: int = 0
    recall_query: str = ""
    over_budget: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "prompt_tokens": self.prompt_tokens,
            "memory_tokens": self.memory_tokens,
            "memory_blocks": list(self.memory_blocks),
            "memory_dropped": self.memory_dropped,
            "recall_query": self.recall_query,
            "over_budget": self.over_budget,
        }


def base_prompt(task: Task) -> str:
    """The prompt both arms share, derived only from the reported issue."""
    return f"{task.task_statement}\n{INSTRUCTIONS}"


def _recall_section(task: Task, workspace: str, budget: Budget, allowance: int) -> tuple[str, tuple[str, ...], int, int]:
    """Recall, pack to the memory sub-budget, and render the section.

    Routed through the shipped production surfaces -- ``recall`` with a
    pinned ``scoring_instant``, ``pack_to_budget`` (what ``mm context``
    uses) and ``AgentFormatter.inject`` (what ``mm inject`` uses) -- so the
    memory arm exercises the product, not a benchmark-only shortcut.
    """
    from ..agent_bridge import AgentFormatter
    from ..cognitive_forget import pack_to_budget
    from ..recall import recall

    ceiling = min(budget.memory_tokens, allowance)
    if ceiling <= 0:
        return "", (), 0, 0
    hits = recall(workspace, task.recall_query, limit=RECALL_LIMIT, scoring_instant=task.scoring_date)
    packed = pack_to_budget(hits, max_tokens=ceiling)
    if not packed.included:
        # Recall found nothing. Emitting the formatter's bare header would
        # put a difference in the prompt that carries no memory at all, so
        # the arms stay literally identical and the record says the memory
        # arm received nothing.
        return "", (), len(packed.dropped), 0
    section = AgentFormatter(max_blocks=len(packed.included)).inject("generic", task.recall_query, packed.included)
    section = f"{section.rstrip()}\n\n"
    return section, tuple(str(b.get("_id", "")) for b in packed.included), len(packed.dropped), estimate_tokens(section)


def build_prompt(task: Task, arm: str, budget: Budget, workspace: str = "") -> PromptBuild:
    """Build one arm's prompt under the shared ceiling."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    if arm == ARM_MEMORY and not workspace:
        # A memory arm with nowhere to recall from is a second control arm
        # wearing the wrong label; it would report a null result that the
        # harness caused. Refuse rather than degrade.
        raise ValueError("the memory arm requires a seeded workspace")
    base = base_prompt(task)
    allowance = budget.prompt_tokens - estimate_tokens(base)
    if arm == ARM_CONTROL or allowance <= 0:
        return PromptBuild(
            arm=arm,
            memory_section="",
            prompt=base,
            prompt_tokens=estimate_tokens(base),
            memory_tokens=0,
            over_budget=allowance <= 0,
        )
    section, ids, dropped, tokens = _recall_section(task, workspace, budget, allowance)
    prompt = section + base
    return PromptBuild(
        arm=arm,
        memory_section=section,
        prompt=prompt,
        prompt_tokens=estimate_tokens(prompt),
        memory_tokens=tokens,
        memory_blocks=ids,
        memory_dropped=dropped,
        recall_query=task.recall_query,
        over_budget=estimate_tokens(prompt) > budget.prompt_tokens,
    )


def build_env(tree: str, home: str, passthrough: Mapping[str, str] | None = None) -> dict[str, str]:
    """The single environment both arms run in.

    ``passthrough`` carries whatever credentials or trust flags the chosen
    agent needs.  It is applied to both arms identically and only its
    **keys** are ever recorded, never its values.
    """
    from .repo_task_validation import sandbox_env

    env = sandbox_env(tree, home)
    for key, value in sorted(dict(passthrough or {}).items()):
        if key.startswith(_FORBIDDEN_ENV_PREFIX) and key not in _ALLOWED_MIND_MEM_ENV:
            raise ControlLeak(f"passthrough variable {key} would hand the control arm a memory")
        env[key] = value
    return env


def assert_control_isolated(env: Mapping[str, str], tree: str, workspace: str) -> tuple[str, ...]:
    """Prove the agent cannot reach memory except through its prompt."""
    leaked = sorted(k for k in env if k.startswith(_FORBIDDEN_ENV_PREFIX) and k not in _ALLOWED_MIND_MEM_ENV)
    if leaked:
        raise ControlLeak(f"environment carries {leaked}")
    for directory in env.get("PATH", "").split(os.pathsep):
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if name in _MEMORY_EXECUTABLES or name.startswith(_MEMORY_EXECUTABLE_PREFIX):
                raise ControlLeak(f"{os.path.join(directory, name)} is on PATH")
    home = os.path.realpath(env.get("HOME", ""))
    if not home or home == os.path.realpath(os.path.expanduser("~")):
        raise ControlLeak("HOME is the real user home; agent configuration and MCP servers would load")
    if workspace and os.path.realpath(workspace).startswith(os.path.realpath(tree) + os.sep):
        raise ControlLeak("the seeded workspace is inside the work tree and could simply be read")
    return ("no_mind_mem_env", "no_memory_cli_on_path", "sandboxed_home_no_mcp", "workspace_outside_tree")


def assert_tree_has_no_corpus(tree: str) -> tuple[str, ...]:
    """Refuse a work tree that recall would read as a corpus of its own.

    The arms share the tree, so a corpus inside it could not bias the
    delta -- but it would make "the control arm has no memory" imprecise,
    and an imprecise isolation claim is the kind that quietly stops being
    true.  Checked after extraction, against the real corpus registry
    rather than a hand-copied list of paths.
    """
    from .._recall_constants import CORPUS_FILES

    present = sorted(rel for rel in CORPUS_FILES.values() if os.path.isfile(os.path.join(tree, rel)))
    if present:
        raise ControlLeak(f"the work tree carries recallable corpus file(s): {present}")
    return ("no_corpus_inside_work_tree",)


def scan_tree_for_memory_pointers(tree: str) -> tuple[str, ...]:
    """Name the files in the work tree that point an agent at memory.

    Recorded, not deleted: both arms get the same tree, and in both the
    ``mm`` CLI is off ``PATH`` and ``HOME`` is a sandbox, so an agent that
    follows such an instruction finds nothing.  Publishing the list is what
    keeps "the control arm cannot reach memory" an accurate sentence rather
    than an approximate one.
    """
    found: list[str] = []
    for root, dirs, files in os.walk(tree):
        if root == tree:
            dirs[:] = sorted(d for d in dirs if d not in _POINTER_SCAN_SKIP)
        for name in sorted(files):
            path = os.path.join(root, name)
            if os.path.getsize(path) > MAX_POINTER_FILE_BYTES:
                continue
            try:
                with open(path, encoding="utf-8") as handle:
                    text = handle.read()
            except (OSError, UnicodeDecodeError):
                continue
            if _MEMORY_POINTER.search(text):
                # POSIX separators, always. This list is PUBLISHED in the run
                # artifact and compared across arms and across machines, so an
                # os-native separator would make the same tree produce a
                # different report on Windows than on Linux -- and a benchmark
                # whose output depends on the host it ran on is not a benchmark.
                found.append(os.path.relpath(path, tree).replace(os.sep, "/"))
    return tuple(sorted(found))


def assert_arms_equal(memory: PromptBuild, control: PromptBuild, budget: Budget) -> tuple[str, ...]:
    """Refuse a pair of arms that differ in anything but the memory prefix."""
    if control.memory_section != "":
        raise ArmMismatch("the control arm carries a memory section")
    if memory.prompt != memory.memory_section + control.prompt:
        raise ArmMismatch("the memory arm's prompt is not the control prompt plus a recalled prefix")
    for build in (memory, control):
        if build.prompt_tokens > budget.prompt_tokens:
            raise ArmMismatch(f"{build.arm} arm prompt is {build.prompt_tokens} tokens, over the {budget.prompt_tokens} ceiling")
    return ("control_has_no_memory_section", "prompts_differ_only_by_prefix", "both_arms_within_prompt_ceiling")
