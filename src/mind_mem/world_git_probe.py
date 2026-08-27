# Copyright 2026 STARGA, Inc.
"""Deterministic git-ref liveness probe — "is this commit still where we left it?".

The third anchor kind :mod:`mind_mem.world_staleness` verifies. A block
that pins a commit, tag or branch is making a claim about a repository;
this module answers whether that claim still holds, using nothing but
local plumbing commands:

* ``rev-parse --verify`` — does the ref resolve at all?
* ``merge-base --is-ancestor`` + ``rev-list --count`` — has ``HEAD``
  moved past it, and by how many commits?

Strictly local and strictly read-only. ``GIT_TERMINAL_PROMPT=0`` and
``GIT_OPTIONAL_LOCKS=0`` are forced into the child environment so a
probe can neither prompt nor reach the network nor take a repository
lock. No ``fetch``, no ``ls-remote``, ever.

Refs are re-validated at this boundary even though extraction already
checked them: a value that reached ``git`` as an option would be an
argument-injection bug, so the check is repeated where it matters.

Stdlib only (``subprocess``).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess  # nosec B404 — local, read-only git plumbing; fixed argv, shell=False
from dataclasses import dataclass
from typing import Final, Sequence

__all__ = [
    "GIT_LIVE",
    "GIT_MISSING_REF",
    "GIT_MOVED",
    "GIT_UNVERIFIABLE",
    "GitProbeResult",
    "is_git_repo",
    "probe_ref",
]

GIT_LIVE: Final = "live"
GIT_MISSING_REF: Final = "missing_ref"
GIT_MOVED: Final = "ref_moved"
GIT_UNVERIFIABLE: Final = "unverifiable"

_REF_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/^~{}\-]*$")
_MAX_REF_LEN: Final = 200
DEFAULT_TIMEOUT_SECONDS: Final = 15.0


@dataclass(frozen=True)
class GitProbeResult:
    """Outcome of one git-ref probe."""

    status: str
    detail: str = ""
    resolved: str = ""
    distance: int = 0


def _child_env() -> dict[str, str]:
    """Environment that forbids prompting, network auth and lock-taking."""
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_ASKPASS"] = ""
    env["GCM_INTERACTIVE"] = "never"
    return env


def _git_executable() -> str | None:
    """Absolute path to ``git``, or ``None`` when it is not installed.

    Resolved rather than spelled as a bare name so the child process can
    never be picked up from a relative ``PATH`` entry.
    """
    return shutil.which("git")


def _run(root: str, args: Sequence[str], timeout: float) -> tuple[int, str]:
    """Run ``git -C root <args>``; return ``(returncode, stripped stdout)``.

    A missing ``git`` binary or a timeout is reported as returncode
    ``-1`` so callers degrade to "unverifiable" instead of raising.
    """
    executable = _git_executable()
    if executable is None:
        return -1, ""
    try:
        proc = subprocess.run(  # nosec B603 — fixed argv, shell=False, ref validated by _validate_ref
            [executable, "-C", root, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_child_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return -1, ""
    return proc.returncode, (proc.stdout or "").strip()


def is_git_repo(root: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> bool:
    """True when *root* sits inside a git work tree."""
    if not root or not os.path.isdir(root):
        return False
    code, out = _run(root, ["rev-parse", "--is-inside-work-tree"], timeout)
    return code == 0 and out == "true"


def _validate_ref(ref: str) -> str | None:
    """Return why *ref* is unusable, else None."""
    if not ref:
        return "empty ref"
    if len(ref) > _MAX_REF_LEN:
        return "ref too long"
    if ref.startswith("-"):
        return "ref may not start with '-'"
    if ".." in ref:
        return "ref may not contain a range expression"
    if not _REF_RE.match(ref):
        return "malformed ref"
    return None


def probe_ref(
    root: str,
    ref: str,
    *,
    max_drift: int = 0,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> GitProbeResult:
    """Report whether *ref* still resolves in *root*, and whether HEAD moved past it.

    Args:
        root:      Directory inside the repository to probe.
        ref:       Commit sha, tag or branch name as cited by the block.
        max_drift: Commits ``HEAD`` may sit ahead of *ref* before the
                   anchor counts as moved. ``0`` — the default — means
                   any forward movement counts.
        timeout:   Per-command timeout in seconds.

    Returns:
        A :class:`GitProbeResult` whose ``status`` is one of
        :data:`GIT_LIVE`, :data:`GIT_MISSING_REF`, :data:`GIT_MOVED` or
        :data:`GIT_UNVERIFIABLE`.

    Raises:
        ValueError: *ref* is malformed — a corpus defect the caller must
            surface, not a world-drift signal.
    """
    problem = _validate_ref(ref)
    if problem is not None:
        raise ValueError(problem)
    if max_drift < 0:
        raise ValueError("max_drift must be >= 0")

    if not is_git_repo(root, timeout=timeout):
        return GitProbeResult(GIT_UNVERIFIABLE, detail=f"{root} is not a git work tree")

    code, sha = _run(root, ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], timeout)
    if code != 0 or not sha:
        return GitProbeResult(GIT_MISSING_REF, detail=f"ref '{ref}' does not resolve in {root}")

    head_code, head = _run(root, ["rev-parse", "--verify", "--quiet", "HEAD^{commit}"], timeout)
    if head_code != 0 or not head:
        return GitProbeResult(GIT_LIVE, detail="HEAD is unborn; nothing to compare", resolved=sha)
    if head == sha:
        return GitProbeResult(GIT_LIVE, resolved=sha)

    anc_code, _ = _run(root, ["merge-base", "--is-ancestor", sha, head], timeout)
    if anc_code != 0:
        return GitProbeResult(
            GIT_MOVED,
            detail=f"ref '{ref}' is not an ancestor of HEAD",
            resolved=sha,
        )

    count_code, count_out = _run(root, ["rev-list", "--count", f"{sha}..{head}"], timeout)
    if count_code != 0 or not count_out.isdigit():
        return GitProbeResult(GIT_UNVERIFIABLE, detail="commit distance unavailable", resolved=sha)
    distance = int(count_out)
    if distance > max_drift:
        return GitProbeResult(
            GIT_MOVED,
            detail=f"HEAD is {distance} commit(s) past ref '{ref}'",
            resolved=sha,
            distance=distance,
        )
    return GitProbeResult(GIT_LIVE, resolved=sha, distance=distance)
