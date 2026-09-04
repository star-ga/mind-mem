# Copyright 2026 STARGA, Inc.
"""Originating-project key — provenance for lineage edges (Group S).

A lineage edge records that one block corroborates another.  It does not,
on its own, record *where that assertion came from*.  Five edges asserted
from inside one repository during one campaign are not five independent
observations; five edges arriving from five separate repositories are.
This module answers the only question that distinction needs: **which
project is a given filesystem path part of?**

Why the git *common* directory, and not the work-tree root
----------------------------------------------------------
The obvious key — the repository root (``git rev-parse --show-toplevel``)
— is wrong at exactly the margin that matters.  A linked work tree has
its own toplevel but shares its parent's object store, so keying on the
toplevel would score a repository and its own work tree as two
independent observers and *manufacture* breadth that does not exist.
Over-counting independence is worse than not scoring it at all, so the
key is the shared common directory instead::

    <repo>            toplevel=<repo>       common=<repo>/.git
    <repo>-worktree   toplevel=<repo>-wt    common=<repo>/.git   ← same key
    <other-repo>      toplevel=<other>      common=<other>/.git  ← distinct

Work trees of one repository therefore collapse to one project, while
genuinely separate repositories stay distinct.

Resolution ladder (deterministic, documented, total)
-----------------------------------------------------
``resolve_project_key`` always returns a string and never raises:

1. **In a git repository** → ``git:<absolute common dir>``.  Resolved via
   ``git rev-parse --path-format=absolute --git-common-dir``, falling
   back to the plain ``--git-common-dir`` form (absolutised against the
   probe directory) for git releases that predate ``--path-format``.
2. **git is unavailable** → the same common directory, found by walking
   the filesystem for ``.git`` (a work tree's ``.git`` is a file naming
   its parent's ``worktrees/`` path, which recovers the shared common
   dir).  Without this rung a host lacking the ``git`` binary fell to the
   path form, so one repository written from a git-having and a git-less
   environment counted as **two** projects — silently, and in the
   over-counting direction.
3. **Not a repository** → ``path:<nearest ancestor holding a project
   marker>`` (``pyproject.toml``, ``package.json``, ``Cargo.toml``,
   ``go.mod``, …).  Keying the *leaf* directory instead mints a new
   "project" per subdirectory and per fresh scratch cwd; measured, two
   subdirectories of one tree scored breadth 2 where the honest answer is
   1.  Genuinely separate projects still key distinctly.
4. **No marker anywhere above** → ``path:<absolute directory>``.  This is
   the one rung that can still over-count: two sibling directories of an
   unmarked tree read as two projects.  It is the residue after the
   marker walk, not the common case, and it is named here rather than
   claimed away.
5. **Nothing resolvable** (empty path, unreadable path, probe failure) →
   :data:`PROJECT_KEY_UNKNOWN`.  Everything here shares one bucket, which
   *under*-counts breadth.

Under-counting is the direction this module prefers, and rungs 1–3 and 5
only ever fail that way.  Rung 4 is the exception, stated rather than
hidden: a signal that claims one-sidedness it does not have is worse than
one that names its edge.

Keys longer than :data:`PROJECT_KEY_MAX_LEN` are replaced by a digest
form (``git#<hex>`` / ``path#<hex>``) so a pathological path cannot bloat
every row of the edge table.  The ``#`` separator is reserved for the
digest form, so a digest key can never collide with a literal one.

Cost
----
Resolution runs at most one short, read-only ``git`` subprocess per
directory per process; results are memoised.  The cache is keyed on the
real path of the probe directory, so a long-lived process that later sees
a repository created under a previously non-repository path keeps the old
answer until :func:`clear_project_key_cache` is called.  That trade is
deliberate: this sits on a write path and must stay cheap.

Stdlib only (``subprocess``, ``hashlib``).
"""

from __future__ import annotations

import functools
import hashlib
import os
import shutil
import subprocess  # nosec B404 — local, read-only git plumbing; fixed argv, shell=False
from typing import Final

__all__ = [
    "PROJECT_KEY_MAX_LEN",
    "PROJECT_KEY_UNKNOWN",
    "clear_project_key_cache",
    "resolve_project_key",
]

#: Key used when no project can be determined at all.  Shared by every
#: unresolvable path, which under-counts breadth rather than inventing it.
PROJECT_KEY_UNKNOWN: Final = "unknown"

#: Maximum literal key length before the digest form is used instead.
PROJECT_KEY_MAX_LEN: Final = 256

#: Wall-clock ceiling for one ``git`` probe.  A probe that overruns is a
#: failed probe, not an exception.
PROBE_TIMEOUT_SECONDS: Final = 5.0

_DIGEST_CHARS: Final = 32


def _child_env() -> dict[str, str]:
    """Environment that forbids prompting, network auth and lock-taking."""
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_ASKPASS"] = ""
    env["GCM_INTERACTIVE"] = "never"
    return env


def _run(directory: str, args: list[str]) -> tuple[int, str]:
    """Run ``git -C directory <args>``; return ``(returncode, stdout)``.

    A missing ``git`` binary, a timeout or an OS error is reported as
    returncode ``-1`` so callers degrade instead of raising.
    """
    executable = shutil.which("git")
    if executable is None:
        return -1, ""
    try:
        proc = subprocess.run(  # nosec B603 — fixed argv, shell=False, directory is a resolved local path
            [executable, "-C", directory, *args],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
            env=_child_env(),
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return -1, ""
    return proc.returncode, (proc.stdout or "").strip()


def _probe_directory(path: str | None) -> str | None:
    """Return the existing directory to probe, or ``None``.

    A file path resolves to its containing directory; ``None`` or an
    empty path resolves to the process working directory.
    """
    candidate = path if isinstance(path, str) and path.strip() else None
    try:
        if candidate is None:
            candidate = os.getcwd()
        resolved = os.path.realpath(os.path.abspath(candidate))
        if os.path.isdir(resolved):
            return resolved
        parent = os.path.dirname(resolved)
        return parent if parent and os.path.isdir(parent) else None
    except OSError:
        return None


def _git_common_dir(directory: str) -> str | None:
    """Absolute path of the shared git directory for *directory*, or ``None``."""
    code, out = _run(directory, ["rev-parse", "--path-format=absolute", "--git-common-dir"])
    if code != 0 or not out:
        # Older git has no --path-format; the plain form answers relative
        # to the -C directory, so absolutise it here.
        code, out = _run(directory, ["rev-parse", "--git-common-dir"])
        if code != 0 or not out:
            return None
        if not os.path.isabs(out):
            out = os.path.join(directory, out)
    try:
        return os.path.realpath(os.path.abspath(out))
    except OSError:
        return None


def _format_key(scheme: str, value: str) -> str:
    """Build ``scheme:value``, or a bounded ``scheme#digest`` when too long."""
    literal = f"{scheme}:{value}"
    if len(literal) <= PROJECT_KEY_MAX_LEN:
        return literal
    digest = hashlib.sha256(value.encode("utf-8", "surrogateescape")).hexdigest()[:_DIGEST_CHARS]
    return f"{scheme}#{digest}"


# Files that mark the root of a project when there is no repository. Ordered
# only for readability; the walk takes the NEAREST ancestor holding any of them.
_PROJECT_MARKERS = (
    ".git",
    ".hg",
    ".svn",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
)


def _git_common_dir_from_filesystem(directory: str) -> str | None:
    """Find the git common dir by walking the tree, with no ``git`` binary.

    The subprocess probe is authoritative, but it is not always available: on a
    host without ``git`` on PATH the old code fell straight through to the path
    form, so ONE repository written from a git-having and a git-less environment
    counted as TWO independent projects. That is the over-count direction, and
    it was silent. Reading the filesystem gives the same answer without the
    dependency.

    A work tree's ``.git`` is a FILE containing ``gitdir: <path>``, where the
    path lies under ``<common>/worktrees/<name>``; splitting there recovers the
    shared common dir, which is exactly what makes a work tree and its parent
    one project.
    """
    current = directory
    while True:
        candidate = os.path.join(current, ".git")
        if os.path.isdir(candidate):
            return os.path.realpath(candidate)
        if os.path.isfile(candidate):
            try:
                with open(candidate, encoding="utf-8", errors="replace") as handle:
                    head = handle.read(4096)
            except OSError:
                return None
            for line in head.splitlines():
                if not line.startswith("gitdir:"):
                    continue
                target = line.partition(":")[2].strip()
                if not target:
                    return None
                if not os.path.isabs(target):
                    target = os.path.join(current, target)
                marker = os.sep + "worktrees" + os.sep
                if marker in target:
                    target = target.split(marker, 1)[0]
                return os.path.realpath(target)
            return None
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def _project_marker_root(directory: str) -> str | None:
    """Nearest ancestor holding a project marker, or None.

    Keying a non-repository on the leaf DIRECTORY manufactures independence:
    one project spread across subdirectories -- or an agent whose cwd is a fresh
    per-session scratch dir -- mints a new "project" per directory, per run.
    Measured on the leaf-keyed version: two subdirectories of ONE non-git tree
    scored breadth 2 where the honest answer is 1. The marker walk keeps
    genuinely separate projects distinct without inventing observers.
    """
    current = directory
    while True:
        for marker in _PROJECT_MARKERS:
            if os.path.exists(os.path.join(current, marker)):
                return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


@functools.lru_cache(maxsize=512)
def _resolve_cached(directory: str) -> str:
    common = _git_common_dir(directory)
    if common:
        return _format_key("git", common)
    # Same question, asked of the filesystem rather than the binary.
    common = _git_common_dir_from_filesystem(directory)
    if common:
        return _format_key("git", common)
    root = _project_marker_root(directory)
    if root:
        return _format_key("path", root)
    return _format_key("path", directory)


def resolve_project_key(path: str | None = None) -> str:
    """Return the originating-project key for *path*.

    Args:
        path: Any filesystem path inside the project.  A file resolves to
            its directory.  ``None`` (the default) uses the process
            working directory, which is what identifies the project a
            session is writing *from*.

    Returns:
        ``git:<common dir>`` inside a repository, ``path:<directory>``
        outside one, or :data:`PROJECT_KEY_UNKNOWN` when nothing can be
        resolved.  Never raises, for any input.
    """
    try:
        directory = _probe_directory(path)
        if directory is None:
            return PROJECT_KEY_UNKNOWN
        return _resolve_cached(directory)
    except Exception:  # pragma: no cover - defensive: this sits on a write path
        return PROJECT_KEY_UNKNOWN


def clear_project_key_cache() -> None:
    """Drop memoised resolutions (tests, and long-lived processes)."""
    _resolve_cached.cache_clear()
