# Copyright 2026 STARGA, Inc.
"""Locate and install the bundled ``mind-mem`` agent skill (the CLI user manual).

The skill lives at ``skills/mind-mem/`` in the source tree: a table-of-contents
``SKILL.md`` plus ``references/*.md`` for progressive discovery. A wheel ships
the same files as data files under ``share/mind-mem/skills/mind-mem`` (see
``[tool.setuptools.data-files]`` in ``pyproject.toml``), so both a source
checkout and a pip/pipx/uv install can find them.

``mm skill install`` copies the directory into an agent's skills folder
(default ``~/.claude/skills``). It never overwrites an existing, different
copy unless ``force`` is given, and ``dry_run`` reports the plan without
touching the filesystem.
"""

from __future__ import annotations

import filecmp
import os
import shutil
import sys
import sysconfig
from pathlib import Path
from typing import Any

SKILL_NAME = "mind-mem"
DEFAULT_TARGET = "~/.claude/skills"
_SHARE_SUBPATH = Path("share") / "mind-mem" / "skills" / SKILL_NAME


def _candidate_dirs() -> list[Path]:
    """Every place the bundled skill can live, most specific first."""
    candidates: list[Path] = []
    # 1. Source checkout / editable install: <repo>/src/mind_mem/ -> <repo>/skills/
    pkg_dir = Path(__file__).resolve().parent
    candidates.append(pkg_dir.parent.parent / "skills" / SKILL_NAME)
    # 2. Installed wheel data files: <scheme data root>/share/mind-mem/skills/
    roots: list[str] = []
    for scheme in (None, f"{os.name}_user"):
        try:
            path = sysconfig.get_path("data", scheme) if scheme else sysconfig.get_path("data")
        except KeyError:
            continue
        if path:
            roots.append(path)
    roots.extend([sys.prefix, sys.base_prefix])
    seen: set[str] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        candidates.append(Path(root) / _SHARE_SUBPATH)
    return candidates


def bundled_skill_dir() -> Path | None:
    """Return the directory holding the bundled ``SKILL.md``, or ``None``."""
    for cand in _candidate_dirs():
        if (cand / "SKILL.md").is_file() and (cand / "references").is_dir():
            return cand
    return None


def _skill_files(src: Path) -> list[Path]:
    """Relative paths of every Markdown file in the bundle, sorted."""
    return sorted(p.relative_to(src) for p in src.rglob("*.md") if p.is_file())


def _same_tree(src: Path, dest: Path, files: list[Path]) -> bool:
    if not dest.is_dir():
        return False
    for rel in files:
        target = dest / rel
        if not target.is_file() or not filecmp.cmp(src / rel, target, shallow=False):
            return False
    return True


def install_skill(target_root: str | os.PathLike[str] = DEFAULT_TARGET, *, force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """Copy the bundled skill to ``<target_root>/mind-mem``.

    Returns a JSON-serialisable report. ``status`` is one of ``installed``,
    ``up_to_date``, ``would_install``, ``exists`` (a different copy is present
    and ``force`` was not given) or ``missing_bundle``.
    """
    src = bundled_skill_dir()
    dest = Path(os.path.expanduser(os.fspath(target_root))) / SKILL_NAME
    report: dict[str, Any] = {"skill": SKILL_NAME, "source": str(src) if src else None, "destination": str(dest)}
    if src is None:
        report["status"] = "missing_bundle"
        report["error"] = "bundled skill files not found; reinstall mind-mem or copy skills/mind-mem from the source repository"
        return report
    files = _skill_files(src)
    report["files"] = [str(rel) for rel in files]
    if _same_tree(src, dest, files):
        report["status"] = "up_to_date"
        return report
    if dest.exists() and not force:
        report["status"] = "exists"
        report["error"] = f"{dest} already exists and differs from the bundled copy; pass --force to replace it"
        return report
    if dry_run:
        report["status"] = "would_install"
        return report
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.exists():
        shutil.rmtree(dest)
    for rel in files:
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, out)
    report["status"] = "installed"
    return report
