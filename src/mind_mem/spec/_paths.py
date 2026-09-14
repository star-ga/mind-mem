# Copyright 2026 STARGA, Inc.
"""Resolve specification sources without assuming a repository checkout."""

from __future__ import annotations

from pathlib import Path


def package_root(module_file: str | Path) -> Path:
    """Return the installed ``mind_mem`` package directory."""
    return Path(module_file).resolve().parents[1]


def repository_root(module_file: str | Path) -> Path | None:
    """Return a verified checkout root, or ``None`` for an installed wheel.

    The marker set deliberately requires both the Python source tree and the
    SDK artifact directory.  A parent such as ``site-packages`` is therefore
    never mistaken for a checkout merely because it happens to contain a
    similarly named directory.
    """
    resolved = Path(module_file).resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "mind_mem").is_dir() and (candidate / "sdk" / "spec").is_dir():
            return candidate
    return None


def source_root(module_file: str | Path) -> Path:
    """Return the live emitter tree for a checkout or installed package."""
    repo = repository_root(module_file)
    root = repo / "src" / "mind_mem" if repo is not None else package_root(module_file)
    if not root.is_dir():
        raise RuntimeError(f"event source root is missing or not a directory: {root}")
    return root


def default_artifact(module_file: str | Path, filename: str) -> Path | None:
    """Return the checkout artifact path, never a guessed install path."""
    repo = repository_root(module_file)
    return None if repo is None else repo / "sdk" / "spec" / filename
