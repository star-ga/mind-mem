# Copyright 2026 STARGA, Inc.
"""``tomllib`` compatibility for the tests, and the pyproject reader.

``tomllib`` entered the stdlib in 3.11, but ``requires-python`` is
``>=3.10`` and the CI matrix runs 3.10 — so a bare ``import tomllib``
inside a test is a ModuleNotFoundError on a supported interpreter. The
product already handles this (``check_version`` falls back to a regex);
the tests did not, and two of them turned the 3.10 job red.

``tomli`` is installed on 3.10 via the ``test`` extra, so this resolves
to a real parser on every supported runner rather than skipping. If it
somehow resolves to neither, callers get ``None`` and skip loudly — a
test that cannot read pyproject must say so, not quietly pass.
"""

from __future__ import annotations

import pathlib
from typing import Any

try:  # 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - only on 3.10
    try:
        import tomli as _toml  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover - no parser available
        _toml = None  # type: ignore[assignment]

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

__all__ = ["declared_extras", "load_pyproject", "toml_available"]


def toml_available() -> bool:
    return _toml is not None


def load_pyproject() -> dict[str, Any] | None:
    """Parsed ``pyproject.toml``, or ``None`` when it cannot be read."""
    if _toml is None:
        return None
    path = _REPO_ROOT / "pyproject.toml"
    if not path.is_file():
        return None
    return _toml.loads(path.read_text(encoding="utf-8"))


def declared_extras() -> set[str] | None:
    """Names in ``[project.optional-dependencies]``, or ``None``."""
    data = load_pyproject()
    if data is None:
        return None
    return set(data.get("project", {}).get("optional-dependencies", {}))
