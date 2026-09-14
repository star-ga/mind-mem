"""Checks for the reproducible optional dependency lock."""

from __future__ import annotations

import re
from pathlib import Path

LOCK = Path(__file__).parents[1] / "requirements-optional.txt"
INPUT = Path(__file__).parents[1] / "requirements-optional.in"
ROOTS = {
    "onnxruntime": "1.24.3",
    "tokenizers": "0.23.2",
    "sentence-transformers": "6.0.1",
}
_REQUIREMENT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^ ;\\]+)")


def _stanzas() -> list[tuple[str, str, list[str]]]:
    stanzas: list[tuple[str, str, list[str]]] = []
    current: tuple[str, str, list[str]] | None = None
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        match = _REQUIREMENT.match(line)
        if match:
            if current is not None:
                stanzas.append(current)
            current = (match.group(1), match.group(2), [line])
        elif current is not None:
            current[2].append(line)
    if current is not None:
        stanzas.append(current)
    return stanzas


def test_every_locked_requirement_has_a_version_and_hash() -> None:
    stanzas = _stanzas()
    assert stanzas
    assert all(any("--hash=sha256:" in line for line in lines) for _, _, lines in stanzas)


def test_direct_roots_keep_the_pinned_versions() -> None:
    versions = {name: version for name, version, _ in _stanzas() if name in ROOTS}
    input_versions = {
        match.group(1): match.group(2) for line in INPUT.read_text(encoding="utf-8").splitlines() if (match := _REQUIREMENT.match(line))
    }
    assert input_versions == ROOTS
    assert versions == input_versions


def test_lock_contains_marker_branches_for_supported_python_profiles() -> None:
    text = LOCK.read_text(encoding="utf-8")
    assert "numpy==2.2.6 ; python_full_version < '3.11'" in text
    assert "numpy==2.4.6 ; python_full_version == '3.11.*'" in text
    assert "numpy==2.5.3 ; python_full_version >= '3.12'" in text
    assert "triton==3.8.0 ; python_full_version < '3.15' and sys_platform == 'linux'" in text
