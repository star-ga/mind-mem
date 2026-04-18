"""Pin the runtime deprecation warning on validate.sh.

The bash validator is being replaced by ``mind_mem.validate_py`` in
v3.2.0. v3.1.x ships a runtime warning so anyone scripting the bash
engine migrates before the forwarder lands. This test guards the
warning text + the env-var opt-out so the next bash refactor doesn't
silently regress the deprecation signal.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

VALIDATE_SH = Path(__file__).parent.parent / "src" / "mind_mem" / "validate.sh"


def _bash() -> str | None:
    return shutil.which("bash")


@pytest.mark.skipif(
    sys.platform.startswith("win") or _bash() is None,
    reason="validate.sh requires bash; Windows native runs use validate_py",
)
def test_validate_sh_emits_deprecation_warning_to_stderr(tmp_path: Path) -> None:
    """Default invocation must surface the deprecation notice on stderr."""
    result = subprocess.run(
        [_bash(), str(VALIDATE_SH), str(tmp_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "MIND_MEM_VALIDATE_BASH": "0"},
    )
    assert "[mind-mem][deprecation] validate.sh is deprecated" in result.stderr
    assert "python3 -m mind_mem.validate_py" in result.stderr
    assert "MIND_MEM_VALIDATE_BASH=1" in result.stderr


@pytest.mark.skipif(
    sys.platform.startswith("win") or _bash() is None,
    reason="validate.sh requires bash; Windows native runs use validate_py",
)
def test_validate_sh_opt_out_suppresses_warning(tmp_path: Path) -> None:
    """Setting MIND_MEM_VALIDATE_BASH=1 must silence the warning."""
    result = subprocess.run(
        [_bash(), str(VALIDATE_SH), str(tmp_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "MIND_MEM_VALIDATE_BASH": "1"},
    )
    assert "[mind-mem][deprecation]" not in result.stderr
