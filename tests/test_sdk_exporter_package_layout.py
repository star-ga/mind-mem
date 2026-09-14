"""The spec exporters must work after installation outside a checkout."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mind_mem.spec import export_asyncapi
from mind_mem.spec._paths import repository_root

REPO_ROOT = Path(__file__).resolve().parents[1]


def _installed_package(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    site = tmp_path / "site-packages"
    site.mkdir()
    shutil.copytree(REPO_ROOT / "src" / "mind_mem", site / "mind_mem")
    return site, {**os.environ, "PYTHONPATH": str(site)}


def _run(site: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-P", *args],
        cwd=site.parent,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )


def test_installed_asyncapi_scans_package_and_requires_explicit_artifact(tmp_path: Path) -> None:
    site, env = _installed_package(tmp_path)
    imported = _run(site, env, "-c", "from mind_mem.spec import export_asyncapi as x; print(x.SPEC_PATH); print(x.observed_event_kinds())")
    assert imported.returncode == 0, imported.stderr
    assert "None" in imported.stdout
    assert "proposal_applied" in imported.stdout

    default_check = _run(site, env, "-m", "mind_mem.spec.export_asyncapi", "--check")
    assert default_check.returncode == 1
    assert "--input/--path" in default_check.stderr

    output = tmp_path / "generated" / "asyncapi.json"
    written = _run(site, env, "-m", "mind_mem.spec.export_asyncapi", "--write", "--output", str(output))
    assert written.returncode == 0, written.stderr
    assert output.is_file()

    checked = _run(site, env, "-m", "mind_mem.spec.export_asyncapi", "--check", "--path", str(output))
    assert checked.returncode == 0, checked.stderr
    assert "matches the live outbound event contract" in checked.stdout

    streamed = _run(site, env, "-m", "mind_mem.spec.export_asyncapi", "--write", "--stdout")
    assert streamed.returncode == 0, streamed.stderr
    assert len(json.loads(streamed.stdout)["x-mind-mem"]["observed_source_event_kinds"]) == 5


def test_missing_emitter_root_is_a_failure_not_an_empty_inventory(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="missing or not a directory"):
        export_asyncapi.observed_event_kinds(tmp_path / "does-not-exist")


def test_empty_emitter_root_is_not_a_valid_schema(tmp_path: Path) -> None:
    source = tmp_path / "mind_mem"
    source.mkdir()
    (source / "empty.py").write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no production event emitters"):
        export_asyncapi.observed_event_kinds(source)


def test_installed_package_below_checkout_does_not_bind_checkout_artifact(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    module = checkout / ".venv/lib/python3.12/site-packages/mind_mem/spec/export_asyncapi.py"
    module.parent.mkdir(parents=True)
    module.touch()
    (checkout / "pyproject.toml").touch()
    (checkout / "src/mind_mem").mkdir(parents=True)
    (checkout / "sdk/spec").mkdir(parents=True)
    assert repository_root(module) is None
