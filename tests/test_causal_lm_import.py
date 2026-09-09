from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "train" / "_causal_lm_import.py"
CANONICAL = ROOT / "src" / "mind_mem" / "causal_lm_loader.py"


def _installed_stub(parent: Path, body: str = "") -> Path:
    package = parent / "mind_mem"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "causal_lm_loader.py").write_text(
        "# conflicting installed revision\n" + body,
        encoding="utf-8",
    )
    return parent


def _run_loader(helper_dir: Path, stub_dir: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(stub_dir), str(helper_dir)))
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import _causal_lm_import as h; print(h._load_module().__file__)",
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_checkout_loader_precedes_conflicting_installed_stub(tmp_path: Path) -> None:
    stub = _installed_stub(tmp_path / "installed")
    result = _run_loader(ROOT / "train", stub, ROOT)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == CANONICAL.resolve()


def test_bare_bundle_loader_precedes_conflicting_installed_stub(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    shutil.copy2(HELPER, bundle / HELPER.name)
    staged = bundle / "mind_mem_causal_lm_loader.py"
    shutil.copy2(CANONICAL, staged)
    stub = _installed_stub(tmp_path / "installed")

    result = _run_loader(bundle, stub, tmp_path)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == staged.resolve()


def test_installed_loader_is_only_the_last_fallback(tmp_path: Path) -> None:
    helper_dir = tmp_path / "bundle"
    helper_dir.mkdir()
    shutil.copy2(HELPER, helper_dir / HELPER.name)
    stub = _installed_stub(tmp_path / "installed")

    result = _run_loader(helper_dir, stub, tmp_path)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == (stub / "mind_mem" / "causal_lm_loader.py").resolve()
