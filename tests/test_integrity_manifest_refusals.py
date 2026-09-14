"""An absent, incomplete or substituted manifest cannot satisfy strict mode."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mind_mem import protection


@pytest.fixture
def package(tmp_path, monkeypatch):
    root = tmp_path / "package"
    root.mkdir()
    for name in ("recall.py", "apply_engine.py"):
        (root / name).write_text(f"# {name}\n", encoding="utf-8")
    monkeypatch.setattr(protection, "_package_root", lambda: root)
    monkeypatch.setattr(protection, "_CRITICAL_MODULES", ("recall.py", "apply_engine.py"))
    monkeypatch.delenv("MIND_MEM_INTEGRITY", raising=False)
    return root


def manifest(root):
    return {"version": 1, "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in protection._CRITICAL_MODULES}}


def store(root, value):
    (root / protection._MANIFEST_FILENAME).write_text(json.dumps(value), encoding="utf-8")


def test_strict_missing_manifest_refuses_while_default_development_remains_available(package, monkeypatch):
    assert protection.verify_integrity().ok
    monkeypatch.setenv("MIND_MEM_INTEGRITY", "strict")
    with pytest.raises(RuntimeError, match="requires an integrity manifest"):
        protection.verify_integrity()
    store(package, manifest(package))
    result = protection.verify_integrity()
    assert result.ok and result.manifest_present and result.checked == 2


@pytest.mark.parametrize("removed_file", [False, True])
def test_omitted_critical_entry_refuses_even_when_file_was_also_removed(package, monkeypatch, removed_file):
    value = manifest(package)
    del value["files"]["apply_engine.py"]
    store(package, value)
    if removed_file:
        (package / "apply_engine.py").unlink()
    result = protection.verify_integrity()
    assert not result.ok and result.extra == ("apply_engine.py",)
    monkeypatch.setenv("MIND_MEM_INTEGRITY", "strict")
    with pytest.raises(RuntimeError, match="integrity check failed"):
        protection.verify_integrity()


@pytest.mark.parametrize(
    "bad",
    [{"version": 1, "files": {}}, {"version": True, "files": {"recall.py": "a" * 64}}, {"version": 2, "files": {"recall.py": "a" * 64}}],
)
def test_invalid_manifest_schema_refuses(package, monkeypatch, bad):
    store(package, bad)
    monkeypatch.setenv("MIND_MEM_INTEGRITY", "strict")
    with pytest.raises(RuntimeError, match="malformed"):
        protection.verify_integrity()


@pytest.mark.parametrize("name", ["../outside.py", "/absolute.py", "dir//file.py", "dir/../file.py", "dir\\file.py", "C:outside.py"])
def test_manifest_paths_cannot_escape_or_alias(package, monkeypatch, name):
    value = manifest(package)
    value["files"][name] = "a" * 64
    store(package, value)
    monkeypatch.setenv("MIND_MEM_INTEGRITY", "strict")
    with pytest.raises(RuntimeError, match="noncanonical module path"):
        protection.verify_integrity()


@pytest.mark.parametrize("digest", [None, 42, "", "a" * 63, "z" * 64])
def test_malformed_digest_is_not_silently_discarded(package, monkeypatch, digest):
    value = manifest(package)
    value["files"]["recall.py"] = digest
    store(package, value)
    monkeypatch.setenv("MIND_MEM_INTEGRITY", "strict")
    with pytest.raises(RuntimeError, match="invalid module digest"):
        protection.verify_integrity()


def test_duplicate_manifest_keys_refuse(package, monkeypatch):
    raw = json.dumps(manifest(package)).replace('"version": 1', '"version": 2, "version": 1')
    (package / protection._MANIFEST_FILENAME).write_text(raw, encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_INTEGRITY", "strict")
    with pytest.raises(RuntimeError, match="unreadable"):
        protection.verify_integrity()


def test_manifest_size_limit_refuses(package, monkeypatch):
    (package / protection._MANIFEST_FILENAME).write_bytes(b" " * (protection._MAX_MANIFEST_BYTES + 1))
    monkeypatch.setenv("MIND_MEM_INTEGRITY", "strict")
    with pytest.raises(RuntimeError, match="size limit"):
        protection.verify_integrity()


@pytest.mark.skipif(os.name != "posix", reason="requires unprivileged symlinks")
@pytest.mark.parametrize("replace_manifest", [False, True])
def test_symlink_substitution_refuses_even_with_matching_bytes(package, monkeypatch, replace_manifest):
    store(package, manifest(package))
    selected = package / (protection._MANIFEST_FILENAME if replace_manifest else "recall.py")
    outside = package.parent / "outside"
    selected.rename(outside)
    selected.symlink_to(outside)
    monkeypatch.setenv("MIND_MEM_INTEGRITY", "strict")
    with pytest.raises(RuntimeError, match="integrity check failed"):
        protection.verify_integrity()


def test_unreadable_critical_module_refuses(package, monkeypatch):
    store(package, manifest(package))

    def unreadable(_path):
        raise OSError("unreadable control")

    monkeypatch.setattr(protection, "_sha256", unreadable)
    monkeypatch.setenv("MIND_MEM_INTEGRITY", "strict")
    with pytest.raises(RuntimeError, match="critical module is unreadable"):
        protection.verify_integrity()


def test_builder_refuses_missing_critical_file_before_output_replacement(package):
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_integrity_manifest.py"
    spec = importlib.util.spec_from_file_location("test_integrity_builder", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output = package / protection._MANIFEST_FILENAME
    builder.build(package)
    before = output.read_bytes()
    (package / "apply_engine.py").unlink()
    with pytest.raises(ValueError, match="critical module missing"):
        builder.build(package)
    assert output.read_bytes() == before


def test_import_does_not_swallow_unexpected_verifier_failure_in_strict_mode():
    code = """
import importlib, os
import mind_mem
from mind_mem import protection
def broken():
    raise OSError('control')
protection.verify_integrity = broken
os.environ['MIND_MEM_INTEGRITY'] = 'strict'
try:
    importlib.reload(mind_mem)
except RuntimeError as exc:
    assert 'verifier failed (strict mode)' in str(exc)
else:
    raise AssertionError('strict import swallowed verifier failure')
"""
    # Bind the child to this test's source tree, not an unrelated editable
    # installation inherited from the developer's interpreter.
    source = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ, MIND_MEM_INTEGRITY="off", PYTHONPATH=str(source))
    completed = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=20, encoding="utf-8")
    assert completed.returncode == 0, completed.stderr
