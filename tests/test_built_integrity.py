"""Source-bound controls for the wheel/sdist integrity release gate."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import tarfile
import zipfile
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "check_built_integrity.py"
_SPEC = importlib.util.spec_from_file_location("check_built_integrity", _SCRIPT)
assert _SPEC and _SPEC.loader
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)


def _source(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    package = tmp_path / "src" / "mind_mem"
    package.mkdir(parents=True)
    (package / "protection.py").write_text(
        "_CRITICAL_MODULES = ('recall.py', 'storage/sharded_pg.py')\n", encoding="utf-8"
    )
    files = {"recall.py": b"source recall\n", "storage/sharded_pg.py": b"source pg\n"}
    for rel, data in files.items():
        path = package / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return tmp_path / "src", {rel: hashlib.sha256(data).hexdigest() for rel, data in files.items()}


def _manifest(digests: dict[str, str], *, extra: dict[str, str] | None = None) -> bytes:
    files = dict(digests)
    if extra:
        files.update(extra)
    return json.dumps({"version": 1, "files": files}, sort_keys=True).encode() + b"\n"


def _archives(
    tmp_path: Path,
    source_digests: dict[str, str],
    *,
    manifest: bytes | None = None,
    wheel_files: dict[str, bytes] | None = None,
    omit_sdist: bool = False,
    duplicate_wheel_name: str | None = None,
    symlink_wheel_name: str | None = None,
) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    manifest = manifest or _manifest(source_digests)
    wheel = dist / "mind_mem-1.0-py3-none-any.whl"
    wf = {
        "mind_mem/recall.py": b"source recall\n",
        "mind_mem/storage/sharded_pg.py": b"source pg\n",
        "mind_mem/_integrity_manifest.json": manifest,
        "mind_mem-1.0.dist-info/METADATA": b"Name: mind-mem\n",
    }
    for name, data in (wheel_files or {}).items():
        if data is None:
            wf.pop(name, None)
        else:
            wf[name] = data
    with zipfile.ZipFile(wheel, "w") as zf:
        for name, data in wf.items():
            zf.writestr(name, data)
        if duplicate_wheel_name:
            zf.writestr(duplicate_wheel_name, b"duplicate")
        if symlink_wheel_name:
            link = zipfile.ZipInfo(symlink_wheel_name)
            link.external_attr = 0o120777 << 16
            zf.writestr(link, b"target")
    if not omit_sdist:
        sdist = dist / "mind_mem-1.0.tar.gz"
        with tarfile.open(sdist, "w:gz") as tf:
            for name in ("mind_mem-1.0/", "mind_mem-1.0/src/", "mind_mem-1.0/src/mind_mem/"):
                directory = tarfile.TarInfo(name)
                directory.type = tarfile.DIRTYPE
                tf.addfile(directory)
            for name, data in {
                "mind_mem-1.0/src/mind_mem/recall.py": b"source recall\n",
                "mind_mem-1.0/src/mind_mem/storage/sharded_pg.py": b"source pg\n",
                "mind_mem-1.0/src/mind_mem/_integrity_manifest.json": manifest,
            }.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
    return dist


def test_realistic_wheel_and_sdist_pass(tmp_path: Path) -> None:
    source, digests = _source(tmp_path)
    report = gate.verify_dist(_archives(tmp_path, digests), source)
    assert report["status"] == "ok"
    assert [item["checked"] for item in report["archives"]] == [2, 2]
    assert report["critical_modules"] == ["recall.py", "storage/sharded_pg.py"]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"manifest": b'{"version":1,"files":{"recall.py":"' + b"0" * 64 + b'"}}'}, "coverage mismatch"),
        ({"wheel_files": {"mind_mem/recall.py": b"tampered\n"}}, "differ from source"),
        (
            {"manifest": b'{"version":1,"files":{"recall.py":"' + b"f" * 64 + b'","storage/sharded_pg.py":"' + b"0" * 64 + b'"}}'},
            "disagrees with current source",
        ),
        ({"omit_sdist": True}, "exactly one wheel and one sdist"),
    ],
)
def test_semantic_archive_failures_are_refused(tmp_path: Path, kwargs: dict[str, object], match: str) -> None:
    source, digests = _source(tmp_path)
    with pytest.raises(gate.IntegrityGateError, match=match):
        gate.verify_dist(_archives(tmp_path, digests, **kwargs), source)


def test_duplicate_archive_entry_is_refused(tmp_path: Path) -> None:
    source, digests = _source(tmp_path)
    with pytest.raises(gate.IntegrityGateError, match="duplicate archive entry"):
        gate.verify_dist(_archives(tmp_path, digests, duplicate_wheel_name="mind_mem/recall.py"), source)


def test_symlink_archive_entry_is_refused(tmp_path: Path) -> None:
    source, digests = _source(tmp_path)
    with pytest.raises(gate.IntegrityGateError, match="non-regular archive entry"):
        gate.verify_dist(_archives(tmp_path, digests, symlink_wheel_name="mind_mem/link.py"), source)


@pytest.mark.skipif(os.name == "nt", reason="source symlink control requires POSIX symlink support")
def test_source_critical_symlink_is_refused(tmp_path: Path) -> None:
    source, digests = _source(tmp_path)
    critical = source / "mind_mem" / "recall.py"
    outside = tmp_path / "outside.py"
    critical.rename(outside)
    critical.symlink_to(outside)
    with pytest.raises(gate.IntegrityGateError, match="outside package"):
        gate.verify_dist(_archives(tmp_path, digests), source)


def test_manifest_size_limit_is_refused(tmp_path: Path) -> None:
    source, digests = _source(tmp_path)
    oversized = b" " * (gate._MAX_MANIFEST_BYTES + 1)
    with pytest.raises(gate.IntegrityGateError, match="manifest exceeds size limit"):
        gate.verify_dist(_archives(tmp_path, digests, manifest=oversized), source)


@pytest.mark.parametrize("limit", [1, 4])
def test_archive_entry_count_limit_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    source, digests = _source(tmp_path)
    monkeypatch.setattr(gate, "_MAX_ARCHIVE_ENTRIES", limit)
    with pytest.raises(gate.IntegrityGateError, match="entry count exceeds limit"):
        gate.verify_dist(_archives(tmp_path, digests), source)


def test_archive_directory_count_limit_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, digests = _source(tmp_path)
    monkeypatch.setattr(gate, "_MAX_ARCHIVE_DIRECTORIES", 2)
    with pytest.raises(gate.IntegrityGateError, match="directory count exceeds limit"):
        gate.verify_dist(_archives(tmp_path, digests), source)


def test_missing_manifest_is_refused(tmp_path: Path) -> None:
    source, digests = _source(tmp_path)
    with pytest.raises(gate.IntegrityGateError, match="exactly one integrity manifest"):
        gate.verify_dist(
            _archives(tmp_path, digests, wheel_files={"mind_mem/_integrity_manifest.json": None}),  # type: ignore[arg-type]
            source,
        )
