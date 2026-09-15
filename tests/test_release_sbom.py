"""Fail-closed controls for the wheel-bound release SBOM helper."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import subprocess
import venv
import zipfile
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "build_release_sbom.py"
_SPEC = importlib.util.spec_from_file_location("build_release_sbom", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _wheel(tmp_path: Path, version: str = "5.0.3") -> Path:
    path = tmp_path / f"mind_mem-{version}-py3-none-any.whl"
    dist_info = f"mind_mem-{version}.dist-info"
    members = {
        "mind_mem/__init__.py": b'__version__ = "5.0.3"\n',
        f"{dist_info}/METADATA": f"Metadata-Version: 2.1\nName: mind-mem\nVersion: {version}\n".encode(),
    }
    record = []
    for name, content in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        record.append(f"{name},sha256={digest},{len(content)}")
    record.append(f"{dist_info}/RECORD,,")
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
        archive.writestr(f"{dist_info}/RECORD", "\n".join(record) + "\n")
    return path


def _wheel_with_member(tmp_path: Path, member: str) -> Path:
    path = tmp_path / "mind_mem-5.0.3-py3-none-any.whl"
    dist_info = "mind_mem-5.0.3.dist-info"
    members = {
        member: b"unsafe member\n",
        f"{dist_info}/METADATA": b"Metadata-Version: 2.1\nName: mind-mem\nVersion: 5.0.3\n",
    }
    record = []
    for name, content in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        record.append(f"{name},sha256={digest},{len(content)}")
    record.append(f"{dist_info}/RECORD,,")
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
        archive.writestr(f"{dist_info}/RECORD", "\n".join(record) + "\n")
    return path


@pytest.mark.parametrize(
    "member",
    [
        "../outside.txt",
        "/absolute.txt",
        "\\\\server\\outside.txt",
        "mind_mem\\outside.py",
        "mind_mem/../outside.py",
        "mind_mem/./outside.py",
        "mind_mem//outside.py",
        "C:/outside.py",
        "C:outside.py",
    ],
)
def test_wheel_rejects_unsafe_member_paths(tmp_path: Path, member: str) -> None:
    wheel = _wheel_with_member(tmp_path, member)
    with pytest.raises(_MODULE.SbomValidationError, match="unsafe archive member path"):
        _MODULE._wheel_manifest(wheel, "5.0.3")


def _target(tmp_path: Path, wheel: Path, *, install: bool = True, extra: bool = False) -> Path:
    target = tmp_path / "target"
    venv.EnvBuilder(with_pip=False, clear=True).create(target)
    interpreter = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    paths = json.loads(
        subprocess.check_output(
            [
                str(interpreter),
                "-I",
                "-c",
                "import json,sysconfig; print(json.dumps({'purelib': sysconfig.get_path('purelib'), 'data': sysconfig.get_path('data')}))",
            ],
            text=True,
            encoding="utf-8",
        )
    )
    if install:
        purelib = Path(paths["purelib"])
        data = Path(paths["data"])
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.namelist():
                if member.endswith("/RECORD"):
                    continue
                if ".data/data/" in member:
                    destination = data / member.split(".data/data/", 1)[1]
                else:
                    destination = purelib / member
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
        if extra:
            extra_info = purelib / "cyclonedx_bom-7.dist-info"
            extra_info.mkdir()
            (extra_info / "METADATA").write_text("Metadata-Version: 2.1\nName: cyclonedx-bom\nVersion: 7\n", encoding="utf-8")
    return interpreter


def _bom(module, wheel: Path, *, root: dict | None = None, components: list | None = None) -> dict:
    root = root or {
        "bom-ref": "root-component",
        "name": "mind-mem",
        "type": "application",
        "version": "5.0.3",
        "hashes": [{"alg": "SHA-256", "content": module._sha256(wheel)}],
        "properties": [{"name": "mind-mem:sbom-scope", "value": "base-wheel-install-only"}],
    }
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "metadata": {"component": root},
        "components": components or [],
        "dependencies": [{"ref": "root-component"}],
    }


def _write_bom(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "sbom.cdx.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_wheel_bound_bom_is_accepted(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    result = _MODULE.validate_sbom(_write_bom(tmp_path, _bom(_MODULE, wheel)), wheel, "5.0.3")
    assert result["wheel_sha256"] == _MODULE._sha256(wheel)
    assert result["scope"] == "base-wheel-install-only"


def test_historical_environment_bom_is_refused_without_root(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    path = _write_bom(
        tmp_path,
        {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "metadata": {"tools": {"components": [{"name": "cyclonedx-py"}]}},
            "components": [{"name": "cyclonedx-bom"}],
        },
    )
    with pytest.raises(_MODULE.SbomValidationError, match="metadata.component"):
        _MODULE.validate_sbom(path, wheel, "5.0.3")


def test_generator_package_in_target_components_is_refused(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    path = _write_bom(
        tmp_path,
        _bom(_MODULE, wheel, components=[{"name": "cyclonedx-python-lib", "version": "11"}]),
    )
    with pytest.raises(_MODULE.SbomValidationError, match="generator packages"):
        _MODULE.validate_sbom(path, wheel, "5.0.3")


def test_normalized_cyclonedx_bom_target_name_is_refused(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    path = _write_bom(tmp_path, _bom(_MODULE, wheel, components=[{"name": "cyclonedx_bom"}]))
    with pytest.raises(_MODULE.SbomValidationError, match="generator packages"):
        _MODULE.validate_sbom(path, wheel, "5.0.3")


def test_empty_target_and_payload_drift_are_refused(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    empty = _target(tmp_path / "empty", wheel, install=False)
    with pytest.raises(_MODULE.SbomValidationError, match="no unambiguous"):
        _MODULE._verify_target_payload(wheel, "5.0.3", empty)

    target = _target(tmp_path / "drift", wheel)
    package = (
        Path(
            subprocess.check_output(
                [str(target), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                text=True,
                encoding="utf-8",
            ).strip()
        )
        / "mind_mem"
        / "__init__.py"
    )
    package.write_bytes(b"tampered\n")
    with pytest.raises(_MODULE.SbomValidationError, match="payload differs"):
        _MODULE._verify_target_payload(wheel, "5.0.3", target)


def test_target_payload_is_bound_to_all_wheel_members(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    target = _target(tmp_path, wheel)
    result = _MODULE._verify_target_payload(wheel, "5.0.3", target)
    assert result["verified_member_count"] == result["wheel_member_count"] == 2


def test_real_target_extra_distribution_is_refused(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    target = _target(tmp_path, wheel, extra=True)
    with pytest.raises(_MODULE.SbomValidationError, match="outside the base wheel"):
        _MODULE._verify_target_payload(wheel, "5.0.3", target)


def test_refusal_preserves_existing_output(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    target = _target(tmp_path / "empty", wheel, install=False)
    output = tmp_path / "existing.cdx.json"
    output.write_text("sentinel", encoding="utf-8")
    with pytest.raises(_MODULE.SbomValidationError, match="no unambiguous"):
        _MODULE.build_sbom(
            wheel=wheel,
            sbom=output,
            expected_version="5.0.3",
            expected_wheel_sha256=_MODULE._sha256(wheel),
            cyclonedx=tmp_path / "missing-cyclonedx",
            target_python=target,
            pyproject=tmp_path / "missing-pyproject",
        )
    assert output.read_text(encoding="utf-8") == "sentinel"


def test_producer_requires_expected_hash(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    producer_args = [
        "--wheel",
        str(wheel),
        "--sbom",
        str(tmp_path / "out.json"),
        "--version",
        "5.0.3",
        "--cyclonedx",
        str(tmp_path / "missing-cyclonedx"),
        "--target-python",
        str(tmp_path / "missing-python"),
        "--pyproject",
        str(tmp_path / "missing-pyproject"),
    ]
    assert _MODULE.main(producer_args) == 2
    assert _MODULE.main(producer_args + ["--expected-wheel-sha256", "0" * 64]) == 2


def test_hash_or_scope_mutation_is_refused(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    document = _bom(_MODULE, wheel)
    document["metadata"]["component"]["hashes"][0]["content"] = "0" * 64
    with pytest.raises(_MODULE.SbomValidationError, match="bound to the wheel"):
        _MODULE.validate_sbom(_write_bom(tmp_path, document), wheel, "5.0.3")
