"""Fail-closed controls for the wheel-bound release SBOM helper."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
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


def _target(tmp_path: Path, wheel: Path, *, names: list[str] | None = None) -> Path:
    target = tmp_path / "target"
    purelib = target / "site-packages"
    data = target / "data"
    purelib.mkdir(parents=True)
    data.mkdir()
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
    probe = {
        "name": "mind-mem",
        "version": "5.0.3",
        "dist_info": str(purelib / "mind_mem-5.0.3.dist-info"),
        "purelib": str(purelib),
        "data": str(data),
        "distributions": names if names is not None else ["mind-mem"],
    }
    interpreter = target / "python"
    interpreter.write_text("#!/bin/sh\nprintf '%s\\n' '" + json.dumps(probe).replace("'", "'\\''") + "'\n", encoding="utf-8")
    interpreter.chmod(0o755)
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
    empty = _target(tmp_path / "empty", wheel, names=[])
    with pytest.raises(_MODULE.SbomValidationError, match="no unambiguous"):
        _MODULE._verify_target_payload(wheel, "5.0.3", empty)

    target = _target(tmp_path / "drift", wheel)
    package = target.parent / "site-packages" / "mind_mem" / "__init__.py"
    package.write_bytes(b"tampered\n")
    with pytest.raises(_MODULE.SbomValidationError, match="payload differs"):
        _MODULE._verify_target_payload(wheel, "5.0.3", target)


def test_target_payload_is_bound_to_all_wheel_members(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    target = _target(tmp_path, wheel)
    result = _MODULE._verify_target_payload(wheel, "5.0.3", target)
    assert result["verified_member_count"] == result["wheel_member_count"] == 2


def test_producer_requires_expected_hash(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    assert (
        _MODULE.main(
            [
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
        )
        == 2
    )


def test_hash_or_scope_mutation_is_refused(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    document = _bom(_MODULE, wheel)
    document["metadata"]["component"]["hashes"][0]["content"] = "0" * 64
    with pytest.raises(_MODULE.SbomValidationError, match="bound to the wheel"):
        _MODULE.validate_sbom(_write_bom(tmp_path, document), wheel, "5.0.3")
