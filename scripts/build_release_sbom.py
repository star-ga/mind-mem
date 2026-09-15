#!/usr/bin/env python3
"""Generate and validate the release SBOM for the built wheel.

CycloneDX runs from a separate tool environment.  The target interpreter must
contain the wheel being released and the wheel's base dependencies only; this
keeps the generator itself out of the product inventory.  Optional extras are
outside this artifact's scope.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import ntpath
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


class SbomValidationError(ValueError):
    """Raised when an SBOM is not bound to the supplied release wheel."""


_WHEEL_NAME = re.compile(r"^mind_mem-(?P<version>[^-]+)-.+\.whl$")
_FORBIDDEN_TARGET_COMPONENTS = {
    "cyclonedx-bom",
    "cyclonedx-py",
    "cyclonedx-python-lib",
}
_GENERATOR_VERSION_ARGUMENT = "--version"


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    return environment


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_wheel_member_path(path: str) -> None:
    """Reject archive names that can escape or alias their install path."""
    parts = path.split("/")
    if (
        not path
        or path.startswith(("/", "\\"))
        or "\\" in path
        or "\x00" in path
        or any(part in {"", ".", ".."} for part in parts)
        or ntpath.splitdrive(path)[0]
    ):
        raise SbomValidationError(f"wheel contains unsafe archive member path: {path!r}")


def _wheel_manifest(wheel: Path, expected_version: str) -> tuple[str, str, dict[str, bytes]]:
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise SbomValidationError(f"wheel is not a regular .whl file: {wheel}")
    match = _WHEEL_NAME.fullmatch(wheel.name)
    if match is None or match.group("version") != expected_version:
        raise SbomValidationError(f"wheel filename must be mind_mem-{expected_version}-<tags>.whl: {wheel.name}")
    try:
        with zipfile.ZipFile(wheel) as archive:
            names = []
            for entry in archive.infolist():
                # ZipInfo normalizes Windows separators and truncates NULs.
                # Check the raw name before any normalized-key lookup can
                # hide it or alias another member in NameToInfo.
                _validate_wheel_member_path(entry.orig_filename)
                if entry.filename != entry.orig_filename:
                    raise SbomValidationError(
                        f"wheel contains unsafe archive member path normalization: {entry.orig_filename!r} -> {entry.filename!r}"
                    )
                names.append(entry.filename)
            if len(names) != len(set(names)):
                raise SbomValidationError("wheel contains duplicate archive members")
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(metadata_names) != 1:
                raise SbomValidationError("wheel must contain exactly one dist-info/METADATA")
            metadata = archive.read(metadata_names[0]).decode("utf-8")
            dist_info = metadata_names[0].rsplit("/", 1)[0]
            record_name = f"{dist_info}/RECORD"
            if record_name not in names:
                raise SbomValidationError("wheel has no matching dist-info/RECORD")
            record_rows = list(csv.reader(archive.read(record_name).decode("utf-8").splitlines(), strict=True))
            if any(len(row) != 3 or not row[0] for row in record_rows):
                raise SbomValidationError("wheel RECORD contains a malformed row")
            record_paths = [row[0] for row in record_rows]
            if len(record_paths) != len(set(record_paths)) or set(record_paths) != set(names):
                raise SbomValidationError("wheel RECORD does not exactly describe archive members")
            payload: dict[str, bytes] = {}
            for path, encoded_hash, encoded_size in record_rows:
                content = archive.read(path)
                if path == record_name:
                    if encoded_hash or encoded_size:
                        raise SbomValidationError("wheel RECORD row for RECORD is invalid")
                    continue
                if not encoded_hash.startswith("sha256="):
                    raise SbomValidationError(f"wheel member lacks a SHA-256 RECORD hash: {path}")
                digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")
                if encoded_hash.removeprefix("sha256=") != digest or encoded_size != str(len(content)):
                    raise SbomValidationError(f"wheel RECORD hash/size mismatch: {path}")
                payload[path] = content
    except (OSError, KeyError, csv.Error, zipfile.BadZipFile, UnicodeDecodeError) as exc:
        raise SbomValidationError(f"cannot read wheel metadata: {exc}") from exc
    from email.parser import Parser

    headers = Parser().parsestr(metadata, headersonly=True)
    names = headers.get_all("Name", [])
    versions = headers.get_all("Version", [])
    if names != ["mind-mem"] or versions != [expected_version]:
        raise SbomValidationError(f"wheel metadata does not match the requested mind-mem release ({names!r}, {versions!r})")
    return "mind-mem", expected_version, payload


def _wheel_identity(wheel: Path, expected_version: str) -> tuple[str, str]:
    name, version, _ = _wheel_manifest(wheel, expected_version)
    return name, version


def _target_installation(target_python: Path) -> dict[str, Any]:
    """Read the target interpreter's installed distributions in isolation."""

    probe = (
        "import importlib.metadata as m, json, sys, sysconfig\n"
        "names = [d.metadata.get('Name', '') for d in m.distributions()]\n"
        "matches = [d for d in m.distributions() if d.metadata.get('Name', '').lower().replace('_', '-') == 'mind-mem']\n"
        "if len(matches) != 1: raise SystemExit('expected exactly one installed mind-mem distribution')\n"
        "d = matches[0]\n"
        "print(json.dumps({"
        "'name': d.metadata.get('Name'), 'version': d.version, "
        "'dist_info': str(d._path), 'purelib': sysconfig.get_path('purelib'), "
        "'data': sysconfig.get_path('data'), 'distributions': names}))\n"
    )
    try:
        completed = subprocess.run(
            [str(target_python), "-I", "-c", probe],
            check=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=_clean_environment(),
        )
        result = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise SbomValidationError(f"target interpreter has no unambiguous installed mind-mem wheel: {exc}") from exc
    if not isinstance(result, dict):
        raise SbomValidationError("target installation probe returned a non-object")
    if result.get("name") != "mind-mem" or not isinstance(result.get("version"), str):
        raise SbomValidationError("target installation identity is not mind-mem")
    distributions = result.get("distributions")
    if not isinstance(distributions, list) or any(not isinstance(name, str) for name in distributions):
        raise SbomValidationError("target installation distribution inventory is malformed")
    normalized = {_normalize_distribution_name(name) for name in distributions}
    if not normalized:
        raise SbomValidationError("target interpreter has no unambiguous installed mind-mem wheel")
    if normalized != {"mind-mem"}:
        raise SbomValidationError("target environment contains packages outside the base wheel: " + ", ".join(sorted(normalized)))
    for key in ("dist_info", "purelib", "data"):
        if not isinstance(result.get(key), str) or not result[key]:
            raise SbomValidationError(f"target installation probe has no {key} path")
    return result


def _verify_target_payload(wheel: Path, expected_version: str, target_python: Path) -> dict[str, Any]:
    name, version, payload = _wheel_manifest(wheel, expected_version)
    installation = _target_installation(target_python)
    if installation["version"] != version:
        raise SbomValidationError(f"target mind-mem version {installation['version']!r} does not match wheel {version!r}")
    purelib = Path(installation["purelib"]).resolve()
    data_root = Path(installation["data"]).resolve()
    dist_info = next(path for path in payload if ".dist-info/" in path).split("/", 1)[0]
    wheel_data = dist_info.removesuffix(".dist-info") + ".data"
    installed_dist_info = Path(installation["dist_info"]).resolve()
    if installed_dist_info != (purelib / dist_info).resolve():
        raise SbomValidationError("target mind-mem distribution is outside its expected wheel site-packages")
    verified = 0
    for member, expected_bytes in payload.items():
        if member.startswith(f"{wheel_data}/data/"):
            target_path = data_root / member.removeprefix(f"{wheel_data}/data/")
        elif member.startswith(f"{wheel_data}/"):
            raise SbomValidationError(f"unsupported wheel data scheme: {member}")
        else:
            target_path = purelib / member
        if not target_path.is_file():
            raise SbomValidationError(f"target installation is missing wheel member: {member}")
        try:
            installed_bytes = target_path.read_bytes()
        except OSError as exc:
            raise SbomValidationError(f"target installation member is unreadable: {member}") from exc
        if installed_bytes != expected_bytes:
            raise SbomValidationError(f"target installation payload differs from wheel member: {member}")
        verified += 1
    return {
        "name": name,
        "version": version,
        "wheel_member_count": len(payload),
        "verified_member_count": verified,
        "target_distributions": sorted({_normalize_distribution_name(item) for item in installation["distributions"]}),
    }


def _component_hashes(component: dict[str, Any]) -> set[str]:
    hashes = component.get("hashes", [])
    if not isinstance(hashes, list):
        raise SbomValidationError("SBOM root component hashes must be a JSON array")
    return {str(entry.get("content", "")).lower() for entry in hashes if isinstance(entry, dict) and entry.get("alg") == "SHA-256"}


def validate_sbom(
    sbom: Path,
    wheel: Path,
    expected_version: str,
    *,
    expected_wheel_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a BOM's root identity and artifact binding, failing closed."""

    name, version = _wheel_identity(wheel, expected_version)
    wheel_sha256 = _sha256(wheel)
    if expected_wheel_sha256 is not None and wheel_sha256 != expected_wheel_sha256.lower():
        raise SbomValidationError(f"wheel SHA-256 mismatch: expected {expected_wheel_sha256}, got {wheel_sha256}")
    try:
        document = json.loads(sbom.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SbomValidationError(f"invalid SBOM JSON: {exc}") from exc
    if not isinstance(document, dict) or document.get("bomFormat") != "CycloneDX":
        raise SbomValidationError("SBOM is not a CycloneDX object")
    metadata = document.get("metadata")
    component = metadata.get("component") if isinstance(metadata, dict) else None
    if not isinstance(component, dict):
        raise SbomValidationError("SBOM metadata.component is missing")
    if component.get("name") != name or component.get("version") != version:
        raise SbomValidationError(
            f"SBOM root component does not match the release wheel: {component.get('name')!r} {component.get('version')!r}"
        )
    if wheel_sha256 not in _component_hashes(component):
        raise SbomValidationError("SBOM root component is not bound to the wheel SHA-256")
    properties = component.get("properties", [])
    if not isinstance(properties, list):
        raise SbomValidationError("SBOM root component properties must be a JSON array")
    scope = {entry.get("value") for entry in properties if isinstance(entry, dict) and entry.get("name") == "mind-mem:sbom-scope"}
    if scope != {"base-wheel-install-only"}:
        raise SbomValidationError("SBOM scope must state base-wheel-install-only; optional extras are not covered")
    components = document.get("components", [])
    if not isinstance(components, list):
        raise SbomValidationError("SBOM components must be a JSON array")
    if any(not isinstance(entry, dict) for entry in components):
        raise SbomValidationError("SBOM components must contain JSON objects")
    if any(
        isinstance(entry, dict) and _normalize_distribution_name(str(entry.get("name", ""))) in _FORBIDDEN_TARGET_COMPONENTS
        for entry in components
    ):
        raise SbomValidationError("SBOM includes CycloneDX generator packages in target components")
    root_ref = component.get("bom-ref")
    if not isinstance(root_ref, str) or not root_ref:
        raise SbomValidationError("SBOM root component has no bom-ref")
    dependencies = document.get("dependencies")
    if not isinstance(dependencies, list):
        raise SbomValidationError("SBOM dependencies must be a JSON array")
    if any(not isinstance(entry, dict) for entry in dependencies):
        raise SbomValidationError("SBOM dependencies must contain JSON objects")
    if not any(isinstance(entry, dict) and entry.get("ref") == root_ref for entry in dependencies):
        raise SbomValidationError("SBOM dependency graph does not reference its root component")
    return {
        "wheel": wheel.name,
        "wheel_sha256": wheel_sha256,
        "version": version,
        "root_component": component.get("name"),
        "target_component_count": len(components),
        "scope": "base-wheel-install-only",
    }


def _run_generator(
    *,
    cyclonedx: Path,
    target_python: Path,
    pyproject: Path,
    output: Path,
) -> None:
    if not cyclonedx.is_file() or not os.access(cyclonedx, os.X_OK):
        raise SbomValidationError(f"CycloneDX executable is missing or not executable: {cyclonedx}")
    if not target_python.is_file():
        raise SbomValidationError(f"target environment interpreter is missing: {target_python}")
    if not pyproject.is_file():
        raise SbomValidationError(f"pyproject is missing: {pyproject}")
    try:
        subprocess.run(
            [
                str(cyclonedx),
                "environment",
                str(target_python),
                "--pyproject",
                str(pyproject),
                "--output-format",
                "JSON",
                "--output-file",
                str(output),
            ],
            check=True,
            stdin=subprocess.DEVNULL,
            env=_clean_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SbomValidationError(f"CycloneDX generation failed: {exc}") from exc


def _measure_generator_version(cyclonedx: Path) -> str:
    try:
        completed = subprocess.run(
            [str(cyclonedx), _GENERATOR_VERSION_ARGUMENT],
            check=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=_clean_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SbomValidationError(f"cannot measure CycloneDX executable version: {exc}") from exc
    version = completed.stdout.strip()
    if not version or "\n" in version:
        raise SbomValidationError("CycloneDX executable returned no single version")
    return version


def build_sbom(
    *,
    wheel: Path,
    sbom: Path,
    expected_version: str,
    expected_wheel_sha256: str,
    cyclonedx: Path,
    target_python: Path,
    pyproject: Path,
) -> dict[str, Any]:
    name, version, _ = _wheel_manifest(wheel, expected_version)
    wheel_sha256 = _sha256(wheel)
    if wheel_sha256 != expected_wheel_sha256.lower():
        raise SbomValidationError(f"wheel SHA-256 mismatch: expected {expected_wheel_sha256}, got {wheel_sha256}")
    target_receipt = _verify_target_payload(wheel, expected_version, target_python)
    generator_version = _measure_generator_version(cyclonedx)
    sbom.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{sbom.name}.", suffix=".tmp", dir=sbom.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        _run_generator(
            cyclonedx=cyclonedx,
            target_python=target_python,
            pyproject=pyproject,
            output=temporary,
        )
        document = json.loads(temporary.read_text(encoding="utf-8"))
        component = document["metadata"]["component"]
        if not isinstance(component, dict):
            raise TypeError("metadata.component is not an object")
        existing_properties = component.get("properties", [])
        if not isinstance(existing_properties, list):
            raise TypeError("metadata.component.properties is not an array")
        component["hashes"] = [{"alg": "SHA-256", "content": wheel_sha256}]
        component["properties"] = [
            *[entry for entry in existing_properties if isinstance(entry, dict)],
            {"name": "mind-mem:sbom-scope", "value": "base-wheel-install-only"},
            {"name": "mind-mem:wheel-filename", "value": wheel.name},
        ]
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = validate_sbom(temporary, wheel, expected_version, expected_wheel_sha256=wheel_sha256)
        os.replace(temporary, sbom)
    except (OSError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SbomValidationError(f"CycloneDX output has no usable root component: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return result | {"generator": "cyclonedx-py", "generator_version": generator_version, "target": target_receipt}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sbom", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--expected-wheel-sha256")
    parser.add_argument("--cyclonedx", type=Path)
    parser.add_argument("--target-python", type=Path)
    parser.add_argument("--pyproject", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.validate_only:
            if any(value is not None for value in (args.cyclonedx, args.target_python, args.pyproject)):
                raise SbomValidationError("--validate-only cannot be combined with generator paths")
            result = validate_sbom(
                args.sbom,
                args.wheel,
                args.version,
                expected_wheel_sha256=args.expected_wheel_sha256,
            )
        else:
            if None in (args.cyclonedx, args.target_python, args.pyproject) or args.expected_wheel_sha256 is None:
                raise SbomValidationError("producer mode requires --cyclonedx, --target-python, --pyproject, and --expected-wheel-sha256")
            result = build_sbom(
                wheel=args.wheel,
                sbom=args.sbom,
                expected_version=args.version,
                expected_wheel_sha256=args.expected_wheel_sha256,
                cyclonedx=args.cyclonedx,
                target_python=args.target_python,
                pyproject=args.pyproject,
            )
    except SbomValidationError as exc:
        print(f"SBOM REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
