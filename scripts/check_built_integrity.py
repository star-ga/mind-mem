#!/usr/bin/env python3
"""Fail-closed verification of the integrity manifest in release archives.

The runtime protection check cannot prove that a wheel or source distribution
actually contains its manifest: an omitted data file makes an installed
editable-style package report ``checked == 0``.  This gate inspects the bytes
that will be uploaded.  It requires exactly one wheel and one sdist, exactly
one canonical manifest in each, complete coverage of the current source
package's ``_CRITICAL_MODULES``, and equality between source, manifest, and
archived critical-module bytes.

Usage::

    python scripts/check_built_integrity.py --dist dist --source-root src
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

_MANIFEST = "_integrity_manifest.json"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class IntegrityGateError(ValueError):
    """The release artifact cannot be trusted by the integrity gate."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_package(source_root: Path) -> Path:
    """Resolve either ``src`` or the package directory supplied by a caller."""
    root = source_root.resolve()
    package = root / "mind_mem"
    if package.is_dir():
        return package
    if root.name == "mind_mem" and root.is_dir():
        return root
    raise IntegrityGateError(f"source package not found below {root}")


def _critical_modules(source_root: Path) -> tuple[str, ...]:
    """Read the source-defined tuple without importing package code.

    Importing the package while checking a build would execute its runtime
    integrity hook.  The release gate needs the source authority itself, so it
    reads only the literal assignment from ``protection.py`` and refuses any
    non-literal or malformed definition.
    """
    package = _source_package(source_root)
    path = package / "protection.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise IntegrityGateError(f"cannot parse source protection module: {exc}") from exc
    values: list[Any] = []
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if value is not None and any(isinstance(t, ast.Name) and t.id == "_CRITICAL_MODULES" for t in targets):
            try:
                values.append(ast.literal_eval(value))
            except (ValueError, TypeError, SyntaxError) as exc:
                raise IntegrityGateError("_CRITICAL_MODULES must be a literal sequence") from exc
    if len(values) != 1 or not isinstance(values[0], (tuple, list)) or not values[0]:
        raise IntegrityGateError("source must define one non-empty _CRITICAL_MODULES sequence")
    modules = tuple(values[0])
    if any(not isinstance(item, str) or not item for item in modules):
        raise IntegrityGateError("_CRITICAL_MODULES entries must be non-empty strings")
    if len(set(modules)) != len(modules):
        raise IntegrityGateError("_CRITICAL_MODULES contains duplicate entries")
    return modules


def _normal_path(name: str) -> str:
    if not name or "\\" in name:
        raise IntegrityGateError(f"archive path is not canonical POSIX: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise IntegrityGateError(f"archive path is unsafe or non-canonical: {name!r}")
    normalized = path.as_posix()
    if normalized != name:
        raise IntegrityGateError(f"archive path is not canonical: {name!r}")
    return normalized


def _manifest_bytes(raw: bytes, *, archive: Path) -> dict[str, str]:
    try:
        pairs: list[tuple[str, Any]] = []

        def pairs_hook(items: list[tuple[str, Any]]) -> dict[str, Any]:
            seen: set[str] = set()
            for key, _ in items:
                if key in seen:
                    raise IntegrityGateError(f"{archive.name}: duplicate manifest key {key!r}")
                seen.add(key)
            pairs.extend(items)
            return dict(items)

        data = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs_hook)
    except IntegrityGateError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityGateError(f"{archive.name}: manifest is not valid UTF-8 JSON") from exc
    if not isinstance(data, dict) or data.get("version") != 1 or set(data) != {"version", "files"}:
        raise IntegrityGateError(f"{archive.name}: manifest must have exactly version=1 and files")
    files = data.get("files")
    if not isinstance(files, dict) or not files:
        raise IntegrityGateError(f"{archive.name}: manifest files must be non-empty")
    out: dict[str, str] = {}
    for rel, digest in files.items():
        if not isinstance(rel, str) or not isinstance(digest, str):
            raise IntegrityGateError(f"{archive.name}: manifest entries must be string pairs")
        _normal_path(rel)
        if rel.startswith("/") or rel.startswith("mind_mem/") or not _DIGEST.fullmatch(digest):
            raise IntegrityGateError(f"{archive.name}: invalid manifest entry {rel!r}")
        out[rel] = digest
    return out


def _archive_entries(path: Path) -> dict[str, bytes]:
    """Read regular archive entries once and reject duplicate names."""
    entries: dict[str, bytes] = {}
    if path.name.endswith(".whl"):
        try:
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    name = _normal_path(info.filename)
                    if name in entries:
                        raise IntegrityGateError(f"{path.name}: duplicate archive entry {name!r}")
                    if (info.external_attr >> 16) & 0o170000 == 0o120000:
                        continue
                    entries[name] = archive.read(info)
        except (OSError, zipfile.BadZipFile) as exc:
            raise IntegrityGateError(f"{path.name}: cannot read wheel: {exc}") from exc
    elif path.name.endswith(".tar.gz"):
        try:
            with tarfile.open(path, mode="r:gz") as archive:
                for info in archive.getmembers():
                    if info.isdir():
                        continue
                    name = _normal_path(info.name)
                    if name in entries:
                        raise IntegrityGateError(f"{path.name}: duplicate archive entry {name!r}")
                    if not info.isfile():
                        continue
                    handle = archive.extractfile(info)
                    if handle is None:
                        raise IntegrityGateError(f"{path.name}: cannot read regular entry {name!r}")
                    entries[name] = handle.read()
        except (OSError, tarfile.TarError) as exc:
            raise IntegrityGateError(f"{path.name}: cannot read sdist: {exc}") from exc
    else:  # pragma: no cover - callers select known suffixes
        raise IntegrityGateError(f"unsupported archive: {path.name}")
    return entries


def _manifest_location(entries: Mapping[str, bytes], archive: Path) -> tuple[str, bytes]:
    candidates = [name for name in entries if name.endswith("/" + _MANIFEST) or name == _MANIFEST]
    if len(candidates) != 1:
        raise IntegrityGateError(f"{archive.name}: expected exactly one integrity manifest, found {len(candidates)}")
    name = candidates[0]
    parts = PurePosixPath(name).parts
    if archive.name.endswith(".whl"):
        if parts != ("mind_mem", _MANIFEST):
            raise IntegrityGateError(f"{archive.name}: manifest is outside canonical wheel package")
        return "mind_mem", entries[name]
    if len(parts) < 4 or parts[-3:] != ("src", "mind_mem", _MANIFEST):
        raise IntegrityGateError(f"{archive.name}: manifest is outside canonical sdist package")
    prefix = PurePosixPath(*parts[:-3]).as_posix()
    if not prefix or "/" in prefix:
        raise IntegrityGateError(f"{archive.name}: invalid sdist package prefix")
    return f"{prefix}/src/mind_mem", entries[name]


def _verify_archive(
    archive: Path,
    source_package: Path,
    expected: Mapping[str, str],
) -> dict[str, Any]:
    entries = _archive_entries(archive)
    package_prefix, raw_manifest = _manifest_location(entries, archive)
    manifest = _manifest_bytes(raw_manifest, archive=archive)
    if set(manifest) != set(expected):
        missing = sorted(set(expected) - set(manifest))
        extra = sorted(set(manifest) - set(expected))
        raise IntegrityGateError(f"{archive.name}: critical coverage mismatch missing={missing} extra={extra}")
    checked = 0
    for rel, source_digest in expected.items():
        source_path = source_package / rel
        if not source_path.is_file():
            raise IntegrityGateError(f"source critical module missing: {rel}")
        archive_name = f"{package_prefix}/{rel}"
        if archive_name not in entries:
            raise IntegrityGateError(f"{archive.name}: critical module missing from archive: {archive_name}")
        archived_digest = _sha256(entries[archive_name])
        if manifest[rel] != source_digest:
            raise IntegrityGateError(f"{archive.name}: manifest disagrees with current source: {rel}")
        if archived_digest != source_digest:
            raise IntegrityGateError(f"{archive.name}: archived critical bytes differ from source: {rel}")
        checked += 1
    return {"archive": archive.name, "manifest": f"{package_prefix}/{_MANIFEST}", "checked": checked}


def verify_dist(
    dist: Path,
    source_root: Path,
    *,
    critical_modules: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Verify one wheel and one sdist in *dist* against current source bytes."""
    source_package = _source_package(source_root)
    modules = tuple(critical_modules) if critical_modules is not None else _critical_modules(source_root)
    if not modules or len(set(modules)) != len(modules):
        raise IntegrityGateError("critical module list must be non-empty and unique")
    expected: dict[str, str] = {}
    for rel in modules:
        _normal_path(rel)
        if rel.startswith("mind_mem/"):
            raise IntegrityGateError(f"critical module must be package-relative: {rel!r}")
        path = source_package / rel
        if not path.is_file():
            raise IntegrityGateError(f"source critical module missing: {rel}")
        expected[rel] = _sha256(path.read_bytes())
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise IntegrityGateError(f"dist must contain exactly one wheel and one sdist (wheel={len(wheels)}, sdist={len(sdists)})")
    reports = [_verify_archive(item, source_package, expected) for item in (wheels[0], sdists[0])]
    return {"status": "ok", "critical_modules": list(modules), "archives": reports}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dist", type=Path, required=True, help="directory containing exactly one wheel and one sdist")
    parser.add_argument("--source-root", type=Path, default=Path("src"), help="repository src directory or mind_mem package directory")
    args = parser.parse_args(argv)
    try:
        report = verify_dist(args.dist, args.source_root)
    except (IntegrityGateError, OSError) as exc:
        print(f"[error] built integrity gate refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
