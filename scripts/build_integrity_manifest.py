#!/usr/bin/env python3
"""Bake ``_integrity_manifest.json`` into the package before wheel build.

Runs once from the release workflow (``.github/workflows/release.yml``)
immediately before ``python -m build``. The manifest hashes the critical
modules listed in ``mind_mem.protection._CRITICAL_MODULES`` with SHA-256;
the runtime verifier (``mind_mem.protection.verify_integrity``)
recomputes the same hashes at import and flags any mismatch.

No-op (and exits 0) in development: editable installs never build the
manifest, so the verifier stays in fail-open mode and local edits don't
trip it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build(package_root: Path, out_path: Path | None = None) -> Path:
    sys.path.insert(0, str(package_root.parent))
    from mind_mem.protection import _CRITICAL_MODULES  # noqa: PLC0415

    files: dict[str, str] = {}
    for rel in _CRITICAL_MODULES:
        path = package_root / rel
        if not path.is_file():
            print(f"[skip] {rel} (not present)", file=sys.stderr)
            continue
        files[rel] = _sha256(path)

    manifest = {
        "version": 1,
        "files": files,
    }
    target = out_path or (package_root / "_integrity_manifest.json")
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] integrity manifest → {target} ({len(files)} files)")
    return target


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--package-root",
        default=None,
        help="Path to src/mind_mem (default: auto-detected)",
    )
    parser.add_argument("--out", default=None, help="Output path (default: inside package)")
    args = parser.parse_args(argv)

    if args.package_root:
        root = Path(args.package_root).resolve()
    else:
        here = Path(__file__).resolve().parent
        root = here.parent / "src" / "mind_mem"
    if not root.is_dir():
        print(f"[err] package root not found: {root}", file=sys.stderr)
        return 2

    out = Path(args.out).resolve() if args.out else None
    build(root, out_path=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
