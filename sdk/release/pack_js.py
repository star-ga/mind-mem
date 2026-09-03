# Copyright 2026 STARGA, Inc.
"""Stage the JS client for publication, and prove the staged package coheres.

Roadmap RM-2321 asks for the npm publish of the in-tree JS client. The publish
itself needs an operator decision that is not ours to make — the manifest on
disk says ``@mind-mem/sdk`` and the roadmap says ``@star-ga/mind-mem-client``,
and an npm name is not reclaimable once taken. So what ships here is
everything up to that decision: the packaging, the version derivation, and the
gate that says the artifact is coherent.

Two properties make the source tree unable to publish itself by accident:

* ``sdk/js/package.json`` carries ``"private": true``. npm refuses to publish
  such a manifest, so no stray ``npm publish`` in that directory can claim a
  name while the name is still undecided.
* The version in that manifest is NOT the authority. It is stamped here from
  ``pyproject.toml`` at pack time, so a release cannot ship whatever number
  was last typed by hand — the manifest currently reads 0.1.0 while the
  package is 5.0.1, which is exactly the drift this removes.

Coherence, without a toolchain
------------------------------
The interesting failure is a manifest whose entry points name files the build
does not produce: ``npm publish`` succeeds, and every consumer's ``import``
fails. :func:`manifest_problems` checks that against ``tsconfig.json`` rather
than against a build directory, so the gate runs on a machine with no Node
installed — ``main``/``types``/``exports`` must live under the compiler's
``outDir``, ``declaration`` must be on for the ``.d.ts`` the manifest
promises, ``files`` must cover them, and the entry module must exist in
``rootDir``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
JS_DIR = REPO_ROOT / "sdk" / "js"
PACKAGE_JSON = JS_DIR / "package.json"
TSCONFIG = JS_DIR / "tsconfig.json"
LICENSE = REPO_ROOT / "LICENSE"

_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _load_jsonc(path: Path) -> dict[str, Any]:
    """Parse a JSON file that may carry ``//`` comments and trailing commas.

    ``tsconfig.json`` is JSONC. Only whole-line comments and trailing commas
    are tolerated — anything else is a genuine syntax error and should raise.
    """
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("//"):
            continue
        lines.append(line)
    text = _TRAILING_COMMA_RE.sub(r"\1", "\n".join(lines))
    loaded: Any = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return loaded


def load_source_manifest(path: Path | None = None) -> dict[str, Any]:
    """The manifest as committed, ``private`` flag and stale version included.

    ``path`` resolves ``PACKAGE_JSON`` at CALL time. A ``Path = PACKAGE_JSON``
    default would bake the module-level value into the signature, so
    redirecting it — which is how the positive controls prove these checks can
    fail — would silently keep reading the committed manifest and pass.
    """
    resolved = PACKAGE_JSON if path is None else path
    loaded: Any = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{resolved} does not contain a JSON object")
    return loaded


def staged_manifest(version: str, source: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the publishable manifest: version stamped, ``private`` dropped.

    Key order is preserved so a diff against the source manifest shows only
    the two intended changes.
    """
    manifest = dict(source if source is not None else load_source_manifest())
    manifest.pop("private", None)
    manifest["version"] = version
    return manifest


def manifest_problems(
    manifest: dict[str, Any],
    tsconfig: dict[str, Any] | None = None,
    js_dir: Path | None = None,
) -> list[str]:
    """Every way the staged manifest and the TypeScript build disagree."""
    js_dir = JS_DIR if js_dir is None else js_dir
    options = (tsconfig if tsconfig is not None else _load_jsonc(TSCONFIG)).get("compilerOptions", {})
    out_dir = str(options.get("outDir", "")).strip("./")
    root_dir = str(options.get("rootDir", "")).strip("./")
    declaration = bool(options.get("declaration", False))

    found: list[str] = []
    if not out_dir:
        found.append("tsconfig.json declares no outDir, so no entry point can be checked")
        return found

    entry_points: dict[str, str] = {}
    for key in ("main", "types"):
        value = manifest.get(key)
        if not isinstance(value, str):
            found.append(f"manifest has no {key!r} entry point")
        else:
            entry_points[key] = value

    exports = manifest.get("exports")
    if isinstance(exports, dict):
        root = exports.get(".")
        if isinstance(root, dict):
            for key, value in root.items():
                if isinstance(value, str):
                    entry_points[f"exports['.']['{key}']"] = value
        else:
            found.append("manifest exports has no '.' entry")
    else:
        found.append("manifest declares no exports map")

    for label, target in entry_points.items():
        relative = target.lstrip("./")
        if not relative.startswith(out_dir + "/"):
            found.append(f"{label} points at {target!r}, which is outside the compiler outDir {out_dir!r}")
        if relative.endswith(".d.ts") and not declaration:
            found.append(f"{label} promises {target!r} but tsconfig has declaration off, so no .d.ts is emitted")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        found.append("manifest has no non-empty 'files' allowlist, so the tarball contents are unbounded")
    else:
        covered = {str(entry).strip("./") for entry in files}
        if out_dir not in covered:
            found.append(f"manifest 'files' {sorted(covered)} does not include the build output {out_dir!r}")

    # The entry module must exist in the compiler's source root, otherwise the
    # build emits nothing at the path the manifest advertises.
    main = manifest.get("main")
    if isinstance(main, str) and root_dir:
        stem = main.lstrip("./")[len(out_dir) + 1 :].removesuffix(".js")
        if stem and not (js_dir / root_dir / f"{stem}.ts").is_file():
            found.append(f"main {main!r} has no source at {root_dir}/{stem}.ts")

    if manifest.get("private"):
        found.append("staged manifest still carries private:true and cannot be published")

    version = manifest.get("version")
    if not isinstance(version, str) or not version:
        found.append("staged manifest has no version")

    return found


def stage(dest: Path, version: str, require_build: bool = False) -> Path:
    """Write a publishable copy of the package into *dest*. Returns *dest*."""
    manifest = staged_manifest(version)
    found = manifest_problems(manifest)
    if found:
        raise ValueError("staged manifest is not coherent: " + "; ".join(found))

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "package.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    readme = JS_DIR / "README.md"
    if readme.is_file():
        shutil.copy2(readme, dest / "README.md")
    if LICENSE.is_file():
        shutil.copy2(LICENSE, dest / "LICENSE")

    built = JS_DIR / "dist"
    if built.is_dir():
        shutil.copytree(built, dest / "dist", dirs_exist_ok=True)
    elif require_build:
        raise FileNotFoundError(f"{built} does not exist — run `npm run build` in {JS_DIR} first")

    return dest


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage / check the npm package for the JS client")
    parser.add_argument("--check", action="store_true", help="validate the staged manifest and exit")
    parser.add_argument("--stage", metavar="DIR", help="write the publishable package into DIR")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import version as version_module  # noqa: PLC0415 — sibling module, path just set

    resolved = version_module.npm_version(version_module.read_package_version())

    manifest = staged_manifest(resolved)
    found = manifest_problems(manifest)
    for problem in found:
        print(f"PROBLEM: {problem}", file=sys.stderr)
    if found:
        return 1

    print(f"staged manifest ok: {manifest.get('name')}@{manifest['version']}")
    if args.stage:
        target = stage(Path(args.stage), resolved)
        print(f"staged into {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entry point
    raise SystemExit(_main())
