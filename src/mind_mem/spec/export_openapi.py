# Copyright 2026 STARGA, Inc.
"""Export and drift-check the committed OpenAPI artifact for the REST API.

Roadmap RM-2323 ("OpenAPI + AsyncAPI specs") asked for a declarative spec so
the hand-rolled clients under ``sdk/`` stop being their own source of truth.
A committed spec on its own would make things *worse*: a file that can
silently disagree with the routes the server actually serves is a contract
nobody can trust. So the deliverable is the pair — the artifact plus the
check that keeps it honest.

Two modes:

``--write``
    Regenerate ``sdk/spec/openapi.json`` (the artifact this writes) from :func:`mind_mem.api.rest.create_app`.

``--check``
    Fail (exit 1) when the committed artifact and the live app disagree.

What the check compares
-----------------------
Everything except ``info.version``. The version is deliberately excluded from
the *structural* comparison and asserted separately (see
``tests/test_sdk_openapi_drift.py``) for one reason: a package version bump
would otherwise turn every release commit red for a reason that has nothing
to do with route drift, which trains people to regenerate the artifact
without reading the diff. Structure is what a client breaks on.

Determinism
-----------
The spec is built against a throwaway workspace so no host path can leak into
a committed artifact, and ``create_app`` mutates ``MIND_MEM_WORKSPACE`` as a
documented back-compat side effect (rest.py:629), so the environment is saved
and restored around the call. Two runs in different workspaces produce
byte-identical output.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from mind_mem.spec._paths import default_artifact

# The GENERATOR lives in the package (it must import the live app); the
# ARTIFACT lives under sdk/ where an SDK consumer looks for it. So the path is
# resolved from the repository root rather than from this file's directory —
# the two deliberately no longer sit together.
_DEFAULT_ARTIFACT = default_artifact(__file__, "openapi.json")
SPEC_PATH = _DEFAULT_ARTIFACT

#: Placeholder substituted for ``info.version`` before the structural
#: comparison. Never written to the artifact — ``--write`` stores the real
#: package version so the published spec is self-describing.
_VERSION_SENTINEL = "<version>"


def build_live_spec() -> dict[str, Any]:
    """Return the OpenAPI document the running server would serve.

    Built against a fresh temporary workspace. ``MIND_MEM_WORKSPACE`` is
    restored afterwards because ``create_app`` exports it process-wide.
    """
    from mind_mem.api.rest import create_app  # noqa: PLC0415 — import cost is real

    saved = os.environ.get("MIND_MEM_WORKSPACE")
    with tempfile.TemporaryDirectory(prefix="mind-mem-openapi-") as workspace:
        try:
            spec = create_app(workspace).openapi()
        finally:
            if saved is None:
                os.environ.pop("MIND_MEM_WORKSPACE", None)
            else:
                os.environ["MIND_MEM_WORKSPACE"] = saved
    return dict(spec)


def canonical_json(spec: dict[str, Any], *, version: str) -> str:
    """Serialise *spec* deterministically with ``info.version`` pinned.

    Sorted keys and a fixed indent so a diff of two artifacts is a diff of
    the API, not of dict ordering.
    """
    document = copy.deepcopy(spec)
    info = document.get("info")
    if isinstance(info, dict):
        info["version"] = version
    return json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def load_committed_spec(path: Path | None = None) -> dict[str, Any]:
    """Parse the committed artifact. Raises if it is absent or malformed.

    ``path`` resolves ``SPEC_PATH`` at CALL time, not at definition time. A
    default argument would freeze the module-level value into the function
    signature, so redirecting ``SPEC_PATH`` — which is how the drift check's
    own positive control proves ``--check`` can fail — would silently keep
    reading the real artifact and report success.
    """
    resolved = SPEC_PATH if path is None else path
    if resolved is None:
        raise FileNotFoundError("no checkout artifact is available; pass --input or --path")
    loaded: Any = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{resolved} does not contain a JSON object")
    return loaded


def structural_diff(committed: dict[str, Any], live: dict[str, Any]) -> str:
    """Return a unified diff of the two specs, empty string when identical."""
    left = canonical_json(committed, version=_VERSION_SENTINEL)
    right = canonical_json(live, version=_VERSION_SENTINEL)
    if left == right:
        return ""
    return "".join(
        difflib.unified_diff(
            left.splitlines(keepends=True),
            right.splitlines(keepends=True),
            fromfile="sdk/spec/openapi.json (committed)",
            tofile="mind_mem.api.rest.create_app() (live)",
        )
    )


def write_spec(path: Path | None = None) -> str:
    """Regenerate the artifact from the live app. Returns the text written.

    ``path`` resolves ``SPEC_PATH`` at call time — see :func:`load_committed_spec`.
    """
    resolved = SPEC_PATH if path is None else path
    if resolved is None:
        raise RuntimeError("no checkout artifact is available; pass --output, --path, or --stdout")
    spec = build_live_spec()
    version = str(spec.get("info", {}).get("version", ""))
    text = canonical_json(spec, version=version)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")
    return text


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export or drift-check sdk/spec/openapi.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate the committed artifact")
    mode.add_argument("--check", action="store_true", help="fail when the artifact has drifted")
    artifact = parser.add_mutually_exclusive_group()
    artifact.add_argument("--input", type=Path, help="artifact to verify")
    artifact.add_argument("--output", type=Path, help="artifact to write")
    artifact.add_argument("--path", type=Path, help="artifact path (input for --check, output for --write)")
    parser.add_argument("--stdout", action="store_true", help="write the generated artifact to stdout (with --write)")
    args = parser.parse_args(argv)
    if args.stdout and not args.write:
        parser.error("--stdout requires --write")
    if args.write and args.input is not None:
        parser.error("--input is only valid with --check")
    if args.check and args.output is not None:
        parser.error("--output is only valid with --write")

    if args.write:
        if args.stdout:
            live = build_live_spec()
            sys.stdout.write(canonical_json(live, version=str(live.get("info", {}).get("version", ""))))
            return 0
        target = args.output or args.path
        try:
            write_spec(target)
        except (OSError, RuntimeError) as exc:
            print(f"CANNOT WRITE SPEC: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {target or SPEC_PATH}")
        return 0

    target = args.input or args.path
    try:
        live = build_live_spec()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"CANNOT BUILD LIVE SPEC: {exc}", file=sys.stderr)
        return 1
    try:
        committed = load_committed_spec(target)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        shown = target or SPEC_PATH or "<no checkout artifact>"
        print(f"MISSING OR INVALID: {shown}: {exc}\nPass --input/--path to verify an artifact", file=sys.stderr)
        return 1

    diff = structural_diff(committed, live)
    if diff:
        sys.stderr.write(diff)
        print(
            "\nDRIFT: the committed OpenAPI artifact no longer matches the live routes.\n"
            f"Regenerate with: python3 {Path(__file__)} --write --output <path>",
            file=sys.stderr,
        )
        return 1

    committed_version = str(committed.get("info", {}).get("version", ""))
    live_version = str(live.get("info", {}).get("version", ""))
    if committed_version != live_version:
        print(
            f"STALE VERSION: artifact says {committed_version!r}, package is {live_version!r}.\n"
            f"Regenerate with: python3 {Path(__file__)} --write --output <path>",
            file=sys.stderr,
        )
        return 1

    print(f"ok: {target or SPEC_PATH} matches the live routes (version {live_version})")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entry point
    raise SystemExit(_main())
