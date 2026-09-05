#!/usr/bin/env python3
"""Write a repro package: raw rows, recomputed metrics, and a manifest that pins
everything a third party needs to get the same answer.

A repro package is a directory:

    raw.ndjson        one JSON object per unit of work -- the evidence. Every
                      published number for this benchmark is recomputable from
                      this file alone (see ``benchmarks/repro_metrics.py``).
    metrics.json      the numbers, produced by running the metric function over
                      raw.ndjson. Not authored by hand, ever.
    dataset.json      the exact pinned inputs, with a content hash -- so
                      "which dataset" is checkable rather than asserted.
    environment.json  hardware / OS / python / package versions / wall clock.
    manifest.json     the pins (commit, config + sha256, seeds, adapter,
                      embedder, k, exclusions, headline) plus the sha256 of
                      every file above, plus the two commands: one that
                      regenerates the package, one that verifies it.

``manifest.json`` is the only file not covered by its own hash, because it
carries the hashes. Everything else is content-addressed, so a package whose
raw rows were edited fails the very first check in the verifier.

Copyright (c) STARGA Inc. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any

from benchmarks.repro_metrics import SCHEMA

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Files that a rerun on a different box must reproduce byte-for-byte. The
#: others record measurement facts (timings, hardware) that legitimately vary;
#: calling them stable would be the sort of overclaim this package prevents.
RUN_TO_RUN_STABLE = ("dataset.json",)


def canonical_json(obj: Any) -> bytes:
    """Sorted-key, separator-pinned, newline-terminated JSON. Hashable, diffable."""
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(obj: Any) -> str:
    """Hash of an object's canonical form -- an identity, not a filename."""
    return sha256_bytes(canonical_json(obj))


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def git_tracked_dirty() -> bool:
    """True when a TRACKED file differs from the pinned commit.

    A package produced from a modified tree is not reproducible from its own
    commit id, so the fact is recorded rather than discovered later. Untracked
    files are excluded on purpose: the package being written is itself untracked
    while it is being written, and counting it would make every first
    generation report dirty for a reason that has nothing to do with the code
    that produced it.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else True
    except Exception:
        return True


def pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "absent"


def environment_facts(packages: tuple[str, ...] = ()) -> dict[str, Any]:
    """Hardware / OS / interpreter / dependency versions. Descriptive only."""
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "mind_mem_version": pkg_version("mind-mem"),
        "packages": {p: pkg_version(p) for p in packages},
    }


def build_manifest(
    *,
    benchmark: str,
    headline_claim: bool,
    sampling: str,
    dataset: dict[str, Any],
    config: dict[str, Any],
    seeds: dict[str, Any],
    adapter: dict[str, Any],
    embedder: dict[str, Any] | None,
    k: int,
    exclusions: dict[str, Any],
    run: dict[str, Any],
    headline: dict[str, Any],
    commands: dict[str, str],
    notes: str = "",
) -> dict[str, Any]:
    """Assemble the pins. ``artifacts`` is filled in by :func:`write_package`."""
    return {
        "schema": SCHEMA,
        "benchmark": benchmark,
        # An explicit false here is the difference between a machinery smoke
        # run and a number anyone may cite. It is never inferred.
        "headline_claim": headline_claim,
        "sampling": sampling,
        "notes": notes,
        "pinned": {
            "repo_commit": git_commit(),
            "repo_tracked_files_dirty_at_run": git_tracked_dirty(),
            "dataset": dataset,
            "config": {"effective": config, "sha256": content_hash(config)},
            "seeds": seeds,
            "adapter": adapter,
            # An absent embedder is stated, never left to be inferred from
            # silence -- a lexical-only run that reads as dense is exactly the
            # confusion the pipeline probe exists to kill.
            "embedder": embedder if embedder is not None else {"name": "none", "note": "no dense leg in this run"},
            "k": k,
            "exclusions": exclusions,
        },
        "run": run,
        "headline": headline,
        "commands": commands,
        "artifacts": {},
    }


def write_package(
    out_dir: str,
    *,
    manifest: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    dataset: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Write the five files and stamp the manifest with their hashes."""
    os.makedirs(out_dir, exist_ok=True)
    files: dict[str, bytes] = {
        "raw.ndjson": b"".join(canonical_json(row) for row in raw_rows),
        "metrics.json": canonical_json(metrics),
        "dataset.json": canonical_json(dataset),
        "environment.json": canonical_json(environment),
    }
    for name, data in files.items():
        with open(os.path.join(out_dir, name), "wb") as handle:
            handle.write(data)
    manifest["artifacts"] = {
        name: {"sha256": sha256_bytes(data), "bytes": len(data), "run_to_run_stable": name in RUN_TO_RUN_STABLE}
        for name, data in sorted(files.items())
    }
    with open(os.path.join(out_dir, "manifest.json"), "wb") as handle:
        handle.write(canonical_json(manifest))
    return manifest


def read_rows(path: str) -> list[dict[str, Any]]:
    """Load an NDJSON artifact. A malformed line is an error, not a skip."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({exc})") from exc
    return rows
