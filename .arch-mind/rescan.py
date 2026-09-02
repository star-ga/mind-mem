#!/usr/bin/env python3
# Copyright 2026 STARGA, Inc. — Apache-2.0 (see ../LICENSE).
"""Regenerate the arch-mind governance fixtures from *tracked content only*.

Why this script exists
----------------------
``.arch-mind/rules.mind`` used to carry its scan precondition as prose: "the
fixture must be produced with nested working trees pruned". Nothing performed
that pruning, and nothing checked that it had happened. Measured on
2026-09-01 against commit 6e60759, the documented command --

    arch-mind sidecar-scan --repo . --lang python,typescript,go,rust,mind

-- run in a live checkout of this repository produced ``module_count`` 1302,
``intra_package_edges`` 77 of ``total_edges`` 350 and
``max_mcp_tool_overlap`` 6, which fails ``NO_CROSS_PKG`` (modularity 2200,
comparator ``eq`` 10000) and ``MCP_ISOLATION_FLOOR`` (isolation 9400, floor
9500). The same commit, extracted with ``git archive`` and scanned with the
same binary, produced 391 / 77 of 77 / overlap 2 -- modularity 10000 and
isolation 9800, all nine rules green.

The whole difference was eight nested working trees (``git worktree``
checkouts under ``.wt/`` and ``.claude/worktrees/``) that agents create and
destroy while they work. Every copy re-imports the same canonical module
names, so each copy's imports land as *cross-package* edges (273 of the 350),
and each copy of an MCP-tool module gets paired against the original when the
overlap metric compares transitive dependency sets -- the overlap-6 reading
was ``src/mind_mem/mcp/tools/memory_ops.py`` matched against three copies of
itself.

So the fixture must be a function of the *commit*, never of scan-time
filesystem state. This script makes it one: it extracts ``HEAD`` with
``git archive`` into a scratch directory -- which by construction contains
only tracked files and therefore no nested working tree, no ``.venv``, no
build output -- and scans that.

That is strictly stronger than asking the scanner to prune. Pruning depends on
the scanner version you happen to have installed (the ``arch-mind`` binary on
this machine predates its own prune and cannot do it); extracting the commit
depends on nothing but ``git``.

Usage
-----
    python3 .arch-mind/rescan.py                 # rewrite the fixtures
    python3 .arch-mind/rescan.py --check         # fail if they would change
    python3 .arch-mind/rescan.py --rev <commit>  # scan a different commit

Requires the ``arch-mind`` CLI on ``PATH`` (or ``--arch-mind <path>``) to
rescan. The pure-Python halves -- extraction, nested-working-tree detection
and fixture normalisation -- need nothing but git, and are what
``tests/test_arch_mind_fixture_provenance.py`` exercises.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ARCH_MIND_DIR = Path(__file__).resolve().parent
REPO_ROOT = ARCH_MIND_DIR.parent

# Every language the fixture is scanned under. Order is part of the fixture
# (`_languages` is a joined string), so it is fixed here rather than passed in.
LANGUAGES = ("python", "typescript", "go", "rust", "mind")

# The fixtures this script owns. All three describe the SAME commit and are
# written from ONE scan, so they cannot drift apart -- before this script
# existed, `scan.json` said 388 modules and `last_summary.json` said 232 for
# the same repository, and both were committed.
#
#   scan.json        the CI-gate fixture (tests/test_arch_mind_rules_gate.py)
#   fixture.json     the `check-rules --fixture` input
#   last_summary.json the `session-start` / `session-end` hook fixture
#
# Historical snapshots (`baseline_*.json`, `scan_v*.json`) are deliberately
# NOT in this list: they are dated records of what a past commit measured and
# rewriting them would destroy the record.
FIXTURE_NAMES = ("fixture.json", "last_summary.json", "scan.json")

# `_repo_root` is normalised to this. The scan runs against a temp directory
# whose name is different on every run and on every machine; recording it
# would make the fixture non-reproducible for a reason that has nothing to do
# with the repository's architecture.
NORMALISED_REPO_ROOT = "."

FIXTURE_COMMENT = (
    "Produced by `.arch-mind/rescan.py` from a `git archive` extraction of "
    "this commit's tracked content, scanned with `arch-mind sidecar-scan`. "
    "The extraction is why the numbers are a function of the commit and not "
    "of scan-time filesystem state: it contains tracked files only, so a "
    "nested working tree (a `git worktree` checkout under `.wt/`, "
    "`.claude/worktrees/`, or any other name) cannot be in scope whatever "
    "the scanner does. `_repo_root` is normalised to '.' because the scan "
    "runs in a scratch directory."
)


class RescanError(RuntimeError):
    """A step of the rescan failed; the message says which."""


# ---------------------------------------------------------------------------
# Nested working trees.
# ---------------------------------------------------------------------------


def nested_working_trees(root: Path) -> list[Path]:
    """Directories below ``root`` that are themselves git working trees.

    Detected structurally, by the presence of a ``.git`` entry -- a *file* for
    a ``git worktree`` checkout, a *directory* for a nested clone. Not by
    name: ``.wt/``, ``.claude/worktrees/`` and ``.worktrees/`` are three
    naming conventions for one structure and name-matching keeps losing to
    the next one.

    ``root`` itself is never reported, so a repository's own source can never
    be mistaken for a nested copy. A found tree is not descended into: one
    result per nested tree, not one per nested tree inside it.
    """
    found: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except (NotADirectoryError, PermissionError, FileNotFoundError):
            continue
        for entry in entries:
            if not entry.is_dir() or entry.is_symlink():
                continue
            if entry.name == ".git":
                continue
            if (entry / ".git").exists():
                found.append(entry)
                continue
            stack.append(entry)
    return sorted(found)


# ---------------------------------------------------------------------------
# Extraction.
# ---------------------------------------------------------------------------


def extract_tracked_tree(repo_root: Path, dest: Path, rev: str = "HEAD") -> None:
    """Materialise ``rev``'s tracked content into ``dest``.

    Uses ``git archive``, so the result is exactly what the commit contains:
    no untracked file, no ignored file, and -- the point of the exercise -- no
    nested working tree, because a worktree checkout is untracked in its host.
    """
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "-C", str(repo_root), "archive", "--format=tar", rev],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if archive.returncode != 0:
        raise RescanError(f"git archive {rev} failed ({archive.returncode}): {archive.stderr.decode('utf-8', errors='replace').strip()}")
    if not archive.stdout:
        raise RescanError(f"git archive {rev} produced an empty archive")
    extract = subprocess.run(
        ["tar", "-x", "-C", str(dest)],
        input=archive.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if extract.returncode != 0:
        raise RescanError(f"tar extract failed ({extract.returncode}): {extract.stderr.decode('utf-8', errors='replace').strip()}")


# ---------------------------------------------------------------------------
# Scan + normalisation.
# ---------------------------------------------------------------------------


def run_sidecar_scan(arch_mind: str, tree: Path, out: Path) -> None:
    """Run ``arch-mind sidecar-scan`` over ``tree``, writing raw JSON to ``out``."""
    proc = subprocess.run(
        [
            arch_mind,
            "sidecar-scan",
            "--repo",
            str(tree),
            "--lang",
            ",".join(LANGUAGES),
            "--out",
            str(out),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RescanError(f"arch-mind sidecar-scan failed ({proc.returncode}): {proc.stderr.decode('utf-8', errors='replace').strip()}")


def normalise_fixture(payload: dict) -> dict:
    """Strip the scan-location fields that would make the fixture unstable.

    Everything else -- above all ``_aggregated_for_phase_a``, the counters the
    nine kernels consume -- is passed through untouched. Normalising a counter
    would be editing the measurement.
    """
    if "_aggregated_for_phase_a" not in payload:
        raise RescanError("scan output has no _aggregated_for_phase_a block")
    normalised = dict(payload)
    normalised["_comment"] = FIXTURE_COMMENT
    normalised["_repo_root"] = NORMALISED_REPO_ROOT
    return normalised


def render(payload: dict) -> str:
    """Serialise a fixture the way arch-mind does: sorted keys, 2-space indent."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def rescan(arch_mind: str, rev: str = "HEAD") -> dict:
    """Extract ``rev``, scan it, and return the normalised fixture payload."""
    with tempfile.TemporaryDirectory(prefix="arch-mind-rescan-") as tmp:
        tmp_dir = Path(tmp)
        tree = tmp_dir / "tree"
        extract_tracked_tree(REPO_ROOT, tree, rev)
        stowaways = nested_working_trees(tree)
        if stowaways:  # pragma: no cover - git archive cannot produce one
            raise RescanError(
                f"extracted tree contains nested working trees, which git archive cannot produce: {[str(p) for p in stowaways]}"
            )
        raw = tmp_dir / "scan.json"
        run_sidecar_scan(arch_mind, tree, raw)
        payload = json.loads(raw.read_text(encoding="utf-8"))
    return normalise_fixture(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rev", default="HEAD", help="commit-ish to scan")
    parser.add_argument("--arch-mind", default="arch-mind", help="path to the arch-mind CLI")
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if a committed fixture would change",
    )
    args = parser.parse_args(argv)

    try:
        text = render(rescan(args.arch_mind, args.rev))
    except (RescanError, OSError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2

    drifted: list[str] = []
    for name in FIXTURE_NAMES:
        path = ARCH_MIND_DIR / name
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == text:
            continue
        drifted.append(name)
        if not args.check:
            path.write_text(text, encoding="utf-8")

    if args.check:
        if drifted:
            print(f"::error::fixtures are stale: {', '.join(drifted)}", file=sys.stderr)
            return 1
        print("fixtures match a fresh scan of the commit")
        return 0
    print(f"rewrote {len(drifted)} of {len(FIXTURE_NAMES)} fixtures" if drifted else "fixtures already current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
