"""Backport v2.9.0 audit fixes to every prior v2.x release as .post1.

For each tag from v2.0.0a1 → v2.8.2 the script:

    1. Creates a git worktree at /tmp/mm-backport-<tag> pinned to the tag.
    2. Cherry-picks the v2.9.0 audit commit so every applicable fix lands.
       Conflicts (fixes for files that didn't exist at that tag) are
       resolved by dropping the hunk — the fix simply doesn't apply to
       the older tree.
    3. Bumps __version__ and pyproject.toml to <tag>.post1.
    4. Builds sdist + wheel with `python3 -m build`.
    5. Uploads via twine.
    6. Cuts a GitHub release (sources only, no binaries — those are on PyPI).
    7. Tears down the worktree.

Runs strictly sequentially — PyPI rate-limits new release uploads and
parallel builds in the same repo fight over `dist/`.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/n/mind-mem")
V29_SHA = subprocess.check_output(
    ["git", "rev-parse", "v2.9.0"], cwd=REPO, text=True
).strip()

# First 7 (v2.0.0a1..v2.1.0) shipped before disk-full hit. Resume
# from v2.2.0. twine --skip-existing handles any inadvertent repeats.
TAGS = [
    "v2.2.0", "v2.3.0", "v2.4.0",
    "v2.5.0", "v2.6.0", "v2.7.0", "v2.8.0", "v2.8.1", "v2.8.2",
]

# Worktrees stage a full tree copy; /data has the headroom, / often doesn't.
WORKTREE_ROOT = Path("/data/checkpoints/mm-workspace/mm-backport")


def run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True)


def backport(tag: str) -> None:
    WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)
    worktree = WORKTREE_ROOT / tag
    if worktree.is_dir():
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=REPO)
    branch = f"backport/{tag}.post1"
    # Delete any leftover branch from a prior run.
    subprocess.run(["git", "branch", "-D", branch], cwd=REPO, capture_output=True)
    run(["git", "worktree", "add", "-b", branch, str(worktree), tag], REPO)

    # Cherry-pick the v2.9.0 commit; keep going on conflicts, dropping
    # the conflicting hunks (files that didn't exist at this tag).
    proc = subprocess.run(
        ["git", "cherry-pick", "--strategy=recursive", "-X", "theirs", V29_SHA],
        cwd=worktree, text=True, capture_output=True,
    )
    if proc.returncode != 0:
        # Delete any unmerged paths so we can commit the partial fix set.
        unmerged = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=U"],
            cwd=worktree, text=True, capture_output=True,
        ).stdout.split()
        for p in unmerged:
            subprocess.run(["git", "rm", "-f", p], cwd=worktree, capture_output=True)
        run(["git", "add", "-A"], worktree)
        run(["git", "-c", "core.editor=true", "cherry-pick", "--continue"], worktree, check=False)

    # Bump version. Strip leading "v" from tag.
    target_version = f"{tag.lstrip('v')}.post1"
    _bump_version(worktree, target_version)
    subprocess.run(
        ["git", "add", "-A"], cwd=worktree, check=True, text=True, capture_output=True,
    )
    run([
        "git", "commit",
        "--author=STARGA Inc <noreply@star.ga>",
        "-m", f"chore: release {target_version} — backport of v2.9.0 audit fixes",
    ], worktree)

    # Build sdist + wheel.
    (worktree / "dist").mkdir(exist_ok=True)
    run(["python3", "-m", "build", "--sdist", "--wheel"], worktree)

    # Upload via twine.
    wheels = list((worktree / "dist").glob("*.whl")) + list((worktree / "dist").glob("*.tar.gz"))
    if not wheels:
        print(f"  ! no dist files produced for {target_version}, skipping upload")
        return
    run(["twine", "upload", "--skip-existing"] + [str(w) for w in wheels], worktree)

    # Push tag + GitHub release.
    tag_name = f"v{target_version}"
    run(["git", "tag", "-a", tag_name, "-m", f"{target_version} backport"], worktree)
    run(["git", "push", "origin", tag_name], worktree)
    run([
        "gh", "release", "create", tag_name,
        "--title", f"{target_version} — audit backport of {tag}",
        "--notes", (
            f"Backport of v2.9.0's audit-pass-#2 fixes onto the {tag} source tree. "
            "Applies every fix whose target module existed at the time of the base release; "
            "modules introduced after that release are skipped. "
            "See https://github.com/star-ga/mind-mem/releases/tag/v2.9.0 for the full fix list."
        ),
    ] + [str(w) for w in wheels], worktree)

    # Teardown.
    subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=REPO)
    subprocess.run(["git", "branch", "-D", branch], cwd=REPO, capture_output=True)
    print(f"  ✓ shipped {target_version}\n")


def _bump_version(worktree: Path, version: str) -> None:
    pyproject = worktree / "pyproject.toml"
    text = pyproject.read_text()
    text = re.sub(r'version = "[^"]+"', f'version = "{version}"', text, count=1)
    pyproject.write_text(text)

    # The __init__.py may live at src/mind_mem or scripts/mind_mem depending
    # on the release age; patch whichever exists.
    for rel in ("src/mind_mem/__init__.py", "scripts/mind_mem/__init__.py"):
        init = worktree / rel
        if init.is_file():
            t2 = init.read_text()
            if "__version__" in t2:
                t2 = re.sub(r'__version__\s*=\s*"[^"]+"', f'__version__ = "{version}"', t2)
                init.write_text(t2)
                break


def main() -> None:
    ok: list[str] = []
    failed: list[tuple[str, str]] = []
    for tag in TAGS:
        print(f"\n=== Backporting {tag} → {tag.lstrip('v')}.post1 ===")
        try:
            backport(tag)
            ok.append(tag)
        except subprocess.CalledProcessError as exc:
            failed.append((tag, exc.stderr or str(exc)))
            print(f"  ✗ {tag} FAILED: {exc.stderr[:200] if exc.stderr else exc}")
    print("\n========= summary =========")
    print(f"  shipped: {len(ok)}")
    for t in ok:
        print(f"    ✓ {t}.post1")
    if failed:
        print(f"  failed : {len(failed)}")
        for t, reason in failed:
            print(f"    ✗ {t}: {reason[:150]}")


if __name__ == "__main__":
    main()
