"""Provenance of the arch-mind governance fixtures.

Why this file exists
--------------------
``tests/test_arch_mind_rules_gate.py`` proves the committed fixtures satisfy
``.arch-mind/rules.mind``. It cannot prove the fixtures describe *this
repository*: a fixture is a JSON file of counters, and counters can be stale,
hand-edited, or captured from a polluted scan and still pass every rule.

That gap was not hypothetical. Measured on 2026-09-01 against commit
6e60759, the command the rules file documented --

    arch-mind sidecar-scan --repo . --lang python,typescript,go,rust,mind

-- run in a live checkout produced ``module_count`` 1302,
``intra_package_edges`` 77 of ``total_edges`` 350 and
``max_mcp_tool_overlap`` 6, failing ``NO_CROSS_PKG`` (modularity 2200 against
``eq`` 10000) and ``MCP_ISOLATION_FLOOR`` (isolation 9400 against floor
9500). The same binary, given the same commit extracted with ``git
archive``, produced 391 / 77 of 77 / overlap 2 -- modularity 10000, isolation
9800, all nine rules green. The difference was eight nested ``git worktree``
checkouts that agents create and destroy while they work.

The prose precondition ("produce the fixture with nested working trees
pruned") could not enforce itself, and the installed scanner could not honour
it. ``.arch-mind/rescan.py`` replaces the precondition with a mechanism --
scan an extraction of the commit, which contains tracked files only and
therefore cannot contain a nested working tree whatever the scanner does --
and this module gates the parts of that mechanism which need nothing but git.

Every negative assertion here carries a positive control proving the same
code path reports the defect when the defect is present.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCH_MIND_DIR = REPO_ROOT / ".arch-mind"
RESCAN_SCRIPT = ARCH_MIND_DIR / "rescan.py"


def _load_rescan():
    """Import ``.arch-mind/rescan.py`` by path (its directory is not a package)."""
    spec = importlib.util.spec_from_file_location("_arch_mind_rescan", RESCAN_SCRIPT)
    assert spec is not None and spec.loader is not None, f"cannot import {RESCAN_SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rescan = _load_rescan()


# ---------------------------------------------------------------------------
# Mirror of arch-mind's own MCP-tool accounting.
# ---------------------------------------------------------------------------
#
# Copied deliberately rather than approximated: the point of the check below
# is to agree with the scanner exactly, so every predicate is the scanner's.
#
#   is_mcp_tool_path  sidecar/src/main.rs -- path contains "/mcp/" or "/tools/"
#   count_loc         sidecar/src/main.rs -- lines that are not blank and do
#                     not start with '#', '//', '/*' or '*'
#   _is_architectural tools/summary.py    -- production source only
#   INDEXED_SUFFIXES  sidecar/src/lang.rs + tools/mind_reflect.py

INDEXED_SUFFIXES = frozenset({".py", ".pyi", ".ts", ".tsx", ".mts", ".cts", ".go", ".rs", ".mind"})

NOT_INDEXED_SEGMENTS = (
    "/.git/",
    "/.claude/",
    "/.worktrees/",
    "/node_modules/",
    "/__pycache__/",
    "/target/",
    "/build/",
    "/dist/",
    "/.venv/",
    "/venv/",
    "/.tox/",
    "/.pytest_cache/",
)

NON_ARCHITECTURAL_PREFIXES = (
    "tests.",
    "test_",
    "benchmarks.",
    "benchmark.",
    "examples.",
    "example.",
    "scripts.",
    "script.",
    "conftest",
    "docs.",
    "doc.",
    "site.",
    "site-packages.",
)


def count_loc(text: str) -> int:
    """arch-mind's line count: non-blank, non-comment-leading lines."""
    total = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("#", "//", "/*", "*")):
            continue
        total += 1
    return total


def module_name(rel_path: str) -> str:
    """Dotted module name for a repo-relative path, arch-mind semantics."""
    name = rel_path.rsplit(".", 1)[0].replace("/", ".")
    while True:
        if name.startswith("src."):
            name = name[len("src.") :]
        elif name.startswith("lib."):
            name = name[len("lib.") :]
        else:
            break
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def is_architectural(name: str) -> bool:
    if name.startswith(NON_ARCHITECTURAL_PREFIXES):
        return False
    return not any(segment in name.split(".") for segment in ("tests", "test", "__pycache__"))


def mcp_tool_modules(tree: Path) -> list[str]:
    """Repo-relative paths the scanner counts toward ``total_mcp_tools``."""
    found: list[str] = []
    for path in sorted(tree.rglob("*")):
        if not path.is_file() or path.suffix not in INDEXED_SUFFIXES:
            continue
        rel = path.relative_to(tree).as_posix()
        if any(segment in f"/{rel}" for segment in NOT_INDEXED_SEGMENTS):
            continue
        if "/mcp/" not in rel and "/tools/" not in rel:
            continue
        if not is_architectural(module_name(rel)):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # pragma: no cover - defensive
            continue
        if count_loc(text) > 0:
            found.append(rel)
    return found


def summary_of(fixture_name: str) -> dict[str, int]:
    payload = json.loads((ARCH_MIND_DIR / fixture_name).read_text(encoding="utf-8"))
    return {k: int(v) for k, v in payload["_aggregated_for_phase_a"].items()}


@pytest.fixture(scope="module")
def extracted_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """This commit's tracked content, materialised the way rescan.py does."""
    dest = tmp_path_factory.mktemp("tracked-tree")
    rescan.extract_tracked_tree(REPO_ROOT, dest)
    return dest


# ---------------------------------------------------------------------------
# The mechanism exists and is runnable.
# ---------------------------------------------------------------------------


def test_rescan_script_is_present_and_declares_the_fixtures_it_owns() -> None:
    """The fixtures are regenerated by a script, not by a remembered command."""
    assert RESCAN_SCRIPT.exists(), f"missing fixture generator: {RESCAN_SCRIPT}"
    assert set(rescan.FIXTURE_NAMES) == {"fixture.json", "last_summary.json", "scan.json"}
    for name in rescan.FIXTURE_NAMES:
        assert (ARCH_MIND_DIR / name).exists(), f"declared fixture is missing: {name}"


# ---------------------------------------------------------------------------
# Nested working trees.
# ---------------------------------------------------------------------------


def test_nested_working_tree_detector_finds_a_planted_checkout(tmp_path: Path) -> None:
    """Positive control: the detector fires on both shapes of nested tree.

    A ``git worktree`` checkout carries ``.git`` as a *file*; a nested clone
    carries it as a *directory*. Both are second copies of some source tree
    and both are what turned modularity 10000 into 2200.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    worktree = tmp_path / ".wt" / "branch-a"
    worktree.mkdir(parents=True)
    clone = tmp_path / "vendor" / "nested-clone"
    clone.mkdir(parents=True)

    assert rescan.nested_working_trees(tmp_path) == []

    (worktree / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")
    (clone / ".git").mkdir()
    assert rescan.nested_working_trees(tmp_path) == sorted([worktree, clone])


def test_nested_working_tree_detector_never_reports_the_root_itself(tmp_path: Path) -> None:
    """A repository is not a nested copy of itself, whatever it carries."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "src").mkdir()
    assert rescan.nested_working_trees(tmp_path) == []


def test_extraction_of_this_commit_contains_no_nested_working_tree(extracted_tree: Path) -> None:
    """The mechanical form of the scan precondition.

    The negative assertion is paired with a positive control immediately
    below it: planting one ``.git`` marker in the extraction must make the
    same call report it, so an empty result cannot mean "the walk found
    nothing because the walk is broken".
    """
    assert rescan.nested_working_trees(extracted_tree) == []

    planted = extracted_tree / "src" / "planted-worktree"
    planted.mkdir(parents=True)
    (planted / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")
    try:
        assert rescan.nested_working_trees(extracted_tree) == [planted]
    finally:
        (planted / ".git").unlink()
        planted.rmdir()


def test_extraction_really_contains_this_repository(extracted_tree: Path) -> None:
    """Guards the check above against passing on an empty directory."""
    assert (extracted_tree / "src" / "mind_mem" / "__init__.py").is_file()
    assert (extracted_tree / ".arch-mind" / "rules.mind").is_file()
    indexed = [p for p in extracted_tree.rglob("*.py") if p.is_file()]
    assert len(indexed) > 500, f"extraction collapsed to {len(indexed)} python files"


# ---------------------------------------------------------------------------
# The committed fixtures describe one commit.
# ---------------------------------------------------------------------------


def test_committed_fixtures_describe_one_and_the_same_scan() -> None:
    """All three live fixtures must carry identical counters.

    Before ``rescan.py`` wrote them from a single scan, ``scan.json`` said 388
    modules and ``last_summary.json`` said 232 for the same repository, and
    both were committed. Two fixtures that disagree mean at least one is
    describing a commit nobody can name.
    """
    summaries = {name: summary_of(name) for name in rescan.FIXTURE_NAMES}
    reference_name, reference = next(iter(summaries.items()))
    for name, summary in summaries.items():
        assert summary == reference, f"{name} disagrees with {reference_name}: {summary} != {reference}"


def test_fixture_disagreement_is_detectable() -> None:
    """Positive control for the comparison above."""
    reference = summary_of("scan.json")
    drifted = dict(reference, module_count=reference["module_count"] + 1)
    assert drifted != reference


# ---------------------------------------------------------------------------
# The fixture's MCP-tool count is bound to the source.
# ---------------------------------------------------------------------------


def test_fixture_mcp_tool_count_matches_the_tracked_source(extracted_tree: Path) -> None:
    """``total_mcp_tools`` is measured from the commit, not trusted.

    This is the counter that no rule constrains and that pollution moves
    hardest: 40 for this commit's tracked content, 145 when the live checkout
    with its worktree copies was scanned. ``mcp_tool_isolation`` is computed
    from ``max_mcp_tool_overlap``, which is only meaningful relative to the
    set of tools that produced it, so pinning the set size is what stops a
    fixture captured under pollution from being committed and passing.
    """
    measured = mcp_tool_modules(extracted_tree)
    assert summary_of("scan.json")["total_mcp_tools"] == len(measured), (
        f"fixture total_mcp_tools disagrees with the tracked source; measured {len(measured)} modules:\n  " + "\n  ".join(measured)
    )


def test_mcp_tool_counter_sees_a_planted_tool(extracted_tree: Path) -> None:
    """Positive control: the counter is not returning a constant."""
    before = len(mcp_tool_modules(extracted_tree))
    planted = extracted_tree / "src" / "mind_mem" / "mcp" / "tools" / "planted_probe.py"
    planted.write_text("VALUE = 1\n", encoding="utf-8")
    try:
        assert len(mcp_tool_modules(extracted_tree)) == before + 1
    finally:
        planted.unlink()
    assert len(mcp_tool_modules(extracted_tree)) == before


def test_mcp_tool_counter_ignores_a_comment_only_module(extracted_tree: Path) -> None:
    """arch-mind counts a module only when ``count_loc > 0``; so must this."""
    before = len(mcp_tool_modules(extracted_tree))
    planted = extracted_tree / "src" / "mind_mem" / "mcp" / "tools" / "planted_comment.py"
    planted.write_text("# nothing but a comment\n\n# and a blank line\n", encoding="utf-8")
    try:
        assert len(mcp_tool_modules(extracted_tree)) == before
    finally:
        planted.unlink()


# ---------------------------------------------------------------------------
# Normalisation touches location, never measurement.
# ---------------------------------------------------------------------------


def test_normalise_pins_the_scan_location_and_leaves_counters_alone() -> None:
    """``_repo_root`` is normalised because the scan runs in a scratch dir."""
    counters = {"module_count": 391, "total_edges": 77}
    payload = {
        "_aggregated_for_phase_a": dict(counters),
        "_repo_root": "/tmp/arch-mind-rescan-abc123/tree",
        "_comment": "whatever the scanner wrote",
        "_languages": "python",
    }
    normalised = rescan.normalise_fixture(payload)
    assert normalised["_repo_root"] == rescan.NORMALISED_REPO_ROOT
    assert normalised["_comment"] == rescan.FIXTURE_COMMENT
    assert normalised["_aggregated_for_phase_a"] == counters
    assert payload["_repo_root"] == "/tmp/arch-mind-rescan-abc123/tree", "input was mutated"


def test_normalise_refuses_a_scan_with_no_counters() -> None:
    """A scan that produced no summary block must fail loudly, not silently."""
    with pytest.raises(rescan.RescanError):
        rescan.normalise_fixture({"_repo_root": "/tmp/x", "nodes": [], "edges": []})
