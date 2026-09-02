"""Mechanical enforcement of ``.arch-mind/rules.mind``.

Why this file exists
--------------------
The nine architectural-governance rules in ``.arch-mind/rules.mind`` were,
until this gate landed, evaluated only by running the external ``arch-mind``
CLI by hand. Nothing in this repository's CI looked at them, so the committed
governance fixture could drift arbitrarily far from the declared invariants
and nothing went red. This module closes that gap using nothing but the
standard library, so the rules are enforced by the same ``pytest`` run as
everything else.

Two rules were reported failing when this gate was written:

* ``NO_CROSS_PKG``        (``modularity_q16``    ``eq`` 10000) read 2155
* ``MCP_ISOLATION_FLOOR`` (``mcp_tool_isolation`` ``ge`` 9500)  read 9400

Neither was a defect in this repository and neither was a miscalibrated
threshold. Both readings came from a scanner that walked three ``git
worktree`` checkouts under ``.wt/`` -- untracked, gitignored, second copies
of this very source tree. Each copy re-imports the same canonical module
names, so every import a copy makes lands as a *cross-package* edge into
``mind_mem`` (273 of 348 edges), and every copy of an MCP tool module gets
paired against the original when the scanner looks for shared transitive
dependencies (max overlap 6 instead of 2). Pruning the nested working trees
and rescanning the same commit gives ``intra_package_edges == total_edges``
(75/75, modularity 10000) and ``max_mcp_tool_overlap == 2`` (isolation 9800),
and all nine rules pass.

That is worth stating plainly, because it is the reason the ``eq 10000``
comparator on ``modularity_q16`` must NOT be softened into a floor: it is
precisely the canary that fires when the governance scan measures something
other than this repository. It fired correctly.

What is checked here
--------------------
1. Every rule in ``.arch-mind/rules.mind`` passes against every committed
   governance fixture (the ``_aggregated_for_phase_a`` counter blocks), using
   the same Q16.16 kernel arithmetic as ``arch-mind``'s reference scorer.
2. No rule threshold has been weakened relative to the values recorded here.
3. This repository's own source really does have zero cross-package
   dependency edges -- measured directly from the import graph rather than
   trusting the committed fixture, so a stale or polluted fixture cannot make
   this claim true by accident.

Every negative assertion below has a positive control that proves the same
code path reports a violation when one exists.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCH_MIND_DIR = REPO_ROOT / ".arch-mind"
RULES_FILE = ARCH_MIND_DIR / "rules.mind"

Q16_ONE = 65_536
Q16_MAX_SCORE = 655_360_000  # 10000.0 in Q16.16

# The governance fixtures that actually feed the gate. `baseline_*.json` files
# are dated historical snapshots, not gate inputs; they are deliberately not
# evaluated here (the 2026-08-29 baseline was itself taken with the nested
# working trees indexed and is a known-bad historical record, retained rather
# than rewritten).
ACTIVE_FIXTURES = ("scan.json", "last_summary.json")

# Mirrors arch-mind's rules grammar: `[arch_rule(metric, op)] const NAME: i32 = N`.
RULE_PATTERN = re.compile(
    r"\[\s*arch_rule\s*\(\s*([a-z0-9_]+)\s*,\s*([a-z]+)\s*\)\s*\]\s*"
    r"const\s+([A-Z][A-Z0-9_]*)\s*:\s*i32\s*=\s*(-?\d+)\s*",
    re.MULTILINE,
)
COMMENT_PATTERN = re.compile(r"//[^\n]*")

# Directory names arch-mind's scanner never indexes, plus the caches this
# repository generates. Kept in one place so the module walk below and the
# scanner agree on what "this repository's source" means.
PRUNED_DIRS = frozenset(
    {
        ".git",
        ".claude",
        ".worktrees",
        "node_modules",
        "__pycache__",
        "target",
        "build",
        "dist",
        ".venv",
        "venv",
        ".tox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
    }
)

# Module-name prefixes arch-mind treats as non-architectural: tests, benches,
# examples, scripts, docs, vendored trees.
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


# ---------------------------------------------------------------------------
# Rules parsing + Q16.16 kernel arithmetic (mirrors arch-mind's reference).
# ---------------------------------------------------------------------------


def parse_rules(text: str) -> list[tuple[str, str, str, int]]:
    """Return ``(rule_name, metric, comparator, threshold_raw)`` for each rule."""
    sanitized = COMMENT_PATTERN.sub("", text)
    return [(name, metric, op, int(raw)) for metric, op, name, raw in RULE_PATTERN.findall(sanitized)]


def _ratio_q16(num: int, den: int, scale: int = 10_000) -> int:
    return ((num * scale) << 16) // den


def _clamp(q: int) -> int:
    return 0 if q < 0 else (Q16_MAX_SCORE if q > Q16_MAX_SCORE else q)


def score_metrics(s: dict[str, int]) -> dict[str, int]:
    """Compute the nine fixture-derived Q16.16 metric scores from raw counters."""
    scores: dict[str, int] = {}

    total_edges = s["total_edges"]
    scores["modularity_q16"] = Q16_MAX_SCORE if total_edges <= 0 else _clamp(_ratio_q16(s["intra_package_edges"], total_edges))

    cyclic = s["cyclic_edges"]
    if total_edges <= 0 or cyclic <= 0:
        scores["acyclicity_q16"] = Q16_MAX_SCORE
    elif cyclic >= total_edges:
        scores["acyclicity_q16"] = 0
    else:
        scores["acyclicity_q16"] = _clamp(_ratio_q16(total_edges - cyclic, total_edges))

    longest, modules = s["longest_path"], s["module_count"]
    if modules <= 0:
        scores["depth_q16"] = Q16_MAX_SCORE
    elif longest <= 0:
        scores["depth_q16"] = 0
    elif longest == 1:
        scores["depth_q16"] = Q16_MAX_SCORE
    else:
        penalty = (longest - 1) * 1000
        scores["depth_q16"] = 0 if penalty >= 10_000 else _clamp((10_000 - penalty) << 16)

    sum_symbols = s["sum_symbols"]
    if modules <= 1 or sum_symbols <= 0:
        scores["equality_q16"] = Q16_MAX_SCORE
    else:
        gini_q = ((s["sum_abs_symbol_diffs"] * 10_000) << 16) // (2 * modules * sum_symbols)
        scores["equality_q16"] = 0 if gini_q >= Q16_MAX_SCORE else _clamp(Q16_MAX_SCORE - gini_q)

    excess = s["redundancy_excess"]
    if excess <= 0:
        scores["redundancy_q16"] = Q16_MAX_SCORE
    elif excess >= 10_000:
        scores["redundancy_q16"] = 0
    else:
        scores["redundancy_q16"] = _clamp((10_000 - excess) << 16)

    pure = s["pure_modules"]
    if modules <= 0:
        scores["q16_determinism_purity"] = Q16_MAX_SCORE
    elif pure <= 0:
        scores["q16_determinism_purity"] = 0
    elif pure >= modules:
        scores["q16_determinism_purity"] = Q16_MAX_SCORE
    else:
        scores["q16_determinism_purity"] = _clamp(_ratio_q16(pure, modules))

    ev, dec = s["sum_evidence_calls"], s["sum_decision_points"]
    if dec <= 0 or ev <= 0:
        scores["evidence_chain_density"] = 0
    elif ev >= dec:
        scores["evidence_chain_density"] = Q16_MAX_SCORE
    else:
        scores["evidence_chain_density"] = _clamp(_ratio_q16(ev, dec))

    tools, overlap = s["total_mcp_tools"], s["max_mcp_tool_overlap"]
    if tools < 2 or overlap <= 0:
        scores["mcp_tool_isolation"] = Q16_MAX_SCORE
    else:
        penalty = overlap * 100
        scores["mcp_tool_isolation"] = 0 if penalty >= 10_000 else _clamp((10_000 - penalty) << 16)

    prot = s["sum_protected_decls"]
    if sum_symbols <= 0 or prot <= 0:
        scores["governance_kernel_coverage"] = 0
    elif prot >= sum_symbols:
        scores["governance_kernel_coverage"] = Q16_MAX_SCORE
    else:
        scores["governance_kernel_coverage"] = _clamp(_ratio_q16(prot, sum_symbols))

    return scores


_COMPARATORS = {
    "ge": lambda a, t: a >= t,
    "gt": lambda a, t: a > t,
    "le": lambda a, t: a <= t,
    "lt": lambda a, t: a < t,
    "eq": lambda a, t: a == t,
    "ne": lambda a, t: a != t,
}


def rule_violations(summary: dict[str, int], rules: list[tuple[str, str, str, int]]) -> list[str]:
    """Return a human-readable line per failing rule (empty list == all pass)."""
    scores = score_metrics(summary)
    failures = []
    for name, metric, op, threshold_raw in rules:
        actual = scores[metric]
        if not _COMPARATORS[op](actual, threshold_raw * Q16_ONE):
            failures.append(f"{name}: {metric} {op} {threshold_raw} -- actual raw={actual // Q16_ONE} (q16={actual})")
    return failures


def load_summary(path: Path) -> dict[str, int]:
    block = json.loads(path.read_text(encoding="utf-8"))["_aggregated_for_phase_a"]
    return {k: int(v) for k, v in block.items()}


# ---------------------------------------------------------------------------
# Direct measurement of the cross-package invariant from this repo's source.
# ---------------------------------------------------------------------------


def module_name(rel_path: Path) -> str:
    """Dotted module name for a repo-relative source path, arch-mind semantics."""
    parts = list(rel_path.with_suffix("").parts)
    if parts and parts[0] in ("src", "lib"):
        parts = parts[1:]
    name = ".".join(parts)
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def is_architectural(name: str) -> bool:
    """True for production source: not a test / bench / example / script / doc tree."""
    if name.startswith(NON_ARCHITECTURAL_PREFIXES):
        return False
    return not any(seg in name.split(".") for seg in ("tests", "test", "__pycache__"))


def discover_modules(root: Path) -> dict[str, Path]:
    """Map architectural module name -> absolute path, pruning nested working trees.

    A directory below ``root`` that carries its own ``.git`` entry is a second
    checkout of some source tree; indexing it double-counts modules and makes
    every result a function of which worktrees happened to exist. That is the
    exact defect this gate was written to catch, so the walk refuses to descend
    into one.
    """
    modules: dict[str, Path] = {}
    stack = [root]
    while stack:
        current = stack.pop()
        for entry in sorted(current.iterdir()):
            if entry.is_dir():
                if entry.name in PRUNED_DIRS or (entry / ".git").exists():
                    continue
                stack.append(entry)
            elif entry.suffix in (".py", ".pyi"):
                name = module_name(entry.relative_to(root))
                if is_architectural(name):
                    modules.setdefault(name, entry)
    return modules


def _resolve(candidate: str, known: set[str]) -> str | None:
    """Walk a dotted target upwards until it names a module we know about."""
    while candidate:
        if candidate in known:
            return candidate
        if "." not in candidate:
            return None
        candidate = candidate.rsplit(".", 1)[0]
    return None


def import_edges(root: Path, modules: dict[str, Path]) -> set[tuple[str, str]]:
    """Directed import edges between the given architectural modules."""
    known = set(modules)
    edges: set[tuple[str, str]] = set()
    for name, path in sorted(modules.items()):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        own_package = name if path.name == "__init__.py" else (name.rsplit(".", 1)[0] if "." in name else "")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = _resolve(alias.name, known)
                    if target and target != name:
                        edges.add((name, target))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    segments = own_package.split(".") if own_package else []
                    for _ in range(node.level - 1):
                        if segments:
                            segments.pop()
                    base = ".".join(segments)
                    module = f"{base}.{node.module}" if node.module else base
                else:
                    module = node.module or ""
                if not module:
                    continue
                for alias in node.names:
                    target = _resolve(f"{module}.{alias.name}", known) or _resolve(module, known)
                    if target and target != name:
                        edges.add((name, target))
    return edges


def cross_package_edges(edges: set[tuple[str, str]]) -> list[tuple[str, str]]:
    """Edges whose endpoints sit in different top-level packages."""
    return sorted(e for e in edges if e[0].split(".")[0] != e[1].split(".")[0])


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rules() -> list[tuple[str, str, str, int]]:
    parsed = parse_rules(RULES_FILE.read_text(encoding="utf-8"))
    assert len(parsed) == 9, f"expected 9 arch-mind rules, parsed {len(parsed)}: {parsed}"
    return parsed


@pytest.mark.parametrize("fixture_name", ACTIVE_FIXTURES)
def test_committed_fixture_satisfies_every_rule(rules: list[tuple[str, str, str, int]], fixture_name: str) -> None:
    """Every declared architectural rule passes against the committed scan."""
    path = ARCH_MIND_DIR / fixture_name
    assert path.exists(), f"governance fixture missing: {path}"
    failures = rule_violations(load_summary(path), rules)
    assert failures == [], f"{fixture_name} violates .arch-mind/rules.mind:\n  " + "\n  ".join(failures)


def test_rule_evaluator_reports_the_two_reported_violations(rules: list[tuple[str, str, str, int]]) -> None:
    """Positive control: the evaluator reproduces both originally-reported failures.

    Feeds it the counters a worktree-polluted scan produced for this same
    commit -- 75 intra-package edges out of 348, max MCP-tool overlap 6 across
    145 tools -- and asserts the two rules that were reported failing are the
    two it reports. Without this, the passing assertion above could be green
    because the evaluator cannot fail at all.
    """
    polluted = dict(load_summary(ARCH_MIND_DIR / "scan.json"))
    polluted.update(total_edges=348, intra_package_edges=75, max_mcp_tool_overlap=6, total_mcp_tools=145)
    failures = rule_violations(polluted, rules)
    assert len(failures) == 2, f"expected exactly the two reported violations, got: {failures}"
    assert any(f.startswith("NO_CROSS_PKG:") and "raw=2155" in f for f in failures), failures
    assert any(f.startswith("MCP_ISOLATION_FLOOR:") and "raw=9400" in f for f in failures), failures


def test_rule_thresholds_are_not_weakened(rules: list[tuple[str, str, str, int]]) -> None:
    """No rule may be relaxed; raising a floor is allowed, lowering one is not."""
    recorded = {
        "NO_CYCLES": ("acyclicity_q16", "eq", 10_000),
        "REDUNDANCY_FLOOR": ("redundancy_q16", "ge", 9_500),
        "PURITY_FLOOR": ("q16_determinism_purity", "ge", 6_000),
        "EQUALITY_FLOOR": ("equality_q16", "ge", 2_000),
        "DEPTH_FLOOR": ("depth_q16", "ge", 6_000),
        "GOVERNANCE_FLOOR": ("governance_kernel_coverage", "ge", 0),
        "NO_CROSS_PKG": ("modularity_q16", "eq", 10_000),
        "MCP_ISOLATION_FLOOR": ("mcp_tool_isolation", "ge", 9_500),
        "EVIDENCE_FLOOR": ("evidence_chain_density", "ge", 0),
    }
    live = {name: (metric, op, threshold) for name, metric, op, threshold in rules}
    assert set(live) == set(recorded), f"rule set changed: {sorted(set(live) ^ set(recorded))}"
    for name, (metric, op, threshold) in recorded.items():
        live_metric, live_op, live_threshold = live[name]
        assert (live_metric, live_op) == (metric, op), f"{name} changed metric/comparator: {live[name]} != {(metric, op, threshold)}"
        assert live_threshold >= threshold, f"{name} threshold weakened: {live_threshold} < {threshold}"


def test_source_has_no_cross_package_dependency_edges() -> None:
    """NO_CROSS_PKG, measured from this repository's own imports.

    Independent of the committed fixture and of arch-mind's scanner: if a
    module outside ``mind_mem`` ever imports into it (or vice versa), this
    fails even when the fixture is stale. The import graph measured here is
    denser than the one arch-mind aggregates, so it is a strictly stronger
    check than the rule it mirrors.
    """
    modules = discover_modules(REPO_ROOT)
    assert len(modules) > 300, f"module discovery collapsed to {len(modules)} modules -- the walk is broken, not the repo"
    edges = import_edges(REPO_ROOT, modules)
    assert len(edges) > 500, f"import-graph discovery collapsed to {len(edges)} edges -- the parser is broken, not the repo"
    offenders = cross_package_edges(edges)
    assert offenders == [], "cross-package dependency edges (NO_CROSS_PKG):\n  " + "\n  ".join(f"{s} -> {t}" for s, t in offenders)


def test_cross_package_detector_sees_a_planted_edge(tmp_path: Path) -> None:
    """Positive control for the check above, on a synthetic two-package tree."""
    (tmp_path / "src" / "widget").mkdir(parents=True)
    (tmp_path / "reporting").mkdir()
    (tmp_path / "src" / "widget" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "widget" / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "reporting" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "reporting" / "summary.py").write_text("from widget.core import VALUE\n", encoding="utf-8")

    modules = discover_modules(tmp_path)
    assert "widget.core" in modules and "reporting.summary" in modules, modules
    offenders = cross_package_edges(import_edges(tmp_path, modules))
    assert offenders == [("reporting.summary", "widget.core")], offenders


def test_module_walk_refuses_to_descend_into_a_nested_working_tree(tmp_path: Path) -> None:
    """Positive control for the pruning that the two reported failures turned on.

    A nested checkout re-declares the same module names; counting it is what
    produced ``modularity_q16`` 2155 and ``mcp_tool_isolation`` 9400 on a
    commit whose real readings are 10000 and 9800.
    """
    (tmp_path / "src" / "widget").mkdir(parents=True)
    (tmp_path / "src" / "widget" / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    copy_root = tmp_path / ".wt" / "branch-a" / "src" / "widget"
    copy_root.mkdir(parents=True)
    (copy_root / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    (copy_root / "extra.py").write_text("from widget.core import VALUE\n", encoding="utf-8")

    # `src` is stripped only at position 0, so the copy's modules keep their
    # `.wt.` prefix -- which is why every edge they make reads as cross-package.
    assert set(discover_modules(tmp_path)) == {
        "widget.core",
        ".wt.branch-a.src.widget.core",
        ".wt.branch-a.src.widget.extra",
    }

    # Mark the copy as a working tree the way `git worktree` does, and it vanishes.
    (tmp_path / ".wt" / "branch-a" / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")
    assert set(discover_modules(tmp_path)) == {"widget.core"}
