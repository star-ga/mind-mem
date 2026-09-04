"""Group S — cross-project corroboration as a maturity component.

Four things are under test, and they are deliberately kept apart:

1. **The project key** (``mind_mem.project_key``).  The whole component
   is worthless if the key manufactures independence that is not there,
   so the first test is that a work tree and its parent repository
   collapse to *one* project while separate repositories stay distinct.
2. **Edge provenance** (``co_retrieval.origin_project``).  Additive,
   idempotent, defaulted for pre-existing rows, and never able to raise
   into the write path.
3. **The score component** (``block_maturity``), including the hard
   statelessness constraint — that module must not grow a dependency on
   ``block_lineage`` — and a golden proof that the default path is
   unchanged bit for bit.
4. **The corpus delta scan**, which must be able to answer "what would
   this rebalance do" without changing anything.
"""

from __future__ import annotations

import ast
import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from mind_mem.block_lineage import (
    NO_ORIGIN_PROJECT,
    add_block_edge,
    distinct_project_count,
    distinct_project_counts,
    ensure_lineage_schema,
)
from mind_mem.block_maturity import (
    MATURITY_EDGE_SATURATION,
    MATURITY_EDGE_WEIGHT,
    MATURITY_EDGE_WEIGHT_WITH_BREADTH,
    MATURITY_LIFECYCLE_WEIGHT,
    MATURITY_PROJECT_SATURATION,
    MATURITY_PROJECT_WEIGHT,
    MATURITY_STATUS_WEIGHT,
    maturity_score,
)
from mind_mem.maturity_breadth_scan import scan_maturity_breadth
from mind_mem.project_key import (
    PROJECT_KEY_MAX_LEN,
    PROJECT_KEY_UNKNOWN,
    clear_project_key_cache,
    resolve_project_key,
)
from mind_mem.retrieval_graph import ensure_graph_tables

_SRC = Path(__file__).resolve().parents[1] / "src" / "mind_mem"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
        errors="replace",
    )


def _make_repo(root: Path) -> Path:
    """Initialise a repository with one commit (work trees need a commit)."""
    root.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", cwd=root)
    (root / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git("add", "seed.txt", cwd=root)
    _git("commit", "-q", "-m", "seed", cwd=root)
    return root


def _db(workspace: Path) -> Path:
    return workspace / ".mind-mem-index" / "recall.db"


def _rows(workspace: Path, sql: str, params: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(str(_db(workspace)))
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _clear_cache():
    """The key resolver memoises; every test starts from a cold cache."""
    clear_project_key_cache()
    yield
    clear_project_key_cache()


requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


# ---------------------------------------------------------------------------
# 1. The project key — independence must not be manufactured
# ---------------------------------------------------------------------------


@pytest.mark.unit
@requires_git
class TestProjectKeyIndependence:
    def test_worktree_and_parent_are_one_project(self, tmp_path: Path) -> None:
        """A linked work tree is not an independent observer of its parent."""
        repo = _make_repo(tmp_path / "repo")
        worktree = tmp_path / "repo-wt"
        _git("worktree", "add", "-q", str(worktree), "-b", "side", cwd=repo)

        assert resolve_project_key(str(repo)) == resolve_project_key(str(worktree))

    def test_worktree_toplevel_would_have_manufactured_independence(self, tmp_path: Path) -> None:
        """The naive key (work-tree root) differs where the real key does not.

        This is the test that pins *why* the common directory is the key:
        if the two toplevels were equal the first test would pass for a
        trivial reason and prove nothing.
        """
        repo = _make_repo(tmp_path / "repo")
        worktree = tmp_path / "repo-wt"
        _git("worktree", "add", "-q", str(worktree), "-b", "side", cwd=repo)

        top_repo = _git("rev-parse", "--show-toplevel", cwd=repo).stdout.strip()
        top_wt = _git("rev-parse", "--show-toplevel", cwd=worktree).stdout.strip()

        assert os.path.realpath(top_repo) != os.path.realpath(top_wt)
        assert resolve_project_key(str(repo)) == resolve_project_key(str(worktree))

    def test_separate_repositories_are_distinct_projects(self, tmp_path: Path) -> None:
        one = _make_repo(tmp_path / "one")
        two = _make_repo(tmp_path / "two")
        assert resolve_project_key(str(one)) != resolve_project_key(str(two))

    def test_subdirectory_resolves_to_its_repository(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        nested = repo / "a" / "b"
        nested.mkdir(parents=True)
        assert resolve_project_key(str(nested)) == resolve_project_key(str(repo))

    def test_file_path_resolves_to_its_directory(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        assert resolve_project_key(str(repo / "seed.txt")) == resolve_project_key(str(repo))

    def test_repository_key_names_the_common_directory(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        key = resolve_project_key(str(repo))
        assert key.startswith("git:")
        assert key.endswith(os.path.realpath(str(repo / ".git")))


@pytest.mark.unit
class TestProjectKeyFallback:
    def test_non_repositories_are_not_pooled_together(self, tmp_path: Path) -> None:
        """Two non-repository directories are two projects, not one bucket."""
        a = tmp_path / "plain-a"
        b = tmp_path / "plain-b"
        a.mkdir()
        b.mkdir()
        key_a = resolve_project_key(str(a))
        key_b = resolve_project_key(str(b))
        assert key_a != key_b
        assert key_a.startswith("path:")
        assert key_b.startswith("path:")

    def test_missing_path_falls_back_to_a_parent_or_unknown(self, tmp_path: Path) -> None:
        key = resolve_project_key(str(tmp_path / "does-not-exist"))
        assert isinstance(key, str) and key

    def test_never_raises_on_hostile_input(self) -> None:
        for candidate in ("", "   ", "\0bad", "relative/path", "/proc/self/mem", None):
            key = resolve_project_key(candidate)  # type: ignore[arg-type]
            assert isinstance(key, str) and key

    def test_unresolvable_path_is_the_unknown_bucket(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import mind_mem.project_key as pk

        monkeypatch.setattr(pk, "_probe_directory", lambda _path: None)
        assert resolve_project_key("/anywhere") == PROJECT_KEY_UNKNOWN

    def test_absent_git_degrades_to_the_path_form(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No git binary is a failed probe, not an exception."""
        repo_like = tmp_path / "repo"
        repo_like.mkdir()
        monkeypatch.setattr("mind_mem.project_key.shutil.which", lambda _name: None)
        key = resolve_project_key(str(repo_like))
        assert key.startswith("path:")

    def test_key_length_is_bounded(self, tmp_path: Path) -> None:
        deep = tmp_path
        while len(str(deep)) <= PROJECT_KEY_MAX_LEN + 40:
            deep = deep / ("segment" * 4)
        deep.mkdir(parents=True)
        key = resolve_project_key(str(deep))
        assert len(key) <= PROJECT_KEY_MAX_LEN
        assert key.startswith("path#")

    def test_resolution_is_memoised(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """One probe per directory per process — this sits on a write path."""
        import mind_mem.project_key as pk

        calls = {"n": 0}
        real = pk._git_common_dir

        def counting(directory: str):
            calls["n"] += 1
            return real(directory)

        monkeypatch.setattr(pk, "_git_common_dir", counting)
        for _ in range(5):
            resolve_project_key(str(tmp_path))
        assert calls["n"] == 1


# ---------------------------------------------------------------------------
# 2. Edge provenance
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeProvenanceMigration:
    def test_column_is_added_to_a_legacy_table(self, tmp_path: Path) -> None:
        """A store created before provenance existed gains the column."""
        ws = str(tmp_path)
        ensure_graph_tables(ws)  # legacy shape: no kind, no origin_project
        conn = sqlite3.connect(str(_db(tmp_path)))
        try:
            conn.execute(
                "INSERT INTO co_retrieval (mem1_id, mem2_id, weight, hit_count, updated_at) VALUES ('OLD-A','OLD-B',1.0,1,'2026-01-01')"
            )
            conn.commit()
            before = {r[1] for r in conn.execute("PRAGMA table_info(co_retrieval)").fetchall()}
        finally:
            conn.close()
        assert "origin_project" not in before

        ensure_lineage_schema(ws)

        cols = {r[1] for r in _rows(tmp_path, "PRAGMA table_info(co_retrieval)")}
        assert "origin_project" in cols
        legacy = _rows(tmp_path, "SELECT origin_project FROM co_retrieval WHERE mem1_id='OLD-A'")
        assert legacy == [(NO_ORIGIN_PROJECT,)]

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        for _ in range(3):
            ensure_lineage_schema(ws)
        cols = [r[1] for r in _rows(tmp_path, "PRAGMA table_info(co_retrieval)")]
        assert cols.count("origin_project") == 1


@pytest.mark.unit
class TestAddBlockEdgeProvenance:
    def test_explicit_project_is_recorded(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        add_block_edge(ws, "A", "B", "supports", origin_project="git:/p/one")
        assert _rows(tmp_path, "SELECT origin_project FROM co_retrieval") == [("git:/p/one",)]

    def test_default_records_the_resolved_project(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        add_block_edge(ws, "A", "B", "supports")
        stored = _rows(tmp_path, "SELECT origin_project FROM co_retrieval")[0][0]
        assert stored == resolve_project_key()
        assert stored != NO_ORIGIN_PROJECT

    def test_provenance_can_be_declined(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        add_block_edge(ws, "A", "B", "supports", origin_project=NO_ORIGIN_PROJECT)
        assert _rows(tmp_path, "SELECT origin_project FROM co_retrieval") == [(NO_ORIGIN_PROJECT,)]

    def test_unprovenanced_row_adopts_the_next_writer(self, tmp_path: Path) -> None:
        """Same first-writer-wins rule the `kind` column already uses."""
        ws = str(tmp_path)
        add_block_edge(ws, "A", "B", "supports", origin_project=NO_ORIGIN_PROJECT)
        add_block_edge(ws, "A", "B", "supports", origin_project="git:/p/two")
        assert _rows(tmp_path, "SELECT origin_project FROM co_retrieval") == [("git:/p/two",)]

    def test_first_project_wins_on_re_assertion(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        add_block_edge(ws, "A", "B", "supports", origin_project="git:/p/one")
        add_block_edge(ws, "A", "B", "supports", origin_project="git:/p/two")
        assert _rows(tmp_path, "SELECT origin_project, hit_count FROM co_retrieval") == [("git:/p/one", 2)]

    def test_existing_validation_is_unchanged(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        with pytest.raises(ValueError):
            add_block_edge(ws, "A", "A", "supports")
        with pytest.raises(ValueError):
            add_block_edge(ws, "A", "B", "not-a-kind")


@pytest.mark.unit
class TestDistinctProjectCounts:
    def test_counts_distinct_projects_on_incoming_edges(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        add_block_edge(ws, "A", "TARGET", "supports", origin_project="git:/p/one")
        add_block_edge(ws, "B", "TARGET", "supports", origin_project="git:/p/two")
        add_block_edge(ws, "C", "TARGET", "supports", origin_project="git:/p/two")
        assert distinct_project_count(ws, "TARGET") == 2

    def test_edge_count_and_breadth_are_different_signals(self, tmp_path: Path) -> None:
        """Three edges from one project are not three independent observers."""
        ws = str(tmp_path)
        for src in ("A", "B", "C"):
            add_block_edge(ws, src, "TARGET", "supports", origin_project="git:/p/one")
        assert distinct_project_count(ws, "TARGET") == 1
        assert len(_rows(tmp_path, "SELECT 1 FROM co_retrieval WHERE mem2_id='TARGET'")) == 3

    def test_outgoing_edges_do_not_count(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        add_block_edge(ws, "TARGET", "X", "supports", origin_project="git:/p/one")
        assert distinct_project_count(ws, "TARGET") == 0

    def test_unprovenanced_edges_are_excluded(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        add_block_edge(ws, "A", "TARGET", "supports", origin_project=NO_ORIGIN_PROJECT)
        assert distinct_project_count(ws, "TARGET") == 0

    def test_kind_filter_narrows_the_count(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        add_block_edge(ws, "A", "TARGET", "supports", origin_project="git:/p/one")
        add_block_edge(ws, "B", "TARGET", "contradicts", origin_project="git:/p/two")
        assert distinct_project_count(ws, "TARGET") == 2
        assert distinct_project_count(ws, "TARGET", kind_filter="supports") == 1

    def test_batch_and_single_agree(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        add_block_edge(ws, "A", "ONE", "supports", origin_project="git:/p/one")
        add_block_edge(ws, "B", "TWO", "supports", origin_project="git:/p/two")
        batch = distinct_project_counts(ws, ["ONE", "TWO", "ABSENT"])
        assert batch == {"ONE": 1, "TWO": 1}
        assert distinct_project_count(ws, "ONE") == batch["ONE"]

    def test_empty_id_list_returns_empty(self, tmp_path: Path) -> None:
        assert distinct_project_counts(str(tmp_path), []) == {}


# ---------------------------------------------------------------------------
# 3. The score component
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBreadthComponent:
    def test_saturation_is_low_because_independence_saturates_fast(self) -> None:
        assert MATURITY_PROJECT_SATURATION == 3
        assert MATURITY_PROJECT_SATURATION < MATURITY_EDGE_SATURATION

    def test_more_projects_score_higher_at_equal_edge_count(self) -> None:
        one = maturity_score({"Lifecycle": "ephemeral"}, incoming_edge_count=5, distinct_project_count=1)
        three = maturity_score({"Lifecycle": "ephemeral"}, incoming_edge_count=5, distinct_project_count=3)
        assert three > one

    def test_breadth_saturates(self) -> None:
        at = maturity_score({"Lifecycle": "ephemeral"}, incoming_edge_count=5, distinct_project_count=MATURITY_PROJECT_SATURATION)
        beyond = maturity_score({"Lifecycle": "ephemeral"}, incoming_edge_count=5, distinct_project_count=MATURITY_PROJECT_SATURATION * 4)
        assert at == beyond

    def test_zero_and_negative_breadth_contribute_nothing(self) -> None:
        base = maturity_score({"Lifecycle": "ephemeral"}, incoming_edge_count=5, distinct_project_count=0)
        negative = maturity_score({"Lifecycle": "ephemeral"}, incoming_edge_count=5, distinct_project_count=-7)
        assert base == negative
        assert abs(base - MATURITY_EDGE_WEIGHT_WITH_BREADTH) < 1e-9

    def test_saturated_breadth_contributes_its_full_weight(self) -> None:
        s = maturity_score(
            {"Status": "deprecated", "Lifecycle": "ephemeral"},
            incoming_edge_count=0,
            distinct_project_count=MATURITY_PROJECT_SATURATION,
        )
        assert abs(s - MATURITY_PROJECT_WEIGHT) < 1e-9

    def test_score_stays_in_unit_range(self) -> None:
        for status in ("active", "wip", "deprecated", ""):
            for lifecycle in ("durable", "generated", "ephemeral", ""):
                for edges in (0, 1, 5, 50):
                    for projects in (None, 0, 1, 3, 99):
                        s = maturity_score(
                            {"Status": status, "Lifecycle": lifecycle},
                            incoming_edge_count=edges,
                            distinct_project_count=projects,
                        )
                        assert 0.0 <= s <= 1.0


@pytest.mark.unit
class TestWeightsRebalancedNotAppended:
    def test_both_profiles_sum_to_exactly_one(self) -> None:
        legacy = MATURITY_STATUS_WEIGHT + MATURITY_LIFECYCLE_WEIGHT + MATURITY_EDGE_WEIGHT
        breadth = MATURITY_STATUS_WEIGHT + MATURITY_LIFECYCLE_WEIGHT + MATURITY_EDGE_WEIGHT_WITH_BREADTH + MATURITY_PROJECT_WEIGHT
        assert legacy == 1.0
        assert breadth == 1.0

    def test_breadth_weight_is_taken_out_of_the_edge_weight(self) -> None:
        """Appended, the clamp would become load-bearing. It must not."""
        assert MATURITY_EDGE_WEIGHT_WITH_BREADTH + MATURITY_PROJECT_WEIGHT == MATURITY_EDGE_WEIGHT
        assert MATURITY_EDGE_WEIGHT_WITH_BREADTH < MATURITY_EDGE_WEIGHT

    def test_the_clamp_is_never_load_bearing(self) -> None:
        """The maximum composite lands exactly on 1.0, not above it."""
        best = maturity_score(
            {"Status": "active", "Lifecycle": "durable"},
            incoming_edge_count=MATURITY_EDGE_SATURATION,
            distinct_project_count=MATURITY_PROJECT_SATURATION,
        )
        unclamped = MATURITY_STATUS_WEIGHT + MATURITY_LIFECYCLE_WEIGHT + MATURITY_EDGE_WEIGHT_WITH_BREADTH + MATURITY_PROJECT_WEIGHT
        assert best == unclamped == 1.0


# --- The default path, proved unchanged against a frozen reference ---------


def _reference_score_pre_breadth(block: dict, incoming_edge_count: int | None) -> float:
    """The scoring code exactly as it stood before Group S landed.

    Copied verbatim from the pre-change implementation so the golden does
    not drift with the module under test.  If a future change moves the
    default path, this comparison fails — which is the point.
    """
    _missing = object()
    raw = block.get("Maturity", _missing)
    if raw is _missing:
        raw = block.get("maturity", _missing)
    if raw is not _missing:
        try:
            return max(0.0, min(1.0, float(raw)))
        except (ValueError, TypeError):
            pass

    status = str(block.get("Status") or block.get("status") or "").strip().lower()
    if status == "active":
        status_c = 1.0
    elif status in ("wip", "in-progress", "in_progress"):
        status_c = 0.5
    else:
        status_c = 0.0

    lifecycle = str(block.get("Lifecycle") or block.get("lifecycle") or "durable").strip().lower()
    if lifecycle == "durable":
        life_c = 1.0
    elif lifecycle == "generated":
        life_c = 0.5
    else:
        life_c = 0.0

    if incoming_edge_count is None or incoming_edge_count <= 0:
        edge_c = 0.0
    else:
        edge_c = min(1.0, incoming_edge_count / 5)

    return min(1.0, status_c * 0.3 + life_c * 0.2 + edge_c * 0.5)


@pytest.mark.unit
class TestDefaultPathUnchanged:
    def test_every_pre_breadth_call_is_bit_identical(self) -> None:
        """Exhaustive grid, compared on the float's bits, not on a tolerance."""
        maturities = (None, "0.0", "0.42", "1.0", "-3", "nonsense", 0.75)
        checked = 0
        for status in ("active", "wip", "deprecated", "archived", "", "ACTIVE"):
            for lifecycle in ("durable", "generated", "ephemeral", "", "DURABLE"):
                for edges in (None, 0, -1, 1, 2, 4, 5, 6, 500):
                    for maturity in maturities:
                        block: dict = {"Status": status, "Lifecycle": lifecycle}
                        if maturity is not None:
                            block["Maturity"] = maturity
                        got = maturity_score(block, incoming_edge_count=edges)
                        want = _reference_score_pre_breadth(block, edges)
                        assert got.hex() == want.hex(), f"{block} edges={edges}: {got!r} != {want!r}"
                        checked += 1
        assert checked == 6 * 5 * 9 * 7

    def test_omitting_breadth_matches_passing_none(self) -> None:
        for edges in (None, 0, 3, 5, 99):
            a = maturity_score({"Status": "active"}, incoming_edge_count=edges)
            b = maturity_score({"Status": "active"}, incoming_edge_count=edges, distinct_project_count=None)
            assert a.hex() == b.hex()

    def test_unknown_breadth_is_not_penalised_relative_to_known_breadth(self) -> None:
        """A caller that cannot supply breadth keeps the full edge weight."""
        unknown = maturity_score({"Lifecycle": "ephemeral"}, incoming_edge_count=MATURITY_EDGE_SATURATION)
        assert abs(unknown - MATURITY_EDGE_WEIGHT) < 1e-9


# --- Statelessness: a hard constraint, not a preference -------------------


@pytest.mark.unit
class TestStatelessness:
    def test_source_imports_nothing_from_the_lineage_layer(self) -> None:
        tree = ast.parse((_SRC / "block_maturity.py").read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(f"{node.module or ''}.{alias.name}" for alias in node.names)
        forbidden = {"block_lineage", "retrieval_graph", "sqlite3", "sqlite_index"}
        offenders = {name for name in imported if any(bad in name for bad in forbidden)}
        assert not offenders, f"block_maturity must stay stateless; found imports: {sorted(offenders)}"

    def test_importing_the_module_loads_no_lineage_code(self, tmp_path: Path) -> None:
        """Catches a *transitive* dependency the AST scan cannot see."""
        probe = tmp_path / "probe.py"
        probe.write_text(
            textwrap.dedent(
                f"""
                import importlib.util, sys
                spec = importlib.util.spec_from_file_location("_bm_probe", {str(_SRC / "block_maturity.py")!r})
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                leaked = [n for n in sys.modules if "block_lineage" in n or "retrieval_graph" in n]
                print("LEAKED:" + ",".join(sorted(leaked)))
                """
            ),
            encoding="utf-8",
        )
        out = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, check=True, encoding="utf-8", errors="replace")
        assert out.stdout.strip() == "LEAKED:", out.stdout

    def test_scoring_needs_no_workspace(self) -> None:
        """The signature proves it: breadth arrives as an int, not a path."""
        assert maturity_score({"Status": "active"}, incoming_edge_count=5, distinct_project_count=3) > 0.0


# ---------------------------------------------------------------------------
# 4. The corpus delta scan
# ---------------------------------------------------------------------------


def _seed_corpus(workspace: Path, blocks: dict[str, str]) -> None:
    """Create the index `blocks` table with the real schema and fill it."""
    from mind_mem.sqlite_index import _init_schema

    _db(workspace).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_db(workspace)))
    try:
        _init_schema(conn)
        for block_id, status in blocks.items():
            conn.execute(
                "INSERT OR REPLACE INTO blocks (id, type, file, line, status, json_blob) VALUES (?,?,?,?,?,?)",
                (block_id, "adr", "corpus.md", 0, status, '{"metadata": {"Lifecycle": "durable"}}'),
            )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.unit
class TestBreadthScan:
    def test_reports_deflation_when_no_edge_carries_provenance(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        _seed_corpus(tmp_path, {"TARGET": "active", "A": "active", "B": "active"})
        add_block_edge(ws, "A", "TARGET", "supports", origin_project=NO_ORIGIN_PROJECT)
        add_block_edge(ws, "B", "TARGET", "supports", origin_project=NO_ORIGIN_PROJECT)

        report = scan_maturity_breadth(ws)

        assert report.edges_total == 2
        assert report.edges_with_provenance == 0
        assert report.blocks_reduced == 1
        assert report.mean_delta < 0.0
        assert any("provenance" in note for note in report.notes)

    def test_breadth_offsets_the_lower_edge_weight(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        _seed_corpus(tmp_path, {"TARGET": "active", "A": "active", "B": "active", "C": "active"})
        for src, project in (("A", "git:/p/one"), ("B", "git:/p/two"), ("C", "git:/p/three")):
            add_block_edge(ws, src, "TARGET", "supports", origin_project=project)

        report = scan_maturity_breadth(ws)

        assert report.edges_with_provenance == 3
        assert report.distinct_projects_seen == 3
        target = next(s for s in report.samples if s["block_id"] == "TARGET")
        assert target["distinct_projects"] == 3
        assert target["rebalanced_score"] > target["current_score"]

    def test_changes_nothing(self, tmp_path: Path) -> None:
        """Read-only means the bytes on disk are the same afterwards."""
        ws = str(tmp_path)
        _seed_corpus(tmp_path, {"TARGET": "active", "A": "active"})
        add_block_edge(ws, "A", "TARGET", "supports", origin_project="git:/p/one")

        before = hashlib.sha256(_db(tmp_path).read_bytes()).hexdigest()
        scan_maturity_breadth(ws)
        after = hashlib.sha256(_db(tmp_path).read_bytes()).hexdigest()
        assert before == after

    def test_absent_lineage_graph_is_reported_as_absent(self, tmp_path: Path) -> None:
        """An index that never held an edge is not the same as one holding none."""
        _seed_corpus(tmp_path, {"A": "active"})
        report = scan_maturity_breadth(str(tmp_path))
        assert report.blocks_total == 1
        assert report.edges_total == 0
        assert any("no co_retrieval table" in note for note in report.notes)

    def test_missing_index_is_a_note_not_an_exception(self, tmp_path: Path) -> None:
        report = scan_maturity_breadth(str(tmp_path / "empty"))
        assert report.blocks_total == 0
        assert any("no recall index" in note for note in report.notes)

    def test_rejects_an_empty_workspace_argument(self) -> None:
        with pytest.raises(ValueError):
            scan_maturity_breadth("")

    def test_report_is_json_serialisable(self, tmp_path: Path) -> None:
        import json

        _seed_corpus(tmp_path, {"A": "active"})
        json.dumps(scan_maturity_breadth(str(tmp_path)).to_dict())
