"""RA.5 — the lifecycle-tier dashboard, and the four refusals it inherits.

RA.0's collapse to one ladder exists so this dashboard can exist; RA.2's
derived views exist so it has numbers to show. What has to be tested is not
the arithmetic — it is that a *render surface* over a governed store did not
quietly become a write path, a read surface, or a false green.

D1  **It stores nothing and creates nothing.** A dashboard over a workspace
    with no tier store leaves it with no tier store. Proven behaviourally and
    structurally, each with a positive control, because "the scanner found no
    writes" from a scanner that cannot see a write is a vacuous pass.

D2  **It never names a block id.** The tier panel publishes counts only. The
    negative assertion is paired with proof the ids exist in the tier store
    AND in the admitted corpus, so their absence is a refusal rather than an
    empty workspace.

D3  **An absent source is unavailable with a reason, never a zero.** No tier
    store means ``available=False``, never "zero blocks are VERIFIED".

D4  **The ledger's chain verdict is reachable.** ``verify_served_chain`` has
    existed since RA.1 and was callable from the test suite and a docs
    paragraph and from nothing else. The dashboard runs it, and ``mm
    dashboard`` exits non-zero when it fails — checked by breaking a row and
    watching the exit code move.

D5  **The store path cannot drift.** ``TIER_STORE_RELPATH`` is a second
    spelling of a path built inline in ``compaction.py``; a ratchet pins them
    together so moving the store fails the build instead of silently emptying
    the panel.

D6  **Nothing on the scoring path can reach it.** This module imports the
    served-set ledger, so an import edge into scoring would be RA.1's rail
    breached two hops out.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

import mind_mem
from mind_mem import accountability_dashboard as dash
from mind_mem.accountability_dashboard import (
    DASHBOARD_TAG,
    TIER_NAMES,
    TIER_STORE_RELPATH,
    _line,
    dashboard,
    ledger_panel,
    render,
    tier_census,
)
from mind_mem.admissibility import RELEASE_FIELD
from mind_mem.memory_tiers import MemoryTier, TierManager
from mind_mem.recall_digests import query_hash, served_set_digest
from mind_mem.retention_class import GOVERNED, PROTECTED
from mind_mem.served_ledger import append_served_run, ledger_path
from mind_mem.storage import get_block_store

SRC = pathlib.Path(mind_mem.__file__).parent

PIPELINE = "b" * 64
ANCHOR = "c" * 64
INSTANT = "2026-09-01"

TIERED_A = "D-20260901-101"
TIERED_B = "D-20260901-102"
UNTIERED = "D-20260901-103"
RELEASE_DECISION = "D-20260901-104"
WITHHELD = "SIG-20260901-101"
RELEASABLE_ID = "IMP-20260901-101"


# ---------------------------------------------------------------------------
# Seeding — through the real writer for every store touched.
# ---------------------------------------------------------------------------


def _write_blocks(workspace: str, relpath: str, blocks: list[dict]) -> None:
    path = os.path.join(workspace, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for block in blocks:
            handle.write(f"[{block['_id']}]\n")
            for key, value in block.items():
                if key == "_id":
                    continue
                handle.write(f"{key}: {value}\n")
            handle.write("\n")


def _seed_corpus(workspace: str) -> None:
    """Four admitted blocks (one a live release decision) + one withheld."""
    _write_blocks(
        workspace,
        "decisions/DECISIONS.md",
        [
            {"_id": TIERED_A, "Statement": "the ladder moved this one", "Status": "active", "Date": "2026-08-27"},
            {"_id": TIERED_B, "Statement": "and this one", "Status": "active", "Date": "2026-08-26"},
            {"_id": UNTIERED, "Statement": "no promotion cycle ever saw this", "Status": "active", "Date": "2026-08-25"},
            {
                "_id": RELEASE_DECISION,
                "Statement": f"admit {RELEASABLE_ID}",
                "Status": "active",
                "Date": "2026-08-24",
                RELEASE_FIELD: RELEASABLE_ID,
            },
        ],
    )
    _write_blocks(
        workspace,
        "intelligence/CAPTURED.md",
        [{"_id": WITHHELD, "Statement": "an unreviewed captured signal", "Status": "quarantined", "Date": "2026-08-27"}],
    )


def _seed_tiers(workspace: str, *, include_withheld: bool = False) -> None:
    """Register + promote through ``TierManager`` itself, not raw SQL."""
    store = os.path.join(workspace, TIER_STORE_RELPATH)
    os.makedirs(os.path.dirname(store), exist_ok=True)
    with TierManager(store) as mgr:
        mgr._register_block(TIERED_A, MemoryTier.WORKING)
        mgr._register_block(TIERED_B, MemoryTier.WORKING)
        mgr._register_block(RELEASE_DECISION, MemoryTier.WORKING)
        assert mgr.promote(TIERED_B, MemoryTier.SHARED) is True
        if include_withheld:
            mgr._register_block(WITHHELD, MemoryTier.WORKING)


def _enable_ledger(workspace: str) -> None:
    with open(os.path.join(workspace, "mind-mem.json"), "w", encoding="utf-8") as handle:
        json.dump({"served_ledger": {"enabled": True}}, handle)


def _append_run(workspace: str, query: str, ids: tuple[str, ...]) -> None:
    row = append_served_run(
        workspace,
        query_hash=query_hash(query),
        served_digest=served_set_digest(ids),
        ids=ids,
        pipeline_hash=PIPELINE,
        index_anchor=ANCHOR,
        scoring_instant=INSTANT,
    )
    assert row is not None, "positive control: the ledger must be enabled or this test means nothing"


@pytest.fixture
def workspace(tmp_path) -> str:
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    return ws


def _cli(workspace: str, *argv: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONPATH": str(SRC.parent),
        "MIND_MEM_WORKSPACE": workspace,
    }
    return subprocess.run(
        [sys.executable, "-m", "mind_mem.mm_cli", *argv],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        encoding="utf-8",
        errors="replace",
    )


# ---------------------------------------------------------------------------
# D1 — stores nothing, creates nothing
# ---------------------------------------------------------------------------


def test_d1_a_dashboard_over_a_bare_workspace_creates_nothing(workspace: str) -> None:
    before = sorted(os.listdir(workspace))
    dashboard(workspace)
    assert sorted(os.listdir(workspace)) == before == []


def test_d1_the_creation_check_can_see_a_creation(workspace: str) -> None:
    """Positive control: the same listing DOES move when something is written."""
    before = sorted(os.listdir(workspace))
    store = os.path.join(workspace, TIER_STORE_RELPATH)
    os.makedirs(os.path.dirname(store), exist_ok=True)
    TierManager(store).close()
    assert sorted(os.listdir(workspace)) != before


def test_d1_the_dashboard_does_not_create_a_tier_store(workspace: str) -> None:
    """The one creation this module could plausibly cause, named explicitly."""
    _seed_corpus(workspace)
    dashboard(workspace)
    assert not os.path.exists(os.path.join(workspace, TIER_STORE_RELPATH))


_WRITE_SQL = re.compile(r"\b(insert|update|delete|create|drop|alter|replace)\s+(into|table|from|set|index|view)\b", re.I)


def _sql_literals(path: pathlib.Path) -> list[str]:
    """String constants that look like SQL, excluding docstrings."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        ast.get_docstring(node, clean=False)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value not in docstrings
    ]


def test_d1_the_module_contains_no_write_sql() -> None:
    literals = _sql_literals(SRC / "accountability_dashboard.py")
    offenders = [s for s in literals if _WRITE_SQL.search(s)]
    assert offenders == [], f"the dashboard names write SQL: {offenders}"


def test_d1_the_write_sql_scanner_can_see_write_sql() -> None:
    """Positive control for the scan above — it is not a regex that matches nothing."""
    assert _WRITE_SQL.search("INSERT INTO block_tiers VALUES (1)")
    offenders = [s for s in _sql_literals(SRC / "memory_tiers.py") if _WRITE_SQL.search(s)]
    assert offenders, "scanner found no write SQL in the module that writes tiers — it is broken"


def test_d1_every_connection_comes_from_the_shared_read_only_opener() -> None:
    """No second opener: ``sqlite3.connect`` must not be called in this module."""
    tree = ast.parse((SRC / "accountability_dashboard.py").read_text(encoding="utf-8"))
    calls = {
        ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "sqlite3.connect" not in calls, "the dashboard opens its own connection instead of the ro rail"
    assert "_read_only_connect" in calls, "the dashboard does not use the shared read-only opener at all"


# ---------------------------------------------------------------------------
# D2 — counts, never ids
# ---------------------------------------------------------------------------


def test_d2_the_tier_panel_names_no_block_id(workspace: str) -> None:
    _seed_corpus(workspace)
    _seed_tiers(workspace, include_withheld=True)

    # Positive controls: the ids exist in BOTH stores this panel reads, so an
    # absent id is a refusal to publish it and not an empty workspace.
    store = os.path.join(workspace, TIER_STORE_RELPATH)
    with TierManager(store) as mgr:
        assert mgr.get_tier(TIERED_B) is MemoryTier.SHARED
        assert mgr.get_tier(WITHHELD) is MemoryTier.WORKING
    assert {b["_id"] for b in get_block_store(workspace).get_all()} >= {TIERED_A, WITHHELD}

    census = json.dumps(tier_census(workspace).to_dict())
    for block_id in (TIERED_A, TIERED_B, UNTIERED, RELEASE_DECISION, WITHHELD):
        assert block_id not in census, f"the tier panel published {block_id}"


def test_d2_a_tiered_but_withheld_block_is_counted_and_never_named(workspace: str) -> None:
    _seed_corpus(workspace)
    _seed_tiers(workspace, include_withheld=True)
    census = tier_census(workspace)
    assert census.tracked == 4, "positive control: four blocks were registered on the ladder"
    assert census.tracked_not_admitted == 1, "the quarantined block is tiered and not admitted"
    assert WITHHELD not in json.dumps(census.to_dict())


# ---------------------------------------------------------------------------
# D3 — unavailable with a reason, never a zero
# ---------------------------------------------------------------------------


def test_d3_no_tier_store_is_unavailable_with_a_reason(workspace: str) -> None:
    _seed_corpus(workspace)
    census = tier_census(workspace)
    assert census.available is False
    assert "promotion cycle has never run" in census.reason
    assert census.by_tier == {}, "an unavailable census must not report a ladder of zeroes"


def test_d3_a_real_ladder_is_counted_per_rung(workspace: str) -> None:
    """Positive control for the test above: with a store, the counts are real."""
    _seed_corpus(workspace)
    _seed_tiers(workspace)
    census = tier_census(workspace)
    assert census.available is True
    assert census.by_tier == {"WORKING": 2, "SHARED": 1, "LONG_TERM": 0, "VERIFIED": 0}
    assert set(census.by_tier) == set(TIER_NAMES)


def test_d3_untracked_admitted_is_not_folded_into_working(workspace: str) -> None:
    """``get_tier`` answers WORKING for an unregistered block; a census may not.

    Three of the four admitted blocks are on the ladder, one has never been
    registered. Folding it into WORKING would report a promotion cycle that
    covered the whole corpus when it covered three quarters of it.
    """
    _seed_corpus(workspace)
    _seed_tiers(workspace)
    census = tier_census(workspace)
    assert census.corpus_admitted == 4
    assert census.tracked_admitted == 3
    assert census.untracked_admitted == 1
    assert census.by_tier["WORKING"] == 2, "the untracked block leaked into WORKING"


def test_d3_the_tier_and_retention_axes_are_crossed_not_merged(workspace: str) -> None:
    """A WORKING block can be PROTECTED — that is why both axes are rendered."""
    _seed_corpus(workspace)
    _seed_tiers(workspace)
    grid = tier_census(workspace).by_tier_retention
    assert grid["WORKING"][PROTECTED] == 1, "the live release decision is PROTECTED and on the first rung"
    assert grid["WORKING"][GOVERNED] == 1
    assert grid["SHARED"][GOVERNED] == 1


# ---------------------------------------------------------------------------
# D4 — the ledger's verdict, reachable at last
# ---------------------------------------------------------------------------


def test_d4_a_clean_ledger_verifies_through_the_dashboard(workspace: str) -> None:
    _enable_ledger(workspace)
    _append_run(workspace, "why did the rollout land", (TIERED_A, TIERED_B))
    panel = ledger_panel(workspace)
    assert panel.ok is True
    assert panel.rows == 1, "positive control: the row must exist for the verdict to mean anything"
    assert panel.enabled is True and panel.present is True


def test_d4_a_tampered_row_fails_the_panel_and_names_it(workspace: str) -> None:
    _enable_ledger(workspace)
    _append_run(workspace, "first", (TIERED_A,))
    _append_run(workspace, "second", (TIERED_B,))
    assert ledger_panel(workspace).ok is True, "positive control: clean before the edit"

    path = ledger_path(workspace)
    rows = [json.loads(line) for line in open(path, encoding="utf-8").read().splitlines() if line.strip()]
    rows[0]["ids"] = [UNTIERED]
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    panel = ledger_panel(workspace)
    assert panel.ok is False
    assert panel.bad_seq == 0
    assert "served_digest does not match ids" in panel.reason


def test_d4_an_absent_ledger_is_a_stated_pass_not_a_bare_green(workspace: str) -> None:
    """An absent ledger is green, and the reason must say WHY it is green.

    This assertion used to pin the phrase "default OFF". That was true
    through 5.0.1 and is false from 5.0.2: ``served_ledger.ledger_enabled``
    returns True for a workspace whose config sets nothing, which
    ``tests/test_docs_alignment.py::TestServedLedgerDefault`` computes from
    the function rather than quoting from a comment. The gate was not
    measuring the default -- it was pinning a sentence, and the sentence it
    pinned was printed into an operator-facing verdict. Pinned to the live
    default instead, and additionally to the opt-out spelling, which is what
    an operator who reads "no ledger" needs next. Strictly stronger than the
    single substring it replaces; nothing was relaxed.
    """
    panel = ledger_panel(workspace)
    assert panel.ok is True
    assert panel.present is False
    assert "records by default since 5.0.2" in panel.reason
    assert '{"served_ledger": {"enabled": false}}' in panel.reason


# ---------------------------------------------------------------------------
# D5 — the store path cannot drift away from its writer
# ---------------------------------------------------------------------------


def test_d5_the_tier_store_path_matches_the_one_compaction_builds() -> None:
    """A second spelling of a path, pinned to the first by a ratchet.

    ``compaction.run_promotion_cycle`` builds ``<ws>/intelligence/tiers.db``
    inline; it is not a constant this module can import. If it moves, this
    fails in the same commit rather than leaving the panel permanently
    ``available=False`` — which would read as "no promotion cycle has run".
    """
    tree = ast.parse((SRC / "compaction.py").read_text(encoding="utf-8"))
    joins = [
        [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("path.join")
    ]
    assert joins, "vacuity guard: found no os.path.join call in compaction.py at all"
    expected = list(pathlib.PurePath(TIER_STORE_RELPATH).parts)
    assert any(parts == expected for parts in joins), (
        f"compaction.py no longer builds {TIER_STORE_RELPATH!r}; the dashboard reads a store nothing writes"
    )


# ---------------------------------------------------------------------------
# D6 — the rail: the scoring path cannot reach a ledger through this module
# ---------------------------------------------------------------------------


def _eager_imports(module: str) -> set[str]:
    """First-party modules *module* imports at module level."""
    path = SRC.joinpath(*module.split(".")).with_suffix(".py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found |= {a.name[len("mind_mem.") :] for a in node.names if a.name.startswith("mind_mem.")}
        elif isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            found.add(node.module)
    return found


def _closure(start: str) -> set[str]:
    seen = {start}
    stack = [start]
    while stack:
        for nxt in _eager_imports(stack.pop()):
            if nxt not in seen and SRC.joinpath(*nxt.split(".")).with_suffix(".py").is_file():
                seen.add(nxt)
                stack.append(nxt)
    return seen


def test_d6_the_dashboard_does_reach_the_ledger() -> None:
    """Vacuity guard for the rail below: the walker can see the edge that EXISTS."""
    assert "served_ledger" in _closure("accountability_dashboard")


def test_d6_the_scoring_path_cannot_reach_the_dashboard() -> None:
    """An import edge here would be RA.1's rail breached two hops out."""
    for scorer in ("_recall_core", "recall_attestation"):
        assert "accountability_dashboard" not in _closure(scorer), f"{scorer} reaches the dashboard"


# ---------------------------------------------------------------------------
# The render + the CLI verb
# ---------------------------------------------------------------------------


def test_render_is_pure_and_deterministic(workspace: str) -> None:
    _seed_corpus(workspace)
    _seed_tiers(workspace)
    report = dashboard(workspace)
    assert render(report) == render(report)
    assert render(json.loads(json.dumps(report))) == render(report), "render must survive a JSON round trip"


def test_render_shows_every_rung_of_the_one_ladder(workspace: str) -> None:
    _seed_corpus(workspace)
    _seed_tiers(workspace)
    text = render(dashboard(workspace))
    for name in TIER_NAMES:
        assert name in text
    assert "memory_tiers.MemoryTier" in text, "the dashboard must say which axis it is rendering"


def test_the_cli_verb_is_wired_and_reports_the_schema(workspace: str) -> None:
    _seed_corpus(workspace)
    _seed_tiers(workspace)
    result = _cli(workspace, "dashboard", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema"] == DASHBOARD_TAG
    assert payload["tiers"]["available"] is True


def test_the_cli_verb_exits_nonzero_on_a_broken_chain(workspace: str) -> None:
    """The gate: a tamper-evidence check nobody can fail on is decoration."""
    _enable_ledger(workspace)
    _append_run(workspace, "first", (TIERED_A,))
    _append_run(workspace, "second", (TIERED_B,))
    assert _cli(workspace, "dashboard").returncode == 0, "positive control: clean chain exits 0"

    path = ledger_path(workspace)
    rows = [json.loads(line) for line in open(path, encoding="utf-8").read().splitlines() if line.strip()]
    rows[0]["ids"] = [UNTIERED]
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    broken = _cli(workspace, "dashboard")
    assert broken.returncode == 1
    assert "FAILED" in broken.stdout


def test_the_dashboard_reuses_the_accountability_report_verbatim(workspace: str) -> None:
    """One set of numbers: a re-derived copy is a second place to disagree."""
    from mind_mem.accountability_views import accountability_report

    _seed_corpus(workspace)
    assert dashboard(workspace)["accountability"] == accountability_report(workspace)
    assert dash.accountability_report is accountability_report


# ---------------------------------------------------------------------------
# D7 — the renderer may not invent a number
# ---------------------------------------------------------------------------


def _seed_served(workspace: str) -> None:
    """A workspace with corpus, ladder, a logged recall and a ledger row."""
    from mind_mem.retrieval_graph import log_retrieval

    _seed_corpus(workspace)
    _seed_tiers(workspace)
    _enable_ledger(workspace)
    log_retrieval(
        workspace,
        "why did the rollout land",
        [{"_id": TIERED_A, "score": 9.0}, {"_id": TIERED_B, "score": 4.0}],
        intent_type="WHY",
    )
    _append_run(workspace, "why did the rollout land", (TIERED_A, TIERED_B))


def test_d7_the_rendered_numbers_are_the_views_own_numbers(workspace: str) -> None:
    """The defect this panel shipped with, and the assertion that pins it.

    The first cut asked ``precision_by_intent`` for ``served`` / ``credited``
    — keys that view has never published — and printed ``0`` for both while
    dropping the per-intent rows. Every number on the page is now compared
    against the view it claims to come from.
    """
    from mind_mem.accountability_views import precision_by_intent, serve_counts, waste_view

    _seed_served(workspace)
    text = render(dashboard(workspace))

    precision = precision_by_intent(workspace)
    assert precision.available is True, "positive control: there is serve evidence to render"
    assert precision.rows, "positive control: there is at least one intent row"
    row = precision.rows[0]
    assert row.served_blocks == 2 and row.credited_blocks == 0
    assert f"{row.intent}" in text
    assert f"{row.credited_blocks}/{row.served_blocks} = {row.precision}" in text
    assert _line("serve observations", precision.observations) in text

    counts = serve_counts(workspace)
    assert counts.durable_serves == 2, "positive control: the ledger row contributed two durable serves"
    assert _line("durable serves", counts.durable_serves) in text

    waste = waste_view(workspace)
    assert waste.corpus_withheld == 1, "positive control: one quarantined block exists to be withheld"
    assert _line("withheld", waste.corpus_withheld) in text
    assert _line("unserved ratio", waste.unserved_ratio) in text


def test_d7_the_two_precisions_are_rendered_apart(workspace: str) -> None:
    """Block-level and run-level are different denominators, not one number."""
    _seed_served(workspace)
    text = render(dashboard(workspace))
    assert "PRECISION — block level" in text
    assert "PRECISION — run level" in text


@pytest.mark.parametrize(
    ("panel", "key"),
    [
        ("tiers", "tracked"),
        ("ledger", "ok"),
        ("ledger", "rows"),
    ],
)
def test_d7_a_missing_panel_key_raises_instead_of_printing_a_zero(workspace: str, panel: str, key: str) -> None:
    _seed_served(workspace)
    report = dashboard(workspace)
    assert render(report), "positive control: the intact report renders"
    del report[panel][key]
    with pytest.raises(KeyError, match=key):
        render(report)


@pytest.mark.parametrize("key", ["observations", "credit_rows", "rows", "window"])
def test_d7_a_missing_accountability_key_raises_too(workspace: str, key: str) -> None:
    """The drift that actually happened was inside the nested report, not the top level."""
    _seed_served(workspace)
    report = dashboard(workspace)
    assert render(report), "positive control: the intact report renders"
    del report["accountability"]["precision_by_intent"][key]
    with pytest.raises(KeyError, match=key):
        render(report)


# ---------------------------------------------------------------------------
# D8 — the determinism claim, gated rather than asserted
# ---------------------------------------------------------------------------

#: Names that read a wall clock. Group RA's determinism constraint is that a
#: report is reproducible on any host on any day, and a panel that read "now"
#: would silently break it — ``updated_at`` is in the tier store precisely so a
#: future author can be tempted.
_CLOCK_NAMES = frozenset({"now", "utcnow", "today", "time", "monotonic", "perf_counter"})


def _clock_reads(path: pathlib.Path) -> set[str]:
    """Attribute/function names in *path* that read a clock."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func).rsplit(".", 1)[-1]
            if name in _CLOCK_NAMES:
                found.add(ast.unparse(node.func))
    return found


def test_d8_no_panel_in_the_dashboard_reads_a_clock() -> None:
    """Every module a panel's numbers come from, not just this one."""
    for module in (
        "accountability_dashboard.py",
        "accountability_views.py",
        "retention_class.py",
        "replay_check.py",
    ):
        assert _clock_reads(SRC / module) == set(), f"{module} reads a clock on a report path"


def test_d8_the_clock_scanner_can_see_a_clock_read() -> None:
    """Positive control: the module that stamps ``updated_at`` DOES read one."""
    assert _clock_reads(SRC / "memory_tiers.py"), "scanner found no clock in the module that timestamps tiers"


def test_d8_two_renders_of_one_workspace_are_identical(workspace: str) -> None:
    """The behavioural half: no hidden input moves between two calls."""
    _seed_served(workspace)
    assert render(dashboard(workspace)) == render(dashboard(workspace))


def test_d3_a_store_without_the_ladders_table_says_so(workspace: str, tmp_path) -> None:
    """ "Missing table" and "empty table" are different facts about different things."""
    import sqlite3

    store = os.path.join(workspace, TIER_STORE_RELPATH)
    os.makedirs(os.path.dirname(store), exist_ok=True)
    conn = sqlite3.connect(store)
    conn.execute("CREATE TABLE something_else (id TEXT)")
    conn.commit()
    conn.close()
    assert os.path.isfile(store), "positive control: there IS a database to read"

    census = tier_census(workspace)
    assert census.available is False
    assert "no block_tiers table" in census.reason


def test_d3_an_empty_ladder_table_says_something_different(workspace: str) -> None:
    """Positive control for the test above: the two reasons are not one reason."""
    store = os.path.join(workspace, TIER_STORE_RELPATH)
    os.makedirs(os.path.dirname(store), exist_ok=True)
    TierManager(store).close()
    census = tier_census(workspace)
    assert census.available is False
    assert "holds no tier rows" in census.reason
