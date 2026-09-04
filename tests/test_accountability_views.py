"""RA.2 — precision and waste as derived views, and the four refusals that shape them.

The views themselves are arithmetic. What has to be tested is what the
module refuses to do, because each refusal is the difference between an
accountability report and a new place for a score to live.

A1  **It stores nothing and creates nothing.** A report over a workspace
    with no index leaves it with no index. Proven behaviourally (the
    directory listing is unchanged) and structurally (the source contains
    no write SQL, and every connection is opened ``mode=ro``) — with a
    positive control for each, because "found no writes" from a scanner
    that cannot see a write is the vacuous pass this repo has been bitten
    by twice.

A2  **It never names withheld content.** The waste view walks the corpus,
    so it is a read surface; it goes through ``admit_corpus``, the shared
    gate, and reports withheld blocks as a bare count. The negative
    assertion is paired with proof the block exists on disk AND that the
    store hands it over, so an absent id means the gate withheld it rather
    than that nothing was there.

A3  **"Unserved" is not "worthless".** A PROTECTED block with no serve
    evidence is reported under its own key with the reason it is
    protected, never in the waste list. This is the 5.0.0 deletion
    mistake made mechanically impossible in the one view whose whole
    subject is unused content.

A4  **An absent source is unavailable, not zero.** With no serve
    evidence at all, precision reports ``available=False`` with a reason.
    A precision of 0.0 computed over a table that does not exist is a
    measurement nobody made.

And the join RA.1 exists for: ``run_id_of_attestation`` mints the ledger
run id from the three values a recall attestation already publishes, so a
credit row keyed on it joins to the exact run that served the blocks.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import pathlib
import re
import sqlite3
import subprocess
import sys

import pytest

from mind_mem import accountability_views as av
from mind_mem.accountability_views import (
    INTENT_UNKNOWN,
    accountability_report,
    precision_by_intent,
    run_id_of_attestation,
    run_precision,
    waste_view,
)
from mind_mem.admissibility import RELEASE_FIELD
from mind_mem.calibration import CalibrationManager
from mind_mem.recall_digests import query_hash, served_set_digest
from mind_mem.retention_class import GOVERNED, PROTECTED
from mind_mem.retrieval_graph import graph_db_path, log_retrieval
from mind_mem.served_ledger import append_served_run, run_id
from mind_mem.storage import get_block_store

PIPELINE = "b" * 64
ANCHOR = "c" * 64
INSTANT = "2026-09-01"

SERVED_A = "D-20260901-001"
SERVED_B = "D-20260901-002"
UNSERVED = "D-20260901-003"
RELEASE_DECISION = "D-20260901-004"
WITHHELD = "SIG-20260901-001"
RELEASABLE_ID = "IMP-20260901-001"


# ---------------------------------------------------------------------------
# Seeding — every fixture writes through the real writer for that store, so a
# view that passes here is reading what the product actually records.
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
    """Four admitted blocks (one of them a live release decision) + one withheld."""
    _write_blocks(
        workspace,
        "decisions/DECISIONS.md",
        [
            {"_id": SERVED_A, "Statement": "retrieval rollout shipped", "Status": "active", "Date": "2026-08-27"},
            {"_id": SERVED_B, "Statement": "retrieval rollout reviewed", "Status": "active", "Date": "2026-08-26"},
            {"_id": UNSERVED, "Statement": "nobody has asked about this", "Status": "active", "Date": "2026-08-25"},
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


def _seed_runs(workspace: str) -> None:
    """Two logged recalls, through ``retrieval_graph.log_retrieval`` itself."""
    log_retrieval(
        workspace,
        "why did the retrieval rollout land",
        [{"_id": SERVED_A, "score": 9.0}, {"_id": SERVED_B, "score": 4.0}],
        intent_type="WHY",
    )
    log_retrieval(workspace, "when was the rollout reviewed", [{"_id": SERVED_B, "score": 5.0}], intent_type="WHEN")


def _enable_ledger(workspace: str) -> None:
    with open(os.path.join(workspace, "mind-mem.json"), "w", encoding="utf-8") as handle:
        json.dump({"served_ledger": {"enabled": True}}, handle)


def _append_run(workspace: str, query: str, ids: tuple[str, ...]) -> str:
    """Append one ledger row through the real writer; return its ``run_id``."""
    row = append_served_run(
        workspace,
        query_hash=query_hash(query),
        served_digest=served_set_digest(ids),
        ids=ids,
        pipeline_hash=PIPELINE,
        index_anchor=ANCHOR,
        scoring_instant=INSTANT,
    )
    assert row is not None, "positive control: the ledger must be enabled for this test to mean anything"
    return row.run_id


@pytest.fixture
def workspace(tmp_path) -> str:
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    return ws


# ---------------------------------------------------------------------------
# A1 — stores nothing, creates nothing
# ---------------------------------------------------------------------------


def test_a_report_over_a_bare_workspace_creates_nothing(workspace: str) -> None:
    before = sorted(os.listdir(workspace))
    accountability_report(workspace)
    assert sorted(os.listdir(workspace)) == before == []


def test_the_creation_check_can_see_a_creation(workspace: str) -> None:
    """Positive control for the test above — the same listing DOES move on a write."""
    from mind_mem.retrieval_graph import ensure_graph_tables

    assert sorted(os.listdir(workspace)) == []
    ensure_graph_tables(workspace)
    assert os.path.isfile(graph_db_path(workspace))
    assert sorted(os.listdir(workspace)) != []


def _db_content_digest(db: str) -> str:
    """SHA-256 over the database's LOGICAL content, via ``iterdump``.

    Hashing ``recall.db``'s bytes does NOT work here and looked like it did:
    the store runs in WAL mode, so a committed INSERT lands in the ``-wal``
    file and leaves the main database byte-identical. A digest over the raw
    file would therefore have been unable to move on a write — a positive
    control that cannot go green, i.e. an assertion proving nothing. The dump
    reflects every row wherever it currently lives.
    """
    conn = sqlite3.connect(db)
    try:
        return hashlib.sha256("\n".join(conn.iterdump()).encode("utf-8")).hexdigest()
    finally:
        conn.close()


def test_the_content_digest_can_see_a_write(workspace: str) -> None:
    """Positive control — without this the byte-identity assertion below is empty."""
    _seed_runs(workspace)
    db = graph_db_path(workspace)
    before = _db_content_digest(db)
    CalibrationManager(workspace).record_outcome([SERVED_A], "success", query_id="r1")
    assert _db_content_digest(db) != before


def test_a_report_leaves_the_store_unchanged(workspace: str) -> None:
    """A report writes no row anywhere, and creates no file of its own.

    Reading a WAL database materialises its ``-wal`` / ``-shm`` sidecars —
    SQLite's own bookkeeping for a reader, not a write by this module — so
    those two names are the only additions tolerated, and they are named
    rather than waved through by a loose comparison.
    """
    _seed_corpus(workspace)
    _seed_runs(workspace)
    CalibrationManager(workspace).record_outcome([SERVED_A], "success", query_id="r1")
    db = graph_db_path(workspace)

    before_files = {str(path) for path in pathlib.Path(workspace).rglob("*")}
    before_digest = _db_content_digest(db)

    accountability_report(workspace)

    assert _db_content_digest(db) == before_digest
    added = {str(path) for path in pathlib.Path(workspace).rglob("*")} - before_files
    assert all(name.endswith(("-wal", "-shm")) for name in added), f"the report created {sorted(added)}"


_WRITE_SQL = re.compile(r"\b(insert\s+into|update\s+\w+\s+set|delete\s+from|create\s+(table|index)|drop\s+)", re.IGNORECASE)


def _write_sql_constants(source: str) -> list[str]:
    return [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _WRITE_SQL.search(node.value)
    ]


def test_the_write_sql_scan_can_actually_see_one() -> None:
    """Positive control: the scanner finds a write in a source that has one."""
    assert _write_sql_constants('def f(c):\n    c.execute("INSERT INTO retrieval_log (a) VALUES (1)")\n')


def test_the_module_contains_no_write_sql() -> None:
    source = pathlib.Path(av.__file__).read_text(encoding="utf-8")
    assert _write_sql_constants(source) == []


def test_every_connection_is_opened_read_only() -> None:
    """Structural: ``sqlite3.connect`` on a plain path CREATES the file."""
    source = pathlib.Path(av.__file__).read_text(encoding="utf-8")
    connects = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "connect"
    ]
    assert connects, "positive control: the module must actually open a connection"
    for call in connects:
        rendered = ast.unparse(call)
        assert "mode=ro" in rendered, f"connection at line {call.lineno} is not read-only: {rendered}"


# ---------------------------------------------------------------------------
# A4 — an absent source is unavailable, never zero
# ---------------------------------------------------------------------------


def test_precision_with_no_serve_evidence_is_unavailable_not_zero(workspace: str) -> None:
    view = precision_by_intent(workspace)
    assert view.available is False
    assert "no serve evidence" in view.reason
    assert view.rows == ()


def test_credit_with_no_serve_evidence_is_counted_not_dropped(workspace: str) -> None:
    """A credit row that cannot be attributed is reported, so the gap is visible."""
    CalibrationManager(workspace).record_outcome([SERVED_A], "success", query_id="whatever")
    view = precision_by_intent(workspace)
    assert view.available is False
    assert view.credit_rows == 1
    assert view.credit_rows_on_unserved_blocks == 1


def test_run_precision_names_the_flag_when_the_ledger_is_off(workspace: str) -> None:
    verdict = run_precision(workspace)
    assert verdict.available is False
    assert "served_ledger.enabled" in verdict.reason


# ---------------------------------------------------------------------------
# The precision view
# ---------------------------------------------------------------------------


def test_precision_is_credited_over_served_per_intent(workspace: str) -> None:
    _seed_corpus(workspace)
    _seed_runs(workspace)
    CalibrationManager(workspace).record_outcome([SERVED_A], "success", query_id="run-1")

    view = precision_by_intent(workspace)
    assert view.available is True
    rows = {row.intent: row for row in view.rows}
    assert set(rows) == {"WHY", "WHEN"}
    # WHY served A and B; only A is credited.
    assert (rows["WHY"].observations, rows["WHY"].served_blocks, rows["WHY"].credited_blocks, rows["WHY"].precision) == (1, 2, 1, 0.5)
    # WHEN served only B, which carries no credit.
    assert (rows["WHEN"].served_blocks, rows["WHEN"].credited_blocks, rows["WHEN"].precision) == (1, 0, 0.0)
    assert view.credit_rows_on_unserved_blocks == 0


def test_an_implicated_block_is_counted_separately_from_a_credited_one(workspace: str) -> None:
    _seed_corpus(workspace)
    _seed_runs(workspace)
    cal = CalibrationManager(workspace)
    cal.record_outcome([SERVED_A], "success", query_id="r1")
    cal.record_outcome([SERVED_B], "failure", query_id="r2")
    rows = {row.intent: row for row in precision_by_intent(workspace).rows}
    assert rows["WHY"].credited_blocks == 1
    assert rows["WHY"].implicated_blocks == 1


def test_an_unclassified_run_lands_in_its_own_bucket(workspace: str) -> None:
    log_retrieval(workspace, "no intent recorded", [{"_id": SERVED_A, "score": 1.0}])
    rows = {row.intent: row for row in precision_by_intent(workspace).rows}
    assert set(rows) == {INTENT_UNKNOWN}


def test_both_sources_are_counted_as_observations_not_runs(workspace: str) -> None:
    """Pins the semantics the field name promises.

    The retrieval log and the ledger are two windows over one history, so
    the same answer recorded in both is two OBSERVATIONS. Only the distinct
    served blocks — and therefore the precision — are source-independent,
    which is the property that makes the number safe to act on.
    """
    query, ids = "why did the retrieval rollout land", (SERVED_A, SERVED_B)
    log_retrieval(workspace, query, [{"_id": i, "score": 1.0} for i in ids], intent_type="WHY")
    _enable_ledger(workspace)
    _append_run(workspace, query, ids)

    view = precision_by_intent(workspace)
    rows = {row.intent: row for row in view.rows}
    assert view.observations == 2
    assert rows["WHY"].observations == 2
    assert rows["WHY"].served_blocks == 2  # the answer, counted once
    assert set(view.sources) == {av.SOURCE_RETRIEVAL_LOG, av.SOURCE_SERVED_LEDGER}


def test_the_view_is_recomputed_and_never_cached(workspace: str) -> None:
    _seed_corpus(workspace)
    _seed_runs(workspace)
    first = precision_by_intent(workspace)
    CalibrationManager(workspace).record_outcome([SERVED_B], "success", query_id="later")
    second = precision_by_intent(workspace)
    assert first.rows != second.rows


# ---------------------------------------------------------------------------
# The run-level join RA.1 exists for
# ---------------------------------------------------------------------------


def test_run_id_of_attestation_is_the_ledgers_own_run_id(workspace: str) -> None:
    """The join key, minted from the three values an attestation already publishes.

    No second identity: this is the value ``served_ledger.run_id`` computes,
    reached through the attestation's field names.
    """
    _enable_ledger(workspace)
    ids = (SERVED_A, SERVED_B)
    query = "why did the retrieval rollout land"
    appended = _append_run(workspace, query, ids)

    attestation = {
        "query_hash": query_hash(query),
        "results_digest": served_set_digest(ids),
        "config_hash": PIPELINE,
    }
    assert run_id_of_attestation(attestation) == appended
    assert run_id_of_attestation(attestation) == run_id(
        query_hash=query_hash(query),
        served_digest=served_set_digest(ids),
        pipeline_hash=PIPELINE,
    )


def test_run_id_of_attestation_refuses_a_malformed_record(workspace: str) -> None:
    with pytest.raises(ValueError):
        run_id_of_attestation({"query_hash": "short", "results_digest": "a" * 64, "config_hash": PIPELINE})
    with pytest.raises(KeyError):
        run_id_of_attestation({"query_hash": "a" * 64, "config_hash": PIPELINE})


def test_run_precision_joins_a_credit_row_keyed_on_the_run_id(workspace: str) -> None:
    _enable_ledger(workspace)
    ids = (SERVED_A, SERVED_B)
    rid = _append_run(workspace, "why did the retrieval rollout land", ids)
    CalibrationManager(workspace).record_outcome([SERVED_A], "success", query_id=rid)

    verdict = run_precision(workspace)
    assert verdict.available is True
    assert (verdict.joined_runs, verdict.served, verdict.credited, verdict.precision) == (1, 2, 1, 0.5)
    assert verdict.credit_rows_with_unjoinable_query_id == 0


def test_a_credit_row_that_cannot_join_is_named_not_dropped(workspace: str) -> None:
    """Positive control: the run exists and the join works when the key is right.

    The test above proves the join fires, so ``available is False`` here is a
    statement about the key, not about a view that never joins anything.
    """
    _enable_ledger(workspace)
    _append_run(workspace, "why did the retrieval rollout land", (SERVED_A, SERVED_B))
    CalibrationManager(workspace).record_outcome([SERVED_A], "success", query_id="cal-abc-123")

    verdict = run_precision(workspace)
    assert verdict.available is False
    # RA.1's residual closed: the envelope publishes the id, so an unjoinable
    # credit row is now a caller that did not pass it, and the reason says so
    # rather than blaming a wiring gap that no longer exists.
    assert "no credit row carries a run_id matching a ledger row" in verdict.reason
    assert 'envelope["attestation"]["query_id"]' in verdict.reason
    assert verdict.runs == 1
    assert verdict.credit_rows_with_unjoinable_query_id == 1
    assert verdict.by_intent == ()


# ---------------------------------------------------------------------------
# A2 — the waste view is a read surface and goes through the shared gate
# ---------------------------------------------------------------------------


def test_the_waste_view_never_names_a_withheld_block(workspace: str) -> None:
    _seed_corpus(workspace)
    _seed_runs(workspace)

    # Positive control, in two parts: the block is on disk with the status we
    # seeded, and the store hands it to the caller. An absent id below is
    # therefore the admission gate withholding it, not an empty corpus.
    on_disk = {b["_id"]: b for b in get_block_store(workspace).get_all()}
    assert WITHHELD in on_disk
    assert on_disk[WITHHELD]["Status"] == "quarantined"

    view = waste_view(workspace)
    assert WITHHELD not in view.unserved_ids
    assert WITHHELD not in {block_id for block_id, _ in view.protected_unserved}
    assert view.corpus_withheld == 1
    assert view.corpus_admitted == 4


def test_the_waste_view_counts_serve_evidence_from_the_real_recall_log(workspace: str) -> None:
    _seed_corpus(workspace)
    _seed_runs(workspace)
    view = waste_view(workspace)
    assert view.served_at_least_once == 2
    assert view.unserved == 2  # the plain unserved decision + the release decision
    assert view.sources == (av.SOURCE_RETRIEVAL_LOG,)
    assert "30 days" in view.window


def test_the_ledger_widens_the_window_when_it_is_enabled(workspace: str) -> None:
    _seed_corpus(workspace)
    _seed_runs(workspace)
    _enable_ledger(workspace)
    _append_run(workspace, "a query whose log row has aged out", (UNSERVED,))
    view = waste_view(workspace)
    assert set(view.sources) == {av.SOURCE_RETRIEVAL_LOG, av.SOURCE_SERVED_LEDGER}
    assert UNSERVED not in view.unserved_ids
    assert "not pruned" in view.window


# ---------------------------------------------------------------------------
# A3 — unserved is a question, not a verdict
# ---------------------------------------------------------------------------


def test_a_protected_block_is_never_called_waste(workspace: str) -> None:
    _seed_corpus(workspace)
    _seed_runs(workspace)
    view = waste_view(workspace)

    protected = dict(view.protected_unserved)
    assert RELEASE_DECISION in protected
    assert "release decision" in protected[RELEASE_DECISION]
    assert RELEASE_DECISION not in view.unserved_ids
    assert view.unserved_ids == (UNSERVED,)
    assert view.unserved_by_retention_class[PROTECTED] == 1
    assert view.unserved_by_retention_class[GOVERNED] == 1


def test_the_report_carries_both_views_and_its_schema(workspace: str) -> None:
    _seed_corpus(workspace)
    _seed_runs(workspace)
    report = accountability_report(workspace)
    assert report["schema"] == "MM_ACCOUNTABILITY_v1"
    assert report["precision_by_intent"]["available"] is True
    assert report["waste"]["corpus_withheld"] == 1
    json.dumps(report)  # the report is serialisable, which is what makes it a surface


# ---------------------------------------------------------------------------
# The entry point — `python -m mind_mem.accountability_views`
# ---------------------------------------------------------------------------


def test_the_module_entry_point_prints_a_report(workspace: str) -> None:
    _seed_corpus(workspace)
    _seed_runs(workspace)
    env = dict(os.environ, PYTHONPATH=str(pathlib.Path(av.__file__).parents[2]))
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "mind_mem.accountability_views", "--workspace", workspace, "--indent", "0"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["schema"] == "MM_ACCOUNTABILITY_v1"
    assert report["waste"]["unserved_ids"] == [UNSERVED]


def test_the_entry_point_creates_nothing(workspace: str) -> None:
    assert av.main(["--workspace", workspace, "--indent", "0"]) == 0
    assert sorted(os.listdir(workspace)) == []


# ---------------------------------------------------------------------------
# Wiring — "imported" is not "wired", so the call path is a test
# ---------------------------------------------------------------------------


def test_the_mm_accountability_verb_reaches_the_report(workspace: str, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """Trace the path a user actually takes: ``mm accountability`` -> the views.

    Parsed through ``mm_cli.build_parser`` rather than by calling the handler
    directly, so the assertion covers the registration too: a handler nothing
    dispatches to is exactly the "registered but unreachable" shape this repo
    keeps paying for.
    """
    from mind_mem import mm_cli

    _seed_corpus(workspace)
    _seed_runs(workspace)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)

    args = mm_cli.build_parser().parse_args(["accountability"])
    assert args.func is mm_cli._cmd_accountability
    assert args.func(args) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "MM_ACCOUNTABILITY_v1"
    assert report["waste"]["unserved_ids"] == [UNSERVED]
    assert report["precision_by_intent"]["available"] is True


def test_the_verb_writes_nothing_to_the_workspace(workspace: str, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """The verb is safe to run against a live workspace — that is the whole point."""
    from mind_mem import mm_cli

    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    args = mm_cli.build_parser().parse_args(["accountability"])
    assert args.func(args) == 0
    capsys.readouterr()
    assert sorted(os.listdir(workspace)) == []


# ---------------------------------------------------------------------------
# One file, two names — a ratchet on the store the views read
# ---------------------------------------------------------------------------


def test_the_calibration_store_and_the_retrieval_log_are_one_database(workspace: str) -> None:
    """Both credit tables and ``retrieval_log`` live in the same ``recall.db``.

    The views open it once, through ``graph_db_path``. If either owner ever
    moves its file, this fails here rather than silently reporting a zero.
    """
    from mind_mem.calibration import _db_path

    assert os.path.abspath(graph_db_path(workspace)) == os.path.abspath(_db_path(workspace))


# ---------------------------------------------------------------------------
# Mutation twins — a gate that cannot fail is not a gate
# ---------------------------------------------------------------------------


class TestMutationTwin:
    def test_neutering_admit_corpus_leaks_the_withheld_block(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        _seed_corpus(workspace)
        _seed_runs(workspace)
        monkeypatch.setattr(av, "admit_corpus", lambda blocks, **kwargs: [dict(b) for b in blocks])
        view = waste_view(workspace)
        with pytest.raises(AssertionError):
            assert WITHHELD not in view.unserved_ids

    def test_flattening_the_retention_class_calls_a_protected_block_waste(
        self,
        workspace: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed_corpus(workspace)
        _seed_runs(workspace)
        monkeypatch.setattr(av, "retention_class", lambda block, **kwargs: GOVERNED)
        view = waste_view(workspace)
        with pytest.raises(AssertionError):
            assert RELEASE_DECISION not in view.unserved_ids
