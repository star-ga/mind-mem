"""RA.1's residual, closed: the recall envelope publishes the run identity.

The right-hand side of the accountability join has accepted a ``query_id``
since ``outcome_attribution.report_outcome`` was written. The left-hand side
never published one, so a client holding a recall envelope had nothing
joinable to pass and ``accountability_views.run_precision`` reported
``available=False`` on every workspace in the world. That is the whole of the
gap, and it was one field.

What this file proves, in the order the risk sits:

R1  **The encoding moved without changing.** ``run_id`` now lives in
    ``recall_digests``, the leaf both the ledger and the attestation may
    depend on, because the attestation may not reach a ledger module on any
    import path. A moved hash that changed value would silently orphan every
    ledger row already on disk, so the digest is pinned against a literal.

R2  **The rail survived the move.** Importing the attestation must still pull
    in no ledger module, checked in a fresh interpreter (a static walk cannot
    see ``importlib``) and guarded against vacuity.

R3  **The published id is derived, not asserted.** It is a pure function of
    three hash-bound fields, absent rather than guessed when one of them is
    unmintable, and refused on parse when it disagrees with them.

R4  **It is wired end to end.** Not "the attribute exists": a recall driven
    through the real MCP handler produces an envelope whose ``query_id``
    equals the ledger row's ``run_id``, and reporting an outcome against that
    value moves ``run_precision`` from unavailable to a computed number.

R5  **The gate can fail.** Each protective assertion has a mutation twin that
    breaks what it guards and requires the assertion to go red.

Also here: RA.2's per-intent-type run-level precision, and RA.1's
``block_serve_counts`` residual — serve counts that outlive the 30-day
``retrieval_log`` prune because they are derived from the append-only ledger
rather than stored in the pruned table.
"""

from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import subprocess
import sys
from typing import Any

import pytest

import mind_mem
from mind_mem import accountability_views as av
from mind_mem.accountability_views import (
    accountability_report,
    run_id_of_attestation,
    run_precision,
    serve_counts,
)
from mind_mem.outcome_attribution import report_outcome
from mind_mem.recall_attestation import (
    RECALL_ATTEST_TAG,
    RecallAttestation,
    build_recall_attestation,
    verify_recall_attestation,
)
from mind_mem.recall_digests import query_hash, run_id, served_set_digest
from mind_mem.retrieval_graph import graph_db_path, log_retrieval
from mind_mem.served_ledger import append_served_run, read_served_runs

PIPELINE = "b" * 64
ANCHOR = "c" * 64
INSTANT = "2026-09-01"

#: ``run_id`` over three fixed digests, computed by the pre-move implementation
#: in ``served_ledger`` and pasted here as a literal. A test that recomputed the
#: expected value with the same function it is testing would agree with any
#: change, including a wrong one; this one disagrees with all of them.
RUN_ID_OF_ABC = "071c9e0fd797aebda989925f1dd619ecea827d9b4ceab5608f0bedb64dc78a07"


def _attestation(*, config_hash: str = PIPELINE, served: tuple[str, ...] = ("D-1", "D-2")) -> RecallAttestation:
    return build_recall_attestation(
        legs_ran=("bm25",),
        legs_degraded=(),
        config_hash=config_hash,
        degraded=None,
        index_anchor=ANCHOR,
        result_count=len(served),
        served_ids=served,
        query="why did the retrieval rollout land",
        scoring_instant=INSTANT,
    )


# ---------------------------------------------------------------------------
# R1 — the encoding moved to the leaf and did not change value
# ---------------------------------------------------------------------------


def test_the_run_identity_has_exactly_one_owner() -> None:
    """``served_ledger.run_id`` IS ``recall_digests.run_id`` — not a copy of it.

    Identity, not equality of output. Two functions that agree today are two
    functions that can disagree tomorrow, and "one object, one encoding, one
    owner" is the rule this module split exists to enforce.
    """
    import mind_mem.recall_digests as digests
    import mind_mem.served_ledger as ledger

    assert ledger.run_id is digests.run_id
    assert ledger.RUN_TAG == digests.RUN_TAG == "MM_RUN_v1"
    assert "run_id" in ledger.__all__, "the historical import path must keep working"


def test_the_moved_digest_is_byte_identical_to_the_pre_move_value() -> None:
    """A ledger row written before the move must still be joinable after it.

    ``run_id`` is on disk in every shipped ledger. If the move had changed the
    tag, the separator or the concatenation order, existing rows would stop
    matching any id a client could compute — a silent, unrecoverable orphaning
    of the exact evidence RA.1 exists to keep.
    """
    assert run_id(query_hash="a" * 64, served_digest="b" * 64, pipeline_hash="c" * 64) == RUN_ID_OF_ABC


def test_the_width_contract_is_enforced_on_every_input() -> None:
    """Fixed width is what makes an unseparated concatenation unambiguous."""
    for bad in ("", "z" * 64, "a" * 63, "A" * 64):
        with pytest.raises(ValueError):
            run_id(query_hash=bad, served_digest="b" * 64, pipeline_hash="c" * 64)
        with pytest.raises(ValueError):
            run_id(query_hash="a" * 64, served_digest=bad, pipeline_hash="c" * 64)
        with pytest.raises(ValueError):
            run_id(query_hash="a" * 64, served_digest="b" * 64, pipeline_hash=bad)


def test_the_leaf_is_still_a_leaf() -> None:
    """``recall_digests`` imports nothing first-party — that is why it can hold this.

    The move is only safe because the destination has no first-party imports.
    A leaf that acquired one would put whatever it acquired on the import path
    of everything that depends on it, including the attestation.
    """
    import ast

    source = (pathlib.Path(mind_mem.__file__).parent / "recall_digests.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"relative import {ast.dump(node)} in the leaf"
            assert not (node.module or "").startswith("mind_mem"), node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("mind_mem"), alias.name


# ---------------------------------------------------------------------------
# R2 — the rail the move could have broken
# ---------------------------------------------------------------------------


_RAIL_CHILD = """
import json, sys
import mind_mem.recall_attestation  # noqa: F401
print(json.dumps(sorted(m for m in sys.modules if m.startswith("mind_mem."))))
"""


def _loaded_by(child: str) -> set[str]:
    env = {**os.environ, "PYTHONPATH": str(pathlib.Path(mind_mem.__file__).parent.parent)}
    out = subprocess.run(
        [sys.executable, "-c", child],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=True,
    ).stdout
    return set(json.loads(out))


def test_publishing_the_id_did_not_put_a_ledger_on_the_attestation_path() -> None:
    """The reason the encoding moved, checked at runtime in a clean process.

    The obvious implementation of this feature — ``from .served_ledger import
    run_id`` inside ``recall_attestation`` — is precisely the import edge the
    attestation's rail forbids, and a static scan of the *current* source
    would not notice a lazy one. Loading the module for real does.
    """
    loaded = _loaded_by(_RAIL_CHILD)
    assert "mind_mem.recall_attestation" in loaded, "child did not import the module — the check is vacuous"
    assert "mind_mem.recall_digests" in loaded, "the id encoding was not reached at all"
    for ledger in ("served_ledger", "ledger_anchor", "usage_meter"):
        assert f"mind_mem.{ledger}" not in loaded, f"{ledger} is now on the attestation's import path"


def test_the_rail_check_can_see_a_breach() -> None:
    """Positive control: the same probe DOES report a ledger when one is loaded."""
    loaded = _loaded_by("import json, sys\nimport mind_mem.served_ledger\n" + _RAIL_CHILD)
    assert "mind_mem.served_ledger" in loaded


# ---------------------------------------------------------------------------
# R3 — derived, absent when unmintable, refused when edited
# ---------------------------------------------------------------------------


def test_the_envelope_attestation_publishes_the_run_identity() -> None:
    att = _attestation()
    published = att.to_dict()
    assert published["query_id"] == run_id(
        query_hash=published["query_hash"],
        served_digest=published["results_digest"],
        pipeline_hash=published["config_hash"],
    )
    assert published["query_id"] == run_id_of_attestation(published)
    assert len(published["query_id"]) == 64


def test_the_identity_is_a_property_not_a_stored_field() -> None:
    """Nothing to forge: it is recomputed, so there is no value to move.

    This is what lets it sit on a record whose stated invariant is that every
    *field* is bound into the hash. A dataclass field would have been an
    unbound sibling; a property is the derivation itself.
    """
    import dataclasses

    assert "query_id" not in {f.name for f in dataclasses.fields(RecallAttestation)}
    assert isinstance(type(RecallAttestation.query_id), type(property))


def test_two_different_answers_to_one_question_get_different_ids() -> None:
    """The id names the ANSWER, which is the property the join depends on."""
    a = _attestation(served=("D-1", "D-2")).to_dict()["query_id"]
    b = _attestation(served=("D-2", "D-1")).to_dict()["query_id"]
    c = _attestation(served=("D-1", "D-3")).to_dict()["query_id"]
    assert len({a, b, c}) == 3


def test_an_unmintable_id_is_absent_and_never_invented() -> None:
    """``config_hash=""`` is a live degraded path, not a hypothetical.

    ``derive_recall_attestation_for_workspace`` binds an empty config hash
    when the pipeline probe fails, because an attestation with an unresolved
    hash is honest and a crashed recall is not. An id cannot be minted from
    it, so the envelope says nothing rather than something wrong.
    """
    degraded = _attestation(config_hash="").to_dict()
    assert degraded["query_id"] == ""
    assert verify_recall_attestation(degraded) is True, "an absent id must not make the record invalid"


def test_an_edited_run_identity_is_refused_on_parse() -> None:
    good = _attestation().to_dict()
    assert verify_recall_attestation(good) is True  # positive control

    forged = dict(good)
    forged["query_id"] = "f" * 64
    assert verify_recall_attestation(forged) is False
    with pytest.raises(ValueError, match="edited in transit"):
        RecallAttestation.from_dict(forged)


def test_an_envelope_without_the_key_still_parses() -> None:
    """Back-compatible: a pre-RA.1-residual envelope claims nothing to disagree with."""
    older = {k: v for k, v in _attestation().to_dict().items() if k != "query_id"}
    assert older["schema"] == RECALL_ATTEST_TAG
    assert verify_recall_attestation(older) is True
    assert RecallAttestation.from_dict(older).query_id == _attestation().query_id


def test_the_forgery_check_is_not_vacuous() -> None:
    """Mutation twin: with the guard removed, the forged record verifies.

    A refusal that would hold anyway proves nothing about the refusal. This
    replaces ``from_dict`` with a version that skips the comparison and shows
    the forged dict sailing through — so the assertion above is load-bearing.
    """
    forged = dict(_attestation().to_dict())
    forged["query_id"] = "f" * 64
    original = RecallAttestation.from_dict

    def _unguarded(d: dict[str, Any]) -> RecallAttestation:
        return original({k: v for k, v in d.items() if k != "query_id"})

    RecallAttestation.from_dict = classmethod(lambda cls, d: _unguarded(d))  # type: ignore[assignment,method-assign]
    try:
        assert verify_recall_attestation(forged) is True, "the twin did not disable the guard"
    finally:
        RecallAttestation.from_dict = original  # type: ignore[method-assign]
    assert verify_recall_attestation(forged) is False, "the guard did not come back"


# ---------------------------------------------------------------------------
# R4 — wired: a real recall, a real outcome report, a real join
# ---------------------------------------------------------------------------


def _live_workspace(tmp_path: pathlib.Path, name: str) -> str:
    ws = tmp_path / name
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "decisions" / "DECISIONS.md").write_text(
        "[D-20260901-001]\nStatement: the latency decision landed\nDate: 2026-09-01\nStatus: active\n\n---\n\n"
        "[D-20260901-002]\nStatement: the latency decision was reviewed\nDate: 2026-09-01\nStatus: active\n\n---\n\n",
        encoding="utf-8",
    )
    (ws / "mind-mem.json").write_text(
        json.dumps(
            {
                "recall": {"vector_enabled": False, "provider": "local"},
                "cache": {"enabled": False},
                "served_ledger": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )
    return str(ws)


def _mcp_recall(monkeypatch: pytest.MonkeyPatch, ws: str) -> Any:
    """The real MCP recall handler, pointed at *ws*. Not a stand-in for it."""
    import mind_mem.mcp.tools.recall as mcp_recall

    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)
    return mcp_recall


def _live_run(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, name: str) -> tuple[str, dict[str, Any]]:
    ws = _live_workspace(tmp_path, name)
    handler = _mcp_recall(monkeypatch, ws)
    envelope = json.loads(handler._recall_impl("latency decision", limit=5, scoring_instant="2026-09-01"))
    return ws, envelope


def test_a_live_recall_publishes_an_id_that_names_its_own_ledger_row(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The two sides of the join agree, traced from the real entry point.

    ``_recall_impl`` is the choke point ``recall`` and ``hybrid_search`` both
    delegate to. The envelope's ``query_id`` and the ledger row's ``run_id``
    are computed on opposite sides of a rail that forbids either module from
    importing the other; if they ever disagree the join is unmakeable and the
    ledger records evidence nobody can use.
    """
    ws, envelope = _live_run(tmp_path, monkeypatch, "wired")

    rows = read_served_runs(ws)
    assert len(rows) == 1, "positive control: there must be a ledger row to join to"
    assert envelope["attestation"]["query_id"], "the envelope published no run identity"
    assert envelope["attestation"]["query_id"] == rows[0].run_id
    assert envelope["results"], "positive control: the run must have served something"


def test_reporting_an_outcome_against_the_published_id_makes_the_join(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RA.1 residual → RA.2 run precision, end to end, with two controls.

    Three states, in order, so the final number is a measurement and not a
    coincidence: nothing reported (unavailable, nothing unjoinable), an
    outcome reported under a *wrong* id (still unavailable, one unjoinable
    row — the join discriminates), then an outcome reported under the id the
    envelope published (available, with the arithmetic).

    The two reports name different blocks deliberately.
    ``outcome_attribution.canonical_outcome_id`` does NOT hash ``query_id``,
    so the same verdict on the same block under two different run ids is one
    payload and the second report is deduplicated away. That is a real
    property of the store, recorded here rather than tripped over.
    """
    ws, envelope = _live_run(tmp_path, monkeypatch, "join")
    query_id = envelope["attestation"]["query_id"]
    served = [hit["_id"] for hit in envelope["results"]]
    assert len(served) >= 2, "positive control: the run must serve enough blocks for precision to be a fraction"

    before = run_precision(ws)
    assert before.available is False
    assert before.credit_rows_with_unjoinable_query_id == 0

    report_outcome(ws, [served[1]], "success", query_id="cal-not-a-run-id")
    wrong_key = run_precision(ws)
    assert wrong_key.available is False, "an outcome under a foreign id must not join"
    assert wrong_key.credit_rows_with_unjoinable_query_id == 1
    assert 'envelope["attestation"]["query_id"]' in wrong_key.reason

    report_outcome(ws, [served[0]], "success", query_id=query_id)
    after = run_precision(ws)
    assert after.available is True, after.reason
    assert after.joined_runs == 1
    assert after.served == len(served)
    assert after.credited == 1
    assert after.precision == round(1 / len(served), 6)
    assert after.credit_rows_with_unjoinable_query_id == 1, "an unjoinable row stays named after the join fires"


def test_the_join_test_can_fail(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation twin for the wiring: drop the key, watch the join go red.

    The published field is the entire deliverable, so the test that proves it
    must be shown to depend on it. With ``to_dict`` stripped of ``query_id``
    the envelope carries nothing joinable and the assertion that the join
    fires raises — which is what makes its green a fact.
    """
    original = RecallAttestation.to_dict

    def _without_query_id(self: RecallAttestation) -> dict[str, Any]:
        return {k: v for k, v in original(self).items() if k != "query_id"}

    monkeypatch.setattr(RecallAttestation, "to_dict", _without_query_id)
    ws, envelope = _live_run(tmp_path, monkeypatch, "mutant")

    assert envelope["attestation"].get("query_id", "") == "", "the twin did not remove the field"
    assert read_served_runs(ws), "positive control: the ledger row is still written"
    with pytest.raises(KeyError):
        _ = envelope["attestation"]["query_id"]


# ---------------------------------------------------------------------------
# RA.2 — precision per intent type, at run level
# ---------------------------------------------------------------------------


def _seeded_join(ws: str) -> None:
    """Two logged recalls under two intents, each joined by an outcome report.

    The WHEN run reports a *failure*, not nothing. A run with no outcome at
    all does not join, so it would be absent from the breakdown rather than
    present with a precision of zero — and "this intent converts badly" is
    exactly the row RA.2 is for. A failure verdict joins and credits nothing.
    """
    for question, intent, ids, verdict, subject in (
        ("why did the rollout land", "WHY", ("D-1", "D-2"), "success", "D-1"),
        ("when was the rollout reviewed", "WHEN", ("D-2", "D-3", "D-4"), "failure", "D-3"),
    ):
        log_retrieval(ws, question, [{"_id": bid, "score": 1.0} for bid in ids], intent_type=intent)
        row = append_served_run(
            ws,
            query_hash=query_hash(question),
            served_digest=served_set_digest(ids),
            ids=ids,
            pipeline_hash=PIPELINE,
            index_anchor=ANCHOR,
            scoring_instant=INSTANT,
        )
        assert row is not None, "positive control: the ledger must be enabled"
        report_outcome(ws, [subject], verdict, query_id=row.run_id)


@pytest.fixture
def joined_workspace(tmp_path: pathlib.Path) -> str:
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as handle:
        json.dump({"served_ledger": {"enabled": True}}, handle)
    _seeded_join(ws)
    return ws


def test_run_precision_breaks_down_by_intent_type(joined_workspace: str) -> None:
    """The item's own wording — precision per intent type — at run level."""
    verdict = run_precision(joined_workspace)
    assert verdict.available is True, verdict.reason
    rows = {row.intent: row for row in verdict.by_intent}
    assert set(rows) == {"WHY", "WHEN"}
    assert (rows["WHY"].served, rows["WHY"].credited, rows["WHY"].precision) == (2, 1, 0.5)
    assert (rows["WHEN"].served, rows["WHEN"].credited, rows["WHEN"].precision) == (3, 0, 0.0)


def test_the_intent_rows_sum_to_the_totals(joined_workspace: str) -> None:
    """A breakdown that does not reconcile is a second measurement, not a view."""
    verdict = run_precision(joined_workspace)
    assert sum(row.served for row in verdict.by_intent) == verdict.served
    assert sum(row.credited for row in verdict.by_intent) == verdict.credited
    assert sum(row.runs for row in verdict.by_intent) == verdict.joined_runs


def test_a_run_whose_intent_aged_out_is_named_unknown_not_dropped(joined_workspace: str) -> None:
    """The ledger outlives the log, so the intent can be gone while the run is not.

    Deleting every ``retrieval_log`` row is not an approximation of the 30-day
    prune — the prune IS a ``DELETE FROM retrieval_log``. The intent is
    unrecoverable afterwards; the run, its served set and its credit are not,
    and dropping the row would understate served work.
    """
    before = run_precision(joined_workspace)
    assert {row.intent for row in before.by_intent} == {"WHY", "WHEN"}  # positive control

    conn = sqlite3.connect(graph_db_path(joined_workspace))
    try:
        conn.execute("DELETE FROM retrieval_log")
        conn.commit()
    finally:
        conn.close()

    after = run_precision(joined_workspace)
    assert after.available is True, after.reason
    assert {row.intent for row in after.by_intent} == {av.INTENT_UNKNOWN}
    assert after.served == before.served
    assert after.credited == before.credited


# ---------------------------------------------------------------------------
# RA.1 residual — serve counts that outlive the window
# ---------------------------------------------------------------------------


def test_serve_counts_split_durable_evidence_from_windowed_evidence(joined_workspace: str) -> None:
    counts = serve_counts(joined_workspace)
    assert counts.available is True
    by_id = {row.block_id: row for row in counts.top}
    assert set(by_id) == {"D-1", "D-2", "D-3", "D-4"}
    # D-2 was served by both runs, under both sources.
    assert (by_id["D-2"].durable, by_id["D-2"].windowed) == (2, 2)
    assert (by_id["D-1"].durable, by_id["D-1"].windowed) == (1, 1)
    assert counts.durable_serves == 5
    assert counts.windowed_serves == 5
    assert counts.sources == (av.SOURCE_RETRIEVAL_LOG, av.SOURCE_SERVED_LEDGER)


def test_the_counts_survive_the_thirty_day_prune(joined_workspace: str) -> None:
    """The residual, stated as a before/after over the actual prune statement.

    ``retrieval_graph.log_retrieval`` runs ``DELETE FROM retrieval_log WHERE
    timestamp < datetime('now','-30 days')`` every hundredth call. Executing
    that delete is therefore the prune itself, not a stand-in for it. What has
    to hold afterwards is that the durable half is untouched — that is the
    difference between a counter in a pruned table and a count derived from an
    append-only one.
    """
    before = serve_counts(joined_workspace)
    assert before.windowed_serves == 5  # positive control: there IS something to lose

    conn = sqlite3.connect(graph_db_path(joined_workspace))
    try:
        conn.execute("DELETE FROM retrieval_log WHERE timestamp < datetime('now', '+1 day')")
        conn.commit()
    finally:
        conn.close()

    after = serve_counts(joined_workspace)
    assert after.windowed_serves == 0, "the prune did not fire — the test proves nothing"
    assert after.durable_serves == before.durable_serves == 5
    assert after.durable_blocks == before.durable_blocks == 4
    assert after.sources == (av.SOURCE_SERVED_LEDGER,)
    assert av.SERVED_LEDGER_WINDOW in after.window


def test_the_durability_claim_depends_on_the_source_split(joined_workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation twin: classify ledger rows as windowed and the claim collapses.

    "Survives the prune" is a claim about which bucket the ledger lands in. If
    the split were wrong the previous test would still be green on a workspace
    where the ledger happened to be enabled, so break the split and require the
    durable count to go to zero.

    The twin mislabels the ledger reader's own output rather than renaming the
    constant. Renaming the constant moves *both* sides of the comparison at
    once — the reader stamps it and the view tests for it — so the mutation
    would cancel out and the twin would pass while proving nothing.
    """
    real = av._observed_from_ledger

    def _mislabelled(workspace: str, intents: dict[str, str]) -> tuple[av.ObservedRun, ...]:
        return tuple(
            av.ObservedRun(run.query_hash, run.intent, run.ids, av.SOURCE_RETRIEVAL_LOG, run.run_id) for run in real(workspace, intents)
        )

    monkeypatch.setattr(av, "_observed_from_ledger", _mislabelled)
    counts = serve_counts(joined_workspace)
    assert counts.durable_serves == 0, "the twin did not change the classification"
    assert counts.windowed_serves == 10


def test_an_empty_workspace_reports_unavailable_not_zero(tmp_path: pathlib.Path) -> None:
    """A count of zero over a table that does not exist is a measurement nobody made."""
    ws = str(tmp_path / "bare")
    os.makedirs(ws)
    counts = serve_counts(ws)
    assert counts.available is False
    assert counts.reason
    assert (counts.blocks, counts.durable_serves, counts.windowed_serves) == (0, 0, 0)
    assert sorted(os.listdir(ws)) == [], "a read-only view created a database"


def test_the_report_carries_the_new_views(joined_workspace: str) -> None:
    """The surface: ``mm accountability`` prints this dict verbatim."""
    report = accountability_report(joined_workspace)
    assert report["schema"] == "MM_ACCOUNTABILITY_v1"
    assert report["serve_counts"]["available"] is True
    assert report["run_precision"]["available"] is True
    assert [row["intent"] for row in report["run_precision"]["by_intent"]] == ["WHEN", "WHY"]
    assert json.dumps(report, sort_keys=True), "the report must be JSON-serialisable"
