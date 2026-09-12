"""Edge extraction on the write path — the block's OWN apply stages its edges.

ROADMAP ("Auto-extract edges on the write path (HITL-gated)", called out as "the
single highest-leverage item"): "wire lightweight entity/relation extraction ... so
writing a block *proposes* typed KG edges. **Extracted edges land as proposals, never
auto-committed** — same approval gate as blocks, honoring the Group H wedge guardrail
(source-of-truth graph never self-modifies)."

Two design decisions worth stating, because both were the alternative to something
worse:

WHY THE APPLY STEP, NOT `propose_update`. The roadmap names propose_update, and that
is where a proposal is born — but a proposal has no block id yet, and
`RelationTriple.__post_init__` REQUIRES `source_block_id` "for provenance". Staging at
propose time would have meant inventing an id for a block that may never exist: if the
block proposal is then rejected, the edge proposal survives citing a block that was
never written, and an operator can approve it. The apply step is where the id becomes
real, so that is where the edges are staged.

WHY `graph_ingest.backfill` AND NOT NEW STAGING CODE. backfill already accepts an
injected `extract_fn` and a `restrict_to_blocks` set, already stages to SIGNALS.md,
and its output is already consumed by `approve_relation_signals` — the operator gate
that commits an edge with a real block id inside an `admit_edge` scope. Restricting it
to one id is the whole write-path feature. A second staging path would have been a
second thing to keep in step with that gate, and this repo has already paid for two
hand-maintained lists of one concept.

The extractor is `edge_extraction.candidate_edges` — deterministic, no model call — so
the write path never blocks on inference.
"""

from __future__ import annotations

import mind_mem.write_path_edges as wpe
from mind_mem.write_path_edges import (
    EDGE_FLAG,
    StageOutcome,
    extract_fn_for_block,
    stage_edges_for_block,
)

# `candidate_edges` is BLOCK-CENTRIC (the block id is the subject) and its predicate
# vocabulary is the CLOSED governance set -- supersedes / refines / depends_on /
# part_of / authored_by -- not social relations. An earlier fixture here said "works
# with", which no predicate covers, and the adapter correctly produced nothing; the
# fixture was wrong, not the extractor.
BLOCK_ID = "DEC-7"
STATEMENT = "This decision supersedes Retry Policy V1 and depends on Circuit Breaker."


def test_the_flag_is_declared_in_the_authoritative_registry():
    """is_enabled_quiet returns False for any flag not in ALL_V4_FLAGS, so an
    undeclared flag is not "off by default" — it is permanently unreachable, and the
    feature would look shipped while never running."""
    from mind_mem.v4.feature_flags import ALL_V4_FLAGS

    assert EDGE_FLAG.split(".", 1)[-1] in ALL_V4_FLAGS


def test_it_is_off_by_default_and_says_DISABLED_not_nothing(tmp_path, monkeypatch):
    """The load-bearing distinction. "off" and "ran and found no edges" must not
    produce the same answer: a caller that cannot tell them apart reads silence as
    "this statement has no relations" when the truth is "nobody looked"."""
    monkeypatch.setattr(wpe, "_flag_on", lambda: False)
    got = stage_edges_for_block(str(tmp_path), "DEC-1")
    assert got.outcome is StageOutcome.DISABLED, got
    assert got.signals_written == 0


def test_a_statement_with_no_relations_says_NO_CANDIDATES_not_disabled(tmp_path, monkeypatch):
    """POSITIVE CONTROL for the test above — the two must be distinguishable in both
    directions, or the enum is decoration."""
    monkeypatch.setattr(wpe, "_flag_on", lambda: True)
    monkeypatch.setattr(wpe, "_backfill",
                        lambda ws, **kw: {"blocks_scanned": 1, "signals_written": 0})
    got = stage_edges_for_block(str(tmp_path), "DEC-1")
    assert got.outcome is StageOutcome.NO_CANDIDATES, got
    assert got.signals_written == 0


def test_scanning_zero_blocks_is_NOT_reported_as_no_edges(tmp_path, monkeypatch):
    """A block absent from the loaded corpus — wrong workspace, an id the loader does
    not index, a store that has not caught up — yields the same zero as a statement
    with no relations, and only one of those is a healthy answer. Reporting the first
    as the second is how a broken wiring passes for a quiet one."""
    monkeypatch.setattr(wpe, "_flag_on", lambda: True)
    monkeypatch.setattr(wpe, "_backfill",
                        lambda ws, **kw: {"blocks_scanned": 0, "signals_written": 0})
    got = stage_edges_for_block(str(tmp_path), "DEC-404")
    assert got.outcome is StageOutcome.BLOCK_NOT_FOUND, got
    assert "DEC-404" in got.detail


def test_nothing_happens_when_the_block_id_is_missing(tmp_path, monkeypatch):
    """An edge needs a real source block. An empty id is refused rather than staged
    against an empty string, which would read downstream as a real provenance value."""
    monkeypatch.setattr(wpe, "_flag_on", lambda: True)
    got = stage_edges_for_block(str(tmp_path), "")
    assert got.outcome is StageOutcome.NO_BLOCK_ID, got


def test_it_stages_through_backfill_restricted_to_exactly_one_block(tmp_path, monkeypatch):
    """The wiring claim, checked by capturing the call rather than trusting it.

    `restrict_to_blocks` must carry EXACTLY this block: None would re-scan the whole
    corpus on every apply, and an empty set would scan nothing while reporting
    success. `dry_run` must be False or nothing is staged at all.
    """
    seen: dict[str, object] = {}

    def _fake_backfill(workspace, **kwargs):
        seen.update(kwargs)
        seen["workspace"] = workspace
        return {"signals_written": 2, "edges_extracted": 2, "blocks_scanned": 1}

    monkeypatch.setattr(wpe, "_flag_on", lambda: True)
    monkeypatch.setattr(wpe, "_backfill", _fake_backfill)
    got = stage_edges_for_block(str(tmp_path), "DEC-7")

    assert got.outcome is StageOutcome.STAGED, got
    assert got.signals_written == 2
    assert seen["restrict_to_blocks"] == ["DEC-7"], seen
    assert seen["dry_run"] is False, seen
    assert seen["workspace"] == str(tmp_path)


def test_the_injected_extractor_makes_no_model_call(tmp_path, monkeypatch):
    """The write path must not block on inference. backfill's DEFAULT extract_fn is
    the configured extraction MODEL, so omitting the injection would put a network
    call on every apply — this asserts one is passed."""
    seen: dict[str, object] = {}
    monkeypatch.setattr(wpe, "_flag_on", lambda: True)
    monkeypatch.setattr(wpe, "_backfill",
                        lambda workspace, **kw: seen.update(kw) or {"signals_written": 1, "blocks_scanned": 1})
    stage_edges_for_block(str(tmp_path), "DEC-7")
    assert callable(seen.get("extract_fn")), seen


def test_the_adapter_supplies_the_block_id_as_the_subject():
    """A REAL interface mismatch, and the reason this test exists.

    `candidate_edges` is block-centric: the block's own `_id` is the subject of every
    candidate. backfill's contract hands an extractor only `text`. Passing
    `{"Statement": text}` with no `_id` makes candidate_edges return [] for EVERY
    input — staging nothing while every metric reads clean. The adapter closes over
    the id, which is sound only because staging is one block at a time.
    """
    rows = extract_fn_for_block(BLOCK_ID)(STATEMENT)
    assert rows, "the adapter produced no triples for a statement that has one"
    assert all(r["subject"] == BLOCK_ID for r in rows), rows


def test_the_adapter_emits_the_shape_backfill_requires():
    """A shape mismatch would be silently dropped and counted as
    edges_dropped_invalid — staging nothing while reporting a clean run."""
    rows = extract_fn_for_block(BLOCK_ID)(STATEMENT)
    assert rows, "the adapter produced no triples for a statement that has one"
    for row in rows:
        assert set(row) >= {"subject", "predicate", "object"}, row
        assert row["subject"] and row["predicate"] and row["object"]


def test_the_adapter_predicates_are_all_accepted_by_RelationTriple():
    """Every predicate must pass `Predicate.from_str`, or the triple is dropped at
    the boundary and the feature stages nothing while looking like it worked."""
    from mind_mem.knowledge_graph import Predicate

    for row in extract_fn_for_block(BLOCK_ID)(STATEMENT):
        Predicate.from_str(row["predicate"])  # raises on an unknown predicate


def test_a_failing_extraction_never_breaks_an_apply_but_is_NOT_silent(tmp_path, monkeypatch, caplog):
    """The block is already committed when this runs, so raising would report failure
    for a completed apply. But a swallowed error makes "writing a block proposes
    edges" unfalsifiable, so the outcome is ERROR, the reason is carried, and it
    logs. Visible, not silent."""
    def _boom(workspace, **kwargs):
        raise RuntimeError("index is locked")

    monkeypatch.setattr(wpe, "_flag_on", lambda: True)
    monkeypatch.setattr(wpe, "_backfill", _boom)
    got = stage_edges_for_block(str(tmp_path), "DEC-7")
    assert got.outcome is StageOutcome.ERROR, got
    assert "index is locked" in got.detail


def test_it_never_writes_the_graph_directly(tmp_path, monkeypatch):
    """The Group H wedge guardrail: the source-of-truth graph never self-modifies.
    The only writer reached is backfill's staging path; add_edge must not be called."""
    import mind_mem.knowledge_graph as kg

    def _forbidden(*args, **kwargs):  # pragma: no cover — must never run
        raise AssertionError("the write path called add_edge — edges must only stage")

    monkeypatch.setattr(kg.KnowledgeGraph, "add_edge", _forbidden)
    monkeypatch.setattr(wpe, "_flag_on", lambda: True)
    stage_edges_for_block(str(tmp_path), "DEC-7")


def test_the_outcome_set_is_closed():
    assert {o.name for o in StageOutcome} == {
        "DISABLED", "NO_BLOCK_ID", "BLOCK_NOT_FOUND", "NO_CANDIDATES", "STAGED",
        "ERROR"}


# ---------------------------------------------------------------------------
# Wiring: the apply path must actually CALL this. "Imported" is not "wired".
# ---------------------------------------------------------------------------


def test_the_apply_path_calls_the_hook_on_its_success_route():
    """A module can be complete, tested and imported while nothing reaches it.

    Asserted on the SOURCE of the success route rather than by driving a full apply,
    because a full apply needs a staged proposal, a snapshot and a lock — and a test
    that heavy would get skipped, which is the failure mode this repo's evidence gate
    exists to forbid. What is checked is specific: the call sits on the same route as
    the receipt commit and the belief update, so it cannot be reached only on a
    failure branch.
    """
    import inspect

    from mind_mem import apply_engine

    source = inspect.getsource(apply_engine._apply_proposal_locked)
    assert "_stage_write_path_edges" in source, (
        "the apply path does not call the write-path edge hook; the module would be "
        "complete and unreachable"
    )
    # On the SUCCESS route: after the receipt is written, not in an error branch.
    receipt_at = source.index("update_receipt(receipt_path")
    hook_at = source.index("_stage_write_path_edges")
    assert hook_at > receipt_at, (
        "the hook runs before the receipt commit, so it could stage edges for an "
        "apply that then fails"
    )


def test_the_hook_swallows_nothing_it_should_report(monkeypatch, capsys):
    """The hook must not raise (the apply is already committed) but must SAY when it
    staged something — a silent success is indistinguishable from a no-op, and the
    roadmap's claim is that a write proposes edges."""
    from mind_mem import apply_engine
    from mind_mem.write_path_edges import StageResult

    monkeypatch.setattr(
        "mind_mem.write_path_edges.stage_edges_for_block",
        lambda ws, bid: StageResult(StageOutcome.STAGED, signals_written=3),
    )
    apply_engine._stage_write_path_edges("/tmp/ws", "DEC-9")
    out = capsys.readouterr().out
    assert "3 edge proposal" in out, out
    assert "nothing was written to the graph" in out, out


def test_the_hook_never_raises_even_when_staging_explodes(monkeypatch):
    """POSITIVE CONTROL for the contract that matters most: the apply has already
    committed, so an exception here would report failure for completed work."""
    from mind_mem import apply_engine

    def _boom(ws, bid):
        raise RuntimeError("store offline")

    monkeypatch.setattr("mind_mem.write_path_edges.stage_edges_for_block", _boom)
    apply_engine._stage_write_path_edges("/tmp/ws", "DEC-9")  # must not raise


def test_the_hook_is_a_noop_without_a_block_id(monkeypatch):
    """An apply with no TargetBlock must not reach staging at all."""
    from mind_mem import apply_engine

    def _forbidden(ws, bid):  # pragma: no cover — must never run
        raise AssertionError("staging was reached with no block id")

    monkeypatch.setattr("mind_mem.write_path_edges.stage_edges_for_block", _forbidden)
    apply_engine._stage_write_path_edges("/tmp/ws", "")
