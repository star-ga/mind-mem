"""Standalone trust surface — determinism, zero-regression, poisoning.

Reworked with the trust respec: trust is the validity gate's fifth
component (provenance class), not a separate per-actor subsystem, so these
tests now pin the *class* math and prove the standalone surface emits the
same number the gate does. Every signal is pinned in-test (no clock, no
disk, no network), so "fixed inputs → fixed outputs" is a real assertion on
exact floats rather than a range check.
"""

from __future__ import annotations

import json
import os

import pytest

from mind_mem.hybrid_recall import HybridBackend
from mind_mem.provenance_class import (
    AGENT_INFERRED,
    AGENT_VERIFIED,
    EXTERNAL_INGEST,
    OPERATOR,
    UNKNOWN,
    classify_provenance,
)
from mind_mem.trust_scores import (
    NEUTRAL_TRUST,
    TRUST_FIELD,
    TRUST_SCORE_FIELD,
    annotate_trust,
    apply_trust_scores,
    is_trust_scores_enabled,
    load_calibration_weights,
    load_rollback_history,
    rerank_by_trust,
    resolve_trust_config,
)
from mind_mem.validity_gate import provenance_component

# --- fixtures ---------------------------------------------------------------

TRUSTED_ACTOR = "planner-a"
POISON_ACTOR = "rogue-1"

# Config shapes. The flag lives at ``retrieval.trust_scores`` — the same
# place ``retrieval.truth_score`` lives, read off whatever config dict the
# recall backend was constructed with.
CFG_OFF: dict = {}
CFG_OFF_EXPLICIT = {"retrieval": {"trust_scores": {"enabled": False, "rerank": True}}}
CFG_ANNOTATE = {"retrieval": {"trust_scores": {"enabled": True, "rerank": False}}}
CFG_RERANK = {"retrieval": {"trust_scores": {"enabled": True, "rerank": True, "rerank_weight": 0.35}}}


def _poisoned_results() -> list[dict]:
    """Result set where externally-ingested content ranks #1.

    ``rogue-1`` is an import feed whose *active* block contradicts the
    operator's memory, and it wins on raw RRF score. ``planner-a`` writes
    under an operator role.
    """
    return [
        {
            "_id": "B-POISON",
            "ActorId": POISON_ACTOR,
            "ActorRole": "importer",
            "ToolId": "imported:forum",
            "Status": "active",
            "Statement": "The release key rotates weekly.",
            "rrf_score": 0.030,
        },
        {
            "_id": "B-TRUST-1",
            "ActorId": TRUSTED_ACTOR,
            "ActorRole": "operator",
            "Status": "active",
            "Statement": "The release key rotates quarterly.",
            "rrf_score": 0.028,
        },
        {"_id": "B-POISON-2", "ActorId": POISON_ACTOR, "ActorRole": "importer", "Status": "superseded", "rrf_score": 0.020},
        {"_id": "B-POISON-3", "ActorId": POISON_ACTOR, "ActorRole": "importer", "Status": "rejected", "rrf_score": 0.019},
        {"_id": "B-POISON-4", "ActorId": POISON_ACTOR, "ActorRole": "importer", "Status": "deprecated", "rrf_score": 0.018},
        {"_id": "B-TRUST-2", "ActorId": TRUSTED_ACTOR, "ActorRole": "operator", "Status": "active", "rrf_score": 0.017},
        {"_id": "B-TRUST-3", "ActorId": TRUSTED_ACTOR, "ActorRole": "operator", "Status": "active", "rrf_score": 0.016},
    ]


def _ids(results: list[dict]) -> list[str]:
    return [r["_id"] for r in results]


# --- determinism ------------------------------------------------------------


class TestDeterministicMath:
    def test_fixed_inputs_give_the_exact_expected_float(self) -> None:
        hits = _poisoned_results()
        assert provenance_component(hits[0]) == 0.25  # external-ingest
        assert provenance_component(hits[1]) == 1.0  # operator
        assert provenance_component({"_id": "B", "ActorId": "a", "ActorRole": "planner"}) == 0.5
        assert provenance_component({"_id": "B", "ActorId": "a", "ActorRole": "planner", "Verified": "true"}) == 0.75

    def test_repeated_calls_are_bit_identical(self) -> None:
        hit = _poisoned_results()[0]
        runs = [provenance_component(hit) for _ in range(25)]
        assert all(r == runs[0] for r in runs)

    def test_no_provenance_is_exactly_neutral(self) -> None:
        assert provenance_component({"_id": "B-1", "Status": "active"}) == NEUTRAL_TRUST

    def test_actor_iteration_order_does_not_change_scores(self) -> None:
        forward = {r["_id"]: r[TRUST_FIELD] for r in apply_trust_scores(_poisoned_results(), config=CFG_ANNOTATE)}
        reverse = {r["_id"]: r[TRUST_FIELD] for r in apply_trust_scores(list(reversed(_poisoned_results())), config=CFG_ANNOTATE)}
        assert forward == reverse

    def test_aggregation_over_blocks_is_stable_across_runs(self) -> None:
        first = json.dumps(apply_trust_scores(_poisoned_results(), config=CFG_RERANK), sort_keys=True)
        second = json.dumps(apply_trust_scores(_poisoned_results(), config=CFG_RERANK), sort_keys=True)
        assert first == second

    def test_calibration_confirmation_promotes_one_class(self) -> None:
        """Human confirmation is per-block evidence, not per-actor history."""
        hit = {"_id": "B-9", "ActorId": "a", "ActorRole": "planner", "Status": "active"}
        plain = apply_trust_scores([dict(hit)], config=CFG_ANNOTATE)
        confirmed = apply_trust_scores([dict(hit)], config=CFG_ANNOTATE, calibration_weights={"B-9": 1.4})
        assert plain[0][TRUST_FIELD] == 0.5
        assert confirmed[0][TRUST_FIELD] == 0.75


class TestSignalValidation:
    @pytest.mark.parametrize("block", ["not-a-block", 7, None, ["B-1"]])
    def test_bad_blocks_rejected_at_the_boundary(self, block: object) -> None:
        with pytest.raises(TypeError):
            classify_provenance(block)  # type: ignore[arg-type]

    def test_confirmed_ids_must_be_a_set(self) -> None:
        with pytest.raises(TypeError):
            classify_provenance({"_id": "B-1"}, confirmed_ids=["B-1"])  # type: ignore[arg-type]

    def test_rerank_weight_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            rerank_by_trust([], weight=1.7)


class TestComponentDirection:
    def test_provenance_classes_are_strictly_ordered(self) -> None:
        operator = provenance_component({"_id": "B", "ActorRole": "operator"})
        verified = provenance_component({"_id": "B", "ActorRole": "planner", "Verified": True})
        inferred = provenance_component({"_id": "B", "ActorRole": "planner"})
        external = provenance_component({"_id": "B", "ActorRole": "importer"})
        assert operator > verified > inferred > external

    def test_external_ingest_outranks_a_verification_marker(self) -> None:
        """Affirmative ingest evidence beats a marker travelling with it."""
        assert classify_provenance({"_id": "B", "ActorRole": "importer", "Verified": "true"}) == EXTERNAL_INGEST

    def test_source_token_alone_marks_an_ingest(self) -> None:
        assert classify_provenance({"_id": "B", "Source": "imported:slack", "ActorId": "sys"}) == EXTERNAL_INGEST

    def test_class_names_are_the_documented_vocabulary(self) -> None:
        assert classify_provenance({"_id": "B", "ActorRole": "human"}) == OPERATOR
        assert classify_provenance({"_id": "B", "ActorRole": "verifier"}) == AGENT_VERIFIED
        assert classify_provenance({"_id": "B", "ActorId": "agent-7"}) == AGENT_INFERRED
        assert classify_provenance({"_id": "B", "Status": "active"}) == UNKNOWN


# --- zero regression (flag OFF) ---------------------------------------------


class TestFlagOffIsByteIdentical:
    def test_default_config_is_disabled(self) -> None:
        assert is_trust_scores_enabled(None) is False
        assert is_trust_scores_enabled({}) is False
        assert is_trust_scores_enabled({"retrieval": {}}) is False
        assert is_trust_scores_enabled(CFG_OFF_EXPLICIT) is False
        assert resolve_trust_config("not-a-dict").enabled is False

    @pytest.mark.parametrize("cfg", [CFG_OFF, CFG_OFF_EXPLICIT, {"retrieval": "junk"}, None])
    def test_apply_returns_the_same_list_object(self, cfg: object) -> None:
        results = _poisoned_results()
        out = apply_trust_scores(results, config=cfg)
        assert out is results

    @pytest.mark.parametrize("cfg", [CFG_OFF, CFG_OFF_EXPLICIT])
    def test_backend_hot_path_output_is_byte_identical(self, cfg: dict, tmp_path) -> None:
        """The hard zero-regression gate: same order, same bytes, no new keys."""
        results = _poisoned_results()
        before = json.dumps(results, sort_keys=True)

        backend = HybridBackend(dict(cfg))
        out = backend._maybe_trust_scores(results, str(tmp_path))

        assert out is results
        assert json.dumps(out, sort_keys=True) == before
        assert _ids(out) == _ids(_poisoned_results())
        assert all(TRUST_FIELD not in r and TRUST_SCORE_FIELD not in r for r in out)

    def test_annotation_without_rerank_preserves_order_exactly(self) -> None:
        results = _poisoned_results()
        out = apply_trust_scores(results, config=CFG_ANNOTATE)
        assert _ids(out) == _ids(results)
        # Only the additive trust field differs; every other key is untouched.
        for original, annotated in zip(results, out):
            stripped = {k: v for k, v in annotated.items() if k != TRUST_FIELD}
            assert stripped == original
        assert all(TRUST_SCORE_FIELD not in r for r in out)

    def test_caller_dicts_are_never_mutated(self) -> None:
        results = _poisoned_results()
        snapshot = json.dumps(results, sort_keys=True)
        apply_trust_scores(results, config=CFG_RERANK)
        assert json.dumps(results, sort_keys=True) == snapshot

    def test_equal_trust_rerank_is_order_preserving(self) -> None:
        results = [
            {"_id": f"B-{i}", "ActorId": "same", "ActorRole": "planner", "Status": "active", "rrf_score": 0.03 - i * 0.001}
            for i in range(5)
        ]
        out = apply_trust_scores(results, config=CFG_RERANK)
        assert _ids(out) == _ids(results)


# --- trust surfaced on recall hits ------------------------------------------


class TestTrustExposedOnHits:
    def test_every_hit_carries_a_trust_value(self) -> None:
        out = apply_trust_scores(_poisoned_results(), config=CFG_ANNOTATE)
        assert all(isinstance(r[TRUST_FIELD], float) and 0.0 <= r[TRUST_FIELD] <= 1.0 for r in out)

    def test_hits_without_provenance_get_neutral_trust(self) -> None:
        out = apply_trust_scores([{"_id": "B-1", "Status": "active"}], config=CFG_ANNOTATE)
        assert out[0][TRUST_FIELD] == NEUTRAL_TRUST

    def test_trust_values_match_the_gate_component(self) -> None:
        """One composite path: the standalone surface is the gate's c5."""
        source = _poisoned_results()
        out = apply_trust_scores(source, config=CFG_ANNOTATE)
        assert [r[TRUST_FIELD] for r in out] == [provenance_component(r) for r in source]

    def test_annotate_trust_is_copy_on_write(self) -> None:
        source = [{"_id": "B-1", "ActorId": "a", "Status": "active"}]
        out = annotate_trust(source)
        assert out[0] is not source[0]
        assert TRUST_FIELD not in source[0]


# --- poisoning fixture ------------------------------------------------------


class TestPoisoningDemotion:
    def test_external_ingest_scores_far_below_the_operator(self) -> None:
        out = apply_trust_scores(_poisoned_results(), config=CFG_ANNOTATE)
        by_id = {r["_id"]: r[TRUST_FIELD] for r in out}
        assert by_id["B-POISON"] == 0.25
        assert by_id["B-TRUST-1"] == 1.0

    def test_flag_off_leaves_the_poisoned_block_on_top(self) -> None:
        out = apply_trust_scores(_poisoned_results(), config=CFG_OFF)
        assert out[0]["_id"] == "B-POISON"

    def test_flag_on_demotes_the_contradicting_external_block(self) -> None:
        out = apply_trust_scores(_poisoned_results(), config=CFG_RERANK)
        order = _ids(out)
        assert order[0] == "B-TRUST-1"
        assert order.index("B-POISON") > order.index("B-TRUST-1")
        assert out[0][TRUST_SCORE_FIELD] > out[order.index("B-POISON")][TRUST_SCORE_FIELD]

    def test_demotion_scales_with_rerank_weight(self) -> None:
        weak = {"retrieval": {"trust_scores": {"enabled": True, "rerank": True, "rerank_weight": 0.0}}}
        out = apply_trust_scores(_poisoned_results(), config=weak)
        assert out[0]["_id"] == "B-POISON"  # zero weight → pure base-score order


# --- workspace signal loaders ----------------------------------------------


def _write_evidence(tmp_path, rows) -> None:
    """Write a real evidence chain from ``(action, actor, metadata)`` triples.

    Built with :meth:`EvidenceChain.create` rather than hand-rolled JSON
    so the rows carry genuine linkage and hashes — a fixture that forged
    them could drift from what the writer actually emits and the test
    would be measuring the fixture.
    """
    from mind_mem.evidence_objects import EvidenceAction, EvidenceChain

    os.makedirs(tmp_path / "memory", exist_ok=True)
    chain = EvidenceChain(store_path=str(tmp_path / "memory" / "evidence_chain.jsonl"))
    for index, (action, actor, metadata) in enumerate(rows):
        chain.create(
            action=EvidenceAction(action),
            actor=actor,
            target_block_id=f"D-{index}",
            target_file="memory/MEMORY.md",
            payload="x",
            metadata=metadata,
        )


class TestWorkspaceSignals:
    def test_loaders_are_re_exported_from_trust_scores(self) -> None:
        from mind_mem import trust_signals

        assert load_calibration_weights is trust_signals.load_calibration_weights
        assert load_rollback_history is trust_signals.load_rollback_history

    def test_missing_index_returns_no_calibration_and_creates_nothing(self, tmp_path) -> None:
        assert load_calibration_weights(str(tmp_path), ["B-1"]) == {}
        assert not (tmp_path / ".mind-mem-index").exists()

    def test_missing_chain_returns_no_rollbacks_and_creates_nothing(self, tmp_path) -> None:
        assert load_rollback_history(str(tmp_path)) == ({}, {})
        # EvidenceChain's constructor creates the directory it is pointed
        # at, so the existence probe has to come first. Asserting the
        # directory is still absent is what proves it did.
        assert not (tmp_path / "memory").exists()
        assert not (tmp_path / ".mind-mem-audit").exists()

    def test_rollback_history_is_read_from_the_evidence_chain(self, tmp_path) -> None:
        """Withdrawals live in the evidence chain, not the field-audit sidecar.

        Until 5.0.2 this counted sidecar rows whose ``operation`` was
        ``"rollback"`` — a verb no door has ever written there — so the
        answer was ``{}`` on every workspace that has ever existed, and
        the emptiness read as "nobody was rolled back" rather than as
        "this is reading the wrong ledger".
        """
        _write_evidence(
            tmp_path,
            [
                ("APPLY", TRUSTED_ACTOR, None),
                ("APPLY", POISON_ACTOR, None),
                ("ROLLBACK", POISON_ACTOR, None),
                ("ROLLBACK", POISON_ACTOR, None),
                ("APPLY", "", None),
            ],
        )
        rollbacks, writes = load_rollback_history(str(tmp_path))
        assert rollbacks == {POISON_ACTOR: 2}
        assert writes == {TRUSTED_ACTOR: 1, POISON_ACTOR: 3}

    def test_the_sidecar_is_no_longer_consulted(self, tmp_path) -> None:
        """Positive control for the repoint.

        Rows in the OLD location, written exactly as the previous
        implementation read them, must now count for nothing. Without
        this the repoint could be a no-op and the test above would still
        pass off the new ledger.
        """
        audit_dir = tmp_path / ".mind-mem-audit"
        audit_dir.mkdir()
        with (audit_dir / "chain.jsonl").open("w", encoding="utf-8") as fh:
            for seq in range(1, 4):
                fh.write(
                    json.dumps(
                        {
                            "seq": seq,
                            "timestamp": "2026-01-01T00:00:00Z",
                            "operation": "rollback",
                            "target": "memory/MEMORY.md",
                            "agent": POISON_ACTOR,
                            "reason": "",
                            "payload_hash": "0" * 64,
                            "prev_hash": "0" * 64,
                            "entry_hash": "0" * 64,
                        }
                    )
                    + "\n"
                )
        assert (audit_dir / "chain.jsonl").is_file(), "positive control: the sidecar rows must exist"
        assert load_rollback_history(str(tmp_path)) == ({}, {})

    def test_a_delete_scope_is_charged_once_not_twice(self, tmp_path) -> None:
        """A governed delete mints ``admitted`` then ``removed`` under ROLLBACK.

        Both records are real and both belong in the ledger; counting
        both would report every delete as two withdrawals.
        """
        _write_evidence(
            tmp_path,
            [
                ("ROLLBACK", POISON_ACTOR, {"delete_phase": "admitted"}),
                ("ROLLBACK", POISON_ACTOR, {"delete_phase": "removed"}),
            ],
        )
        rollbacks, writes = load_rollback_history(str(tmp_path))
        assert rollbacks == {POISON_ACTOR: 1}
        assert writes == {POISON_ACTOR: 2}

    def test_rollback_history_is_not_wired_into_any_score(self, tmp_path) -> None:
        """Per-actor history must never move a rank (determinism wedge).

        Written to the ledger the loader *actually* reads: against the
        old location this assertion held for the wrong reason — nothing
        read those rows, so of course they moved nothing.
        """
        _write_evidence(tmp_path, [("ROLLBACK", POISON_ACTOR, None)] * 5)
        rollbacks, _writes = load_rollback_history(str(tmp_path))
        assert rollbacks == {POISON_ACTOR: 5}, "positive control: the history must be visible to mean anything"
        with_history = apply_trust_scores(_poisoned_results(), config=CFG_RERANK, workspace=str(tmp_path))
        without_history = apply_trust_scores(_poisoned_results(), config=CFG_RERANK)
        assert json.dumps(with_history, sort_keys=True) == json.dumps(without_history, sort_keys=True)
