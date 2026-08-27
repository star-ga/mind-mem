"""Per-actor memory trust scores — determinism, zero-regression, poisoning.

Every signal fed to the scorer is pinned in-test (no clock, no disk, no
network), so the acceptance gate "fixed inputs → fixed outputs" is a real
assertion on exact floats rather than a range check.
"""

from __future__ import annotations

import json

import pytest

from mind_mem.hybrid_recall import HybridBackend
from mind_mem.trust_scores import (
    NEUTRAL_TRUST,
    TRUST_FIELD,
    TRUST_SCORE_FIELD,
    ActorSignals,
    aggregate_actor_signals,
    annotate_trust,
    apply_trust_scores,
    compute_actor_trust,
    compute_trust_map,
    is_trust_scores_enabled,
    load_calibration_weights,
    load_rollback_history,
    rerank_by_trust,
    resolve_trust_config,
)

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
    """Result set where a low-trust actor's contradicting block ranks #1.

    ``rogue-1`` has three governance-rejected blocks plus one *active*
    block that contradicts the trusted actor's memory, and it wins on raw
    RRF score. ``planner-a`` has a clean record.
    """
    return [
        {
            "_id": "B-POISON",
            "ActorId": POISON_ACTOR,
            "Status": "active",
            "Statement": "The release key rotates weekly.",
            "truth_score": 0.6,
            "rrf_score": 0.030,
        },
        {
            "_id": "B-TRUST-1",
            "ActorId": TRUSTED_ACTOR,
            "Status": "active",
            "Statement": "The release key rotates quarterly.",
            "truth_score": 0.85,
            "rrf_score": 0.028,
        },
        {"_id": "B-POISON-2", "ActorId": POISON_ACTOR, "Status": "superseded", "truth_score": 0.15, "rrf_score": 0.020},
        {"_id": "B-POISON-3", "ActorId": POISON_ACTOR, "Status": "rejected", "truth_score": 0.15, "rrf_score": 0.019},
        {"_id": "B-POISON-4", "ActorId": POISON_ACTOR, "Status": "deprecated", "truth_score": 0.15, "rrf_score": 0.018},
        {"_id": "B-TRUST-2", "ActorId": TRUSTED_ACTOR, "Status": "active", "truth_score": 0.85, "rrf_score": 0.017},
        {"_id": "B-TRUST-3", "ActorId": TRUSTED_ACTOR, "Status": "active", "truth_score": 0.85, "rrf_score": 0.016},
    ]


POISON_HISTORY = {
    "rollback_counts": {POISON_ACTOR: 4, TRUSTED_ACTOR: 0},
    "write_counts": {POISON_ACTOR: 5, TRUSTED_ACTOR: 6},
}


def _ids(results: list[dict]) -> list[str]:
    return [r["_id"] for r in results]


# --- determinism ------------------------------------------------------------


class TestDeterministicMath:
    SIGNALS = ActorSignals(
        block_truth=(0.8, 0.6),
        calibration_weights=(1.0,),
        contradicted_blocks=1,
        total_blocks=4,
        rollbacks=1,
        total_writes=5,
    )

    def test_fixed_inputs_give_the_exact_expected_float(self) -> None:
        # truth .7 · cal .5 · contradiction .75 · rollback .8
        # raw = .45*.7 + .25*.5 + .20*.75 + .10*.8 = 0.67
        # evidence = max(2, 1, 4) + 5 = 9
        # shrunk = (0.67*9 + 0.5*3) / (9 + 3) = 0.6275
        trust = compute_actor_trust("a", self.SIGNALS)
        assert trust.trust == 0.6275
        assert trust.evidence_count == 9
        assert (trust.truth, trust.calibration, trust.contradiction, trust.rollback) == (0.7, 0.5, 0.75, 0.8)

    def test_repeated_calls_are_bit_identical(self) -> None:
        runs = [compute_actor_trust("a", self.SIGNALS) for _ in range(25)]
        assert all(r == runs[0] for r in runs)

    def test_no_evidence_is_exactly_neutral(self) -> None:
        assert compute_actor_trust("a", ActorSignals()).trust == NEUTRAL_TRUST

    def test_actor_iteration_order_does_not_change_scores(self) -> None:
        forward = compute_trust_map(aggregate_actor_signals(_poisoned_results(), **POISON_HISTORY))
        reverse = compute_trust_map(aggregate_actor_signals(list(reversed(_poisoned_results())), **POISON_HISTORY))
        assert {k: v.trust for k, v in forward.items()} == {k: v.trust for k, v in reverse.items()}

    def test_aggregation_over_blocks_is_stable_across_runs(self) -> None:
        first = json.dumps(
            {k: v.as_dict() for k, v in compute_trust_map(aggregate_actor_signals(_poisoned_results(), **POISON_HISTORY)).items()},
            sort_keys=True,
        )
        second = json.dumps(
            {k: v.as_dict() for k, v in compute_trust_map(aggregate_actor_signals(_poisoned_results(), **POISON_HISTORY)).items()},
            sort_keys=True,
        )
        assert first == second

    def test_more_evidence_moves_further_from_neutral(self) -> None:
        thin = compute_actor_trust("a", ActorSignals(block_truth=(0.95,), total_blocks=1))
        thick = compute_actor_trust("a", ActorSignals(block_truth=(0.95,) * 20, total_blocks=20))
        assert NEUTRAL_TRUST < thin.trust < thick.trust


class TestSignalValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"total_blocks": -1},
            {"rollbacks": -2, "total_writes": 3},
            {"contradicted_blocks": 3, "total_blocks": 2},
            {"rollbacks": 4, "total_writes": 1},
            {"block_truth": (1.5,)},
            {"calibration_weights": (9.0,)},
            {"block_truth": ("x",)},
        ],
    )
    def test_bad_signals_rejected_at_the_boundary(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            ActorSignals(**kwargs)

    def test_compute_rejects_wrong_type(self) -> None:
        with pytest.raises(TypeError):
            compute_actor_trust("a", {"block_truth": (0.5,)})  # type: ignore[arg-type]

    def test_rerank_weight_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            rerank_by_trust([], weight=1.7)


class TestComponentDirection:
    def test_contradictions_lower_trust(self) -> None:
        clean = compute_actor_trust("a", ActorSignals(block_truth=(0.7,) * 4, total_blocks=4))
        dirty = compute_actor_trust("a", ActorSignals(block_truth=(0.7,) * 4, contradicted_blocks=3, total_blocks=4))
        assert dirty.trust < clean.trust

    def test_rollbacks_lower_trust(self) -> None:
        clean = compute_actor_trust("a", ActorSignals(total_blocks=4, block_truth=(0.7,) * 4, total_writes=10))
        rolled = compute_actor_trust("a", ActorSignals(total_blocks=4, block_truth=(0.7,) * 4, rollbacks=9, total_writes=10))
        assert rolled.trust < clean.trust

    def test_calibration_feedback_moves_trust(self) -> None:
        demoted = compute_actor_trust("a", ActorSignals(calibration_weights=(0.5,) * 4, total_blocks=4))
        boosted = compute_actor_trust("a", ActorSignals(calibration_weights=(1.5,) * 4, total_blocks=4))
        assert demoted.trust < boosted.trust


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
        out = apply_trust_scores(results, config=cfg, **POISON_HISTORY)
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
        out = apply_trust_scores(results, config=CFG_ANNOTATE, **POISON_HISTORY)
        assert _ids(out) == _ids(results)
        # Only the additive trust field differs; every other key is untouched.
        for original, annotated in zip(results, out):
            stripped = {k: v for k, v in annotated.items() if k != TRUST_FIELD}
            assert stripped == original
        assert all(TRUST_SCORE_FIELD not in r for r in out)

    def test_caller_dicts_are_never_mutated(self) -> None:
        results = _poisoned_results()
        snapshot = json.dumps(results, sort_keys=True)
        apply_trust_scores(results, config=CFG_RERANK, **POISON_HISTORY)
        assert json.dumps(results, sort_keys=True) == snapshot

    def test_equal_trust_rerank_is_order_preserving(self) -> None:
        results = [
            {"_id": f"B-{i}", "ActorId": "same", "Status": "active", "truth_score": 0.7, "rrf_score": 0.03 - i * 0.001} for i in range(5)
        ]
        out = apply_trust_scores(results, config=CFG_RERANK)
        assert _ids(out) == _ids(results)


# --- trust surfaced on recall hits ------------------------------------------


class TestTrustExposedOnHits:
    def test_every_hit_carries_a_trust_value(self) -> None:
        out = apply_trust_scores(_poisoned_results(), config=CFG_ANNOTATE, **POISON_HISTORY)
        assert all(isinstance(r[TRUST_FIELD], float) and 0.0 <= r[TRUST_FIELD] <= 1.0 for r in out)

    def test_hits_without_provenance_get_neutral_trust(self) -> None:
        out = apply_trust_scores([{"_id": "B-1", "Status": "active", "truth_score": 0.9}], config=CFG_ANNOTATE)
        assert out[0][TRUST_FIELD] == NEUTRAL_TRUST

    def test_trust_values_match_the_pure_computation(self) -> None:
        trust_map = compute_trust_map(aggregate_actor_signals(_poisoned_results(), **POISON_HISTORY))
        out = apply_trust_scores(_poisoned_results(), config=CFG_ANNOTATE, **POISON_HISTORY)
        assert out[0][TRUST_FIELD] == trust_map[POISON_ACTOR].trust
        assert out[1][TRUST_FIELD] == trust_map[TRUSTED_ACTOR].trust

    def test_annotate_trust_is_copy_on_write(self) -> None:
        source = [{"_id": "B-1", "ActorId": "a", "Status": "active"}]
        out = annotate_trust(source, {})
        assert out[0] is not source[0]
        assert TRUST_FIELD not in source[0]


# --- poisoning fixture ------------------------------------------------------


class TestPoisoningDemotion:
    def test_low_trust_actor_scores_far_below_the_clean_actor(self) -> None:
        trust_map = compute_trust_map(aggregate_actor_signals(_poisoned_results(), **POISON_HISTORY))
        assert trust_map[POISON_ACTOR].trust == 0.3598
        assert trust_map[TRUSTED_ACTOR].trust == 0.7306

    def test_flag_off_leaves_the_poisoned_block_on_top(self) -> None:
        out = apply_trust_scores(_poisoned_results(), config=CFG_OFF, **POISON_HISTORY)
        assert out[0]["_id"] == "B-POISON"

    def test_flag_on_demotes_the_contradicting_low_trust_block(self) -> None:
        out = apply_trust_scores(_poisoned_results(), config=CFG_RERANK, **POISON_HISTORY)
        order = _ids(out)
        assert order[0] == "B-TRUST-1"
        assert order.index("B-POISON") > order.index("B-TRUST-1")
        assert out[0][TRUST_SCORE_FIELD] > out[order.index("B-POISON")][TRUST_SCORE_FIELD]

    def test_demotion_scales_with_rerank_weight(self) -> None:
        weak = {"retrieval": {"trust_scores": {"enabled": True, "rerank": True, "rerank_weight": 0.0}}}
        out = apply_trust_scores(_poisoned_results(), config=weak, **POISON_HISTORY)
        assert out[0]["_id"] == "B-POISON"  # zero weight → pure base-score order


# --- workspace signal loaders ----------------------------------------------


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
        assert not (tmp_path / ".mind-mem-audit").exists()

    def test_rollback_history_is_read_from_the_audit_chain(self, tmp_path) -> None:
        audit_dir = tmp_path / ".mind-mem-audit"
        audit_dir.mkdir()
        rows = [
            ("apply_proposal", TRUSTED_ACTOR),
            ("apply_proposal", POISON_ACTOR),
            ("rollback", POISON_ACTOR),
            ("rollback", POISON_ACTOR),
            ("apply_proposal", ""),
        ]
        with (audit_dir / "chain.jsonl").open("w", encoding="utf-8") as fh:
            for seq, (operation, agent) in enumerate(rows, start=1):
                fh.write(
                    json.dumps(
                        {
                            "seq": seq,
                            "timestamp": "2026-01-01T00:00:00Z",
                            "operation": operation,
                            "target": "memory/MEMORY.md",
                            "agent": agent,
                            "reason": "",
                            "payload_hash": "0" * 64,
                            "prev_hash": "0" * 64,
                            "entry_hash": "0" * 64,
                        }
                    )
                    + "\n"
                )
        rollbacks, writes = load_rollback_history(str(tmp_path))
        assert rollbacks == {POISON_ACTOR: 2}
        assert writes == {TRUSTED_ACTOR: 1, POISON_ACTOR: 3}
