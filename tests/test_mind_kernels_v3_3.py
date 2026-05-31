"""Kernel-loading tests for v3.3.0 features.

Each v3.3.0 feature ships a ``.mind`` config kernel so operators can
tune bounds without patching code. This file pins the expected section
names + key names for each kernel — if the module's ``FeatureGate``
contract changes, the kernel must move in lockstep and this test is
the tripwire.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_MIND = Path(__file__).resolve().parent.parent / "mind"


def _load(name: str) -> dict:
    from mind_mem.mind_ffi import load_kernel_config

    return load_kernel_config(str(_REPO_MIND / f"{name}.mind"))


@pytest.mark.parametrize(
    "name,required_sections",
    [
        ("query_plan", ["query_decomposition", "llm_decomposer"]),
        ("graph", ["expansion", "detection", "seed_score"]),
        ("session", ["boost", "session_extraction"]),
        ("truth", ["priors", "age_decay", "contradictions", "access_bonus"]),
        ("answer", ["self_consistency", "verification", "prompts"]),
        ("evidence", ["bundle", "relations", "confidence"]),
        ("ensemble", ["ensemble", "cross_encoder", "bge", "llm", "borda"]),
    ],
)
def test_v3_3_kernel_has_required_sections(name: str, required_sections: list[str]) -> None:
    cfg = _load(name)
    assert cfg, f"{name}.mind failed to parse"
    for section in required_sections:
        assert section in cfg, f"{name}.mind missing section [{section}]"


class TestGraphKernel:
    def test_max_hops_respects_security_cap(self) -> None:
        cfg = _load("graph")
        assert cfg["expansion"]["max_hops"] <= 3, "kernel must not request more hops than _MAX_HOPS=3"

    def test_decay_in_bounds(self) -> None:
        cfg = _load("graph")
        decay = cfg["expansion"]["decay"]
        assert 0.0 < decay <= 1.0

    def test_seed_score_constant_flag(self) -> None:
        cfg = _load("graph")
        # Fixes architect audit #4 — do not accept a kernel that would
        # re-enable the buggy double-decay behaviour.
        assert cfg["seed_score"]["constant_across_hops"] is True


class TestTruthKernel:
    def test_priors_sum_check(self) -> None:
        cfg = _load("truth")
        priors = cfg["priors"]
        assert priors["verified"] > priors["active"] > priors["draft"] > priors["superseded"]

    def test_age_decay_half_life_positive(self) -> None:
        cfg = _load("truth")
        assert cfg["age_decay"]["half_life_days"] > 0
        assert cfg["age_decay"]["floor"] > 0


class TestAnswerKernel:
    def test_self_consistency_samples_odd(self) -> None:
        # Plurality voting wants an odd count to avoid ties.
        cfg = _load("answer")
        samples = cfg["self_consistency"]["samples"]
        assert samples % 2 == 1, f"samples={samples} should be odd"

    def test_all_category_prompts_present(self) -> None:
        cfg = _load("answer")
        prompts = cfg["prompts"]
        for category in ("adversarial", "temporal", "multi_hop", "single_hop", "open_domain"):
            assert category in prompts, f"prompt missing for category={category}"

    def test_verification_patterns_are_strings(self) -> None:
        cfg = _load("answer")
        # The loader flattens [verification.patterns] onto [verification].
        verification = cfg["verification"]
        for name in ("date", "time", "number", "proper_noun", "yes_no"):
            assert isinstance(verification[name], str), f"{name} pattern not a string"


class TestEnsembleKernel:
    def test_bge_model_is_reranker_not_retriever(self) -> None:
        cfg = _load("ensemble")
        bge_model = cfg["bge"]["model"]
        assert "reranker" in bge_model, (
            f"ensemble.mind points to {bge_model} which isn't a reranker; use bge-reranker-v2-m3, not bge-large-en-v1.5 (retrieval model)"
        )

    def test_ensemble_members_non_empty(self) -> None:
        cfg = _load("ensemble")
        assert cfg["ensemble"]["members"], "ensemble has no members — nothing to fuse"
