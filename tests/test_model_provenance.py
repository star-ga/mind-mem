"""Tests for ``mind_mem.model_provenance`` — base_model allowlist."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.model_provenance import (
    DEFAULT_PUBLISHERS,
    Publisher,
    check_provenance,
    list_publishers,
)


def _write_config(root: Path, body: dict) -> None:
    """Helper — emit a HF-flavoured ``config.json``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(body))


# ---------------------------------------------------------------------------
# Happy paths — known publishers
# ---------------------------------------------------------------------------


class TestKnownPublishers:
    """Each canonical publisher should clear the gate on its real slug."""

    @pytest.mark.parametrize(
        ("base_model", "expected_publisher"),
        [
            ("Qwen/Qwen2.5-7B", "Alibaba Qwen"),
            ("meta-llama/Llama-3.1-8B", "Meta Llama"),
            ("facebook/opt-1.3b", "Meta Llama"),
            ("mistralai/Mistral-7B-v0.1", "Mistral AI"),
            ("google/gemma-2-9b", "Google Gemma"),
            ("ibm-granite/granite-7b-base", "IBM Granite"),
            ("ibm/granite-7b-base", "IBM Granite"),
            ("openai/whisper-large-v3", "OpenAI"),
            ("anthropic/something-public", "Anthropic"),
            ("deepseek-ai/deepseek-v3", "DeepSeek"),
            ("microsoft/phi-3-mini", "Microsoft Phi"),
            ("tiiuae/falcon-7b", "TII Falcon"),
        ],
    )
    def test_canonical_namespace_matches(self, tmp_path: Path, base_model: str, expected_publisher: str) -> None:
        _write_config(tmp_path, {"base_model": base_model})
        result = check_provenance(tmp_path)
        assert result.passed
        assert result.matched_publisher == expected_publisher
        assert result.base_model == base_model

    def test_namespace_match_is_case_insensitive(self, tmp_path: Path) -> None:
        # HF org slugs ARE case-sensitive in URLs, but real configs often
        # mis-case ("QWEN/qwen2..."), and we'd rather pass the audit than
        # block the real Qwen weights on a typo.
        _write_config(tmp_path, {"base_model": "QWEN/Qwen3-8B"})
        result = check_provenance(tmp_path)
        assert result.passed
        assert result.matched_publisher == "Alibaba Qwen"


# ---------------------------------------------------------------------------
# Negative paths — non-allowlisted, malformed, attacker-flavoured
# ---------------------------------------------------------------------------


class TestRejection:
    def test_unknown_namespace_fails(self, tmp_path: Path) -> None:
        _write_config(tmp_path, {"base_model": "evil-org/malicious-fork"})
        result = check_provenance(tmp_path)
        assert not result.passed
        assert "evil-org" in result.detail
        assert result.base_model == "evil-org/malicious-fork"
        assert any("evil-org" in e for e in result.evidence)

    def test_typo_squat_fails(self, tmp_path: Path) -> None:
        # Common attacker pattern: typo-squat the canonical org slug.
        # ``meta-Ilama`` (capital-I) would slip past a naive prefix
        # match — ours requires an exact slug equality after lowercasing.
        _write_config(tmp_path, {"base_model": "meta-Ilama/Llama-3-8B"})
        result = check_provenance(tmp_path)
        assert not result.passed
        assert result.matched_publisher is None

    def test_missing_namespace_fails(self, tmp_path: Path) -> None:
        # base_model present but no slash → no namespace at all.
        _write_config(tmp_path, {"base_model": "raw-name-no-org"})
        result = check_provenance(tmp_path)
        assert not result.passed
        assert "not in allowlist" in result.detail

    def test_empty_namespace_fails(self, tmp_path: Path) -> None:
        # Edge case: leading slash → empty namespace string.
        _write_config(tmp_path, {"base_model": "/no-org-prefix"})
        result = check_provenance(tmp_path)
        assert not result.passed
        assert "no namespace" in result.detail


# ---------------------------------------------------------------------------
# Missing / non-string base_model — pass open
# ---------------------------------------------------------------------------


class TestMissingBaseModel:
    def test_no_config_passes(self, tmp_path: Path) -> None:
        # Pretrain checkpoint with no config.json at all → not our problem.
        result = check_provenance(tmp_path)
        assert result.passed
        assert "no base_model declared" in result.detail

    def test_config_without_base_model_passes(self, tmp_path: Path) -> None:
        _write_config(tmp_path, {"model_type": "qwen3"})
        result = check_provenance(tmp_path)
        assert result.passed
        assert "no base_model declared" in result.detail

    def test_base_model_empty_string_passes(self, tmp_path: Path) -> None:
        _write_config(tmp_path, {"base_model": "  "})
        result = check_provenance(tmp_path)
        assert result.passed

    def test_base_model_non_string_passes(self, tmp_path: Path) -> None:
        _write_config(tmp_path, {"base_model": ["nested", "list"]})
        result = check_provenance(tmp_path)
        # Non-string base_model is treated as "not declared" — the
        # safetensors header check + remote-code check still fire.
        assert result.passed


# ---------------------------------------------------------------------------
# Allow-extra and custom publisher tuples
# ---------------------------------------------------------------------------


class TestAllowExtra:
    def test_allow_extra_extends_default(self, tmp_path: Path) -> None:
        _write_config(tmp_path, {"base_model": "internal-org/internal-finetune"})
        # Default allowlist would reject this.
        before = check_provenance(tmp_path)
        assert not before.passed
        # With operator opt-in, it passes.
        after = check_provenance(tmp_path, allow_extra=("internal-org",))
        assert after.passed
        assert "operator-allowlist" in (after.matched_publisher or "")

    def test_custom_publisher_tuple_replaces_default(self, tmp_path: Path) -> None:
        # When ``publishers`` is passed, the default list is ignored
        # entirely — useful for air-gapped operators that only trust
        # a single internal mirror.
        _write_config(tmp_path, {"base_model": "Qwen/Qwen3-8B"})
        only_internal = (
            Publisher(
                name="Internal Mirror",
                slugs=frozenset({"internal-mirror"}),
                description="Local mirror only.",
            ),
        )
        result = check_provenance(tmp_path, publishers=only_internal)
        # Even Qwen is rejected when the default list is replaced.
        assert not result.passed


# ---------------------------------------------------------------------------
# Helpers — list_publishers
# ---------------------------------------------------------------------------


class TestListPublishers:
    def test_default_publishers_serializable(self) -> None:
        rows = list_publishers()
        assert len(rows) == len(DEFAULT_PUBLISHERS)
        for row in rows:
            assert "name" in row
            assert "slugs" in row
            assert "description" in row
            assert isinstance(row["slugs"], list)
            # JSON-serializable round trip.
            json.dumps(row)

    def test_default_publishers_have_disjoint_slugs(self) -> None:
        # Two publishers shouldn't claim the same HF org slug — that
        # would make the matched_publisher field non-deterministic.
        seen: set[str] = set()
        for pub in DEFAULT_PUBLISHERS:
            for slug in pub.slugs:
                assert slug not in seen, f"duplicate slug: {slug}"
                seen.add(slug)
