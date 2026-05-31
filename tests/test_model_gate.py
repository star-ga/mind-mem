"""Tests for ``mind_mem.model_gate`` — load-gate registry."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from mind_mem.model_gate import (
    REASON_AUDIT_FAILED,
    REASON_AUDIT_FAILED_OVERRIDE,
    REASON_AUDITED_NOW,
    REASON_DRIFT_RE_AUDITED,
    REASON_NEVER_AUDITED_OVERRIDE,
    REASON_PATH_NOT_FOUND,
    REASON_TRUSTED_FRESH,
    gate_check,
    gate_list,
    gate_remove,
)


@pytest.fixture
def isolated_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``MIND_MEM_GATE_REGISTRY`` at a per-test JSON file."""
    reg = tmp_path / "registry.json"
    monkeypatch.setenv("MIND_MEM_GATE_REGISTRY", str(reg))
    return reg


@pytest.fixture
def clean_ckpt(tmp_path: Path) -> Path:
    """A safetensors-only checkpoint with a Qwen-flavoured config — passes
    every audit check including provenance."""
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3","base_model":"Qwen/Qwen3-8B"}')
    body = b'{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(body)) + body + b"\x00" * 8)
    return root


@pytest.fixture
def evil_ckpt(tmp_path: Path) -> Path:
    """A checkpoint that fails the provenance check (unknown publisher)."""
    root = tmp_path / "evil"
    root.mkdir()
    (root / "config.json").write_text('{"base_model":"evil-org/malicious-fork"}')
    body = b'{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(body)) + body + b"\x00" * 8)
    return root


# ---------------------------------------------------------------------------
# Happy path — never seen, audited now, then trusted_fresh
# ---------------------------------------------------------------------------


class TestFirstAudit:
    def test_first_audit_passes(self, clean_ckpt: Path, isolated_registry: Path) -> None:
        decision = gate_check(clean_ckpt)
        assert decision.passed
        assert decision.reason == REASON_AUDITED_NOW
        assert decision.audit_passed is True
        assert decision.manifest_sha256 != ""
        # Registry persists a single entry.
        reg = json.loads(isolated_registry.read_text())
        assert len(reg) == 1
        entry = next(iter(reg.values()))
        assert entry["audit_passed"] is True
        assert entry["trust_without_audit"] is False

    def test_second_call_uses_fast_path(self, clean_ckpt: Path, isolated_registry: Path) -> None:
        gate_check(clean_ckpt)
        decision = gate_check(clean_ckpt)
        assert decision.passed
        assert decision.reason == REASON_TRUSTED_FRESH


# ---------------------------------------------------------------------------
# Drift detection — file mutation forces re-audit
# ---------------------------------------------------------------------------


class TestDrift:
    def test_drift_triggers_re_audit(self, clean_ckpt: Path, isolated_registry: Path) -> None:
        first = gate_check(clean_ckpt)
        # Mutate a file — manifest_sha256 changes.
        (clean_ckpt / "config.json").write_text('{"model_type":"qwen3","base_model":"Qwen/Qwen3-14B"}')
        second = gate_check(clean_ckpt)
        assert second.passed
        assert second.reason == REASON_DRIFT_RE_AUDITED
        assert second.manifest_sha256 != first.manifest_sha256


# ---------------------------------------------------------------------------
# Audit-failure path — refuses to load
# ---------------------------------------------------------------------------


class TestAuditFailure:
    def test_unknown_publisher_blocks_load(self, evil_ckpt: Path, isolated_registry: Path) -> None:
        decision = gate_check(evil_ckpt)
        assert not decision.passed
        assert decision.reason == REASON_AUDIT_FAILED
        assert decision.audit_passed is False
        # Registry still records the failure so the next caller sees it.
        reg = json.loads(isolated_registry.read_text())
        assert reg[str(evil_ckpt)]["audit_passed"] is False

    def test_allow_extra_publishers_lets_evil_org_pass(self, evil_ckpt: Path, isolated_registry: Path) -> None:
        decision = gate_check(evil_ckpt, allow_extra_publishers=("evil-org",))
        assert decision.passed
        assert decision.reason == REASON_AUDITED_NOW


# ---------------------------------------------------------------------------
# trust_without_audit override — operator escape hatch
# ---------------------------------------------------------------------------


class TestTrustWithoutAudit:
    def test_never_audited_override(self, evil_ckpt: Path, isolated_registry: Path) -> None:
        decision = gate_check(evil_ckpt, trust_without_audit=True)
        assert decision.passed
        assert decision.reason == REASON_NEVER_AUDITED_OVERRIDE
        # Registry records the override so it's auditable.
        reg = json.loads(isolated_registry.read_text())
        assert reg[str(evil_ckpt)]["trust_without_audit"] is True

    def test_failed_audit_override(self, evil_ckpt: Path, isolated_registry: Path) -> None:
        # First call — audit fails normally.
        first = gate_check(evil_ckpt)
        assert not first.passed
        # Second call — operator overrides.
        second = gate_check(evil_ckpt, trust_without_audit=True)
        assert second.passed
        assert second.reason == REASON_AUDIT_FAILED_OVERRIDE
        reg = json.loads(isolated_registry.read_text())
        assert reg[str(evil_ckpt)]["trust_without_audit"] is True


# ---------------------------------------------------------------------------
# Path errors
# ---------------------------------------------------------------------------


class TestPathErrors:
    def test_missing_path_blocks(self, tmp_path: Path, isolated_registry: Path) -> None:
        decision = gate_check(tmp_path / "nope")
        assert not decision.passed
        assert decision.reason == REASON_PATH_NOT_FOUND


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


class TestRegistryHelpers:
    def test_gate_list_returns_entries(self, clean_ckpt: Path, isolated_registry: Path) -> None:
        gate_check(clean_ckpt)
        rows = gate_list()
        assert len(rows) == 1
        assert rows[0]["path"] == str(clean_ckpt)
        assert rows[0]["audit_passed"] is True

    def test_gate_remove_drops_entry(self, clean_ckpt: Path, isolated_registry: Path) -> None:
        gate_check(clean_ckpt)
        assert gate_remove(clean_ckpt) is True
        # Second remove returns False (idempotent).
        assert gate_remove(clean_ckpt) is False
        # Registry empty.
        reg = json.loads(isolated_registry.read_text() or "{}")
        assert reg == {}


# ---------------------------------------------------------------------------
# Atomic writes — corrupt registry doesn't crash subsequent reads
# ---------------------------------------------------------------------------


class TestRegistryRobustness:
    def test_corrupt_json_is_treated_as_empty(self, clean_ckpt: Path, isolated_registry: Path) -> None:
        # Pre-corrupt the registry — gate_check should still work.
        isolated_registry.parent.mkdir(parents=True, exist_ok=True)
        isolated_registry.write_text("{not valid json")
        decision = gate_check(clean_ckpt)
        assert decision.passed
        assert decision.reason == REASON_AUDITED_NOW
        # And the registry is now valid JSON again.
        reg = json.loads(isolated_registry.read_text())
        assert isinstance(reg, dict)
        assert len(reg) == 1

    def test_non_dict_registry_treated_as_empty(self, clean_ckpt: Path, isolated_registry: Path) -> None:
        isolated_registry.parent.mkdir(parents=True, exist_ok=True)
        isolated_registry.write_text("[1, 2, 3]")
        decision = gate_check(clean_ckpt)
        assert decision.passed
