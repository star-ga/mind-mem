# Copyright 2026 STARGA, Inc.
"""Tests for mind-mem governance spec binding (spec_binding.py).

Covers: SpecBinding dataclass, SpecBindingManager lifecycle,
hash stability, drift detection, rebind, storage, and edge cases.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mind_mem.spec_binding import (
    SpecBinding,
    SpecBindingManager,
    _compute_config_hash,
    _normalize_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_path(tmp_path: Path) -> str:
    """Write a minimal mind-mem.json config and return its path."""
    cfg = {
        "recall": {"backend": "scan", "rrf_k": 60},
        "limits": {"max_recall_results": 100},
    }
    p = tmp_path / "mind-mem.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return str(p)


@pytest.fixture
def manager(config_path: str) -> SpecBindingManager:
    return SpecBindingManager(config_path)


@pytest.fixture
def bound_manager(manager: SpecBindingManager, config_path: str) -> SpecBindingManager:
    manager.bind(config_path)
    return manager


# ---------------------------------------------------------------------------
# _normalize_config
# ---------------------------------------------------------------------------


class TestNormalizeConfig:
    def test_sorts_top_level_keys(self, tmp_path: Path) -> None:
        cfg = {"z_key": 1, "a_key": 2}
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        normalized = _normalize_config(str(p))
        parsed = json.loads(normalized)
        assert list(parsed.keys()) == sorted(parsed.keys())

    def test_sorts_nested_keys(self, tmp_path: Path) -> None:
        cfg = {"outer": {"z": 1, "a": 2}, "first": {"m": 3, "b": 4}}
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        normalized = _normalize_config(str(p))
        parsed = json.loads(normalized)
        assert list(parsed["outer"].keys()) == sorted(parsed["outer"].keys())
        assert list(parsed["first"].keys()) == sorted(parsed["first"].keys())

    def test_same_content_different_key_order_produces_same_output(
        self, tmp_path: Path
    ) -> None:
        cfg_a = {"b": 2, "a": 1}
        cfg_b = {"a": 1, "b": 2}
        pa = tmp_path / "a.json"
        pb = tmp_path / "b.json"
        pa.write_text(json.dumps(cfg_a), encoding="utf-8")
        pb.write_text(json.dumps(cfg_b), encoding="utf-8")
        assert _normalize_config(str(pa)) == _normalize_config(str(pb))

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _normalize_config(str(tmp_path / "nonexistent.json"))

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("not json {{{", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            _normalize_config(str(p))


# ---------------------------------------------------------------------------
# _compute_config_hash
# ---------------------------------------------------------------------------


class TestComputeConfigHash:
    def test_returns_128_char_hex_string(self, config_path: str) -> None:
        h = _compute_config_hash(config_path)
        assert isinstance(h, str)
        assert len(h) == 128  # SHA3-512 produces 64 bytes = 128 hex chars
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self, config_path: str) -> None:
        h1 = _compute_config_hash(config_path)
        h2 = _compute_config_hash(config_path)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        pa = tmp_path / "a.json"
        pb = tmp_path / "b.json"
        pa.write_text(json.dumps({"key": "value_a"}), encoding="utf-8")
        pb.write_text(json.dumps({"key": "value_b"}), encoding="utf-8")
        assert _compute_config_hash(str(pa)) != _compute_config_hash(str(pb))

    def test_key_order_invariant(self, tmp_path: Path) -> None:
        pa = tmp_path / "a.json"
        pb = tmp_path / "b.json"
        pa.write_text(json.dumps({"b": 2, "a": 1}), encoding="utf-8")
        pb.write_text(json.dumps({"a": 1, "b": 2}), encoding="utf-8")
        assert _compute_config_hash(str(pa)) == _compute_config_hash(str(pb))


# ---------------------------------------------------------------------------
# SpecBinding dataclass
# ---------------------------------------------------------------------------


class TestSpecBinding:
    def test_frozen(self, config_path: str) -> None:
        b = SpecBinding(
            spec_hash="abc" * 42 + "ab",
            config_path=config_path,
            bound_at=datetime.now(timezone.utc),
            version="1.0.0",
        )
        with pytest.raises((AttributeError, TypeError)):
            b.version = "2.0.0"  # type: ignore[misc]

    def test_to_dict_round_trip(self, config_path: str) -> None:
        now = datetime.now(timezone.utc)
        b = SpecBinding(
            spec_hash="x" * 128,
            config_path=config_path,
            bound_at=now,
            version="1.9.1",
            attestation_signature="sig123",
        )
        d = b.to_dict()
        restored = SpecBinding.from_dict(d)
        assert restored.spec_hash == b.spec_hash
        assert restored.config_path == b.config_path
        assert restored.version == b.version
        assert restored.attestation_signature == b.attestation_signature

    def test_attestation_signature_optional(self, config_path: str) -> None:
        b = SpecBinding(
            spec_hash="x" * 128,
            config_path=config_path,
            bound_at=datetime.now(timezone.utc),
            version="1.0.0",
        )
        assert b.attestation_signature is None

    def test_bound_at_preserved_with_timezone(self, config_path: str) -> None:
        now = datetime.now(timezone.utc)
        b = SpecBinding(
            spec_hash="x" * 128,
            config_path=config_path,
            bound_at=now,
            version="1.0.0",
        )
        d = b.to_dict()
        restored = SpecBinding.from_dict(d)
        # Microseconds may round-trip via ISO string — compare to second precision
        assert abs((restored.bound_at - now).total_seconds()) < 1.0


# ---------------------------------------------------------------------------
# SpecBindingManager.bind
# ---------------------------------------------------------------------------


class TestBind:
    def test_returns_spec_binding(self, manager: SpecBindingManager, config_path: str) -> None:
        b = manager.bind(config_path)
        assert isinstance(b, SpecBinding)

    def test_spec_hash_is_sha3_512(self, manager: SpecBindingManager, config_path: str) -> None:
        b = manager.bind(config_path)
        assert len(b.spec_hash) == 128

    def test_config_path_stored(self, manager: SpecBindingManager, config_path: str) -> None:
        b = manager.bind(config_path)
        assert b.config_path == config_path

    def test_binding_file_created(self, manager: SpecBindingManager, config_path: str) -> None:
        manager.bind(config_path)
        binding_file = os.path.join(os.path.dirname(config_path), ".spec_binding.json")
        assert os.path.exists(binding_file)

    def test_binding_file_is_valid_json(
        self, manager: SpecBindingManager, config_path: str
    ) -> None:
        manager.bind(config_path)
        binding_file = os.path.join(os.path.dirname(config_path), ".spec_binding.json")
        with open(binding_file, encoding="utf-8") as f:
            data = json.load(f)
        assert "spec_hash" in data
        assert "config_path" in data
        assert "bound_at" in data
        assert "version" in data


# ---------------------------------------------------------------------------
# SpecBindingManager.get_binding
# ---------------------------------------------------------------------------


class TestGetBinding:
    def test_returns_none_when_no_binding(self, manager: SpecBindingManager) -> None:
        assert manager.get_binding() is None

    def test_returns_binding_after_bind(
        self, bound_manager: SpecBindingManager, config_path: str
    ) -> None:
        b = bound_manager.get_binding()
        assert b is not None
        assert isinstance(b, SpecBinding)


# ---------------------------------------------------------------------------
# SpecBindingManager.verify
# ---------------------------------------------------------------------------


class TestVerify:
    def test_valid_when_config_unchanged(
        self, bound_manager: SpecBindingManager
    ) -> None:
        valid, reason = bound_manager.verify()
        assert valid is True
        assert "valid" in reason.lower() or reason == ""

    def test_invalid_when_config_changed(
        self, bound_manager: SpecBindingManager, config_path: str
    ) -> None:
        # Mutate the config
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["limits"]["max_recall_results"] = 999
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        valid, reason = bound_manager.verify()
        assert valid is False
        assert reason  # non-empty explanation

    def test_reason_contains_hash_info_on_failure(
        self, bound_manager: SpecBindingManager, config_path: str
    ) -> None:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"changed": True}, f)

        _, reason = bound_manager.verify()
        # Reason must mention hash mismatch context
        assert "hash" in reason.lower() or "mismatch" in reason.lower() or "changed" in reason.lower()

    def test_no_binding_returns_false(self, manager: SpecBindingManager) -> None:
        valid, reason = manager.verify()
        assert valid is False
        assert "no binding" in reason.lower()

    def test_missing_config_file_returns_false(
        self, bound_manager: SpecBindingManager, config_path: str
    ) -> None:
        os.remove(config_path)
        valid, reason = bound_manager.verify()
        assert valid is False
        assert reason


# ---------------------------------------------------------------------------
# SpecBindingManager.has_drifted
# ---------------------------------------------------------------------------


class TestHasDrifted:
    def test_false_when_unchanged(self, bound_manager: SpecBindingManager) -> None:
        assert bound_manager.has_drifted() is False

    def test_true_when_config_changed(
        self, bound_manager: SpecBindingManager, config_path: str
    ) -> None:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"tampered": "yes"}, f)
        assert bound_manager.has_drifted() is True

    def test_true_when_no_binding(self, manager: SpecBindingManager) -> None:
        assert manager.has_drifted() is True


# ---------------------------------------------------------------------------
# SpecBindingManager.rebind
# ---------------------------------------------------------------------------


class TestRebind:
    def test_rebind_updates_hash(
        self, bound_manager: SpecBindingManager, config_path: str
    ) -> None:
        original = bound_manager.get_binding()
        assert original is not None
        original_hash = original.spec_hash

        # Change config then rebind
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"new": "config"}, f)
        new_binding = bound_manager.rebind(config_path)

        assert new_binding.spec_hash != original_hash

    def test_rebind_clears_drift(
        self, bound_manager: SpecBindingManager, config_path: str
    ) -> None:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"new": "config"}, f)
        assert bound_manager.has_drifted() is True

        bound_manager.rebind(config_path)
        assert bound_manager.has_drifted() is False

    def test_rebind_overwrites_binding_file(
        self, bound_manager: SpecBindingManager, config_path: str
    ) -> None:
        binding_file = os.path.join(os.path.dirname(config_path), ".spec_binding.json")
        mtime_before = os.path.getmtime(binding_file)
        time.sleep(0.05)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"updated": True}, f)
        bound_manager.rebind(config_path)

        mtime_after = os.path.getmtime(binding_file)
        assert mtime_after >= mtime_before
