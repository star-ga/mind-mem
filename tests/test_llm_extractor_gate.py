"""Backend wiring — :func:`mind_mem.llm_extractor._gate_check_local`.

Covers the v3.8.6 contract: every local-directory checkpoint passed to
the in-process transformers backend must clear
:func:`mind_mem.model_gate.gate_check` before ``from_pretrained`` is
called. HF hub IDs and single-file binaries fall outside the gate's
scope and pass through unchanged. Two env-vars control the policy:

* ``MIND_MEM_SKIP_GATE=1`` — bypass the gate entirely.
* ``MIND_MEM_TRUST_WITHOUT_AUDIT=1`` — forward ``trust_without_audit``
  to ``gate_check``; the override is recorded in the gate ledger.

Tests don't load a real LLM — they exercise ``_gate_check_local``
directly with a controlled checkpoint directory + monkeypatched
``gate_check``.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from mind_mem.llm_extractor import _gate_check_local


@pytest.fixture
def isolated_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    reg = tmp_path / "registry.json"
    monkeypatch.setenv("MIND_MEM_GATE_REGISTRY", str(reg))
    return reg


@pytest.fixture
def clean_ckpt(tmp_path: Path) -> Path:
    """Minimal HF-shaped directory that passes every audit check."""
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3","base_model":"Qwen/Qwen3-8B"}')
    body = b'{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(body)) + body + b"\x00" * 8)
    return root


@pytest.fixture
def evil_ckpt(tmp_path: Path) -> Path:
    """Directory whose provenance check fails — namespace not on the
    canonical publisher allowlist."""
    root = tmp_path / "evil"
    root.mkdir()
    (root / "config.json").write_text('{"base_model":"evil-org/malicious-fork"}')
    body = b'{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(body)) + body + b"\x00" * 8)
    return root


# ---------------------------------------------------------------------------
# Bypass paths — gate doesn't apply
# ---------------------------------------------------------------------------


class TestBypass:
    def test_skip_env_var_bypasses_gate(self, evil_ckpt: Path, isolated_registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Even an evil checkpoint passes when the operator opts out.
        monkeypatch.setenv("MIND_MEM_SKIP_GATE", "1")
        _gate_check_local(str(evil_ckpt))  # must not raise
        # Registry stays empty because we bypassed gate_check entirely.
        assert not isolated_registry.exists() or isolated_registry.read_text().strip() in ("", "{}")

    def test_hub_id_bypasses_gate(self, isolated_registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # "Qwen/Qwen3-8B" is an HF hub ID, not a local path. The gate
        # only handles directory checkpoints, so this passes through.
        monkeypatch.delenv("MIND_MEM_SKIP_GATE", raising=False)
        _gate_check_local("Qwen/Qwen3-8B")  # must not raise

    def test_single_file_path_bypasses_gate(self, tmp_path: Path, isolated_registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A standalone .gguf file — also not in scope for the
        # directory-oriented gate.
        gguf = tmp_path / "weights.gguf"
        gguf.write_bytes(b"GGUF\x00" * 16)
        monkeypatch.delenv("MIND_MEM_SKIP_GATE", raising=False)
        _gate_check_local(str(gguf))  # must not raise


# ---------------------------------------------------------------------------
# Gate active path — clean checkpoint passes
# ---------------------------------------------------------------------------


class TestGateAllow:
    def test_clean_checkpoint_passes(self, clean_ckpt: Path, isolated_registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MIND_MEM_SKIP_GATE", raising=False)
        monkeypatch.delenv("MIND_MEM_TRUST_WITHOUT_AUDIT", raising=False)
        _gate_check_local(str(clean_ckpt))  # must not raise
        # Gate registered the checkpoint as audited_now / passed.
        import json

        reg = json.loads(isolated_registry.read_text())
        entry = next(iter(reg.values()))
        assert entry["audit_passed"] is True

    def test_repeat_call_uses_fast_path(self, clean_ckpt: Path, isolated_registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MIND_MEM_SKIP_GATE", raising=False)
        _gate_check_local(str(clean_ckpt))
        # Second call hits trusted_fresh — still no raise, registry
        # unchanged in shape (one entry).
        _gate_check_local(str(clean_ckpt))
        import json

        reg = json.loads(isolated_registry.read_text())
        assert len(reg) == 1


# ---------------------------------------------------------------------------
# Gate active path — failing checkpoint refuses load
# ---------------------------------------------------------------------------


class TestGateBlock:
    def test_failed_audit_raises_runtimeerror(self, evil_ckpt: Path, isolated_registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MIND_MEM_SKIP_GATE", raising=False)
        monkeypatch.delenv("MIND_MEM_TRUST_WITHOUT_AUDIT", raising=False)
        with pytest.raises(RuntimeError, match="gate refused load"):
            _gate_check_local(str(evil_ckpt))
        # The failure is still logged in the registry so the next
        # caller sees the prior decision.
        import json

        reg = json.loads(isolated_registry.read_text())
        assert reg[str(evil_ckpt.resolve())]["audit_passed"] is False

    def test_error_message_mentions_overrides(self, evil_ckpt: Path, isolated_registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MIND_MEM_SKIP_GATE", raising=False)
        monkeypatch.delenv("MIND_MEM_TRUST_WITHOUT_AUDIT", raising=False)
        with pytest.raises(RuntimeError) as exc_info:
            _gate_check_local(str(evil_ckpt))
        msg = str(exc_info.value)
        # The operator needs to know which env-vars unblock the load.
        assert "MIND_MEM_TRUST_WITHOUT_AUDIT" in msg
        assert "MIND_MEM_SKIP_GATE" in msg


# ---------------------------------------------------------------------------
# trust_without_audit env override
# ---------------------------------------------------------------------------


class TestTrustOverride:
    def test_trust_env_var_lets_evil_pass(self, evil_ckpt: Path, isolated_registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MIND_MEM_SKIP_GATE", raising=False)
        monkeypatch.setenv("MIND_MEM_TRUST_WITHOUT_AUDIT", "1")
        _gate_check_local(str(evil_ckpt))  # must not raise
        # Override is recorded in the ledger for audit trail.
        import json

        reg = json.loads(isolated_registry.read_text())
        assert reg[str(evil_ckpt.resolve())]["trust_without_audit"] is True


# ---------------------------------------------------------------------------
# Label propagation — operator sees which backend tripped the gate
# ---------------------------------------------------------------------------


class TestLabel:
    def test_default_label_is_transformers(self, evil_ckpt: Path, isolated_registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MIND_MEM_SKIP_GATE", raising=False)
        monkeypatch.delenv("MIND_MEM_TRUST_WITHOUT_AUDIT", raising=False)
        with pytest.raises(RuntimeError, match="transformers checkpoint"):
            _gate_check_local(str(evil_ckpt))

    def test_custom_label_appears_in_error(self, evil_ckpt: Path, isolated_registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MIND_MEM_SKIP_GATE", raising=False)
        monkeypatch.delenv("MIND_MEM_TRUST_WITHOUT_AUDIT", raising=False)
        with pytest.raises(RuntimeError, match="rerank checkpoint"):
            _gate_check_local(str(evil_ckpt), label="rerank")


# ---------------------------------------------------------------------------
# Drift detection round-trip — file mutation triggers re-audit
# ---------------------------------------------------------------------------


class TestDriftReAudit:
    def test_drift_re_audit_passes_for_clean_mutation(
        self, clean_ckpt: Path, isolated_registry: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MIND_MEM_SKIP_GATE", raising=False)
        _gate_check_local(str(clean_ckpt))
        # Mutate config in a way that's still safe — gate re-audits and
        # passes again.
        (clean_ckpt / "config.json").write_text('{"model_type":"qwen3","base_model":"Qwen/Qwen3-14B"}')
        _gate_check_local(str(clean_ckpt))  # must not raise
