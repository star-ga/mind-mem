"""Pinned-model audit pipeline — release-CI gate.

Covers the v3.8.7 contract: ``mm audit-pinned`` reads
``audit_pinned_models`` from ``mind-mem.json`` and runs the seven-check
audit (plus optional Ed25519 verify) on each. Exit-code mapping:

* ``0`` — every entry passed; also returned when the config is absent
  or the list is empty.
* ``1`` — at least one entry produced a HIGH finding or failed verify.
* ``2`` — config could not be parsed, or a pinned path is missing
  while ``--fail-on-missing`` was set.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from mind_mem.audit_pinned import (
    PinnedConfigError,
    audit_pinned,
    format_pinned_report_text,
    load_pinned_models,
)


@pytest.fixture
def clean_ckpt(tmp_path: Path) -> Path:
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3","base_model":"Qwen/Qwen3-8B"}')
    body = b'{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(body)) + body + b"\x00" * 8)
    return root


@pytest.fixture
def evil_ckpt(tmp_path: Path) -> Path:
    root = tmp_path / "evil"
    root.mkdir()
    (root / "config.json").write_text('{"base_model":"evil-org/malicious-fork"}')
    body = b'{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(body)) + body + b"\x00" * 8)
    return root


def _write_config(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "mind-mem.json"
    p.write_text(json.dumps(payload))
    return p


# ---------------------------------------------------------------------------
# Schema parsing
# ---------------------------------------------------------------------------


class TestLoadPinnedModels:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_pinned_models(tmp_path / "nope.json") == []

    def test_no_key_returns_empty(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, {"extraction": {"model": "qwen"}})
        assert load_pinned_models(cfg) == []

    def test_empty_list_returns_empty(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, {"audit_pinned_models": []})
        assert load_pinned_models(cfg) == []

    def test_string_entry_normalised(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, {"audit_pinned_models": ["/x/y"]})
        models = load_pinned_models(cfg)
        assert len(models) == 1
        assert models[0].path == "/x/y"
        assert models[0].verify is False
        assert models[0].allow_publishers == ()

    def test_object_entry_with_overrides(self, tmp_path: Path) -> None:
        cfg = _write_config(
            tmp_path,
            {"audit_pinned_models": [{"path": "/a", "verify": True, "allow_publishers": ["my-org"]}]},
        )
        models = load_pinned_models(cfg)
        assert models[0].verify is True
        assert models[0].allow_publishers == ("my-org",)

    def test_mixed_list_accepted(self, tmp_path: Path) -> None:
        cfg = _write_config(
            tmp_path,
            {
                "audit_pinned_models": [
                    "/p1",
                    {"path": "/p2", "verify": True},
                ]
            },
        )
        assert len(load_pinned_models(cfg)) == 2

    def test_invalid_top_level_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "mind-mem.json"
        p.write_text('["not", "an", "object"]')
        with pytest.raises(PinnedConfigError, match="JSON object"):
            load_pinned_models(p)

    def test_non_array_value_raises(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, {"audit_pinned_models": "/x"})
        with pytest.raises(PinnedConfigError, match="JSON array"):
            load_pinned_models(cfg)

    def test_object_missing_path_raises(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, {"audit_pinned_models": [{"verify": True}]})
        with pytest.raises(PinnedConfigError, match="missing required 'path'"):
            load_pinned_models(cfg)

    def test_bad_allow_publishers_type_raises(self, tmp_path: Path) -> None:
        cfg = _write_config(
            tmp_path,
            {"audit_pinned_models": [{"path": "/x", "allow_publishers": [1, 2]}]},
        )
        with pytest.raises(PinnedConfigError, match="list of strings"):
            load_pinned_models(cfg)

    def test_bad_json_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "mind-mem.json"
        p.write_text("{not valid json")
        with pytest.raises(PinnedConfigError, match="could not read"):
            load_pinned_models(p)


# ---------------------------------------------------------------------------
# Pipeline — happy paths
# ---------------------------------------------------------------------------


class TestAuditPinned:
    def test_no_config_passes(self, tmp_path: Path) -> None:
        report = audit_pinned(tmp_path / "nope.json", workspace=tmp_path)
        assert report.passed is True
        assert report.findings == []
        assert report.config_present is False

    def test_empty_list_passes(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, {"audit_pinned_models": []})
        report = audit_pinned(cfg, workspace=tmp_path)
        assert report.passed is True
        assert report.findings == []
        assert report.config_present is True

    def test_single_clean_checkpoint_passes(self, tmp_path: Path, clean_ckpt: Path) -> None:
        cfg = _write_config(tmp_path, {"audit_pinned_models": [str(clean_ckpt)]})
        report = audit_pinned(cfg, workspace=tmp_path)
        assert report.passed is True
        assert len(report.findings) == 1
        assert report.findings[0].audit_passed is True
        assert report.findings[0].verify_attempted is False

    def test_failing_checkpoint_fails(self, tmp_path: Path, evil_ckpt: Path) -> None:
        cfg = _write_config(tmp_path, {"audit_pinned_models": [str(evil_ckpt)]})
        report = audit_pinned(cfg, workspace=tmp_path)
        assert report.passed is False
        assert report.findings[0].audit_passed is False
        # Provenance is the failed check on this fixture.
        assert "provenance" in report.findings[0].audit_summary["checks_failed"][0]

    def test_allow_publishers_rescues_evil_org(self, tmp_path: Path, evil_ckpt: Path) -> None:
        cfg = _write_config(
            tmp_path,
            {"audit_pinned_models": [{"path": str(evil_ckpt), "allow_publishers": ["evil-org"]}]},
        )
        report = audit_pinned(cfg, workspace=tmp_path)
        assert report.passed is True

    def test_relative_path_resolved_against_workspace(self, tmp_path: Path, clean_ckpt: Path) -> None:
        # Pinned path is stored relative to the config file's dir.
        cfg = _write_config(tmp_path, {"audit_pinned_models": [clean_ckpt.name]})
        report = audit_pinned(cfg, workspace=clean_ckpt.parent)
        assert report.passed is True
        assert report.findings[0].audit_passed is True


# ---------------------------------------------------------------------------
# Missing-path handling
# ---------------------------------------------------------------------------


class TestMissingPath:
    def test_missing_path_does_not_fail_by_default(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, {"audit_pinned_models": [str(tmp_path / "no-such-ckpt")]})
        report = audit_pinned(cfg, workspace=tmp_path)
        # passed is True because missing paths don't count toward
        # audit failure unless --fail-on-missing was set (CLI-level).
        assert report.passed is True
        assert report.findings[0].exists is False
        assert report.findings[0].error is not None


# ---------------------------------------------------------------------------
# Verify integration
# ---------------------------------------------------------------------------


class TestVerifyIntegration:
    def test_verify_fails_when_no_signature(self, tmp_path: Path, clean_ckpt: Path) -> None:
        # Clean checkpoint passes audit but has no signature sidecar →
        # verify_attempted=True, verify_passed=False, overall fails.
        cfg = _write_config(
            tmp_path,
            {"audit_pinned_models": [{"path": str(clean_ckpt), "verify": True}]},
        )
        report = audit_pinned(cfg, workspace=tmp_path)
        assert report.findings[0].audit_passed is True
        assert report.findings[0].verify_attempted is True
        assert report.findings[0].verify_passed is False
        assert report.passed is False

    def test_verify_passes_when_signed(self, tmp_path: Path, clean_ckpt: Path) -> None:
        from mind_mem.model_signing import generate_keypair, sign_model

        sk, _pk = generate_keypair()
        sign_model(clean_ckpt, sk, write_sidecars=True)
        cfg = _write_config(
            tmp_path,
            {"audit_pinned_models": [{"path": str(clean_ckpt), "verify": True}]},
        )
        report = audit_pinned(cfg, workspace=tmp_path)
        assert report.passed is True
        assert report.findings[0].verify_attempted is True
        assert report.findings[0].verify_passed is True


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_no_config_message(self, tmp_path: Path) -> None:
        report = audit_pinned(tmp_path / "nope.json", workspace=tmp_path)
        text = format_pinned_report_text(report)
        assert "no mind-mem.json found" in text

    def test_empty_list_message(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, {"audit_pinned_models": []})
        report = audit_pinned(cfg, workspace=tmp_path)
        text = format_pinned_report_text(report)
        assert "no models pinned" in text

    def test_pass_includes_overall(self, tmp_path: Path, clean_ckpt: Path) -> None:
        cfg = _write_config(tmp_path, {"audit_pinned_models": [str(clean_ckpt)]})
        report = audit_pinned(cfg, workspace=tmp_path)
        text = format_pinned_report_text(report)
        assert "PASS" in text
        assert "overall: PASS" in text

    def test_fail_includes_failed_check(self, tmp_path: Path, evil_ckpt: Path) -> None:
        cfg = _write_config(tmp_path, {"audit_pinned_models": [str(evil_ckpt)]})
        report = audit_pinned(cfg, workspace=tmp_path)
        text = format_pinned_report_text(report)
        assert "FAIL" in text
        assert "failed check:" in text
        assert "overall: FAIL" in text


# ---------------------------------------------------------------------------
# Report serialization (CLI --json mode)
# ---------------------------------------------------------------------------


class TestReportToDict:
    def test_round_trip(self, tmp_path: Path, clean_ckpt: Path) -> None:
        cfg = _write_config(tmp_path, {"audit_pinned_models": [str(clean_ckpt)]})
        report = audit_pinned(cfg, workspace=tmp_path)
        d = report.to_dict()
        # Must be JSON-serialisable for the --json output path.
        s = json.dumps(d)
        loaded = json.loads(s)
        assert loaded["passed"] is True
        assert len(loaded["findings"]) == 1
        assert loaded["findings"][0]["audit_passed"] is True
