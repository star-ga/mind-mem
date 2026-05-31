"""Tests for ``mind_mem.mcp.tools.model`` — MCP wrappers for audit / sign / verify."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from mind_mem.mcp.tools.model import (
    audit_model_tool,
    sign_model_tool,
    verify_model_tool,
)


@pytest.fixture
def tiny_ckpt(tmp_path: Path) -> Path:
    """Two-file synthetic checkpoint with a real-shaped safetensors header."""
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3","base_model":"Qwen/Qwen3-8B"}')
    body = b'{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(body)) + body + b"\x00" * 8)
    return root


# ---------------------------------------------------------------------------
# audit_model_tool
# ---------------------------------------------------------------------------


class TestAuditModelTool:
    def test_clean_checkpoint_passes(self, tiny_ckpt: Path) -> None:
        result = json.loads(audit_model_tool(str(tiny_ckpt)))
        assert result["ok"] is True
        assert result["model_path"] == str(tiny_ckpt)
        # Provenance must be in the check stream because audit_model
        # appends it as the seventh check.
        names = [c["name"] for c in result["checks"]]
        assert "provenance" in names

    def test_manifest_omitted_by_default(self, tiny_ckpt: Path) -> None:
        result = json.loads(audit_model_tool(str(tiny_ckpt)))
        # Default include_manifest=False — keeps payload small for big checkpoints.
        assert "manifest" not in result

    def test_manifest_included_when_requested(self, tiny_ckpt: Path) -> None:
        result = json.loads(audit_model_tool(str(tiny_ckpt), include_manifest=True))
        assert "manifest" in result
        # Manifest is a {relpath -> sha256} mapping.
        for path_, digest in result["manifest"].items():
            assert isinstance(path_, str)
            assert len(digest) == 64

    def test_unknown_publisher_fails_provenance(self, tiny_ckpt: Path) -> None:
        (tiny_ckpt / "config.json").write_text('{"base_model":"evil-org/x"}')
        result = json.loads(audit_model_tool(str(tiny_ckpt)))
        prov = next(c for c in result["checks"] if c["name"] == "provenance")
        assert prov["passed"] is False
        assert result["ok"] is False

    def test_allow_publisher_extends_allowlist(self, tiny_ckpt: Path) -> None:
        (tiny_ckpt / "config.json").write_text('{"base_model":"internal-org/x"}')
        result = json.loads(audit_model_tool(str(tiny_ckpt), allow_publisher=["internal-org"]))
        prov = next(c for c in result["checks"] if c["name"] == "provenance")
        assert prov["passed"] is True

    def test_empty_path_rejected(self) -> None:
        result = json.loads(audit_model_tool(""))
        assert "error" in result
        assert "non-empty" in result["error"]

    def test_nul_byte_path_rejected(self) -> None:
        result = json.loads(audit_model_tool("/tmp/has\x00nul"))
        assert "error" in result
        assert "NUL" in result["error"]

    def test_missing_path_returns_error(self, tmp_path: Path) -> None:
        result = json.loads(audit_model_tool(str(tmp_path / "nope")))
        assert result["ok"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# sign_model_tool
# ---------------------------------------------------------------------------


class TestSignModelTool:
    def test_generate_key_writes_sidecars(self, tiny_ckpt: Path, tmp_path: Path) -> None:
        prefix = str(tmp_path / "k")
        result = json.loads(sign_model_tool(str(tiny_ckpt), generate_key_prefix=prefix))
        assert result["ok"] is True
        assert (tmp_path / "k.sk").is_file()
        assert (tmp_path / "k.pub").is_file()
        assert len(result["signature_hex"]) == 128  # 64 bytes hex
        assert len(result["public_key_hex"]) == 64  # 32 bytes hex
        assert (tiny_ckpt / "MODEL_MANIFEST.txt").is_file()
        assert (tiny_ckpt / "MODEL_MANIFEST.txt.sig").is_file()
        assert (tiny_ckpt / "MODEL_PUBKEY.pub").is_file()

    def test_key_file_signs_and_can_be_reused(self, tiny_ckpt: Path, tmp_path: Path) -> None:
        # First call: generate-key path produces a usable .sk on disk.
        prefix = str(tmp_path / "kk")
        first = json.loads(sign_model_tool(str(tiny_ckpt), generate_key_prefix=prefix))
        assert first["ok"] is True
        # Second call: re-sign with the saved .sk — manifest_sha256 should
        # be identical because the file set didn't change (sidecars are
        # skipped).
        second = json.loads(sign_model_tool(str(tiny_ckpt), key_file=str(tmp_path / "kk.sk")))
        assert second["ok"] is True
        assert second["manifest_sha256"] == first["manifest_sha256"]

    def test_no_sidecars_skips_disk_writes(self, tiny_ckpt: Path, tmp_path: Path) -> None:
        prefix = str(tmp_path / "k2")
        result = json.loads(
            sign_model_tool(
                str(tiny_ckpt),
                generate_key_prefix=prefix,
                write_sidecars=False,
            )
        )
        assert result["ok"] is True
        assert result["manifest_path"] is None
        assert not (tiny_ckpt / "MODEL_MANIFEST.txt").exists()

    def test_no_key_source_rejected(self, tiny_ckpt: Path) -> None:
        result = json.loads(sign_model_tool(str(tiny_ckpt)))
        assert result["ok"] is False
        assert "exactly one" in result["error"]

    def test_both_key_sources_rejected(self, tiny_ckpt: Path, tmp_path: Path) -> None:
        result = json.loads(
            sign_model_tool(
                str(tiny_ckpt),
                key_file=str(tmp_path / "x.sk"),
                generate_key_prefix=str(tmp_path / "y"),
            )
        )
        assert result["ok"] is False
        assert "exactly one" in result["error"]

    def test_missing_key_file_returns_error(self, tiny_ckpt: Path, tmp_path: Path) -> None:
        result = json.loads(sign_model_tool(str(tiny_ckpt), key_file=str(tmp_path / "missing.sk")))
        assert result["ok"] is False
        assert "not found" in result["error"]

    def test_bad_length_key_file_returns_error(self, tiny_ckpt: Path, tmp_path: Path) -> None:
        bad = tmp_path / "bad.sk"
        bad.write_bytes(b"\x00" * 16)  # 16, not 32
        result = json.loads(sign_model_tool(str(tiny_ckpt), key_file=str(bad)))
        assert result["ok"] is False
        assert "32 raw bytes" in result["error"]


# ---------------------------------------------------------------------------
# verify_model_tool
# ---------------------------------------------------------------------------


class TestVerifyModelTool:
    def test_signed_then_verified(self, tiny_ckpt: Path, tmp_path: Path) -> None:
        sign = json.loads(sign_model_tool(str(tiny_ckpt), generate_key_prefix=str(tmp_path / "k")))
        assert sign["ok"]
        result = json.loads(verify_model_tool(str(tiny_ckpt)))
        assert result["ok"] is True
        assert result["error_kind"] is None

    def test_tampered_file_triggers_manifest_mismatch(self, tiny_ckpt: Path, tmp_path: Path) -> None:
        json.loads(sign_model_tool(str(tiny_ckpt), generate_key_prefix=str(tmp_path / "k")))
        (tiny_ckpt / "config.json").write_text('{"model_type":"BACKDOORED"}')
        result = json.loads(verify_model_tool(str(tiny_ckpt)))
        assert result["ok"] is False
        assert result["error_kind"] == "manifest_mismatch"

    def test_explicit_pubkey_path(self, tiny_ckpt: Path, tmp_path: Path) -> None:
        json.loads(sign_model_tool(str(tiny_ckpt), generate_key_prefix=str(tmp_path / "k")))
        # Replace the sidecar pubkey with garbage but pass the real one
        # explicitly — verify must use the explicit key.
        (tiny_ckpt / "MODEL_PUBKEY.pub").write_bytes(b"\xff" * 32)
        result = json.loads(verify_model_tool(str(tiny_ckpt), pubkey_path=str(tmp_path / "k.pub")))
        assert result["ok"] is True

    def test_missing_pubkey_path_returns_error(self, tiny_ckpt: Path, tmp_path: Path) -> None:
        json.loads(sign_model_tool(str(tiny_ckpt), generate_key_prefix=str(tmp_path / "k")))
        result = json.loads(verify_model_tool(str(tiny_ckpt), pubkey_path=str(tmp_path / "missing.pub")))
        assert result["ok"] is False
        assert "not found" in result["error"]

    def test_missing_manifest_reports_missing_file(self, tiny_ckpt: Path) -> None:
        # Never signed — verify reports missing manifest cleanly.
        result = json.loads(verify_model_tool(str(tiny_ckpt)))
        assert result["ok"] is False
        assert result["error_kind"] == "missing_file"

    def test_empty_path_rejected(self) -> None:
        result = json.loads(verify_model_tool(""))
        assert "error" in result
        assert "non-empty" in result["error"]
