# Copyright 2026 STARGA, Inc.
"""RE.1/RE.2 controls for the local served-ledger receipt adapter."""

from __future__ import annotations

import base64
import io
import json
import sys
import threading
from pathlib import Path

import pytest

import mind_mem.mm_cli as mm_cli
import mind_mem.retrieval_receipts as receipts
from mind_mem.recall_digests import served_set_digest
from mind_mem.served_ledger import append_served_run


def _workspace(tmp_path: Path) -> Path:
    (tmp_path / ".mind-mem-ledger").mkdir()
    (tmp_path / "mind-mem.json").write_text(json.dumps({"served_ledger": {"enabled": True}}), encoding="utf-8")
    return tmp_path


def _append(ws: Path, *, v2: bool = False) -> dict:
    ids = ("D-20260914-001",)
    kwargs = {
        "query_hash": "1" * 64,
        "served_digest": served_set_digest(ids),
        "ids": ids,
        "pipeline_hash": "2" * 64,
        "index_anchor": "3" * 64,
        "scoring_instant": "2026-09-14",
    }
    if v2:
        kwargs.update({"serve_kind": "attested", "context_digest": "4" * 64})
    row = append_served_run(ws, **kwargs)
    assert row is not None
    return row.to_row()


def _package_dict(payload: bytes) -> dict:
    return json.loads(payload.decode("utf-8"))


class TestReceiptRoundTrip:
    def test_nonempty_v1_v2_snapshot_is_locally_consistent(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        _append(ws)
        _append(ws, v2=True)
        before = (ws / ".mind-mem-ledger" / "served.jsonl").read_bytes()

        payload = receipts.export_receipt(ws)
        report = receipts.verify_receipt(payload)

        assert report["status"] == "locally_consistent"
        assert report["rows_checked"] == 2
        assert report["checks"] == {
            "schema": True,
            "profile": True,
            "manifest_shape": True,
            "manifest_scope": True,
            "manifest": True,
            "payload_encoding": True,
            "ledger_digest": True,
            "head_digest": True,
            "rows": True,
            "local_chain": True,
        }
        assert (ws / ".mind-mem-ledger" / "served.jsonl").read_bytes() == before
        assert report["issuer_trust"] == "unknown"
        assert report["external_anchor"] == "unknown"

    def test_repeated_snapshot_has_identical_bytes(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        _append(ws)
        assert receipts.export_receipt(ws) == receipts.export_receipt(ws)

    def test_missing_or_empty_ledger_is_not_an_empty_proof(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        with pytest.raises(receipts.ReceiptUnavailable):
            receipts.export_receipt(ws)
        (ws / ".mind-mem-ledger" / "served.jsonl").write_bytes(b"")
        with pytest.raises(receipts.ReceiptUnavailable):
            receipts.export_receipt(ws)

        no_parent = tmp_path / "no-ledger-parent"
        (no_parent / "mind-mem.json").parent.mkdir()
        (no_parent / "mind-mem.json").write_text(json.dumps({"served_ledger": {"enabled": True}}), encoding="utf-8")
        with pytest.raises(receipts.ReceiptUnavailable, match="directory is absent"):
            receipts.export_receipt(no_parent)


class TestReceiptIntegrity:
    def test_changed_row_is_rejected_even_when_package_digest_is_updated(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        _append(ws)
        package = _package_dict(receipts.export_receipt(ws))
        raw = base64.b64decode(package["ledger_b64"])
        altered_raw = raw.replace(b'"seq":0', b'"seq":7')
        package["ledger_b64"] = base64.b64encode(altered_raw).decode("ascii")
        package["manifest"]["ledger_sha256"] = receipts._sha256(altered_raw)
        package["manifest"]["ledger_bytes"] = len(altered_raw)
        package["manifest_sha256"] = receipts._sha256(receipts._package_bytes(receipts._manifest_payload(package)))
        report = receipts.verify_receipt(package)
        assert report["status"] == "integrity_failed"
        assert report["checks"]["local_chain"] is False

    def test_truncation_and_missing_link_are_rejected(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        _append(ws)
        _append(ws, v2=True)
        package = _package_dict(receipts.export_receipt(ws))
        raw = base64.b64decode(package["ledger_b64"])
        package["ledger_b64"] = base64.b64encode(raw.splitlines(keepends=True)[0]).decode("ascii")
        assert receipts.verify_receipt(package)["status"] == "integrity_failed"

        package = _package_dict(receipts.export_receipt(ws))
        lines = base64.b64decode(package["ledger_b64"]).splitlines()
        second = json.loads(lines[1])
        second["prev_row_hash"] = "f" * 64
        lines[1] = json.dumps(second, sort_keys=True, separators=(",", ":")).encode()
        package["ledger_b64"] = base64.b64encode(b"\n".join(lines) + b"\n").decode("ascii")
        package["manifest_sha256"] = receipts._sha256(receipts._package_bytes(receipts._manifest_payload(package)))
        assert receipts.verify_receipt(package)["status"] == "integrity_failed"

    def test_external_retained_manifest_detects_rewritten_package(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        _append(ws)
        package = receipts.export_receipt(ws)
        manifest_hash = _package_dict(package)["manifest_sha256"]
        altered = _package_dict(package)
        altered["head_b64"] = base64.b64encode(b"0" * 64 + b"\n").decode("ascii")
        report = receipts.verify_receipt(altered, expected_manifest_sha256=manifest_hash)
        assert report["status"] == "integrity_failed"
        assert report["checks"]["retained_manifest"] is False

    def test_unsupported_and_malformed_packages_fail_closed(self) -> None:
        assert receipts.verify_receipt(b"{}\n")["status"] == "malformed"
        assert receipts.verify_receipt(json.dumps({"schema": "other"}))["status"] == "malformed"
        assert (
            receipts.verify_receipt(
                json.dumps(
                    {
                        "schema": "other",
                        "profile": "x",
                        "manifest": {},
                        "ledger_b64": "",
                        "head_b64": "",
                        "manifest_sha256": "0" * 64,
                    }
                )
            )["status"]
            == "unsupported"
        )

    def test_ledger_byte_and_row_bounds_refuse_capture(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        _append(ws)
        with pytest.raises(receipts.ReceiptUnavailable, match="snapshot limit"):
            receipts.export_receipt(ws, max_bytes=1)


class TestReceiptConcurrency:
    def test_append_waits_for_capture_and_does_not_change_captured_package(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _workspace(tmp_path)
        _append(ws)
        captured = threading.Event()
        release = threading.Event()
        original = receipts._read_stable
        calls = 0

        def hooked(path: Path, *, limit: int, optional: bool = False):
            nonlocal calls
            result = original(path, limit=limit, optional=optional)
            if path.name == "served.jsonl":
                calls += 1
                captured.set()
                assert release.wait(timeout=2)
            return result

        monkeypatch.setattr(receipts, "_read_stable", hooked)
        result: list[bytes] = []
        exporter = threading.Thread(target=lambda: result.append(receipts.export_receipt(ws)), daemon=True)
        exporter.start()
        assert captured.wait(timeout=2)
        worker = threading.Thread(target=lambda: _append(ws), daemon=True)
        worker.start()
        release.set()
        exporter.join(timeout=2)
        worker.join(timeout=2)
        assert result
        first = result[0]
        assert not exporter.is_alive()
        assert not worker.is_alive()
        assert calls == 1
        assert receipts.verify_receipt(first)["status"] == "locally_consistent"
        assert len(receipts.verify_receipt(receipts.export_receipt(ws))["checks"]) > 0


class TestReceiptCLI:
    def test_canonical_vector_and_schema_are_committed(self) -> None:
        root = Path(__file__).parents[1]
        schema = json.loads((root / "docs/specs/retrieval-receipt-local-v1.json").read_text(encoding="utf-8"))
        vector = json.loads((root / "docs/specs/retrieval-receipt-canonical-vectors.json").read_text(encoding="utf-8"))
        assert schema["properties"]["schema"]["const"] == receipts.RECEIPT_SCHEMA
        assert schema["properties"]["profile"]["const"] == vector["profile"] == "local-served-ledger-v1"
        canonical = receipts._canonical(vector["canonicalization"]["input"])
        assert canonical.decode("utf-8") == vector["canonicalization"]["output_utf8"]
        assert receipts._sha256(canonical) == vector["canonicalization"]["sha256"]

    def test_help_exposes_both_receipt_commands(self) -> None:
        parser = mm_cli.build_parser()
        receipt = parser._subparsers._group_actions[0].choices["receipt"]
        assert "export" in receipt._subparsers._group_actions[0].choices
        assert "verify" in receipt._subparsers._group_actions[0].choices

    def test_actual_export_verify_and_corruption_commands(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        ws = _workspace(tmp_path)
        _append(ws)
        output = tmp_path / "receipt.json"
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ws))
        monkeypatch.setattr(mm_cli, "_run_auto_update_hook", lambda args: None)

        assert mm_cli.main(["receipt", "export", "--out", str(output)]) == 0
        exported = json.loads(capsys.readouterr().out)
        assert exported == {"bytes": output.stat().st_size, "schema": receipts.RECEIPT_SCHEMA, "status": "exported"}
        assert mm_cli.main(["receipt", "verify", "--input", str(output)]) == 0
        assert json.loads(capsys.readouterr().out)["status"] == "locally_consistent"

        corrupted = json.loads(output.read_text(encoding="utf-8"))
        corrupted["head_b64"] = base64.b64encode(b"0" * 64 + b"\n").decode("ascii")
        output.write_text(json.dumps(corrupted), encoding="utf-8")
        assert mm_cli.main(["receipt", "verify", "--input", str(output)]) == 1
        assert json.loads(capsys.readouterr().out)["status"] == "integrity_failed"

    def test_stdin_duplicate_keys_and_output_no_clobber(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        ws = _workspace(tmp_path)
        _append(ws)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ws))
        monkeypatch.setattr(mm_cli, "_run_auto_update_hook", lambda args: None)
        monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b'{"schema":1,"schema":2}\n'), encoding="utf-8"))
        assert mm_cli.main(["receipt", "verify", "--stdin"]) == 1
        assert json.loads(capsys.readouterr().out)["status"] == "malformed"

        existing = tmp_path / "existing.json"
        existing.write_bytes(b"keep")
        assert mm_cli.main(["receipt", "export", "--out", str(existing)]) == 1
        assert existing.read_bytes() == b"keep"
        assert mm_cli.main(["receipt", "export", "--out", str(ws / ".mind-mem-ledger" / "served.jsonl")]) == 1
        assert (ws / ".mind-mem-ledger" / "served.jsonl").read_bytes()

    def test_verify_rejects_symlink_input_and_custom_bounds(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        _append(ws)
        package = tmp_path / "package.json"
        package.write_bytes(receipts.export_receipt(ws))
        link = tmp_path / "package-link.json"
        link.symlink_to(package)
        assert receipts.verify_receipt(link)["status"] == "unavailable"
        assert receipts.verify_receipt(package.read_bytes(), max_package_bytes=1)["status"] == "unavailable"
        assert receipts.verify_receipt(package.read_bytes(), max_package_bytes=False)["status"] == "malformed"
        assert (
            receipts.verify_receipt(package.read_bytes(), max_package_bytes=receipts.DEFAULT_MAX_PACKAGE_BYTES + 1)["status"] == "malformed"
        )
        with pytest.raises(ValueError, match="no greater"):
            receipts.export_receipt(ws, max_bytes=receipts.DEFAULT_MAX_LEDGER_BYTES + 1)

    def test_write_receipt_refuses_output_symlink(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        _append(ws)
        target = tmp_path / "target"
        target.write_bytes(b"keep")
        link = tmp_path / "output"
        link.symlink_to(target)
        with pytest.raises(receipts.ReceiptError, match="already exists"):
            receipts.write_receipt(ws, link)
        assert target.read_bytes() == b"keep"
