"""Tests for mind_mem.protection (v3.3.0+)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from mind_mem import protection


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fake_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "mind_mem"
    root.mkdir()
    for rel in ("recall.py", "apply_engine.py"):
        (root / rel).write_text(f"# stub {rel}\n", encoding="utf-8")
    monkeypatch.setattr(protection, "_package_root", lambda: root)
    monkeypatch.setattr(protection, "_CRITICAL_MODULES", ("recall.py", "apply_engine.py"))
    return root


class TestNoManifest:
    def test_returns_ok_when_manifest_absent(self, fake_package: Path) -> None:
        report = protection.verify_integrity()
        assert report.ok is True
        assert report.manifest_present is False
        assert report.checked == 0
        assert report.mismatched == ()
        assert report.missing == ()


class TestWithManifest:
    def _write_manifest(self, root: Path, overrides: dict[str, str] | None = None) -> None:
        files = {rel: _sha256(root / rel) for rel in ("recall.py", "apply_engine.py")}
        if overrides:
            files.update(overrides)
        manifest = {"version": 1, "files": files}
        (root / protection._MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    def test_clean_package_passes(self, fake_package: Path) -> None:
        self._write_manifest(fake_package)
        report = protection.verify_integrity()
        assert report.ok is True
        assert report.manifest_present is True
        assert report.checked == 2
        assert report.mismatched == ()

    def test_detects_tamper(self, fake_package: Path) -> None:
        self._write_manifest(fake_package)
        (fake_package / "recall.py").write_text("# TAMPERED\n", encoding="utf-8")
        report = protection.verify_integrity()
        assert report.ok is False
        assert "recall.py" in report.mismatched

    def test_detects_missing_file(self, fake_package: Path) -> None:
        self._write_manifest(fake_package)
        (fake_package / "apply_engine.py").unlink()
        report = protection.verify_integrity()
        assert report.ok is False
        assert "apply_engine.py" in report.missing

    def test_malformed_manifest_returns_none(self, fake_package: Path) -> None:
        (fake_package / protection._MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")
        report = protection.verify_integrity()
        # Falls back to no-manifest behaviour — silent pass.
        assert report.ok is True
        assert report.manifest_present is False


class TestStrictMode:
    def test_strict_raises_on_tamper(
        self, fake_package: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        files = {rel: _sha256(fake_package / rel) for rel in ("recall.py", "apply_engine.py")}
        (fake_package / protection._MANIFEST_FILENAME).write_text(
            json.dumps({"version": 1, "files": files}),
            encoding="utf-8",
        )
        (fake_package / "recall.py").write_text("# tampered\n", encoding="utf-8")
        monkeypatch.setenv(protection._STRICT_ENV, "strict")
        with pytest.raises(RuntimeError, match="integrity check failed"):
            protection.verify_integrity()

    def test_strict_env_truthy_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for value in ("1", "true", "yes", "STRICT"):
            monkeypatch.setenv(protection._STRICT_ENV, value)
            assert protection._strict() is True

    def test_strict_env_falsy_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for value in ("", "0", "off", "no", "false"):
            monkeypatch.setenv(protection._STRICT_ENV, value)
            assert protection._strict() is False


class TestConstants:
    def test_author_is_starga(self) -> None:
        assert "STARGA" in protection.__author__

    def test_license_is_apache(self) -> None:
        assert protection.__license__ == "Apache-2.0"

    def test_auth_header_stable(self) -> None:
        # Downstream consumers pin on this exact string; changing it
        # is a breaking change that needs a major version bump.
        assert protection.AUTH_HEADER == "X-MindMem-Token"

    def test_audit_tag_stable(self) -> None:
        assert protection.AUDIT_TAG == "TAG_v1"


class TestWorldWritable:
    def test_non_posix_returns_false(self, tmp_path: Path) -> None:
        with patch.object(protection.os, "name", "nt"):
            assert protection._world_writable(tmp_path) is False

    @pytest.mark.skipif(os.name != "posix", reason="POSIX-only")
    def test_detects_world_writable(self, tmp_path: Path) -> None:
        import stat as _stat

        bad = tmp_path / "bad"
        bad.mkdir()
        bad.chmod(0o777)
        try:
            assert protection._world_writable(bad) is True
        finally:
            bad.chmod(0o755)

    @pytest.mark.skipif(os.name != "posix", reason="POSIX-only")
    def test_normal_mode_is_safe(self, tmp_path: Path) -> None:
        assert protection._world_writable(tmp_path) is False


class TestReportImmutability:
    def test_report_is_frozen(self) -> None:
        report = protection.IntegrityReport(
            ok=True,
            mode="fail-open",
            manifest_present=False,
            checked=0,
        )
        with pytest.raises((AttributeError, Exception)):
            report.ok = False  # type: ignore[misc]
