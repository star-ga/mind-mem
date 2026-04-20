"""v3.2.0 §1.4 PR-6 — apply_engine routes through configured BlockStore."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    for d in (
        "decisions",
        "tasks",
        "entities",
        "intelligence",
        "intelligence/applied",
        "intelligence/proposed",
        "memory",
        "maintenance/tracked",
        "maintenance/append-only",
    ):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


def _write_config(ws: Path, config: dict) -> None:
    (ws / "mind-mem.json").write_text(json.dumps(config))


class TestBackendRouting:
    def test_default_uses_markdown_store(self, ws: Path) -> None:
        _write_config(ws, {})
        from mind_mem.apply_engine import _store_for
        from mind_mem.block_store import MarkdownBlockStore

        store = _store_for(str(ws))
        assert isinstance(store, MarkdownBlockStore)

    def test_explicit_markdown_backend(self, ws: Path) -> None:
        _write_config(ws, {"block_store": {"backend": "markdown"}})
        from mind_mem.apply_engine import _store_for
        from mind_mem.block_store import MarkdownBlockStore

        assert isinstance(_store_for(str(ws)), MarkdownBlockStore)

    def test_encrypted_backend_with_passphrase(self, ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIND_MEM_ENCRYPTION_PASSPHRASE", "test-pass-1234567890-abcdefgh")
        _write_config(ws, {"block_store": {"backend": "encrypted"}})

        from mind_mem.apply_engine import _store_for
        from mind_mem.block_store_encrypted import EncryptedBlockStore

        assert isinstance(_store_for(str(ws)), EncryptedBlockStore)

    def test_create_snapshot_delegates_to_store(self, ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Proves the apply_engine wrapper calls through the factory-resolved store."""
        _write_config(ws, {})
        fake_store = MagicMock()

        with patch("mind_mem.apply_engine._store_for", return_value=fake_store):
            from mind_mem.apply_engine import create_snapshot

            create_snapshot(str(ws), "20260420-000000")
            fake_store.snapshot.assert_called_once()
            # snap_dir is computed by apply_engine, not the store.
            # Normalise separators so the assertion holds on Windows
            # where os.path.join emits backslashes.
            args, kwargs = fake_store.snapshot.call_args
            snap_dir_posix = args[0].replace(os.sep, "/")
            assert snap_dir_posix.endswith("intelligence/applied/20260420-000000")

    def test_restore_snapshot_delegates_to_store(self, ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_config(ws, {})
        fake_store = MagicMock()

        with patch("mind_mem.apply_engine._store_for", return_value=fake_store):
            from mind_mem.apply_engine import restore_snapshot

            restore_snapshot(str(ws), "/path/to/snap")
            fake_store.restore.assert_called_once_with("/path/to/snap")

    def test_snapshot_diff_delegates_to_store(self, ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_config(ws, {})
        fake_store = MagicMock()
        fake_store.diff.return_value = ["decisions/DECISIONS.md"]

        with patch("mind_mem.apply_engine._store_for", return_value=fake_store):
            from mind_mem.apply_engine import snapshot_diff

            result = snapshot_diff(str(ws), "/path/to/snap")
            fake_store.diff.assert_called_once_with("/path/to/snap")
            assert result == ["decisions/DECISIONS.md"]

    def test_factory_failure_falls_back_to_markdown(self, ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A broken config must not crash apply_engine — falls back to MarkdownBlockStore."""
        # Write a malformed config that the factory will choke on.
        (ws / "mind-mem.json").write_text("{not valid json")

        from mind_mem.apply_engine import _store_for
        from mind_mem.block_store import MarkdownBlockStore

        store = _store_for(str(ws))
        assert isinstance(store, MarkdownBlockStore)
