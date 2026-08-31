"""Regression tests for VaultBridge.scan's sync_dirs handling.

A requested subdirectory that did not exist (a case typo, say) was
dropped without a word, so a forward sync of zero notes was
indistinguishable from an empty vault; an explicitly empty allowlist
meant "scan everything", the opposite of "sync nothing"; and an
unreadable note vanished from the sync silently.
"""

from __future__ import annotations

import builtins
import os
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from mind_mem.agent_bridge import VaultBridge


@pytest.fixture
def vault() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        root = Path(td)
        (root / "notes").mkdir()
        (root / "notes" / "a.md").write_text("---\nid: N-A\n---\n\nAlpha.\n", encoding="utf-8")
        (root / "notes" / "b.md").write_text("---\nid: N-B\n---\n\nBeta.\n", encoding="utf-8")
        yield root


class TestSyncDirs:
    def test_missing_sync_dir_is_an_error(self, vault: Path) -> None:
        """A named sync_dir that does not exist must not read as an empty vault."""
        bridge = VaultBridge(vault_root=str(vault))
        with pytest.raises(FileNotFoundError, match="sync_dir not found"):
            bridge.scan(sync_dirs=["definitely-not-a-directory"])

    def test_a_case_mismatched_sync_dir_is_an_error(self, vault: Path) -> None:
        """'Notes' is not 'notes' — where the filesystem agrees.

        Skipped on a case-INSENSITIVE filesystem, which macOS and Windows use by
        default: there ``Notes`` genuinely resolves to ``notes``, the scan
        legitimately succeeds, and asserting a raise tests the host rather than
        the product. Detected by probing the actual vault rather than by
        branching on sys.platform, since either filesystem can be mounted
        anywhere.
        """
        probe = vault / "notes"
        if not probe.is_dir():  # pragma: no cover - fixture guarantees it
            pytest.skip("fixture layout changed")
        if (vault / "NOTES").is_dir():
            pytest.skip("case-insensitive filesystem: 'Notes' and 'notes' are one directory")
        bridge = VaultBridge(vault_root=str(vault))
        with pytest.raises(FileNotFoundError, match="sync_dir not found"):
            bridge.scan(sync_dirs=["Notes"])

    def test_empty_allowlist_scans_nothing(self, vault: Path) -> None:
        bridge = VaultBridge(vault_root=str(vault))
        assert bridge.scan(sync_dirs=[]) == []

    def test_none_still_scans_the_whole_vault(self, vault: Path) -> None:
        bridge = VaultBridge(vault_root=str(vault))
        assert {b.block_id for b in bridge.scan()} == {"N-A", "N-B"}

    def test_existing_sync_dir_still_works(self, vault: Path) -> None:
        bridge = VaultBridge(vault_root=str(vault))
        assert {b.block_id for b in bridge.scan(sync_dirs=["notes"])} == {"N-A", "N-B"}

    def test_escaping_sync_dir_still_rejected(self, vault: Path) -> None:
        bridge = VaultBridge(vault_root=str(vault))
        with pytest.raises(ValueError, match="escapes vault root"):
            bridge.scan(sync_dirs=["../"])

    def test_unreadable_note_is_reported(self, vault: Path, monkeypatch) -> None:
        """The note is still skipped — but the operator hears about it."""
        from mind_mem import agent_bridge

        blocked = os.path.join(str(vault), "notes", "b.md")
        real_open = builtins.open

        def _open(path, *args, **kwargs):
            if str(path) == blocked:
                raise PermissionError(13, "Permission denied", blocked)
            return real_open(path, *args, **kwargs)

        warnings: list[tuple[str, dict]] = []

        class _Recorder:
            def warning(self, event: str, **kwargs) -> None:
                warnings.append((event, kwargs))

            def info(self, event: str, **kwargs) -> None:
                pass

        monkeypatch.setattr(builtins, "open", _open)
        monkeypatch.setattr(agent_bridge, "_log", _Recorder())
        blocks = VaultBridge(vault_root=str(vault)).scan()

        assert {b.block_id for b in blocks} == {"N-A"}
        assert [w for w in warnings if w[0] == "vault_note_unreadable"]
        assert warnings[0][1]["path"] == "notes/b.md"
