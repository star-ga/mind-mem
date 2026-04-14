# Copyright 2026 STARGA, Inc.
"""Tests for EncryptedBlockStore + workspace migration (v3.0.0 — GH #504)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from mind_mem.block_store import MarkdownBlockStore
from mind_mem.block_store_encrypted import (
    EncryptedBlockStore,
    encrypt_workspace,
    get_block_store,
)


_PASS = "test-passphrase-for-unit-tests"


def _make_block(bid: str) -> str:
    return (
        f"[{bid}]\n"
        f"Date: 2026-04-13\n"
        f"Status: active\n"
        f"Title: Example\n"
        f"Rationale: seed\n"
    )


@pytest.fixture
def seeded_workspace(tmp_path: Path) -> Path:
    decisions = tmp_path / "decisions"
    decisions.mkdir()
    (decisions / "DECISIONS.md").write_text(
        "\n".join(_make_block(f"D-2026041{i}-001") for i in range(3)) + "\n"
    )
    return tmp_path


class TestEncryptedBlockStoreConstructor:
    def test_rejects_empty_passphrase(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            EncryptedBlockStore(str(tmp_path), passphrase="")


class TestTransparentReadOnPlainFiles:
    def test_plain_files_pass_through(self, seeded_workspace: Path) -> None:
        store = EncryptedBlockStore(str(seeded_workspace), passphrase=_PASS)
        blocks = store.get_all()
        assert len(blocks) == 3
        ids = {b.get("_id") for b in blocks}
        assert "D-20260410-001" in ids

    def test_plain_files_searchable(self, seeded_workspace: Path) -> None:
        store = EncryptedBlockStore(str(seeded_workspace), passphrase=_PASS)
        results = store.search("example")
        assert len(results) >= 1


class TestTransparentReadOnEncryptedFiles:
    def test_encrypted_files_decrypt_on_read(
        self, seeded_workspace: Path
    ) -> None:
        # Encrypt the corpus in place
        os.environ["MIND_MEM_ENCRYPTION_PASSPHRASE"] = _PASS
        try:
            result = encrypt_workspace(str(seeded_workspace))
            assert result["encrypted"] >= 1

            # Open the on-disk file and confirm it IS encrypted (magic header)
            decision_file = seeded_workspace / "decisions" / "DECISIONS.md"
            head = decision_file.read_bytes()[:6]
            from mind_mem.encryption import _MAGIC
            assert head == _MAGIC

            # Plain store would fail to parse; encrypted store
            # decrypts transparently + returns the original blocks
            enc = EncryptedBlockStore(str(seeded_workspace), passphrase=_PASS)
            blocks = enc.get_all()
            assert len(blocks) == 3
        finally:
            os.environ.pop("MIND_MEM_ENCRYPTION_PASSPHRASE", None)


class TestEncryptWorkspaceMigration:
    def test_idempotent(self, seeded_workspace: Path) -> None:
        os.environ["MIND_MEM_ENCRYPTION_PASSPHRASE"] = _PASS
        try:
            r1 = encrypt_workspace(str(seeded_workspace))
            assert r1["encrypted"] >= 1
            r2 = encrypt_workspace(str(seeded_workspace))
            # Second pass: every file is already encrypted → skipped.
            assert r2["skipped"] == r1["encrypted"]
            assert r2["encrypted"] == 0
        finally:
            os.environ.pop("MIND_MEM_ENCRYPTION_PASSPHRASE", None)

    def test_missing_passphrase_raises(self, seeded_workspace: Path) -> None:
        os.environ.pop("MIND_MEM_ENCRYPTION_PASSPHRASE", None)
        with pytest.raises(RuntimeError):
            encrypt_workspace(str(seeded_workspace))


class TestFactory:
    def test_returns_plain_without_env(self, tmp_path: Path) -> None:
        os.environ.pop("MIND_MEM_ENCRYPTION_PASSPHRASE", None)
        store = get_block_store(str(tmp_path))
        assert isinstance(store, MarkdownBlockStore)

    def test_returns_encrypted_with_env(self, tmp_path: Path) -> None:
        os.environ["MIND_MEM_ENCRYPTION_PASSPHRASE"] = _PASS
        try:
            store = get_block_store(str(tmp_path))
            assert isinstance(store, EncryptedBlockStore)
        finally:
            os.environ.pop("MIND_MEM_ENCRYPTION_PASSPHRASE", None)
