"""Tests for mind_mem.storage.get_block_store factory (v3.2.0)."""

from __future__ import annotations

import json
import os

import pytest

from mind_mem.block_store import MarkdownBlockStore
from mind_mem.storage import get_block_store

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workspace(tmp_path, config: dict | None = None) -> str:
    """Return a tmp workspace path, writing mind-mem.json when *config* is given."""
    ws = str(tmp_path)
    if config is not None:
        with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
            json.dump(config, fh)
    return ws


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_returns_markdown(tmp_path):
    """No config at all → MarkdownBlockStore."""
    ws = _make_workspace(tmp_path)
    store = get_block_store(ws)
    assert isinstance(store, MarkdownBlockStore)


def test_explicit_markdown(tmp_path):
    """Explicit backend='markdown' → MarkdownBlockStore."""
    ws = _make_workspace(tmp_path)
    store = get_block_store(ws, config={"block_store": {"backend": "markdown"}})
    assert isinstance(store, MarkdownBlockStore)


def test_encrypted_without_passphrase_raises(tmp_path, monkeypatch):
    """backend='encrypted' without env var → ValueError."""
    monkeypatch.delenv("MIND_MEM_ENCRYPTION_PASSPHRASE", raising=False)
    ws = _make_workspace(tmp_path)
    with pytest.raises(ValueError, match="MIND_MEM_ENCRYPTION_PASSPHRASE"):
        get_block_store(ws, config={"block_store": {"backend": "encrypted"}})


def test_encrypted_with_passphrase_wraps_markdown(tmp_path, monkeypatch):
    """backend='encrypted' with env var → EncryptedBlockStore wrapping markdown."""
    pytest.importorskip("mind_mem.block_store_encrypted")
    monkeypatch.setenv("MIND_MEM_ENCRYPTION_PASSPHRASE", "test-secret-passphrase")
    ws = _make_workspace(tmp_path)

    from mind_mem.block_store_encrypted import EncryptedBlockStore

    store = get_block_store(ws, config={"block_store": {"backend": "encrypted"}})
    assert isinstance(store, EncryptedBlockStore)
    # Inner store must be the markdown implementation
    assert isinstance(store._inner, MarkdownBlockStore)


def test_postgres_raises_not_implemented(tmp_path):
    """backend='postgres' → NotImplementedError (ships in PR-5)."""
    ws = _make_workspace(tmp_path)
    with pytest.raises(NotImplementedError, match="PR-5"):
        get_block_store(ws, config={"block_store": {"backend": "postgres"}})


def test_unknown_backend_raises_value_error(tmp_path):
    """Unrecognised backend name → ValueError listing supported values."""
    ws = _make_workspace(tmp_path)
    with pytest.raises(ValueError, match="Unknown block_store.backend"):
        get_block_store(ws, config={"block_store": {"backend": "redis"}})


def test_reads_mind_mem_json_when_config_none(tmp_path):
    """config=None triggers auto-load from <workspace>/mind-mem.json."""
    ws = _make_workspace(tmp_path, config={"block_store": {"backend": "markdown"}})
    store = get_block_store(ws)  # no explicit config
    assert isinstance(store, MarkdownBlockStore)


def test_missing_mind_mem_json_falls_back_to_markdown(tmp_path):
    """Missing mind-mem.json with config=None → MarkdownBlockStore default."""
    ws = str(tmp_path)
    assert not os.path.exists(os.path.join(ws, "mind-mem.json"))
    store = get_block_store(ws)
    assert isinstance(store, MarkdownBlockStore)


def test_empty_block_store_section_is_markdown(tmp_path):
    """mind-mem.json with block_store:{} (no backend key) → MarkdownBlockStore."""
    ws = _make_workspace(tmp_path, config={"block_store": {}})
    store = get_block_store(ws)
    assert isinstance(store, MarkdownBlockStore)


def test_config_without_block_store_key_is_markdown(tmp_path):
    """mind-mem.json without a block_store key at all → MarkdownBlockStore."""
    ws = _make_workspace(tmp_path, config={"recall": {"backend": "bm25"}})
    store = get_block_store(ws)
    assert isinstance(store, MarkdownBlockStore)
