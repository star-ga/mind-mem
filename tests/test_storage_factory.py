"""Tests for mind_mem.storage.get_block_store factory (v3.2.0)."""

from __future__ import annotations

import contextlib
import json
import logging
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


def test_postgres_raises_value_error_without_dsn(tmp_path):
    """backend='postgres' without a dsn → ValueError (PR-5)."""
    ws = _make_workspace(tmp_path)
    with pytest.raises(ValueError, match="dsn"):
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


# v3.9: replicas validation lives in the factory now (was previously a
# silent ignore — replicas configured in mind-mem.json had no effect).


def test_postgres_replicas_must_be_list(tmp_path):
    """block_store.replicas must be a list — string value rejected."""
    ws = _make_workspace(tmp_path)
    with pytest.raises(ValueError, match="replicas must be a list"):
        get_block_store(
            ws,
            config={
                "block_store": {
                    "backend": "postgres",
                    "dsn": "postgresql://x@127.0.0.1:5432/x",
                    "replicas": "postgresql://r@host:5432/x",  # wrong shape
                }
            },
        )


def test_postgres_replicas_filtered_to_strings(tmp_path):
    """Non-string entries in the replicas list are silently dropped."""
    pytest.importorskip("psycopg")
    pytest.importorskip("psycopg_pool")
    ws = _make_workspace(tmp_path)
    # We can't actually connect, but we can verify the factory accepts
    # the shape without raising on the validation step. PostgresBlockStore
    # constructs lazily, so this only fails when a query is made.
    from mind_mem.block_store_postgres import PostgresBlockStore

    store = get_block_store(
        ws,
        config={
            "block_store": {
                "backend": "postgres",
                "dsn": "postgresql://x@127.0.0.1:5432/x",
                "replicas": [],  # empty list -> bare PostgresBlockStore, not Replicated
            }
        },
    )
    assert isinstance(store, PostgresBlockStore)


# ---------------------------------------------------------------------------
# A malformed config must be diagnosable, and must not mean two things
# ---------------------------------------------------------------------------


class _RecordingHandler(logging.Handler):
    """Capture records from the storage logger.

    ``observability.StructuredLogger`` sets ``propagate = False`` and owns
    its own stderr handler, so pytest's ``caplog`` (which hooks the root
    logger) never sees these records. Attach directly instead.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def events(self) -> str:
        return " ".join(r.getMessage() for r in self.records)


@contextlib.contextmanager
def _storage_warnings():
    logger = logging.getLogger("mind-mem.storage")
    handler = _RecordingHandler()
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


def test_corrupt_config_is_logged_not_silently_downgraded(tmp_path):
    """A corrupt mind-mem.json downgrades a Postgres workspace to Markdown.

    _load_workspace_config caught OSError/JSONDecodeError/UnicodeDecodeError
    with a bare `pass`, so a corrupt-or-unreadable config was
    indistinguishable from no config at all: get_block_store returned a
    MarkdownBlockStore and the team's decisions went to
    decisions/DECISIONS.md instead of the database, with no warning at
    any level.
    """
    from mind_mem.storage import _backend_name, _load_workspace_config

    # One trailing comma — valid-looking, unparseable.
    (tmp_path / "mind-mem.json").write_text('{"block_store": {"backend": "postgres", "dsn": "x"},}', encoding="utf-8")

    with _storage_warnings() as captured:
        assert _load_workspace_config(str(tmp_path)) == {}
        assert _backend_name(str(tmp_path)) == "markdown"
    assert "workspace_config_unreadable" in captured.events, f"downgrade was silent; captured: {captured.events!r}"


def test_config_that_is_not_an_object_is_logged(tmp_path):
    from mind_mem.storage import _load_workspace_config

    (tmp_path / "mind-mem.json").write_text("[1, 2, 3]", encoding="utf-8")
    with _storage_warnings() as captured:
        assert _load_workspace_config(str(tmp_path)) == {}
    assert "workspace_config_not_an_object" in captured.events


def test_valid_config_logs_nothing(tmp_path):
    """Anti-noise: a good config must not emit a warning."""
    from mind_mem.storage import _backend_name, _load_workspace_config

    (tmp_path / "mind-mem.json").write_text('{"block_store": {"backend": "markdown"}}', encoding="utf-8")
    with _storage_warnings() as captured:
        assert _load_workspace_config(str(tmp_path)) == {"block_store": {"backend": "markdown"}}
        assert _backend_name(str(tmp_path)) == "markdown"
    assert captured.records == []


def test_router_and_constructor_agree_on_a_malformed_block_store_section(tmp_path):
    """`{"block_store": "postgres"}` used to give two different wrong answers.

    _backend_name checked `isinstance(bs_cfg, dict)` and returned
    "markdown" (so reindex/governance quietly read the LOCAL corpus),
    while get_block_store called `.get` on the string and raised
    AttributeError — not the ValueError its docstring promises.
    """
    import pytest as _pytest

    from mind_mem.storage import _backend_name, get_block_store

    (tmp_path / "mind-mem.json").write_text('{"block_store": "postgres"}', encoding="utf-8")

    with _storage_warnings() as captured:
        with _pytest.raises(ValueError, match="block_store must be an object"):
            get_block_store(str(tmp_path))
        assert _backend_name(str(tmp_path)) == "markdown"
    assert "block_store_config_malformed" in captured.events


def test_non_string_backend_is_rejected_by_the_constructor(tmp_path):
    import pytest as _pytest

    from mind_mem.storage import _backend_name, get_block_store

    cfg = {"block_store": {"backend": 7}}
    with _pytest.raises(ValueError, match="backend must be a string"):
        get_block_store(str(tmp_path), config=cfg)
    assert _backend_name(str(tmp_path), config=cfg) == "markdown"


def test_absent_and_null_block_store_sections_still_default(tmp_path):
    """The zero-config and explicit-null paths are unaffected."""
    from mind_mem.block_store import MarkdownBlockStore
    from mind_mem.storage import _backend_name, get_block_store

    for cfg in ({}, {"block_store": None}, {"block_store": {}}):
        assert isinstance(get_block_store(str(tmp_path), config=cfg), MarkdownBlockStore)
        assert _backend_name(str(tmp_path), config=cfg) == "markdown"
