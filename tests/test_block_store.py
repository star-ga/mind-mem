"""Tests for block_store.py — BlockStore protocol and MarkdownBlockStore."""

import os

import pytest

from mind_mem.block_store import BlockStore, MarkdownBlockStore

# ---------------------------------------------------------------------------
# Sample markdown content for tests
# ---------------------------------------------------------------------------

DECISIONS_MD = """\
[D-20260301-001]
Statement: Use JWT for authentication
Status: active
Priority: high

---

[D-20260301-002]
Statement: Prefer PostgreSQL over MySQL
Status: superseded
Priority: medium
"""

TASKS_MD = """\
[T-20260301-001]
Title: Implement login flow
Status: active
Assignee: alice

---

[T-20260301-002]
Title: Write unit tests for auth
Status: done
Assignee: bob
"""


@pytest.fixture()
def corpus_workspace(tmp_path):
    """Create a temporary workspace with sample corpus files."""
    ws = str(tmp_path / "workspace")
    os.makedirs(os.path.join(ws, "decisions"))
    os.makedirs(os.path.join(ws, "tasks"))

    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w") as f:
        f.write(DECISIONS_MD)

    with open(os.path.join(ws, "tasks", "TASKS.md"), "w") as f:
        f.write(TASKS_MD)

    return ws


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestBlockStoreProtocol:
    """Verify MarkdownBlockStore satisfies the BlockStore protocol."""

    def test_is_instance_of_protocol(self, corpus_workspace):
        store = MarkdownBlockStore(corpus_workspace, corpus_dirs=("decisions", "tasks"))
        assert isinstance(store, BlockStore)

    def test_protocol_is_runtime_checkable(self):
        assert hasattr(BlockStore, "__protocol_attrs__") or hasattr(BlockStore, "__abstractmethods__"), (
            "BlockStore should be a runtime-checkable Protocol"
        )


# ---------------------------------------------------------------------------
# get_all
# ---------------------------------------------------------------------------


class TestGetAll:
    def test_returns_all_blocks(self, corpus_workspace):
        store = MarkdownBlockStore(corpus_workspace, corpus_dirs=("decisions", "tasks"))
        blocks = store.get_all()
        ids = {b["_id"] for b in blocks}
        assert ids == {"D-20260301-001", "D-20260301-002", "T-20260301-001", "T-20260301-002"}

    def test_active_only_filters(self, corpus_workspace):
        store = MarkdownBlockStore(corpus_workspace, corpus_dirs=("decisions", "tasks"))
        blocks = store.get_all(active_only=True)
        ids = {b["_id"] for b in blocks}
        assert "D-20260301-001" in ids
        assert "T-20260301-001" in ids
        # superseded and done are not active
        assert "D-20260301-002" not in ids
        assert "T-20260301-002" not in ids

    def test_empty_workspace(self, tmp_path):
        ws = str(tmp_path / "empty")
        os.makedirs(ws)
        store = MarkdownBlockStore(ws, corpus_dirs=("decisions",))
        assert store.get_all() == []


# ---------------------------------------------------------------------------
# get_by_id
# ---------------------------------------------------------------------------


class TestGetById:
    def test_finds_existing_block(self, corpus_workspace):
        store = MarkdownBlockStore(corpus_workspace, corpus_dirs=("decisions", "tasks"))
        block = store.get_by_id("T-20260301-001")
        assert block is not None
        assert block["_id"] == "T-20260301-001"
        assert block["Title"] == "Implement login flow"

    def test_returns_none_for_missing(self, corpus_workspace):
        store = MarkdownBlockStore(corpus_workspace, corpus_dirs=("decisions", "tasks"))
        assert store.get_by_id("NONEXISTENT-999") is None


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------


class TestListFiles:
    def test_discovers_md_files(self, corpus_workspace):
        store = MarkdownBlockStore(corpus_workspace, corpus_dirs=("decisions", "tasks"))
        files = store.list_files()
        assert len(files) == 2
        basenames = {os.path.basename(f) for f in files}
        assert basenames == {"DECISIONS.md", "TASKS.md"}

    def test_ignores_non_md_files(self, corpus_workspace):
        # Add a non-md file — should not appear in list_files
        with open(os.path.join(corpus_workspace, "decisions", "README.txt"), "w") as f:
            f.write("not a markdown block file")
        store = MarkdownBlockStore(corpus_workspace, corpus_dirs=("decisions",))
        files = store.list_files()
        basenames = {os.path.basename(f) for f in files}
        assert "README.txt" not in basenames

    def test_missing_corpus_dir_is_skipped(self, corpus_workspace):
        store = MarkdownBlockStore(corpus_workspace, corpus_dirs=("decisions", "nonexistent"))
        files = store.list_files()
        # Only decisions dir exists
        assert len(files) == 1


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_finds_matching_blocks(self, corpus_workspace):
        store = MarkdownBlockStore(corpus_workspace, corpus_dirs=("decisions", "tasks"))
        results = store.search("JWT")
        assert len(results) == 1
        assert results[0]["_id"] == "D-20260301-001"

    def test_case_insensitive(self, corpus_workspace):
        store = MarkdownBlockStore(corpus_workspace, corpus_dirs=("decisions", "tasks"))
        results = store.search("jwt")
        assert len(results) == 1

    def test_respects_limit(self, corpus_workspace):
        store = MarkdownBlockStore(corpus_workspace, corpus_dirs=("decisions", "tasks"))
        # "active" appears in at least 2 blocks
        results = store.search("active", limit=1)
        assert len(results) == 1

    def test_no_match_returns_empty(self, corpus_workspace):
        store = MarkdownBlockStore(corpus_workspace, corpus_dirs=("decisions", "tasks"))
        results = store.search("zzz_no_such_content_zzz")
        assert results == []


# ---------------------------------------------------------------------------
# invalidate_cache
# ---------------------------------------------------------------------------


class TestInvalidateCache:
    def test_invalidate_clears_cached_files(self, corpus_workspace):
        store = MarkdownBlockStore(corpus_workspace, corpus_dirs=("decisions", "tasks"))
        # Populate the cache
        files_before = store.list_files()
        assert len(files_before) == 2

        # Add a new file
        with open(os.path.join(corpus_workspace, "decisions", "NEW.md"), "w") as f:
            f.write("[D-20260301-099]\nStatement: New decision\nStatus: active\n")

        # Cache is stale — still 2 files
        assert len(store.list_files()) == 2

        # Invalidate and re-discover
        store.invalidate_cache()
        files_after = store.list_files()
        assert len(files_after) == 3


# ---------------------------------------------------------------------------
# Default corpus_dirs
# ---------------------------------------------------------------------------


class TestDefaultCorpusDirs:
    def test_uses_corpus_registry_when_none(self, corpus_workspace):
        """When corpus_dirs is not specified, CORPUS_DIRS from corpus_registry is used."""
        store = MarkdownBlockStore(corpus_workspace)
        # Should not raise; corpus_dirs defaults from corpus_registry
        files = store.list_files()
        # Our workspace has decisions + tasks which overlap with default CORPUS_DIRS
        assert isinstance(files, list)
