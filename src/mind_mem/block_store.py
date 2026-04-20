"""BlockStore abstraction — decouples block access from storage format.

Provides a Protocol-based interface for block CRUD operations, with
MarkdownBlockStore as the default implementation wrapping the existing
file-based Markdown parsing.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Optional, Protocol, runtime_checkable

from .block_parser import get_active, get_by_id, parse_file
from .observability import get_logger

_log = get_logger("block_store")


@runtime_checkable
class BlockStore(Protocol):
    """Protocol for block storage backends.

    v3.2.0 §1.4 renames ``list_files`` → ``list_blocks`` as part of
    the apply-engine routing work. ``list_files`` remains as a thin
    alias with a ``DeprecationWarning``; it will be removed in v4.0.
    """

    def get_all(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        """Return all blocks, optionally filtered to active only."""
        ...

    def get_by_id(self, block_id: str) -> Optional[dict[str, Any]]:
        """Return a single block by ID, or None if not found."""
        ...

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Search blocks by text query."""
        ...

    def list_blocks(self) -> list[str]:
        """Return the list of block-containing artifacts managed by this store.

        For filesystem-backed stores this is the set of .md files in
        the corpus. For database-backed stores this is the logical
        equivalent (e.g., table partitions or rowids). Callers should
        treat it as an opaque identifier list.
        """
        ...


class MarkdownBlockStore:
    """BlockStore backed by Markdown files on disk.

    Wraps the existing block_parser functions to provide a uniform interface.
    """

    def __init__(self, workspace: str, corpus_dirs: tuple[str, ...] | None = None):
        self._workspace = workspace
        if corpus_dirs is None:
            from .corpus_registry import CORPUS_DIRS

            corpus_dirs = CORPUS_DIRS
        self._corpus_dirs = corpus_dirs
        self._files: list[str] | None = None

    def _discover_files(self) -> list[str]:
        """Discover all .md files in corpus directories."""
        if self._files is not None:
            return self._files
        files: list[str] = []
        for d in self._corpus_dirs:
            dir_path = os.path.join(self._workspace, d)
            if os.path.isdir(dir_path):
                for fname in sorted(os.listdir(dir_path)):
                    if fname.endswith(".md"):
                        files.append(os.path.join(dir_path, fname))
        self._files = files
        return files

    def get_all(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        """Return all blocks from all corpus files.

        Args:
            active_only: If True, only return blocks with Status=active.
        """
        blocks: list[dict[str, Any]] = []
        for fpath in self._discover_files():
            parsed = parse_file(fpath)
            if active_only:
                parsed = get_active(parsed)
            blocks.extend(parsed)
        return blocks

    def get_by_id(self, block_id: str) -> Optional[dict[str, Any]]:
        """Return a single block by ID, or None if not found."""
        for fpath in self._discover_files():
            parsed = parse_file(fpath)
            result = get_by_id(parsed, block_id)
            if result:
                return result
        return None

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Simple substring search across all blocks.

        Args:
            query: Case-insensitive substring to match against block values.
            limit: Maximum number of results to return.
        """
        query_lower = query.lower()
        matches: list[dict[str, Any]] = []
        for block in self.get_all():
            text = " ".join(str(v) for v in block.values()).lower()
            if query_lower in text:
                matches.append(block)
                if len(matches) >= limit:
                    break
        return matches

    def list_blocks(self) -> list[str]:
        """Return list of corpus .md file paths managed by this store.

        v3.2.0 §1.4: renamed from ``list_files``. The old name is
        preserved as a deprecation shim on both this class and every
        wrapping store (``EncryptedBlockStore``) — callers migrating
        from v3.1.x should switch to ``list_blocks`` at their
        convenience; the shim stays through v3.2.x.
        """
        return list(self._discover_files())

    def list_files(self) -> list[str]:
        """Deprecated alias for :meth:`list_blocks` — removed in v4.0."""
        warnings.warn(
            "BlockStore.list_files() is deprecated; use list_blocks() instead. "
            "The alias will be removed in v4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.list_blocks()

    def invalidate_cache(self) -> None:
        """Clear file discovery cache (call after corpus changes)."""
        self._files = None
