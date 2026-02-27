"""BlockStore abstraction — decouples block access from storage format.

Provides a Protocol-based interface for block CRUD operations, with
MarkdownBlockStore as the default implementation wrapping the existing
file-based Markdown parsing.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Protocol, runtime_checkable

from .block_parser import get_active, get_by_id, parse_file
from .observability import get_logger

_log = get_logger("block_store")


@runtime_checkable
class BlockStore(Protocol):
    """Protocol for block storage backends."""

    def get_all(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        """Return all blocks, optionally filtered to active only."""
        ...

    def get_by_id(self, block_id: str) -> Optional[dict[str, Any]]:
        """Return a single block by ID, or None if not found."""
        ...

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Search blocks by text query."""
        ...

    def list_files(self) -> list[str]:
        """Return list of corpus files managed by this store."""
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

    def list_files(self) -> list[str]:
        """Return list of corpus .md files managed by this store."""
        return list(self._discover_files())

    def invalidate_cache(self) -> None:
        """Clear file discovery cache (call after corpus changes)."""
        self._files = None
