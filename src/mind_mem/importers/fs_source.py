# Copyright 2026 STARGA, Inc.
"""Note-tree source loading — a directory of markdown notes to records.

The merged importers all took a single JSON dump. The formats agent
memory actually lives in on disk are *directories* of markdown, so this
module is the second source shape: walk a tree, split front matter from
body, and hand the parsers a deterministic tuple of :class:`SourceNote`.

Determinism
-----------
Notes come back sorted by POSIX-normalized relative path and carry **no
filesystem timestamps** — mtime is checkout state, not content, and an
import must be a pure function of the tree. Two clones of the same vault
therefore import to byte-identical blocks.

Boundaries
----------
The walk is bounded three ways (file count, per-note bytes, total bytes)
so a hostile or accidentally enormous tree cannot exhaust a small
runner. Oversized single notes are skipped and logged rather than
failing the whole import; blowing the file-count or total-byte ceiling
is an explicit :class:`ImportParseError`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from ..observability import get_logger
from .records import ImportParseError

_log = get_logger("importers.fs_source")

__all__ = [
    "DEFAULT_EXCLUDED_DIRS",
    "MAX_NOTE_BYTES",
    "MAX_TREE_BYTES",
    "MAX_TREE_FILES",
    "NOTE_EXTENSIONS",
    "SourceNote",
    "load_note_tree",
    "markdown_link_targets",
    "parse_front_matter",
    "wikilink_targets",
]

#: Extensions treated as notes. Everything else in the tree is ignored.
NOTE_EXTENSIONS: tuple[str, ...] = (".md", ".markdown")

#: Directories never descended into — editor state, trash, templates and
#: version-control metadata are not memory.
DEFAULT_EXCLUDED_DIRS: frozenset[str] = frozenset({".git", ".obsidian", ".trash", ".venv", "node_modules", "templates"})

MAX_NOTE_BYTES = 1024 * 1024
MAX_TREE_FILES = 5000
MAX_TREE_BYTES = 64 * 1024 * 1024

# ``[[target]]`` / ``[[target|alias]]`` / ``[[target#heading]]``. The
# target is bounded so a pathological line cannot produce a giant key.
_WIKILINK_RE = re.compile(r"\[\[([^\[\]|#\n]{1,128})(?:[|#][^\[\]\n]{0,256})?\]\]")

# ``[label](some/note.md)`` — the shape an index file uses when it links
# siblings by path instead of by wikilink.
_MD_LINK_RE = re.compile(r"\]\(\s*([^)\s#]{1,256}\.(?:md|markdown))\s*(?:#[^)\s]{0,128})?\)")

_FRONT_MATTER_RE = re.compile(r"^---[ \t]*\n(.*?)\n---[ \t]*\n?", re.DOTALL)

# Front matter is bounded the same way metadata is: this is a header,
# not a payload.
_MAX_FRONT_MATTER_KEYS = 32
_MAX_FRONT_MATTER_VALUE = 512


@dataclass(frozen=True)
class SourceNote:
    """One markdown note lifted off disk.

    Attributes:
        relative_path: POSIX-normalized path relative to the tree root.
            Stable across machines, which is what makes the derived
            block id stable.
        front_matter: Flat ``str -> str`` header fields. Nested blocks
            are flattened to dotted keys (``metadata.type``) so a
            nested value is still addressable and is never dropped.
        body: Note text with the front-matter header removed.
    """

    relative_path: str
    front_matter: Mapping[str, str]
    body: str

    @property
    def stem(self) -> str:
        """Filename without directories or extension."""
        name = self.relative_path.rsplit("/", 1)[-1]
        for suffix in NOTE_EXTENSIONS:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    @property
    def folder(self) -> str:
        """Parent directory relative to the root (``""`` at the root)."""
        head, sep, _ = self.relative_path.rpartition("/")
        return head if sep else ""


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split a ``---`` front-matter header from *text*.

    One level of nesting is flattened to dotted keys and ``- item``
    sequences are joined with commas, which covers the flat-ish headers
    notes actually carry.

    deferred: a full YAML parse (anchors, multi-line scalars, deep
    nesting) is out of scope — the dependency-free posture of the rest
    of the corpus tooling is worth more than the long tail.
    upgrade path: swap this function for a real YAML load behind an
    opt-in config flag, keeping this as the default.
    """
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    body = text[match.end() :]
    out: dict[str, str] = {}
    prefix = ""
    prefix_indent = 0
    pending_list_key = ""
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped.startswith("- "):
            if pending_list_key:
                item = stripped[2:].strip().strip('"').strip("'")
                existing = out.get(pending_list_key, "")
                joined = f"{existing},{item}" if existing else item
                out[pending_list_key] = joined[:_MAX_FRONT_MATTER_VALUE]
            continue
        if ":" not in stripped:
            continue
        raw_key, raw_value = stripped.split(":", 1)
        key = raw_key.strip()
        value = raw_value.strip().strip('"').strip("'")
        if prefix and indent <= prefix_indent:
            prefix = ""
            prefix_indent = 0
        if not value:
            prefix = f"{key}." if not prefix else f"{prefix}{key}."
            prefix_indent = indent
            pending_list_key = f"{prefix[:-1]}"
            continue
        pending_list_key = ""
        if len(out) >= _MAX_FRONT_MATTER_KEYS:
            continue
        out[f"{prefix}{key}"] = value[:_MAX_FRONT_MATTER_VALUE]
    return out, body


def wikilink_targets(text: str) -> tuple[str, ...]:
    """Ordered, de-duplicated ``[[target]]`` names in *text*."""
    return _ordered_unique(match.group(1).strip() for match in _WIKILINK_RE.finditer(text))


def markdown_link_targets(text: str) -> tuple[str, ...]:
    """Ordered, de-duplicated note *stems* linked as ``[label](note.md)``."""
    stems = []
    for match in _MD_LINK_RE.finditer(text):
        target = match.group(1).strip().rsplit("/", 1)[-1]
        for suffix in NOTE_EXTENSIONS:
            if target.endswith(suffix):
                target = target[: -len(suffix)]
                break
        stems.append(target)
    return _ordered_unique(stems)


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for value in values:
        if value and value not in seen:
            seen[value] = None
    return tuple(seen)


def _validate_root(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ImportParseError("note tree path must be a non-empty string")
    if not os.path.exists(path):
        raise ImportParseError(f"note directory not found: {path}")
    if not os.path.isdir(path):
        raise ImportParseError(f"note tree path is not a directory: {path} (this importer reads a directory of notes)")
    return os.path.realpath(path)


def load_note_tree(
    path: str,
    *,
    excludes: frozenset[str] = DEFAULT_EXCLUDED_DIRS,
) -> tuple[SourceNote, ...]:
    """Walk the note tree at *path* and return its notes, sorted by path.

    Raises:
        ImportParseError: the path is missing, is not a directory, holds
            more than :data:`MAX_TREE_FILES` notes, exceeds
            :data:`MAX_TREE_BYTES` in total, or contains a note that is
            not valid UTF-8.
    """
    root = _validate_root(path)
    collected: list[SourceNote] = []
    total_bytes = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in excludes and not d.startswith("."))
        for filename in sorted(filenames):
            if not filename.endswith(NOTE_EXTENSIONS):
                continue
            full = os.path.join(dirpath, filename)
            if not os.path.isfile(full):
                continue
            size = os.path.getsize(full)
            if size > MAX_NOTE_BYTES:
                _log.warning(
                    "import_note_skipped_oversize",
                    extra={"file": os.path.relpath(full, root), "bytes": size, "max_bytes": MAX_NOTE_BYTES},
                )
                continue
            total_bytes += size
            if total_bytes > MAX_TREE_BYTES:
                raise ImportParseError(f"note tree too large: over {MAX_TREE_BYTES} bytes of notes under {path}")
            if len(collected) >= MAX_TREE_FILES:
                raise ImportParseError(f"note tree has more than {MAX_TREE_FILES} notes: {path}")
            try:
                with open(full, "r", encoding="utf-8") as handle:
                    text = handle.read()
            except UnicodeDecodeError as exc:
                raise ImportParseError(f"note is not valid UTF-8: {os.path.relpath(full, root)}") from exc
            except OSError as exc:
                raise ImportParseError(f"cannot read note {os.path.relpath(full, root)}: {exc}") from exc
            front_matter, body = parse_front_matter(text.replace("\r\n", "\n").replace("\r", "\n"))
            collected.append(
                SourceNote(
                    relative_path=os.path.relpath(full, root).replace(os.sep, "/"),
                    front_matter=front_matter,
                    body=body,
                )
            )

    if not collected:
        raise ImportParseError(f"no markdown notes found under {path} (looked for {', '.join(NOTE_EXTENSIONS)})")
    return tuple(sorted(collected, key=lambda note: note.relative_path))
