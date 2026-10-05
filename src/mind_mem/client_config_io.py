# Copyright 2026 STARGA, Inc.
"""Safe read / backup / atomic write for AI-client config files.

Client config files (``~/.cursor/mcp.json``, Zed's ``settings.json``,
``~/.claude/settings.json`` …) are the user's own files, often hand-edited
and often JSONC. The installer must never lose their content, so every
writer in :mod:`mind_mem.hook_installer` and :mod:`mind_mem.opencode_config`
goes through these helpers:

* :func:`read_json_object` distinguishes "absent or empty" (safe to create)
  from "present but not a strict-JSON object" (must be left untouched).
  Treating the second case as ``{}`` is what used to overwrite a commented
  or slightly malformed config with a file holding only the mind-mem entry.
* :func:`backup_file` writes a timestamped copy next to the file
  (``<file>.bak-mind-mem-YYYYmmdd-HHMMSS``) before it is changed.
* :func:`atomic_write_text` writes through a temp file and ``os.replace`` so
  a crash mid-write cannot leave a truncated config, and writes through a
  symlink (dotfile managers) instead of replacing the link.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from typing import Any

BACKUP_INFIX = ".bak-mind-mem-"


class UnsafeConfigError(Exception):
    """The file exists but cannot be merged without losing user content."""


def read_text_file(path: str) -> str | None:
    """Return the file's text, or ``None`` when it does not exist.

    Raises :class:`UnsafeConfigError` when the file exists but cannot be
    read — an unreadable file must never be treated as empty and then
    overwritten.
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise UnsafeConfigError(f"cannot read config: {exc}; left untouched") from exc


def read_json_object(path: str) -> dict[str, Any] | None:
    """Load *path* as a JSON object for merging.

    Returns ``None`` when the file is absent, ``{}`` when it is empty or
    whitespace-only (nothing to lose), and the parsed object otherwise.
    Raises :class:`UnsafeConfigError` when the file is JSONC (comments,
    trailing commas), not valid JSON, or not a JSON object at the root.
    """
    raw = read_text_file(path)
    if raw is None:
        return None
    if not raw.strip():
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise UnsafeConfigError(
            "config is JSONC (comments or trailing commas) or not valid JSON; left untouched — add the snippet in `content` by hand"
        ) from exc
    if not isinstance(loaded, dict):
        raise UnsafeConfigError("config root is not a JSON object; left untouched")
    return loaded


def backup_file(path: str) -> str:
    """Copy *path* to a unique timestamped sibling and return its path."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = f"{path}{BACKUP_INFIX}{stamp}"
    n = 1
    while os.path.exists(dest):
        dest = f"{path}{BACKUP_INFIX}{stamp}-{n}"
        n += 1
    shutil.copy2(path, dest)
    return dest


def atomic_write_text(path: str, text: str) -> None:
    """Replace *path* with *text* atomically, keeping its file mode."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".mind-mem-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        if os.path.isfile(path):
            shutil.copymode(path, tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_config(path: str, text: str) -> str | None:
    """Back up *path* if it exists, then write *text* atomically.

    Writes through a symlink to its target. Returns the backup path, or
    ``None`` when the file did not exist before.
    """
    target = os.path.realpath(path)
    backup = backup_file(target) if os.path.isfile(target) else None
    atomic_write_text(target, text)
    return backup


__all__ = [
    "BACKUP_INFIX",
    "UnsafeConfigError",
    "atomic_write_text",
    "backup_file",
    "read_json_object",
    "read_text_file",
    "write_config",
]
