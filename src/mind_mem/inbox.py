"""Inbox folder ingestion — `mm inbox-watch` (v3.9.0 candidate).

Drop any file into a configured ``inbox/`` directory and mind-mem
classifies it by extension and routes to the right ingestion path.
The text path is stdlib-only. The multimodal paths are NOT: image and
audio are routed and then refused — no embedder or transcriber ships,
and there is no extra to install that adds one — while PDF works if
``pypdf`` is importable, which no mind-mem extra installs either. There
is no ``mind-mem[multimodal]`` extra; ``pyproject.toml`` declares no such
extra, so telling an operator to install one sends them to a pip
warning and a no-op. The handlers below name the package that is
actually missing instead.

Routing rules (file extension → handler)::

    .txt .md .json .csv .log .xml .yaml .yml  → text → markdown block
    .png .jpg .jpeg .gif .webp                → image  (not implemented)
    .mp3 .wav .flac .m4a                      → audio  (not implemented)
    .pdf                                      → text extract  (needs pypdf)

Files are processed atomically — moved to ``inbox/_processed/<ts>/``
on success or ``inbox/_failed/<ts>/`` (with a sidecar ``.error.txt``)
on failure. The ``inbox/`` root and the two staging directories are
created if they don't exist.

Usage::

    from mind_mem.inbox import InboxWatcher

    watcher = InboxWatcher(workspace="/path/ws", inbox="/path/inbox")
    watcher.start()
    # ... later ...
    watcher.stop()
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from .codepoint_sanitize import sanitize_text_for_ingest
from .enums import IngestTier
from .importers.quarantine import QUARANTINE_STATUS, QUARANTINE_TIER, TIER_FIELD

__all__ = [
    "ROUTING_TABLE",
    "InboxWatcher",
    "IngestResult",
    "classify_file",
    "ingest_text_file",
]

_log = logging.getLogger("mind_mem.inbox")

# ---------------------------------------------------------------------------
# Routing table — declarative; lazy-imports keep heavy deps optional.
# ---------------------------------------------------------------------------

# (extension → handler-name). Handlers live below. The multimodal
# handlers raise NotImplementedError naming what is actually missing;
# they must not name a `mind-mem[...]` extra, because none of them has
# one — see the module docstring.
ROUTING_TABLE: dict[str, str] = {
    # Text
    ".txt": "text",
    ".md": "text",
    ".json": "text",
    ".csv": "text",
    ".log": "text",
    ".xml": "text",
    ".yaml": "text",
    ".yml": "text",
    # Image (multimodal; routed, then refused — no embedder ships)
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
    # Audio (multimodal; routed, then refused — no transcriber ships)
    ".mp3": "audio",
    ".wav": "audio",
    ".flac": "audio",
    ".m4a": "audio",
    # Document (multimodal; works only when pypdf is importable)
    ".pdf": "pdf",
}

# Files that match these patterns at the inbox root are ignored (staging dirs,
# hidden files, OS metadata).
_IGNORE_BASENAME_PREFIXES: tuple[str, ...] = (".", "_processed", "_failed")
_INGEST_TEXT_BYTES = 4 * 1024 * 1024  # 4 MiB cap for inbox text files


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestResult:
    path: str
    handler: str
    ok: bool
    block_id: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_file(path: str) -> str | None:
    """Return the handler name for *path*, or ``None`` if unsupported."""
    ext = os.path.splitext(path)[1].lower()
    return ROUTING_TABLE.get(ext)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def ingest_text_file(workspace: str, file_path: str) -> str:
    """Read *file_path* as UTF-8 text and write a markdown block.

    Returns the new block id. Raises on failure (caller routes to the
    ``_failed/`` staging directory).
    """
    size = os.path.getsize(file_path)
    if size > _INGEST_TEXT_BYTES:
        raise ValueError(f"text file too large for inbox ingestion: {size} bytes (max {_INGEST_TEXT_BYTES})")
    with open(file_path, encoding="utf-8", errors="replace") as fh:
        content = fh.read()

    # Security: strip invisible-Unicode (zero-width, tag chars, bidi
    # controls) before the text becomes a block — prompt-injection
    # channel. Config-gated, default ON (ingest.sanitize_codepoints).
    content = sanitize_text_for_ingest(content, workspace, source=file_path)

    base = os.path.splitext(os.path.basename(file_path))[0]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_base = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base)[:64] or "inbox"
    block_id = f"INBOX-{ts}-{safe_base}"

    block = {
        "_id": block_id,
        "type": "INBOX_DOCUMENT",
        "Subject": f"Inbox: {os.path.basename(file_path)}",
        "Statement": content,
        "Source": os.path.basename(file_path),
        "Timestamp": ts,
        # The inbox is a DROP FOLDER: whatever lands in it is untrusted input,
        # exactly like an imported corpus. It arrives quarantined and stays
        # invisible to recall until a governance proposal releases it. Before
        # this, attacker-supplied file text landed Status: active and
        # immediately recallable, with no proposal and no chain entry -- the
        # same injection primitive the importer quarantine was built to close.
        "Status": QUARANTINE_STATUS,
        TIER_FIELD: QUARANTINE_TIER,
    }

    # Lazy import — storage factory is heavy. Keeping it out of module
    # import time means tests that exercise the routing table don't
    # need a workspace at all.
    from .governance_gate import get_gate
    from .pipeline_hash import stamp_transform_hash
    from .storage import get_block_store

    store = get_block_store(workspace)
    with get_gate(workspace).admit_block(
        action="INGEST",
        block_id=block_id,
        content=content,
        tier=IngestTier.EXTERNAL_INGEST,
        actor="inbox",
        metadata={"source": os.path.basename(file_path)},
    ):
        written_id = store.write_block(stamp_transform_hash(workspace, block))
    _log.info("inbox_text_ingested", extra={"block_id": written_id, "source": file_path})
    return written_id


def _ingest_image(workspace: str, file_path: str) -> str:
    raise NotImplementedError(
        "image ingestion is not implemented. The inbox routes image drops to "
        "this multimodal handler, but no image embedder ships and no extra "
        "installs one -- there is no such extra to install. Convert the image "
        "to text yourself and drop that instead."
    )


def _ingest_audio(workspace: str, file_path: str) -> str:
    raise NotImplementedError(
        "audio ingestion is not implemented. The inbox routes audio drops to "
        "this multimodal handler, but no transcriber ships and no extra "
        "installs one -- there is no such extra to install. Transcribe the "
        "audio yourself and drop the transcript instead."
    )


def _ingest_pdf(workspace: str, file_path: str) -> str:
    # Try pypdf if installed; otherwise name the package that is missing.
    try:
        import pypdf  # type: ignore[import-untyped]
    except ImportError as exc:
        raise NotImplementedError(
            "PDF ingestion (the multimodal document path) requires pypdf, which "
            "mind-mem does not depend on and no extra installs. "
            "Install with: pip install pypdf"
        ) from exc

    reader = pypdf.PdfReader(file_path)
    pages = [p.extract_text() or "" for p in reader.pages]
    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError("PDF contained no extractable text")
    # Same invisible-Unicode sanitization as the text path (security).
    text = sanitize_text_for_ingest(text, workspace, source=file_path)

    base = os.path.splitext(os.path.basename(file_path))[0]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_base = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base)[:64] or "inbox-pdf"
    block_id = f"INBOX-PDF-{ts}-{safe_base}"
    block = {
        "_id": block_id,
        "type": "INBOX_DOCUMENT",
        "Subject": f"Inbox PDF: {os.path.basename(file_path)}",
        "Statement": text,
        "Source": os.path.basename(file_path),
        "Timestamp": ts,
        # The inbox is a DROP FOLDER: whatever lands in it is untrusted input,
        # exactly like an imported corpus. It arrives quarantined and stays
        # invisible to recall until a governance proposal releases it. Before
        # this, attacker-supplied file text landed Status: active and
        # immediately recallable, with no proposal and no chain entry -- the
        # same injection primitive the importer quarantine was built to close.
        "Status": QUARANTINE_STATUS,
        TIER_FIELD: QUARANTINE_TIER,
    }

    from .governance_gate import get_gate
    from .pipeline_hash import stamp_transform_hash
    from .storage import get_block_store

    store = get_block_store(workspace)
    with get_gate(workspace).admit_block(
        action="INGEST",
        block_id=block_id,
        content=text,
        tier=IngestTier.EXTERNAL_INGEST,
        actor="inbox",
        metadata={"source": os.path.basename(file_path)},
    ):
        return store.write_block(stamp_transform_hash(workspace, block))


_HANDLERS: dict[str, Callable[[str, str], str]] = {
    "text": ingest_text_file,
    "image": _ingest_image,
    "audio": _ingest_audio,
    "pdf": _ingest_pdf,
}


# ---------------------------------------------------------------------------
# Single-file processing helper
# ---------------------------------------------------------------------------


def _move_to_staging(file_path: str, staging_root: str, error_text: str | None = None) -> str:
    """Move *file_path* under *staging_root*/<ts>/. Returns the new path."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target_dir = os.path.join(staging_root, ts)
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, os.path.basename(file_path))
    shutil.move(file_path, target)
    if error_text is not None:
        with open(target + ".error.txt", "w", encoding="utf-8") as fh:
            fh.write(error_text)
    return target


def process_file(workspace: str, file_path: str, *, processed_dir: str, failed_dir: str) -> IngestResult:
    """Classify, ingest, and stage *file_path*. Never raises."""
    handler_name = classify_file(file_path)
    if handler_name is None:
        msg = f"unsupported extension: {os.path.splitext(file_path)[1]}"
        try:
            _move_to_staging(file_path, failed_dir, error_text=msg)
        except OSError as move_err:
            _log.error("inbox_move_failed", extra={"path": file_path, "error": str(move_err)})
        return IngestResult(path=file_path, handler="unknown", ok=False, error=msg)

    handler = _HANDLERS[handler_name]
    try:
        block_id = handler(workspace, file_path)
        _move_to_staging(file_path, processed_dir)
        return IngestResult(path=file_path, handler=handler_name, ok=True, block_id=block_id)
    except Exception as exc:
        _log.warning(
            "inbox_handler_failed",
            extra={"path": file_path, "handler": handler_name, "error": str(exc)},
        )
        try:
            _move_to_staging(file_path, failed_dir, error_text=str(exc))
        except OSError as move_err:
            _log.error("inbox_move_failed", extra={"path": file_path, "error": str(move_err)})
        return IngestResult(path=file_path, handler=handler_name, ok=False, error=str(exc))


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------


class InboxWatcher:
    """Poll an inbox directory and route new files through ``process_file``.

    Directory layout (auto-created)::

        <inbox_root>/
            file1.md
            ...
            _processed/<ts>/file1.md
            _failed/<ts>/bad.bin
            _failed/<ts>/bad.bin.error.txt

    The watcher is stdlib-only (``threading.Timer`` + ``os.scandir``).
    Files are processed in mtime order. The polling interval is
    configurable; the default 5s is gentle enough for shared
    workspaces.
    """

    def __init__(
        self,
        workspace: str,
        inbox: str,
        *,
        interval: float = 5.0,
        on_result: Callable[[IngestResult], None] | None = None,
    ) -> None:
        if not workspace:
            raise ValueError("workspace must be a non-empty path")
        if not inbox:
            raise ValueError("inbox must be a non-empty path")
        if interval < 0.5:
            raise ValueError("interval must be >= 0.5 seconds")
        self.workspace = workspace
        self.inbox_root = os.path.abspath(inbox)
        self.processed_dir = os.path.join(self.inbox_root, "_processed")
        self.failed_dir = os.path.join(self.inbox_root, "_failed")
        self.interval = interval
        self.on_result = on_result
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for d in (self.inbox_root, self.processed_dir, self.failed_dir):
            os.makedirs(d, exist_ok=True)

    # ----- lifecycle ---------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="mm-inbox", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1.0)

    # ----- main loop ---------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                files = self._list_pending_files()
            except OSError as exc:
                _log.error("inbox_list_failed", extra={"error": str(exc)})
                files = []
            for f in files:
                if self._stop_event.is_set():
                    return
                result = process_file(
                    self.workspace,
                    f,
                    processed_dir=self.processed_dir,
                    failed_dir=self.failed_dir,
                )
                if self.on_result is not None:
                    try:
                        self.on_result(result)
                    except Exception as cb_err:  # callbacks must not kill the loop
                        _log.warning("inbox_callback_failed", extra={"error": str(cb_err)})
            # Sleep in 0.5s ticks so stop() is responsive
            for _ in range(int(self.interval / 0.5)):
                if self._stop_event.is_set():
                    return
                time.sleep(0.5)

    def _list_pending_files(self) -> list[str]:
        """Return sorted list of files at the inbox root (excluding staging)."""
        out: list[tuple[float, str]] = []
        with os.scandir(self.inbox_root) as it:
            for entry in it:
                if not entry.is_file(follow_symlinks=False):
                    continue
                base = entry.name
                if any(base.startswith(p) for p in _IGNORE_BASENAME_PREFIXES):
                    continue
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                out.append((mtime, entry.path))
        out.sort(key=lambda pair: pair[0])
        return [p for _, p in out]

    # ----- one-shot mode ----------------------------------------------

    def process_once(self) -> list[IngestResult]:
        """Process every file currently in the inbox and return results."""
        results: list[IngestResult] = []
        for f in self._list_pending_files():
            result = process_file(
                self.workspace,
                f,
                processed_dir=self.processed_dir,
                failed_dir=self.failed_dir,
            )
            results.append(result)
            if self.on_result is not None:
                try:
                    self.on_result(result)
                except Exception as cb_err:
                    _log.warning("inbox_callback_failed", extra={"error": str(cb_err)})
        return results
