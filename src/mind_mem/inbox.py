"""Inbox folder ingestion — `mm inbox-watch` (v3.9.0 candidate).

Drop any file into a configured ``inbox/`` directory and mind-mem
classifies it by extension and routes to the right ingestion path.
The text path is stdlib-only. Nothing here reads pixels or audio
samples: no embedder and no transcriber ships, and there is no extra to
install that adds one. There is no ``mind-mem[multimodal]`` extra;
``pyproject.toml`` declares no such extra, so telling an operator to
install one sends them to a pip warning and a no-op. PDF works if
``pypdf`` is importable, which no mind-mem extra installs either — the
handlers below name the package that is actually missing instead.

Image and audio drops are refused unless ``v4.multi_modal`` is on, and
even then nothing is derived from the media: the operator supplies a
**sidecar** (``photo.png`` beside ``photo.png.txt``) and its text becomes
the block, with the media file hashed for a stable signature. See the
sidecar section below.

Routing rules (file extension → handler)::

    .txt .md .json .csv .log .xml .yaml .yml  → text → markdown block
    .png .jpg .jpeg .gif .webp                → image  (needs a sidecar)
    .mp3 .wav .flac .m4a                      → audio  (needs a sidecar)
    .pdf                                      → text extract  (needs pypdf)

Every door here writes through ``GovernanceGate.admit_block`` under
``IngestTier.EXTERNAL_INGEST``, so every block a drop produces — text,
PDF, image or audio — arrives ``Status: quarantined`` and is invisible
to recall until a governance proposal releases it.

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

import json
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
from .multi_modal import build_audio_block, build_image_block
from .multi_modal import flag_enabled as _multimodal_enabled
from .v4.backpressure import PRODUCER_INBOX as _BP_INBOX
from .v4.backpressure import batch_limit as _bp_batch_limit
from .v4.backpressure import report_depth as _bp_report_depth

__all__ = [
    "ROUTING_TABLE",
    "InboxWatcher",
    "IngestResult",
    "Sidecar",
    "classify_file",
    "ingest_text_file",
    "sidecar_path_for",
]

_log = logging.getLogger("mind_mem.inbox")

# ---------------------------------------------------------------------------
# Routing table — declarative; lazy-imports keep heavy deps optional.
# ---------------------------------------------------------------------------

# (extension → handler-name). Handlers live below. A handler that
# refuses must name what is actually missing; it must not name a
# `mind-mem[...]` extra, because none of them has one — see the module
# docstring.
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
    # Image (multimodal; needs a sidecar + v4.multi_modal — no embedder ships)
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
    # Audio (multimodal; needs a sidecar + v4.multi_modal — no transcriber ships)
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
# The multimodal sidecar door — v4.multi_modal, default OFF
# ---------------------------------------------------------------------------
#
# Still no embedder and still no transcriber: nothing here reads pixels or
# samples, and no extra installs anything that does. What the door accepts
# is a SIDECAR the operator wrote — ``photo.png`` next to
# ``photo.png.txt``. The media file is hashed, never interpreted; the
# sidecar's text is the block's content. That is the honest shape of
# "multimodal ingest" without a model: the description is SUPPLIED rather
# than derived, and the block says so.
#
# The sidecar name is the FULL media filename plus a suffix, never the
# stem. ``photo.txt`` is a legitimate text drop in its own right, and
# quietly consuming it as somebody's caption would swallow a file the
# operator meant to ingest as a document.
#
# Untrusted like every other drop: the sidecar goes through the same
# codepoint sanitizer and lands under the same EXTERNAL_INGEST tier, so
# the block arrives quarantined and recall cannot see it until a
# governance proposal releases it.
_SIDECAR_SUFFIXES: tuple[str, ...] = (".txt", ".md")
_MEDIA_HANDLERS: frozenset[str] = frozenset({"image", "audio"})
#: A caption or a transcript, not a corpus.
_INGEST_SIDECAR_BYTES = 1024 * 1024
#: Bounds the work the thumbnail hash will do for one drop.
_INGEST_MEDIA_BYTES = 64 * 1024 * 1024
_MAX_DURATION_SECONDS = 24 * 60 * 60
_MAX_SPEAKERS = 64
_MAX_SPEAKER_CHARS = 128
_MAX_PIXELS = 1_000_000


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
# Sidecars — the operator-supplied description of a media file
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sidecar:
    """One media file's description, as the operator wrote it.

    ``text`` is the whole sidecar for a plain-text sidecar, or the
    ``description`` / ``transcript`` / ``text`` member of a JSON object
    sidecar. The remaining fields exist only in the JSON form and are
    absent (zero / empty) otherwise — a plain caption knows nothing about
    duration, speakers or pixel dimensions.

    No ``embedding``. The door deliberately refuses to accept vectors
    from a file: an embedding is a scoring input, and taking one from
    untrusted input would let a drop steer retrieval rather than merely
    supply text for a human to admit.
    """

    path: str
    text: str
    duration_seconds: float = 0.0
    speakers: tuple[str, ...] = ()
    dimensions: tuple[int, int] = (0, 0)


def sidecar_path_for(media_path: str) -> str | None:
    """The existing sidecar for *media_path*, or ``None``.

    Candidates are the full media filename plus each of
    :data:`_SIDECAR_SUFFIXES`, in order, so the answer is deterministic
    when an operator has written both.

    A SYMLINK is not a sidecar. ``_list_pending_files`` scans with
    ``is_file(follow_symlinks=False)``, so the watcher has always refused
    to ingest a symlinked drop; resolving a sidecar with a plain
    ``os.path.isfile`` would follow one, and the new door would read a
    file the old door declines to — a link named ``board.png.txt``
    pointing anywhere the process can read. The two doors get the same
    rule.
    """
    for suffix in _SIDECAR_SUFFIXES:
        candidate = media_path + suffix
        if os.path.isfile(candidate) and not os.path.islink(candidate):
            return candidate
    return None


def _sidecar_owner(name: str) -> str | None:
    """The media file *name* would be a sidecar for, or ``None``.

    Pure string work on the basename — no I/O, no existence check. The
    caller decides whether that owner is actually present.
    """
    for suffix in _SIDECAR_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            owner = name[: -len(suffix)]
            if ROUTING_TABLE.get(os.path.splitext(owner)[1].lower()) in _MEDIA_HANDLERS:
                return owner
    return None


def _sidecar_number(data: dict, key: str, *, maximum: float) -> float:
    """A finite, in-range number from a JSON sidecar. Raises on anything else."""
    value = data.get(key)
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"sidecar field {key!r} must be a number, got {type(value).__name__}")
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        raise ValueError(f"sidecar field {key!r} must be finite")
    if not 0.0 <= out <= maximum:
        raise ValueError(f"sidecar field {key!r} must be in [0, {maximum}], got {out}")
    return out


def _sidecar_speakers(data: dict) -> tuple[str, ...]:
    """Speaker labels from a JSON sidecar, bounded in count and length."""
    value = data.get("speakers")
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(s, str) for s in value):
        raise ValueError("sidecar field 'speakers' must be a list of strings")
    if len(value) > _MAX_SPEAKERS:
        raise ValueError(f"sidecar field 'speakers' holds {len(value)} entries (max {_MAX_SPEAKERS})")
    if any(len(s) > _MAX_SPEAKER_CHARS for s in value):
        raise ValueError(f"a 'speakers' entry exceeds {_MAX_SPEAKER_CHARS} characters")
    return tuple(value)


def _sidecar_dimensions(data: dict) -> tuple[int, int]:
    """Pixel dimensions from a JSON sidecar.

    deferred: nothing here reads the image header, so an operator who does
    not state the dimensions gets ``(0, 0)`` — upgrade path: parse the PNG
    IHDR / JPEG SOF markers in-tree, or accept a Pillow dependency, but
    not a half-table that silently works for one format.
    """
    value = data.get("dimensions")
    if value is None:
        return (0, 0)
    if not isinstance(value, list) or len(value) != 2 or any(isinstance(n, bool) or not isinstance(n, int) for n in value):
        raise ValueError("sidecar field 'dimensions' must be [width, height] integers")
    if any(not 0 <= n <= _MAX_PIXELS for n in value):
        raise ValueError(f"sidecar field 'dimensions' must be in [0, {_MAX_PIXELS}]")
    return (value[0], value[1])


def _parse_sidecar(path: str, raw: str) -> Sidecar:
    """Interpret already-sanitized sidecar text. Pure; no I/O, no clock."""
    stripped = raw.strip()
    if not stripped:
        raise ValueError(f"sidecar {os.path.basename(path)} is empty; there is nothing to describe the media with")
    if not stripped.startswith("{"):
        return Sidecar(path=path, text=raw)
    try:
        data = json.loads(stripped)
    except ValueError:
        # A caption that merely begins with a brace is still a caption.
        return Sidecar(path=path, text=raw)
    if not isinstance(data, dict):
        return Sidecar(path=path, text=raw)

    body = ""
    for key in ("description", "transcript", "text"):
        candidate = data.get(key)
        if isinstance(candidate, str) and candidate.strip():
            body = candidate
            break
    if not body:
        raise ValueError(f"JSON sidecar {os.path.basename(path)} has no non-empty 'description', 'transcript' or 'text'")
    return Sidecar(
        path=path,
        text=body,
        duration_seconds=_sidecar_number(data, "duration_seconds", maximum=_MAX_DURATION_SECONDS),
        speakers=_sidecar_speakers(data),
        dimensions=_sidecar_dimensions(data),
    )


def _read_sidecar(workspace: str, media_path: str) -> Sidecar:
    """Load and sanitize the sidecar for *media_path*.

    Raises ``ValueError`` when there is none, when it is oversized, or
    when its JSON form is malformed — the caller routes the drop to
    ``_failed/`` with the message, so the operator reads exactly what was
    wrong next to the file that caused it.
    """
    path = sidecar_path_for(media_path)
    if path is None:
        wanted = ", ".join(os.path.basename(media_path) + suffix for suffix in _SIDECAR_SUFFIXES)
        raise ValueError(
            f"no sidecar description found for {os.path.basename(media_path)}; drop one of: {wanted}. "
            "Nothing in this package can describe or transcribe a media file, so the description has to be supplied."
        )
    size = os.path.getsize(path)
    if size > _INGEST_SIDECAR_BYTES:
        raise ValueError(f"sidecar too large for inbox ingestion: {size} bytes (max {_INGEST_SIDECAR_BYTES})")
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    # Security: identical treatment to the text door. A sidecar is
    # operator-written but it is still file content arriving through a
    # drop folder, so invisible-Unicode (zero-width, tag chars, bidi
    # controls) is stripped before the text can become a block.
    raw = sanitize_text_for_ingest(raw, workspace, source=path)
    return _parse_sidecar(path, raw)


def _check_media_size(file_path: str) -> None:
    """Refuse a media file too large to hash cheaply."""
    size = os.path.getsize(file_path)
    if size > _INGEST_MEDIA_BYTES:
        raise ValueError(f"media file too large for inbox ingestion: {size} bytes (max {_INGEST_MEDIA_BYTES})")


def _modal_block_id(file_path: str, tag: str) -> tuple[str, str]:
    """``(timestamp, block id)`` for a media drop, shaped like the text door's."""
    base = os.path.splitext(os.path.basename(file_path))[0]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_base = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base)[:64] or "inbox"
    return ts, f"INBOX-{tag}-{ts}-{safe_base}"


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


def _write_modal_block(workspace: str, file_path: str, block: dict[str, object], *, content: str) -> str:
    """Write one media block through the gate the text door uses.

    Same scope, same tier, same stamp: ``admit_block`` under
    :attr:`~mind_mem.enums.IngestTier.EXTERNAL_INGEST`, whose
    ``INITIAL_STATUS`` row is ``QUARANTINED``. The handlers below stamp
    that status onto the block as well, and ``require_admission`` refuses
    the write if the two ever disagree in the escalating direction — a
    media door cannot mint a servable block even by writing the field
    itself.

    Factored out of the two handlers rather than copied into both: this
    is the one place a multimodal drop becomes a stored block, so it is
    the one place a reviewer has to check.
    """
    from .governance_gate import get_gate
    from .pipeline_hash import stamp_transform_hash
    from .storage import get_block_store

    block_id = str(block["_id"])
    store = get_block_store(workspace)
    with get_gate(workspace).admit_block(
        action="INGEST",
        block_id=block_id,
        content=content,
        tier=IngestTier.EXTERNAL_INGEST,
        actor="inbox",
        metadata={"source": os.path.basename(file_path)},
    ):
        return store.write_block(stamp_transform_hash(workspace, block))


def _ingest_image(workspace: str, file_path: str) -> str:
    """A dropped image plus its sidecar description → one quarantined block.

    OFF unless ``v4.multi_modal`` is on for this workspace, and the
    refusal below is the message this handler has always raised, word for
    word: with the flag off, this door does not exist.
    """
    if not _multimodal_enabled(workspace):
        raise NotImplementedError(
            "image ingestion is not implemented. The inbox routes image drops to "
            "this multimodal handler, but no image embedder ships and no extra "
            "installs one -- there is no such extra to install. Convert the image "
            "to text yourself and drop that instead."
        )

    _check_media_size(file_path)
    side = _read_sidecar(workspace, file_path)
    ts, block_id = _modal_block_id(file_path, "IMG")
    # The real path is what gets hashed; only the basename is stored, so
    # the corpus does not learn the operator's directory layout (the text
    # door keeps the basename for the same reason).
    fields = build_image_block(block_id, side.text, file_path, dimensions=side.dimensions).as_dict()
    name = os.path.basename(file_path)
    # CAPITALISED field names, deliberately. `block_parser` only reads keys
    # matching `[A-Z][A-Za-z]+:`, so a lowercase `type:` line is written to
    # the file and then dropped by every reader -- which is exactly what
    # has been happening to the text door's `type: INBOX_DOCUMENT` all
    # along. A field nothing can read is decoration, so these are spelled
    # the way the parser actually parses.
    block: dict[str, object] = {
        "_id": block_id,
        "Type": fields["type"],
        "SourcePath": name,
        "ThumbnailHash": fields["thumbnail_hash"],
        "Subject": f"Inbox image: {name}",
        # The description lives in Statement -- the corpus-native text
        # field every read path already knows -- rather than being
        # duplicated into the schema's own `description` key.
        "Statement": side.text,
        "Source": name,
        "Sidecar": os.path.basename(side.path),
        "Timestamp": ts,
        # A drop folder is untrusted input. Quarantined on arrival,
        # invisible to recall until a governance proposal releases it.
        "Status": QUARANTINE_STATUS,
        TIER_FIELD: QUARANTINE_TIER,
    }
    if side.dimensions != (0, 0):
        block["Dimensions"] = list(fields["dimensions"])

    written_id = _write_modal_block(workspace, file_path, block, content=side.text)
    _log.info("inbox_image_ingested", extra={"block_id": written_id, "source": file_path})
    return written_id


def _ingest_audio(workspace: str, file_path: str) -> str:
    """A dropped audio file plus its sidecar transcript → one quarantined block.

    Same bargain as :func:`_ingest_image`: nothing here transcribes
    anything, and with ``v4.multi_modal`` off the original refusal stands
    unchanged.
    """
    if not _multimodal_enabled(workspace):
        raise NotImplementedError(
            "audio ingestion is not implemented. The inbox routes audio drops to "
            "this multimodal handler, but no transcriber ships and no extra "
            "installs one -- there is no such extra to install. Transcribe the "
            "audio yourself and drop the transcript instead."
        )

    _check_media_size(file_path)
    side = _read_sidecar(workspace, file_path)
    ts, block_id = _modal_block_id(file_path, "AUD")
    fields = build_audio_block(
        block_id,
        side.text,
        file_path,
        duration_seconds=side.duration_seconds,
        speakers=side.speakers,
    ).as_dict()
    name = os.path.basename(file_path)
    block: dict[str, object] = {
        "_id": block_id,
        # Parser-visible spelling — see the note in _ingest_image.
        "Type": fields["type"],
        "SourcePath": name,
        # Kept even at 0.0: it is what modal_token_cost prices an audio
        # block by, and "unstated" is a fact about the drop worth storing.
        "DurationSeconds": fields["duration_seconds"],
        "Subject": f"Inbox audio: {name}",
        "Statement": side.text,
        "Source": name,
        "Sidecar": os.path.basename(side.path),
        "Timestamp": ts,
        "Status": QUARANTINE_STATUS,
        TIER_FIELD: QUARANTINE_TIER,
    }
    if side.speakers:
        block["Speakers"] = list(fields["speakers"])

    written_id = _write_modal_block(workspace, file_path, block, content=side.text)
    _log.info("inbox_audio_ingested", extra={"block_id": written_id, "source": file_path})
    return written_id


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


def _stage_sidecar(media_path: str, processed_dir: str) -> None:
    """Move a consumed sidecar out of the inbox, after its media file.

    Only reachable once a media ingest has SUCCEEDED, which can only
    happen with ``v4.multi_modal`` on — so this is not a second flag
    probe, and with the flag off it never runs.

    Leaving the sidecar behind would be worse than untidy. On the next
    poll it is an orphan ``.txt`` at the inbox root with no media beside
    it, and the text door would ingest the caption a second time as a
    document in its own right.
    """
    path = sidecar_path_for(media_path)
    if path is None:
        return
    try:
        _move_to_staging(path, processed_dir)
    except OSError as exc:
        _log.warning("inbox_sidecar_stage_failed", extra={"path": path, "error": str(exc)})


def _drop_consumed_sidecars(workspace: str, paths: list[str]) -> list[str]:
    """Hide sidecars whose media file is pending in the same listing.

    Without this the watcher would ingest ``photo.png.txt`` as a document
    of its own in the very pass that ingests ``photo.png`` as an image —
    the same caption stored twice, once as a caption and once as a
    document.

    Two properties are load-bearing. An ORPHAN sidecar (no media file
    beside it) is still an ordinary text drop and is left alone. And the
    flag is consulted only once a pair actually exists, so an inbox
    holding no media-plus-sidecar pair — every inbox that never opted in
    — never probes the config at all and its listing is unchanged.
    """
    names = {os.path.basename(p) for p in paths}
    consumed = {p for p in paths if (_sidecar_owner(os.path.basename(p)) or "") in names}
    if not consumed or not _multimodal_enabled(workspace):
        return paths
    return [p for p in paths if p not in consumed]


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
        if handler_name in _MEDIA_HANDLERS:
            _stage_sidecar(file_path, processed_dir)
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

    # ----- backpressure ------------------------------------------------

    def _backlog_limit(self, pending: int, *, bounded: bool) -> int | None:
        """Report the inbox backlog and return a per-tick cap, or ``None``.

        ``bounded=False`` reports the depth and never caps — that is
        ``process_once``, where the operator asked for the whole backlog
        by name. The scheduled loop is ``bounded=True``: while
        overloaded it takes ``low_watermark`` files and yields to the
        next tick.

        Deferring is not dropping. The files it does not take this tick
        stay in the inbox root and are listed again on the next pass, so
        the only thing shed is RATE. Nothing bypasses ``process_file``,
        which means nothing bypasses ``ingest_text_file``'s
        ``admit_block`` — a throttled inbox admits fewer blocks per
        tick, never a different KIND of block.

        Silent and inert when ``v4.backpressure`` is off.
        """
        if not bounded:
            _bp_report_depth(_BP_INBOX, pending)
            return None
        return _bp_batch_limit(_BP_INBOX, pending)

    # ----- main loop ---------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                files = self._list_pending_files()
            except OSError as exc:
                _log.error("inbox_list_failed", extra={"error": str(exc)})
                files = []
            limit = self._backlog_limit(len(files), bounded=True)
            if limit is not None and limit < len(files):
                _log.info(
                    "inbox_backpressure_throttled",
                    extra={"pending": len(files), "batch_limit": limit, "deferred": len(files) - limit},
                )
                files = files[:limit]
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
        return _drop_consumed_sidecars(self.workspace, [p for _, p in out])

    # ----- one-shot mode ----------------------------------------------

    def process_once(self) -> list[IngestResult]:
        """Process every file currently in the inbox and return results."""
        results: list[IngestResult] = []
        files = self._list_pending_files()
        # Report-only: one-shot mode drains everything the operator asked
        # for. The depth still reaches the controller so a scheduled loop
        # in the same process sees the backlog this run created.
        self._backlog_limit(len(files), bounded=False)
        for f in files:
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
