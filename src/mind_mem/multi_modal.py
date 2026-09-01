# Copyright 2026 STARGA, Inc.
"""Multi-modal block types (v2.4.0 — IMAGE / AUDIO schema).

CLIP / SigLIP / Whisper embeddings are intentionally NOT computed
inside this module — that would drag heavy model dependencies into
the core. Instead we ship the block schemas + metadata plumbing so
higher-level components (that can load whichever model they prefer)
can slot in and get cross-modal routing for free.

Each multi-modal block stores:

- IMAGE — description (text), optional embedding (list[float]),
  source path, dimensions (w, h), thumbnail SHA-256.
- AUDIO — transcript, optional embedding, duration seconds, speaker
  labels, source path.

Callers that have loaded a CLIP-style model pass embeddings to
:func:`build_image_block`; the token cost of a block is computed via
:func:`modal_token_cost` so the v2.4.0 token budget packer stays
accurate for mixed-modal context windows.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

# ---------------------------------------------------------------------------
# IMAGE + AUDIO blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageBlock:
    block_id: str
    description: str
    source_path: str
    dimensions: tuple[int, int] = (0, 0)
    thumbnail_hash: str = ""
    embedding: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "_id": self.block_id,
            "type": "image",
            "description": self.description,
            "source_path": self.source_path,
            "dimensions": list(self.dimensions),
            "thumbnail_hash": self.thumbnail_hash,
            "embedding": list(self.embedding),
        }


@dataclass(frozen=True)
class AudioBlock:
    block_id: str
    transcript: str
    source_path: str
    duration_seconds: float = 0.0
    speakers: tuple[str, ...] = ()
    embedding: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "_id": self.block_id,
            "type": "audio",
            "transcript": self.transcript,
            "source_path": self.source_path,
            "duration_seconds": self.duration_seconds,
            "speakers": list(self.speakers),
            "embedding": list(self.embedding),
        }


# ---------------------------------------------------------------------------
# Builders + SHA-256 thumbnail helper
# ---------------------------------------------------------------------------


def thumbnail_hash(path: str, *, block_size: int = 64 * 1024) -> str:
    """SHA-256 of a thumbnail file — cheap signature for change detection.

    The canonical thumbnail is out of scope (needs PIL); here we just
    hash whatever bytes the caller already persisted.
    """
    if not path or not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_image_block(
    block_id: str,
    description: str,
    source_path: str,
    *,
    dimensions: tuple[int, int] = (0, 0),
    embedding: Iterable[float] = (),
) -> ImageBlock:
    return ImageBlock(
        block_id=block_id,
        description=description,
        source_path=source_path,
        dimensions=dimensions,
        thumbnail_hash=thumbnail_hash(source_path),
        embedding=tuple(float(x) for x in embedding),
    )


def build_audio_block(
    block_id: str,
    transcript: str,
    source_path: str,
    *,
    duration_seconds: float = 0.0,
    speakers: Iterable[str] = (),
    embedding: Iterable[float] = (),
) -> AudioBlock:
    return AudioBlock(
        block_id=block_id,
        transcript=transcript,
        source_path=source_path,
        duration_seconds=float(duration_seconds),
        speakers=tuple(speakers),
        embedding=tuple(float(x) for x in embedding),
    )


# ---------------------------------------------------------------------------
# Cross-modal similarity (pure Python)
# ---------------------------------------------------------------------------


def cross_modal_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    ax = list(a)
    bx = list(b)
    if not ax or not bx:
        return 0.0
    # Pad the shorter vector with zeros so mismatched modalities still
    # yield a stable answer instead of raising.
    if len(ax) != len(bx):
        pad = [0.0] * abs(len(ax) - len(bx))
        if len(ax) < len(bx):
            ax = ax + pad
        else:
            bx = bx + pad
    dot = sum(x * y for x, y in zip(ax, bx))
    na = math.sqrt(sum(x * x for x in ax))
    nb = math.sqrt(sum(x * x for x in bx))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Modal-aware token cost
# ---------------------------------------------------------------------------


# Vision models charge roughly 85 tokens per tile (OpenAI gpt-4o-vision
# reference). Callers with a different cost table override via the
# ``model_cost_table`` argument of :func:`modal_token_cost`.
_DEFAULT_IMAGE_TOKENS: int = 85
_DEFAULT_AUDIO_TOKENS_PER_SECOND: float = 1.3


def _as_float(value: Any, default: float) -> float:
    """Coerce a corpus field to a float, or *default* — never raises.

    A block that came back through the Markdown parser carries every
    field as a **string**: ``duration_seconds`` is ``"30.0"``, not
    ``30.0``. A bare ``float(...)`` therefore works on the happy path and
    raises ``ValueError`` on a hand-edited (or hostile) corpus, and this
    function is on the packing path — an exception there would take down
    a recall response over one malformed field. Non-numeric means
    "duration unknown", which is what *default* says.
    """
    if isinstance(value, bool) or value is None:
        return default
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return default if out != out or out in (float("inf"), float("-inf")) else out


def modal_token_cost(
    block: Mapping[str, Any],
    *,
    model_cost_table: Optional[Mapping[str, Any]] = None,
) -> int:
    """Approximate token cost of a multi-modal block.

    Text blocks fall back to the v2.4 ``estimate_tokens`` heuristic;
    image + audio blocks use model-aware per-block costs.
    """
    kind = str(block.get("type", "")).lower()
    table = model_cost_table or {}
    if kind == "image":
        return int(_as_float(table.get("image"), _DEFAULT_IMAGE_TOKENS))
    if kind == "audio":
        duration = _as_float(block.get("duration_seconds"), 0.0)
        per_sec = _as_float(table.get("audio_per_second"), _DEFAULT_AUDIO_TOKENS_PER_SECOND)
        return max(1, int(round(duration * per_sec)))
    # Text / unknown — defer to stdlib estimator.
    text = ""
    for fld in ("text", "excerpt", "statement", "content", "description", "transcript"):
        v = block.get(fld)
        if isinstance(v, str) and v.strip():
            text = v
            break
    from .cognitive_forget import estimate_tokens

    return estimate_tokens(text)


#: Packing fields a text result may carry, in the order the packer reads
#: them. Mirrors ``pack_to_budget``'s ``text_field`` default; kept here so
#: :func:`pack_cost` is a drop-in for the estimator it replaces.
_PACK_TEXT_FIELD: str = "excerpt"


def _modality_of(block: Mapping[str, Any]) -> str:
    """``"image"`` / ``"audio"`` / ``""`` for a result OR a parsed block.

    Two spellings, because the same content wears two shapes on the way
    to the packer. A recall RESULT carries ``type``, derived from the id
    prefix by ``_recall_detection.get_block_type``. A block read back off
    disk carries ``Type``, because ``block_parser`` only sees field keys
    that start with a capital. Reading one spelling and writing the other
    is how a wiring silently does nothing.
    """
    for key in ("type", "Type"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            kind = value.strip().lower()
            if kind in ("image", "audio"):
                return kind
    return ""


def _stated_duration(block: Mapping[str, Any]) -> Optional[float]:
    """Seconds this audio result states, or ``None`` when it states none."""
    for key in ("duration_seconds", "DurationSeconds"):
        if key in block and block.get(key) is not None:
            return _as_float(block.get(key), 0.0)
    return None


def pack_cost(
    block: Mapping[str, Any],
    *,
    text_field: str = _PACK_TEXT_FIELD,
    model_cost_table: Optional[Mapping[str, Any]] = None,
) -> int:
    """Per-result token cost for the budget packer, modality-aware.

    The drop-in the ``pack_recall_budget`` path passes to
    :func:`~mind_mem.cognitive_forget.pack_to_budget` when the
    ``v4.multi_modal`` flag is on. A text result costs **exactly** what
    the packer's own estimator charges it — same field, same heuristic —
    so turning the flag on cannot move a text-only budget by one token;
    only image and audio results are priced differently, because for
    those the excerpt is a caption and the real cost is the tile count or
    the audio duration.

    An audio result that does not state its duration is priced by its
    transcript rather than by :func:`modal_token_cost`'s
    duration-of-zero, which would charge a half-hour recording one token.
    A recall RESULT does not currently carry the duration through — the
    result payload passes a fixed field list — so this is the ordinary
    case, not the exotic one, and understating it would be worse than
    declining to guess.

    Deterministic and total: no clock, no randomness, no exception path.
    """
    kind = _modality_of(block)
    if kind == "image":
        return modal_token_cost({"type": "image"}, model_cost_table=model_cost_table)
    if kind == "audio":
        duration = _stated_duration(block)
        if duration is not None:
            return modal_token_cost({"type": "audio", "duration_seconds": duration}, model_cost_table=model_cost_table)
    from .cognitive_forget import estimate_tokens

    return estimate_tokens(str(block.get(text_field, "")))


# ---------------------------------------------------------------------------
# Flag — the ingest door and the packing path are OFF until asked for
# ---------------------------------------------------------------------------


#: ``mind-mem.json`` → ``v4.multi_modal.enabled``. Off by default: this
#: gates an INGEST DOOR, and a door nobody asked for should not exist.
FLAG: str = "multi_modal"


def flag_enabled(workspace: str) -> bool:
    """``v4.multi_modal`` state for *workspace*, ambient config as fallback.

    Reads only, logs nothing, raises nothing — a caller may ask "is this
    surface on?" without the asking being observable. That is a hard
    requirement, not politeness: with the flag off this build must be
    indistinguishable from the one that never had the feature, and
    ``feature_flags.is_enabled`` would emit ``v4_config_unreadable`` on a
    malformed config and break exactly that.

    The workspace's own ``mind-mem.json`` wins over the ambient config,
    because every caller here (the inbox handlers, ``pack_recall_budget``)
    is already holding one explicit workspace directory.

    deferred: this is the third copy of the shape (``lint.flag_enabled``,
    ``maintenance_migrate.flag_enabled``) — upgrade path: hoist one
    ``feature_flags.is_enabled_for_workspace(ws, flag)`` and retire all
    three, once a slice touches those two modules for another reason.
    """
    config_path = os.path.join(workspace, "mind-mem.json") if workspace else ""
    if config_path:
        try:
            with open(config_path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict):
            block = data.get("v4")
            if isinstance(block, dict) and FLAG in block:
                sub = block.get(FLAG)
                return isinstance(sub, dict) and sub.get("enabled") is True

    from .v4.feature_flags import is_enabled_quiet

    return is_enabled_quiet(FLAG)


__all__ = [
    "FLAG",
    "ImageBlock",
    "AudioBlock",
    "flag_enabled",
    "pack_cost",
    "thumbnail_hash",
    "build_image_block",
    "build_audio_block",
    "cross_modal_similarity",
    "modal_token_cost",
]
