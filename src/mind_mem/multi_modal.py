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
import math
import os
from dataclasses import dataclass, field
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


def cross_modal_similarity(
    a: Iterable[float], b: Iterable[float]
) -> float:
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
        return int(table.get("image", _DEFAULT_IMAGE_TOKENS))
    if kind == "audio":
        duration = float(block.get("duration_seconds", 0.0))
        per_sec = float(table.get("audio_per_second", _DEFAULT_AUDIO_TOKENS_PER_SECOND))
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


__all__ = [
    "ImageBlock",
    "AudioBlock",
    "thumbnail_hash",
    "build_image_block",
    "build_audio_block",
    "cross_modal_similarity",
    "modal_token_cost",
]
