#!/usr/bin/env python3
"""Config seam wiring :mod:`smart_chunker` into the BM25 chunk-boost path.

``_recall_core.recall`` scores long ``Statement`` fields twice: once whole,
once as the best-scoring sub-chunk, then blends the two. The sub-chunks came
from :func:`_recall_detection.chunk_text` — a fixed three-sentence sliding
window that is blind to document structure, so a chunk routinely straddles a
markdown header and mixes two sections into one scoring unit.

This module lets an operator swap that window for
:func:`smart_chunker.smart_chunk`, whose boundaries follow the document's own
structure. It is **off by default**::

    {"retrieval": {"smart_chunking": {"enabled": true}}}

Design constraints this seam exists to enforce:

* **Default-off, byte-identical.** :func:`resolve_smart_chunking_config` is a
  pure read of an already-loaded mapping — no clock, no disk, no logging, no
  raising. A disabled (or absent, or malformed) config resolves to
  ``enabled=False`` and the caller keeps calling ``chunk_text`` unchanged.
* **Deterministic.** ``smart_chunk``'s optional LLM boundary refinement is
  pinned OFF here and is deliberately not exposed as a config key: recall is a
  pure function of (corpus, config, scoring_instant), and an LLM call on the
  scored path would break that.
* **Superset, not replacement.** Structure-aware boundaries only exist where
  the text has structure. When ``smart_chunk`` finds none (one chunk back), the
  seam falls back to ``chunk_text`` so unstructured prose keeps the sentence
  windows — enabling the flag adds header alignment, it does not remove the
  boost from header-less blocks.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ._recall_detection import chunk_text
from .smart_chunker import SmartChunkerConfig, smart_chunk

__all__ = [
    "CHUNK_CACHE_SIZE",
    "DEFAULT_MAX_CHUNK_SIZE",
    "DEFAULT_MIN_CHUNK_SIZE",
    "DEFAULT_OVERLAP_SENTENCES",
    "DEFAULT_SOFT_MAX_BOUNDARY_SCORE",
    "DEFAULT_SOFT_MAX_CHUNK_SIZE",
    "SmartChunkingConfig",
    "chunk_statement",
    "restore_header_gaps",
    "is_smart_chunking_enabled",
    "resolve_smart_chunking_config",
]

#: Hard ceiling per scoring chunk, in characters. Matches
#: ``SmartChunkerConfig.max_chunk_size``.
DEFAULT_MAX_CHUNK_SIZE = 1500

#: ``0`` — undersized chunks are **not** merged back into their neighbour.
#: ``smart_chunker`` defaults this to 100 for document ingestion, where a
#: 40-character chunk is a poor retrieval unit. Here each chunk is only ever
#: scored, never stored or surfaced, so a short one costs nothing — and
#: merging it would undo the very header alignment this seam is for
#: (``# Title`` + one short line is exactly the group that would be swallowed).
DEFAULT_MIN_CHUNK_SIZE = 0

#: ``0`` — no sentence bleed between adjacent scoring chunks. Overlap exists to
#: preserve reading context across stored chunks; here it would let a term from
#: section A raise section B's chunk score, which is the confusion the
#: structure-aware split is meant to remove.
DEFAULT_OVERLAP_SENTENCES = 0

#: ``1`` arms ``smart_chunker``'s soft ceiling from the first character, so a
#: group closes as soon as a *strong* structural boundary appears rather than
#: only when the 1500-character hard limit is hit. Without this every statement
#: shorter than the hard limit would come back as a single chunk and the
#: structure would never be consulted.
DEFAULT_SOFT_MAX_CHUNK_SIZE = 1

#: ``0.5`` is calibrated against ``smart_chunker._score_boundary`` so that a
#: markdown header — and only a header — closes a chunk. A header scores at
#: least 0.70 (0.5 for the header itself, +0.2/+0.1 for level ≤2/≤3, +0.15 for
#: the kind transition, +0.05 for the blank-line gap). Every non-header
#: boundary stays under: paragraph→paragraph tops out at 0.20, and even a
#: code-block edge reaches only 0.45. So running prose is left intact.
DEFAULT_SOFT_MAX_BOUNDARY_SCORE = 0.5


@dataclass(frozen=True)
class SmartChunkingConfig:
    """Resolved ``retrieval.smart_chunking`` settings (default-OFF)."""

    enabled: bool = False
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES
    soft_max_chunk_size: int = DEFAULT_SOFT_MAX_CHUNK_SIZE
    soft_max_boundary_score: float = DEFAULT_SOFT_MAX_BOUNDARY_SCORE

    def to_chunker_config(self) -> SmartChunkerConfig:
        """Project onto a :class:`SmartChunkerConfig`, LLM refinement pinned off."""
        return SmartChunkerConfig(
            max_chunk_size=self.max_chunk_size,
            soft_max_chunk_size=self.soft_max_chunk_size,
            soft_max_boundary_score=self.soft_max_boundary_score,
            min_chunk_size=self.min_chunk_size,
            overlap_sentences=self.overlap_sentences,
            preserve_code_blocks=True,
            llm_refine=False,
        )


def _positive_int(section: Mapping[str, Any], key: str, default: int, *, minimum: int) -> int:
    """Read an int knob defensively; anything unusable falls back to *default*."""
    raw = section.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return default
    value = int(raw)
    return value if value >= minimum else default


def resolve_smart_chunking_config(config: Mapping[str, Any] | None) -> SmartChunkingConfig:
    """Read ``retrieval.smart_chunking`` defensively; unknown shapes → defaults.

    Pure: reads the passed mapping and nothing else. It never logs, never
    touches the clock or the filesystem, and never raises — a probe that
    answers "off" must leave no trace the unwired build did not, and a
    malformed config must not be able to fail a recall.
    """
    if not isinstance(config, Mapping):
        return SmartChunkingConfig()
    retrieval = config.get("retrieval")
    if not isinstance(retrieval, Mapping):
        return SmartChunkingConfig()
    section = retrieval.get("smart_chunking")
    if not isinstance(section, Mapping):
        return SmartChunkingConfig()

    if section.get("enabled") is not True:
        # Explicit-True only, matching retrieval.trust_scores: a truthy
        # non-bool ("yes", 1) is a config typo, not consent to change ranking.
        return SmartChunkingConfig()

    max_chunk_size = _positive_int(section, "max_chunk_size", DEFAULT_MAX_CHUNK_SIZE, minimum=1)
    min_chunk_size = _positive_int(section, "min_chunk_size", DEFAULT_MIN_CHUNK_SIZE, minimum=0)
    overlap_sentences = _positive_int(section, "overlap_sentences", DEFAULT_OVERLAP_SENTENCES, minimum=0)
    soft_max = _positive_int(section, "soft_max_chunk_size", DEFAULT_SOFT_MAX_CHUNK_SIZE, minimum=0)
    # smart_chunker rejects a soft ceiling above the hard one. Clamp rather
    # than propagate the ValueError: this runs inside recall.
    soft_max = min(soft_max, max_chunk_size)

    raw_score = section.get("soft_max_boundary_score", DEFAULT_SOFT_MAX_BOUNDARY_SCORE)
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        boundary_score = DEFAULT_SOFT_MAX_BOUNDARY_SCORE
    else:
        boundary_score = min(1.0, max(0.0, float(raw_score)))

    return SmartChunkingConfig(
        enabled=True,
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        overlap_sentences=overlap_sentences,
        soft_max_chunk_size=soft_max,
        soft_max_boundary_score=boundary_score,
    )


def is_smart_chunking_enabled(config: Mapping[str, Any] | None) -> bool:
    """True only when ``retrieval.smart_chunking.enabled`` is explicitly ``true``."""
    return resolve_smart_chunking_config(config).enabled


#: A markdown header line that is not already preceded by a blank line.
#: ``(?<!\n)`` anchors on a *non-empty* previous line.
_TIGHT_HEADER_RE = re.compile(r"(?<!\n)\n(#{1,6}[ \t])")


def restore_header_gaps(text: str) -> str:
    """Re-insert the blank line a markdown header needs to be seen as one.

    This adapter is not cosmetic — without it the whole seam is inert, which is
    worth stating plainly. ``smart_chunker._segment_document`` cuts segments at
    blank lines, and ``block_parser.parse_blocks`` **drops every blank line**
    inside a field: a continuation line contributes only when it has content,
    so a multi-paragraph ``Statement:`` is stored ``"\n"``-joined with the
    paragraph gaps gone. A statement read back out of the corpus therefore
    always segments as exactly one segment, ``smart_chunk`` always returns one
    chunk, and the structure-aware path would silently never fire.

    Restoring the gap in front of each header — and nothing else — is the
    minimal repair: the header syntax survives the round-trip intact, so the
    break it implies can be reconstructed exactly, whereas an ordinary
    paragraph break leaves no trace to reconstruct from.

    A blank line landing inside a fenced code block is harmless:
    ``_segment_document`` refuses to cut inside a code span, and these chunks
    are scoring units that are never stored or surfaced.
    """
    return _TIGHT_HEADER_RE.sub(r"\n\n\1", text)


#: Chunking is a pure function of (text, config), and recall re-chunks the same
#: statements on every query, so the result is memoised. Bounded so a large
#: corpus cannot grow it without limit; the entries are dropped LRU-first.
CHUNK_CACHE_SIZE = 512


@lru_cache(maxsize=CHUNK_CACHE_SIZE)
def _smart_chunks(text: str, cfg: SmartChunkingConfig) -> tuple[str, ...]:
    """Memoised structure-aware split. Empty tuple = no structure found.

    deferred: ``smart_chunk`` emits an INFO ``smart_chunk_complete`` line per
    call, which on a large corpus means one line per distinct long statement on
    the first query after start-up (the cache silences the repeats, and the
    flag-off path never gets here at all). Left as-is rather than quietly
    lowering another module's log level from this seam — upgrade path: give
    ``smart_chunk`` a caller-set verbosity and pass it from
    ``to_chunker_config``, or move that line to DEBUG in ``smart_chunker``
    itself with its own test."""
    chunks = smart_chunk(restore_header_gaps(text), config=cfg.to_chunker_config())
    if len(chunks) <= 1:
        return ()
    return tuple(c.text for c in chunks)


def chunk_statement(text: str, cfg: SmartChunkingConfig) -> list[str]:
    """Return the sub-chunks a statement is scored by.

    With *cfg* disabled this is ``chunk_text(text)`` — the identical call the
    caller made before this seam existed, same argument, same result. With it
    enabled, boundaries come from the document's own structure, falling back to
    the sentence window when the text has no structure to follow.
    """
    if not cfg.enabled:
        return chunk_text(text)

    chunks = _smart_chunks(text, cfg)
    if not chunks:
        # No structural boundary found — a header-less wall of prose. Keep the
        # sentence window so enabling the flag never *removes* a chunk boost.
        return chunk_text(text)
    # Fresh list per call: the cache owns the tuple, the caller owns its list.
    return list(chunks)
