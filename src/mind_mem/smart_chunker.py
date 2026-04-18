#!/usr/bin/env python3
"""mind-mem Smart Chunker — Semantic-boundary document chunking.

Analyzes document structure (headers, paragraphs, code blocks, lists) and uses
heuristic-based semantic boundary detection to produce high-quality chunks.
Prefers natural document boundaries over fixed-size splits, while respecting
configurable maximum chunk size limits.

Optionally supports LLM-guided boundary refinement (off by default). When
enabled, candidate boundaries are sent to a local LLM for verification.

Zero external dependencies. LLM refinement uses the same ollama/llama-cpp
backends as llm_extractor.

Usage:
    from mind_mem.smart_chunker import smart_chunk, SmartChunkerConfig

    chunks = smart_chunk(document_text, config=SmartChunkerConfig(max_chunk_size=1500))
    for chunk in chunks:
        print(chunk.text, chunk.metadata)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .observability import get_logger

_log = get_logger("smart_chunker")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SmartChunkerConfig:
    """Configuration for the smart chunker.

    Attributes:
        max_chunk_size: Maximum characters per chunk. Hard limit — chunks are
            never larger than this. Default 1500.
        min_chunk_size: Minimum characters per chunk. Chunks below this size
            are merged with their neighbor. Default 100.
        overlap_sentences: Number of trailing sentences to overlap between
            adjacent chunks for context continuity. Default 1.
        preserve_code_blocks: If True, code blocks (fenced or indented) are
            never split mid-block. Default True.
        llm_refine: If True, candidate boundaries are sent to a local LLM for
            refinement. Requires an ollama or llama-cpp backend. Default False.
        llm_model: Model name for LLM refinement. Default "qwen3.5:9b".
        llm_backend: LLM backend — "auto", "ollama", or "llama-cpp".
            Default "auto".
        source: Optional source identifier attached to every chunk's metadata.
            Default "".
    """

    max_chunk_size: int = 1500
    min_chunk_size: int = 100
    overlap_sentences: int = 1
    preserve_code_blocks: bool = True
    llm_refine: bool = False
    llm_model: str = "qwen3.5:9b"
    llm_backend: str = "auto"
    source: str = ""


# ---------------------------------------------------------------------------
# Chunk result
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single chunk produced by the smart chunker.

    Attributes:
        text: The chunk content.
        index: Zero-based chunk index within the document.
        start_char: Character offset where this chunk starts in the original
            document (before overlap is applied).
        end_char: Character offset where this chunk ends in the original
            document.
        metadata: Preserved metadata — section headers, source, position info.
    """

    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Structural element detection
# ---------------------------------------------------------------------------

# Markdown ATX headers: # through ######
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Fenced code block boundaries: ``` or ~~~
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})", re.MULTILINE)

# Blank line (one or more)
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")

# List item start: - , * , + , or 1. 2. etc.
_LIST_ITEM_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+\.)\s+", re.MULTILINE)

# Sentence boundary: period/question/exclamation followed by whitespace
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


# Code boundary patterns for AST-aware splitting of code blocks
_CODE_BOUNDARY_PATTERNS: list[re.Pattern[str]] = [
    # Python: def/class/async def
    re.compile(r"^(?:async\s+)?def\s+\w+", re.MULTILINE),
    re.compile(r"^class\s+\w+", re.MULTILINE),
    # JavaScript/TypeScript: function, class, const/let/var arrow
    re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+\w+", re.MULTILINE),
    re.compile(r"^(?:export\s+)?class\s+\w+", re.MULTILINE),
    re.compile(r"^(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\(?.*?\)?\s*=>", re.MULTILINE),
    # Rust: fn, impl, struct
    re.compile(r"^(?:pub\s+)?(?:async\s+)?fn\s+\w+", re.MULTILINE),
    re.compile(r"^(?:pub\s+)?impl\b", re.MULTILINE),
    re.compile(r"^(?:pub\s+)?struct\s+\w+", re.MULTILINE),
    # Go: func
    re.compile(r"^func\s+(?:\(.*?\)\s*)?\w+", re.MULTILINE),
]


def _detect_code_boundaries(text: str) -> list[int]:
    """Detect function/class/method boundaries in code text.

    Uses lightweight regex patterns (not a real AST parser) to find the
    start offsets of function, class, method, and struct definitions in
    Python, JavaScript/TypeScript, Rust, and Go.

    Args:
        text: Code block text to analyze.

    Returns:
        Sorted list of character offsets where code boundaries occur.
        Only boundaries after position 0 are returned (the start of the
        text is never included as a boundary).
    """
    offsets: set[int] = set()
    for pattern in _CODE_BOUNDARY_PATTERNS:
        for m in pattern.finditer(text):
            if m.start() > 0:
                offsets.add(m.start())
    return sorted(offsets)


@dataclass
class _Segment:
    """Internal representation of a structural segment."""

    text: str
    start: int
    end: int
    kind: str  # "header", "code", "paragraph", "list", "blank"
    header_text: str = ""
    header_level: int = 0


def _identify_code_spans(text: str) -> list[tuple[int, int]]:
    """Find all fenced code block spans as (start, end) character offsets."""
    spans: list[tuple[int, int]] = []
    fence_stack: int | None = None
    for m in _FENCE_RE.finditer(text):
        if fence_stack is None:
            fence_stack = m.start()
        else:
            spans.append((fence_stack, m.end()))
            fence_stack = None
    return spans


def _in_code_span(pos: int, code_spans: list[tuple[int, int]]) -> bool:
    """Check whether a character position falls inside a code block."""
    for start, end in code_spans:
        if start <= pos < end:
            return True
    return False


def _segment_document(text: str) -> list[_Segment]:
    """Split a document into structural segments.

    Identifies headers, code blocks, list regions, and paragraphs separated
    by blank lines. Preserves original character offsets.
    """
    if not text.strip():
        return []

    code_spans = _identify_code_spans(text)
    segments: list[_Segment] = []

    # Collect boundary positions: every blank-line gap and every header
    boundaries: list[int] = [0]

    for m in _BLANK_LINE_RE.finditer(text):
        mid = m.start() + 1  # just after the first \n
        if not _in_code_span(mid, code_spans):
            boundaries.append(m.end())

    boundaries.append(len(text))
    boundaries = sorted(set(boundaries))

    # Active header context
    current_header = ""
    current_header_level = 0

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        chunk = text[start:end].strip()
        if not chunk:
            continue

        # Determine segment kind
        kind = "paragraph"

        # Is this segment overlapping with a code span?
        for cs_start, cs_end in code_spans:
            # Segment overlaps if it starts within the span, or the span
            # starts within the segment, or the segment is fully contained.
            if start < cs_end and end > cs_start:
                kind = "code"
                break

        if kind != "code":
            header_match = _HEADER_RE.match(chunk)
            if header_match:
                kind = "header"
                current_header_level = len(header_match.group(1))
                current_header = header_match.group(2).strip()
            elif _LIST_ITEM_RE.match(chunk):
                kind = "list"

        # For code segments, detect function/class boundaries and split.
        # Only split when there are 2+ boundaries (i.e., multiple
        # function/class definitions worth separating).
        if kind == "code":
            code_bounds = _detect_code_boundaries(chunk)
            if len(code_bounds) >= 2:
                # Split code block at function/class boundaries
                sub_starts = [0] + code_bounds
                for j in range(len(sub_starts)):
                    sub_start = sub_starts[j]
                    sub_end = sub_starts[j + 1] if j + 1 < len(sub_starts) else len(chunk)
                    sub_text = chunk[sub_start:sub_end].strip()
                    if not sub_text:
                        continue
                    segments.append(
                        _Segment(
                            text=sub_text,
                            start=start + sub_start,
                            end=start + sub_end,
                            kind="code",
                            header_text=current_header,
                            header_level=current_header_level,
                        )
                    )
                continue

        seg = _Segment(
            text=chunk,
            start=start,
            end=end,
            kind=kind,
            header_text=current_header,
            header_level=current_header_level,
        )
        segments.append(seg)

    return segments


# ---------------------------------------------------------------------------
# Boundary scoring
# ---------------------------------------------------------------------------


def _score_boundary(before: _Segment, after: _Segment) -> float:
    """Score a candidate boundary between two adjacent segments.

    Higher score = stronger natural boundary = better place to split.
    Range: 0.0 to 1.0.
    """
    score = 0.0

    # Header after the boundary is the strongest signal
    if after.kind == "header":
        score += 0.5
        # Higher-level headers are even stronger
        if after.header_level <= 2:
            score += 0.2
        elif after.header_level <= 3:
            score += 0.1

    # Transition between different segment kinds
    if before.kind != after.kind:
        score += 0.15

    # Code block boundary — good place to split
    if before.kind == "code" or after.kind == "code":
        score += 0.1

    # Blank line gap (implicit in our segmentation)
    score += 0.05

    # Topic shift heuristic: check word overlap between segments
    words_before = set(before.text.lower().split()[-30:])
    words_after = set(after.text.lower().split()[:30])
    if words_before and words_after:
        overlap = len(words_before & words_after)
        max_possible = min(len(words_before), len(words_after))
        if max_possible > 0:
            similarity = overlap / max_possible
            # Low similarity = topic shift = good boundary
            if similarity < 0.1:
                score += 0.15
            elif similarity < 0.2:
                score += 0.08

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Core chunking algorithm
# ---------------------------------------------------------------------------


def _merge_segments_into_chunks(
    segments: list[_Segment],
    config: SmartChunkerConfig,
) -> list[tuple[list[_Segment], str, str, int]]:
    """Group segments into chunks respecting size limits and natural boundaries.

    Returns list of (segments, header_text, header_level_str, start_char) tuples.
    Each tuple represents one chunk's worth of segments.
    """
    if not segments:
        return []

    groups: list[tuple[list[_Segment], str, str, int]] = []
    current_segs: list[_Segment] = []
    current_size = 0
    current_header = ""
    current_header_level = 0

    for i, seg in enumerate(segments):
        seg_size = len(seg.text)

        # If adding this segment would exceed max, close current group
        if current_segs and (current_size + seg_size + 1) > config.max_chunk_size:
            # But first check if the boundary score is low — if so, we might
            # want to force-split the current segment instead
            groups.append(
                (
                    list(current_segs),
                    current_header,
                    str(current_header_level),
                    current_segs[0].start,
                )
            )
            current_segs = []
            current_size = 0

        # Track the current header context
        if seg.kind == "header":
            current_header = seg.header_text
            current_header_level = seg.header_level

        # Handle oversized single segments — force-split them
        if seg_size > config.max_chunk_size:
            sub_chunks = _force_split_text(seg.text, config.max_chunk_size)
            for j, sub_text in enumerate(sub_chunks):
                sub_seg = _Segment(
                    text=sub_text,
                    start=seg.start + j * config.max_chunk_size,
                    end=min(seg.start + (j + 1) * config.max_chunk_size, seg.end),
                    kind=seg.kind,
                    header_text=seg.header_text,
                    header_level=seg.header_level,
                )
                groups.append(
                    (
                        [sub_seg],
                        current_header,
                        str(current_header_level),
                        sub_seg.start,
                    )
                )
            continue

        current_segs.append(seg)
        current_size += seg_size + 1  # +1 for separator

    # Close last group
    if current_segs:
        groups.append(
            (
                list(current_segs),
                current_header,
                str(current_header_level),
                current_segs[0].start,
            )
        )

    return groups


def _force_split_text(text: str, max_size: int) -> list[str]:
    """Force-split oversized text at sentence boundaries where possible.

    Falls back to word boundaries, then character boundaries.
    Returns empty list for empty/whitespace-only input.
    """
    if not text or not text.strip():
        return []
    if len(text) <= max_size:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_size:
            chunks.append(remaining)
            break

        # Try to split at a sentence boundary within the limit
        candidate = remaining[:max_size]
        split_pos = -1

        # Search backwards for sentence-ending punctuation
        for pos in range(len(candidate) - 1, max(0, len(candidate) // 2), -1):
            if candidate[pos] in ".!?" and (pos + 1 >= len(candidate) or candidate[pos + 1] in " \n\t"):
                split_pos = pos + 1
                break

        # Fall back to word boundary
        if split_pos < 0:
            last_space = candidate.rfind(" ")
            if last_space > len(candidate) // 2:
                split_pos = last_space + 1

        # Fall back to hard split
        if split_pos < 0:
            split_pos = max_size

        chunks.append(remaining[:split_pos].rstrip())
        remaining = remaining[split_pos:].lstrip()

    return [c for c in chunks if c.strip()]


def _merge_small_chunks(
    groups: list[tuple[list[_Segment], str, str, int]],
    config: SmartChunkerConfig,
) -> list[tuple[list[_Segment], str, str, int]]:
    """Merge undersized chunk groups with their neighbors."""
    if len(groups) <= 1:
        return groups

    merged: list[tuple[list[_Segment], str, str, int]] = []
    i = 0
    while i < len(groups):
        segs, header, level, start = groups[i]
        total_size = sum(len(s.text) for s in segs)

        # If undersized and can merge with next
        if total_size < config.min_chunk_size and i + 1 < len(groups):
            next_segs, next_header, next_level, _ = groups[i + 1]
            combined_size = total_size + sum(len(s.text) for s in next_segs)
            if combined_size <= config.max_chunk_size:
                merged.append((segs + next_segs, header or next_header, level or next_level, start))
                i += 2
                continue

        # If undersized and can merge with previous
        if total_size < config.min_chunk_size and merged:
            prev_segs, prev_header, prev_level, prev_start = merged[-1]
            combined_size = sum(len(s.text) for s in prev_segs) + total_size
            if combined_size <= config.max_chunk_size:
                merged[-1] = (prev_segs + segs, prev_header, prev_level, prev_start)
                i += 1
                continue

        merged.append((segs, header, level, start))
        i += 1

    return merged


def _extract_overlap_prefix(text: str, n_sentences: int) -> str:
    """Extract the last N sentences from text for overlap with the next chunk."""
    if n_sentences <= 0 or not text.strip():
        return ""
    sentences = _SENTENCE_END_RE.split(text.rstrip())
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return ""
    overlap = sentences[-n_sentences:]
    return " ".join(overlap)


# ---------------------------------------------------------------------------
# LLM-guided boundary refinement (optional)
# ---------------------------------------------------------------------------


def _refine_boundaries_with_llm(
    segments: list[_Segment],
    boundary_scores: list[tuple[int, float]],
    config: SmartChunkerConfig,
) -> list[tuple[int, float]]:
    """Optionally refine boundary scores using a local LLM.

    Sends the text around each candidate boundary to an LLM and asks whether
    it is a good split point. Returns adjusted scores.

    If the LLM is unavailable, returns scores unchanged.
    """
    if not config.llm_refine:
        return boundary_scores

    try:
        from .llm_extractor import _query_llm, is_available

        if not is_available(backend=config.llm_backend):
            _log.info("llm_refine_skip", reason="no_backend_available")
            return boundary_scores
    except ImportError:
        _log.info("llm_refine_skip", reason="llm_extractor_not_importable")
        return boundary_scores

    refined: list[tuple[int, float]] = []

    for idx, (boundary_idx, score) in enumerate(boundary_scores):
        if boundary_idx <= 0 or boundary_idx >= len(segments):
            refined.append((boundary_idx, score))
            continue

        before_text = segments[boundary_idx - 1].text[-200:]
        after_text = segments[boundary_idx].text[:200]

        prompt = (
            "You are a document chunking assistant. Given the text before and "
            "after a candidate split point, rate how good this split point is "
            "on a scale of 0.0 to 1.0. A good split point separates distinct "
            "topics or sections. Return ONLY a number between 0.0 and 1.0.\n\n"
            f"TEXT BEFORE SPLIT:\n{before_text}\n\n"
            f"TEXT AFTER SPLIT:\n{after_text}\n\n"
            "SCORE (0.0-1.0):"
        )

        try:
            response = _query_llm(prompt, config.llm_model, config.llm_backend)
            # Parse the score from LLM response
            llm_score = _parse_llm_score(response)
            if llm_score is not None:
                # Blend heuristic and LLM scores (60% LLM, 40% heuristic)
                blended = 0.6 * llm_score + 0.4 * score
                refined.append((boundary_idx, blended))
                _log.debug(
                    "llm_boundary_refined",
                    boundary=boundary_idx,
                    heuristic=round(score, 3),
                    llm=round(llm_score, 3),
                    blended=round(blended, 3),
                )
                continue
        except (OSError, ValueError, RuntimeError):
            pass

        refined.append((boundary_idx, score))

    return refined


def _parse_llm_score(response: str) -> float | None:
    """Parse a float score from LLM response text."""
    response = response.strip()
    # Try direct float parse
    try:
        val = float(response)
        if 0.0 <= val <= 1.0:
            return val
    except ValueError:
        pass
    # Search for a non-negative decimal number in the response (reject negatives)
    match = re.search(r"(?<![-\d])\b(0\.\d+|1\.0|0|1)\b", response)
    if match:
        try:
            val = float(match.group(1))
            if 0.0 <= val <= 1.0:
                return val
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def smart_chunk(
    text: str,
    *,
    config: SmartChunkerConfig | None = None,
    source: str = "",
) -> list[Chunk]:
    """Chunk a document using semantic boundary detection.

    Analyzes document structure (headers, paragraphs, code blocks, lists)
    and uses heuristic scoring to prefer natural boundaries over fixed-size
    splits. Optionally refines boundaries with an LLM.

    Args:
        text: Full document text to chunk.
        config: Chunking configuration. Uses defaults if not provided.
        source: Source identifier for metadata. Overrides config.source if set.

    Returns:
        List of Chunk objects with text, position info, and metadata.
    """
    if config is None:
        config = SmartChunkerConfig()

    effective_source = source or config.source

    if not text or not text.strip():
        return []

    # Step 1: Segment the document
    segments = _segment_document(text)
    if not segments:
        return [
            Chunk(
                text=text.strip(),
                index=0,
                start_char=0,
                end_char=len(text),
                metadata={"source": effective_source, "section": "", "position": "only"},
            )
        ]

    # Step 2: Score boundaries between segments
    boundary_scores: list[tuple[int, float]] = []
    for i in range(1, len(segments)):
        score = _score_boundary(segments[i - 1], segments[i])
        boundary_scores.append((i, score))

    # Step 3: Optional LLM refinement
    if config.llm_refine:
        boundary_scores = _refine_boundaries_with_llm(segments, boundary_scores, config)

    # Step 4: Group segments into chunks respecting size limits
    groups = _merge_segments_into_chunks(segments, config)

    # Step 5: Merge small chunks
    groups = _merge_small_chunks(groups, config)

    # Step 6: Build final Chunk objects with overlap and metadata
    chunks: list[Chunk] = []
    prev_overlap = ""

    for i, (segs, header, _level, start_char) in enumerate(groups):
        # Assemble chunk text
        parts: list[str] = []
        if prev_overlap and config.overlap_sentences > 0:
            parts.append(prev_overlap)
        for seg in segs:
            parts.append(seg.text)

        chunk_text = "\n\n".join(parts)
        end_char = segs[-1].end if segs else start_char

        # Determine position label
        if len(groups) == 1:
            position = "only"
        elif i == 0:
            position = "first"
        elif i == len(groups) - 1:
            position = "last"
        else:
            position = "middle"

        # Collect section headers from this chunk's segments
        section_headers = []
        for seg in segs:
            if seg.kind == "header":
                section_headers.append(seg.header_text)

        metadata: dict[str, Any] = {
            "source": effective_source,
            "section": header,
            "position": position,
            "chunk_index": i,
            "chunk_total": len(groups),
            "start_char": start_char,
            "end_char": end_char,
        }
        if section_headers:
            metadata["section_headers"] = section_headers

        # Detect content types present in this chunk
        content_types = sorted({seg.kind for seg in segs})
        if content_types:
            metadata["content_types"] = content_types

        chunk = Chunk(
            text=chunk_text,
            index=i,
            start_char=start_char,
            end_char=end_char,
            metadata=metadata,
        )
        chunks.append(chunk)

        # Prepare overlap for next chunk
        raw_text = "\n\n".join(seg.text for seg in segs)
        prev_overlap = _extract_overlap_prefix(raw_text, config.overlap_sentences)

    _log.info(
        "smart_chunk_complete",
        input_chars=len(text),
        segments=len(segments),
        chunks=len(chunks),
        source=effective_source,
    )

    return chunks


def smart_chunk_blocks(
    blocks: list[dict[str, Any]],
    *,
    config: SmartChunkerConfig | None = None,
    text_fields: tuple[str, ...] = ("Statement", "Description", "Summary", "Content"),
) -> list[dict[str, Any]]:
    """Apply smart chunking to a list of parsed blocks.

    For each block, finds the longest text field and applies smart_chunk.
    Short blocks are returned unchanged. Long blocks are split into multiple
    block dicts with _id suffixed by ".N" (compatible with block_parser
    chunk format).

    Args:
        blocks: List of parsed block dicts (from block_parser).
        config: Chunking configuration.
        text_fields: Tuple of field names to consider for chunking (uses the
            longest one found).

    Returns:
        List of block dicts, with long blocks split into chunks.
    """
    if config is None:
        config = SmartChunkerConfig()

    result: list[dict[str, Any]] = []

    for block in blocks:
        # Find the longest text field
        best_field: str | None = None
        best_text = ""
        for fld in text_fields:
            val = block.get(fld, "")
            if isinstance(val, str) and len(val) > len(best_text):
                best_field = fld
                best_text = val

        if not best_field or len(best_text) <= config.max_chunk_size:
            result.append(block)
            continue

        # Smart-chunk the text field
        source = str(block.get("_id", ""))
        chunks = smart_chunk(best_text, config=config, source=source)

        if len(chunks) <= 1:
            result.append(block)
            continue

        base_id = block.get("_id", "?")
        for chunk in chunks:
            chunk_block: dict[str, Any] = dict(block)  # shallow copy
            chunk_block["_id"] = f"{base_id}.{chunk.index}"
            chunk_block["_chunk_index"] = chunk.index
            chunk_block["_chunk_parent"] = base_id
            chunk_block["_chunk_metadata"] = chunk.metadata
            chunk_block[best_field] = chunk.text
            result.append(chunk_block)

    return result
