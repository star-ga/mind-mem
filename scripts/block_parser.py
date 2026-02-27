#!/usr/bin/env python3
"""Mind Mem Block Parser v1.0 — Self-hosted, zero external dependencies.

Parses Schema v1.0 + v2.0 blocks from markdown files.
Returns JSON array of blocks with fields, lists, and nested structures.

Usage:
    python3 maintenance/block_parser.py <file> [--id-pattern REGEX] [--json] [--active-only]
    python3 maintenance/block_parser.py decisions/DECISIONS.md --json
    python3 maintenance/block_parser.py tasks/TASKS.md --active-only --json

As library:
    from .block_parser import parse_file, parse_blocks
    blocks = parse_file("decisions/DECISIONS.md")
    active = [b for b in blocks if b.get("Status") == "active"]
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, cast

from .observability import get_logger

_log = get_logger("block_parser")

# Maximum input size to parse (100KB). Larger files are truncated with a warning.
MAX_PARSE_SIZE = 100_000


class BlockCorruptedError(ValueError):
    """Raised when a block fails to parse due to malformed content.

    Attributes:
        block_line_number: 1-based line where the block header was found.
        file_path: Path of the file being parsed (if available).
        context: Surrounding lines for debugging.
    """

    def __init__(
        self,
        message: str,
        *,
        block_line_number: int = 0,
        file_path: str = "",
        context: str = "",
    ):
        super().__init__(message)
        self.block_line_number = block_line_number
        self.file_path = file_path
        self.context = context


# Entity ID patterns recognized in block content
_ENTITY_ID_RE = re.compile(
    r"\b(D-\d{8}-\d{3}|T-\d{8}-\d{3}|INC-\d{8}-[a-z0-9-]+"
    r"|PRJ-[a-z0-9-]+|PER-[a-z0-9-]+|TOOL-[a-z0-9-]+"
    r"|C-\d{8}-\d{3}|DREF-\d{8}-\d{3}|I-\d{8}-\d{3})\b"
)

# Date pattern: YYYY-MM-DD
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

# Negation patterns for detecting negated blocks
_NEGATION_RE = re.compile(
    r"\b(not|never|don't|won't|shouldn't|cannot|can't|doesn't|didn't"
    r"|wasn't|isn't|aren't|weren't|hasn't|haven't|wouldn't|couldn't"
    r"|no\s+\w+|none)\b",
    re.IGNORECASE,
)


# Nested dict fields recognized in ConstraintSignature parsing
_DICT_FIELDS = {"scope", "axis", "lifecycle"}


def parse_blocks(text: str) -> list[dict]:
    """Parse all [ID] blocks from text. Returns list of dicts.

    Each dict has:
      - _id: the block ID (e.g. "D-20260213-001")
      - _line: line number where block starts (1-based)
      - Key: Value fields as strings
      - List fields (Sources, History, etc.) as lists of strings
      - ConstraintSignatures as list of dicts (v2.0)
    """
    # Strip UTF-8 BOM if present so the first block header is not masked.
    if text.startswith("﻿"):
        text = text[1:]
    lines = text.split("\n")
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_field: str | None = None
    in_constraint_sigs = False
    current_sig: dict[str, Any] | None = None
    current_sig_field: str | None = None
    in_ops = False
    current_op: dict[str, Any] | None = None
    current_op_field: str | None = None
    in_patch = False

    for i, line in enumerate(lines):
        lineno = i + 1

        # New block header: [ID]
        id_match = re.match(r"^\[([A-Z]+-[^\]]+)\]\s*$", line)
        if id_match:
            if current is not None:
                _finalize_ops(current, in_ops, current_op)
                _finalize_block(current, in_constraint_sigs, current_sig)
                _enrich_fact_keys(current)
                blocks.append(current)
            current = {"_id": id_match.group(1), "_line": lineno}
            current_field = None
            in_constraint_sigs = False
            current_sig = None
            current_sig_field = None
            in_ops = False
            current_op = None
            current_op_field = None
            in_patch = False
            continue

        if current is None:
            continue

        # ConstraintSignatures: or ConstraintSignature: section start (accept both)
        if re.match(r"^ConstraintSignatures?:\s*$", line):
            in_constraint_sigs = True
            current["ConstraintSignatures"] = []
            current_field = "ConstraintSignatures"
            current_sig = None
            current_sig_field = None
            continue

        # Inside ConstraintSignatures block
        if in_constraint_sigs:
            # New signature item: - id: CS-...
            sig_start = re.match(r"^- id:\s*(.+)$", line)
            if sig_start:
                if current_sig is not None:
                    _finalize_sig(current_sig)
                    sigs = current["ConstraintSignatures"]
                    assert isinstance(sigs, list)
                    sigs.append(current_sig)
                current_sig = {"id": sig_start.group(1).strip()}
                current_sig_field = None
                continue

            # Nested list inside sig (e.g. evidence:, conditions:, exceptions:, tags:)
            if current_sig is not None:
                # Sub-field with inline value
                sig_kv = re.match(r"^  ([a-z_]+):\s+(.+)$", line)
                if sig_kv:
                    key, val = sig_kv.group(1), sig_kv.group(2).strip()
                    current_sig_field = key
                    # Parse inline lists [a, b, c]
                    if val.startswith("[") and val.endswith("]"):
                        current_sig[key] = _parse_inline_list(val)
                    # Parse inline dicts { key: val, ... }
                    elif val.startswith("{") and val.endswith("}"):
                        current_sig[key] = _parse_inline_dict(val)
                    else:
                        current_sig[key] = _coerce_value(val)
                    continue

                # Sub-field with no inline value (block list follows)
                sig_key_only = re.match(r"^  ([a-z_]+):\s*$", line)
                if sig_key_only:
                    key = sig_key_only.group(1)
                    current_sig_field = key
                    current_sig[key] = []
                    continue

                # Nested dict sub-fields at 4-space indent (scope, axis, lifecycle)
                nested_kv = re.match(r"^    ([a-z_]+):\s+(.+)$", line)
                if nested_kv and current_sig_field in _DICT_FIELDS:
                    parent = current_sig_field
                    assert parent is not None
                    if not isinstance(current_sig.get(parent), dict):
                        current_sig[parent] = {}
                    parent_dict = cast(dict[str, Any], current_sig[parent])
                    key, val = nested_kv.group(1), nested_kv.group(2).strip()
                    if val.startswith("[") and val.endswith("]"):
                        parent_dict[key] = _parse_inline_list(val)
                    elif val.startswith("{") and val.endswith("}"):
                        parent_dict[key] = _parse_inline_dict(val)
                    else:
                        parent_dict[key] = _coerce_value(val)
                    continue

                # Nested sub-sub-fields at 6-space indent (e.g. scope.time)
                time_kv = re.match(r"^      ([a-z_]+):\s+(.+)$", line)
                if time_kv and current_sig_field in _DICT_FIELDS:
                    parent = current_sig_field
                    assert parent is not None
                    if not isinstance(current_sig.get(parent), dict):
                        current_sig[parent] = {}
                    parent_dict = cast(dict[str, Any], current_sig[parent])
                    if "time" not in parent_dict:
                        parent_dict["time"] = {}
                    key, val = time_kv.group(1), time_kv.group(2).strip()
                    time_dict = cast(dict[str, Any], parent_dict["time"])
                    time_dict[key] = _coerce_value(val)
                    continue

                # List items under sig sub-fields
                sig_list_item = re.match(r"^  - (.+)$", line)
                if sig_list_item and current_sig_field:
                    sig_field_val = current_sig.get(current_sig_field)
                    if isinstance(sig_field_val, list):
                        sig_field_val.append(sig_list_item.group(1).strip())
                    continue

                # End of ConstraintSignatures: blank line or new top-level field
                if line.strip() == "" or re.match(r"^[A-Z][A-Za-z]+:", line):
                    if current_sig is not None:
                        _finalize_sig(current_sig)
                        sigs2 = current["ConstraintSignatures"]
                        assert isinstance(sigs2, list)
                        sigs2.append(current_sig)
                    in_constraint_sigs = False
                    current_sig = None
                    current_sig_field = None
                    # Fall through to normal field parsing if it's a field line
                    if line.strip() == "":
                        continue

            else:
                # Blank line with no current sig = end of ConstraintSignatures
                if line.strip() == "":
                    in_constraint_sigs = False
                    continue

        # Ops: section start
        if re.match(r"^Ops:\s*$", line):
            in_ops = True
            current["Ops"] = []
            current_field = "Ops"
            current_op = None
            current_op_field = None
            in_patch = False
            continue

        # Inside Ops block
        if in_ops:
            # Multiline patch content (4+ space indent while in_patch)
            if in_patch:
                assert current_op is not None
                patch_line = re.match(r"^    (.*)$", line)
                if patch_line:
                    current_op.setdefault("patch", [])
                    patch_val = current_op["patch"]
                    assert isinstance(patch_val, list)
                    patch_val.append(patch_line.group(1))
                    continue
                else:
                    # End of patch block
                    in_patch = False
                    # Join patch lines into single string
                    if "patch" in current_op and isinstance(current_op["patch"], list):
                        current_op["patch"] = "\n".join(current_op["patch"])
                    # Fall through to check if this line starts a new op or field

            # New operation: - op: <type>
            op_start = re.match(r"^\s*-\s*op:\s*(\S+)\s*$", line)
            if op_start:
                if current_op is not None:
                    ops_list = current["Ops"]
                    assert isinstance(ops_list, list)
                    ops_list.append(current_op)
                current_op = {"op": op_start.group(1)}
                current_op_field = None
                in_patch = False
                continue

            # Op sub-fields at 2-space indent
            if current_op is not None:
                op_kv = re.match(r"^\s{2,}([a-z_]+):\s+(.+)$", line)
                if op_kv:
                    key, val = op_kv.group(1), op_kv.group(2).strip()
                    current_op_field = key
                    if val == "|":
                        # Start multiline patch
                        in_patch = True
                        current_op["patch"] = []
                    else:
                        current_op[key] = _coerce_value(val)
                    continue

                # Op sub-field with no inline value
                op_key_only = re.match(r"^\s{2,}([a-z_]+):\s*$", line)
                if op_key_only:
                    key = op_key_only.group(1)
                    current_op_field = key
                    current_op[key] = {}
                    continue

                # Range sub-fields at deeper indent (for replace_range)
                range_kv = re.match(r"^\s{4,}([a-z_]+):\s+(.+)$", line)
                if range_kv and current_op_field == "range":
                    if not isinstance(current_op.get("range"), dict):
                        current_op["range"] = {}
                    range_dict = cast(dict[str, Any], current_op["range"])
                    key, val = range_kv.group(1), range_kv.group(2).strip()
                    range_dict[key] = val
                    continue

            # End of Ops: blank line or new top-level field
            if line.strip() == "" or re.match(r"^[A-Z][A-Za-z]+:", line):
                if current_op is not None:
                    ops_list2 = current["Ops"]
                    assert isinstance(ops_list2, list)
                    ops_list2.append(current_op)
                in_ops = False
                current_op = None
                current_op_field = None
                in_patch = False
                if line.strip() == "":
                    continue
                # Fall through to normal field parsing

        # Normal field: Key: Value
        field_match = re.match(r"^([A-Z][A-Za-z]+):\s+(.+)$", line)
        if field_match:
            key, val = field_match.group(1), field_match.group(2).strip()
            current[key] = val
            current_field = key
            continue

        # Field with no inline value (list follows)
        field_empty = re.match(r"^([A-Z][A-Za-z]+):\s*$", line)
        if field_empty:
            key = field_empty.group(1)
            current[key] = []
            current_field = key
            continue

        # List item under current field
        list_item = re.match(r"^- (.+)$", line)
        if list_item and current_field:
            field_val = current.get(current_field)
            if isinstance(field_val, list):
                field_val.append(list_item.group(1).strip())
            elif isinstance(field_val, str):
                # Convert scalar to list
                current[current_field] = [field_val, list_item.group(1).strip()]
            continue

        # Indented list item (History entries etc.)
        indented_item = re.match(r"^\s+- (.+)$", line)
        if indented_item and current_field:
            field_val2 = current.get(current_field)
            if isinstance(field_val2, list):
                field_val2.append(indented_item.group(1).strip())
            continue

        # Continuation line: indented text (2+ spaces, not a list item) appends to previous scalar field
        if current_field and line.startswith("  ") and not line.lstrip().startswith("-"):
            continuation = line.strip()
            cont_val = current.get(current_field)
            if continuation and isinstance(cont_val, str):
                current[current_field] = cont_val + "\n" + continuation
                continue

        # Blank line: don't break current block
        if line.strip() == "":
            continue

        # Section separator
        if line.startswith("---"):
            if current is not None:
                _finalize_ops(current, in_ops, current_op)
                _finalize_block(current, in_constraint_sigs, current_sig)
                _enrich_fact_keys(current)
                blocks.append(current)
                current = None
                current_field = None
                in_constraint_sigs = False
                current_sig = None
                in_ops = False
                current_op = None
                in_patch = False
            continue

    # Last block
    if current is not None:
        _finalize_ops(current, in_ops, current_op)
        _finalize_block(current, in_constraint_sigs, current_sig)
        _enrich_fact_keys(current)
        blocks.append(current)

    return blocks


def _enrich_fact_keys(block: dict) -> None:
    """Extract and attach fact key fields: _entities, _dates, _has_negation."""
    # Collect all text content from the block (non-underscore fields)
    texts = []
    for key, val in block.items():
        if key.startswith("_"):
            continue
        if isinstance(val, str):
            texts.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    texts.append(item)
    full_text = " ".join(texts)

    block["_entities"] = sorted(set(_ENTITY_ID_RE.findall(full_text)))
    block["_dates"] = sorted(set(_DATE_RE.findall(full_text)))
    block["_has_negation"] = bool(_NEGATION_RE.search(full_text))


def _finalize_ops(block: dict[str, Any], in_ops: bool, current_op: dict[str, Any] | None) -> None:
    """Finalize Ops section, closing any open operation."""
    if in_ops and current_op is not None:
        if "Ops" not in block:
            block["Ops"] = []
        # Join patch lines if still a list
        if "patch" in current_op and isinstance(current_op["patch"], list):
            current_op["patch"] = "\n".join(current_op["patch"])
        ops = block["Ops"]
        assert isinstance(ops, list)
        ops.append(current_op)


def _finalize_block(block: dict[str, Any], in_sigs: bool, current_sig: dict[str, Any] | None) -> None:
    """Finalize a block, closing any open ConstraintSignatures."""
    if in_sigs and current_sig is not None:
        _finalize_sig(current_sig)
        if "ConstraintSignatures" not in block:
            block["ConstraintSignatures"] = []
        sigs = block["ConstraintSignatures"]
        assert isinstance(sigs, list)
        sigs.append(current_sig)


def _finalize_sig(sig: dict[str, Any]) -> None:
    """Ensure sig has all expected fields with defaults."""
    defaults = {
        "domain": "other",
        "subject": "",
        "predicate": "",
        "object": "",
        "modality": "may",
        "priority": 1,
        "scope": {"projects": [], "channels": [], "users": [], "time": {"start": None, "end": None}},
        "conditions": [],
        "exceptions": [],
        "evidence": [],
        "tags": [],
    }
    for k, v in defaults.items():
        if k not in sig:
            sig[k] = v


def _parse_inline_list(s: str) -> list[Any]:
    """Parse [a, b, c] -> ['a', 'b', 'c']. Supports quoted strings for values with commas."""
    inner = s[1:-1].strip()
    if not inner:
        return []
    # Quote-aware splitting: "value, with comma", other
    if '"' in inner:
        items: list[str] = []
        current: list[str] = []
        in_quotes = False
        for ch in inner:
            if ch == '"':
                in_quotes = not in_quotes
            elif ch == "," and not in_quotes:
                items.append("".join(current).strip().strip('"'))
                current = []
            else:
                current.append(ch)
        if current:
            items.append("".join(current).strip().strip('"'))
        return [_coerce_value(x) for x in items if x]
    return [_coerce_value(x.strip()) for x in inner.split(",") if x.strip()]


def _parse_inline_dict(s: str) -> dict[str, Any]:
    """Parse { key: val, key2: val2 } -> dict. Quote-aware comma splitting."""
    inner = s[1:-1].strip()
    if not inner:
        return {}
    # Quote-aware split on commas (same pattern as _parse_inline_list)
    if '"' in inner:
        pairs: list[str] = []
        current: list[str] = []
        in_quotes = False
        for ch in inner:
            if ch == '"':
                in_quotes = not in_quotes
            elif ch == "," and not in_quotes:
                pairs.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            pairs.append("".join(current).strip())
    else:
        pairs = [p.strip() for p in inner.split(",")]
    result = {}
    for pair in pairs:
        if ":" in pair:
            k, v = pair.split(":", 1)
            result[k.strip()] = _coerce_value(v.strip().strip('"'))
    return result


def _coerce_value(s: str) -> Any:
    """Coerce string to int/float/bool/None if appropriate."""
    if s in ("null", "none", "None"):
        return None
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    try:
        return int(s)
    except (ValueError, TypeError):
        pass
    try:
        return float(s)
    except (ValueError, TypeError):
        pass
    return s


def parse_file(filepath: str, *, strict: bool = False) -> list[dict]:
    """Parse blocks from a file path. Files >100KB are truncated.

    Args:
        filepath: Path to the markdown file.
        strict: If True, raise BlockCorruptedError on any per-block
            parse failure instead of skipping the block.

    Returns:
        List of parsed block dicts.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read(MAX_PARSE_SIZE + 1)
    if len(content) > MAX_PARSE_SIZE:
        content = content[:MAX_PARSE_SIZE]
        # Try to truncate at a block boundary to avoid corrupt partial blocks
        last_boundary = content.rfind("\n[")
        if last_boundary > 0:
            # Verify it looks like a block header (e.g. \n[D- or \n[T- etc)
            after = content[last_boundary + 2 : last_boundary + 4]
            if after and after[0].isupper():
                content = content[:last_boundary]
        else:
            _log.warning(
                "MAX_PARSE_SIZE truncation could not find a block boundary; result may contain a partial block"
            )

    blocks = parse_blocks(content)
    valid: list[dict] = []
    for block in blocks:
        try:
            _validate_block(block)
            valid.append(block)
        except (ValueError, TypeError, KeyError) as exc:
            line = block.get("_line", 0)
            bid = block.get("_id", "<unknown>")
            lines = content.split("\n")
            start = max(0, line - 3)
            end = min(len(lines), line + 3)
            ctx = "\n".join(lines[start:end])
            _log.warning(
                "block_corrupted",
                block_id=bid,
                line=line,
                file=filepath,
                error=str(exc),
                context=ctx,
            )
            if strict:
                raise BlockCorruptedError(
                    f"Corrupted block {bid} at line {line} in {filepath}: {exc}",
                    block_line_number=line,
                    file_path=filepath,
                    context=ctx,
                ) from exc
    return valid


def _validate_block(block: dict) -> None:
    """Basic validation of a parsed block. Raises ValueError on problems."""
    if not block.get("_id"):
        raise ValueError("Block missing required _id field")


def chunk_block(
    block: dict,
    max_tokens: int = 400,
    overlap: int = 50,
) -> list[dict]:
    """Split a long block into overlapping chunks for improved recall.

    Each chunk preserves all metadata from the parent block. The chunk's _id
    gets a ".N" suffix (e.g., "D-20260129-001.0", "D-20260129-001.1").
    The Statement/Description field is split into overlapping windows.

    Short blocks (under max_tokens words) are returned as-is in a single-item list.

    Args:
        block: Parsed block dict.
        max_tokens: Maximum words per chunk before splitting.
        overlap: Number of words to overlap between adjacent chunks.

    Returns:
        List of chunk dicts (1 item if block is short enough).
    """
    # Find the primary text field to chunk
    text_field = None
    text_value = ""
    for field in ("Statement", "Description", "Summary", "Title"):
        val = block.get(field, "")
        if isinstance(val, str) and len(val) > len(text_value):
            text_field = field
            text_value = val

    if not text_field or not text_value:
        return [block]

    words = text_value.split()
    if len(words) <= max_tokens:
        return [block]

    # Split into overlapping windows
    chunks = []
    base_id = block.get("_id", "?")
    step = max(1, max_tokens - overlap)
    idx = 0

    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk_text = " ".join(words[start:end])

        chunk = dict(block)  # shallow copy
        chunk["_id"] = f"{base_id}.{idx}"
        chunk["_chunk_index"] = idx
        chunk["_chunk_parent"] = base_id
        chunk[text_field] = chunk_text
        chunks.append(chunk)

        idx += 1
        if end >= len(words):
            break
        start += step

    return chunks


def deduplicate_chunks(results: list[dict]) -> list[dict]:
    """Deduplicate chunked results by base block _id, keeping highest score."""
    best: dict[str, dict] = {}
    for r in results:
        rid = r.get("_id", "")
        # Extract base ID (strip .N suffix)
        base = rid.rsplit(".", 1)[0] if "." in rid and rid.rsplit(".", 1)[1].isdigit() else rid
        if base not in best or r.get("score", 0) > best[base].get("score", 0):
            best[base] = r
    # Preserve original order for equal-score items
    seen = set()
    deduped = []
    for r in results:
        rid = r.get("_id", "")
        base = rid.rsplit(".", 1)[0] if "." in rid and rid.rsplit(".", 1)[1].isdigit() else rid
        if base not in seen and best.get(base) is r:
            seen.add(base)
            deduped.append(r)
    return deduped


def get_active(blocks: list[dict], status_field: str = "Status", active_value: str = "active") -> list[dict]:
    """Filter blocks to only active ones."""
    return [b for b in blocks if b.get(status_field) == active_value]


def get_by_id(blocks: list[dict], block_id: str) -> dict | None:
    """Find a block by its _id."""
    for b in blocks:
        if b.get("_id") == block_id:
            return b
    return None


def extract_refs(blocks: list[dict]) -> set[str]:
    """Extract all cross-reference IDs from block fields."""
    refs = set()
    for block in blocks:
        for key, val in block.items():
            if key.startswith("_"):
                continue
            if isinstance(val, str):
                refs.update(_ENTITY_ID_RE.findall(val))
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        refs.update(_ENTITY_ID_RE.findall(item))
    return refs


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mind Mem Block Parser v1.0")
    parser.add_argument("file", help="Path to markdown file to parse")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--active-only", action="store_true", help="Only show active blocks")
    parser.add_argument("--id-pattern", help="Filter by ID regex pattern")
    parser.add_argument("--field", help="Extract specific field from all blocks")
    args = parser.parse_args()

    blocks = parse_file(args.file)

    if args.active_only:
        blocks = get_active(blocks)

    if args.id_pattern:
        pat = re.compile(args.id_pattern)
        blocks = [b for b in blocks if pat.search(b.get("_id", ""))]

    if args.field:
        for b in blocks:
            val = b.get(args.field)
            if val is not None:
                if args.json:
                    print(json.dumps({"_id": b["_id"], args.field: val}))
                else:
                    print(f"{b['_id']}: {val}")
        sys.exit(0)

    if args.json:
        print(json.dumps(blocks, indent=2, default=str))
    else:
        for b in blocks:
            print(f"\n[{b['_id']}] (line {b.get('_line', '?')})")
            for k, v in b.items():
                if k.startswith("_"):
                    continue
                if isinstance(v, list):
                    print(f"  {k}:")
                    for item in v:
                        if isinstance(item, dict):
                            print(f"    - {json.dumps(item, default=str)}")
                        else:
                            print(f"    - {item}")
                else:
                    print(f"  {k}: {v}")
