#!/usr/bin/env python3
"""Mind Mem Block Parser v1.0 — Self-hosted, zero external dependencies.

Parses Schema v1.0 + v2.0 blocks from markdown files.
Returns JSON array of blocks with fields, lists, and nested structures.

Usage:
    python3 maintenance/block_parser.py <file> [--id-pattern REGEX] [--json] [--active-only]
    python3 maintenance/block_parser.py decisions/DECISIONS.md --json
    python3 maintenance/block_parser.py tasks/TASKS.md --active-only --json

As library:
    from block_parser import parse_file, parse_blocks
    blocks = parse_file("decisions/DECISIONS.md")
    active = [b for b in blocks if b.get("Status") == "active"]
"""

from __future__ import annotations

import json
import re
import sys

# Maximum input size to parse (100KB). Larger files are truncated with a warning.
MAX_PARSE_SIZE = 100_000

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


def parse_blocks(text: str) -> list[dict]:
    """Parse all [ID] blocks from text. Returns list of dicts.

    Each dict has:
      - _id: the block ID (e.g. "D-20260213-001")
      - _line: line number where block starts (1-based)
      - Key: Value fields as strings
      - List fields (Sources, History, etc.) as lists of strings
      - ConstraintSignatures as list of dicts (v2.0)
    """
    lines = text.split("\n")
    blocks = []
    current = None
    current_field = None
    in_constraint_sigs = False
    current_sig = None
    current_sig_field = None
    in_ops = False
    current_op = None
    current_op_field = None
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
                    current["ConstraintSignatures"].append(current_sig)
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
                DICT_FIELDS = {"scope", "axis", "lifecycle"}
                nested_kv = re.match(r"^    ([a-z_]+):\s+(.+)$", line)
                if nested_kv and current_sig_field in DICT_FIELDS:
                    parent = current_sig_field
                    if not isinstance(current_sig.get(parent), dict):
                        current_sig[parent] = {}
                    key, val = nested_kv.group(1), nested_kv.group(2).strip()
                    if val.startswith("[") and val.endswith("]"):
                        current_sig[parent][key] = _parse_inline_list(val)
                    elif val.startswith("{") and val.endswith("}"):
                        current_sig[parent][key] = _parse_inline_dict(val)
                    else:
                        current_sig[parent][key] = _coerce_value(val)
                    continue

                # Nested sub-sub-fields at 6-space indent (e.g. scope.time)
                time_kv = re.match(r"^      ([a-z_]+):\s+(.+)$", line)
                if time_kv and current_sig_field in DICT_FIELDS:
                    parent = current_sig_field
                    if not isinstance(current_sig.get(parent), dict):
                        current_sig[parent] = {}
                    if "time" not in current_sig[parent]:
                        current_sig[parent]["time"] = {}
                    key, val = time_kv.group(1), time_kv.group(2).strip()
                    current_sig[parent]["time"][key] = _coerce_value(val)
                    continue

                # List items under sig sub-fields
                sig_list_item = re.match(r"^  - (.+)$", line)
                if sig_list_item and current_sig_field:
                    if isinstance(current_sig.get(current_sig_field), list):
                        current_sig[current_sig_field].append(sig_list_item.group(1).strip())
                    continue

                # End of ConstraintSignatures: blank line or new top-level field
                if line.strip() == "" or re.match(r"^[A-Z][A-Za-z]+:", line):
                    if current_sig is not None:
                        _finalize_sig(current_sig)
                        current["ConstraintSignatures"].append(current_sig)
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
                patch_line = re.match(r"^    (.*)$", line)
                if patch_line:
                    current_op.setdefault("patch", [])
                    current_op["patch"].append(patch_line.group(1))
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
                    current["Ops"].append(current_op)
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
                    key, val = range_kv.group(1), range_kv.group(2).strip()
                    current_op["range"][key] = val
                    continue

            # End of Ops: blank line or new top-level field
            if line.strip() == "" or re.match(r"^[A-Z][A-Za-z]+:", line):
                if current_op is not None:
                    current["Ops"].append(current_op)
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
            if isinstance(current.get(current_field), list):
                current[current_field].append(list_item.group(1).strip())
            elif isinstance(current.get(current_field), str):
                # Convert scalar to list
                current[current_field] = [current[current_field], list_item.group(1).strip()]
            continue

        # Indented list item (History entries etc.)
        indented_item = re.match(r"^\s+- (.+)$", line)
        if indented_item and current_field:
            if isinstance(current.get(current_field), list):
                current[current_field].append(indented_item.group(1).strip())
            continue

        # Continuation line: indented text (2+ spaces, not a list item) appends to previous scalar field
        if current_field and line.startswith("  ") and not line.lstrip().startswith("-"):
            continuation = line.strip()
            if continuation and isinstance(current.get(current_field), str):
                current[current_field] += "\n" + continuation
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


def _finalize_ops(block, in_ops, current_op):
    """Finalize Ops section, closing any open operation."""
    if in_ops and current_op is not None:
        if "Ops" not in block:
            block["Ops"] = []
        # Join patch lines if still a list
        if "patch" in current_op and isinstance(current_op["patch"], list):
            current_op["patch"] = "\n".join(current_op["patch"])
        block["Ops"].append(current_op)


def _finalize_block(block, in_sigs, current_sig):
    """Finalize a block, closing any open ConstraintSignatures."""
    if in_sigs and current_sig is not None:
        _finalize_sig(current_sig)
        if "ConstraintSignatures" not in block:
            block["ConstraintSignatures"] = []
        block["ConstraintSignatures"].append(current_sig)


def _finalize_sig(sig):
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


def _parse_inline_list(s):
    """Parse [a, b, c] -> ['a', 'b', 'c']. Supports quoted strings for values with commas."""
    inner = s[1:-1].strip()
    if not inner:
        return []
    # Quote-aware splitting: "value, with comma", other
    if '"' in inner:
        items = []
        current = []
        in_quotes = False
        for ch in inner:
            if ch == '"':
                in_quotes = not in_quotes
            elif ch == ',' and not in_quotes:
                items.append(''.join(current).strip().strip('"'))
                current = []
            else:
                current.append(ch)
        if current:
            items.append(''.join(current).strip().strip('"'))
        return [_coerce_value(x) for x in items if x]
    return [_coerce_value(x.strip()) for x in inner.split(",") if x.strip()]


def _parse_inline_dict(s):
    """Parse { key: val, key2: val2 } -> dict. Quote-aware comma splitting."""
    inner = s[1:-1].strip()
    if not inner:
        return {}
    # Quote-aware split on commas (same pattern as _parse_inline_list)
    if '"' in inner:
        pairs = []
        current = []
        in_quotes = False
        for ch in inner:
            if ch == '"':
                in_quotes = not in_quotes
            elif ch == ',' and not in_quotes:
                pairs.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            pairs.append(''.join(current).strip())
    else:
        pairs = [p.strip() for p in inner.split(",")]
    result = {}
    for pair in pairs:
        if ":" in pair:
            k, v = pair.split(":", 1)
            result[k.strip()] = _coerce_value(v.strip().strip('"'))
    return result


def _coerce_value(s):
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


def parse_file(filepath: str) -> list[dict]:
    """Parse blocks from a file path. Files >100KB are truncated."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read(MAX_PARSE_SIZE + 1)
    if len(content) > MAX_PARSE_SIZE:
        content = content[:MAX_PARSE_SIZE]
    return parse_blocks(content)


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
    pattern = re.compile(
        r"\b(D-\d{8}-\d{3}|T-\d{8}-\d{3}|INC-\d{8}-[a-z0-9-]+"
        r"|PRJ-[a-z0-9-]+|PER-[a-z0-9-]+|TOOL-[a-z0-9-]+"
        r"|C-\d{8}-\d{3}|DREF-\d{8}-\d{3}|I-\d{8}-\d{3})\b"
    )
    for block in blocks:
        for key, val in block.items():
            if key.startswith("_"):
                continue
            if isinstance(val, str):
                refs.update(pattern.findall(val))
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        refs.update(pattern.findall(item))
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
