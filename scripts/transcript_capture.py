#!/usr/bin/env python3
"""mind-mem Transcript JSONL Capture. Zero external deps.

Scans Claude Code transcript JSONL files for:
- Decision-like language (same patterns as capture.py)
- User corrections ("don't do X", "always do Y", "never use Z")
- Convention discoveries ("the pattern is", "the convention is")
- Bug fix insights ("the fix was", "the issue was", "root cause")

Extracts structured signals and writes them to SIGNALS.md for review.

Usage:
    python3 scripts/transcript_capture.py workspace/ --transcript path/to/session.jsonl
    python3 scripts/transcript_capture.py workspace/ --scan-recent  # scans ~/.claude/projects/
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture import CONFIDENCE_TO_PRIORITY, append_signals, extract_structure
from observability import get_logger, metrics

_log = get_logger("transcript_capture")

# Patterns specific to transcript capture (user corrections, conventions)
TRANSCRIPT_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # User corrections — highest value signals
    (re.compile(r"\bdon'?t\s+(ever\s+)?use\b", re.I), "correction", "high"),
    (re.compile(r"\bnever\s+(do|use|add|make|create)\b", re.I), "correction", "high"),
    (re.compile(r"\balways\s+(use|do|add|make|check|ensure)\b", re.I), "correction", "high"),
    (re.compile(r"\bstop\s+(doing|using|adding)\b", re.I), "correction", "high"),
    (re.compile(r"\bthat'?s\s+(wrong|incorrect|not right)\b", re.I), "correction", "high"),
    (re.compile(r"\bno,?\s+(use|do|try|make)\b", re.I), "correction", "medium"),
    (re.compile(r"\binstead\s+(of|use|do)\b", re.I), "correction", "medium"),

    # Convention discoveries
    (re.compile(r"\bthe (pattern|convention|standard|rule) is\b", re.I), "convention", "high"),
    (re.compile(r"\bwe (follow|use|prefer|adopt)\b", re.I), "convention", "medium"),
    (re.compile(r"\bour (convention|pattern|standard|approach) is\b", re.I), "convention", "high"),

    # Bug fix insights
    (re.compile(r"\bthe (fix|solution|answer) (is|was)\b", re.I), "insight", "medium"),
    (re.compile(r"\broot cause (is|was)\b", re.I), "insight", "high"),
    (re.compile(r"\bthe (issue|problem|bug) (is|was)\b", re.I), "insight", "medium"),

    # Architectural decisions in conversation
    (re.compile(r"\blet'?s (go with|use|switch to|adopt)\b", re.I), "decision", "medium"),
    (re.compile(r"\bwe should (use|switch|migrate|adopt)\b", re.I), "decision", "medium"),
    (re.compile(r"\bdecided to\b", re.I), "decision", "high"),
]

# Cross-reference pattern (same as capture.py)
XREF_PATTERN = re.compile(r"\b[A-Z]+-\d{8}-\d{3}\b")


def parse_transcript(jsonl_path: str) -> list[dict]:
    """Parse a JSONL transcript file into message dicts.

    Args:
        jsonl_path: Path to the .jsonl transcript file

    Returns:
        List of message dicts with role, content, and line number
    """
    messages = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Extract text content from various message formats
            content = ""
            if isinstance(msg.get("content"), str):
                content = msg["content"]
            elif isinstance(msg.get("content"), list):
                # Claude API format: content is list of blocks
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content += block.get("text", "") + "\n"
                    elif isinstance(block, str):
                        content += block + "\n"
            elif isinstance(msg.get("message"), str):
                content = msg["message"]

            if content.strip():
                messages.append({
                    "line": i,
                    "role": msg.get("role", msg.get("type", "unknown")),
                    "content": content.strip(),
                })

    return messages


def scan_transcript(jsonl_path: str, role_filter: str | None = None) -> list[dict]:
    """Scan a transcript for decision/correction/convention signals.

    Args:
        jsonl_path: Path to .jsonl transcript
        role_filter: Only scan messages from this role (e.g., "user" for corrections)

    Returns:
        List of signal dicts compatible with append_signals()
    """
    messages = parse_transcript(jsonl_path)
    signals = []

    for msg in messages:
        role = msg["role"]
        if role_filter and role != role_filter:
            continue

        # Split content into lines for per-line scanning
        lines = msg["content"].split("\n")
        for line_offset, text in enumerate(lines):
            text = text.strip()
            if not text or len(text) < 15:
                continue
            if text.startswith("#") or text.startswith("```"):
                continue
            if XREF_PATTERN.search(text):
                continue  # Already cross-referenced

            for pattern, sig_type, confidence in TRANSCRIPT_PATTERNS:
                if pattern.search(text):
                    structure = extract_structure(text, sig_type, pattern.pattern)
                    signals.append({
                        "line": msg["line"],
                        "type": sig_type,
                        "text": text[:200],
                        "pattern": pattern.pattern,
                        "confidence": confidence,
                        "priority": CONFIDENCE_TO_PRIORITY.get(confidence, "P2"),
                        "structure": structure,
                        "source_role": role,
                    })
                    break  # First match wins

    _log.info("transcript_scan_complete", path=jsonl_path,
              messages=len(messages), signals=len(signals))
    metrics.inc("transcript_signals", len(signals))
    return signals


def find_recent_transcripts(days: int = 3) -> list[str]:
    """Find recent Claude Code transcript JSONL files.

    Looks in common locations:
    - ~/.claude/projects/*/
    - Current directory

    Returns list of .jsonl file paths sorted by modification time (newest first).
    """
    paths = []
    cutoff = datetime.now() - timedelta(days=days)

    # Check ~/.claude/projects/
    claude_dir = os.path.expanduser("~/.claude/projects")
    if os.path.isdir(claude_dir):
        for root, dirs, files in os.walk(claude_dir):
            for f in files:
                if f.endswith(".jsonl"):
                    path = os.path.join(root, f)
                    try:
                        mtime = datetime.fromtimestamp(os.path.getmtime(path))
                        if mtime >= cutoff:
                            paths.append((path, mtime))
                    except OSError:
                        pass

    # Sort by modification time, newest first
    paths.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in paths]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="mind-mem Transcript Capture")
    parser.add_argument("workspace", nargs="?", default=".")
    parser.add_argument("--transcript", "-t", help="Path to specific .jsonl transcript")
    parser.add_argument("--scan-recent", action="store_true",
                        help="Scan recent transcripts from ~/.claude/projects/")
    parser.add_argument("--days", type=int, default=3, help="Days to look back (default: 3)")
    parser.add_argument("--role", choices=["user", "assistant"], default=None,
                        help="Only scan messages from this role")
    parser.add_argument("--dry-run", action="store_true", help="Show signals without writing")
    args = parser.parse_args()

    ws = os.path.abspath(args.workspace)
    today = datetime.now().strftime("%Y-%m-%d")

    if args.transcript:
        transcripts = [args.transcript]
    elif args.scan_recent:
        transcripts = find_recent_transcripts(days=args.days)
        if not transcripts:
            print(f"No transcripts found in last {args.days} day(s)")
            return
        print(f"Found {len(transcripts)} recent transcript(s)")
    else:
        parser.print_help()
        return

    total_detected = 0
    total_written = 0

    for t_path in transcripts:
        signals = scan_transcript(t_path, role_filter=args.role)
        total_detected += len(signals)

        if signals and not args.dry_run:
            written = append_signals(ws, signals, today)
            total_written += written

        if signals:
            print(f"\n{t_path}:")
            for sig in signals[:5]:  # Show first 5
                role_tag = f" [{sig.get('source_role', '?')}]" if sig.get("source_role") else ""
                print(f"  [{sig['confidence']}] {sig['type']}{role_tag}: {sig['text'][:80]}...")
            if len(signals) > 5:
                print(f"  ... and {len(signals) - 5} more")

    print(f"\nTotal: {total_detected} signal(s) detected, {total_written} new signal(s) written")


if __name__ == "__main__":
    main()
