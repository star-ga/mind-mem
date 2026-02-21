#!/usr/bin/env python3
"""mind-mem Auto-Capture Engine with Structured Extraction. Zero external deps.

SAFETY: This engine ONLY writes to intelligence/SIGNALS.md.
It NEVER writes to decisions/DECISIONS.md or tasks/TASKS.md directly.
All captured signals must go through /apply to become formal blocks.
This prevents memory poisoning from automated extraction errors.

Structured extraction pipeline:
1. Scans daily log for decision/task-like language
2. Extracts structured fields (subject, predicate, confidence)
3. Classifies signal priority based on language strength
4. Deduplicates against existing signals
5. Appends to intelligence/SIGNALS.md with full metadata

Usage:
    python3 scripts/capture.py [workspace_path]
    python3 scripts/capture.py . --scan-all
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mind_filelock import FileLock
from observability import get_logger, metrics

_log = get_logger("capture")


# ---------------------------------------------------------------------------
# Pattern definitions with confidence and priority
# ---------------------------------------------------------------------------

DECISION_PATTERNS = [
    # High confidence decision patterns
    (r"\bwe(?:'ll| will| decided| agreed| chose| went with)\b", "decision", "high"),
    (r"\bdecided to\b", "decision", "high"),
    (r"\bfrom now on\b", "decision", "high"),
    (r"\bgoing forward\b", "decision", "high"),
    (r"\bno longer\b", "decision", "high"),

    # Medium confidence decision patterns
    (r"\blet'?s go with\b", "decision", "medium"),
    (r"\bswitching to\b", "decision", "medium"),
    (r"\binstead of\b", "decision", "medium"),
    (r"\bwe('re| are) (moving|switching|changing)\b", "decision", "medium"),
    (r"\bapproved\b", "decision", "medium"),
    (r"\bfinalized\b", "decision", "medium"),

    # Low confidence decision patterns (contextual)
    (r"\bprefer\b.*\bover\b", "decision", "low"),
    (r"\bdefault\b.*\bwill be\b", "decision", "low"),

    # High confidence task patterns
    (r"\baction item\b", "task", "high"),
    (r"\bdeadline\b", "task", "high"),
    (r"\bby end of\b", "task", "high"),
    (r"\bmust\b.*\bbefore\b", "task", "high"),
    (r"\bblocked on\b", "task", "high"),

    # Medium confidence task patterns
    (r"\bneed to\b", "task", "medium"),
    (r"\btodo\b", "task", "medium"),
    (r"\bfollow up\b", "task", "medium"),
    (r"\bshould\b.*\bbefore\b", "task", "medium"),
    (r"\bnext step\b", "task", "medium"),
    (r"\brequires\b", "task", "medium"),

    # Low confidence task patterns
    (r"\bwould be nice\b", "task", "low"),
    (r"\bsomeday\b", "task", "low"),
    (r"\bmaybe\b.*\bshould\b", "task", "low"),
]

# Patterns that indicate a line IS already cross-referenced
XREF_PATTERN = re.compile(r"\b[DT]-\d{8}-\d{3}\b")

# Priority mapping from confidence
CONFIDENCE_TO_PRIORITY = {"high": "P1", "medium": "P2", "low": "P3"}


def content_hash(text: str) -> str:
    """SHA256 hash of normalized text for dedup.

    Normalization: lowercase, collapse whitespace, strip.
    """
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Structured extraction
# ---------------------------------------------------------------------------

def extract_structure(text: str, sig_type: str, pattern: str) -> dict:
    """Extract structured fields from captured text.

    Returns dict with subject, predicate, object, and tags.
    Uses simple heuristic extraction — not NLP, but good enough
    for triage purposes.
    """
    structure = {
        "subject": "",
        "predicate": "",
        "object": "",
        "tags": [],
    }

    text_lower = text.lower()

    # Extract subject: first noun-like phrase before the verb
    # Decision: "We decided to use PostgreSQL" -> subject="we"
    # Task: "Need to fix the auth module" -> subject=""
    if sig_type == "decision":
        subject_match = re.match(r"^([\w\s]+?)(?:decided|agreed|chose|will|'ll)", text_lower)
        if subject_match:
            structure["subject"] = subject_match.group(1).strip()

    # Extract object: key phrase after the verb
    obj_patterns = [
        r"(?:use|using|chose|with|to)\s+(\S+(?:\s+\S+)?)",
        r"(?:switching to|moving to)\s+(\S+(?:\s+\S+)?)",
        r"(?:fix|update|implement|add|remove|create)\s+(?:the\s+)?(\S+(?:\s+\S+)?)",
    ]
    for pat in obj_patterns:
        m = re.search(pat, text_lower)
        if m:
            structure["object"] = m.group(1).strip()[:50]
            break

    # Extract tags from common keywords
    tag_keywords = {
        "database": "database", "db": "database", "postgres": "database",
        "auth": "security", "security": "security", "token": "security",
        "api": "api", "endpoint": "api", "rest": "api",
        "deploy": "deployment", "ci": "deployment", "cd": "deployment",
        "test": "testing", "spec": "testing", "coverage": "testing",
        "bug": "bugfix", "fix": "bugfix", "error": "bugfix",
        "infra": "infrastructure", "server": "infrastructure",
        "perf": "performance", "latency": "performance", "slow": "performance",
    }
    for keyword, tag in tag_keywords.items():
        if keyword in text_lower and tag not in structure["tags"]:
            structure["tags"].append(tag)

    return structure


# ---------------------------------------------------------------------------
# Log scanning
# ---------------------------------------------------------------------------

def find_today_log(workspace: str) -> tuple[str | None, str]:
    """Find today's daily log file."""
    today = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(workspace, "memory", f"{today}.md")
    if os.path.isfile(path):
        return path, today
    return None, today


def find_all_logs(workspace: str, days: int = 7) -> list[tuple[str, str]]:
    """Find recent daily log files for batch scanning."""
    logs = []
    memory_dir = os.path.join(workspace, "memory")
    if not os.path.isdir(memory_dir):
        return logs

    cutoff = datetime.now()
    for i in range(days):
        date = cutoff.strftime("%Y-%m-%d")
        path = os.path.join(memory_dir, f"{date}.md")
        if os.path.isfile(path):
            logs.append((path, date))
        cutoff = cutoff - timedelta(days=1)

    return logs


def scan_log(log_path: str) -> list[dict]:
    """Scan a daily log for uncaptured decisions/tasks with structured extraction."""
    signals = []
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Skip if already cross-referenced
        if XREF_PATTERN.search(stripped):
            continue

        for pattern, sig_type, confidence in DECISION_PATTERNS:
            if re.search(pattern, stripped, re.IGNORECASE):
                structure = extract_structure(stripped, sig_type, pattern)
                signals.append({
                    "line": i,
                    "type": sig_type,
                    "text": stripped[:150],
                    "pattern": pattern,
                    "confidence": confidence,
                    "priority": CONFIDENCE_TO_PRIORITY[confidence],
                    "structure": structure,
                })
                break  # one match per line is enough

    return signals


def append_signals(workspace: str, signals: list[dict], date_str: str) -> int:
    """Append captured signals to SIGNALS.md with structured metadata."""
    signals_path = os.path.join(workspace, "intelligence", "SIGNALS.md")
    if not os.path.isfile(signals_path):
        return 0

    # Check existing signals to avoid duplicates via content hash
    with open(signals_path, "r", encoding="utf-8") as f:
        existing = f.read()

    # Build set of existing content hashes for O(1) lookup
    existing_hashes = set(re.findall(r"ContentHash: ([a-f0-9]+)", existing))

    new_signals = []
    for sig in signals:
        sig_hash = content_hash(sig["text"])
        # Skip if content hash already exists, or fallback substring match
        if sig_hash in existing_hashes or sig["text"][:100] in existing:
            continue
        sig["content_hash"] = sig_hash
        new_signals.append(sig)

    if not new_signals:
        return 0

    # Find next signal ID — filter by today's date to avoid cross-date max
    existing_ids = re.findall(r"\[SIG-(\d{8}-\d{3})\]", existing)
    today_compact = date_str.replace("-", "")
    today_ids = [eid for eid in existing_ids if eid.startswith(today_compact)]
    if today_ids:
        counter = max(int(eid[9:]) for eid in today_ids) + 1
    else:
        counter = 1

    with FileLock(signals_path):
        with open(signals_path, "a", encoding="utf-8") as f:
            for sig in new_signals:
                if counter > 999:
                    break  # Cap at 999 signals per day to maintain ID format
                sig_id = f"SIG-{today_compact}-{counter:03d}"
                f.write(f"\n[{sig_id}]\n")
                f.write(f"Date: {date_str}\n")
                f.write(f"Type: auto-capture-{sig['type']}\n")
                f.write(f"Source: memory/{date_str}.md:{sig['line']}\n")
                f.write(f"Confidence: {sig.get('confidence', 'medium')}\n")
                f.write(f"Priority: {sig.get('priority', 'P2')}\n")
                f.write("Status: pending\n")
                f.write(f"Excerpt: {sig['text']}\n")
                if sig.get("content_hash"):
                    f.write(f"ContentHash: {sig['content_hash']}\n")

                # Write structured extraction
                st = sig.get("structure", {})
                if st.get("subject"):
                    f.write(f"Subject: {st['subject']}\n")
                if st.get("object"):
                    f.write(f"Object: {st['object']}\n")
                if st.get("tags"):
                    f.write(f"Tags: {', '.join(st['tags'])}\n")

                prefix = 'D-' if sig['type'] == 'decision' else 'T-'
                f.write(f"Action: Review and formalize as {prefix} block if warranted\n")
                f.write("\n---\n")
                counter += 1

    return len(new_signals)


def main():
    workspace = sys.argv[1] if len(sys.argv) > 1 else "."
    workspace = os.path.abspath(workspace)

    scan_all = "--scan-all" in sys.argv

    if scan_all:
        logs = find_all_logs(workspace, days=7)
        if not logs:
            print("capture: no daily logs found in last 7 days")
            return
        total_detected = 0
        total_written = 0
        for log_path, date_str in logs:
            signals = scan_log(log_path)
            if signals:
                written = append_signals(workspace, signals, date_str)
                total_detected += len(signals)
                total_written += written
        _log.info("batch_scan_complete", logs=len(logs), detected=total_detected, written=total_written)
        metrics.inc("signals_detected", total_detected)
        metrics.inc("signals_written", total_written)
        print(f"capture: scanned {len(logs)} log(s) — {total_detected} detected, {total_written} new signals")
    else:
        log_path, date_str = find_today_log(workspace)
        if not log_path:
            print(f"capture: no daily log for {date_str}, nothing to scan")
            return

        signals = scan_log(log_path)
        if not signals:
            print(f"capture: {date_str} — 0 uncaptured items")
            return

        written = append_signals(workspace, signals, date_str)
        _log.info("scan_complete", date=date_str, detected=len(signals), written=written)
        metrics.inc("signals_detected", len(signals))
        metrics.inc("signals_written", written)
        print(f"capture: {date_str} — {len(signals)} detected, {written} new signals appended")


if __name__ == "__main__":
    main()
