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
    python3 -m mind_mem.capture [workspace_path]
    python3 -m mind_mem.capture . --scan-all
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import re
import sys
from datetime import datetime, timedelta
from typing import Any

from .block_provenance import PROVENANCE_FIELDS, clean_provenance_value
from .enums import INITIAL_STATUS, IngestTier
from .mind_filelock import FileLock
from .observability import get_logger, metrics

_log = get_logger("capture")

#: Lock path suffix for the "read the file, derive the next id, write the
#: blocks" critical section.
#:
#: Deliberately NOT ``SIGNALS.md`` itself. ``BlockStore.write_block`` takes
#: ``FileLock(intelligence/SIGNALS.md)`` for its own splice, and
#: :class:`~mind_mem.mind_filelock.FileLock`'s intra-process layer is a plain
#: ``threading.Lock`` keyed on the lock path — not an ``RLock`` — so holding
#: the file's lock across the store call would deadlock capture against
#: itself and surface as a ``LockTimeout`` ten seconds later. A separate
#: path gives concurrent capture runs the mutual exclusion the daily id
#: counter needs, and leaves the file's own lock to the writer that actually
#: touches the bytes.
_DERIVE_LOCK_SUFFIX = ".capture"


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
    (r"\btodo\b", "task", "medium"),  # English prose detector — not a TaskStatus comparison
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
    tags: list[str] = []
    structure: dict = {
        "subject": "",
        "predicate": "",
        "object": "",
        "tags": tags,
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
    tag_keywords = {  # nosec B105 — values are tag-category labels, not passwords; "security" refers to the tag name
        "database": "database",
        "db": "database",
        "postgres": "database",
        "auth": "security",
        "security": "security",
        "token": "security",
        "api": "api",
        "endpoint": "api",
        "rest": "api",
        "deploy": "deployment",
        "ci": "deployment",
        "cd": "deployment",
        "test": "testing",
        "spec": "testing",
        "coverage": "testing",
        "bug": "bugfix",
        "fix": "bugfix",
        "error": "bugfix",
        "infra": "infrastructure",
        "server": "infrastructure",
        "perf": "performance",
        "latency": "performance",
        "slow": "performance",
    }
    for keyword, tag in tag_keywords.items():
        if keyword in text_lower and tag not in tags:
            tags.append(tag)

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
    logs: list[tuple[str, str]] = []
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
                # Coding-schema classifier (ADR/CODE/PERF/ALGO/BUG) —
                # overrides the pattern type when the text clearly
                # matches a specialised coding schema. Keeps the
                # original pattern-based type as a fallback when no
                # coding schema applies.
                coding_type = _classify_coding_type(stripped)
                if coding_type:
                    sig_type = coding_type
                signals.append(
                    {
                        "line": i,
                        "type": sig_type,
                        "text": stripped[:150],
                        "pattern": pattern,
                        "confidence": confidence,
                        "priority": CONFIDENCE_TO_PRIORITY[confidence],
                        "structure": structure,
                    }
                )
                break  # one match per line is enough

    return signals


def _classify_coding_type(text: str) -> str | None:
    """Best-effort coding-schema classification. Never raises."""
    try:
        from .coding_schemas import classify_coding_block

        return classify_coding_block(text)
    except Exception:  # pragma: no cover — best-effort
        return None


def _signal_status() -> str:
    """The one status a captured signal may carry.

    Read off :data:`~mind_mem.enums.INITIAL_STATUS` rather than spelled as
    a literal, for the same reason ``intel_scan._finding_status`` is: the
    tier table is the only place a status is decided, and
    :func:`~mind_mem.admission.require_admission` refuses an
    ``AUTO_CAPTURE`` receipt carrying anything else — so a literal here
    would be a second definition free to drift, and its only possible
    destination is a refused write.
    """
    status = INITIAL_STATUS[IngestTier.AUTO_CAPTURE]
    if status is None:  # pragma: no cover — pinned by test_capture_governed_signals
        raise RuntimeError("IngestTier.AUTO_CAPTURE has no INITIAL_STATUS row; captured signals cannot be stamped")
    return status.value


def _one_line(value: str) -> str:
    """Flatten a value onto the single line its field occupies.

    ``mcp/tools/governance._recent_statements`` compares a candidate
    proposal against the stored ``Excerpt``, and its docstring pins the
    stored form as "truncated to 500 chars with newlines collapsed" — so
    this collapsing is part of that reader's contract, not cosmetics.
    Escaping a value so it cannot break out of its block is a separate
    job, and belongs to ``block_store._neutralise_value``, which every
    write through the store passes through.
    """
    return value.replace("\n", " ").replace("\r", "")


def _signal_block(sig: dict, sig_id: str, date_str: str) -> dict[str, Any]:
    """The governed block form of one captured signal.

    Built entirely before any scope opens and before any byte lands, so a
    value this refuses — ``clean_provenance_value`` raises on a
    vocabulary-bound field — aborts the batch with nothing written. That
    ordering used to be a separate pre-validation loop above the file
    open; it is now a property of building the blocks first.
    """
    block: dict[str, Any] = {
        "_id": sig_id,
        "Date": date_str,
        # Full UTC instant beside the day. ``Date:`` is day-granular, so a
        # reader can only place the block at 00:00 of its day, which makes
        # it look up to 24h older than it is — the recency window in
        # mcp/tools/governance.py then has an effective retention of
        # (24h - time-of-day of the write) and silently under-fires.
        "Captured": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "Type": f"auto-capture-{sig['type']}",
        "Source": f"memory/{date_str}.md:{sig['line']}",
        "Confidence": sig.get("confidence", "medium"),
        "Priority": sig.get("priority", "P2"),
        "Status": _signal_status(),
        "Excerpt": _one_line(str(sig["text"])[:500]),
    }
    if sig.get("content_hash"):
        block["ContentHash"] = sig["content_hash"]

    structure = sig.get("structure") or {}
    if structure.get("subject"):
        block["Subject"] = _one_line(str(structure["subject"]))
    if structure.get("object"):
        block["Object"] = _one_line(str(structure["object"]))
    if structure.get("tags"):
        block["Tags"] = ", ".join(str(tag) for tag in structure["tags"])

    # Provenance fields (Group E + T-001 ContentSource) — optional, written
    # only when the caller attached them (e.g. propose_update).
    provenance = sig.get("provenance") or {}
    for prov_param, prov_field in PROVENANCE_FIELDS.items():
        prov_val = provenance.get(prov_param)
        if not prov_val:
            continue
        prov_clean = clean_provenance_value(prov_param, str(prov_val))
        if prov_clean:
            block[prov_field] = prov_clean

    prefix = "D-" if sig["type"] == "decision" else "T-"
    block["Action"] = f"Review and formalize as {prefix} block if warranted"
    return block


def append_signals(workspace: str, signals: list[dict], date_str: str) -> int:
    """Land captured signals as governed ``SIG-`` blocks. Returns the count.

    Every signal goes through ``BlockStore.write_block`` inside one
    :meth:`~mind_mem.governance_gate.GovernanceGate.admit_batch` scope at
    :attr:`~mind_mem.enums.IngestTier.AUTO_CAPTURE`, so the receipt is
    minted before a byte lands and ``write_block``'s first statement —
    :func:`~mind_mem.admission.require_admission` — refuses any write the
    receipt does not cover, and any status the tier may not mint.

    **Why it is not an ``open(..., "a")`` any more.** This function used
    to hand-write the ``[SIG-…]`` text into ``intelligence/SIGNALS.md``
    with the admission as a *conditional* scope::

        _gate = _get_gate(workspace)          # except Exception: return None
        _scope = _gate.admit_batch(...) if _gate is not None else nullcontext()

    which is fail-OPEN in the one direction that matters: a gate that
    cannot be constructed produced ``None``, the conditional silently
    substituted a no-op scope, and the signal was written anyway with a
    success returned. Measured on a fresh workspace, with
    ``memory/hash_chain_v2.db`` replaced by a directory (an unwritable or
    corrupted ledger — the realistic trigger, not a monkeypatch):
    ``get_gate`` raised ``OperationalError`` and ``append_signals``
    returned 1 with the block in the corpus, the hash chain at +0 and the
    evidence chain at +0. ``tests/test_governed_write_paths`` passed
    throughout, because ``admit_batch`` appears textually in the function
    and the scanner could not see that a conditional expression had an
    ungated arm — which is why that scanner now rejects the shape.

    There is no fallback now, by construction rather than by an
    ``except`` clause someone must remember to keep narrow: nothing here
    can write a corpus file, because nothing here opens one. A gate that
    cannot open cannot mint a receipt, and ``write_block`` without a
    receipt raises :class:`~mind_mem.admission.UngatedWriteError`. The
    caller sees the failure instead of a count.

    deferred: the daily id counter and the ``ContentHash`` dedup are still
    derived from ``intelligence/SIGNALS.md`` on disk, which is authoritative
    only while ``get_block_store`` resolves to the Markdown backend (the
    default). On a Postgres-backed workspace the canonical file stays
    empty, so the counter restarts and a new signal can take an id an
    existing one already holds — the same hazard ``intel_scan``
    documents. Upgrade path: derive the used-id set through the store as
    ``intel_scan._recorded_findings`` does, once it can be done without
    paying a whole-corpus parse per call (measured 0.25 s for
    ``get_all`` against 0.005 s for the file read on a 2,750-block
    workspace, on a path that includes the ``observe_signal`` MCP tool).
    """
    signals_path = os.path.join(workspace, "intelligence", "SIGNALS.md")
    if not os.path.isfile(signals_path):
        return 0

    with FileLock(signals_path + _DERIVE_LOCK_SUFFIX):
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

        # Cap at 999 signals per day to keep the ``SIG-YYYYMMDD-###`` id
        # format. The blocks are built first so the batch names exactly the
        # ids it will write — a receipt that covers an id no one writes is
        # noise, and one that misses an id refuses that write.
        blocks = [
            _signal_block(sig, f"SIG-{today_compact}-{counter + offset:03d}", date_str)
            for offset, sig in enumerate(new_signals[: max(0, 1000 - counter)])
        ]
        if not blocks:
            return 0

        from .governance_gate import get_gate
        from .storage import get_block_store

        # Built before the scope opens: a store that cannot be constructed
        # must not leave an authorisation record behind for a write that
        # never happened.
        store = get_block_store(workspace)
        with get_gate(workspace).admit_batch(
            action="WRITE",
            batch_id=f"capture-{today_compact}",
            block_ids=[str(block["_id"]) for block in blocks],
            content="\n".join(sig["text"] for sig in new_signals),
            tier=IngestTier.AUTO_CAPTURE,
            actor="capture",
            target_file=signals_path,
        ):
            for block in blocks:
                store.write_block(block)

    # The number of blocks that actually landed, not ``len(new_signals)``:
    # the two differ once the daily cap bites, and the old return reported
    # signals it had just dropped.
    return len(blocks)


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
        log_path, date_str = find_today_log(workspace)  # type: ignore[assignment]
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
