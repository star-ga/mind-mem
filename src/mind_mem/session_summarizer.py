#!/usr/bin/env python3
"""mind-mem Session Summarizer. Zero external deps.

Generates structured session summaries from Claude Code JSONL
transcripts. Extracts topics, files touched, decisions, and duration.

Writes:
  - Summary blocks [SESS-YYYYMMDD-NNN] to summaries/daily/YYYY-MM-DD.md
  - Linking signal to intelligence/SIGNALS.md

GOVERNANCE. Both writes are admitted under
:attr:`~mind_mem.enums.IngestTier.AUTO_CAPTURE` before a byte lands. That
tier's :data:`~mind_mem.enums.INITIAL_STATUS` row is ``PENDING``, which is
not in :data:`~mind_mem.enums.SERVABLE`, so nothing this module writes is
recallable until a governance proposal releases it. The summary block
carries that status explicitly rather than inheriting it by accident:
``summaries/`` is outside ``_recall_constants.CORPUS_FILES`` today, and a
door whose safety rests on a directory listing somewhere else is one
refactor away from being a leak.

The content is transcript text the operator never reviewed -- a summary is
a DERIVED artifact, not a trusted one -- so it goes through the same door
as any other auto-capture.

Usage:
    python3 -m mind_mem.session_summarizer workspace/ --transcript path/to.jsonl
    python3 -m mind_mem.session_summarizer workspace/ --scan-recent
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from datetime import datetime

from .capture import append_signals
from .enums import INITIAL_STATUS, IngestTier
from .mind_filelock import FileLock
from .observability import get_logger, metrics
from .transcript_capture import TRANSCRIPT_PATTERNS, find_recent_transcripts, parse_transcript

_log = get_logger("session_summarizer")

#: The tier every write in this module is admitted under. A session summary
#: is transcript-derived text nobody reviewed, which is exactly what
#: AUTO_CAPTURE names -- the same tier ``capture.append_signals`` already
#: uses for the linking signal this module emits beside the summary.
SUMMARY_TIER = IngestTier.AUTO_CAPTURE

#: The status the summary block declares, READ OFF the one table that
#: decides it rather than spelled as a literal here. A second hand-written
#: copy of "pending" is a second place for the answer to drift; this way a
#: change to INITIAL_STATUS moves the block with it.
_SUMMARY_STATUS_OR_NONE = INITIAL_STATUS[SUMMARY_TIER]
#: Fail at import if this tier mints no status. ``INITIAL_STATUS`` maps some
#: tiers to ``None`` on purpose (RESTAMP and STORE_MIGRATION carry the existing
#: status forward), and an UNSTATED status is SERVABLE -- so a ``None`` here
#: would silently publish every session summary instead of withholding it.
#: This is the invariant, not a type-checker appeasement: if someone retiers
#: the summarizer to a carry-forward tier, the import fails loudly rather than
#: the summaries quietly becoming world-readable.
if _SUMMARY_STATUS_OR_NONE is None:  # pragma: no cover - import-time invariant
    # NOT an ``assert``: ``python -O`` strips those, and an invariant that
    # disappears under an optimisation flag is not an invariant. This one
    # decides whether session summaries are withheld or published, so it has
    # to hold in every interpreter mode.
    raise RuntimeError(
        f"session summaries are written under tier {SUMMARY_TIER!r}, which mints "
        "no initial status; an unstated status is SERVABLE, so this would "
        "publish them"
    )
SUMMARY_STATUS = _SUMMARY_STATUS_OR_NONE


# ---------------------------------------------------------------------------
# File path extraction
# ---------------------------------------------------------------------------

FILE_PATH_RE = re.compile(r"(?:/home/\w+/[\w./-]+|(?:\./|\.\./)[\w./-]+|[\w-]+/[\w./-]+\.\w{1,6})")

# Proper noun / topic extraction (capitalized words, 2+ chars, not common words)
COMMON_WORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "has",
        "her",
        "was",
        "one",
        "our",
        "out",
        "its",
        "his",
        "how",
        "man",
        "new",
        "now",
        "old",
        "see",
        "way",
        "may",
        "day",
        "too",
        "any",
        "who",
        "boy",
        "did",
        "get",
        "let",
        "say",
        "she",
        "use",
        "been",
        "call",
        "each",
        "from",
        "have",
        "just",
        "make",
        "than",
        "that",
        "them",
        "then",
        "this",
        "very",
        "when",
        "with",
        "also",
        "back",
        "been",
        "come",
        "could",
        "first",
        "into",
        "like",
        "long",
        "look",
        "only",
        "over",
        "such",
        "take",
        "will",
        "about",
        "could",
        "every",
        "found",
        "great",
        "here",
        "still",
        "their",
        "these",
        "think",
        "those",
        "under",
        "where",
        "which",
        "while",
        "would",
        "after",
        "before",
        "should",
        "right",
        "through",
        "true",
        "false",
        "none",
        "null",
        "error",
        "warning",
        "info",
        "debug",
        "import",
        "export",
        "return",
        "function",
        "class",
        "const",
        "file",
        "line",
        "type",
        "name",
        "path",
        "data",
        "code",
        "test",
        "added",
        "instead",
        "using",
        "here",
        "need",
        "want",
        "sure",
        "okay",
        "done",
        "note",
        "update",
        "change",
        "check",
        "create",
        "delete",
        "read",
        "write",
        "run",
        "start",
        "stop",
        "open",
        "close",
    }
)

TOPIC_RE = re.compile(r"\b([A-Z][a-zA-Z0-9_-]{2,})\b")


def file_hash(path: str) -> str:
    """SHA256 hash of a file's contents (first 64KB) for dedup."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            h.update(f.read(65536))
    except OSError:
        h.update(path.encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Summary extraction
# ---------------------------------------------------------------------------


def extract_summary(messages: list[dict]) -> dict:
    """Extract structured summary from parsed transcript messages.

    Returns dict with: topics, files, decisions, message_count, roles.
    """
    all_text = []
    files: Counter[str] = Counter()
    topics: Counter[str] = Counter()
    decisions = []
    roles: Counter[str] = Counter()

    for msg in messages:
        text = msg["content"]
        role = msg["role"]
        all_text.append(text)
        roles[role] += 1

        # Extract file paths
        for fp in FILE_PATH_RE.findall(text):
            # Normalize: strip trailing punctuation
            fp = fp.rstrip(".,;:)")
            if len(fp) > 5:
                files[fp] += 1

        # Extract topics (proper nouns / identifiers)
        for t in TOPIC_RE.findall(text):
            if t.lower() not in COMMON_WORDS and len(t) > 2:
                topics[t] += 1

        # Check for decision/correction patterns
        for pattern, sig_type, confidence in TRANSCRIPT_PATTERNS:
            if pattern.search(text):
                excerpt = text[:150].strip()
                if excerpt and len(excerpt) > 15:
                    decisions.append(
                        {
                            "type": sig_type,
                            "confidence": confidence,
                            "excerpt": excerpt,
                            "role": role,
                        }
                    )
                break

    return {
        "topics": topics.most_common(15),
        "files": files.most_common(20),
        "decisions": decisions[:10],
        "message_count": len(messages),
        "roles": dict(roles),
    }


def format_summary_block(
    sess_id: str,
    transcript_path: str,
    summary: dict,
    transcript_hash: str,
) -> str:
    """Format a SESS block for summaries/daily/*.md."""
    lines = [
        f"[{sess_id}]",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        # Declared, not implied. `is_servable` fails closed on a block with
        # no Status at all, so the field is belt AND braces -- but a block
        # that states its own withheld status is auditable by reading it,
        # which a block relying on an absent field is not.
        f"Status: {SUMMARY_STATUS.value}",
        f"Source: {transcript_path}",
        f"TranscriptHash: {transcript_hash}",
        f"Messages: {summary['message_count']}",
        f"Roles: {', '.join(f'{k}={v}' for k, v in summary['roles'].items())}",
    ]

    if summary["topics"]:
        top_topics = [f"{t} ({c})" for t, c in summary["topics"][:10]]
        lines.append(f"Topics: {', '.join(top_topics)}")

    if summary["files"]:
        top_files = [f for f, _ in summary["files"][:10]]
        lines.append(f"Files: {', '.join(top_files)}")

    if summary["decisions"]:
        lines.append("Decisions:")
        for d in summary["decisions"][:5]:
            lines.append(f"  - [{d['confidence']}] {d['type']}: {d['excerpt'][:80]}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Write summary
# ---------------------------------------------------------------------------


def write_summary(
    workspace: str,
    transcript_path: str,
    messages: list[dict],
    dry_run: bool = False,
) -> str | None:
    """Generate and write a session summary.

    Returns the session ID if written, None if skipped (dedup).
    """
    today = datetime.now().strftime("%Y-%m-%d")
    summary_dir = os.path.join(workspace, "summaries", "daily")
    os.makedirs(summary_dir, exist_ok=True)
    summary_file = os.path.join(summary_dir, f"{today}.md")

    # Dedup: check if this transcript was already summarized
    t_hash = file_hash(transcript_path)
    if os.path.isfile(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            existing = f.read()
        if t_hash in existing:
            _log.info("summary_skipped_dedup", transcript=transcript_path, hash=t_hash)
            return None

    summary = extract_summary(messages)
    if summary["message_count"] < 3:
        _log.info("summary_skipped_too_short", messages=summary["message_count"])
        return None

    # Determine next SESS ID
    today_compact = today.replace("-", "")
    if os.path.isfile(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            content = f.read()
        existing_ids = re.findall(rf"\[SESS-{today_compact}-(\d{{3}})\]", content)
        counter = max((int(x) for x in existing_ids), default=0) + 1
    else:
        counter = 1

    sess_id = f"SESS-{today_compact}-{counter:03d}"
    block = format_summary_block(sess_id, transcript_path, summary, t_hash)

    if dry_run:
        print(f"[DRY RUN] Would write {sess_id}:\n{block}")
        return sess_id

    # Admit BEFORE the bytes land, and let a refusal propagate. Admitting
    # after the append is what `capture.append_signals` used to do, inside a
    # bare except that downgraded the refusal to a warning -- so a rejected
    # write stayed on disk anyway. There is no `_get_gate`-returns-None
    # fallback here for the same reason: a door that writes when it cannot
    # reach the gate is a door with no gate.
    #
    # Lazy import -- governance_gate pulls in the hash chain and the evidence
    # chain, and this module is also a plain CLI that parses transcripts.
    from .governance_gate import get_gate

    with get_gate(workspace).admit_block(
        action="WRITE",
        block_id=sess_id,
        content=block,
        tier=SUMMARY_TIER,
        actor="session_summarizer",
        target_file=summary_file,
        metadata={"source": os.path.basename(transcript_path), "transcript_hash": t_hash},
    ):
        # Write summary block
        with FileLock(summary_file):
            header_needed = not os.path.isfile(summary_file)
            with open(summary_file, "a", encoding="utf-8") as f:
                if header_needed:
                    f.write(f"# Session Summaries — {today}\n\n---\n\n")
                f.write(block)
                f.write("\n---\n\n")

    # Write linking signal to SIGNALS.md
    linking_signal = [
        {
            "line": 0,
            "type": "summary",
            "text": f"Session summary {sess_id} from {os.path.basename(transcript_path)} "
            f"({summary['message_count']} msgs, "
            f"{len(summary['topics'])} topics, "
            f"{len(summary['decisions'])} decisions)",
            "pattern": "auto-capture-summary",
            "confidence": "medium",
            "priority": "P2",
            "structure": {
                "subject": sess_id,
                "object": os.path.basename(transcript_path),
                "tags": ["session-summary"],
            },
        }
    ]
    append_signals(workspace, linking_signal, today)

    _log.info(
        "summary_written",
        sess_id=sess_id,
        messages=summary["message_count"],
        topics=len(summary["topics"]),
        decisions=len(summary["decisions"]),
    )
    metrics.inc("summaries_written")
    return sess_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="mind-mem Session Summarizer")
    parser.add_argument("workspace", nargs="?", default=".")
    parser.add_argument("--transcript", "-t", help="Path to specific .jsonl transcript")
    parser.add_argument("--scan-recent", action="store_true", help="Scan recent transcripts from ~/.claude/projects/")
    parser.add_argument("--days", type=int, default=3, help="Days to look back (default: 3)")
    parser.add_argument("--dry-run", action="store_true", help="Show summaries without writing")
    args = parser.parse_args()

    ws = os.path.abspath(args.workspace)

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

    written = 0
    skipped = 0

    for t_path in transcripts:
        messages = parse_transcript(t_path)
        if not messages:
            continue
        sess_id = write_summary(ws, t_path, messages, dry_run=args.dry_run)
        if sess_id:
            written += 1
            print(f"  {sess_id} <- {os.path.basename(t_path)} ({len(messages)} msgs)")
        else:
            skipped += 1

    print(f"\nSummaries: {written} written, {skipped} skipped (dedup/too short)")


if __name__ == "__main__":
    main()
