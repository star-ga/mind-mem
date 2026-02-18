#!/usr/bin/env python3
"""mind-mem Bootstrap Corpus — one-time backfill from existing knowledge sources.

Populates the mind-mem corpus by scanning:
1. ALL JSONL transcripts in ~/.claude/projects/ (all time)
2. ALL daily logs in workspace/memory/ (extended window)
3. ~/CLAUDE.md and ~/.claude/projects/-home-n/memory/MEMORY.md for patterns
4. Entity extraction on all collected text

Safe to re-run: content_hash dedup in append_signals() prevents double-writing.

Usage:
    python3 scripts/bootstrap_corpus.py <workspace> [--dry-run] [--max-transcripts N]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture import append_signals, scan_log, find_all_logs
from transcript_capture import parse_transcript, scan_transcript, find_recent_transcripts
from session_summarizer import write_summary
from entity_ingest import extract_entities, filter_new_entities, load_existing_entities, entities_to_signals
from observability import get_logger

_log = get_logger("bootstrap_corpus")


def scan_markdown_file(file_path: str) -> list[dict]:
    """Scan a markdown file (CLAUDE.md, MEMORY.md) for decision/entity patterns.

    Reuses scan_log which already matches decision/task patterns on markdown lines.
    """
    if not os.path.isfile(file_path):
        return []
    return scan_log(file_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="mind-mem Bootstrap Corpus Backfill")
    parser.add_argument("workspace", help="Path to mind-mem workspace")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be written without writing")
    parser.add_argument("--max-transcripts", type=int, default=0,
                        help="Limit number of transcripts to process (0 = unlimited)")
    args = parser.parse_args()

    ws = os.path.abspath(args.workspace)
    today = datetime.now().strftime("%Y-%m-%d")
    dry_run = args.dry_run

    print("mind-mem Bootstrap Corpus")
    print(f"  Workspace: {ws}")
    print(f"  Dry run:   {dry_run}")
    print(f"  Date:      {today}")
    print()

    total_signals_detected = 0
    total_signals_written = 0
    total_summaries_created = 0
    total_entities_proposed = 0
    all_entity_text = []  # Collect text for entity extraction at the end

    # ------------------------------------------------------------------
    # Phase 1: Scan ALL JSONL transcripts
    # ------------------------------------------------------------------
    print("Phase 1: Scanning JSONL transcripts...")
    transcripts = find_recent_transcripts(days=3650)  # 10 years = effectively all time
    if args.max_transcripts > 0:
        transcripts = transcripts[:args.max_transcripts]
    print(f"  Found {len(transcripts)} transcript(s)")

    for i, t_path in enumerate(transcripts, 1):
        if i % 10 == 0 or i == 1:
            print(f"  Processing transcript {i}/{len(transcripts)}...")

        # Extract signals
        signals = scan_transcript(t_path)
        total_signals_detected += len(signals)

        if signals and not dry_run:
            written = append_signals(ws, signals, today)
            total_signals_written += written

        # Create session summary
        messages = parse_transcript(t_path)
        if messages:
            all_entity_text.append(" ".join(m["content"] for m in messages[:50]))
            sess_id = write_summary(ws, t_path, messages, dry_run=dry_run)
            if sess_id:
                total_summaries_created += 1

    print(f"  Transcripts done: {total_signals_detected} signals detected, "
          f"{total_signals_written} written, {total_summaries_created} summaries")
    print()

    # ------------------------------------------------------------------
    # Phase 2: Scan ALL daily logs (extended window)
    # ------------------------------------------------------------------
    print("Phase 2: Scanning daily logs...")
    logs = find_all_logs(ws, days=3650)  # 10 years = all logs
    print(f"  Found {len(logs)} daily log(s)")

    log_signals_detected = 0
    log_signals_written = 0
    for log_path, date_str in logs:
        signals = scan_log(log_path)
        log_signals_detected += len(signals)

        if signals and not dry_run:
            written = append_signals(ws, signals, date_str)
            log_signals_written += written

        # Collect text for entity extraction
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                all_entity_text.append(f.read())
        except OSError:
            pass

    total_signals_detected += log_signals_detected
    total_signals_written += log_signals_written
    print(f"  Logs done: {log_signals_detected} signals detected, {log_signals_written} written")
    print()

    # ------------------------------------------------------------------
    # Phase 3: Parse CLAUDE.md and MEMORY.md
    # ------------------------------------------------------------------
    print("Phase 3: Scanning CLAUDE.md and MEMORY.md...")
    md_files = [
        os.path.expanduser("~/CLAUDE.md"),
        os.path.expanduser("~/.claude/projects/-home-n/memory/MEMORY.md"),
    ]

    md_signals_detected = 0
    md_signals_written = 0
    for md_path in md_files:
        if not os.path.isfile(md_path):
            print(f"  Skipped (not found): {md_path}")
            continue

        signals = scan_markdown_file(md_path)
        md_signals_detected += len(signals)
        print(f"  {os.path.basename(md_path)}: {len(signals)} signals")

        if signals and not dry_run:
            written = append_signals(ws, signals, today)
            md_signals_written += written

        # Collect text for entity extraction
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                all_entity_text.append(f.read())
        except OSError:
            pass

    total_signals_detected += md_signals_detected
    total_signals_written += md_signals_written
    print(f"  Markdown done: {md_signals_detected} detected, {md_signals_written} written")
    print()

    # ------------------------------------------------------------------
    # Phase 4: Entity extraction on all collected text
    # ------------------------------------------------------------------
    print("Phase 4: Running entity extraction...")
    existing_entities = load_existing_entities(ws)
    combined_text = "\n".join(all_entity_text)

    entities = extract_entities(combined_text)
    new_entities = filter_new_entities(entities, existing_entities)

    # Deduplicate by (type, slug)
    seen = set()
    unique_entities = []
    for ent in new_entities:
        key = (ent["entity_type"], ent["slug"])
        if key not in seen:
            seen.add(key)
            unique_entities.append(ent)

    total_entities_proposed = len(unique_entities)
    print(f"  Entities found: {len(entities)} total, {total_entities_proposed} new")

    if unique_entities and not dry_run:
        entity_signals = entities_to_signals(unique_entities, "bootstrap_corpus")
        entity_written = append_signals(ws, entity_signals, today)
        total_signals_written += entity_written
        print(f"  Entity signals written: {entity_written}")
    elif unique_entities and dry_run:
        for ent in unique_entities[:20]:
            print(f"    NEW {ent['entity_type']}: {ent['slug']} (via {ent['source_pattern']})")
        if len(unique_entities) > 20:
            print(f"    ... and {len(unique_entities) - 20} more")

    print()

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Bootstrap Corpus Complete")
    print("=" * 60)
    print(f"  Transcripts processed:  {len(transcripts)}")
    print(f"  Daily logs scanned:     {len(logs)}")
    print(f"  Markdown files scanned: {sum(1 for p in md_files if os.path.isfile(p))}")
    print(f"  Total signals found:    {total_signals_detected}")
    print(f"  New signals written:    {total_signals_written}")
    print(f"  Entities proposed:      {total_entities_proposed}")
    print(f"  Summaries created:      {total_summaries_created}")
    if dry_run:
        print("  (DRY RUN — nothing was written)")
    print()

    _log.info("bootstrap_complete",
              transcripts=len(transcripts),
              logs=len(logs),
              signals_detected=total_signals_detected,
              signals_written=total_signals_written,
              entities_proposed=total_entities_proposed,
              summaries_created=total_summaries_created,
              dry_run=dry_run)


if __name__ == "__main__":
    main()
