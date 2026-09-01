#!/usr/bin/env python3
"""mind-mem Bootstrap Corpus — one-time backfill from existing knowledge sources.

Populates the mind-mem corpus by scanning:

1. ALL JSONL transcripts in ``~/.claude/projects/`` (all time)
2. ALL daily logs in ``workspace/memory/`` (extended window)
3. ``~/CLAUDE.md`` and ``MEMORY.md`` files for patterns
4. Entity extraction on all collected text

Safe to re-run: ``content_hash`` dedup in :func:`~mind_mem.capture.append_signals`
prevents double-writing.

THIS IS AN INGEST DOOR
----------------------
Everything it reads originates OUTSIDE the workspace: session transcripts hold
whatever an agent was shown — pasted text, tool output, fetched pages — and a
CLAUDE.md is a file on disk like any other. So nothing this door mints is
servable. Both write legs run under a :class:`~mind_mem.governance_gate.GovernanceGate`
admission with :attr:`~mind_mem.enums.IngestTier.AUTO_CAPTURE`, which
:data:`~mind_mem.enums.INITIAL_STATUS` maps to :attr:`~mind_mem.enums.Status.PENDING`
— withheld, because :data:`~mind_mem.enums.SERVABLE` holds ``ACTIVE`` alone:

* **signals** → :func:`~mind_mem.capture.append_signals`, which opens its own
  ``admit_batch`` before a byte lands and stamps ``Status: pending``. Those
  blocks live in ``intelligence/SIGNALS.md``, inside
  :data:`~mind_mem.corpus_registry.CORPUS_DIRS`, so recall *parses* them and
  ``admit_corpus`` *withholds* them until a proposal releases them.
* **session summaries** → :func:`~mind_mem.session_summarizer.write_summary`,
  wrapped here in :func:`_write_summary_admitted`. That function appends to
  ``summaries/daily/*.md``, which is outside ``CORPUS_DIRS`` and therefore not
  a recall surface at all; the admission is what makes the write auditable and
  refusable rather than silent. Fail-closed: no gate, no summary.

Neither leg may be handed a tier that mints a servable status —
``GovernanceGate._check_tier`` refuses one, and ``tests/test_quarantine_redteam.py``
pins the table it reads.

Flag-gated, default OFF: ``mind-mem.json`` → ``v4.bootstrap_corpus.enabled``.
With the flag off :func:`main` writes nothing, reads nothing, and probes the
flag through :func:`~mind_mem.v4.feature_flags.is_enabled_quiet` so the OFF path
emits no log line the unwired build would not have emitted.

Usage::

    mind-mem-bootstrap <workspace> [--dry-run] [--max-transcripts N]
    python3 -m mind_mem.bootstrap_corpus <workspace> --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import IO, Optional

from .capture import append_signals, find_all_logs, scan_log
from .entity_ingest import entities_to_signals, extract_entities, filter_new_entities, load_existing_entities
from .enums import IngestTier
from .observability import get_logger
from .session_summarizer import file_hash, write_summary
from .transcript_capture import find_recent_transcripts, parse_transcript, scan_transcript

__all__ = [
    "FLAG",
    "BootstrapReport",
    "flag_enabled",
    "main",
    "run_bootstrap",
    "scan_markdown_file",
]

_log = get_logger("bootstrap_corpus")

#: ``mind-mem.json`` → ``v4.bootstrap_corpus.enabled``. Registered in
#: :data:`~mind_mem.v4.feature_flags.ALL_V4_FLAGS`; an unregistered name is
#: fail-closed there, so a typo cannot turn the door on.
FLAG = "bootstrap_corpus"

#: The tier every write on this door is admitted under. Named once so the
#: signal leg and the summary leg cannot drift apart, and so a reader can see
#: in one place that the door mints ``PENDING`` and nothing else.
INGEST_TIER = IngestTier.AUTO_CAPTURE

#: Ten years — the whole point of a backfill is that it is not a rolling window.
_ALL_TIME_DAYS = 3650


def flag_enabled() -> bool:
    """True when ``v4.bootstrap_corpus.enabled`` is set.

    Uses the QUIET probe. :func:`~mind_mem.v4.feature_flags.is_enabled` warns
    ``v4_config_unreadable`` on a malformed config; a probe that logs on an OFF
    path makes the flag-off build observably different from the build that
    never had the door, and this restoration lands under "flag-off is
    byte-identical".
    """
    from .v4.feature_flags import is_enabled_quiet

    return is_enabled_quiet(FLAG)


@dataclass(frozen=True)
class BootstrapReport:
    """What one backfill run found and wrote. Immutable by construction."""

    transcripts: int = 0
    logs: int = 0
    markdown_files: int = 0
    signals_detected: int = 0
    signals_written: int = 0
    entities_proposed: int = 0
    summaries_created: int = 0
    dry_run: bool = False


def scan_markdown_file(file_path: str) -> list[dict]:
    """Scan a markdown file (CLAUDE.md, MEMORY.md) for decision/entity patterns.

    Reuses scan_log which already matches decision/task patterns on markdown lines.
    """
    if not os.path.isfile(file_path):
        return []
    return scan_log(file_path)


def _write_summary_admitted(workspace: str, transcript_path: str, messages: list[dict]) -> Optional[str]:
    """Write one session summary INSIDE a governance admission, or not at all.

    ``write_summary`` appends to ``summaries/daily/*.md`` with a plain
    ``open(..., "a")``. That file is outside ``CORPUS_DIRS``, so no recall leg
    can reach it — but "unreachable today" is exactly the argument the 5.0.0
    sweep made about whole modules, and it is evidence about wiring, not about
    safety. The admission is opened BEFORE the bytes land, so a gate that
    refuses (a drifted spec binding, a governance mode that blocks) aborts the
    write instead of annotating one that already happened.

    Fail-closed: if the gate cannot be constructed, this writes nothing and
    returns ``None``. An ingest door with no gate is not a door that should
    fall through to an ungoverned append.

    The id is derived from the transcript's content hash, not from a counter,
    so re-running the backfill re-admits the same id for the same source.
    """
    try:
        from .governance_gate import get_gate

        gate = get_gate(workspace)
    except Exception as exc:  # noqa: BLE001 - any gate failure means "do not write"
        _log.warning("bootstrap_summary_ungated_skipped", transcript=transcript_path, error=str(exc))
        return None

    digest = file_hash(transcript_path)
    with gate.admit_block(
        action="INGEST",
        block_id=f"SESS-BOOTSTRAP-{digest}",
        content="\n".join(str(m.get("content", "")) for m in messages),
        tier=INGEST_TIER,
        actor="bootstrap_corpus",
        target_file=os.path.join(workspace, "summaries", "daily"),
        metadata={"source": os.path.basename(transcript_path), "transcript_hash": digest},
    ):
        return write_summary(workspace, transcript_path, messages, dry_run=False)


def _phase_transcripts(
    workspace: str,
    *,
    dry_run: bool,
    max_transcripts: int,
    today: str,
    entity_text: list[str],
    out: IO[str],
) -> tuple[int, int, int, int]:
    """Phase 1 — mine every JSONL transcript. Returns (found, detected, written, summaries)."""
    print("Phase 1: Scanning JSONL transcripts...", file=out)
    transcripts = find_recent_transcripts(days=_ALL_TIME_DAYS)
    if max_transcripts > 0:
        transcripts = transcripts[:max_transcripts]
    print(f"  Found {len(transcripts)} transcript(s)", file=out)

    detected = written = summaries = 0
    for i, t_path in enumerate(transcripts, 1):
        if i % 10 == 0 or i == 1:
            print(f"  Processing transcript {i}/{len(transcripts)}...", file=out)

        signals = scan_transcript(t_path)
        detected += len(signals)
        if signals and not dry_run:
            written += append_signals(workspace, signals, today)

        messages = parse_transcript(t_path)
        if messages:
            entity_text.append(" ".join(str(m.get("content", "")) for m in messages[:50]))
            if dry_run:
                sess_id = write_summary(workspace, t_path, messages, dry_run=True)
            else:
                sess_id = _write_summary_admitted(workspace, t_path, messages)
            if sess_id:
                summaries += 1

    print(f"  Transcripts done: {detected} signals detected, {written} written, {summaries} summaries", file=out)
    print(file=out)
    return len(transcripts), detected, written, summaries


def _phase_logs(
    workspace: str,
    *,
    dry_run: bool,
    entity_text: list[str],
    out: IO[str],
) -> tuple[int, int, int]:
    """Phase 2 — mine every daily log. Returns (found, detected, written)."""
    print("Phase 2: Scanning daily logs...", file=out)
    logs = find_all_logs(workspace, days=_ALL_TIME_DAYS)
    print(f"  Found {len(logs)} daily log(s)", file=out)

    detected = written = 0
    for log_path, date_str in logs:
        signals = scan_log(log_path)
        detected += len(signals)
        if signals and not dry_run:
            written += append_signals(workspace, signals, date_str)
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                entity_text.append(f.read())
        except OSError:
            pass

    print(f"  Logs done: {detected} signals detected, {written} written", file=out)
    print(file=out)
    return len(logs), detected, written


def _phase_markdown(
    workspace: str,
    *,
    dry_run: bool,
    today: str,
    entity_text: list[str],
    out: IO[str],
) -> tuple[int, int, int]:
    """Phase 3 — mine the two well-known markdown files. Returns (found, detected, written)."""
    print("Phase 3: Scanning CLAUDE.md and MEMORY.md...", file=out)
    md_files = [
        os.path.expanduser("~/CLAUDE.md"),
        os.path.expanduser("~/.claude/MEMORY.md"),
    ]

    found = detected = written = 0
    for md_path in md_files:
        if not os.path.isfile(md_path):
            print(f"  Skipped (not found): {md_path}", file=out)
            continue
        found += 1

        signals = scan_markdown_file(md_path)
        detected += len(signals)
        print(f"  {os.path.basename(md_path)}: {len(signals)} signals", file=out)
        if signals and not dry_run:
            written += append_signals(workspace, signals, today)

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                entity_text.append(f.read())
        except OSError:
            pass

    print(f"  Markdown done: {detected} detected, {written} written", file=out)
    print(file=out)
    return found, detected, written


def _phase_entities(
    workspace: str,
    *,
    dry_run: bool,
    today: str,
    entity_text: list[str],
    out: IO[str],
) -> tuple[int, int]:
    """Phase 4 — entity extraction over everything collected. Returns (proposed, written)."""
    print("Phase 4: Running entity extraction...", file=out)
    existing_entities = load_existing_entities(workspace)
    entities = extract_entities("\n".join(entity_text))
    new_entities = filter_new_entities(entities, existing_entities)

    seen: set[tuple[str, str]] = set()
    unique_entities = []
    for ent in new_entities:
        key = (ent["entity_type"], ent["slug"])
        if key not in seen:
            seen.add(key)
            unique_entities.append(ent)

    print(f"  Entities found: {len(entities)} total, {len(unique_entities)} new", file=out)

    written = 0
    if unique_entities and not dry_run:
        written = append_signals(workspace, entities_to_signals(unique_entities, "bootstrap_corpus"), today)
        print(f"  Entity signals written: {written}", file=out)
    elif unique_entities and dry_run:
        for ent in unique_entities[:20]:
            print(f"    NEW {ent['entity_type']}: {ent['slug']} (via {ent['source_pattern']})", file=out)
        if len(unique_entities) > 20:
            print(f"    ... and {len(unique_entities) - 20} more", file=out)

    print(file=out)
    return len(unique_entities), written


def run_bootstrap(
    workspace: str,
    *,
    dry_run: bool = False,
    max_transcripts: int = 0,
    out: IO[str] | None = None,
) -> BootstrapReport:
    """Run the four backfill phases and return what they found.

    The caller is responsible for the flag check — :func:`main` does it. This
    is the programmatic entry point, so a test (or an embedder) can drive the
    pipeline without going through argv.
    """
    stream: IO[str] = out if out is not None else sys.stdout
    ws = os.path.abspath(workspace)
    today = datetime.now().strftime("%Y-%m-%d")

    print("mind-mem Bootstrap Corpus", file=stream)
    print(f"  Workspace: {ws}", file=stream)
    print(f"  Dry run:   {dry_run}", file=stream)
    print(f"  Date:      {today}", file=stream)
    print(file=stream)

    entity_text: list[str] = []

    n_transcripts, t_detected, t_written, summaries = _phase_transcripts(
        ws, dry_run=dry_run, max_transcripts=max_transcripts, today=today, entity_text=entity_text, out=stream
    )
    n_logs, l_detected, l_written = _phase_logs(ws, dry_run=dry_run, entity_text=entity_text, out=stream)
    n_md, m_detected, m_written = _phase_markdown(ws, dry_run=dry_run, today=today, entity_text=entity_text, out=stream)
    entities_proposed, e_written = _phase_entities(ws, dry_run=dry_run, today=today, entity_text=entity_text, out=stream)

    report = BootstrapReport(
        transcripts=n_transcripts,
        logs=n_logs,
        markdown_files=n_md,
        signals_detected=t_detected + l_detected + m_detected,
        signals_written=t_written + l_written + m_written + e_written,
        entities_proposed=entities_proposed,
        summaries_created=summaries,
        dry_run=dry_run,
    )

    print("=" * 60, file=stream)
    print("Bootstrap Corpus Complete", file=stream)
    print("=" * 60, file=stream)
    print(f"  Transcripts processed:  {report.transcripts}", file=stream)
    print(f"  Daily logs scanned:     {report.logs}", file=stream)
    print(f"  Markdown files scanned: {report.markdown_files}", file=stream)
    print(f"  Total signals found:    {report.signals_detected}", file=stream)
    print(f"  New signals written:    {report.signals_written}", file=stream)
    print(f"  Entities proposed:      {report.entities_proposed}", file=stream)
    print(f"  Summaries created:      {report.summaries_created}", file=stream)
    print("  Everything written is PENDING — recall withholds it until a", file=stream)
    print("  governance proposal releases it. Review with: mm scan", file=stream)
    if dry_run:
        print("  (DRY RUN — nothing was written)", file=stream)
    print(file=stream)

    _log.info(
        "bootstrap_complete",
        transcripts=report.transcripts,
        logs=report.logs,
        signals_detected=report.signals_detected,
        signals_written=report.signals_written,
        entities_proposed=report.entities_proposed,
        summaries_created=report.summaries_created,
        dry_run=report.dry_run,
    )
    return report


#: Printed when the door is invoked with its flag off. Names the exact key, so
#: an operator does not have to grep for it.
FLAG_OFF_MESSAGE = (
    f"mind-mem-bootstrap is disabled. It is an INGEST DOOR and ships default-OFF.\n"
    f'Enable it in mind-mem.json: "v4": {{ "{FLAG}": {{ "enabled": true }} }}\n'
    "Everything it writes lands PENDING and stays invisible to recall until a\n"
    "governance proposal releases it."
)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``mind-mem-bootstrap``. Returns a process exit code."""
    parser = argparse.ArgumentParser(prog="mind-mem-bootstrap", description="mind-mem Bootstrap Corpus Backfill")
    parser.add_argument("workspace", help="Path to mind-mem workspace")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be written without writing")
    parser.add_argument("--max-transcripts", type=int, default=0, help="Limit number of transcripts to process (0 = unlimited)")
    args = parser.parse_args(argv)

    # The flag check comes BEFORE anything is read, so an OFF run touches no
    # transcript, opens no workspace, and creates no gate.
    if not flag_enabled():
        print(FLAG_OFF_MESSAGE, file=sys.stderr)
        return 2

    run_bootstrap(args.workspace, dry_run=args.dry_run, max_transcripts=args.max_transcripts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
