#!/usr/bin/env python3
"""mind-mem Dream Cycle — autonomous memory enrichment. Zero external deps.

Five enrichment passes run periodically (nightly/cron) to improve memory
quality without human intervention:

    1. Entity Discovery      — find untracked entities in recent logs
    2. Citation Repair       — detect broken internal references
    3. Stale Block Detection — flag blocks untouched for >30 days
    4. Consolidation         — identify facts repeated across daily logs
    5. Integrity Summary     — write a report of all findings

Usage:
    python3 -m mind_mem.dream_cycle workspace/
    python3 -m mind_mem.dream_cycle workspace/ --dry-run
    python3 -m mind_mem.dream_cycle workspace/ --pass entity_discovery
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from .observability import get_logger, metrics, timed

_log = get_logger("dream_cycle")


# --- Data classes (immutable results) ---

@dataclass(frozen=True)
class EntityProposal:
    """A proposed new entity discovered during Pass 1."""

    entity_type: str  # "project", "tool", "person"
    slug: str
    source_pattern: str
    excerpt: str
    source_file: str


@dataclass(frozen=True)
class BrokenCitation:
    """A broken internal reference found during Pass 2."""

    source_file: str
    cited_id: str
    line_number: int
    context: str


@dataclass(frozen=True)
class StaleBlock:
    """A block that has not been updated recently (Pass 3)."""

    block_id: str
    source_file: str
    last_modified_date: str  # ISO date string or "unknown"
    days_stale: int


@dataclass(frozen=True)
class ConsolidationCandidate:
    """A fact appearing in multiple daily logs (Pass 4)."""

    fact_text: str
    occurrences: int
    source_files: tuple[str, ...]


@dataclass(frozen=True)
class DreamCycleReport:
    """Complete report from a dream cycle run. Immutable."""

    timestamp: str
    workspace: str
    entity_proposals: tuple[EntityProposal, ...] = ()
    broken_citations: tuple[BrokenCitation, ...] = ()
    stale_blocks: tuple[StaleBlock, ...] = ()
    consolidation_candidates: tuple[ConsolidationCandidate, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def total_findings(self) -> int:
        return (
            len(self.entity_proposals)
            + len(self.broken_citations)
            + len(self.stale_blocks)
            + len(self.consolidation_candidates)
        )


# --- Entity extraction patterns (mirrors entity_ingest.py) ---

_PROJECT_PATTERNS = [
    (re.compile(r"https?://github\.com/([\w-]+)/([\w.-]+)"), "github_repo"),
    (re.compile(r"/home/\w+/([\w.-]+)(?:/|$)"), "local_project"),
    (re.compile(r"\bPRJ-([\w-]+)\b"), "prj_ref"),
]

_TOOL_PATTERNS = [
    (re.compile(r"\bmcp[_-](?:server[_-])?([\w-]+)\b", re.I), "mcp_server"),
    (re.compile(
        r"\b(codex|gemini|claude|docker|kubectl|npm|pip|cargo|rustc|gcc|make)\b",
        re.I,
    ), "cli_tool"),
    (re.compile(r"\bTOOL-([\w-]+)\b"), "tool_ref"),
]

_PEOPLE_PATTERNS = [
    (re.compile(r"@([\w-]{2,30})\b"), "at_mention"),
    (re.compile(r"\bPER-([\w-]+)\b"), "per_ref"),
]

_IGNORE_DIRS = frozenset({
    "bin", "lib", "etc", "var", "tmp", "usr", "opt", "dev", "proc", "sys",
    "snap", "cache", "config", "local", "share", "node_modules", ".git",
    ".cache", ".local", ".config", ".openclaw", ".claude", ".ssh", ".gnupg",
    ".npm", ".cargo", ".rustup", "documents", "downloads", "desktop",
})

_TOOL_IGNORE = frozenset({"server", "make", "cargo", "rustc", "gcc"})

_PERSON_IGNORE = frozenset({
    "mention", "handle", "user", "admin", "bot", "system", "star",
    "here", "everyone", "channel", "all",
})

# Block ID patterns for citation checking
_BLOCK_ID_RE = re.compile(
    r"\b(D-\d{8}-\d{3}|T-\d{8}-\d{3}|INC-\d{8}-[a-z0-9-]+"
    r"|C-\d{8}-\d{3}|DREF-\d{8}-\d{3}|I-\d{8}-\d{3})\b"
)

# Block header pattern (lines starting with [ID])
_BLOCK_HEADER_RE = re.compile(r"^\[([\w-]+(?:-\d{8}-\d{3})?)\]", re.MULTILINE)

# Date pattern in file names
_DATE_FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.md$")


# --- Pass 1: Entity Discovery ---


def _extract_slug(m: re.Match, source: str) -> str | None:
    """Extract and validate a slug from a regex match."""
    if source == "github_repo":
        return m.group(2).lower().rstrip(".")
    slug = m.group(1).lower() if m.lastindex else m.group(0).lower()
    if source == "local_project" and (slug in _IGNORE_DIRS or len(slug) < 3):
        return None
    return slug


# Entity type -> (patterns, ignore set, min_len)
_ENTITY_GROUPS: list[tuple[str, list[tuple[re.Pattern, str]], frozenset[str], int]] = [
    ("project", _PROJECT_PATTERNS, frozenset(), 1),
    ("tool", _TOOL_PATTERNS, _TOOL_IGNORE, 2),
    ("person", _PEOPLE_PATTERNS, _PERSON_IGNORE, 2),
]


def _extract_raw_entities(text: str) -> list[dict[str, str]]:
    """Extract entity references from text. Returns list of entity dicts."""
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for etype, patterns, ignore, min_len in _ENTITY_GROUPS:
        for pattern, source in patterns:
            for m in pattern.finditer(text):
                slug = _extract_slug(m, source)
                if not slug or slug in ignore or slug in seen or len(slug) < min_len:
                    continue
                seen.add(slug)
                start = max(0, m.start() - 30)
                end = min(len(text), m.end() + 50)
                found.append({
                    "entity_type": etype, "slug": slug,
                    "source_pattern": source, "excerpt": text[start:end].strip(),
                })
    return found


def _load_tracked_slugs(workspace: str) -> set[str]:
    """Load all entity slugs already tracked in entities/*.md."""
    tracked: set[str] = set()
    entities_dir = os.path.join(workspace, "entities")
    if not os.path.isdir(entities_dir):
        return tracked

    for fname in os.listdir(entities_dir):
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(entities_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        for m in re.finditer(r"\[(PRJ|TOOL|PER)-([\w-]+)\]", content):
            tracked.add(m.group(2).lower())
    return tracked


def pass_entity_discovery(
    workspace: str,
    lookback_days: int = 7,
) -> list[EntityProposal]:
    """Scan recent daily logs for untracked entities."""
    memory_dir = os.path.join(workspace, "memory")
    if not os.path.isdir(memory_dir):
        _log.info("entity_discovery_skip", reason="no memory/ dir")
        return []

    cutoff = datetime.now() - timedelta(days=lookback_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    tracked = _load_tracked_slugs(workspace)
    proposals: list[EntityProposal] = []
    seen_global: set[str] = set()

    for fname in sorted(os.listdir(memory_dir)):
        match = _DATE_FILE_RE.match(fname)
        if not match:
            continue
        date_str = match.group(1)
        if date_str < cutoff_str:
            continue

        fpath = os.path.join(memory_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue

        entities = _extract_raw_entities(content)
        for ent in entities:
            slug = ent["slug"]
            if slug in tracked or slug in seen_global:
                continue
            seen_global.add(slug)
            proposals.append(EntityProposal(
                entity_type=ent["entity_type"],
                slug=slug,
                source_pattern=ent["source_pattern"],
                excerpt=ent["excerpt"][:120],
                source_file=fname,
            ))

    _log.info("entity_discovery_complete", found=len(proposals))
    metrics.inc("dream_entities_found", len(proposals))
    return proposals


# --- Pass 2: Citation Repair ---


def _scan_workspace_md_files(workspace: str) -> list[str]:
    """Collect all scannable .md files (decisions, tasks, entities)."""
    files: list[str] = []
    for rel in ("decisions/DECISIONS.md", "tasks/TASKS.md"):
        fpath = os.path.join(workspace, rel)
        if os.path.isfile(fpath):
            files.append(fpath)
    entities_dir = os.path.join(workspace, "entities")
    if os.path.isdir(entities_dir):
        for fname in os.listdir(entities_dir):
            if fname.endswith(".md"):
                files.append(os.path.join(entities_dir, fname))
    return files


def _collect_defined_ids(workspace: str) -> set[str]:
    """Collect all block IDs defined in the workspace (headers like [D-...])."""
    defined: set[str] = set()
    for fpath in _scan_workspace_md_files(workspace):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        for m in _BLOCK_HEADER_RE.finditer(content):
            defined.add(m.group(1))
    return defined


def pass_citation_repair(workspace: str) -> list[BrokenCitation]:
    """Find broken internal references (IDs citing nonexistent blocks)."""
    defined_ids = _collect_defined_ids(workspace)
    broken: list[BrokenCitation] = []

    for fpath in _scan_workspace_md_files(workspace):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            continue

        rel_path = os.path.relpath(fpath, workspace)
        for line_num, line in enumerate(lines, start=1):
            # Skip block header lines (definitions, not references)
            if line.startswith("[") and _BLOCK_HEADER_RE.match(line):
                continue
            for m in _BLOCK_ID_RE.finditer(line):
                cited_id = m.group(1)
                if cited_id not in defined_ids:
                    context = line.strip()[:120]
                    broken.append(BrokenCitation(
                        source_file=rel_path,
                        cited_id=cited_id,
                        line_number=line_num,
                        context=context,
                    ))

    _log.info("citation_repair_complete", broken=len(broken))
    metrics.inc("dream_broken_citations", len(broken))
    return broken


# --- Pass 3: Stale Block Detection ---


def _parse_date_from_id(block_id: str) -> str | None:
    """Extract YYYY-MM-DD from a block ID like D-20260213-001."""
    m = re.match(r"[A-Z]+-(\d{4})(\d{2})(\d{2})-\d{3}", block_id)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def pass_stale_detection(
    workspace: str,
    stale_days: int = 30,
) -> list[StaleBlock]:
    """Find blocks not updated in >stale_days (file mtime + embedded date)."""
    cutoff = datetime.now() - timedelta(days=stale_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    cutoff_ts = cutoff.timestamp()

    stale: list[StaleBlock] = []
    scan_files = {
        "decisions/DECISIONS.md": os.path.join(workspace, "decisions", "DECISIONS.md"),
        "tasks/TASKS.md": os.path.join(workspace, "tasks", "TASKS.md"),
    }

    for rel_path, fpath in scan_files.items():
        if not os.path.isfile(fpath):
            continue

        file_mtime = os.path.getmtime(fpath)
        file_is_old = file_mtime < cutoff_ts

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue

        for m in _BLOCK_HEADER_RE.finditer(content):
            block_id = m.group(1)
            embedded_date = _parse_date_from_id(block_id)

            # Determine staleness
            if embedded_date and embedded_date < cutoff_str:
                # Block date is old — check if file was also not touched recently
                if file_is_old:
                    days = (datetime.now() - datetime.strptime(
                        embedded_date, "%Y-%m-%d"
                    )).days
                    stale.append(StaleBlock(
                        block_id=block_id,
                        source_file=rel_path,
                        last_modified_date=embedded_date,
                        days_stale=days,
                    ))
            elif not embedded_date and file_is_old:
                # No date in ID, but file is old
                days = int((datetime.now().timestamp() - file_mtime) / 86400)
                last_mod = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d")
                stale.append(StaleBlock(
                    block_id=block_id,
                    source_file=rel_path,
                    last_modified_date=last_mod,
                    days_stale=days,
                ))

    _log.info("stale_detection_complete", stale=len(stale))
    metrics.inc("dream_stale_blocks", len(stale))
    return stale


# --- Pass 4: Consolidation ---


def _normalize_line(line: str) -> str:
    """Normalize a line for duplicate detection.

    Strips whitespace, lowercases, removes markdown formatting and dates.
    """
    text = line.strip().lower()
    # Remove markdown headers
    text = re.sub(r"^#+\s*", "", text)
    # Remove inline formatting
    text = re.sub(r"[*_`~]", "", text)
    # Remove date prefixes like "2026-03-15:"
    text = re.sub(r"^\d{4}-\d{2}-\d{2}[:\s]*", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pass_consolidation(
    workspace: str,
    lookback_days: int = 30,
    min_occurrences: int = 3,
) -> list[ConsolidationCandidate]:
    """Find facts repeated in min_occurrences+ daily logs (promotion candidates)."""
    memory_dir = os.path.join(workspace, "memory")
    if not os.path.isdir(memory_dir):
        _log.info("consolidation_skip", reason="no memory/ dir")
        return []

    cutoff = datetime.now() - timedelta(days=lookback_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    # Map: normalized_line -> set of source files
    line_sources: dict[str, set[str]] = {}
    # Map: normalized_line -> original (first seen) text
    line_originals: dict[str, str] = {}

    # Minimum line length to avoid matching trivial lines
    min_line_len = 20

    for fname in sorted(os.listdir(memory_dir)):
        match = _DATE_FILE_RE.match(fname)
        if not match:
            continue
        date_str = match.group(1)
        if date_str < cutoff_str:
            continue

        fpath = os.path.join(memory_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            continue

        seen_in_file: set[str] = set()
        for line in lines:
            normalized = _normalize_line(line)
            if len(normalized) < min_line_len:
                continue
            if normalized in seen_in_file:
                continue
            seen_in_file.add(normalized)

            if normalized not in line_sources:
                line_sources[normalized] = set()
                line_originals[normalized] = line.strip()
            line_sources[normalized].add(fname)

    # Filter to those appearing in >= min_occurrences files
    candidates: list[ConsolidationCandidate] = []
    for normalized, sources in line_sources.items():
        if len(sources) >= min_occurrences:
            candidates.append(ConsolidationCandidate(
                fact_text=line_originals[normalized],
                occurrences=len(sources),
                source_files=tuple(sorted(sources)),
            ))

    # Sort by occurrence count descending
    candidates.sort(key=lambda c: c.occurrences, reverse=True)

    _log.info("consolidation_complete", candidates=len(candidates))
    metrics.inc("dream_consolidation_candidates", len(candidates))
    return candidates


# --- Pass 5: Integrity Summary ---


def _format_report_markdown(report: DreamCycleReport) -> str:
    """Format a DreamCycleReport as markdown for writing to disk."""
    lines: list[str] = [
        f"# Dream Cycle Report — {report.timestamp}",
        "", f"Workspace: `{report.workspace}`",
        f"Total findings: {report.total_findings}", "",
    ]

    _prefix_map = {"project": "PRJ", "tool": "TOOL", "person": "PER"}

    # Pass 1
    lines += ["## Pass 1: Entity Discovery", ""]
    if report.entity_proposals:
        for ep in report.entity_proposals:
            p = _prefix_map.get(ep.entity_type, "UNK")
            lines.append(f"- **{p}-{ep.slug}** ({ep.source_pattern}) in `{ep.source_file}`")
    else:
        lines.append("No untracked entities found.")
    lines.append("")

    # Pass 2
    lines += ["## Pass 2: Citation Repair", ""]
    if report.broken_citations:
        for bc in report.broken_citations:
            lines.append(f"- `{bc.cited_id}` in `{bc.source_file}` (line {bc.line_number})")
    else:
        lines.append("All citations are valid.")
    lines.append("")

    # Pass 3
    lines += ["## Pass 3: Stale Block Detection", ""]
    if report.stale_blocks:
        for sb in report.stale_blocks:
            lines.append(f"- `{sb.block_id}` in `{sb.source_file}` ({sb.days_stale}d stale)")
    else:
        lines.append("No stale blocks found.")
    lines.append("")

    # Pass 4
    lines += ["## Pass 4: Consolidation", ""]
    if report.consolidation_candidates:
        for cc in report.consolidation_candidates:
            files = ", ".join(f"`{f}`" for f in cc.source_files[:5])
            lines.append(f"- \"{cc.fact_text}\" ({cc.occurrences}x in {files})")
    else:
        lines.append("No consolidation candidates found.")
    lines.append("")

    if report.errors:
        lines += ["## Errors", ""]
        lines += [f"- {err}" for err in report.errors]
        lines.append("")

    lines += ["---", f"*Generated by mind-mem dream cycle at {report.timestamp}*", ""]
    return "\n".join(lines)


def pass_integrity_summary(
    workspace: str,
    report: DreamCycleReport,
    dry_run: bool = False,
) -> str:
    """Write dream cycle summary to memory/dream-cycle-YYYY-MM-DD.md."""
    content = _format_report_markdown(report)

    if not dry_run:
        memory_dir = os.path.join(workspace, "memory")
        os.makedirs(memory_dir, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        report_path = os.path.join(memory_dir, f"dream-cycle-{today}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        _log.info("integrity_summary_written", path=report_path)

    return content


# --- Main entry point ---


def run_dream_cycle(
    workspace: str,
    dry_run: bool = False,
    lookback_days: int = 7,
    stale_days: int = 30,
    consolidation_lookback: int = 30,
    min_occurrences: int = 3,
) -> DreamCycleReport:
    """Run all 5 enrichment passes. Errors in one pass do not block others."""
    ws = os.path.abspath(workspace)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    errors: list[str] = []

    _log.info("dream_cycle_start", workspace=ws, dry_run=dry_run)

    # Pass 1: Entity Discovery
    entity_proposals: list[EntityProposal] = []
    with timed("dream_pass_entity_discovery", _log):
        try:
            entity_proposals = pass_entity_discovery(ws, lookback_days=lookback_days)
        except Exception as exc:
            msg = f"Pass 1 (entity discovery) failed: {exc}"
            _log.error(msg)
            errors.append(msg)

    # Pass 2: Citation Repair
    broken_citations: list[BrokenCitation] = []
    with timed("dream_pass_citation_repair", _log):
        try:
            broken_citations = pass_citation_repair(ws)
        except Exception as exc:
            msg = f"Pass 2 (citation repair) failed: {exc}"
            _log.error(msg)
            errors.append(msg)

    # Pass 3: Stale Block Detection
    stale_blocks: list[StaleBlock] = []
    with timed("dream_pass_stale_detection", _log):
        try:
            stale_blocks = pass_stale_detection(ws, stale_days=stale_days)
        except Exception as exc:
            msg = f"Pass 3 (stale detection) failed: {exc}"
            _log.error(msg)
            errors.append(msg)

    # Pass 4: Consolidation
    consolidation_candidates: list[ConsolidationCandidate] = []
    with timed("dream_pass_consolidation", _log):
        try:
            consolidation_candidates = pass_consolidation(
                ws,
                lookback_days=consolidation_lookback,
                min_occurrences=min_occurrences,
            )
        except Exception as exc:
            msg = f"Pass 4 (consolidation) failed: {exc}"
            _log.error(msg)
            errors.append(msg)

    report = DreamCycleReport(
        timestamp=timestamp,
        workspace=ws,
        entity_proposals=tuple(entity_proposals),
        broken_citations=tuple(broken_citations),
        stale_blocks=tuple(stale_blocks),
        consolidation_candidates=tuple(consolidation_candidates),
        errors=tuple(errors),
    )

    # Pass 5: Integrity Summary
    with timed("dream_pass_integrity_summary", _log):
        try:
            pass_integrity_summary(ws, report, dry_run=dry_run)
        except Exception as exc:
            msg = f"Pass 5 (integrity summary) failed: {exc}"
            _log.error(msg)
            errors = list(report.errors) + [msg]
            report = DreamCycleReport(
                timestamp=report.timestamp,
                workspace=report.workspace,
                entity_proposals=report.entity_proposals,
                broken_citations=report.broken_citations,
                stale_blocks=report.stale_blocks,
                consolidation_candidates=report.consolidation_candidates,
                errors=tuple(errors),
            )

    _log.info(
        "dream_cycle_complete",
        entities=len(entity_proposals),
        citations=len(broken_citations),
        stale=len(stale_blocks),
        consolidation=len(consolidation_candidates),
        errors=len(report.errors),
        total=report.total_findings,
    )
    metrics.inc("dream_cycle_runs")
    metrics.inc("dream_total_findings", report.total_findings)

    return report


# --- CLI ---


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="mind-mem Dream Cycle")
    parser.add_argument("workspace", nargs="?", default=".")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--pass", dest="single_pass", default=None,
        choices=["entity_discovery", "citation_repair", "stale_detection", "consolidation"],
    )
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--stale-days", type=int, default=30)
    parser.add_argument("--consolidation-lookback", type=int, default=30)
    parser.add_argument("--min-occurrences", type=int, default=3)
    args = parser.parse_args()

    ws = os.path.abspath(args.workspace)
    print(f"mind-mem dream cycle: {ws}")
    if args.dry_run:
        print("  (dry-run mode)")
    print()

    if args.single_pass:
        dispatch = {
            "entity_discovery": lambda: pass_entity_discovery(ws, lookback_days=args.lookback_days),
            "citation_repair": lambda: pass_citation_repair(ws),
            "stale_detection": lambda: pass_stale_detection(ws, stale_days=args.stale_days),
            "consolidation": lambda: pass_consolidation(
                ws, lookback_days=args.consolidation_lookback,
                min_occurrences=args.min_occurrences,
            ),
        }
        results = dispatch[args.single_pass]()
        print(f"{args.single_pass}: {len(results)} result(s)")
        for r in results:
            print(f"  {r}")
        return

    report = run_dream_cycle(
        ws, dry_run=args.dry_run, lookback_days=args.lookback_days,
        stale_days=args.stale_days, consolidation_lookback=args.consolidation_lookback,
        min_occurrences=args.min_occurrences,
    )
    labels = [
        ("Entity Discovery", len(report.entity_proposals)),
        ("Citation Repair", len(report.broken_citations)),
        ("Stale Detection", len(report.stale_blocks)),
        ("Consolidation", len(report.consolidation_candidates)),
    ]
    for label, count in labels:
        print(f"  {label}: {count}")
    print(f"  Summary: {'written' if not args.dry_run else 'dry-run'}")
    print(f"\nTotal findings: {report.total_findings}")
    if report.errors:
        for err in report.errors:
            print(f"  ERROR: {err}")


if __name__ == "__main__":
    main()
