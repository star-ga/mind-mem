#!/usr/bin/env python3
"""mind-mem Compiled Truth — synthesized entity pages with append-only evidence.

Implements the "compiled truth + append-only timeline" pattern. Each entity or
topic gets a page consisting of:

    - **Current best understanding** at the top (the "compiled" section —
      synthesized, up-to-date).
    - **Evidence trail** at the bottom (append-only timeline of raw
      observations that led to the current understanding).

The module manages the full lifecycle: create, load, save, add evidence,
recompile, detect contradictions, supersede, and scan for promotable facts.

Usage (library):
    from .compiled_truth import (
        CompiledTruthPage, EvidenceEntry,
        load_truth_page, save_truth_page, add_evidence, recompile_truth,
    )

    page = CompiledTruthPage(
        entity_id="PRJ-mind-mem",
        entity_type="project",
        compiled_section="mind-mem is a persistent AI memory system.",
        evidence_entries=[],
        last_compiled="2026-04-10T00:00:00+00:00",
        version=1,
    )
    entry = EvidenceEntry(
        timestamp="2026-04-10T12:00:00+00:00",
        source="memory/2026-04-10.md",
        observation="mind-mem v1.9.0 released with 8 new modules.",
        confidence="high",
        superseded=False,
    )
    page = add_evidence(page, entry)
    page = recompile_truth(page)
    save_truth_page("/path/to/workspace", page)

Usage (CLI):
    python3 -m mind_mem.compiled_truth load  workspace/ PRJ-mind-mem
    python3 -m mind_mem.compiled_truth scan  workspace/ --min-mentions 3
    python3 -m mind_mem.compiled_truth contradictions workspace/ PRJ-mind-mem

Zero external deps — dataclasses, os, re, collections (all stdlib).
"""

from __future__ import annotations

import dataclasses
import os
import re
from collections import Counter
from datetime import datetime, timezone

from .observability import get_logger, metrics

_log = get_logger("compiled_truth")

# ---------------------------------------------------------------------------
# Valid entity types and confidence levels
# ---------------------------------------------------------------------------

VALID_ENTITY_TYPES: frozenset[str] = frozenset({
    "project", "person", "tool", "topic",
})
VALID_CONFIDENCE_LEVELS: frozenset[str] = frozenset({
    "high", "medium", "low",
})

# Subdirectory under workspace for compiled truth pages
_COMPILED_DIR = os.path.join("entities", "compiled")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EvidenceEntry:
    """A single observation in the append-only evidence trail.

    Attributes:
        timestamp: ISO-8601 timestamp of when the observation was recorded.
        source: File path or session ID where the observation originated.
        observation: Raw text of what was observed.
        confidence: One of "high", "medium", "low".
        superseded: True when newer evidence replaces this entry.
    """

    timestamp: str
    source: str
    observation: str
    confidence: str
    superseded: bool = False


@dataclasses.dataclass(frozen=True)
class CompiledTruthPage:
    """A compiled truth page for a single entity or topic.

    Attributes:
        entity_id: Unique identifier (e.g. "PRJ-mind-mem").
        entity_type: One of "project", "person", "tool", "topic".
        compiled_section: Current best understanding (markdown text).
        evidence_entries: Append-only list of evidence entries.
        last_compiled: ISO-8601 timestamp of the most recent compilation.
        version: Incremented on each recompile.
    """

    entity_id: str
    entity_type: str
    compiled_section: str
    evidence_entries: list[EvidenceEntry]
    last_compiled: str
    version: int


# ---------------------------------------------------------------------------
# Markdown serialisation constants
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n",
    re.DOTALL,
)

_EVIDENCE_HEADER_RE = re.compile(
    r"^### "
    r"(\d{4}-\d{2}-\d{2}(?:T[\d:.+Z-]+)?)\s+"
    r"\[(\w+)\]\s+"
    r"\(source:\s+(.+?)\)"
    r"(?:\s+~~SUPERSEDED~~)?\s*$",
)


# ---------------------------------------------------------------------------
# Format / Parse
# ---------------------------------------------------------------------------


def format_truth_page(page: CompiledTruthPage) -> str:
    """Render a CompiledTruthPage as markdown.

    Layout:
        - YAML-style frontmatter (entity_id, entity_type, last_compiled, version)
        - ``# {entity_id} — Compiled Truth`` heading
        - ``## Current Understanding`` section with the compiled section body
        - ``## Evidence Trail`` section with one ``### date [CONF] (source: ...)``
          sub-heading per evidence entry, newest first.

    Returns:
        The full markdown string.
    """
    lines: list[str] = []

    # Frontmatter
    lines.append("---")
    lines.append(f"entity_id: {page.entity_id}")
    lines.append(f"entity_type: {page.entity_type}")
    lines.append(f"last_compiled: {page.last_compiled}")
    lines.append(f"version: {page.version}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {page.entity_id} — Compiled Truth")
    lines.append("")
    lines.append("## Current Understanding")
    lines.append("")
    lines.append(page.compiled_section.strip())
    lines.append("")
    lines.append("## Evidence Trail")
    lines.append("")

    for entry in page.evidence_entries:
        conf_tag = entry.confidence.upper()
        header = f"### {entry.timestamp} [{conf_tag}] (source: {entry.source})"
        if entry.superseded:
            header += " ~~SUPERSEDED~~"
        lines.append(header)
        lines.append("")
        lines.append(entry.observation.strip())
        lines.append("")

    return "\n".join(lines)


def parse_truth_page(markdown: str) -> CompiledTruthPage:
    """Parse a markdown truth page back into a CompiledTruthPage.

    Expects the format produced by ``format_truth_page``.

    Raises:
        ValueError: If required frontmatter fields are missing or the document
            structure cannot be parsed.
    """
    # --- Frontmatter ---
    fm_match = _FRONTMATTER_RE.match(markdown)
    if not fm_match:
        raise ValueError("Missing or malformed frontmatter")

    fm_text = fm_match.group(1)
    fm: dict[str, str] = {}
    for line in fm_text.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip()

    for required in ("entity_id", "entity_type", "last_compiled", "version"):
        if required not in fm:
            raise ValueError(f"Frontmatter missing required field: {required}")

    entity_id = fm["entity_id"]
    entity_type = fm["entity_type"]
    last_compiled = fm["last_compiled"]
    version = int(fm["version"])

    body = markdown[fm_match.end():]

    # --- Compiled section ---
    compiled_section = ""
    evidence_text = ""

    evidence_marker = "## Evidence Trail"
    compiled_marker = "## Current Understanding"

    if compiled_marker in body:
        after_compiled = body.split(compiled_marker, 1)[1]
        if evidence_marker in after_compiled:
            compiled_section = after_compiled.split(evidence_marker, 1)[0].strip()
            evidence_text = after_compiled.split(evidence_marker, 1)[1]
        else:
            compiled_section = after_compiled.strip()
    elif evidence_marker in body:
        evidence_text = body.split(evidence_marker, 1)[1]

    # Strip the heading line from compiled_section if present
    # (the split may leave the heading itself)
    compiled_section = compiled_section.strip()

    # --- Evidence entries ---
    entries: list[EvidenceEntry] = []
    if evidence_text.strip():
        # Split on ### headers
        chunks = re.split(r"(?=^### )", evidence_text.strip(), flags=re.MULTILINE)
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            first_line, _, rest = chunk.partition("\n")
            hdr_match = _EVIDENCE_HEADER_RE.match(first_line.strip())
            if not hdr_match:
                continue
            timestamp = hdr_match.group(1)
            confidence = hdr_match.group(2).lower()
            source = hdr_match.group(3)
            superseded = "~~SUPERSEDED~~" in first_line
            observation = rest.strip()

            entries.append(EvidenceEntry(
                timestamp=timestamp,
                source=source,
                observation=observation,
                confidence=confidence,
                superseded=superseded,
            ))

    return CompiledTruthPage(
        entity_id=entity_id,
        entity_type=entity_type,
        compiled_section=compiled_section,
        evidence_entries=entries,
        last_compiled=last_compiled,
        version=version,
    )


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------


def load_truth_page(workspace: str, entity_id: str) -> CompiledTruthPage | None:
    """Load a compiled truth page from disk.

    Reads from ``{workspace}/entities/compiled/{entity_id}.md``.

    Args:
        workspace: Root workspace directory.
        entity_id: Entity identifier (used as filename stem).

    Returns:
        The parsed page, or ``None`` if the file does not exist.
    """
    path = os.path.join(workspace, _COMPILED_DIR, f"{entity_id}.md")
    if not os.path.isfile(path):
        _log.debug("truth_page_not_found", entity_id=entity_id, path=path)
        return None

    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()

    page = parse_truth_page(content)
    _log.info("truth_page_loaded", entity_id=entity_id, version=page.version)
    metrics.inc("truth_pages_loaded")
    return page


def save_truth_page(workspace: str, page: CompiledTruthPage) -> str:
    """Save a compiled truth page to disk.

    Writes to ``{workspace}/entities/compiled/{entity_id}.md``.

    Args:
        workspace: Root workspace directory.
        page: The page to persist.

    Returns:
        The absolute path of the written file.
    """
    dir_path = os.path.join(workspace, _COMPILED_DIR)
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, f"{page.entity_id}.md")
    content = format_truth_page(page)

    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    _log.info(
        "truth_page_saved",
        entity_id=page.entity_id,
        version=page.version,
        evidence_count=len(page.evidence_entries),
    )
    metrics.inc("truth_pages_saved")
    return os.path.abspath(file_path)


# ---------------------------------------------------------------------------
# Immutable mutation helpers
# ---------------------------------------------------------------------------


def add_evidence(
    page: CompiledTruthPage,
    entry: EvidenceEntry,
) -> CompiledTruthPage:
    """Append an evidence entry to the page (immutable).

    Returns a **new** ``CompiledTruthPage`` with the entry appended.  The
    original page is not modified.

    Args:
        page: Existing truth page.
        entry: New evidence to append.

    Returns:
        A new page with the evidence entry appended.

    Raises:
        ValueError: If the confidence level is invalid.
    """
    if entry.confidence not in VALID_CONFIDENCE_LEVELS:
        raise ValueError(
            f"Invalid confidence '{entry.confidence}'; "
            f"must be one of {sorted(VALID_CONFIDENCE_LEVELS)}"
        )

    new_entries = list(page.evidence_entries) + [entry]

    _log.info(
        "evidence_added",
        entity_id=page.entity_id,
        source=entry.source,
        confidence=entry.confidence,
    )
    metrics.inc("evidence_entries_added")

    return dataclasses.replace(page, evidence_entries=new_entries)


def supersede_evidence(
    page: CompiledTruthPage,
    entry_index: int,
    reason: str,
) -> CompiledTruthPage:
    """Mark an evidence entry as superseded (immutable).

    Returns a **new** ``CompiledTruthPage`` with the indicated entry replaced
    by a copy that has ``superseded=True``.

    Args:
        page: Existing truth page.
        entry_index: Zero-based index into ``evidence_entries``.
        reason: Human-readable reason for superseding.

    Returns:
        A new page with the entry marked as superseded.

    Raises:
        IndexError: If ``entry_index`` is out of range.
    """
    if entry_index < 0 or entry_index >= len(page.evidence_entries):
        raise IndexError(
            f"entry_index {entry_index} out of range "
            f"(page has {len(page.evidence_entries)} entries)"
        )

    old_entry = page.evidence_entries[entry_index]
    if old_entry.superseded:
        _log.debug(
            "evidence_already_superseded",
            entity_id=page.entity_id,
            index=entry_index,
        )
        return page

    new_entry = dataclasses.replace(old_entry, superseded=True)
    new_entries = list(page.evidence_entries)
    new_entries[entry_index] = new_entry

    _log.info(
        "evidence_superseded",
        entity_id=page.entity_id,
        index=entry_index,
        reason=reason,
    )
    metrics.inc("evidence_entries_superseded")

    return dataclasses.replace(page, evidence_entries=new_entries)


# ---------------------------------------------------------------------------
# Recompilation
# ---------------------------------------------------------------------------


def recompile_truth(page: CompiledTruthPage) -> CompiledTruthPage:
    """Regenerate the compiled section from non-superseded evidence.

    Produces a bullet-point summary ordered by timestamp (newest first).
    Each active evidence entry contributes a single bullet.  The version
    is incremented and ``last_compiled`` is set to the current UTC time.

    Returns:
        A **new** page with updated compiled section, version, and timestamp.
    """
    active = [e for e in page.evidence_entries if not e.superseded]

    if not active:
        compiled = "(No active evidence.)"
    else:
        # Sort newest-first for the summary
        sorted_entries = sorted(active, key=lambda e: e.timestamp, reverse=True)
        bullets: list[str] = []
        for entry in sorted_entries:
            conf = entry.confidence.upper()
            bullets.append(f"- **[{conf}]** {entry.observation.strip()}")
        compiled = "\n".join(bullets)

    now_iso = datetime.now(timezone.utc).isoformat()
    new_version = page.version + 1

    _log.info(
        "truth_recompiled",
        entity_id=page.entity_id,
        version=new_version,
        active_evidence=len(active),
    )
    metrics.inc("truth_recompilations")

    return dataclasses.replace(
        page,
        compiled_section=compiled,
        last_compiled=now_iso,
        version=new_version,
    )


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------


def _normalise_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def _word_set(text: str) -> set[str]:
    """Return the set of words in normalised text."""
    return set(_normalise_text(text).split())


_NEGATION_WORDS: frozenset[str] = frozenset({
    "not", "no", "never", "none", "neither", "nor", "cannot",
    "without", "lack", "lacks", "remove", "removed", "disable",
    "disabled", "drop", "dropped", "reject", "rejected",
})

_ANTONYM_PAIRS: list[tuple[str, str]] = [
    ("increase", "decrease"),
    ("add", "remove"),
    ("enable", "disable"),
    ("active", "inactive"),
    ("start", "stop"),
    ("success", "failure"),
    ("pass", "fail"),
    ("true", "false"),
    ("yes", "no"),
    ("up", "down"),
    ("open", "close"),
    ("high", "low"),
    ("fast", "slow"),
    ("new", "old"),
]


def detect_contradictions(
    page: CompiledTruthPage,
) -> list[tuple[EvidenceEntry, EvidenceEntry, str]]:
    """Find evidence entries that may contradict each other.

    Compares every pair of non-superseded entries and flags contradictions
    based on two heuristics:

    1. **Negation asymmetry**: entries share significant word overlap but
       one contains negation words the other does not.
    2. **Antonym presence**: entries share topic words and contain known
       antonym pairs.

    Args:
        page: The truth page to analyse.

    Returns:
        A list of ``(entry_a, entry_b, reason)`` tuples describing
        detected contradictions.
    """
    active = [e for e in page.evidence_entries if not e.superseded]
    contradictions: list[tuple[EvidenceEntry, EvidenceEntry, str]] = []

    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a = active[i]
            b = active[j]
            words_a = _word_set(a.observation)
            words_b = _word_set(b.observation)
            overlap = words_a & words_b

            # Heuristic 1: negation asymmetry with shared topic
            if len(overlap) >= 3:
                neg_a = words_a & _NEGATION_WORDS
                neg_b = words_b & _NEGATION_WORDS
                if neg_a and not neg_b:
                    contradictions.append((
                        a, b,
                        f"Negation asymmetry: entry A contains {sorted(neg_a)} "
                        f"but entry B does not, while sharing topics: "
                        f"{sorted(overlap - _NEGATION_WORDS)[:5]}",
                    ))
                    continue
                if neg_b and not neg_a:
                    contradictions.append((
                        a, b,
                        f"Negation asymmetry: entry B contains {sorted(neg_b)} "
                        f"but entry A does not, while sharing topics: "
                        f"{sorted(overlap - _NEGATION_WORDS)[:5]}",
                    ))
                    continue

            # Heuristic 2: antonym pairs in entries with shared context
            if len(overlap) >= 2:
                for word_x, word_y in _ANTONYM_PAIRS:
                    if word_x in words_a and word_y in words_b:
                        contradictions.append((
                            a, b,
                            f"Antonym pair: A has '{word_x}', B has '{word_y}' "
                            f"with shared context: {sorted(overlap)[:5]}",
                        ))
                        break
                    if word_y in words_a and word_x in words_b:
                        contradictions.append((
                            a, b,
                            f"Antonym pair: A has '{word_y}', B has '{word_x}' "
                            f"with shared context: {sorted(overlap)[:5]}",
                        ))
                        break

    _log.info(
        "contradictions_detected",
        entity_id=page.entity_id,
        count=len(contradictions),
    )
    metrics.inc("truth_contradictions_detected", len(contradictions))
    return contradictions


# ---------------------------------------------------------------------------
# Scan for promotable facts
# ---------------------------------------------------------------------------


def scan_for_promotable_facts(
    workspace: str,
    min_mentions: int = 3,
) -> list[dict]:
    """Scan daily memory logs for facts mentioned repeatedly.

    Reads markdown files from ``{workspace}/memory/`` and counts sentence-level
    occurrences.  Any sentence (normalised) appearing at least ``min_mentions``
    times is returned as a candidate for promotion to a compiled truth page.

    Args:
        workspace: Root workspace directory.
        min_mentions: Minimum occurrences to qualify (default 3).

    Returns:
        A list of dicts, each with keys ``fact``, ``mentions``, and
        ``sources`` (list of filenames where the fact appeared).
    """
    memory_dir = os.path.join(workspace, "memory")
    if not os.path.isdir(memory_dir):
        _log.debug("no_memory_dir", workspace=workspace)
        return []

    # Sentence -> list of source filenames
    sentence_sources: dict[str, list[str]] = {}

    for fname in sorted(os.listdir(memory_dir)):
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(memory_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                content = fh.read()
        except OSError:
            continue

        # Split into paragraphs first, then into sentences
        paragraphs = re.split(r"\n\s*\n", content)
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # Skip markdown headers, frontmatter, and list items
            if para.startswith("#") or para.startswith("---"):
                continue
            # Split paragraph into sentences
            raw_sentences = re.split(r"(?<=[.!?])\s+", para)
            for raw_sentence in raw_sentences:
                sentence = raw_sentence.strip()
                if len(sentence) < 20:
                    continue
                if sentence.startswith("#") or sentence.startswith("---"):
                    continue
                normalised = _normalise_text(sentence)
                if normalised not in sentence_sources:
                    sentence_sources[normalised] = []
                if fname not in sentence_sources[normalised]:
                    sentence_sources[normalised].append(fname)

    promotable: list[dict] = []
    for fact, sources in sentence_sources.items():
        if len(sources) >= min_mentions:
            promotable.append({
                "fact": fact,
                "mentions": len(sources),
                "sources": sources,
            })

    # Sort by mention count descending
    promotable.sort(key=lambda d: d["mentions"], reverse=True)

    _log.info(
        "promotable_facts_scanned",
        workspace=workspace,
        total_sentences=len(sentence_sources),
        promotable=len(promotable),
    )
    metrics.inc("promotable_facts_scanned", len(promotable))
    return promotable


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="mind-mem compiled truth page manager",
    )
    sub = parser.add_subparsers(dest="command")

    # load
    p_load = sub.add_parser("load", help="Load and display a truth page")
    p_load.add_argument("workspace", help="Workspace directory")
    p_load.add_argument("entity_id", help="Entity ID to load")

    # scan
    p_scan = sub.add_parser("scan", help="Scan for promotable facts")
    p_scan.add_argument("workspace", help="Workspace directory")
    p_scan.add_argument(
        "--min-mentions", type=int, default=3,
        help="Minimum mentions to qualify (default: 3)",
    )

    # contradictions
    p_contra = sub.add_parser(
        "contradictions", help="Detect contradictions in a truth page",
    )
    p_contra.add_argument("workspace", help="Workspace directory")
    p_contra.add_argument("entity_id", help="Entity ID to analyse")

    args = parser.parse_args()

    if args.command == "load":
        page = load_truth_page(args.workspace, args.entity_id)
        if page is None:
            print(f"No truth page found for {args.entity_id}", file=sys.stderr)
            sys.exit(1)
        print(format_truth_page(page))

    elif args.command == "scan":
        facts = scan_for_promotable_facts(args.workspace, args.min_mentions)
        if not facts:
            print("No promotable facts found.")
        else:
            for f in facts:
                print(f"[{f['mentions']}x] {f['fact']}")
                print(f"    sources: {', '.join(f['sources'])}")
                print()

    elif args.command == "contradictions":
        page = load_truth_page(args.workspace, args.entity_id)
        if page is None:
            print(f"No truth page found for {args.entity_id}", file=sys.stderr)
            sys.exit(1)
        conflicts = detect_contradictions(page)
        if not conflicts:
            print("No contradictions detected.")
        else:
            for a, b, reason in conflicts:
                print(f"CONTRADICTION: {reason}")
                print(f"  A [{a.timestamp}]: {a.observation[:80]}")
                print(f"  B [{b.timestamp}]: {b.observation[:80]}")
                print()

    else:
        parser.print_help()
        sys.exit(1)
