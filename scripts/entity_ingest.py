#!/usr/bin/env python3
"""mind-mem Entity Ingestion — regex-based entity extraction. Zero external deps.

Scans SIGNALS.md and daily logs for entity references not yet tracked in
entities/*.md. Proposes new entities as auto-capture-entity signals.

Entity types:
  - Projects: GitHub repo URLs, /home/n/*/ paths, PRJ-* refs
  - Tools: Known CLI names, MCP server refs, TOOL-* refs
  - People: @handle mentions, PER-* refs

Usage:
    python3 scripts/entity_ingest.py workspace/
    python3 scripts/entity_ingest.py workspace/ --dry-run
    python3 scripts/entity_ingest.py workspace/ --source path/to/file.md
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture import append_signals
from observability import get_logger, metrics

_log = get_logger("entity_ingest")


# ---------------------------------------------------------------------------
# Entity extraction patterns
# ---------------------------------------------------------------------------

# Projects: GitHub URLs and local paths
PROJECT_PATTERNS = [
    # GitHub repo URLs
    (re.compile(r"https?://github\.com/([\w-]+)/([\w.-]+)"), "github_repo"),
    # Local project paths under /home/n/
    (re.compile(r"/home/\w+/([\w.-]+)(?:/|$)"), "local_project"),
    # Explicit PRJ- references
    (re.compile(r"\bPRJ-([\w-]+)\b"), "prj_ref"),
]

# Tools: CLI names and MCP references
TOOL_PATTERNS = [
    # MCP server references
    (re.compile(r"\bmcp[_-](?:server[_-])?([\w-]+)\b", re.I), "mcp_server"),
    # Known CLI tool invocations
    (re.compile(r"\b(codex|gemini|claude|naestro|docker|kubectl|npm|pip|cargo|rustc|gcc|make)\b",
                re.I), "cli_tool"),
    # Explicit TOOL- references
    (re.compile(r"\bTOOL-([\w-]+)\b"), "tool_ref"),
]

# People: @mentions
PEOPLE_PATTERNS = [
    (re.compile(r"@([\w-]{2,30})\b"), "at_mention"),
    (re.compile(r"\bPER-([\w-]+)\b"), "per_ref"),
]

# Directories that are likely projects (not system dirs)
IGNORE_DIRS = frozenset({
    "bin", "lib", "etc", "var", "tmp", "usr", "opt", "dev", "proc", "sys",
    "snap", "cache", "config", "local", "share", "node_modules", ".git",
    ".cache", ".local", ".config", ".openclaw", ".claude", ".ssh", ".gnupg",
    ".npm", ".cargo", ".rustup", ".moltbot", ".claude-ultimate",
    "documents", "downloads", "desktop",
})

# Known tool slug aliases — maps detected slug to canonical registry slug
TOOL_ALIASES: dict[str, str] = {
    "claude": "codex-cli",  # claude CLI is tracked as TOOL-codex-cli
    "codex": "codex-cli",   # same tool, different name
    "gemini": "gemini-cli",
    "docker": "docker",
    "kubectl": "kubectl",
    "npm": "npm",
    "pip": "pip",
}

# Tool slugs to ignore (false positives from pattern matching)
TOOL_IGNORE = frozenset({
    "server",  # mcp_server without suffix matches as "server"
    "make",    # build tool, too generic
    "cargo", "rustc", "gcc",  # low-signal system tools
})


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

def load_existing_entities(workspace: str) -> dict[str, set[str]]:
    """Load existing entity IDs from entities/*.md.

    Returns dict mapping entity type to set of lowercase slugs.
    """
    entities: dict[str, set[str]] = {
        "projects": set(),
        "tools": set(),
        "people": set(),
    }

    entities_dir = os.path.join(workspace, "entities")
    if not os.path.isdir(entities_dir):
        return entities

    for fname in os.listdir(entities_dir):
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(entities_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract all entity IDs: [PRJ-xxx], [TOOL-xxx], [PER-xxx]
        for m in re.finditer(r"\[(PRJ|TOOL|PER)-([\w-]+)\]", content):
            prefix, slug = m.group(1), m.group(2).lower()
            if prefix == "PRJ":
                entities["projects"].add(slug)
            elif prefix == "TOOL":
                entities["tools"].add(slug)
            elif prefix == "PER":
                entities["people"].add(slug)

        # Also extract Name: fields for fuzzy matching
        for m in re.finditer(r"^Name:\s*(.+)$", content, re.MULTILINE):
            name = m.group(1).strip().lower().replace(" ", "-")
            # Add to appropriate set based on file name
            if "project" in fname.lower():
                entities["projects"].add(name)
            elif "tool" in fname.lower():
                entities["tools"].add(name)
            elif "people" in fname.lower() or "person" in fname.lower():
                entities["people"].add(name)

    return entities


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> list[dict]:
    """Extract entity references from text.

    Returns list of dicts: {type, slug, source_pattern, excerpt}.
    """
    found = []
    seen_slugs: set[str] = set()

    # Projects
    for pattern, source in PROJECT_PATTERNS:
        for m in pattern.finditer(text):
            if source == "github_repo":
                slug = m.group(2).lower().rstrip(".")
            elif source == "local_project":
                slug = m.group(1).lower()
                if slug in IGNORE_DIRS or len(slug) < 3:
                    continue
            else:
                slug = m.group(1).lower()

            if slug not in seen_slugs:
                seen_slugs.add(slug)
                # Get surrounding context
                start = max(0, m.start() - 30)
                end = min(len(text), m.end() + 50)
                found.append({
                    "entity_type": "project",
                    "slug": slug,
                    "source_pattern": source,
                    "excerpt": text[start:end].strip(),
                })

    # Tools
    for pattern, source in TOOL_PATTERNS:
        for m in pattern.finditer(text):
            slug = m.group(1).lower() if m.lastindex else m.group(0).lower()
            if slug in TOOL_IGNORE:
                continue
            if slug not in seen_slugs and len(slug) > 1:
                seen_slugs.add(slug)
                start = max(0, m.start() - 30)
                end = min(len(text), m.end() + 50)
                found.append({
                    "entity_type": "tool",
                    "slug": slug,
                    "source_pattern": source,
                    "excerpt": text[start:end].strip(),
                })

    # People
    for pattern, source in PEOPLE_PATTERNS:
        for m in pattern.finditer(text):
            slug = m.group(1).lower()
            # Filter out common non-person mentions
            if slug in {"mention", "handle", "user", "admin", "bot", "system",
                         "star", "here", "everyone", "channel", "all"}:
                continue
            if slug not in seen_slugs and len(slug) > 1:
                seen_slugs.add(slug)
                start = max(0, m.start() - 30)
                end = min(len(text), m.end() + 50)
                found.append({
                    "entity_type": "person",
                    "slug": slug,
                    "source_pattern": source,
                    "excerpt": text[start:end].strip(),
                })

    return found


def filter_new_entities(
    entities: list[dict],
    existing: dict[str, set[str]],
) -> list[dict]:
    """Filter out entities already tracked in the registry."""
    new = []
    for ent in entities:
        slug = ent["slug"]
        etype = ent["entity_type"]
        registry_key = "projects" if etype == "project" else (
            "tools" if etype == "tool" else "people"
        )
        registry = existing.get(registry_key, set())
        # Check both the raw slug and any known alias
        canonical = TOOL_ALIASES.get(slug, slug) if etype == "tool" else slug
        if slug not in registry and canonical not in registry:
            new.append(ent)
    return new


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def entities_to_signals(entities: list[dict], source_file: str) -> list[dict]:
    """Convert extracted entities to signal dicts for append_signals."""
    signals = []
    for ent in entities:
        prefix = {"project": "PRJ", "tool": "TOOL", "person": "PER"}[ent["entity_type"]]
        signals.append({
            "line": 0,
            "type": "entity",
            "text": f"New {ent['entity_type']} detected: {prefix}-{ent['slug']} "
                    f"(via {ent['source_pattern']}) — {ent['excerpt'][:100]}",
            "pattern": "auto-capture-entity",
            "confidence": "medium",
            "priority": "P2",
            "structure": {
                "subject": f"{prefix}-{ent['slug']}",
                "object": ent["entity_type"],
                "tags": [f"entity-{ent['entity_type']}", "auto-ingest"],
            },
        })
    return signals


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="mind-mem Entity Ingestion")
    parser.add_argument("workspace", nargs="?", default=".")
    parser.add_argument("--source", help="Scan a specific file instead of workspace")
    parser.add_argument("--dry-run", action="store_true", help="Show entities without writing")
    args = parser.parse_args()

    ws = os.path.abspath(args.workspace)
    existing = load_existing_entities(ws)
    _log.info("registry_loaded",
              projects=len(existing["projects"]),
              tools=len(existing["tools"]),
              people=len(existing["people"]))

    # Collect text to scan
    texts: list[tuple[str, str]] = []  # (source_name, content)

    if args.source:
        with open(args.source, "r", encoding="utf-8") as f:
            texts.append((args.source, f.read()))
    else:
        # Scan SIGNALS.md
        signals_path = os.path.join(ws, "intelligence", "SIGNALS.md")
        if os.path.isfile(signals_path):
            with open(signals_path, "r", encoding="utf-8") as f:
                texts.append(("SIGNALS.md", f.read()))

        # Scan recent daily logs
        memory_dir = os.path.join(ws, "memory")
        if os.path.isdir(memory_dir):
            for fname in sorted(os.listdir(memory_dir), reverse=True)[:7]:
                if fname.endswith(".md"):
                    fpath = os.path.join(memory_dir, fname)
                    with open(fpath, "r", encoding="utf-8") as f:
                        texts.append((fname, f.read()))

    # Extract and filter
    all_entities = []
    for source_name, content in texts:
        entities = extract_entities(content)
        new_entities = filter_new_entities(entities, existing)
        if new_entities:
            all_entities.extend(new_entities)
            for ent in new_entities:
                print(f"  NEW {ent['entity_type']}: {ent['slug']} (via {ent['source_pattern']})")

    # Deduplicate across sources
    seen = set()
    unique_entities = []
    for ent in all_entities:
        key = (ent["entity_type"], ent["slug"])
        if key not in seen:
            seen.add(key)
            unique_entities.append(ent)

    if not unique_entities:
        print("No new entities found.")
        return

    print(f"\n{len(unique_entities)} new entity/entities found")

    if not args.dry_run:
        signals = entities_to_signals(unique_entities, "entity_ingest")
        today = datetime.now().strftime("%Y-%m-%d")
        written = append_signals(ws, signals, today)
        print(f"{written} entity signal(s) written to SIGNALS.md")
        metrics.inc("entities_detected", len(unique_entities))
        metrics.inc("entity_signals_written", written)
    else:
        print("[DRY RUN] Would write signals for the above entities")


if __name__ == "__main__":
    main()
