#!/usr/bin/env python3
"""mind-mem Category Distiller — auto-generates thematic summary files from memory blocks.

Deterministic category detection system (NO LLM required). Scans memory blocks
(decisions, tasks, entities, signals) and produces thematic summary files in
the ``categories/`` directory of a workspace.

Categories are auto-detected from block tags, entity types, and content keyword
matching. Each category file is a Markdown summary that references source blocks
by ID, suitable for anticipatory context assembly.

Usage (CLI):
    python3 scripts/category_distiller.py /path/to/workspace
    python3 scripts/category_distiller.py /path/to/workspace --json
    python3 scripts/category_distiller.py /path/to/workspace --query "deploy pipeline"

As library:
    from category_distiller import CategoryDistiller
    cd = CategoryDistiller()
    written = cd.distill("/path/to/workspace")
    context = cd.get_category_context("deploy workflow", "/path/to/workspace")
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

# Same-directory imports (block_parser, observability live alongside this file)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from block_parser import get_active, parse_file
except ImportError:
    # Minimal fallback when block_parser is unavailable — parse nothing gracefully.
    def parse_file(path: str) -> list[dict]:  # type: ignore[misc]
        return []

    def get_active(blocks: list[dict], status_field: str = "Status",  # type: ignore[misc]
                   active_value: str = "active") -> list[dict]:
        return [b for b in blocks if b.get(status_field) == active_value]

try:
    from observability import get_logger
except ImportError:
    # Lightweight fallback logger — prints to stderr with a component prefix.
    class _FallbackLogger:
        def __init__(self, name: str) -> None:
            self.name = name

        def _emit(self, level: str, event: str, **kwargs: object) -> None:
            extra = f" {kwargs}" if kwargs else ""
            print(f"[{level}] {self.name}: {event}{extra}", file=sys.stderr)

        def debug(self, event: str, **kwargs: object) -> None:
            self._emit("DEBUG", event, **kwargs)

        def info(self, event: str, **kwargs: object) -> None:
            self._emit("INFO", event, **kwargs)

        def warning(self, event: str, **kwargs: object) -> None:
            self._emit("WARN", event, **kwargs)

        def error(self, event: str, **kwargs: object) -> None:
            self._emit("ERROR", event, **kwargs)

    def get_logger(component: str):  # type: ignore[misc]
        return _FallbackLogger(component)

log = get_logger("category_distiller")

# Optional MIND kernel acceleration
_mind_ffi = None
try:
    from mind_ffi import MindMemKernel
    _k = MindMemKernel()
    if _k._lib is not None:
        _mind_ffi = _k
except (ImportError, OSError):
    pass


# ---------------------------------------------------------------------------
# CategoryDistiller
# ---------------------------------------------------------------------------

class CategoryDistiller:
    """Periodically scans memory blocks and produces category summary files.

    Categories are auto-detected from block tags, entity types, and content
    clustering via deterministic keyword matching.  Each category file is a
    Markdown summary that references source blocks by ID.
    """

    DEFAULT_CATEGORIES = [
        "architecture", "decisions", "people", "preferences",
        "workflows", "bugs", "credentials", "integrations",
        "goals", "constraints",
    ]

    # Keyword maps for deterministic categorisation.
    # Each keyword list is checked against block text (case-insensitive).
    CATEGORY_KEYWORDS: dict[str, list[str]] = {
        "architecture": [
            "architecture", "design", "pattern", "schema", "layer",
            "api", "endpoint", "microservice", "database", "infra", "deploy",
        ],
        "decisions": [
            "decision", "decided", "chose", "adopted", "rejected",
            "migrated", "switched",
        ],
        "people": [
            "team", "person", "member", "hire", "onboard",
            "contact", "stakeholder", "@",
        ],
        "preferences": [
            "prefer", "always", "never", "convention", "style",
            "format", "standard", "rule",
        ],
        "workflows": [
            "workflow", "process", "pipeline", "ci", "cd",
            "deploy", "release", "review", "merge",
        ],
        "bugs": [
            "bug", "fix", "error", "crash", "regression",
            "issue", "debug", "patch", "hotfix",
        ],
        "credentials": [
            "key", "token", "secret", "password", "auth",
            "oauth", "credential", "api-key", "certificate",
        ],
        "integrations": [
            "integration", "webhook", "plugin", "extension",
            "connector", "sdk", "library", "package",
        ],
        "goals": [
            "goal", "objective", "milestone", "target", "roadmap",
            "plan", "deadline", "priority",
        ],
        "constraints": [
            "constraint", "limitation", "requirement", "must",
            "cannot", "blocker", "dependency",
        ],
    }

    # Minimum combined score for a block to be assigned to a category.
    _SCORE_THRESHOLD = 2

    def __init__(self, extra_categories: dict[str, list[str]] | None = None) -> None:
        """Initialise with optional extra category keyword maps.

        Parameters
        ----------
        extra_categories:
            Mapping of ``category_name -> [keywords]``.  If the category
            already exists in ``CATEGORY_KEYWORDS`` the keywords are appended;
            otherwise a new category is created.
        """
        self.categories: dict[str, list[str]] = {
            k: list(v) for k, v in self.CATEGORY_KEYWORDS.items()
        }
        if extra_categories:
            for cat, keywords in extra_categories.items():
                if cat in self.categories:
                    self.categories[cat].extend(keywords)
                else:
                    self.categories[cat] = list(keywords)

    # ------------------------------------------------------------------
    # Block scanning
    # ------------------------------------------------------------------

    def _scan_blocks(self, workspace: str) -> list[dict]:
        """Scan all memory block files and return parsed blocks."""
        blocks: list[dict] = []

        # Well-known single-file sources
        scan_dirs: dict[str, list[str]] = {
            "decisions": ["DECISIONS.md"],
            "tasks": ["TASKS.md"],
            "intelligence": ["SIGNALS.md"],
        }

        for subdir, filenames in scan_dirs.items():
            for fn in filenames:
                path = os.path.join(workspace, subdir, fn)
                if os.path.isfile(path):
                    try:
                        parsed = parse_file(path)
                        for b in parsed:
                            b["_source_dir"] = subdir
                        blocks.extend(parsed)
                    except Exception as exc:
                        log.warning("block_parse_failed", path=path, error=str(exc))

        # Scan entities directory — every .md file
        entities_dir = os.path.join(workspace, "entities")
        if os.path.isdir(entities_dir):
            for fn in sorted(os.listdir(entities_dir)):
                if fn.endswith(".md"):
                    path = os.path.join(entities_dir, fn)
                    try:
                        parsed = parse_file(path)
                        for b in parsed:
                            b["_source_dir"] = "entities"
                        blocks.extend(parsed)
                    except Exception as exc:
                        log.warning("entity_parse_failed", path=path, error=str(exc))

        log.info("scan_complete", total_blocks=len(blocks))
        return blocks

    # ------------------------------------------------------------------
    # Categorisation (pure deterministic)
    # ------------------------------------------------------------------

    def categorize_block(self, block: dict) -> list[str]:
        """Assign categories to a block based on tags, content keywords, and fields.

        Scoring rules (all matching is case-insensitive):
        - Exact tag match:     +3
        - Content keyword hit: +1
        A category is assigned when its combined score >= ``_SCORE_THRESHOLD``
        (default 2), meaning at least two keyword hits or one exact tag match.

        Returns ``["uncategorized"]`` when nothing matches.
        """
        matched: list[str] = []

        # Assemble searchable text from relevant fields
        text_parts: list[str] = []
        for field in ("Statement", "Description", "Tags", "Subject", "Rationale"):
            val = block.get(field, "")
            if val:
                text_parts.append(str(val).lower())
        text = " ".join(text_parts)

        # Parse tags into a list for exact matching
        tags_str = block.get("Tags", "")
        tags_list: list[str] = (
            [t.strip().lower() for t in tags_str.split(",") if t.strip()]
            if tags_str else []
        )

        for category, keywords in self.categories.items():
            score = 0
            for kw in keywords:
                kw_lower = kw.lower()
                # Exact tag match is weighted higher
                if kw_lower in tags_list:
                    score += 3
                # Content keyword match
                if kw_lower in text:
                    score += 1
            if score >= self._SCORE_THRESHOLD:
                matched.append(category)

        if not matched:
            matched.append("uncategorized")

        return matched

    # ------------------------------------------------------------------
    # Distillation pipeline
    # ------------------------------------------------------------------

    def _batch_categorize_mind(self, blocks: list[dict]) -> dict[str, list[dict]]:
        """Batch categorize using MIND C kernels when available.

        Pre-computes keyword and tag overlap matrices [N, C], then uses
        category_affinity + category_assign kernels for O(1)-per-element
        scoring instead of nested Python loops.
        """
        cat_names = list(self.categories.keys())
        n_blocks = len(blocks)
        n_cats = len(cat_names)

        # Pre-compute text and tags for each block
        block_texts: list[str] = []
        block_tags: list[list[str]] = []
        for block in blocks:
            parts = []
            for field in ("Statement", "Description", "Tags", "Subject", "Rationale"):
                val = block.get(field, "")
                if val:
                    parts.append(str(val).lower())
            block_texts.append(" ".join(parts))
            tags_str = block.get("Tags", "")
            block_tags.append(
                [t.strip().lower() for t in tags_str.split(",") if t.strip()]
                if tags_str else []
            )

        # Build overlap matrices [N*C] flat
        kw_overlap = [0.0] * (n_blocks * n_cats)
        tag_match = [0.0] * (n_blocks * n_cats)
        ent_match = [0.0] * (n_blocks * n_cats)  # placeholder for entity overlap

        for bi in range(n_blocks):
            text = block_texts[bi]
            tags = block_tags[bi]
            for ci, cat in enumerate(cat_names):
                idx = bi * n_cats + ci
                keywords = self.categories[cat]
                kw_score = 0.0
                tag_score = 0.0
                for kw in keywords:
                    kw_lower = kw.lower()
                    if kw_lower in text:
                        kw_score += 1.0
                    if kw_lower in tags:
                        tag_score += 1.0
                kw_overlap[idx] = kw_score
                tag_match[idx] = tag_score

        # Call C kernels: affinity = kw_w * kw + tag_w * tag + ent_w * ent
        affinity = _mind_ffi.category_affinity_py(
            kw_overlap, tag_match, ent_match,
            n_blocks=n_blocks, n_cats=n_cats,
            kw_w=1.0, tag_w=3.0, ent_w=0.0,
        )

        # Assign categories with threshold
        threshold = float(self._SCORE_THRESHOLD)
        assignments = _mind_ffi.category_assign_py(
            affinity, threshold, n_blocks=n_blocks, n_cats=n_cats,
        )

        # Build category map from assignment matrix
        category_map: dict[str, list[dict]] = {}
        for bi in range(n_blocks):
            assigned_any = False
            for ci, cat in enumerate(cat_names):
                idx = bi * n_cats + ci
                if assignments[idx] > 0.5:  # sigmoid output > 0.5 means assigned
                    category_map.setdefault(cat, []).append(blocks[bi])
                    assigned_any = True
            if not assigned_any:
                category_map.setdefault("uncategorized", []).append(blocks[bi])

        log.info("batch_categorize_mind", n_blocks=n_blocks, n_cats=n_cats,
                 categories_found=len(category_map))
        return category_map

    def distill(self, workspace: str) -> list[str]:
        """Scan all blocks, cluster by category, write summary files.

        Returns list of written file paths (category markdowns + manifest).
        Uses MIND C kernels for batch scoring when available, falls back
        to pure Python per-block categorization.
        """
        blocks = self._scan_blocks(workspace)

        # Use MIND kernels for batch scoring when available
        if _mind_ffi is not None and len(blocks) > 0:
            try:
                category_map = self._batch_categorize_mind(blocks)
            except Exception as exc:
                log.warning("mind_batch_fallback", error=str(exc))
                category_map = {}
                for block in blocks:
                    cats = self.categorize_block(block)
                    for cat in cats:
                        category_map.setdefault(cat, []).append(block)
        else:
            # Pure Python fallback
            category_map = {}
            for block in blocks:
                cats = self.categorize_block(block)
                for cat in cats:
                    category_map.setdefault(cat, []).append(block)

        # Ensure output directory exists
        cat_dir = os.path.join(workspace, "categories")
        os.makedirs(cat_dir, exist_ok=True)

        # Write a Markdown file per category
        written: list[str] = []
        for category, cat_blocks in sorted(category_map.items()):
            path = self._write_category_file(category, cat_blocks, workspace)
            written.append(path)

        # Write JSON manifest
        manifest = {
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_blocks": len(blocks),
            "categories": {
                cat: len(blks) for cat, blks in sorted(category_map.items())
            },
        }
        manifest_path = os.path.join(cat_dir, "_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        written.append(manifest_path)

        log.info(
            "distill_complete",
            categories=len(category_map),
            files_written=len(written),
            total_blocks=len(blocks),
        )
        return written

    def _write_category_file(
        self, category: str, blocks: list[dict], workspace: str
    ) -> str:
        """Write ``categories/{category}.md`` with block summaries and source refs."""
        cat_dir = os.path.join(workspace, "categories")
        os.makedirs(cat_dir, exist_ok=True)
        path = os.path.join(cat_dir, f"{category}.md")

        now_str = datetime.now().strftime("%Y-%m-%d")
        lines: list[str] = [
            f"# Category: {category.title()}",
            "",
            f"*Auto-generated by mind-mem category distiller — {now_str}*",
            f"*{len(blocks)} blocks in this category*",
            "",
            "---",
            "",
        ]

        # Sort blocks newest-first by date, then by ID for stability
        def sort_key(b: dict) -> tuple[str, str]:
            date = b.get("Date", b.get("Created", "0000-00-00"))
            return (str(date), b.get("_id", ""))

        sorted_blocks = sorted(blocks, key=sort_key, reverse=True)

        for block in sorted_blocks:
            bid = block.get("_id", "unknown")
            statement = block.get(
                "Statement",
                block.get("Description", block.get("Subject", "---")),
            )
            status = block.get("Status", "---")
            source = block.get("_source_dir", "---")
            date = block.get("Date", block.get("Created", "---"))
            tags = block.get("Tags", "")

            # Truncate long statements for readability
            if isinstance(statement, str) and len(statement) > 200:
                statement = statement[:197] + "..."

            lines.append(f"### {bid}")
            lines.append(f"- **Statement:** {statement}")
            lines.append(f"- **Status:** {status}")
            lines.append(f"- **Date:** {date}")
            lines.append(f"- **Source:** {source}/")
            if tags:
                lines.append(f"- **Tags:** {tags}")
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return path

    # ------------------------------------------------------------------
    # Query-time helpers (anticipatory context)
    # ------------------------------------------------------------------

    def get_category_context(
        self, query: str, workspace: str, limit: int = 3
    ) -> str:
        """Return relevant category summaries for a query.

        Scores each known category by keyword overlap with the query,
        then reads and concatenates the top-N category files (up to
        2000 chars each).  Used for anticipatory context assembly.
        """
        cat_dir = os.path.join(workspace, "categories")
        if not os.path.isdir(cat_dir):
            return ""

        query_lower = query.lower()

        # Score each category by keyword overlap with the query
        scores: list[tuple[str, int]] = []
        for category, keywords in self.categories.items():
            score = sum(1 for kw in keywords if kw.lower() in query_lower)
            if score > 0:
                scores.append((category, score))

        # Descending by score, take top N
        scores.sort(key=lambda x: x[1], reverse=True)
        top_cats = [cat for cat, _ in scores[:limit]]

        # Read and concatenate the matching category files
        parts: list[str] = []
        for cat in top_cats:
            path = os.path.join(cat_dir, f"{cat}.md")
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                if len(content) > 2000:
                    content = content[:2000] + "\n\n*[truncated]*"
                parts.append(content)

        return "\n\n---\n\n".join(parts)

    def get_categories_for_query(self, query: str) -> list[str]:
        """Return category names relevant to a query (for prefetch routing).

        Categories are sorted by descending keyword overlap score.
        """
        query_lower = query.lower()
        scores: list[tuple[str, int]] = []
        for category, keywords in self.categories.items():
            score = sum(1 for kw in keywords if kw.lower() in query_lower)
            if score > 0:
                scores.append((category, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [cat for cat, _ in scores]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the category distiller from the command line.

    Usage:
        python scripts/category_distiller.py /path/to/workspace [--json] [--query "..."]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="mind-mem Category Distiller — deterministic thematic summaries"
    )
    parser.add_argument(
        "workspace",
        help="Path to the mind-mem workspace root",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output manifest as JSON to stdout after distilling",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Instead of distilling, return category context for this query",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Max categories to return for --query (default: 3)",
    )
    args = parser.parse_args()

    workspace = os.path.abspath(args.workspace)
    if not os.path.isdir(workspace):
        print(f"Error: workspace directory does not exist: {workspace}", file=sys.stderr)
        sys.exit(1)

    distiller = CategoryDistiller()

    if args.query:
        # Context-retrieval mode
        context = distiller.get_category_context(args.query, workspace, limit=args.limit)
        if context:
            print(context)
        else:
            print("(no matching categories)", file=sys.stderr)
        return

    # Default: full distillation
    written = distiller.distill(workspace)

    if args.json:
        manifest_path = os.path.join(workspace, "categories", "_manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                print(f.read())
    else:
        print(f"Distilled {len(written)} files:")
        for path in written:
            print(f"  {path}")


if __name__ == "__main__":
    main()
