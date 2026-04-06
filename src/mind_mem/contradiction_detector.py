#!/usr/bin/env python3
"""mind-mem Contradiction Detector — Surface conflicts at the governance gate.

At review time, runs proposal text against committed corpus using the existing
retrieval pipeline (BM25 + optional vector) to find semantically similar
memories. When high-similarity-but-conflicting entries exist, flags them
for the reviewer alongside the proposal.

Usage (library):
    from .contradiction_detector import detect_contradictions
    conflicts = detect_contradictions(workspace, proposal)
    # → List of {block_id, similarity, content_excerpt, conflict_type}

Usage (CLI):
    python3 -m mind_mem.contradiction_detector P-20260227-001 /path/to/workspace

Integrates with approve_apply flow via check_proposal_contradictions().

Related: #429 (corpus isolation), #432 (this feature)
"""

from __future__ import annotations

import json
import os
import re

from .block_parser import parse_file
from .observability import get_logger, metrics

_log = get_logger("contradiction_detector")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maximum characters for extracted text (guards against unbounded input).
_MAX_TEXT_LENGTH = 10_000

# Default similarity threshold for flagging potential conflicts.
# Lower = more flags (false positives cheap), higher = fewer (may miss real ones).
DEFAULT_SIMILARITY_THRESHOLD = 0.7

# Maximum number of similar blocks to retrieve for comparison.
DEFAULT_TOP_K = 10

# Files that contain committed corpus (not proposed/).
COMMITTED_CORPUS_FILES = [
    "decisions/DECISIONS.md",
    "tasks/TASKS.md",
    "entities/ENTITIES.md",
    "memory/MEMORY.md",
    "MEMORY.md",
    "AGENTS.md",
]

# Block statuses that indicate committed (active) knowledge.
COMMITTED_STATUSES = {"active", "in_progress", "monitoring", "completed"}


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def _extract_proposal_text(proposal: dict) -> str:
    """Extract searchable text from a proposal block.

    Combines title, description, evidence, and op patches into a single
    text for retrieval queries.
    """
    parts = []

    # Title / summary
    for field in ("Title", "Summary", "Description", "Rationale"):
        val = proposal.get(field)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())

    # Evidence (list or string)
    evidence = proposal.get("Evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    for e in evidence:
        if isinstance(e, str) and e.strip():
            parts.append(e.strip())

    # Op patches (the actual content being proposed)
    for op in proposal.get("Ops", []):
        patch = op.get("patch", "")
        if isinstance(patch, str) and patch.strip():
            parts.append(patch.strip())
        value = op.get("value", "")
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    return " ".join(parts)[:_MAX_TEXT_LENGTH]


def _extract_block_text(block: dict) -> str:
    """Extract searchable text from a committed block."""
    parts = []
    for field in ("Title", "Summary", "Description", "Rationale", "Decision", "Content", "Action", "Details"):
        val = block.get(field)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())

    # Include list fields
    for field in ("Evidence", "Constraints", "Implications"):
        val = block.get(field, [])
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())

    return " ".join(parts)[:_MAX_TEXT_LENGTH]


# ---------------------------------------------------------------------------
# Similarity computation (zero-dep fallback)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Simple word tokenization with lowering and dedup."""
    return set(re.findall(r"\b[a-z0-9]+(?:'[a-z]+)?\b", text.lower()))


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity between two text strings.

    Fast, zero-dependency fallback when vector search is unavailable.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """TF-based cosine similarity between two text strings.

    Better than Jaccard for longer texts — accounts for term frequency.
    Zero external dependencies.
    """
    import math
    from collections import Counter

    tokens_a = re.findall(r"\b[a-z0-9]+(?:'[a-z]+)?\b", text_a.lower())
    tokens_b = re.findall(r"\b[a-z0-9]+(?:'[a-z]+)?\b", text_b.lower())

    if not tokens_a or not tokens_b:
        return 0.0

    freq_a = Counter(tokens_a)
    freq_b = Counter(tokens_b)

    # All unique terms
    all_terms = set(freq_a.keys()) | set(freq_b.keys())

    # Dot product and magnitudes
    dot = sum(freq_a.get(t, 0) * freq_b.get(t, 0) for t in all_terms)
    mag_a = math.sqrt(sum(v * v for v in freq_a.values()))
    mag_b = math.sqrt(sum(v * v for v in freq_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Conflict classification
# ---------------------------------------------------------------------------


def _classify_conflict(proposal_text: str, existing_text: str, similarity: float) -> str:
    """Classify the type of relationship between a proposal and an existing block.

    Returns one of:
        "contradiction" — Statements appear to conflict
        "refinement"    — Proposal refines/extends existing knowledge
        "duplicate"     — Near-identical content
        "related"       — Similar topic but no clear conflict
    """
    # Very high similarity suggests duplicate or refinement
    if similarity > 0.9:
        return "duplicate"

    # Check for negation patterns that suggest contradiction
    negation_patterns = [
        r"\bnot\b",
        r"\bnever\b",
        r"\bdon't\b",
        r"\bdoesn't\b",
        r"\bshouldn't\b",
        r"\bwon't\b",
        r"\bcannot\b",
        r"\bcan't\b",
        r"\bdisable\b",
        r"\bremove\b",
        r"\bdelete\b",
        r"\bdeprecate\b",
        r"\brevert\b",
        r"\bundo\b",
        r"\broll\s*back\b",
        r"\binstead\b",
        r"\breplace\b",
    ]

    proposal_lower = proposal_text.lower()
    existing_lower = existing_text.lower()

    # Count negation signals in each
    proposal_negations = sum(1 for p in negation_patterns if re.search(p, proposal_lower))
    existing_negations = sum(1 for p in negation_patterns if re.search(p, existing_lower))

    # If one has significantly more negation language than the other,
    # and they're discussing the same topic (high similarity), likely a contradiction
    negation_diff = abs(proposal_negations - existing_negations)
    if negation_diff >= 3 and similarity > 0.5:
        return "contradiction"

    # Check for status change patterns (e.g., "active" → "deprecated")
    status_reversal_pairs = [
        ("active", "deprecated"),
        ("enable", "disable"),
        ("allow", "deny"),
        ("accept", "reject"),
        ("start", "stop"),
        ("add", "remove"),
        ("increase", "decrease"),
        ("true", "false"),
    ]

    for word_a, word_b in status_reversal_pairs:
        a_in_proposal = bool(re.search(r"\b" + re.escape(word_a) + r"\b", proposal_lower))
        b_in_existing = bool(re.search(r"\b" + re.escape(word_b) + r"\b", existing_lower))
        b_in_proposal = bool(re.search(r"\b" + re.escape(word_b) + r"\b", proposal_lower))
        a_in_existing = bool(re.search(r"\b" + re.escape(word_a) + r"\b", existing_lower))
        if (a_in_proposal and b_in_existing) or (b_in_proposal and a_in_existing):
            if similarity > 0.4:
                return "contradiction"

    # High similarity but no contradiction signals = refinement
    if similarity > 0.7:
        return "refinement"

    return "related"


# ---------------------------------------------------------------------------
# Core detection pipeline
# ---------------------------------------------------------------------------


def _load_committed_blocks(workspace: str) -> list[dict]:
    """Load all committed (non-proposed) blocks from the workspace."""
    ws = os.path.abspath(workspace)
    blocks = []

    for rel_path in COMMITTED_CORPUS_FILES:
        path = os.path.join(ws, rel_path)
        if not os.path.isfile(path):
            continue
        try:
            parsed = parse_file(path)
            for block in parsed:
                block["_source_file"] = rel_path
                blocks.append(block)
        except Exception as e:
            _log.warning("parse_error", file=rel_path, error=str(e))

    return blocks


def detect_contradictions(
    workspace: str,
    proposal: dict,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    use_bm25: bool = True,
) -> list[dict]:
    """Detect potential contradictions between a proposal and committed corpus.

    This is the main entry point. Runs the proposal text against all committed
    blocks using text similarity (BM25 when available, TF-IDF cosine fallback).

    Args:
        workspace: Workspace root path.
        proposal: Proposal block dict (from block_parser).
        threshold: Minimum similarity to flag (default 0.7).
        top_k: Maximum similar blocks to return.
        use_bm25: Try to use BM25 recall pipeline (falls back to cosine).

    Returns:
        List of conflict dicts, sorted by similarity (descending):
        [{
            "block_id": str,
            "source_file": str,
            "similarity": float,
            "conflict_type": str,  # "contradiction", "refinement", "duplicate", "related"
            "existing_excerpt": str,
            "proposal_excerpt": str,
        }]
    """
    ws = os.path.abspath(workspace)
    proposal_text = _extract_proposal_text(proposal)

    if not proposal_text.strip():
        _log.info("empty_proposal_text", proposal_id=proposal.get("ProposalId"))
        return []

    # Try BM25 recall first for better ranking
    bm25_results = []
    if use_bm25:
        bm25_results = _bm25_recall(ws, proposal_text, top_k * 3)

    # Load committed blocks for direct comparison
    committed_blocks = _load_committed_blocks(ws)

    # Build results: compare proposal against each candidate
    conflicts = []
    seen_ids = set()

    # If BM25 returned results, use those as candidates (already ranked by relevance)
    if bm25_results:
        for result in bm25_results[: top_k * 2]:
            block_id = result.get("_id", result.get("id", ""))
            if block_id in seen_ids:
                continue
            seen_ids.add(block_id)

            existing_text = result.get("excerpt", result.get("text", ""))
            if not existing_text:
                existing_text = _extract_block_text(result)

            similarity = _tfidf_cosine_similarity(proposal_text, existing_text)
            if similarity < threshold:
                continue

            conflict_type = _classify_conflict(proposal_text, existing_text, similarity)
            conflicts.append(
                {
                    "block_id": block_id,
                    "source_file": result.get("_source_file", result.get("file", "unknown")),
                    "similarity": round(similarity, 4),
                    "conflict_type": conflict_type,
                    "existing_excerpt": existing_text[:300],
                    "proposal_excerpt": proposal_text[:300],
                }
            )

    # Also scan all committed blocks directly (catches things BM25 might miss)
    for block in committed_blocks:
        block_id = block.get("_id", "")
        if not block_id or block_id in seen_ids:
            continue

        # Skip non-committed blocks
        status = block.get("Status", "active")
        if status and status.lower() not in COMMITTED_STATUSES and status.lower() != "":
            continue

        seen_ids.add(block_id)
        existing_text = _extract_block_text(block)
        if not existing_text.strip():
            continue

        similarity = _tfidf_cosine_similarity(proposal_text, existing_text)
        if similarity < threshold:
            continue

        conflict_type = _classify_conflict(proposal_text, existing_text, similarity)
        conflicts.append(
            {
                "block_id": block_id,
                "source_file": block.get("_source_file", "unknown"),
                "similarity": round(similarity, 4),
                "conflict_type": conflict_type,
                "existing_excerpt": existing_text[:300],
                "proposal_excerpt": proposal_text[:300],
            }
        )

    # Sort by similarity (highest first), limit to top_k
    conflicts.sort(key=lambda c: c["similarity"], reverse=True)
    conflicts = conflicts[:top_k]

    metrics.inc("contradiction_checks")
    metrics.inc("contradictions_found", sum(1 for c in conflicts if c["conflict_type"] == "contradiction"))

    _log.info(
        "contradiction_check",
        proposal_id=proposal.get("ProposalId", "?"),
        candidates_checked=len(seen_ids),
        conflicts_found=len(conflicts),
        contradictions=sum(1 for c in conflicts if c["conflict_type"] == "contradiction"),
    )

    return conflicts


def _bm25_recall(workspace: str, query: str, limit: int) -> list[dict]:
    """Try to use the BM25 recall pipeline. Returns empty list on failure."""
    try:
        from .recall import recall

        results = recall(
            workspace=workspace,
            query=query,
            limit=limit,
            active_only=False,
            graph_boost=False,
            include_pending=False,  # Exclude proposed — we only want committed
        )
        return results
    except Exception as e:
        _log.warning("bm25_recall_failed", error=str(e), hint="falling back to direct comparison")
        return []


# ---------------------------------------------------------------------------
# Integration with approve_apply
# ---------------------------------------------------------------------------


def check_proposal_contradictions(
    workspace: str,
    proposal: dict,
    threshold: float | None = None,
) -> dict:
    """Run contradiction check for a proposal at the governance gate.

    Returns a structured report suitable for inclusion in approve_apply output.

    Args:
        workspace: Workspace root path.
        proposal: Proposal block dict.
        threshold: Similarity threshold (reads from mind-mem.json if not provided).

    Returns:
        {
            "has_contradictions": bool,
            "has_conflicts": bool,  # any conflict type
            "contradiction_count": int,
            "conflicts": [...],  # full conflict list
            "summary": str,  # human-readable summary
        }
    """
    ws = os.path.abspath(workspace)

    # Read threshold from config if not provided
    if threshold is None:
        threshold = _get_config_threshold(ws)

    conflicts = detect_contradictions(ws, proposal, threshold=threshold)

    contradictions = [c for c in conflicts if c["conflict_type"] == "contradiction"]
    duplicates = [c for c in conflicts if c["conflict_type"] == "duplicate"]
    refinements = [c for c in conflicts if c["conflict_type"] == "refinement"]

    # Build human-readable summary
    summary_parts = []
    if contradictions:
        ids = ", ".join(c["block_id"] for c in contradictions)
        summary_parts.append(f"⚠️  {len(contradictions)} potential contradiction(s) with committed memory: {ids}")
    if duplicates:
        ids = ", ".join(c["block_id"] for c in duplicates)
        summary_parts.append(f"🔄 {len(duplicates)} near-duplicate(s) found: {ids}")
    if refinements:
        ids = ", ".join(c["block_id"] for c in refinements)
        summary_parts.append(f"📝 {len(refinements)} related block(s) that may be refined by this proposal: {ids}")
    if not conflicts:
        summary_parts.append("✅ No conflicts detected with committed corpus.")

    return {
        "has_contradictions": len(contradictions) > 0,
        "has_conflicts": len(conflicts) > 0,
        "contradiction_count": len(contradictions),
        "duplicate_count": len(duplicates),
        "refinement_count": len(refinements),
        "total_conflicts": len(conflicts),
        "conflicts": conflicts,
        "summary": "\n".join(summary_parts),
    }


def _get_config_threshold(workspace: str) -> float:
    """Read contradiction threshold from mind-mem.json config."""
    config_path = os.path.join(workspace, "mind-mem.json")
    try:
        with open(config_path) as f:
            config = json.load(f)
        threshold = float(config.get("contradiction", {}).get("threshold", DEFAULT_SIMILARITY_THRESHOLD))
        if not (0.0 <= threshold <= 1.0):
            return DEFAULT_SIMILARITY_THRESHOLD
        return threshold
    except (FileNotFoundError, OSError):
        return DEFAULT_SIMILARITY_THRESHOLD
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        _log.warning("config_threshold_parse_error", error=str(exc))
        return DEFAULT_SIMILARITY_THRESHOLD


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="mind-mem Contradiction Detector")
    parser.add_argument("proposal_id", help="Proposal ID to check (e.g., P-20260227-001)")
    parser.add_argument("workspace", nargs="?", default=".", help="Workspace path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Similarity threshold (default: {DEFAULT_SIMILARITY_THRESHOLD})",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    ws = os.path.abspath(args.workspace)

    from .apply_engine import find_proposal

    proposal, source_file = find_proposal(ws, args.proposal_id)
    if not proposal:
        print(f"Proposal {args.proposal_id} not found.")
        sys.exit(1)

    report = check_proposal_contradictions(ws, proposal, threshold=args.threshold)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"\n{'=' * 60}")
        print(f"Contradiction Check: {args.proposal_id}")
        print(f"{'=' * 60}")
        print(report["summary"])
        if report["conflicts"]:
            print(f"\nDetails ({len(report['conflicts'])} conflict(s)):")
            for c in report["conflicts"]:
                print(
                    f"  [{c['conflict_type'].upper()}] {c['block_id']} "
                    f"(similarity: {c['similarity']:.2f}, file: {c['source_file']})"
                )
                print(f"    Existing: {c['existing_excerpt'][:120]}...")
        print()
