#!/usr/bin/env python3
"""mind-mem Automated Conflict Resolution Pipeline. Zero external deps.

Goes beyond contradiction detection to graduated resolution:
1. Detection (already in intel_scan.py)
2. Strategy selection (timestamp, confidence, scope priority)
3. Proposal generation (supersede proposals for high-confidence resolutions)
4. Human veto loop (pending-review queue, never auto-applies without approval)

Resolution strategies:
- TIMESTAMP_PRIORITY: Newest decision wins (most recent intent)
- CONFIDENCE_PRIORITY: Highest ConstraintSignature priority wins
- SCOPE_PRIORITY: More specific scope wins over general
- MANUAL: Cannot auto-resolve, requires human review

Usage:
    from conflict_resolver import resolve_contradictions, generate_resolution_proposals
    proposals = resolve_contradictions(workspace)
    # → List of resolution proposals with strategy, confidence, and supersede actions
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_parser import get_by_id, parse_file
from mind_filelock import FileLock
from observability import get_logger, metrics

_log = get_logger("conflict_resolver")


# ---------------------------------------------------------------------------
# Resolution strategies
# ---------------------------------------------------------------------------

class ResolutionStrategy:
    TIMESTAMP = "timestamp_priority"
    CONFIDENCE = "confidence_priority"
    SCOPE = "scope_priority"
    MANUAL = "manual_review"


def _extract_date(block: dict) -> str | None:
    """Extract date from block, trying multiple fields."""
    for field in ("Date", "Created", "Timestamp"):
        val = block.get(field, "")
        if isinstance(val, str) and re.match(r"\d{4}-\d{2}-\d{2}", val):
            return val[:10]
    # Try extracting from block ID (D-YYYYMMDD-NNN)
    bid = block.get("_id", "")
    m = re.match(r"[A-Z]+-(\d{8})-\d{3}", bid)
    if m:
        raw = m.group(1)
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return None


def _get_cs_priority(block: dict) -> int:
    """Get highest ConstraintSignature priority from a block."""
    best = 0
    for sig in block.get("ConstraintSignatures", []):
        p = sig.get("priority", 1)
        if isinstance(p, int) and p > best:
            best = p
    return best


def _get_scope_specificity(block: dict) -> int:
    """Score how specific a block's scope is (higher = more specific)."""
    specificity = 0
    for sig in block.get("ConstraintSignatures", []):
        scope = sig.get("scope", {})
        if isinstance(scope, dict):
            for key, val in scope.items():
                if isinstance(val, list):
                    specificity += len(val)
                elif isinstance(val, str) and val:
                    specificity += 1
                elif isinstance(val, dict):
                    specificity += sum(1 for v in val.values() if v)
    return specificity


def _block_hash(block: dict) -> str:
    """Compute a stable hash of a block's content fields."""
    content = json.dumps(
        {k: v for k, v in sorted(block.items()) if not k.startswith("_")},
        default=str, sort_keys=True
    )
    return hashlib.sha256(content.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Core resolution logic
# ---------------------------------------------------------------------------

def analyze_contradiction(block_a: dict, block_b: dict) -> dict:
    """Analyze a contradiction pair and recommend a resolution strategy.

    Returns:
        Dict with: strategy, confidence, winner_id, loser_id, rationale
    """
    date_a = _extract_date(block_a)
    date_b = _extract_date(block_b)
    prio_a = _get_cs_priority(block_a)
    prio_b = _get_cs_priority(block_b)
    scope_a = _get_scope_specificity(block_a)
    scope_b = _get_scope_specificity(block_b)

    id_a = block_a.get("_id", "?")
    id_b = block_b.get("_id", "?")

    # Strategy 1: Confidence/priority wins (clear difference)
    if prio_a != prio_b and abs(prio_a - prio_b) >= 2:
        winner = id_a if prio_a > prio_b else id_b
        loser = id_b if winner == id_a else id_a
        return {
            "strategy": ResolutionStrategy.CONFIDENCE,
            "confidence": "high",
            "winner_id": winner,
            "loser_id": loser,
            "rationale": f"ConstraintSignature priority: {max(prio_a, prio_b)} vs {min(prio_a, prio_b)} (delta >= 2)",
        }

    # Strategy 2: Scope specificity wins (one is clearly more targeted)
    if scope_a != scope_b and abs(scope_a - scope_b) >= 2:
        winner = id_a if scope_a > scope_b else id_b
        loser = id_b if winner == id_a else id_a
        return {
            "strategy": ResolutionStrategy.SCOPE,
            "confidence": "medium",
            "winner_id": winner,
            "loser_id": loser,
            "rationale": f"Scope specificity: {max(scope_a, scope_b)} vs {min(scope_a, scope_b)} fields",
        }

    # Strategy 3: Timestamp wins (newer = more recent intent)
    if date_a and date_b and date_a != date_b:
        winner = id_a if date_a > date_b else id_b
        loser = id_b if winner == id_a else id_a
        newer = max(date_a, date_b)
        older = min(date_a, date_b)
        return {
            "strategy": ResolutionStrategy.TIMESTAMP,
            "confidence": "medium",
            "winner_id": winner,
            "loser_id": loser,
            "rationale": f"Newer decision ({newer}) supersedes older ({older})",
        }

    # Fallback: cannot auto-resolve
    return {
        "strategy": ResolutionStrategy.MANUAL,
        "confidence": "low",
        "winner_id": None,
        "loser_id": None,
        "rationale": "Cannot auto-resolve: same date, similar priority and scope. Requires human review.",
    }


def resolve_contradictions(workspace: str) -> list[dict]:
    """Analyze all detected contradictions and produce resolution recommendations.

    Reads CONTRADICTIONS.md (produced by intel_scan.py) and cross-references
    with the actual blocks to determine the best resolution strategy.

    Returns:
        List of resolution dicts with strategy, confidence, block IDs, and rationale.
    """
    ws = os.path.abspath(workspace)
    contradictions_path = os.path.join(ws, "intelligence", "CONTRADICTIONS.md")
    decisions_path = os.path.join(ws, "decisions", "DECISIONS.md")

    if not os.path.isfile(contradictions_path) or not os.path.isfile(decisions_path):
        return []

    # Parse contradiction entries
    contra_blocks = parse_file(contradictions_path)
    decision_blocks = parse_file(decisions_path)

    resolutions = []
    _ID_RE = re.compile(r"[A-Z]+-\d{8}-\d{3}")

    for contra in contra_blocks:
        # Extract the two conflicting block IDs from the contradiction entry
        text = " ".join(
            str(v) for v in contra.values() if isinstance(v, str) and not v.startswith("_")
        )
        ids = _ID_RE.findall(text)
        if len(ids) < 2:
            continue

        # Find the actual blocks
        block_a = get_by_id(decision_blocks, ids[0])
        block_b = get_by_id(decision_blocks, ids[1])

        if not block_a or not block_b:
            continue

        resolution = analyze_contradiction(block_a, block_b)
        resolution["contradiction_id"] = contra.get("_id", "?")
        resolution["block_a"] = ids[0]
        resolution["block_b"] = ids[1]
        resolution["hash_a"] = _block_hash(block_a)
        resolution["hash_b"] = _block_hash(block_b)
        resolutions.append(resolution)

    _log.info("contradictions_analyzed", count=len(resolutions),
              auto_resolvable=sum(1 for r in resolutions if r["strategy"] != ResolutionStrategy.MANUAL))
    metrics.inc("contradictions_analyzed", len(resolutions))
    return resolutions


def generate_resolution_proposals(workspace: str, resolutions: list[dict] | None = None) -> int:
    """Generate supersede proposals for auto-resolvable contradictions.

    Writes proposals to intelligence/proposed/RESOLUTIONS_PROPOSED.md
    for human review before application.

    Returns:
        Number of proposals generated.
    """
    ws = os.path.abspath(workspace)

    if resolutions is None:
        resolutions = resolve_contradictions(ws)

    # Only generate proposals for auto-resolvable contradictions
    auto = [r for r in resolutions if r["strategy"] != ResolutionStrategy.MANUAL and r["winner_id"]]
    if not auto:
        return 0

    proposed_dir = os.path.join(ws, "intelligence", "proposed")
    os.makedirs(proposed_dir, exist_ok=True)
    proposed_path = os.path.join(proposed_dir, "RESOLUTIONS_PROPOSED.md")

    now = datetime.now()
    ts = now.strftime("%Y-%m-%dT%H:%M:%S")
    date_compact = now.strftime("%Y%m%d")

    with FileLock(proposed_path):
        with open(proposed_path, "a", encoding="utf-8") as f:
            for i, res in enumerate(auto, 1):
                proposal_id = f"R-{date_compact}-{i:03d}"
                f.write(f"\n[{proposal_id}]\n")
                f.write(f"Date: {ts}\n")
                f.write("Type: auto-resolution\n")
                f.write(f"Strategy: {res['strategy']}\n")
                f.write(f"Confidence: {res['confidence']}\n")
                f.write(f"Contradiction: {res['contradiction_id']}\n")
                f.write(f"Winner: {res['winner_id']} (hash: {res['hash_a']})\n")
                f.write(f"Loser: {res['loser_id']} (hash: {res['hash_b']})\n")
                f.write(f"Action: Supersede {res['loser_id']} with SupersededBy: {res['winner_id']}\n")
                f.write(f"Rationale: {res['rationale']}\n")
                f.write("Status: pending-review\n")
                f.write("\n---\n")

    _log.info("resolution_proposals_generated", count=len(auto))
    metrics.inc("resolution_proposals", len(auto))
    return len(auto)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="mind-mem Conflict Resolution Pipeline")
    parser.add_argument("workspace", nargs="?", default=".")
    parser.add_argument("--analyze", action="store_true", help="Analyze contradictions and show resolutions")
    parser.add_argument("--propose", action="store_true", help="Generate resolution proposals")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    ws = os.path.abspath(args.workspace)

    if args.analyze or not args.propose:
        resolutions = resolve_contradictions(ws)
        if not resolutions:
            print("No contradictions found to resolve.")
            return

        if args.json:
            print(json.dumps(resolutions, indent=2, default=str))
        else:
            print(f"Found {len(resolutions)} contradiction(s):\n")
            for r in resolutions:
                status = "AUTO-RESOLVABLE" if r["strategy"] != ResolutionStrategy.MANUAL else "NEEDS HUMAN REVIEW"
                print(f"  [{r['contradiction_id']}] {r['block_a']} vs {r['block_b']}")
                print(f"    Strategy: {r['strategy']}")
                print(f"    Confidence: {r['confidence']}")
                print(f"    Status: {status}")
                if r.get("winner_id"):
                    print(f"    Winner: {r['winner_id']}")
                print(f"    Rationale: {r['rationale']}")
                print()

        if args.propose:
            count = generate_resolution_proposals(ws, resolutions)
            print(f"\nGenerated {count} resolution proposal(s) → intelligence/proposed/RESOLUTIONS_PROPOSED.md")

    elif args.propose:
        count = generate_resolution_proposals(ws)
        print(f"Generated {count} resolution proposal(s)")


if __name__ == "__main__":
    main()
