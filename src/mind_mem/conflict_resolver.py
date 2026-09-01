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
- CONSENSUS: A multi-agent quorum picked the winner (opt-in, default OFF —
  ``governance.consensus.enabled`` in ``mind-mem.json``). Only ever consulted
  where MANUAL would otherwise be the verdict, and its winner becomes a
  pending-review PROPOSAL, never an apply.

Usage:
    from .conflict_resolver import resolve_contradictions, generate_resolution_proposals
    proposals = resolve_contradictions(workspace)
    # → List of resolution proposals with strategy, confidence, and supersede actions
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime

from .admissibility import admit_corpus
from .block_parser import get_by_id, parse_file
from .consensus_vote import Vote, reach_consensus, resolve_consensus_config
from .mind_filelock import FileLock
from .observability import get_logger, metrics

_log = get_logger("conflict_resolver")


# ---------------------------------------------------------------------------
# Resolution strategies
# ---------------------------------------------------------------------------


class ResolutionStrategy:
    TIMESTAMP = "timestamp_priority"
    CONFIDENCE = "confidence_priority"
    SCOPE = "scope_priority"
    MANUAL = "manual_review"
    #: Opt-in (``governance.consensus.enabled``). Reached only from the MANUAL
    #: fallback, so it can never override a deterministic strategy above it.
    CONSENSUS = "consensus_quorum"


#: Workspace-relative corpus of agent votes on contradictions. Optional: a
#: workspace without it simply has no votes, which is the single-operator
#: case and resolves exactly as it did before consensus existed.
VOTES_FILE = os.path.join("intelligence", "VOTES.md")


def _load_workspace_config(ws: str) -> dict:
    """Read ``mind-mem.json`` from *ws*, silently, returning ``{}`` on any problem.

    Deliberately does NOT reuse ``init_workspace.load_config`` /
    ``cron_runner.load_config``: both of those log a warning when the file is
    absent or unparseable, and this read happens on the DEFAULT-OFF path of a
    flag. A flag probe that emits a log line is observable, which would make
    flag-off behaviour differ from before the flag existed. The parsing and
    validation of what comes back is not duplicated — that is
    :func:`consensus_vote.resolve_consensus_config`, the tested incumbent.
    """
    try:
        with open(os.path.join(ws, "mind-mem.json"), "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _load_consensus_votes(ws: str) -> dict[str, list[Vote]]:
    """Parse ``intelligence/VOTES.md`` into ``{contradiction_id: [Vote, ...]}``.

    **Admission-filtered.** The vote blocks go through
    :func:`~mind_mem.admissibility.admit_corpus` before any of them is turned
    into a :class:`~mind_mem.consensus_vote.Vote`. A vote is a block like any
    other, so a block minted by a withheld ingest tier (``quarantined``,
    ``pending``) — or carrying a status nobody has named — must not be able to
    push a contradiction over quorum and stage a supersede proposal off the
    back of it. Reading the ``Status`` field and *selecting* on it is not the
    same as filtering: ``admit_corpus`` is the one predicate that is
    fail-closed on statuses it has never heard of.

    Only reached when the consensus flag is ON; the flag-off path never opens
    this file.
    """
    path = os.path.join(ws, VOTES_FILE)
    if not os.path.isfile(path):
        return {}
    try:
        blocks = parse_file(path)
    except (OSError, ValueError) as exc:
        _log.warning("consensus_votes_unreadable", path=path, error=str(exc))
        return {}

    votes: dict[str, list[Vote]] = {}
    for block in admit_corpus(blocks):
        contradiction = _vote_field(block, "Contradiction")
        agent = _vote_field(block, "Agent")
        choice = _vote_field(block, "Choice")
        if not (contradiction and agent and choice):
            continue
        votes.setdefault(contradiction, []).append(
            Vote(
                agent_id=agent,
                choice=choice,
                # Left UNSPECIFIED on purpose: the weight comes from
                # ``namespaces.<agent>.trust_weight``, so a vote file cannot
                # award itself trust the operator did not configure.
                trust_weight=None,
                rationale=_vote_field(block, "Rationale") or None,
            )
        )
    return votes


def _vote_field(block: dict, key: str) -> str:
    """A single trimmed string field of a vote block, or ``""``.

    The block parser renders a bare ``Key:`` as an empty list and repeated
    keys as lists, so anything that is not a populated string is treated as
    absent rather than coerced.
    """
    value = block.get(key)
    return value.strip() if isinstance(value, str) else ""


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
    content = json.dumps({k: v for k, v in sorted(block.items()) if not k.startswith("_")}, default=str, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Core resolution logic
# ---------------------------------------------------------------------------


def analyze_contradiction(
    block_a: dict,
    block_b: dict,
    *,
    votes: list[Vote] | None = None,
    consensus: dict | None = None,
    namespace_config: dict | None = None,
) -> dict:
    """Analyze a contradiction pair and recommend a resolution strategy.

    Pure: no clock, no filesystem, no configuration read. Everything the
    consensus leg needs is injected by :func:`resolve_contradictions`.

    Args:
        votes: Agent votes for THIS contradiction. ``None`` (the default) is
            the flag-off shape and is byte-identical to the pre-consensus
            function.
        consensus: Resolved ``governance.consensus`` settings from
            :func:`~mind_mem.consensus_vote.resolve_consensus_config`. The
            quorum is consulted only when this says ``enabled``.
        namespace_config: ``namespaces`` map, for per-agent ``trust_weight``.

    Returns:
        Dict with: strategy, confidence, winner_id, loser_id, rationale
        (plus ``consensus`` detail when the quorum decided it).
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
    manual = {
        "strategy": ResolutionStrategy.MANUAL,
        "confidence": "low",
        "winner_id": None,
        "loser_id": None,
        "rationale": "Cannot auto-resolve: same date, similar priority and scope. Requires human review.",
    }

    # Opt-in quorum leg. Nothing above this point can reach it, so an
    # auto-resolvable contradiction is never overridden by a vote, and with
    # the flag off (or no votes) the function returns exactly what it always
    # did — no logging, no probe, no observable difference.
    if not consensus or not consensus.get("enabled") or not votes:
        return manual

    threshold = float(consensus.get("quorum_threshold", 0.66))
    decision = reach_consensus(
        votes,
        quorum_threshold=threshold,
        min_votes=int(consensus.get("min_votes", 2)),
        namespace_config=namespace_config,
    )
    # The winner must be one of the two blocks actually in contradiction.
    # Without this, a vote file naming any id at all could put that id on the
    # "Winner:" line of a supersede proposal for a pair it has nothing to do
    # with. A vote for a third id is not filtered out before the tally either
    # — it dilutes the quorum, which is the fail-closed direction.
    if decision.reason != "quorum" or decision.winner not in (id_a, id_b):
        return manual

    winner = decision.winner
    loser = id_b if winner == id_a else id_a
    return {
        "strategy": ResolutionStrategy.CONSENSUS,
        # Never "high": a quorum is agreement, not evidence, and this outcome
        # is staged for review like every other non-MANUAL strategy.
        "confidence": "medium",
        "winner_id": winner,
        "loser_id": loser,
        "rationale": (
            f"Quorum consensus of {len({v.agent_id for v in votes})} agent(s): "
            f"weighted margin {decision.margin} >= threshold {threshold} "
            f"(confidence {decision.confidence}). Staged for review, never auto-applied."
        ),
        "consensus": {
            "margin": decision.margin,
            "confidence": decision.confidence,
            "threshold": threshold,
            "agents": sorted({v.agent_id for v in votes}),
            "vote_counts": dict(decision.vote_counts),
        },
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

    # Opt-in consensus leg (``governance.consensus.enabled``, default OFF).
    # Flag off: no votes file is opened, no namespace map is resolved, and
    # analyze_contradiction is called with exactly the arguments it took
    # before this leg existed.
    workspace_config = _load_workspace_config(ws)
    consensus_cfg = resolve_consensus_config(workspace_config)
    consensus_on = bool(consensus_cfg["enabled"])
    votes_by_contradiction: dict[str, list[Vote]] = {}
    namespace_config: dict | None = None
    if consensus_on:
        votes_by_contradiction = _load_consensus_votes(ws)
        namespaces = workspace_config.get("namespaces")
        namespace_config = namespaces if isinstance(namespaces, dict) else None

    resolutions = []
    _ID_RE = re.compile(r"[A-Z]+-\d{8}-\d{3}")

    for contra in contra_blocks:
        # Extract the two conflicting block IDs from the contradiction entry
        text = " ".join(str(v) for v in contra.values() if isinstance(v, str) and not v.startswith("_"))
        ids = _ID_RE.findall(text)
        if len(ids) < 2:
            continue

        # Find the actual blocks
        block_a = get_by_id(decision_blocks, ids[0])
        block_b = get_by_id(decision_blocks, ids[1])

        if not block_a or not block_b:
            continue

        contra_id = contra.get("_id", "?")
        resolution = analyze_contradiction(
            block_a,
            block_b,
            votes=votes_by_contradiction.get(str(contra_id)) if consensus_on else None,
            consensus=consensus_cfg if consensus_on else None,
            namespace_config=namespace_config,
        )
        resolution["contradiction_id"] = contra_id
        resolution["block_a"] = ids[0]
        resolution["block_b"] = ids[1]
        resolution["hash_a"] = _block_hash(block_a)
        resolution["hash_b"] = _block_hash(block_b)
        resolutions.append(resolution)

    _log.info(
        "contradictions_analyzed",
        count=len(resolutions),
        auto_resolvable=sum(1 for r in resolutions if r["strategy"] != ResolutionStrategy.MANUAL),
    )
    metrics.inc("contradictions_analyzed", len(resolutions))
    if consensus_on:
        by_quorum = sum(1 for r in resolutions if r["strategy"] == ResolutionStrategy.CONSENSUS)
        if by_quorum:
            _log.info("contradictions_resolved_by_quorum", count=by_quorum)
            metrics.inc("consensus_resolutions", by_quorum)
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
        # Start the counter at max(existing IDs for today) + 1 so
        # repeated calls on the same day do not collide with earlier
        # batches. New file → regex finds zero matches → counter starts at 1.
        start_idx = 1
        if os.path.isfile(proposed_path):
            try:
                with open(proposed_path, "r", encoding="utf-8") as fh:
                    existing = fh.read()
            except OSError as exc:
                # The file exists but cannot be read, so the highest id
                # already in it is unknown. Falling back to 1 and appending
                # would re-mint R-<date>-001.. over live ids and break
                # everything keyed on proposal_id (apply, rollback, the
                # audit trail). Refuse loudly instead of corrupting it.
                _log.error("resolution_proposals_id_scan_failed", path=proposed_path, error=str(exc))
                raise OSError(
                    f"cannot read {proposed_path} to determine the next proposal id; "
                    "refusing to append proposals that would duplicate existing ids"
                ) from exc
            pat = re.compile(rf"R-{re.escape(date_compact)}-(\d{{3,}})")
            nums = [int(m.group(1)) for m in pat.finditer(existing)]
            if nums:
                start_idx = max(nums) + 1
        with open(proposed_path, "a", encoding="utf-8") as f:
            for i, res in enumerate(auto, start_idx):
                proposal_id = f"R-{date_compact}-{i:03d}"
                # hash_a/hash_b are keyed by POSITION (block_a==ids[0],
                # block_b==ids[1]). The winner can be either side, so map the
                # hashes to the winner/loser by identity — otherwise when the
                # winner is block_b the audit trail prints the loser's hash
                # next to the winner (and vice versa), defeating the
                # tamper-evidence fingerprint check. When no positional anchor
                # is present (hand-built resolution dicts), fall back to the
                # hash_a==winner / hash_b==loser convention.
                block_a = res.get("block_a")
                if block_a is not None:
                    winner_is_a = res["winner_id"] == block_a
                    winner_hash = res["hash_a"] if winner_is_a else res["hash_b"]
                    loser_hash = res["hash_b"] if winner_is_a else res["hash_a"]
                else:
                    winner_hash = res.get("hash_a", "")
                    loser_hash = res.get("hash_b", "")
                f.write(f"\n[{proposal_id}]\n")
                f.write(f"Date: {ts}\n")
                f.write("Type: auto-resolution\n")
                f.write(f"Strategy: {res['strategy']}\n")
                f.write(f"Confidence: {res['confidence']}\n")
                f.write(f"Contradiction: {res['contradiction_id']}\n")
                f.write(f"Winner: {res['winner_id']} (hash: {winner_hash})\n")
                f.write(f"Loser: {res['loser_id']} (hash: {loser_hash})\n")
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
