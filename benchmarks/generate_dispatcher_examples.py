"""Generate synthetic training examples for the v3.2.x 7-dispatcher MCP surface.

mind-mem-4b v2 needs to natively emit calls like
``recall(mode="similar")`` and ``staged_change(phase="propose")``
instead of the 19 legacy tool names. This script produces N synthetic
(prompt, expected-call) pairs covering every dispatcher × mode
combination.

Usage:
    python3 benchmarks/generate_dispatcher_examples.py \\
        --output /path/to/out.jsonl --count 10000

Each line is JSONL with schema::

    {
      "prompt": "<user message>",
      "expected_call": {
        "tool": "recall",
        "args": {"mode": "similar", "query": "...", "limit": 10}
      }
    }

Pure synthetic data — no workspace scrape, no LLM calls. Safe to run
in any environment. Output is deterministic given ``--seed``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable, Iterator

# ---------------------------------------------------------------------------
# 7 consolidated dispatchers from v3.2.x
# ---------------------------------------------------------------------------


_RECALL_MODES = ("similar", "verify", "intent", "diagnose", "prefetch", "bundle")
_STAGED_PHASES = ("propose", "approve", "rollback", "list")
_MEMORY_VERIFY_CHECKS = ("chain", "replay", "contradictions", "drift")
_GRAPH_ACTIONS = ("traverse", "neighbors", "shortest_path", "subgraph")
_CORE_ACTIONS = ("index_stats", "reindex", "scan", "export")
_KERNEL_ACTIONS = ("list", "get", "invoke")
_COMPILED_TRUTH_ACTIONS = ("query", "refresh")


# Natural-language prompt templates per dispatcher × mode. Each template
# is a callable(random_state) -> str so callers can sample with their
# own RNG. Real diversity comes from the topic + entity pools below.
_TOPIC_POOL = [
    "PostgreSQL migration",
    "API rate limiting",
    "OAuth token rotation",
    "Redis caching layer",
    "CI/CD pipeline failure",
    "database backup strategy",
    "vector search indexing",
    "blue-green deployment",
    "contract renewal with AcmeCorp",
    "runtime memory leak",
    "auth middleware rewrite",
    "observability dashboard",
    "storage backend migration",
    "rate-limit incident on 2026-03-04",
    "DR runbook update",
]
_PERSON_POOL = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace"]
_BLOCK_ID_PREFIXES = ("D-", "T-", "PER-", "PRJ-", "INC-")


def _pick(rng: random.Random, pool: list[str]) -> str:
    return rng.choice(pool)


def _rand_block_id(rng: random.Random) -> str:
    prefix = rng.choice(_BLOCK_ID_PREFIXES)
    if prefix in ("D-", "T-"):
        return f"{prefix}202604{rng.randint(1, 28):02d}-{rng.randint(1, 999):03d}"
    return f"{prefix}{rng.randint(1, 999):03d}"


# ---------------------------------------------------------------------------
# Per-dispatcher generators
# ---------------------------------------------------------------------------


def _gen_recall(rng: random.Random) -> dict:
    mode = rng.choice(_RECALL_MODES)
    topic = _pick(rng, _TOPIC_POOL)
    prompts = {
        "similar": [
            f"What do we know about {topic}?",
            f"Find all memory about {topic}.",
            f"Search for {topic} context.",
        ],
        "verify": [
            f"Is it true we decided to use {topic}?",
            f"Did we already agree on {topic}?",
        ],
        "intent": [
            f"Classify this query: '{topic}'",
            f"What kind of question is 'Did {topic} fail?'",
        ],
        "diagnose": [
            f"Why is recall returning poor results for '{topic}'?",
            f"Diagnose the retrieval trace for '{topic}'",
        ],
        "prefetch": [
            f"Warm the cache for {topic} queries.",
            f"Pre-fetch memory for tomorrow's {topic} review.",
        ],
        "bundle": [
            f"Give me the structured evidence bundle for '{topic}'.",
            f"Return facts + timeline for {topic}, not raw blocks.",
        ],
    }
    return {
        "prompt": _pick(rng, prompts[mode]),
        "expected_call": {
            "tool": "recall",
            "args": {"mode": mode, "query": topic, "limit": rng.choice([5, 10, 18])},
        },
    }


def _gen_staged_change(rng: random.Random) -> dict:
    phase = rng.choice(_STAGED_PHASES)
    topic = _pick(rng, _TOPIC_POOL)
    prompts = {
        "propose": [
            f"Propose a new decision: switch to {topic}.",
            f"Stage a change for {topic} — we need to try a different approach.",
        ],
        "approve": [
            f"Approve the pending proposal for {topic}.",
            f"Apply the staged change about {topic}.",
        ],
        "rollback": [
            f"Roll back the {topic} change — it broke staging.",
            f"Undo yesterday's {topic} proposal.",
        ],
        "list": [
            "Show me all pending proposals.",
            "What's staged for review?",
        ],
    }
    args: dict = {"phase": phase}
    if phase == "propose":
        args["statement"] = f"Use {topic} as the default."
        args["block_type"] = "decision"
    elif phase in ("approve", "rollback"):
        args["proposal_id"] = _rand_block_id(rng).replace("D-", "P-")
    return {"prompt": _pick(rng, prompts[phase]), "expected_call": {"tool": "staged_change", "args": args}}


def _gen_memory_verify(rng: random.Random) -> dict:
    check = rng.choice(_MEMORY_VERIFY_CHECKS)
    prompts = {
        "chain": ["Verify the audit chain integrity.", "Is the hash chain intact?"],
        "replay": ["Replay the audit log.", "Reconstruct the audit history."],
        "contradictions": ["List any contradictions in memory.", "Find conflicting decisions."],
        "drift": ["Is there semantic drift?", "Detect any memory drift."],
    }
    return {
        "prompt": _pick(rng, prompts[check]),
        "expected_call": {"tool": "memory_verify", "args": {"check": check}},
    }


def _gen_graph(rng: random.Random) -> dict:
    action = rng.choice(_GRAPH_ACTIONS)
    bid = _rand_block_id(rng)
    prompts = {
        "traverse": [f"Walk the graph from {bid}.", f"Show the causal chain from {bid}."],
        "neighbors": [f"What blocks reference {bid}?", f"List neighbours of {bid}."],
        "shortest_path": [f"How does {bid} connect to D-20260420-099?", "Find shortest path."],
        "subgraph": [f"Extract the subgraph around {bid}.", "Give me the related cluster."],
    }
    args: dict = {"action": action, "block_id": bid}
    if action == "shortest_path":
        args["target_id"] = _rand_block_id(rng)
    return {"prompt": _pick(rng, prompts[action]), "expected_call": {"tool": "graph", "args": args}}


def _gen_core(rng: random.Random) -> dict:
    action = rng.choice(_CORE_ACTIONS)
    prompts = {
        "index_stats": ["Show index stats.", "How big is the memory?"],
        "reindex": ["Rebuild the index.", "Reindex everything."],
        "scan": ["Run a workspace scan.", "Full integrity scan."],
        "export": ["Export the memory.", "Dump all decisions."],
    }
    return {"prompt": _pick(rng, prompts[action]), "expected_call": {"tool": "core", "args": {"action": action}}}


def _gen_kernels(rng: random.Random) -> dict:
    action = rng.choice(_KERNEL_ACTIONS)
    kernel_name = rng.choice(["bm25f", "rrf_fusion", "temporal_decay", "cross_encoder"])
    args: dict = {"action": action}
    if action == "list":
        prompt = "List all MIND kernels."
    elif action == "get":
        prompt = f"Show me the {kernel_name} kernel."
        args["name"] = kernel_name
    else:
        prompt = f"Invoke {kernel_name} with test input."
        args["name"] = kernel_name
        args["input"] = {"test": True}
    return {"prompt": prompt, "expected_call": {"tool": "kernels", "args": args}}


def _gen_compiled_truth(rng: random.Random) -> dict:
    action = rng.choice(_COMPILED_TRUTH_ACTIONS)
    topic = _pick(rng, _TOPIC_POOL)
    prompts = {
        "query": [f"What's the compiled truth about {topic}?", f"Give me the verified answer for {topic}."],
        "refresh": [f"Refresh the compiled truth for {topic}.", "Rebuild compiled truths."],
    }
    args: dict = {"action": action}
    if action == "query":
        args["topic"] = topic
    return {"prompt": _pick(rng, prompts[action]), "expected_call": {"tool": "compiled_truth", "args": args}}


_DISPATCHERS: tuple[Callable[[random.Random], dict], ...] = (
    _gen_recall,
    _gen_staged_change,
    _gen_memory_verify,
    _gen_graph,
    _gen_core,
    _gen_kernels,
    _gen_compiled_truth,
)


def generate(count: int, seed: int = 42) -> Iterator[dict]:
    rng = random.Random(seed)
    for _ in range(count):
        gen = rng.choice(_DISPATCHERS)
        yield gen(rng)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for example in generate(args.count, seed=args.seed):
            f.write(json.dumps(example, default=str) + "\n")

    print(f"Wrote {args.count} examples to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
