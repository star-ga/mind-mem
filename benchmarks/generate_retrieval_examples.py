"""Generate training examples for v3.3.0 retrieval shapes.

Produces JSONL examples that teach mind-mem-4b v2:

1. Query decomposition — compound question → list of sub-queries
2. Query reformulation — question → N paraphrases
3. Entity extraction — prompt → {people, projects, tools, incidents}
4. LLM-as-reranker — (query, candidates) → relevance scores

Pure synthetic — no LLM / workspace. Deterministic via --seed.

Usage:
    python3 benchmarks/generate_retrieval_examples.py \\
        --output /path/to/out.jsonl --count 5000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable, Iterator

_TOPICS = [
    "PostgreSQL",
    "Redis",
    "Kubernetes",
    "OAuth2",
    "GraphQL",
    "vector search",
    "rate limiting",
    "feature flags",
    "secret rotation",
    "audit chain",
]
_VERBS = ["decide", "choose", "migrate", "remove", "adopt", "replace", "evaluate", "deprecate"]
_PEOPLE = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]
_PROJECTS = ["Atlas", "Bifrost", "Citadel", "Delphi", "Eclipse"]
_TOOLS = ["Grafana", "Prometheus", "Jaeger", "Vault", "Terraform"]


def _pick(rng: random.Random, pool: list[str]) -> str:
    return rng.choice(pool)


# ---------------------------------------------------------------------------
# Task 1 — Query decomposition
# ---------------------------------------------------------------------------


_DECOMP_TEMPLATES = [
    # Temporal compound
    ("{a_verb} {a_topic} after we {b_verb} {b_topic}?", "temporal"),
    ("What happened to {a_topic} before we rolled out {b_topic}?", "temporal"),
    # Causal compound
    ("Why did we {a_verb} {a_topic} because {b_topic} failed?", "causal"),
    # Comparative
    ("{a_topic} vs {b_topic} for our scale?", "comparison"),
    # Conjunction
    ("{a_person} and {b_person}'s stance on {a_topic}?", "conjunction"),
    # Multi-question
    ("What did {a_person} decide? When did {b_person} agree?", "multi_qmark"),
]


def _gen_decomposition(rng: random.Random) -> dict:
    template, kind = _pick(rng, _DECOMP_TEMPLATES)
    a_topic = _pick(rng, _TOPICS)
    b_topic = _pick(rng, _TOPICS)
    a_verb = _pick(rng, _VERBS)
    b_verb = _pick(rng, _VERBS)
    a_person = _pick(rng, _PEOPLE)
    b_person = _pick(rng, _PEOPLE)
    query = template.format(
        a_topic=a_topic,
        b_topic=b_topic,
        a_verb=a_verb,
        b_verb=b_verb,
        a_person=a_person,
        b_person=b_person,
    )
    # Expected decomposition — sub-queries are the simpler clauses.
    sub_queries = [query]
    if kind == "temporal":
        sub_queries.extend([f"{a_verb} {a_topic}", f"{b_verb} {b_topic}"])
    elif kind == "causal":
        sub_queries.extend([f"{a_verb} {a_topic}", f"{b_topic} failure"])
    elif kind == "comparison":
        sub_queries.extend([a_topic, b_topic])
    elif kind == "conjunction":
        sub_queries.extend([f"{a_person} {a_topic}", f"{b_person} {a_topic}"])
    elif kind == "multi_qmark":
        sub_queries.extend([f"{a_person} decision", f"{b_person} agreement"])
    return {
        "task": "query_decomposition",
        "prompt": query,
        "expected_output": sub_queries[:4],
    }


# ---------------------------------------------------------------------------
# Task 2 — Query reformulation
# ---------------------------------------------------------------------------


def _gen_reformulation(rng: random.Random) -> dict:
    topic = _pick(rng, _TOPICS)
    verb = _pick(rng, _VERBS)
    original = f"Why did we {verb} {topic}?"
    paraphrases = [
        original,
        f"What was the reasoning behind {verb}ing {topic}?",
        f"{topic}: rationale for {verb}ing?",
        f"Justification to {verb} {topic}",
    ]
    return {
        "task": "query_reformulation",
        "prompt": original,
        "expected_output": paraphrases,
    }


# ---------------------------------------------------------------------------
# Task 3 — Entity extraction
# ---------------------------------------------------------------------------


def _gen_entity_extraction(rng: random.Random) -> dict:
    person = _pick(rng, _PEOPLE)
    project = _pick(rng, _PROJECTS)
    tool = _pick(rng, _TOOLS)
    prompt = f"{person} pushed {project} to use {tool} for the new rollout."
    return {
        "task": "entity_extraction",
        "prompt": prompt,
        "expected_output": {
            "people": [person],
            "projects": [project],
            "tools": [tool],
            "incidents": [],
        },
    }


# ---------------------------------------------------------------------------
# Task 4 — LLM-as-reranker
# ---------------------------------------------------------------------------


def _gen_rerank_judgment(rng: random.Random) -> dict:
    topic = _pick(rng, _TOPICS)
    query = f"Why {topic}?"
    # 3 candidates: 1 relevant, 1 tangential, 1 irrelevant.
    candidates = [
        {"id": "D-1", "text": f"We adopted {topic} because it scales to our load."},
        {"id": "D-2", "text": f"Unrelated note about {_pick(rng, _TOPICS)}."},
        {"id": "D-3", "text": f"{topic} has weekly maintenance windows on Sundays."},
    ]
    rng.shuffle(candidates)
    # Deterministic scoring: top-ranked has highest, etc.
    scores = {c["id"]: 0 for c in candidates}
    for c in candidates:
        if c["id"] == "D-1":
            scores[c["id"]] = 95
        elif c["id"] == "D-3":
            scores[c["id"]] = 55
        else:
            scores[c["id"]] = 10
    return {
        "task": "llm_rerank",
        "prompt": json.dumps(
            {"query": query, "candidates": candidates},
            default=str,
        ),
        "expected_output": scores,
    }


_GENS: tuple[Callable[[random.Random], dict], ...] = (
    _gen_decomposition,
    _gen_reformulation,
    _gen_entity_extraction,
    _gen_rerank_judgment,
)


def generate(count: int, seed: int = 42) -> Iterator[dict]:
    rng = random.Random(seed)
    for _ in range(count):
        g = rng.choice(_GENS)
        yield g(rng)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=5000)
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
