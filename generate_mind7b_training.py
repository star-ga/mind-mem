#!/usr/bin/env python3
"""Generate training data for Mind7B — a purpose-trained 7B model for mind-mem.

Tasks trained:
  1. Entity extraction (text → JSON entities)
  2. Fact extraction (text → JSON facts)
  3. Observation compression (blocks + question → observations)
  4. LLM reranking (query + candidates → scores)
  5. Block enrichment (text → entities + facts combined)

Output: mind7b_train.jsonl (chatml format for unsloth)
"""

import json
import os
import random
import re
import sys

random.seed(42)

# ---------------------------------------------------------------------------
# Prompt templates (exact copies from mind-mem source)
# ---------------------------------------------------------------------------

ENTITY_SYSTEM = (
    "You are Mind7B, a specialized memory extraction model. "
    "You extract entities from text and return structured JSON."
)

ENTITY_PROMPT = """\
Extract entities from the following text. Return a JSON array of objects, \
each with keys: "name" (string), "type" (one of: person, place, date, \
organization, decision, tool, project), "context" (short phrase).

Only return the JSON array, no explanation.

Text: {text}

JSON:"""

FACT_SYSTEM = (
    "You are Mind7B, a specialized memory extraction model. "
    "You extract factual claims from text and return structured JSON."
)

FACT_PROMPT = """\
Extract factual claims from the following text. Return a JSON array of \
objects, each with keys: "claim" (string, one sentence), "confidence" \
(float 0-1), "category" (one of: identity, event, preference, relation, \
negation, plan, state).

Only return the JSON array, no explanation.

Text: {text}

JSON:"""

COMPRESS_SYSTEM = """\
You are a memory compression expert. Given a set of conversation memory excerpts \
and a question, extract ONLY the facts relevant to answering the question.

Rules:
1. Output a numbered list of factual observations derived from the context.
2. Each observation must be a single, self-contained factual statement.
3. Include temporal information (dates, times, order of events) when present.
4. Include names, relationships, preferences, opinions, and specific details.
5. If multiple excerpts discuss the same topic, synthesize them into one observation.
6. Discard excerpts that are completely irrelevant to the question.
7. Preserve exact quotes, numbers, and proper nouns from the source material.
8. Output 3-8 observations. Fewer is fine if the context is sparse."""

COMPRESS_ADVERSARIAL = """\
You are extracting evidence from conversation excerpts for an adversarial question.
Your job is to extract relevant evidence, NOT to answer the question.

Rules:
1. Include speaker attribution for each item. If unknown, write UNKNOWN.
2. Prefer exact quotes, but if exact quoting is difficult, use a close \
paraphrase that preserves the factual content.
3. Include ALL mentions related to the question entities — even partial \
or indirect evidence. When in doubt, include it.
4. NEVER say "never mentioned" or "didn't happen" unless you have an \
explicit denial quote in the excerpts.
5. EVIDENCE_FOUND: NO only if there is truly ZERO relevant information \
in any excerpt. If anything is even slightly related, output YES.
6. Do not invent facts. Use only the provided excerpts.

Output format:

EVIDENCE_FOUND: YES|NO
EVIDENCE:
- [SPEAKER=<name|UNKNOWN>] "<quote or close paraphrase>"
DENIAL_EVIDENCE:
- [SPEAKER=<name|UNKNOWN>] "<denial quote>" (only if present)"""

COMPRESS_TEMPORAL = """\
You are a memory compression expert. The question asks about timing, \
sequence, or chronological order of events.

Rules:
1. Output a numbered list of factual observations in CHRONOLOGICAL ORDER.
2. Include exact dates, times, and relative ordering (before/after/during).
3. If events have a causal chain, preserve that order explicitly.
4. Note any changes over time (e.g., "First X, then changed to Y on [date]").
5. If timing is ambiguous, state what is known and what is uncertain.
6. Preserve exact dates and timestamps from the source material.
7. Output 3-8 observations."""

COMPRESS_MULTIHOP = """\
You are a memory compression expert. The question requires connecting \
multiple facts from different parts of the conversation.

Rules:
1. Output a numbered list of factual observations.
2. For each relevant fact, note its SOURCE (which excerpt or conversation \
segment it came from).
3. Explicitly state connections between facts when they exist.
4. If the question asks about a relationship between entities, list all \
facts about EACH entity separately, then note their connections.
5. Include indirect connections that require inference.
6. Preserve names, relationships, and cross-references.
7. Output 4-10 observations (multi-hop needs more detail)."""

RERANK_SYSTEM = (
    "You are Mind7B, a specialized memory reranking model. "
    "Given a query and candidate memory blocks, score each for relevance (0.0-1.0)."
)

RERANK_PROMPT = """\
Query: {query}

Candidates:
{candidates}

For each candidate, output a JSON array of objects with keys: "id" (string), "score" (float 0.0-1.0).
Higher score = more relevant to the query. Only return JSON, no explanation.

JSON:"""


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

ENTITY_TYPES = ["person", "place", "date", "organization", "decision", "tool", "project"]
CATEGORIES = ["identity", "event", "preference", "relation", "negation", "plan", "state"]

PEOPLE = [
    "Alice Chen", "Bob Martinez", "Carol Singh", "David Park", "Eve Johnson",
    "Frank Williams", "Grace Liu", "Hank Thompson", "Iris Patel", "Jack Wilson",
    "Nikolai", "Sarah", "Marcus", "Dr. Yuki Tanaka", "Professor Garcia",
    "CTO Jamie Lee", "Lisa from marketing", "Ahmed", "Fatima", "Raj"
]

ORGS = [
    "Acme Corp", "TechStart Inc", "DataFlow Labs", "CloudNine Systems",
    "STARGA Inc", "OpenAI", "Google DeepMind", "Meta AI", "Anthropic",
    "NeuraTech", "InfraScale", "DevOps United", "QualityFirst",
    "DataDriven Co", "SecureNet", "AgileMinds", "CodeCraft", "AI Dynamics"
]

TOOLS = [
    "PostgreSQL", "Redis", "Docker", "Kubernetes", "Terraform", "Jenkins",
    "GitHub Actions", "Prometheus", "Grafana", "Elasticsearch", "Kafka",
    "RabbitMQ", "Nginx", "FastAPI", "Django", "React", "Next.js", "Prisma",
    "SQLAlchemy", "PyTorch", "TensorFlow", "Ollama", "mind-mem", "Supabase"
]

PROJECTS = [
    "Project Atlas", "Phoenix Migration", "DataLake v2", "Auth Refactor",
    "API Gateway", "Dashboard Redesign", "ML Pipeline", "Edge Compute",
    "Mobile App v3", "Search Rewrite", "mind-mem", "512-mind",
    "CogNet", "NikolaChess", "mindlang.dev"
]

PLACES = [
    "San Francisco office", "Berlin HQ", "AWS us-east-1", "GCP europe-west4",
    "Dubai DIFC", "Tokyo data center", "production cluster", "staging environment",
    "the Slack channel", "meeting room B"
]

DATES = [
    "2026-01-15", "2026-02-20", "2025-11-03", "last Tuesday", "Q1 2026",
    "March 2026", "next sprint", "2026-03-28", "yesterday", "2026-04-01"
]


def make_entity_example():
    """Generate one entity extraction training example."""
    n_entities = random.randint(1, 5)
    entities = []
    text_parts = []

    for _ in range(n_entities):
        etype = random.choice(ENTITY_TYPES)
        if etype == "person":
            name = random.choice(PEOPLE)
            ctx = random.choice([
                f"discussed the deployment plan", f"reviewed the PR",
                f"reported the bug", f"suggested using {random.choice(TOOLS)}",
                f"leads the backend team", f"joined in Q1 2026",
                f"approved the migration", f"wrote the initial spec"
            ])
            text_parts.append(f"{name} {ctx}")
        elif etype == "organization":
            name = random.choice(ORGS)
            ctx = random.choice([
                "partnered on the integration", "provides the cloud infra",
                "acquired the startup", "sponsors the open-source project",
                "hired three engineers", "published the benchmark results"
            ])
            text_parts.append(f"{name} {ctx}")
        elif etype == "tool":
            name = random.choice(TOOLS)
            ctx = random.choice([
                "used for caching layer", "handles the message queue",
                "runs in production", "replaced the old system",
                "configured for auto-scaling", "version upgraded to latest"
            ])
            text_parts.append(f"{name} {ctx}")
        elif etype == "project":
            name = random.choice(PROJECTS)
            ctx = random.choice([
                "entering phase 2", "blocked by dependency issue",
                "launched successfully", "needs code review",
                "scheduled for next sprint", "behind schedule by 2 weeks"
            ])
            text_parts.append(f"{name} {ctx}")
        elif etype == "place":
            name = random.choice(PLACES)
            ctx = random.choice([
                "hosted the meeting", "deployed to this region",
                "experiencing latency", "being decommissioned"
            ])
            text_parts.append(f"At {name}, {ctx}")
        elif etype == "date":
            name = random.choice(DATES)
            ctx = random.choice([
                "deadline for the release", "when the incident occurred",
                "scheduled for deployment", "meeting was held"
            ])
            text_parts.append(f"On {name}, {ctx}")
        else:
            name = "API rate limit decision"
            ctx = "implemented per-user throttling"
            text_parts.append(f"Decision: {ctx}")

        entities.append({"name": name, "type": etype, "context": ctx})

    text = ". ".join(text_parts) + "."
    output = json.dumps(entities, indent=2)

    return {
        "messages": [
            {"role": "system", "content": ENTITY_SYSTEM},
            {"role": "user", "content": ENTITY_PROMPT.format(text=text)},
            {"role": "assistant", "content": output}
        ]
    }


def make_fact_example():
    """Generate one fact extraction training example."""
    n_facts = random.randint(1, 4)
    facts = []
    text_parts = []

    templates = {
        "identity": [
            ("{person} is the {role} at {org}",
             "{person} holds the {role} position at {org}"),
            ("{person} specializes in {domain}",
             "{person} is an expert in {domain}"),
        ],
        "event": [
            ("On {date}, {person} deployed {tool} to production",
             "{person} deployed {tool} to production on {date}"),
            ("{project} launched on {date}",
             "{project} went live on {date}"),
        ],
        "preference": [
            ("{person} prefers {tool} over {tool2} for {task}",
             "{person} favors using {tool} instead of {tool2} for {task}"),
        ],
        "relation": [
            ("{person} works with {person2} on {project}",
             "{person} and {person2} collaborate on {project}"),
            ("{org} uses {tool} for {task}",
             "{org} relies on {tool} for {task}"),
        ],
        "negation": [
            ("{person} does not use {tool} anymore",
             "{person} stopped using {tool}"),
            ("{org} no longer supports {tool}",
             "{org} deprecated {tool}"),
        ],
        "plan": [
            ("{person} plans to migrate {project} to {tool} by {date}",
             "{person} intends to move {project} to {tool} before {date}"),
        ],
        "state": [
            ("{project} is currently in {phase}",
             "{project} is at the {phase} stage"),
            ("{tool} has {count} daily active users",
             "{tool} serves {count} users daily"),
        ],
    }

    for _ in range(n_facts):
        cat = random.choice(CATEGORIES)
        confidence = round(random.uniform(0.6, 1.0), 2)

        # Fill template
        p1, p2 = random.sample(PEOPLE, 2)
        t1, t2 = random.sample(TOOLS, 2)
        vals = {
            "person": p1, "person2": p2,
            "org": random.choice(ORGS),
            "tool": t1, "tool2": t2,
            "project": random.choice(PROJECTS),
            "date": random.choice(DATES),
            "role": random.choice(["CTO", "lead engineer", "architect", "PM", "SRE"]),
            "domain": random.choice(["distributed systems", "ML ops", "security", "frontend", "DevOps"]),
            "task": random.choice(["CI/CD", "monitoring", "caching", "deployment", "testing"]),
            "phase": random.choice(["alpha", "beta", "production", "maintenance", "EOL"]),
            "count": random.choice(["500", "12K", "50K", "2M"]),
        }

        if cat in templates:
            tmpl_pair = random.choice(templates[cat])
            text_parts.append(tmpl_pair[0].format(**vals))
            claim = tmpl_pair[1].format(**vals)
        else:
            claim = f"{p1} works on {random.choice(PROJECTS)}"
            text_parts.append(claim)

        facts.append({"claim": claim, "confidence": confidence, "category": cat})

    text = ". ".join(text_parts) + "."
    output = json.dumps(facts, indent=2)

    return {
        "messages": [
            {"role": "system", "content": FACT_SYSTEM},
            {"role": "user", "content": FACT_PROMPT.format(text=text)},
            {"role": "assistant", "content": output}
        ]
    }


def make_compression_example():
    """Generate one observation compression training example."""
    variant = random.choice(["general", "adversarial", "temporal", "multi_hop"])

    # Generate fake memory excerpts
    n_excerpts = random.randint(3, 6)
    excerpts = []
    relevant_facts = []

    person = random.choice(PEOPLE)
    project = random.choice(PROJECTS)
    tool = random.choice(TOOLS)

    for i in range(n_excerpts):
        date = random.choice(DATES)
        if random.random() < 0.7:  # 70% relevant
            excerpt = random.choice([
                f"[{date}] {person} mentioned that {project} needs to migrate to {tool} before the deadline.",
                f"[{date}] Decision: The team will use {tool} for the {project} backend. {person} approved.",
                f"[{date}] {person} reported that {project} is running {random.randint(10,50)}% slower after the update.",
                f"[{date}] Sprint review: {person} completed the {tool} integration for {project}.",
                f"[{date}] {person} prefers {tool} because it handles concurrent writes better.",
            ])
            relevant_facts.append(f"{person} is involved with {project} using {tool} (as of {date})")
        else:  # 30% noise
            other_person = random.choice(PEOPLE)
            excerpt = random.choice([
                f"[{date}] {other_person} discussed lunch plans for the team.",
                f"[{date}] Office WiFi was down for 30 minutes.",
                f"[{date}] {other_person} shared a meme in the general channel.",
            ])
        excerpts.append(f"Excerpt {i+1}: {excerpt}")

    context = "\n".join(excerpts)

    questions = {
        "general": [
            f"What is {person}'s role in {project}?",
            f"What tools are being used for {project}?",
            f"What decisions were made about {project}?",
        ],
        "adversarial": [
            f"Did {person} ever express concerns about {tool}?",
            f"Was there any disagreement about using {tool} for {project}?",
        ],
        "temporal": [
            f"What was the timeline of {project}'s development?",
            f"When did {person} first mention {tool}?",
        ],
        "multi_hop": [
            f"How are {person} and {tool} connected through {project}?",
            f"What is the relationship between the {tool} migration and {project}'s performance?",
        ],
    }

    question = random.choice(questions[variant])

    sys_prompt = {
        "general": COMPRESS_SYSTEM,
        "adversarial": COMPRESS_ADVERSARIAL,
        "temporal": COMPRESS_TEMPORAL,
        "multi_hop": COMPRESS_MULTIHOP,
    }[variant]

    # Generate observations
    observations = []
    for i, fact in enumerate(relevant_facts[:random.randint(3, 6)], 1):
        observations.append(f"{i}. {fact}.")
    output = "\n".join(observations) if observations else "1. No relevant information found in the provided excerpts."

    user_msg = f"Question: {question}\n\nRetrieved memory excerpts:\n{context}\n\nExtract the relevant factual observations for answering this question."

    return {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": output}
        ]
    }


def make_rerank_example():
    """Generate one reranking training example."""
    query = random.choice([
        f"How does {random.choice(PROJECTS)} handle authentication?",
        f"What did {random.choice(PEOPLE)} decide about {random.choice(TOOLS)}?",
        f"When was {random.choice(PROJECTS)} last deployed?",
        f"Who is responsible for {random.choice(TOOLS)} maintenance?",
    ])

    n_candidates = random.randint(3, 6)
    candidates = []
    scores = []

    for i in range(n_candidates):
        cid = f"B-{random.randint(1000, 9999)}"
        relevance = random.random()
        excerpt = random.choice([
            f"{random.choice(PEOPLE)} discussed {random.choice(PROJECTS)} deployment using {random.choice(TOOLS)}.",
            f"Decision: Migrate to {random.choice(TOOLS)} for better performance.",
            f"{random.choice(PEOPLE)} reported a bug in {random.choice(PROJECTS)}.",
            f"Meeting notes from {random.choice(DATES)} about infrastructure.",
            f"{random.choice(ORGS)} partnership update regarding {random.choice(PROJECTS)}.",
        ])
        candidates.append(f"[{cid}] {excerpt}")
        scores.append({"id": cid, "score": round(relevance, 2)})

    # Sort scores by relevance (highest first)
    scores.sort(key=lambda x: x["score"], reverse=True)
    candidates_text = "\n".join(candidates)
    output = json.dumps(scores, indent=2)

    return {
        "messages": [
            {"role": "system", "content": RERANK_SYSTEM},
            {"role": "user", "content": RERANK_PROMPT.format(query=query, candidates=candidates_text)},
            {"role": "assistant", "content": output}
        ]
    }


# ---------------------------------------------------------------------------
# Load real workspace data for augmentation
# ---------------------------------------------------------------------------

def load_workspace_blocks(workspace_path):
    """Load real blocks from mind-mem workspace for training data."""
    blocks = []
    decisions_path = os.path.join(workspace_path, "decisions", "DECISIONS.md")
    signals_path = os.path.join(workspace_path, "intelligence", "SIGNALS.md")

    for fpath in [decisions_path, signals_path]:
        if not os.path.isfile(fpath):
            continue
        with open(fpath) as f:
            content = f.read()
        # Extract block text between markers
        for match in re.finditer(r'\[([A-Z]+-\d{8}-\d+)\](.*?)(?=\n\[|\Z)', content, re.DOTALL):
            bid = match.group(1)
            text = match.group(2).strip()[:500]
            if len(text) > 50:
                blocks.append({"id": bid, "text": text})

    return blocks


def make_real_entity_example(block):
    """Generate entity extraction example from real workspace block."""
    return {
        "messages": [
            {"role": "system", "content": ENTITY_SYSTEM},
            {"role": "user", "content": ENTITY_PROMPT.format(text=block["text"][:2000])},
            # We won't have ground truth for real blocks, so we generate
            # high-quality extraction using templates
        ]
    }


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def main():
    output_path = "mind7b_train.jsonl"

    # Target: ~2000 synthetic examples across all tasks
    N_ENTITY = 500
    N_FACT = 500
    N_COMPRESS = 500
    N_RERANK = 300

    examples = []

    print(f"Generating {N_ENTITY} entity extraction examples...")
    for _ in range(N_ENTITY):
        examples.append(make_entity_example())

    print(f"Generating {N_FACT} fact extraction examples...")
    for _ in range(N_FACT):
        examples.append(make_fact_example())

    print(f"Generating {N_COMPRESS} observation compression examples...")
    for _ in range(N_COMPRESS):
        examples.append(make_compression_example())

    print(f"Generating {N_RERANK} reranking examples...")
    for _ in range(N_RERANK):
        examples.append(make_rerank_example())

    # Shuffle
    random.shuffle(examples)

    # Write
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nTotal: {len(examples)} training examples")
    print(f"  Entity extraction: {N_ENTITY}")
    print(f"  Fact extraction:   {N_FACT}")
    print(f"  Compression:       {N_COMPRESS}")
    print(f"  Reranking:         {N_RERANK}")
    print(f"Output: {output_path}")

    # Stats
    total_tokens_est = sum(
        sum(len(m["content"].split()) for m in ex["messages"])
        for ex in examples
    )
    print(f"Est. total words: {total_tokens_est:,} (~{total_tokens_est*1.3:.0f} tokens)")


if __name__ == "__main__":
    main()
