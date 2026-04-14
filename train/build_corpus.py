"""Harvest a training corpus for the mind-mem-4b model.

Produces /home/n/mm-train-output/corpus.jsonl — one example per line, in
the chat format expected by the SFTTrainer.  Each example follows::

    {"messages": [
        {"role": "system",  "content": "..."},
        {"role": "user",    "content": "..."},
        {"role": "assistant","content": "..."}
    ]}

Sources (deterministic — no LLM calls, no network):
    1. MCP tool docstrings       → "what does tool X do?"  Q/A pairs.
    2. Block-schema grammars     → "generate a valid <TYPE> block".
    3. CHANGELOG entries         → "what changed in version X?".
    4. docs/ prose               → fill-in-the-blank + summarization.
    5. Governance workflow demos → end-to-end propose→approve→verify
       sequences rendered as multi-turn chats.

The file is idempotent: running build_corpus.py twice produces the
same output bytes.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable, Iterator

REPO = Path("/home/n/mind-mem")
OUT = Path(
    os.environ.get(
        "MM_CORPUS_OUT",
        "/data/checkpoints/mm-workspace/train-output/corpus.jsonl",
    )
)

SYSTEM_PROMPT = (
    "You are mind-mem-4b, a memory-governance assistant specialised in "
    "auditable, contradiction-safe memory for coding agents. You know "
    "the mind-mem Python package, its 57 MCP tools, block schemas, "
    "governance workflows, and CHANGELOG history. You respond "
    "concisely, cite exact tool names / block types, and refuse to "
    "invent APIs that do not exist in the package."
)


# ---------------------------------------------------------------------------
# Source 1: MCP tool docstrings
# ---------------------------------------------------------------------------


def _harvest_mcp_tools() -> Iterator[dict]:
    """Extract every @mcp.tool function + docstring from mcp_server.py.

    Emits multiple Q/A variants per tool to drive name recall under
    different phrasings. Tool names must appear in every assistant
    response so the model learns "cite the exact tool".
    """
    path = REPO / "src" / "mind_mem" / "mcp_server.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any(
            (isinstance(d, ast.Attribute) and d.attr == "tool")
            or (isinstance(d, ast.Name) and d.id == "tool")
            for d in node.decorator_list
        ):
            continue
        doc = ast.get_docstring(node) or ""
        if not doc.strip():
            continue
        summary = doc.strip().split("\n\n")[0].replace("\n", " ")
        args = _fn_arg_repr(node)
        tool_name = node.name
        hint = _usage_hint(summary)

        # Variant 1 — documentation lookup.
        yield {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"What does the mind-mem MCP tool `{tool_name}` do?"},
            {"role": "assistant", "content": f"`{tool_name}({args})` — {summary}"},
        ]}
        # Variant 2 — reverse lookup by intent.
        yield {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Which MCP tool should I call to {hint}?"},
            {"role": "assistant", "content": f"Call `{tool_name}`. Signature: `{tool_name}({args})`."},
        ]}
        # Variant 3 — short direct form.
        yield {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Name the mind-mem tool for: {hint}."},
            {"role": "assistant", "content": f"`{tool_name}`."},
        ]}
        # Variant 4 — signature question.
        yield {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"What is the signature of `{tool_name}`?"},
            {"role": "assistant", "content": f"`{tool_name}({args})`"},
        ]}
        # Variant 5 — what returns.
        yield {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"What does `{tool_name}` return?"},
            {"role": "assistant", "content": (
                f"`{tool_name}` returns a JSON envelope. From the docstring:\n\n{doc.strip()}"
            )},
        ]}
        # Variant 6 — how phrased differently.
        for alt in _paraphrase_usage(hint):
            yield {"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": alt},
                {"role": "assistant", "content": f"Use the `{tool_name}` MCP tool."},
            ]}


def _paraphrase_usage(hint: str) -> list[str]:
    """Generate 3 phrasings of a 'use case' question for tool recall."""
    hint = hint.strip().rstrip(".")
    return [
        f"How do I {hint}?",
        f"I need to {hint}. Which tool?",
        f"What's the right mind-mem call to {hint}?",
    ]


def _fn_arg_repr(node: ast.FunctionDef) -> str:
    parts: list[str] = []
    args = node.args
    defaults = list(args.defaults)
    # positional
    pos = list(args.args)
    default_offset = len(pos) - len(defaults)
    for i, a in enumerate(pos):
        ann = f": {ast.unparse(a.annotation)}" if a.annotation else ""
        if i >= default_offset:
            d = defaults[i - default_offset]
            parts.append(f"{a.arg}{ann} = {ast.unparse(d)}")
        else:
            parts.append(f"{a.arg}{ann}")
    # keyword-only
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        ann = f": {ast.unparse(a.annotation)}" if a.annotation else ""
        if d is not None:
            parts.append(f"{a.arg}{ann} = {ast.unparse(d)}")
        else:
            parts.append(f"{a.arg}{ann}")
    return ", ".join(parts)


def _usage_hint(summary: str) -> str:
    # strip leading article + final period for a smoother Q
    text = summary.strip().rstrip(".")
    for lead in ("Return ", "Returns ", "Get ", "List ", "Compute "):
        if text.startswith(lead):
            text = text[len(lead):]
            break
    return text.lower()


# ---------------------------------------------------------------------------
# Source 2: Block schemas
# ---------------------------------------------------------------------------


_BLOCK_TYPES: list[tuple[str, str, list[str]]] = [
    ("DEC", "Decision block (ADR-style).", ["ProposalId", "Date", "Status", "Title", "Rationale", "Evidence"]),
    ("ADR", "Architecture Decision Record.", ["ADR", "Date", "Status", "Context", "Decision", "Consequences"]),
    ("CODE", "Code-change decision.", ["Id", "Date", "Status", "File", "Change", "Rationale"]),
    ("PERF", "Performance record.", ["Id", "Date", "Metric", "Before", "After", "Rationale"]),
    ("ALGO", "Algorithm choice.", ["Id", "Date", "Problem", "Chosen", "Alternatives", "Rationale"]),
    ("BUG", "Bug report / fix.", ["Id", "Date", "Severity", "Symptom", "RootCause", "Fix"]),
    ("CONV", "Code convention.", ["Id", "Convention", "Example", "Scope"]),
    ("DREF", "Drift signal.", ["Id", "Date", "Severity", "Signal", "Summary"]),
    ("CHECK", "Contradiction.", ["Id", "Date", "BlockA", "BlockB", "Type"]),
    ("EV", "Evidence object.", ["EvidenceId", "Action", "Actor", "TargetBlock", "PayloadHash"]),
    ("FIELD", "Field audit entry.", ["FieldChangeId", "BlockId", "Field", "Old", "New"]),
    ("TIER", "Memory tier record.", ["BlockId", "Tier", "Promotions", "LastAccess"]),
    ("IMAGE", "Image block.", ["Id", "Date", "Path", "Caption", "Hash"]),
    ("AUDIO", "Audio block.", ["Id", "Date", "Path", "Transcript", "Hash"]),
]


def _harvest_block_schemas() -> Iterator[dict]:
    prompts_show = [
        "Show me a mind-mem {short} block template.",
        "Give me the canonical {short} block format.",
        "Print a {short} block skeleton.",
        "What does a {short} block look like?",
        "Template for a {short} block, please.",
    ]
    prompts_fields = [
        "What fields does a {short} block require?",
        "List the required fields of a {short} block.",
        "Which keys live inside a {short} block?",
    ]
    for short, desc, fields in _BLOCK_TYPES:
        # Canonical template (always starts with [SHORT-DATE-NNN] so the
        # model learns the exact ID prefix).
        example = f"[{short}-20260413-001]\n" + "\n".join(
            f"{f}: <{f.lower()}>" for f in fields
        )
        for q in prompts_show:
            yield {"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q.format(short=short)},
                {"role": "assistant", "content": f"{desc}\n\n```\n{example}\n```"},
            ]}
        for q in prompts_fields:
            yield {"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q.format(short=short)},
                {"role": "assistant", "content": (
                    f"A mind-mem {short} block requires: "
                    + ", ".join(f"`{f}`" for f in fields)
                    + "."
                )},
            ]}
        # What type is this by ID? (ID-prefix → type direction)
        yield {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"A block ID starts with `[{short}-`. What type is it?"},
            {"role": "assistant", "content": f"`{short}` — {desc}"},
        ]}


# ---------------------------------------------------------------------------
# Source 3: CHANGELOG entries
# ---------------------------------------------------------------------------


_CHANGELOG_ENTRY_RE = re.compile(r"^## ([0-9][^ \n]*) \((\d{4}-\d{2}-\d{2})\)", re.MULTILINE)


def _harvest_changelog() -> Iterator[dict]:
    path = REPO / "CHANGELOG.md"
    text = path.read_text(encoding="utf-8")
    matches = list(_CHANGELOG_ENTRY_RE.finditer(text))
    for i, m in enumerate(matches):
        version, date = m.group(1), m.group(2)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"What changed in mind-mem {version}?",
                },
                {
                    "role": "assistant",
                    "content": f"mind-mem {version} (released {date}):\n\n{body}",
                },
            ]
        }


# ---------------------------------------------------------------------------
# Source 4: docs/ prose
# ---------------------------------------------------------------------------


def _harvest_docs() -> Iterator[dict]:
    docs = REPO / "docs"
    if not docs.is_dir():
        return
    for md in sorted(docs.rglob("*.md")):
        text = md.read_text(encoding="utf-8", errors="replace")
        for section in _split_sections(text):
            heading, body = section
            if not body.strip() or len(body) < 80:
                continue
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Explain the mind-mem documentation section "
                            f"'{heading}' from {md.name}."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": body.strip(),
                    },
                ]
            }


def _split_sections(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    lines = text.splitlines()
    current_heading = ""
    current_body: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if current_heading:
                out.append((current_heading, "\n".join(current_body)))
            current_heading = line[3:].strip()
            current_body = []
        else:
            current_body.append(line)
    if current_heading:
        out.append((current_heading, "\n".join(current_body)))
    return out


# ---------------------------------------------------------------------------
# Source 5: Governance workflow demos
# ---------------------------------------------------------------------------


_WORKFLOW_PARAPHRASES: dict[str, list[str]] = {
    "contradict-and-resolve": [
        "I see mind-mem reporting a contradiction between two decision blocks. What should I do?",
        "How do I resolve a contradiction?",
        "Walk me through fixing a contradiction between two decision blocks.",
        "Contradiction detected — what's the remediation flow?",
    ],
    "belief-drift": [
        "How do I detect whether a belief has drifted over time?",
        "Which tools surface belief drift?",
        "How do I query drift signals?",
        "I want to check if a belief has drifted. Which tools do I call?",
    ],
    "rollback": [
        "I applied a bad proposal. How do I roll back and what happens to the belief state?",
        "How do I roll back a proposal safely?",
        "Undo a bad apply — what's the procedure?",
        "I applied a bad proposal. How do I roll back safely?",
    ],
    "audit-trail": [
        "Who changed field X on block Y, and when?",
        "How do I see the history of field X on block Y?",
        "Audit who changed field X on block Y.",
        "Show field-level audit history for a block.",
    ],
    "encryption": [
        "Can I encrypt a sensitive memory file at rest?",
        "How do I encrypt a mind-mem file?",
        "How do I enable encryption at rest?",
    ],
    "tiers": [
        "What happens to a block as it's accessed more often?",
        "How does mind-mem promote blocks through tiers?",
        "Explain the memory-tier promotion cycle.",
    ],
    "verify": [
        "How do I verify the audit chain integrity?",
        "Prove my audit chain hasn't been tampered with.",
        "Which tool checks the hash chain?",
    ],
    "governance-bench": [
        "Run the full governance benchmark suite.",
        "How do I benchmark governance health?",
        "What's the governance health bench command?",
    ],
}


_WORKFLOWS: list[dict] = [
    {
        "scenario": "contradict-and-resolve",
        "turns": [
            (
                "user",
                "I see mind-mem reporting a contradiction between two decision blocks. What should I do?",
            ),
            (
                "assistant",
                "Run `list_contradictions` to fetch the analysed list. Each entry now carries `confidence_score`, `side_effects`, and a `strategy` recommendation from the AutoResolver. For strategy=supersede with high confidence and no side_effects, call `propose_update` with the generated patch, then review the proposal and call `approve_apply` with `dry_run=false`. Finally `verify_chain` confirms the audit ledger is intact.",
            ),
        ],
    },
    {
        "scenario": "belief-drift",
        "turns": [
            (
                "user",
                "How do I detect whether a belief has drifted over time?",
            ),
            (
                "assistant",
                "mind-mem surfaces two drift signals. The lexical pass appears in DRIFT.md during `scan`. The semantic pass is stored in SQLite by `DriftDetector` and queryable via `belief_timeline(block_id)` or `recent_signals(min_confidence=0.6)`. In v2.9.0+ both run on every `scan` call, so `recent_signals` is always live.",
            ),
        ],
    },
    {
        "scenario": "rollback",
        "turns": [
            (
                "user",
                "I applied a bad proposal. How do I roll back and what happens to the belief state?",
            ),
            (
                "assistant",
                "Call `rollback_proposal(receipt_ts)` with the receipt timestamp printed at apply time (format YYYYMMDD-HHMMSS). rollback restores the delta snapshot, marks the proposal rolled_back, and pushes an observation=0.0 into BeliefStore under source='rollback', so the Kalman confidence for the target block drops accordingly. Verify with `verify_chain` + `belief_timeline(target_block_id)`.",
            ),
        ],
    },
    {
        "scenario": "audit-trail",
        "turns": [
            (
                "user",
                "Who changed field X on block Y, and when?",
            ),
            (
                "assistant",
                "FieldAuditor records every field mutation with before/after values, agent attribution, and a reason string. Since v2.9.0 `apply_engine._op_update_field` fires FieldAuditor automatically, so `FieldAuditor(ws).field_history(block_id=Y, field='X')` returns the chronological history. Each entry is hash-linked into AuditChain so tampering is detectable.",
            ),
        ],
    },
    {
        "scenario": "encryption",
        "turns": [
            (
                "user",
                "Can I encrypt a sensitive memory file at rest?",
            ),
            (
                "assistant",
                "Set `MIND_MEM_ENCRYPTION_PASSPHRASE` in the MCP server environment, then call the admin tool `encrypt_file(file_path)`. The on-disk file is XChaCha20-Poly1305-style encrypted with a per-file nonce + HMAC, and the magic header prevents double-encryption. To read it back, call the admin `decrypt_file(file_path)` which returns base64-encoded plaintext without modifying the stored ciphertext. Full transparent read/write-path encryption is on the v3.0.0 roadmap.",
            ),
        ],
    },
    {
        "scenario": "tiers",
        "turns": [
            (
                "user",
                "What happens to a block as it's accessed more often?",
            ),
            (
                "assistant",
                "mind-mem tracks 4 memory tiers: WORKING, SHARED, LONG_TERM, VERIFIED. `TierManager.run_promotion_cycle` (invoked by compaction in v2.9.0+) promotes blocks that cross access-frequency + age thresholds. Each tier has a different retrieval_boost so long-lived, frequently-accessed blocks rank higher in recall results.",
            ),
        ],
    },
    {
        "scenario": "verify",
        "turns": [
            ("user", "How do I verify the audit chain integrity?"),
            ("assistant", "Call the `verify_chain` MCP tool. It walks the hash-chained `EvidenceChain` and returns `{ok: true, entries: N}` when the chain is intact or flags the first broken entry otherwise."),
        ],
    },
    {
        "scenario": "governance-bench",
        "turns": [
            ("user", "Run the full governance benchmark suite."),
            ("assistant", "Call `governance_health_bench`. It exercises contradiction detection, audit completeness, drift detection, and scalability in one pass, returning a JSON report with per-suite pass/fail."),
        ],
    },
]


def _harvest_workflow_paraphrases() -> Iterator[dict]:
    """One Q/A entry per scenario × phrasing, so each workflow gets
    multiple entry points but a stable assistant response."""
    canonical_answers: dict[str, str] = {}
    for wf in _WORKFLOWS:
        canonical_answers[wf["scenario"]] = wf["turns"][-1][1]
    for scen, phrasings in _WORKFLOW_PARAPHRASES.items():
        answer = canonical_answers.get(scen)
        if not answer:
            continue
        for q in phrasings:
            yield {"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": answer},
            ]}


# ---------------------------------------------------------------------------
# Source 6: Direct tool-name citations — teach "cite the exact tool"
# ---------------------------------------------------------------------------


_TOOL_CITATIONS: list[tuple[str, str]] = [
    ("How do I verify the audit chain integrity?", "verify_chain"),
    ("Which tool lists all contradictions?", "list_contradictions"),
    ("Which tool applies a staged proposal?", "approve_apply"),
    ("Which tool rolls back an apply?", "rollback_proposal"),
    ("Which tool writes a new memory block?", "propose_update"),
    ("Which tool runs full-text search over memory?", "recall"),
    ("Which tool runs hybrid BM25 + vector search?", "hybrid_search"),
    ("Which tool reindexes FTS tables?", "reindex"),
    ("Which tool scans the workspace for contradictions and drift?", "scan"),
    ("Which tool exports memory to JSONL?", "export_memory"),
    ("Which tool classifies query intent?", "intent_classify"),
    ("Which tool returns index statistics?", "index_stats"),
    ("Which tool runs the governance health benchmark?", "governance_health_bench"),
    ("Which tool encrypts a workspace file at rest?", "encrypt_file"),
    ("Which tool decrypts an encrypted workspace file?", "decrypt_file"),
    ("Which tool finds blocks similar to a given ID?", "find_similar"),
    ("Which tool deletes a memory item?", "delete_memory_item"),
    ("Which tool summarises block categories?", "category_summary"),
    ("Which tool prefetches recall results?", "prefetch"),
    ("Which tool diagnoses why a query returned its results?", "retrieval_diagnostics"),
    ("Which tool traverses the cross-reference graph?", "traverse_graph"),
    ("Which tool lists stale blocks?", "stale_blocks"),
    ("Which tool runs the dream consolidation cycle?", "dream_cycle"),
    ("Which tool returns memory-evolution timeline?", "memory_evolution"),
    ("Which tool lists stored evidence objects?", "list_evidence"),
    ("Which tool returns a single block by ID?", "get_block"),
    ("Which tool reports memory-system health?", "memory_health"),
    ("Which tool records calibration feedback?", "calibration_feedback"),
    ("Which tool returns calibration stats?", "calibration_stats"),
    ("Which tool packs a query result into a recall budget?", "pack_recall_budget"),
    ("Which tool plans a consolidation pass?", "plan_consolidation"),
    ("Which tool walks an Obsidian vault for blocks?", "vault_scan"),
    ("Which tool writes a single block back into a vault?", "vault_sync"),
    ("Which tool injects context for an agent?", "agent_inject"),
    ("Which tool observes a signal event?", "observe_signal"),
    ("Which tool reports signal stats?", "signal_stats"),
    ("Which tool loads a project profile?", "project_profile"),
    ("Which tool validates an ontology?", "ontology_validate"),
    ("Which tool loads an ontology?", "ontology_load"),
    ("Which tool reports change-stream status?", "stream_status"),
    ("Which tool propagates staleness marks?", "propagate_staleness"),
    ("Which tool builds a core snapshot?", "build_core"),
    ("Which tool loads a core snapshot?", "load_core"),
    ("Which tool unloads a core?", "unload_core"),
    ("Which tool lists cores?", "list_cores"),
    ("Which tool adds a graph edge?", "graph_add_edge"),
    ("Which tool queries the graph?", "graph_query"),
    ("Which tool returns graph stats?", "graph_stats"),
    ("Which tool verifies the Merkle root?", "verify_merkle"),
    ("Which tool runs the mind-mem verify suite?", "mind_mem_verify"),
]


_TOOL_IMPERATIVES: list[tuple[str, str]] = [
    ("Verify the audit chain integrity.", "verify_chain"),
    ("Verify the audit chain.", "verify_chain"),
    ("List contradictions.", "list_contradictions"),
    ("List all contradictions.", "list_contradictions"),
    ("Show contradictions.", "list_contradictions"),
    ("Apply a staged proposal.", "approve_apply"),
    ("Apply the proposal.", "approve_apply"),
    ("Approve and apply.", "approve_apply"),
    ("Roll back an apply.", "rollback_proposal"),
    ("Rollback the last apply.", "rollback_proposal"),
    ("Undo an apply.", "rollback_proposal"),
    ("Write a new memory block.", "propose_update"),
    ("Propose an update.", "propose_update"),
    ("Stage a new block.", "propose_update"),
    ("Full-text search over memory.", "recall"),
    ("Search memory.", "recall"),
    ("Find a block by keyword.", "recall"),
    ("Run hybrid BM25 + vector search.", "hybrid_search"),
    ("Run a hybrid search.", "hybrid_search"),
    ("Reindex the FTS tables.", "reindex"),
    ("Reindex memory.", "reindex"),
    ("Rebuild the index.", "reindex"),
    ("Scan the workspace for contradictions and drift.", "scan"),
    ("Scan for contradictions.", "scan"),
    ("Run a drift scan.", "scan"),
    ("Export memory to JSONL.", "export_memory"),
    ("Export memory.", "export_memory"),
    ("Classify the intent of a query.", "intent_classify"),
    ("Classify intent.", "intent_classify"),
    ("Get index statistics.", "index_stats"),
    ("Show index stats.", "index_stats"),
    ("Run the governance health benchmark.", "governance_health_bench"),
    ("Run the governance bench.", "governance_health_bench"),
    ("Encrypt a workspace file at rest.", "encrypt_file"),
    ("Encrypt a file.", "encrypt_file"),
    ("Decrypt an encrypted workspace file.", "decrypt_file"),
    ("Decrypt a file.", "decrypt_file"),
    ("Find blocks similar to a given ID.", "find_similar"),
    ("Find similar blocks.", "find_similar"),
    ("Delete a memory item.", "delete_memory_item"),
    ("Delete a block.", "delete_memory_item"),
    ("Show block category summaries.", "category_summary"),
    ("Summarise categories.", "category_summary"),
    ("Prefetch recall results.", "prefetch"),
    ("Diagnose why a query returned its results.", "retrieval_diagnostics"),
    ("Diagnose a recall result.", "retrieval_diagnostics"),
    ("Traverse the cross-reference graph.", "traverse_graph"),
    ("List stale blocks.", "stale_blocks"),
    ("Run the dream consolidation cycle.", "dream_cycle"),
    ("Return memory-evolution timeline.", "memory_evolution"),
    ("List stored evidence objects.", "list_evidence"),
    ("Return a single block by ID.", "get_block"),
    ("Get a block by ID.", "get_block"),
    ("Report memory-system health.", "memory_health"),
    ("Record calibration feedback.", "calibration_feedback"),
    ("Return calibration stats.", "calibration_stats"),
    ("Pack a query result into a recall budget.", "pack_recall_budget"),
    ("Plan a consolidation pass.", "plan_consolidation"),
    ("Walk an Obsidian vault for blocks.", "vault_scan"),
    ("Write a single block back into a vault.", "vault_sync"),
    ("Inject context for an agent.", "agent_inject"),
    ("Observe a signal event.", "observe_signal"),
    ("Report signal stats.", "signal_stats"),
    ("Load a project profile.", "project_profile"),
    ("Validate an ontology.", "ontology_validate"),
    ("Load an ontology.", "ontology_load"),
    ("Report change-stream status.", "stream_status"),
    ("Propagate staleness marks.", "propagate_staleness"),
    ("Build a core snapshot.", "build_core"),
    ("Load a core snapshot.", "load_core"),
    ("Unload a core.", "unload_core"),
    ("List cores.", "list_cores"),
    ("Add a graph edge.", "graph_add_edge"),
    ("Query the graph.", "graph_query"),
    ("Return graph stats.", "graph_stats"),
    ("Verify the Merkle root.", "verify_merkle"),
    ("Run the mind-mem verify suite.", "mind_mem_verify"),
]


def _harvest_tool_citations() -> Iterator[dict]:
    """Tight Q/A pairs forcing the model to name the exact tool.

    Emits multiple phrasings per tool — interrogative AND imperative —
    so the name is the reinforced signal regardless of phrasing.
    """
    phrasings = [
        ("{q}", "`{tool}`."),
        ("{q}", "Use `{tool}`."),
        ("{q}", "Call `{tool}`."),
        ("In mind-mem, {q_lower}", "`{tool}`."),
        ("For mind-mem, {q_lower}", "You want `{tool}`."),
    ]
    # Interrogative forms.
    for q, tool in _TOOL_CITATIONS:
        q_lower = q[0].lower() + q[1:] if q else q
        for qp, ap in phrasings:
            yield {"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": qp.format(q=q, q_lower=q_lower)},
                {"role": "assistant", "content": ap.format(tool=tool)},
            ]}
    # Imperative forms — "Apply a staged proposal." → "`approve_apply`."
    imp_phrasings = [
        ("{q}", "`{tool}`."),
        ("{q}", "Use `{tool}`."),
        ("{q}", "Call `{tool}`."),
        ("{q}", "The mind-mem tool for that is `{tool}`."),
    ]
    for q, tool in _TOOL_IMPERATIVES:
        for qp, ap in imp_phrasings:
            yield {"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": qp.format(q=q)},
                {"role": "assistant", "content": ap.format(tool=tool)},
            ]}


# ---------------------------------------------------------------------------
# Source 7: Strict workflow tool chains (terse tool-sequence answers)
# ---------------------------------------------------------------------------


_WORKFLOW_CHAINS: list[tuple[str, str]] = [
    (
        "I see a contradiction between two decision blocks. Walk me through the fix.",
        "1. `list_contradictions`\n2. `propose_update`\n3. `approve_apply`\n4. `verify_chain`",
    ),
    (
        "I applied a bad proposal. How do I roll back safely?",
        "1. `rollback_proposal`\n2. `verify_chain`",
    ),
    (
        "I want to check if a belief has drifted. Which tools do I call?",
        "1. `scan`\n2. `recent_signals` (see drift_detector)",
    ),
    (
        "Audit who changed field X on block Y.",
        "Use `FieldAuditor.field_history(block_id=Y, field='X')`.",
    ),
    (
        "Run the full governance benchmark suite.",
        "Call `governance_health_bench`.",
    ),
]


def _harvest_workflow_chains() -> Iterator[dict]:
    """Terse tool-chain answers — for workflow-questions the response
    should BE the sequence of tool names, not an explanation."""
    for q, chain in _WORKFLOW_CHAINS:
        for prefix in ("", "In mind-mem: ", "Mind-mem workflow: "):
            yield {"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prefix + q if prefix else q},
                {"role": "assistant", "content": chain},
            ]}


def _harvest_workflows() -> Iterator[dict]:
    for wf in _WORKFLOWS:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for role, content in wf["turns"]:
            messages.append({"role": role, "content": content})
        yield {"messages": messages}


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def _dedup(entries: Iterable[dict]) -> Iterator[dict]:
    seen: set[str] = set()
    for e in entries:
        key = hashlib.sha256(
            json.dumps(e, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        yield e


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with OUT.open("w", encoding="utf-8") as fh:
        for src in (
            _harvest_mcp_tools(),
            _harvest_block_schemas(),
            _harvest_changelog(),
            _harvest_docs(),
            _harvest_workflows(),
            _harvest_workflow_paraphrases(),
            _harvest_tool_citations(),
            _harvest_workflow_chains(),
        ):
            for entry in _dedup(src):
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
    print(f"wrote {count} examples to {OUT}")


if __name__ == "__main__":
    main()
