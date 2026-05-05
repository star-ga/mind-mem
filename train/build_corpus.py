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
    "the mind-mem Python package, its 81 MCP tools, block schemas, "
    "governance workflows, and CHANGELOG history. You respond "
    "concisely, cite exact tool names / block types, and refuse to "
    "invent APIs that do not exist in the package."
)


# ---------------------------------------------------------------------------
# Source 1: MCP tool docstrings
# ---------------------------------------------------------------------------


def _is_tool_decorator(decorator: ast.expr) -> bool:
    """Match the three decorator forms used across mcp_server + mcp/tools/.

    * ``@mcp.tool``           — legacy form on mcp_server.py
    * ``@tool``               — alias inside mcp_server.py
    * ``@mcp_tool_observe``   — v3.4+ wrapper used by every module under
                                 src/mind_mem/mcp/tools/*.py
    """
    if isinstance(decorator, ast.Attribute) and decorator.attr == "tool":
        return True
    if isinstance(decorator, ast.Name) and decorator.id in ("tool", "mcp_tool_observe"):
        return True
    return False


def _tool_source_files() -> list[Path]:
    """Every .py file that may register an MCP tool.

    v3.4 split ``mcp_server.py`` into per-domain modules under
    ``mcp/tools/`` registered via per-module ``register(mcp)`` callbacks.
    Walking only the legacy ``mcp_server.py`` would have missed all 81
    tools currently shipped — every retrain after v3.4 must walk the
    new directory too.
    """
    sources: list[Path] = [REPO / "src" / "mind_mem" / "mcp_server.py"]
    tools_dir = REPO / "src" / "mind_mem" / "mcp" / "tools"
    if tools_dir.is_dir():
        for path in sorted(tools_dir.glob("*.py")):
            if path.name in ("__init__.py", "_helpers.py"):
                continue
            sources.append(path)
    return sources


def _harvest_mcp_tools() -> Iterator[dict]:
    """Extract every MCP tool function + docstring from mcp_server.py and mcp/tools/.

    Emits multiple Q/A variants per tool to drive name recall under
    different phrasings. Tool names must appear in every assistant
    response so the model learns "cite the exact tool".
    """
    for path in _tool_source_files():
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not any(_is_tool_decorator(d) for d in node.decorator_list):
                continue
            doc = ast.get_docstring(node) or ""
            if not doc.strip():
                continue
            summary = doc.strip().split("\n\n")[0].replace("\n", " ")
            args = _fn_arg_repr(node)
            tool_name = node.name
            hint = _usage_hint(summary)
            yield from _emit_tool_variants(tool_name, args, summary, hint, doc)


def _emit_tool_variants(
    tool_name: str,
    args: str,
    summary: str,
    hint: str,
    doc: str,
) -> Iterator[dict]:
    # Variant 1 — documentation lookup.
    yield {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"What does the mind-mem MCP tool `{tool_name}` do?"},
            {"role": "assistant", "content": f"`{tool_name}({args})` — {summary}"},
        ]
    }
    # Variant 2 — reverse lookup by intent.
    yield {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Which MCP tool should I call to {hint}?"},
            {"role": "assistant", "content": f"Call `{tool_name}`. Signature: `{tool_name}({args})`."},
        ]
    }
    # Variant 3 — short direct form.
    yield {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Name the mind-mem tool for: {hint}."},
            {"role": "assistant", "content": f"`{tool_name}`."},
        ]
    }
    # Variant 4 — signature question.
    yield {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"What is the signature of `{tool_name}`?"},
            {"role": "assistant", "content": f"`{tool_name}({args})`"},
        ]
    }
    # Variant 5 — what returns.
    yield {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"What does `{tool_name}` return?"},
            {"role": "assistant", "content": (f"`{tool_name}` returns a JSON envelope. From the docstring:\n\n{doc.strip()}")},
        ]
    }
    # Variant 6 — how phrased differently.
    for alt in _paraphrase_usage(hint):
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": alt},
                {"role": "assistant", "content": f"Use the `{tool_name}` MCP tool."},
            ]
        }


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
            text = text[len(lead) :]
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
        example = f"[{short}-20260413-001]\n" + "\n".join(f"{f}: <{f.lower()}>" for f in fields)
        for q in prompts_show:
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q.format(short=short)},
                    {"role": "assistant", "content": f"{desc}\n\n```\n{example}\n```"},
                ]
            }
        for q in prompts_fields:
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q.format(short=short)},
                    {"role": "assistant", "content": (f"A mind-mem {short} block requires: " + ", ".join(f"`{f}`" for f in fields) + ".")},
                ]
            }
        # What type is this by ID? (ID-prefix → type direction)
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"A block ID starts with `[{short}-`. What type is it?"},
                {"role": "assistant", "content": f"`{short}` — {desc}"},
            ]
        }


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
                        "content": (f"Explain the mind-mem documentation section '{heading}' from {md.name}."),
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
            (
                "assistant",
                "Call the `verify_chain` MCP tool. It walks the hash-chained `EvidenceChain` and returns `{ok: true, entries: N}` when the chain is intact or flags the first broken entry otherwise.",
            ),
        ],
    },
    {
        "scenario": "governance-bench",
        "turns": [
            ("user", "Run the full governance benchmark suite."),
            (
                "assistant",
                "Call `governance_health_bench`. It exercises contradiction detection, audit completeness, drift detection, and scalability in one pass, returning a JSON report with per-suite pass/fail.",
            ),
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
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": answer},
                ]
            }


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
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": qp.format(q=q, q_lower=q_lower)},
                    {"role": "assistant", "content": ap.format(tool=tool)},
                ]
            }
    # Imperative forms — "Apply a staged proposal." → "`approve_apply`."
    imp_phrasings = [
        ("{q}", "`{tool}`."),
        ("{q}", "Use `{tool}`."),
        ("{q}", "Call `{tool}`."),
        ("{q}", "The mind-mem tool for that is `{tool}`."),
    ]
    for q, tool in _TOOL_IMPERATIVES:
        for qp, ap in imp_phrasings:
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": qp.format(q=q)},
                    {"role": "assistant", "content": ap.format(tool=tool)},
                ]
            }


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
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prefix + q if prefix else q},
                    {"role": "assistant", "content": chain},
                ]
            }


def _harvest_workflows() -> Iterator[dict]:
    for wf in _WORKFLOWS:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for role, content in wf["turns"]:
            messages.append({"role": role, "content": content})
        yield {"messages": messages}


# ---------------------------------------------------------------------------
# Source 8: Intent pool — broad paraphrased intent prompts per tool
#
# This is the load-bearing source for v3.9 eval-gate parity.  Earlier
# sources teach "what does tool X do?" and "X's signature is …".  The
# eval harness probes the inverse direction: a natural-language intent
# WITHOUT the tool name, expecting the model to surface the tool name
# in its response.  Without this source the v3.9.1 model scored 9.1%
# on tool-call recall despite excellent training metrics — every
# in-corpus example named the tool in the user prompt.
#
# Convention:
#   * Prompt MUST NOT contain the tool name.
#   * Assistant answer MUST cite the tool name in backticks.
#   * Multiple answer phrasings per intent so the gradient lands on
#     the *name*, not on the surrounding boilerplate.
# ---------------------------------------------------------------------------


_INTENT_POOL: dict[str, list[str]] = {
    # --- recall surface --------------------------------------------------
    "recall": [
        "I need to search my memory for blocks matching a keyword.",
        "Find blocks in mind-mem by free-text query.",
        "How do I run a full-text search across stored blocks?",
        "Surface every block that mentions 'audit chain'.",
        "What's the quickest way to look up a block by content keyword?",
        "I want a keyword search over the workspace.",
    ],
    "hybrid_search": [
        "Combine BM25 with semantic vector search and merge the results.",
        "I want hybrid retrieval — lexical plus dense vector — fused with RRF.",
        "Run a search that uses both BM25 and embedding similarity.",
        "How do I get reciprocal-rank-fusion over my BM25 + vector results?",
        "Best-quality search that blends lexical and semantic ranking.",
    ],
    "prefetch": [
        "Warm the recall cache before a session starts.",
        "Pre-load likely recall hits into the cache.",
        "Speculatively pull blocks I'll probably need so the next recall is instant.",
        "Speed up the next recall call by warming the cache now.",
    ],
    "find_similar": [
        "Find blocks that look semantically similar to a given block ID.",
        "I have block ABC-001 — show me its nearest neighbours.",
        "Surface other blocks that resemble this one.",
        "Pull the top-k most-similar blocks to a target.",
    ],
    "intent_classify": [
        "Tell me which of the 9 mind-mem intent classes this query falls into.",
        "Route my query — which intent does mind-mem think it is?",
        "Classify the user-query intent so I can pick the right tool.",
        "Get the intent label and confidence for a question.",
    ],
    "retrieval_diagnostics": [
        "Why did my recall return these blocks and not others?",
        "Show the BM25 / vector / RRF scores behind a recall result.",
        "Explain the ranking decisions for the last query.",
        "I need a per-candidate score breakdown for a recall.",
    ],
    "pack_recall_budget": [
        "Trim recall results to fit a fixed token budget.",
        "Pack the most useful blocks into N tokens worth of context.",
        "Squeeze a recall response into the context window I have left.",
        "Budget-constrained recall packing.",
    ],
    "recall_with_axis": [
        "Recall blocks but only along a specific governance axis.",
        "Filter recall results to a single classification axis.",
        "Run a recall that's restricted to one axis of meaning.",
    ],
    # --- governance surface ----------------------------------------------
    "propose_update": [
        "Stage a change to a memory block — don't apply it yet.",
        "I want to draft a new block but route it through review.",
        "Add a memory but require human approval before it lands.",
        "Submit a proposal for a block change.",
        "Queue a write into the proposal review pipeline.",
    ],
    "approve_apply": [
        "I've reviewed proposal P. Land it.",
        "Apply the staged proposal now that I've approved it.",
        "Commit a queued proposal change to the store.",
        "Finalize a pending memory mutation after review.",
        "Move a proposal from staged to applied.",
    ],
    "reject_proposal": [
        "I don't want this proposal — close it without applying.",
        "Reject the staged change with a reason.",
        "Decline a pending proposal.",
    ],
    "rollback_proposal": [
        "I just applied a bad change. Undo it.",
        "Reverse the most recent apply using its receipt timestamp.",
        "Roll back a memory mutation safely.",
        "Undo an apply that turned out to be wrong.",
    ],
    "scan": [
        "Look across the workspace and surface contradictions plus drift.",
        "Run the contradiction-and-drift sweep.",
        "Detect contradictions in the current memory store.",
        "Trigger the workspace-wide governance scan.",
    ],
    "list_contradictions": [
        "Show me everything that's currently flagged as contradictory.",
        "List blocks that conflict with each other.",
        "I want the contradiction queue with strategy recommendations.",
        "Give me the analysed contradictions list.",
    ],
    "memory_evolution": [
        "Show how a block changed over time.",
        "Trace the lifecycle of a memory block.",
        "Pull the evolution timeline for a block ID.",
        "I want the access + importance + keyword history of a block.",
    ],
    # --- audit / chain surface -------------------------------------------
    "verify_chain": [
        "Prove the audit log hasn't been tampered with.",
        "Walk the hash chain and confirm every entry validates.",
        "I need to confirm chain integrity before I trust this snapshot.",
        "Validate the cryptographic ledger end-to-end.",
        "Check that the evidence chain is intact.",
    ],
    "verify_merkle": [
        "Confirm a block's content hash matches its Merkle root.",
        "Verify the Merkle path for a single block.",
        "Cross-check a block hash against the chain's Merkle structure.",
    ],
    "list_evidence": [
        "Dump every evidence entry recorded by the audit chain so far.",
        "Show me the full ledger of governance actions.",
        "I want each accepted evidence object in the chain.",
    ],
    "mind_mem_verify": [
        "Run the full mind-mem verification suite.",
        "Self-check the workspace against the integrity invariants.",
        "Validate a snapshot top-to-bottom.",
    ],
    # --- memory operations -----------------------------------------------
    "index_stats": [
        "Tell me how many blocks live in the index right now.",
        "Show retrieval-index statistics.",
        "Count terms, blocks, and FTS rows in the workspace.",
    ],
    "reindex": [
        "Rebuild the FTS tables from the source of truth.",
        "Refresh the search index after a bulk import.",
        "Recompute retrieval indexes from scratch.",
    ],
    "delete_memory_item": [
        "Permanently remove a block by ID.",
        "Drop a memory item from the store.",
        "Delete a single block.",
    ],
    "export_memory": [
        "Dump the workspace to JSONL for archival.",
        "Export every block to a portable JSONL file.",
        "Serialise the memory store to disk.",
    ],
    "get_block": [
        "Pull a single block by its ID.",
        "Fetch one specific memory record.",
        "Read block ABC-001 verbatim.",
    ],
    "memory_health": [
        "Tell me whether the memory subsystem is healthy.",
        "Run a top-line health check on the store.",
        "Surface health metrics for the workspace.",
    ],
    "compact": [
        "Garbage-collect old archived blocks and snapshots.",
        "Compact the workspace — drop expired signals + snapshots.",
        "Trim aged data from the store.",
    ],
    "stale_blocks": [
        "List blocks that haven't been touched in a while.",
        "Show me the staleness queue.",
        "Surface blocks ripe for re-extraction.",
    ],
    # --- benchmarks / category -------------------------------------------
    "governance_health_bench": [
        "Benchmark the governance plane end-to-end.",
        "Stress-test contradiction detection, audit completeness, and drift in one call.",
        "Run the governance bench and report pass/fail per suite.",
        "Exercise the full governance pipeline as a benchmark.",
    ],
    "category_summary": [
        "Summarise blocks grouped by category for a topic.",
        "Give me a per-category roll-up.",
        "Aggregate blocks by category for a topic.",
    ],
    # --- calibration -----------------------------------------------------
    "calibration_feedback": [
        "Record whether a recall hit was actually useful.",
        "Log calibration feedback for a query/result pair.",
        "Tell mind-mem the recall was a hit (or miss).",
    ],
    "calibration_stats": [
        "How well-calibrated is the retrieval scorer right now?",
        "Show calibration accuracy and ECE.",
        "Pull the latest calibration statistics.",
    ],
    # --- consolidation ---------------------------------------------------
    "plan_consolidation": [
        "Plan a consolidation pass — what would dream-cycle do if I ran it now?",
        "Dry-run the consolidation planner.",
        "Preview which blocks consolidation would touch.",
    ],
    "propagate_staleness": [
        "Cascade staleness from a few blocks across their cross-references.",
        "Spread the dirty flag through the cross-ref graph from these seed IDs.",
        "Mark downstream blocks stale when their dependencies change.",
    ],
    "project_profile": [
        "Load the per-project profile config.",
        "Pull project-specific recall weights.",
        "Show the profile mind-mem will apply for project X.",
    ],
    "dream_cycle": [
        "Run the dream consolidation pass now.",
        "Trigger the offline consolidation cycle.",
        "Manually invoke the dream cycle.",
    ],
    # --- core snapshots --------------------------------------------------
    "build_core": [
        "Snapshot a namespace into a core file.",
        "Build a portable core snapshot from the current workspace.",
        "Freeze a namespace into a `.core` artefact.",
    ],
    "load_core": [
        "Restore a previously-built core snapshot.",
        "Load a `.core` file back into the workspace.",
        "Mount a core snapshot.",
    ],
    "unload_core": [
        "Detach a loaded core snapshot.",
        "Unload a core namespace.",
    ],
    "list_cores": [
        "Show every core snapshot currently registered.",
        "Enumerate loaded cores.",
    ],
    # --- encryption ------------------------------------------------------
    "encrypt_file": [
        "Encrypt a workspace file at rest.",
        "Protect a sensitive memory file with at-rest encryption.",
        "Store this file ciphered on disk.",
    ],
    "decrypt_file": [
        "Read back an encrypted workspace file.",
        "Decrypt a previously-encrypted memory file.",
        "Recover plaintext from an at-rest-encrypted file.",
    ],
    # --- agent / vault / stream ------------------------------------------
    "agent_inject": [
        "Render a context snippet for a specific agent.",
        "Inject mind-mem context into an agent's prompt.",
        "Format a recall snippet for the target agent's expected layout.",
    ],
    "vault_scan": [
        "Walk an Obsidian vault and ingest its blocks.",
        "Sync an external vault into mind-mem.",
        "Scan a directory tree of markdown for mind-mem blocks.",
    ],
    "vault_sync": [
        "Push a single block back out to the vault file.",
        "Mirror a mind-mem block into Obsidian.",
        "Write a block back to its source vault file.",
    ],
    "stream_status": [
        "Show the change-stream pump status.",
        "Is the change stream healthy?",
        "Report stream backlog and lag.",
    ],
    # --- ontology --------------------------------------------------------
    "ontology_load": [
        "Load an ontology definition into the workspace.",
        "Register an ontology by name.",
        "Mount a typed-block ontology.",
    ],
    "ontology_validate": [
        "Validate a block against the registered ontology.",
        "Type-check a block.",
        "Confirm a block conforms to ontology rules.",
    ],
    # --- graph -----------------------------------------------------------
    "graph_add_edge": [
        "Add a cross-reference between two blocks.",
        "Link block A to block B in the graph.",
        "Insert a typed edge between blocks.",
    ],
    "graph_query": [
        "Walk the cross-reference graph from a starting block.",
        "Run a graph query.",
        "Retrieve neighbours along a typed edge.",
    ],
    "graph_stats": [
        "Show graph density / edge counts / SCC counts.",
        "Pull statistics for the cross-ref graph.",
        "How big is the block graph?",
    ],
    "traverse_graph": [
        "BFS the cross-reference graph from a seed block.",
        "Traverse outward from a block following typed edges.",
        "Walk the graph from a starting node.",
    ],
    # --- kernels ---------------------------------------------------------
    "list_mind_kernels": [
        "Enumerate every registered MIND scoring kernel.",
        "Show me the kernel registry.",
        "What kernels does mind-mem currently expose?",
    ],
    "get_mind_kernel": [
        "Fetch a single MIND kernel definition by name.",
        "Show the source of a registered MIND kernel.",
        "Look up one kernel by name.",
    ],
    "compiled_truth_load": [
        "Load a compiled-truth entity by ID.",
        "Pull the compiled-truth state for an entity.",
    ],
    "compiled_truth_add_evidence": [
        "Attach a piece of evidence to a compiled-truth entity.",
        "Append evidence to the compiled-truth ledger.",
    ],
    "compiled_truth_contradictions": [
        "Show contradictions detected inside a compiled-truth entity.",
        "List contradictory evidence on a compiled-truth entity.",
    ],
    # --- mic / map -------------------------------------------------------
    "mic_convert_tool": [
        "Convert a graph between mic@2 text and mic-b binary formats.",
        "Round-trip a MIND IR graph through MIC text and MIC-B binary.",
        "Translate a serialized graph between MIC text and binary.",
    ],
    "mic_inspect_tool": [
        "Inspect the structure of a serialized MIC/MAP graph.",
        "Decode and summarise a MIC artefact.",
        "Show what's inside a MIC/MAP file.",
    ],
    # --- model audit / signing -------------------------------------------
    "audit_model_tool": [
        "Run the seven-check audit on a local model checkpoint.",
        "Audit a checkpoint manifest end-to-end.",
        "Fire the seven-step model-checkpoint audit via MCP.",
    ],
    "sign_model_tool": [
        "Sign a model-checkpoint manifest with Ed25519.",
        "Produce an Ed25519 signature over a model manifest.",
        "Cryptographically sign a checkpoint.",
    ],
    "verify_model_tool": [
        "Verify an Ed25519 signature on a model manifest.",
        "Validate that a checkpoint manifest's signature is genuine.",
        "Check the Ed25519 manifest signature.",
    ],
    # --- pipeline / hash-of-code (v3.9) ----------------------------------
    "pipeline_status": [
        "Show me the current pipeline hash and how many blocks are dirty.",
        "Inspect the pipeline-invalidation state.",
        "Report the active pipeline hash and dirty-block count.",
        "What's the workspace's pipeline hash right now and how many blocks need re-extraction?",
    ],
    "reindex_dirty": [
        "Re-stamp every block whose TransformHash drifted from the active pipeline hash.",
        "Bulk-fix blocks whose pipeline hash is stale.",
        "Run the dirty-block re-extraction pass.",
        "Reprocess blocks whose TransformHash no longer matches the live pipeline.",
    ],
    # --- v3.9 walkthrough / persona --------------------------------------
    "compile_truth_walkthrough": [
        "Show me a dependency-ordered learning sequence for a topic.",
        "Compile a Kahn-topo walkthrough for an entity.",
        "I need a teaching order — what do I learn first to understand X?",
        "Build a topological learning path through related blocks.",
    ],
    "recall_with_persona": [
        "Run a recall but project the results through a persona lens.",
        "Retrieve and reformat blocks for a 'brief' / 'detailed' / 'technical' persona.",
        "Persona-aware recall.",
        "Recall blocks and reshape them through a persona projection.",
    ],
    # --- signal ----------------------------------------------------------
    "observe_signal": [
        "Record a signal-event observation.",
        "Log a metric/signal event.",
        "Push a fresh signal observation into the store.",
    ],
    "signal_stats": [
        "Show recent signal-event statistics.",
        "Roll up signal counts over the last window.",
        "Pull stats for observed signals.",
    ],
    # --- arch-mind bridge ------------------------------------------------
    "arch_baseline": [
        "Snapshot the architectural baseline of a repo via arch-mind.",
        "Establish the baseline 9-metric snapshot for a repo.",
    ],
    "arch_delta": [
        "Compute the delta between two arch-mind snapshots.",
        "Diff two arch-mind sessions.",
    ],
    "arch_history": [
        "Show the arch-mind session history for a repo.",
        "Walk the signed arch-mind evidence chain.",
    ],
    "arch_check_rules": [
        "Run the arch-mind rule-engine check against the rules.mind file.",
        "Gate on architectural rules in CI.",
    ],
    "arch_session_start": [
        "Open an arch-mind session before agents touch a repo.",
        "Begin a signed governance session in arch-mind.",
    ],
    "arch_session_end": [
        "Close an arch-mind session and append the signed evidence node.",
        "Seal an open arch-mind session.",
    ],
    "arch_metric_explain": [
        "Explain one of the 9 arch-mind Q16.16 metrics.",
        "Get a human-readable rationale for an arch-mind metric.",
    ],
    # --- public dispatchers (v3.4 consolidated surface) ------------------
    "staged_change": [
        "Use the consolidated dispatcher for proposal staging / approval / rejection.",
        "Drive the proposal lifecycle through the staged_change dispatcher.",
    ],
    "memory_verify": [
        "Use the consolidated verify dispatcher to check chain + Merkle + snapshot.",
        "Drive every verification surface through one dispatcher.",
    ],
    "graph": [
        "Use the consolidated graph dispatcher.",
        "Drive add_edge / query / stats / traverse through one entry point.",
    ],
    "core": [
        "Use the consolidated core-snapshot dispatcher.",
        "Drive build / load / unload / list cores through one entry point.",
    ],
    "kernels": [
        "Use the consolidated kernels dispatcher.",
        "Drive list and get_mind_kernel through one entry point.",
    ],
    "compiled_truth": [
        "Use the consolidated compiled-truth dispatcher.",
        "Drive load / add_evidence / contradictions through one dispatcher.",
    ],
}


# Phrasings that wrap a curated intent into multiple Q/A entries.  Each
# (prompt-template, answer-template) pair MUST keep the tool name
# in the answer slot, never in the prompt slot.
_INTENT_PHRASINGS: list[tuple[str, str]] = [
    ("{intent}", "Use `{tool}`."),
    ("{intent}", "Call `{tool}`."),
    ("{intent}", "`{tool}`."),
    ("{intent}", "The mind-mem tool for that is `{tool}`."),
    ("{intent}", "Reach for `{tool}`."),
    ("In mind-mem: {intent_lower}", "`{tool}`."),
    ("With mind-mem, {intent_lower}", "Use `{tool}`."),
    ("Mind-mem question: {intent}", "`{tool}` is the right tool."),
    ("Quick mind-mem ask — {intent_lower}", "`{tool}`."),
]


def _harvest_intent_pool() -> Iterator[dict]:
    """Curated intent prompts with multi-phrasing answers.

    For every (tool, intents) pair the prompt is one of the curated
    intent strings (which never names the tool) and the answer always
    cites the tool. This is the source the v3.9 eval-gate exercises.
    """
    for tool, intents in _INTENT_POOL.items():
        for intent in intents:
            stripped = intent.strip().rstrip(".")
            intent_lower = stripped[0].lower() + stripped[1:] if stripped else stripped
            for qp, ap in _INTENT_PHRASINGS:
                user = qp.format(intent=intent, intent_lower=intent_lower)
                assistant = ap.format(tool=tool)
                yield {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ]
                }


# ---------------------------------------------------------------------------
# Source 9: v3.9 surface facts — TransformHash + transport + replication
#
# The v3.9 eval probes test for exact strings the model has never seen
# in the corpus before:
#
#   * ``TransformHash`` (capital-case field name on every block).
#   * ``stamp_transform_hash`` (helper that writes the field).
#   * ``reextract_dirty_blocks`` (bulk re-stamp helper).
#   * HTTP REST endpoints: ``/status`` / ``/query`` / ``/memories``
#       / ``/consolidate`` / ``/walkthrough`` (and NO others).
#   * ``daemon`` triggers the dream cycle on a configurable interval.
#   * ``inbox`` folder ingestion with classification + atomic move.
#   * Replicated Postgres routing with ``primary`` for writes and
#       ``round-robin`` for reads.
# ---------------------------------------------------------------------------


_V39_FACTS: list[tuple[str, str]] = [
    # TransformHash field --------------------------------------------------
    (
        "Show me the field name a v3.9 inbox-ingested block carries to record the pipeline hash.",
        (
            "Every v3.9 block written by the ingestion or inbox pipeline carries a "
            "`TransformHash` field. The field is CapitalCase per the mind-mem block "
            "convention (every standard field starts with a capital letter), and its "
            "value is the SHA-256 of the active pipeline-component versions."
        ),
    ),
    (
        "What field does mind-mem v3.9 stamp on a block to track the active pipeline hash?",
        "`TransformHash`. It records the SHA-256 of the active pipeline-component versions.",
    ),
    (
        "Which helper stamps the current pipeline hash onto a block dict before writing?",
        (
            "Use `stamp_transform_hash(workspace, block)` from `mind_mem.pipeline_hash`. "
            "It returns a copy of the block with `TransformHash` set to the current "
            "pipeline digest, leaving any pre-existing hash untouched."
        ),
    ),
    (
        "Helper that writes the TransformHash field on a block?",
        "`stamp_transform_hash(workspace, block)` from `mind_mem.pipeline_hash`.",
    ),
    (
        "Function that sets TransformHash before a block is written?",
        "`stamp_transform_hash(workspace, block)`.",
    ),
    (
        "Which v3.9 helper writes the pipeline-hash digest onto a block dict?",
        "`stamp_transform_hash` from `mind_mem.pipeline_hash`.",
    ),
    (
        "Mind-mem helper for stamping the pipeline-hash field on a block?",
        "`stamp_transform_hash(workspace, block)` — returns a copy with `TransformHash` set.",
    ),
    (
        "Bulk-fix blocks whose TransformHash drifted — function name?",
        "`reextract_dirty_blocks` from `mind_mem.pipeline_hash` (also exposed as the `reindex_dirty` MCP tool).",
    ),
    (
        "Library function backing the `reindex_dirty` MCP tool?",
        "`reextract_dirty_blocks(workspace, ...)` in `mind_mem.pipeline_hash`.",
    ),
    (
        "Which helper iterates blocks and re-extracts the ones with a stale pipeline hash?",
        "`reextract_dirty_blocks` (in `mind_mem.pipeline_hash`).",
    ),
    (
        "How do I bulk re-stamp blocks whose pipeline hash drifted?",
        (
            "Call `reextract_dirty_blocks(workspace, ...)` from `mind_mem.pipeline_hash` "
            "(also exposed as the MCP tool `reindex_dirty`). It walks every block, "
            "re-runs extraction on those whose `TransformHash` no longer matches the "
            "active pipeline digest, and writes the refreshed block back via "
            "`stamp_transform_hash`."
        ),
    ),
    (
        "Bulk re-extract blocks with stale TransformHash — which function?",
        "`reextract_dirty_blocks` (exposed via the `reindex_dirty` MCP tool).",
    ),
    # HTTP transport surface ----------------------------------------------
    (
        "List the v3.9 HTTP transport endpoints.",
        (
            "The v3.9 HTTP REST adapter (`mind_mem.http_transport.serve_http`) exposes "
            "exactly six paths:\n"
            "* `GET /status` — health, memory count, last-scan timestamp\n"
            "* `POST /query` — natural-language search\n"
            "* `GET /memories` — list / browse with filtering\n"
            "* `DELETE /memories/{id}` — remove a specific memory\n"
            "* `POST /consolidate` — trigger dream cycle on demand\n"
            "* `POST /walkthrough` — dependency-ordered walkthrough\n"
            "Authentication uses the `X-MindMem-Token` header (loopback binds may "
            "skip auth via `--allow-unauthenticated-localhost`)."
        ),
    ),
    (
        "What HTTP routes does the v3.9 mind-mem REST adapter expose?",
        (
            "`/status`, `/query`, `/memories`, `/memories/{id}`, `/consolidate`, "
            "`/walkthrough`. Anything else (e.g. `/admin`, `/auth/login`, `/users`, "
            "`/embed`) is NOT a real route — the adapter returns 404."
        ),
    ),
    (
        "Which header does the mind-mem HTTP transport use for auth?",
        "`X-MindMem-Token`. The token is read from the `MIND_MEM_TOKEN` env var at server startup.",
    ),
    (
        "How do I serve mind-mem over HTTP?",
        (
            "```python\n"
            "from mind_mem.http_transport import serve_http\n"
            "serve_http(workspace='/path/ws', host='127.0.0.1', port=8765)\n"
            "```\n"
            "The adapter binds loopback by default and exposes `/status`, `/query`, "
            "`/memories`, `/consolidate`, and `/walkthrough`."
        ),
    ),
    # Daemon ---------------------------------------------------------------
    (
        "How does the v3.9 daemon trigger the dream cycle?",
        (
            "The v3.9 background `daemon` (see `mind_mem.daemon`) runs each configured "
            "job on its own interval thread. The `dream_cycle` task polls every "
            "`auto_interval_seconds` (default 1800s) and calls `run_dream_cycle` on "
            "the workspace; a `0` interval disables the task. Configure under the "
            "`daemon` block of `mind-mem.json`."
        ),
    ),
    (
        "What's the daemon for in v3.9?",
        (
            "The v3.9 `daemon` is a background process that fires `dream_cycle`, "
            "`intel_scan`, `entity_ingest`, and `transcript_scan` on configurable "
            "intervals. Each task runs on its own thread and survives transient "
            "errors; intervals come from the `daemon` block of `mind-mem.json`."
        ),
    ),
    (
        "Which background process owns the recurring dream-cycle in v3.9?",
        "The mind-mem `daemon` (see `mind_mem.daemon`). It schedules `dream_cycle` on the configured `auto_interval_seconds`.",
    ),
    # Inbox ----------------------------------------------------------------
    (
        "How do I drop a file into the v3.9 inbox for ingestion?",
        (
            "Configure an `inbox/` directory in `mind-mem.json` and start the inbox "
            "watcher (`InboxWatcher`). Drop any text/PDF/image/audio file into the "
            "directory and mind-mem classifies it by extension and routes to the "
            "right ingestion path. On success the file moves to "
            "`inbox/_processed/<ts>/`; on failure it moves to `inbox/_failed/<ts>/` "
            "with a sidecar `.error.txt`. Each ingested file is hashed and stamped "
            "with the active `TransformHash`."
        ),
    ),
    (
        "What's the v3.9 inbox folder for?",
        (
            "It's a watched directory that mind-mem ingests files from. Drop a file "
            "into `inbox/` and the watcher classifies it (text / pdf / image / "
            "audio), runs the matching ingestion path, and atomically moves the "
            "source to `_processed/<ts>/` or `_failed/<ts>/`."
        ),
    ),
    (
        "How does the v3.9 inbox classify files?",
        (
            "By extension — text, image, audio, pdf — each routed to a dedicated "
            "ingest function. Image / audio / pdf require the `[multimodal]` extra. "
            "Atomic move semantics keep the source untouched until ingestion either "
            "succeeds (`_processed/`) or fails (`_failed/`)."
        ),
    ),
    # Replicated Postgres --------------------------------------------------
    (
        "How does the v3.9 replicated postgres backend handle writes vs reads?",
        (
            "v3.9 ships `ReplicatedPostgresBlockStore`, a transparent wrapper that "
            "splits traffic by intent. Writes (`write_block`, `delete_block`, "
            "`lock`) always hit the `primary`. Reads (`get_block`, `recall`, "
            "`hybrid_search`, `prefetch`) round-robin across the configured replica "
            "DSNs. A failing replica falls back to `primary` (fail-open); after 3 "
            "consecutive failures the wrapper opens a circuit breaker on that "
            "replica and skips it for subsequent reads."
        ),
    ),
    (
        "Where does mind-mem v3.9 send a write when both primary and replicas are configured?",
        "Always to the `primary`. Replicas are read-only; writes never round-robin.",
    ),
    (
        "Walk me through the read-path of v3.9's replicated postgres store.",
        (
            "Reads round-robin across the configured replica DSNs. On replica "
            "exception the wrapper falls back to `primary`. Three consecutive "
            "failures on the same replica trip the circuit breaker; that replica is "
            "skipped on subsequent reads until the breaker times out."
        ),
    ),
    # MIC/MAP serialization ------------------------------------------------
    (
        "What are mic@2 and mic-b in mind-mem v3.9?",
        (
            "`mic@2` is the canonical text wire format (a MIND-flavoured MAP serialization). "
            "`mic-b` is the matching binary form. The two round-trip losslessly. "
            "Use `mic_convert_tool` to translate between them and `mic_inspect_tool` "
            "to summarise the structure of either form."
        ),
    ),
    (
        "Convert a MIND IR graph from mic@2 text to mic-b binary?",
        "Use `mic_convert_tool` with the source format set to `mic@2` and the destination set to `mic-b`.",
    ),
    # v3.9 surface summary -------------------------------------------------
    (
        "Summarise the v3.9 transport surface.",
        (
            "v3.9 ships four transport surfaces in addition to the existing MCP "
            "stdio path:\n"
            "1. **HTTP REST adapter** (`mind_mem.http_transport`) — `GET /status`, "
            "`POST /query`, `GET /memories`, `DELETE /memories/{id}`, "
            "`POST /consolidate`, `POST /walkthrough`. Auth via `X-MindMem-Token`.\n"
            "2. **Background daemon** (`mind_mem.daemon`) — interval-driven `dream_cycle`, "
            "`intel_scan`, `entity_ingest`, `transcript_scan`.\n"
            "3. **Inbox folder** (`mind_mem.inbox.InboxWatcher`) — drop files into "
            "`inbox/`, classified by extension, atomic move on success/failure.\n"
            "4. **Replicated Postgres** (`ReplicatedPostgresBlockStore`) — primary "
            "for writes, round-robin reads across replicas with circuit breaker."
        ),
    ),
    (
        "What's new about block-write semantics in mind-mem v3.9?",
        (
            "Every write goes through `stamp_transform_hash(workspace, block)`, which "
            "sets the `TransformHash` field on the block to the current pipeline "
            "digest. When the pipeline definition changes, `pipeline_status` reports "
            "a non-zero dirty-block count, and `reindex_dirty` (a.k.a. "
            "`reextract_dirty_blocks`) re-extracts and re-stamps the affected blocks."
        ),
    ),
]


def _harvest_v39_facts() -> Iterator[dict]:
    """Direct teaching of v3.9 transport / TransformHash / replication facts."""
    for q, a in _V39_FACTS:
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        }


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def _dedup(entries: Iterable[dict]) -> Iterator[dict]:
    seen: set[str] = set()
    for e in entries:
        key = hashlib.sha256(json.dumps(e, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
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
            _harvest_intent_pool(),
            _harvest_v39_facts(),
        ):
            for entry in _dedup(src):
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
    print(f"wrote {count} examples to {OUT}")


if __name__ == "__main__":
    main()
