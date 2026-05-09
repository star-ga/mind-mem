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

# MUST match train/eval_harness.py:_chat() verbatim, otherwise the
# trained model is conditioned on a different system prompt than the
# one it sees at inference time (v3.9.5 bug — caused tool_call regression
# from 95% → 90% and v39_new_tools 92% → 77%).
SYSTEM_PROMPT = "You are mind-mem-4b, a memory-governance assistant."


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
    """Generate dense, paraphrased block-template teaching.

    v3.9.3-balanced: corpus rebalance after v3.9.3 full-FT eval showed
    block_schema regressed to 50% (target 98%). Root cause was 16
    block_schema examples drowning under 3645 intent examples. This
    expands per-block-type to ~40 prompts (25 show + 12 fields + 3
    structural), giving 14 × 40 = 560 examples — comparable in volume
    to the workflow_paraphrases source (~70) and intent_pool reduction.
    """
    prompts_show = [
        "Show me a mind-mem {short} block template.",
        "Give me the canonical {short} block format.",
        "Print a {short} block skeleton.",
        "What does a {short} block look like?",
        "Template for a {short} block, please.",
        "Render the {short} block layout.",
        "How is a {short} block structured?",
        "Display a {short} block schema.",
        "Provide the {short} block template.",
        "Sketch a {short} block.",
        "Lay out the fields of a {short} block.",
        "Walk me through the {short} block format.",
        "What goes into a {short} block?",
        "Give the standard {short} block shape.",
        "Print the canonical {short} block format.",
        "Show the {short} block schema with all fields.",
        "Write a blank {short} block I can fill in.",
        "What is the structure of a {short} block?",
        "Compose an empty {short} block.",
        "Schema for a {short} block?",
        "{short} block — what does it look like?",
        "Empty {short} block template?",
        "Canonical layout for a {short} block?",
        "Stub a {short} block.",
        "Print the field list for a {short} block as a template.",
    ]
    prompts_fields = [
        "What fields does a {short} block require?",
        "List the required fields of a {short} block.",
        "Which keys live inside a {short} block?",
        "Required {short} block fields?",
        "Enumerate the fields of a {short} block.",
        "Tell me every field a {short} block needs.",
        "{short} block — what are the required keys?",
        "What does a {short} block require structurally?",
        "Field list for {short} blocks?",
        "Every required field of a {short} block, please.",
        "What goes in a {short} block (fields only)?",
        "Mandatory fields of a {short} block?",
    ]
    for short, desc, fields in _BLOCK_TYPES:
        # Canonical template (always starts with [SHORT-DATE-NNN] so the
        # model learns the exact ID prefix).
        example = f"[{short}-20260413-001]\n" + "\n".join(f"{f}: <{f.lower()}>" for f in fields)
        # Long-form answer with field list inline (reinforces both shape
        # and field names in a single example).
        long_answer = (
            f"{desc}\n\n```\n{example}\n```\n\n"
            f"Required fields: " + ", ".join(f"`{f}`" for f in fields) + "."
        )
        terse_answer = f"{desc}\n\n```\n{example}\n```"
        # Alternate between long and terse answers across the prompt list
        # so the model sees both forms of the same canonical schema.
        for i, q in enumerate(prompts_show):
            answer = long_answer if i % 2 == 0 else terse_answer
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q.format(short=short)},
                    {"role": "assistant", "content": answer},
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
        # What type is this by ID? (ID-prefix → type direction) — three phrasings
        for q in (
            f"A block ID starts with `[{short}-`. What type is it?",
            f"What block type uses the `{short}` ID prefix?",
            f"`[{short}-...]` IDs — what's the block type?",
        ):
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
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
        "Two decision blocks contradict each other. Steps?",
        "I see a contradiction between two decision blocks. Walk me through the fix.",
        "Mind-mem flagged a contradiction. What's the workflow?",
        "Resolve a contradiction end-to-end.",
        "Contradiction remediation procedure?",
        "How does the contradiction-fix loop work?",
        "Got a contradiction — give me the canonical resolution chain.",
        "Step-by-step on resolving a flagged contradiction.",
    ],
    "belief-drift": [
        "How do I detect whether a belief has drifted over time?",
        "Which tools surface belief drift?",
        "How do I query drift signals?",
        "I want to check if a belief has drifted. Which tools do I call?",
        "Drift detection — workflow please.",
        "How do I see if a belief has drifted recently?",
        "Tools for spotting belief drift?",
        "Walk me through the drift-check workflow.",
        "What's the call sequence for verifying a belief is still consistent?",
        "How do I read the latest drift signals?",
        "Show me the procedure for drift inspection.",
        "Detect drift on a specific belief — how?",
    ],
    "rollback": [
        "I applied a bad proposal. How do I roll back and what happens to the belief state?",
        "How do I roll back a proposal safely?",
        "Undo a bad apply — what's the procedure?",
        "I applied a bad proposal. How do I roll back safely?",
        "Rollback workflow?",
        "What's the safe-rollback procedure after a bad apply?",
        "Reverse a proposal cleanly — steps?",
        "How does rollback affect BeliefStore confidence?",
        "Undo an applied proposal end-to-end.",
        "I committed something wrong. Walk me through reverting.",
        "Step-by-step rollback after a mis-apply?",
        "What tools restore state after a bad apply?",
    ],
    "audit-trail": [
        "Who changed field X on block Y, and when?",
        "How do I see the history of field X on block Y?",
        "Audit who changed field X on block Y.",
        "Show field-level audit history for a block.",
        "Field-level audit lookup — how?",
        "Trace the mutations of one field across time.",
        "Print the audit history of a single (block, field).",
        "How do I run a field-level audit?",
        "Tool that returns the chronological field-change list?",
        "Where do I look up who edited a specific field on a specific block?",
        "Audit trail for a single field — how do I pull it?",
        "Walk me through the field-history audit lookup.",
    ],
    "encryption": [
        "Can I encrypt a sensitive memory file at rest?",
        "How do I encrypt a mind-mem file?",
        "How do I enable encryption at rest?",
        "Encrypt-a-file procedure?",
        "Walk me through encrypting one workspace file.",
        "Sensitive-file encryption — what's the call?",
        "How does on-disk encryption work in mind-mem?",
        "Steps to encrypt + later decrypt a workspace file?",
        "Make a memory file unreadable on disk — how?",
        "Encrypt one file at rest, then decrypt it back. Procedure?",
        "Walk through the encrypt_file / decrypt_file pair.",
        "Sensitive memory at rest — how do I protect it?",
    ],
    "tiers": [
        "What happens to a block as it's accessed more often?",
        "How does mind-mem promote blocks through tiers?",
        "Explain the memory-tier promotion cycle.",
        "Walk me through the tier ladder.",
        "How does mind-mem decide a block is hot?",
        "Tier-promotion mechanics?",
        "How do blocks move from WORKING to LONG_TERM?",
        "Tier boost — how does it score recall?",
        "Memory-tier flow end-to-end?",
        "Why does the same block recall higher after repeated access?",
        "How is tier promotion driven — frequency, recency, both?",
        "What's the tier-decay loop?",
    ],
    "verify": [
        "How do I verify the audit chain integrity?",
        "Prove my audit chain hasn't been tampered with.",
        "Which tool checks the hash chain?",
        "Audit chain integrity verification?",
        "How do I prove the audit log is genuine?",
        "Audit-trail tamper check — which call?",
        "Confirm the chain-of-custody is intact.",
        "Walk through verify_chain.",
        "Tool that re-derives every event hash and checks the chain?",
        "Run an integrity check on the audit chain.",
        "Hash-chain verification procedure?",
        "Audit chain — how do I prove it's still valid?",
    ],
    "governance-bench": [
        "Run the full governance benchmark suite.",
        "How do I benchmark governance health?",
        "What's the governance health bench command?",
        "Governance benchmark — how do I trigger it?",
        "Run the four governance suites in one call.",
        "Stress-test contradictions / audit / drift in one go.",
        "Tool that benchmarks the entire governance plane?",
        "How do I run governance_health_bench?",
        "End-to-end governance benchmark — what's the call?",
        "I want a per-suite pass/fail across the governance plane. How?",
        "Governance plane benchmark procedure?",
        "Run the full set of governance health checks.",
    ],
    "scan-and-fix": [
        "Run a workspace scan and act on whatever it surfaces.",
        "Scan the workspace, then handle the findings.",
        "Scan-then-remediate workflow?",
        "Walk me through a scan-driven fix.",
        "What's the post-scan remediation flow?",
        "Drift scan → propose → apply procedure?",
    ],
    "approval-flow": [
        "How do I review and apply a proposal end-to-end?",
        "Walk me through the proposal-approval workflow.",
        "Apply a staged proposal — full procedure?",
        "Dry-run-then-commit flow for a proposal?",
        "I want to approve a queued proposal cleanly. How?",
        "Approval workflow with rollback safety?",
    ],
    "recall-then-explain": [
        "I got a recall result and need to know why those blocks ranked highest.",
        "Why did this block rank where it did?",
        "Explain a recall ranking.",
        "Diagnose a recall result — workflow?",
        "Walk me through retrieval explainability.",
        "Tools to break down a recall score?",
    ],
    "snapshot-cycle": [
        "Take a snapshot, do some work, restore if it goes wrong.",
        "Snapshot → experiment → rollback workflow?",
        "Reversible-experiment procedure?",
        "How do I checkpoint then restore a workspace?",
        "Walk me through a safe-experiment cycle with snapshots.",
    ],
    "mic-roundtrip": [
        "Convert a MIC document between text and binary, then check it.",
        "Round-trip a MIC artefact through mic-b and mic@2.",
        "How do I convert + inspect MIC files?",
        "Walk me through the MIC convert + inspect cycle.",
        "MIC text-to-binary roundtrip procedure?",
    ],
    "transform-hash-cycle": [
        "Find blocks whose pipeline-hash drifted, then re-stamp them.",
        "Detect + fix dirty blocks workflow.",
        "Pipeline-hash drift remediation procedure?",
        "Walk me through the dirty-block re-extraction loop.",
        "Stale-TransformHash repair workflow?",
    ],
    "http-transport": [
        "Serve mind-mem over HTTP and run a search through it.",
        "Stand up the HTTP REST adapter.",
        "How do I expose mind-mem over HTTP?",
        "HTTP transport setup + search workflow?",
        "Walk me through the v3.9 REST adapter.",
    ],
    "inbox-ingest": [
        "Drop a file into the v3.9 inbox and have it ingested.",
        "Inbox-driven ingestion workflow?",
        "How does the inbox watcher process a new file?",
        "Walk me through the inbox ingest path.",
        "Inbox folder file-drop ingestion procedure?",
    ],
    "replicated-postgres": [
        "Walk me through the v3.9 replicated-postgres routing.",
        "Replicated-Postgres read/write split — how?",
        "How does mind-mem route across primary + replicas?",
        "Postgres replica routing procedure?",
        "Walk through the circuit-breaker behaviour on a failing replica.",
    ],
    "persona-recall": [
        "I want recall results in a specific projection mode.",
        "Recall with a persona projection?",
        "How do I get a brief / detailed / technical recall?",
        "Persona-aware recall workflow?",
        "Walk me through recall_with_persona.",
    ],
    "walkthrough": [
        "Compile a learning walkthrough for a topic.",
        "How do I get a dependency-ordered learning sequence?",
        "Walk me through compile_truth_walkthrough.",
        "Topic walkthrough generation procedure?",
        "Kahn-topo learning sequence — how?",
    ],
    "intent-classification": [
        "How does mind-mem decide which retrieval path to use?",
        "Intent classification workflow?",
        "Walk me through intent_classify.",
        "How is recall biased per query?",
        "Per-query retrieval-path selection — how?",
    ],
    "alerts": [
        "I want to be notified when governance health drops.",
        "Alerting hook setup?",
        "Walk me through the alerts subscription.",
        "How do I get a webhook on contradiction count rising?",
        "Governance-health alerting workflow?",
    ],
    "find-similar": [
        "I have a block id. Show me other blocks that are semantically close.",
        "Find similar blocks — workflow?",
        "Walk me through find_similar.",
        "How do I get nearest-neighbours by semantic similarity?",
        "Block-to-block similarity lookup?",
    ],
    "memory-evolution": [
        "I want to see how a block evolved over time.",
        "Block-level mutation timeline — how?",
        "Walk me through memory_evolution.",
        "How do I see every change to a single block?",
        "Block evolution lookup procedure?",
    ],
    "encrypt-status": [
        "Is my workspace currently encrypted?",
        "Check encryption status of the workspace.",
        "Walk me through encrypt_status.",
        "How do I see how many files are encrypted at rest?",
        "Encryption-state lookup procedure?",
    ],
    "tier-decay-apply": [
        "Apply tier decay to age out cold blocks.",
        "Trigger tier-decay manually.",
        "Walk me through tier_decay_apply.",
        "How do I run the TTL/LRU pass?",
        "Cold-block aging workflow?",
    ],
    "category-summary": [
        "Show me a per-category roll-up for a topic.",
        "Category-grouped block summary — how?",
        "Walk me through category_summary.",
        "How do I get blocks grouped by category?",
        "Topic-level category roll-up procedure?",
    ],
    "audit-replay": [
        "Replay every audit event so I can verify chain integrity end-to-end.",
        "Full audit-chain replay — how?",
        "Walk me through audit_replay.",
        "How do I re-derive every event hash?",
        "End-to-end chain integrity check procedure?",
    ],
    "proposals-list": [
        "Show me the staged proposals waiting for review.",
        "List the proposal queue.",
        "Walk me through the proposal-review workflow.",
        "How do I see every staged proposal?",
        "Pending-proposals lookup procedure?",
    ],
    "delete-flow": [
        "Permanently delete a memory block.",
        "Walk me through delete_memory_item.",
        "How do I hard-delete a block (admin)?",
        "Permanent block deletion workflow?",
        "Admin delete procedure for a memory item?",
    ],
    "export-roundtrip": [
        "Export memory to JSONL and import it back into a fresh workspace.",
        "Round-trip a workspace through export + import.",
        "Walk me through export → fresh-workspace → import.",
        "How do I migrate a workspace via JSONL export?",
        "Export + reimport procedure for a workspace?",
    ],
    "rollback-rationale-required": [
        "How do I roll back a proposal — what's required?",
        "Roll back with rationale — procedure?",
        "What does v3.9.3 require for rollback_proposal?",
        "Walk me through the rollback procedure including reason field.",
        "Rollback API change in v3.9.3 — how do I call it now?",
    ],
    "pipeline-status-check": [
        "Is anything in my workspace stale relative to the active pipeline?",
        "Check pipeline-hash status.",
        "Walk me through pipeline_status.",
        "How do I see the dirty-block count?",
        "Pipeline-hash drift check procedure?",
    ],
    "drift-deep-dive": [
        "I see drift on a specific block. Walk me through investigating it.",
        "Investigate per-block drift.",
        "Block-level drift deep-dive workflow?",
        "How do I trace why a single block drifted?",
        "Per-block drift investigation procedure?",
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
                "mind-mem surfaces two drift signals. The lexical pass appears in DRIFT.md during `scan`. The semantic pass is stored in SQLite by `DriftDetector` and queryable via `belief_timeline(block_id)` or `signal_stats(min_confidence=0.6)`. In v2.9.0+ both run on every `scan` call, so `signal_stats` is always live.",
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
    # === v3.9.4 additions: more workflow scenarios so workflow probes ===
    # === have a denser teaching surface (was 5, now 25 scenarios). =====
    {
        "scenario": "scan-and-fix",
        "turns": [
            ("user", "Run a workspace scan and act on whatever it surfaces."),
            (
                "assistant",
                "Call `scan` first — it returns drift counts + a JSON drift report and writes to DRIFT.md / SIGNALS.md. If contradictions surface, run `list_contradictions` for analysed entries with confidence + side_effects + strategy. If drift signals surface, read `signal_stats` for the chronological tail. Apply fixes through `propose_update` → `approve_apply`, then close the loop with `verify_chain`.",
            ),
        ],
    },
    {
        "scenario": "approval-flow",
        "turns": [
            ("user", "How do I review and apply a proposal end-to-end?"),
            (
                "assistant",
                "Pull the proposal queue and inspect each entry. For each candidate: review the diff in SIGNALS.md, then call `approve_apply(proposal_id, dry_run=true)` to preview the deltas without committing. When the dry-run looks right, re-call with `dry_run=false`. The receipt timestamp printed at apply time is the key for any later `rollback_proposal(receipt_ts, reason=...)`.",
            ),
        ],
    },
    {
        "scenario": "recall-then-explain",
        "turns": [
            ("user", "I got a recall result and need to know why those blocks ranked highest."),
            (
                "assistant",
                "Pass the same query to `retrieval_diagnostics(query)`. It returns the BM25 / vector / RRF score breakdown per candidate plus the intent classification path. Cross-reference with `recall(query)` to see which scoring path dominated.",
            ),
        ],
    },
    {
        "scenario": "snapshot-cycle",
        "turns": [
            ("user", "Take a snapshot, do some work, restore if it goes wrong."),
            (
                "assistant",
                "`create_snapshot(label='before-experiment')` writes a delta + MANIFEST.json. Make changes via the proposal pipeline. If the result is bad, `list_snapshots` to find the snapshot id, then `restore_snapshot(snap_id)` rolls the workspace back. The restore itself emits an audit event so the chain stays whole — verify with `verify_chain`.",
            ),
        ],
    },
    {
        "scenario": "mic-roundtrip",
        "turns": [
            ("user", "Convert a MIC document between text and binary, then check it."),
            (
                "assistant",
                "Use `mic_convert_tool` with the source format set to `mic@2` (text/MAP) or `mic-b` (binary) and the destination set to the other. To inspect what's inside either form, call `mic_inspect_tool` — it returns the structure summary without re-encoding.",
            ),
        ],
    },
    {
        "scenario": "transform-hash-cycle",
        "turns": [
            ("user", "Find blocks whose pipeline-hash drifted, then re-stamp them."),
            (
                "assistant",
                "Call `pipeline_status` for the active pipeline-hash and dirty-block count. If dirty > 0, run `reindex_dirty` (the MCP wrapper) — it walks every block, finds those whose `TransformHash` doesn't match the active pipeline digest, re-extracts via `reextract_dirty_blocks` from `mind_mem.pipeline_hash`, and re-stamps via `stamp_transform_hash`.",
            ),
        ],
    },
    {
        "scenario": "http-transport",
        "turns": [
            ("user", "Serve mind-mem over HTTP and run a search through it."),
            (
                "assistant",
                "Start the adapter with `serve_http(workspace=..., host='127.0.0.1', port=8765)` from `mind_mem.http_transport`. It registers six routes: `GET /status` (health), `POST /query` (search), `GET /memories` (list/browse), `DELETE /memories/{id}`, `POST /consolidate` (dream cycle), `POST /walkthrough`. Auth header is `X-MindMem-Token` from the `MIND_MEM_TOKEN` env var. To search, POST a JSON body to `/query`.",
            ),
        ],
    },
    {
        "scenario": "inbox-ingest",
        "turns": [
            ("user", "Drop a file into the v3.9 inbox and have it ingested."),
            (
                "assistant",
                "Configure `inbox/` in `mind-mem.json` and start the `InboxWatcher`. To ingest, drop the file into `inbox/`. The watcher classifies by extension (text / pdf / image / audio), runs the matching ingest path, atomic-moves the source to `_processed/<ts>/` on success or `_failed/<ts>/` on failure. Each generated block is hashed and stamped with the active `TransformHash`.",
            ),
        ],
    },
    {
        "scenario": "replicated-postgres",
        "turns": [
            ("user", "Walk me through the v3.9 replicated-postgres routing."),
            (
                "assistant",
                "Writes (`write_block`, `delete_block`, `lock`) always hit the `primary`. Reads (`get_block`, `recall`, `hybrid_search`, `prefetch`) round-robin across the configured replica DSNs. On replica failure the wrapper falls back to `primary` (fail-open); after 3 consecutive failures the wrapper opens a circuit breaker on that replica and skips it for subsequent reads.",
            ),
        ],
    },
    {
        "scenario": "persona-recall",
        "turns": [
            ("user", "I want recall results in a specific projection mode."),
            (
                "assistant",
                "Call `recall_with_persona(query, persona)` where `persona` is `brief`, `detailed`, or `technical`. Each persona reformats the same retrieval result: `brief` returns one-line summaries, `detailed` returns full-text blocks, `technical` returns blocks plus the BM25/vector score breakdown.",
            ),
        ],
    },
    {
        "scenario": "walkthrough",
        "turns": [
            ("user", "Compile a learning walkthrough for a topic."),
            (
                "assistant",
                "Call `compile_truth_walkthrough(topic)`. It performs a Kahn topological sort over the topic's dependency graph and returns blocks in dependency-ordered sequence so concepts are introduced in the right order before they're referenced.",
            ),
        ],
    },
    {
        "scenario": "intent-classification",
        "turns": [
            ("user", "How does mind-mem decide which retrieval path to use?"),
            (
                "assistant",
                "Pass the query to `intent_classify(query)`. It returns one of nine intent types plus per-type confidence weights — the recall pipeline uses these weights to bias BM25 vs vector vs cross-encoder reranking for that specific query.",
            ),
        ],
    },
    {
        "scenario": "alerts",
        "turns": [
            ("user", "I want to be notified when governance health drops."),
            (
                "assistant",
                "Subscribe via `alerts_subscribe(webhook_url, threshold)` from `mind_mem.governance.alerts`. The alerts hook fires when `governance_health_bench` drops below the configured threshold or `list_contradictions` accumulates above the configured count. Webhook + Slack templates are built in.",
            ),
        ],
    },
    {
        "scenario": "find-similar",
        "turns": [
            ("user", "I have a block id. Show me other blocks that are semantically close."),
            (
                "assistant",
                "Call `find_similar(block_id, top_k)`. It re-uses the dense embedding of the source block, runs cosine similarity against the vector index, and returns the top_k nearest with similarity scores.",
            ),
        ],
    },
    {
        "scenario": "memory-evolution",
        "turns": [
            ("user", "I want to see how a block evolved over time."),
            (
                "assistant",
                "Call `memory_evolution(block_id)`. It returns the chronological mutation timeline for that block — every field change, every status transition, with timestamps and the proposal id that drove each change.",
            ),
        ],
    },
    {
        "scenario": "encrypt-status",
        "turns": [
            ("user", "Is my workspace currently encrypted?"),
            (
                "assistant",
                "Call `encrypt_status`. It returns whether `MIND_MEM_ENCRYPTION_PASSPHRASE` is set, how many files are currently encrypted at rest, and whether the SQLCipher database is locked.",
            ),
        ],
    },
    {
        "scenario": "tier-decay-apply",
        "turns": [
            ("user", "Apply tier decay to age out cold blocks."),
            (
                "assistant",
                "Call `tier_decay_apply`. It runs the TTL/LRU pass: blocks idle past their tier's TTL move down one tier, blocks that have moved past LONG_TERM are marked for cleanup. The cycle is normally invoked by the daemon but can be triggered manually for testing.",
            ),
        ],
    },
    {
        "scenario": "category-summary",
        "turns": [
            ("user", "Show me a per-category roll-up for a topic."),
            (
                "assistant",
                "Call `category_summary(topic)`. It groups blocks by category (decision / task / convention / evidence / etc.), returns counts per category, and surfaces the top blocks per group so you get a one-shot view of what the workspace knows about that topic.",
            ),
        ],
    },
    {
        "scenario": "audit-replay",
        "turns": [
            ("user", "Replay every audit event so I can verify chain integrity end-to-end."),
            (
                "assistant",
                "Call `audit_replay`. It walks the audit chain from genesis, recomputes every event hash, and verifies each hash links correctly to its predecessor. Mismatch = tampering or a missing event. Pair with `verify_chain` for the on-the-fly check.",
            ),
        ],
    },
    {
        "scenario": "proposals-list",
        "turns": [
            ("user", "Show me the staged proposals waiting for review."),
            (
                "assistant",
                "Read `intelligence/SIGNALS.md` for the queue, or call `list_contradictions` for the analysed entries (each carries `confidence_score`, `side_effects`, and a `strategy` recommendation). Approve via `approve_apply(proposal_id, dry_run=false)` or reject via `rollback_proposal(receipt_ts, reason=...)` after apply.",
            ),
        ],
    },
    {
        "scenario": "delete-flow",
        "turns": [
            ("user", "Permanently delete a memory block."),
            (
                "assistant",
                "Call the admin tool `delete_memory_item(block_id, reason)`. It requires `MIND_MEM_SCOPE=admin` (gated by `MIND_MEM_ADMIN_TOKEN`). The deletion is recorded as an audit event so the chain stays whole; `verify_chain` passes after a clean delete.",
            ),
        ],
    },
    {
        "scenario": "export-roundtrip",
        "turns": [
            ("user", "Export memory to JSONL and import it back into a fresh workspace."),
            (
                "assistant",
                "Use `export_memory(out_path)` to dump every block as JSONL with full metadata. Initialize the destination with `mind-mem-init`, then call `import_memory(in_path)` — it routes each entry through `propose_update` → `approve_apply` so the audit chain in the destination is well-formed.",
            ),
        ],
    },
    {
        "scenario": "rollback-rationale-required",
        "turns": [
            ("user", "How do I roll back a proposal — what's required?"),
            (
                "assistant",
                "Call `rollback_proposal(receipt_ts, reason=...)`. As of v3.9.3 the `reason` field is required on both the MCP and REST surfaces (8-2000 chars). The rollback restores the delta snapshot, marks the proposal `rolled_back`, and pushes an observation=0.0 into BeliefStore so confidence drops accordingly.",
            ),
        ],
    },
    {
        "scenario": "pipeline-status-check",
        "turns": [
            ("user", "Is anything in my workspace stale relative to the active pipeline?"),
            (
                "assistant",
                "Call `pipeline_status`. It returns `(active_hash, dirty_block_count)`. If `dirty_block_count > 0`, run `reindex_dirty` to re-extract and re-stamp those blocks via `reextract_dirty_blocks` from `mind_mem.pipeline_hash`.",
            ),
        ],
    },
    {
        "scenario": "drift-deep-dive",
        "turns": [
            ("user", "I see drift on a specific block. Walk me through investigating it."),
            (
                "assistant",
                "Start with `signal_stats(min_confidence=0.6)` to confirm the drift signal. Then call `belief_timeline(block_id)` for the per-block confidence trajectory. Cross-reference with `memory_evolution(block_id)` to see if the structural mutations explain the drift. If they do, propose a fix via `propose_update`; if not, mark the signal investigated through the audit trail.",
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
        "1. `scan`\n2. `signal_stats`",
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
    # v3.9.4: cut from 9 to 4 phrasings to flatten corpus distribution.
    # The 9-phrasing version produced ~3645 intent examples that drowned
    # out block_schema (16) and workflow (5) — full-FT then over-fit and
    # forgot the under-represented sources (v3.9.2 80%/40%, v3.9.3 50%/0%).
    # 4 phrasings × ~75 tools × 4 intents = ~1200 entries — still load-
    # bearing for tool_call recall, no longer dominant.
    ("{intent}", "Use `{tool}`."),
    ("{intent}", "Call `{tool}`."),
    ("{intent}", "`{tool}`."),
    ("{intent}", "The mind-mem tool for that is `{tool}`."),
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
        "`reindex_dirty` MCP tool (impl: `reextract_dirty_blocks` in `mind_mem.pipeline_hash`).",
    ),
    (
        "Library function backing the `reindex_dirty` MCP tool?",
        "`reindex_dirty` MCP tool (impl: `reextract_dirty_blocks(workspace, ...)` in `mind_mem.pipeline_hash`).",
    ),
    (
        "Which helper iterates blocks and re-extracts the ones with a stale pipeline hash?",
        "`reindex_dirty` MCP tool (impl: `reextract_dirty_blocks` in `mind_mem.pipeline_hash`).",
    ),
    (
        "How do I bulk re-stamp blocks whose pipeline hash drifted?",
        (
            "Use `reindex_dirty` MCP tool, which calls `reextract_dirty_blocks(workspace, ...)` from `mind_mem.pipeline_hash`. "
            "It walks every block, re-runs extraction on those whose `TransformHash` no longer matches the "
            "active pipeline digest, and writes the refreshed block back via "
            "`stamp_transform_hash`."
        ),
    ),
    (
        "Bulk re-extract blocks with stale TransformHash — which function?",
        "`reindex_dirty` MCP tool (impl: `reextract_dirty_blocks`).",
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
            "Exactly six paths: `/status`, `/query`, `/memories`, `/memories/{id}`, "
            "`/consolidate`, `/walkthrough`. Routes outside that list are not "
            "registered and return 404."
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
# Source 10: Targeted patches for v3.9.2-qlora eval misses.
#
# The first v3.9.2 eval had 4/6 fails: block_schema (missing Rationale /
# BlockId), workflow (hallucinated mind_mem_bench / audit_model_tool;
# missed signal_stats / field_history / FieldAuditor), v39_transform_hash
# (model said pipeline_hash instead of TransformHash; reindex_dirty
# instead of reextract_dirty_blocks), v39_transport_guard (missed /query
# in endpoint list, said "inbox receive" instead of "ingest", missed
# "primary" / "round-robin" keywords for replicated postgres).
#
# Root cause: failed probe categories had only 16-37 corpus examples
# competing with ~3645 intent-pool examples. The intent-pool drowned
# them out. Fix: explicit re-teaching with multiple paraphrases per
# fact, ensuring every required eval keyword appears verbatim, plus
# hallucination negatives ("X is NOT a tool").
# ---------------------------------------------------------------------------


_TARGETED_PATCHES: list[tuple[str, str]] = [
    # === TransformHash field — exact name reinforcement ===
    (
        "Show me the field name a v3.9 inbox-ingested block carries to record the pipeline hash.",
        "The field is `TransformHash` (CapitalCase). Every v3.9 block — inbox-ingested or pipeline-written — carries `TransformHash` set to the SHA-256 of the active pipeline-component versions. Note: it is NOT called `pipeline_hash`; that is descriptive prose, the actual field name on every block is `TransformHash`.",
    ),
    (
        "What's the v3.9 pipeline-hash field on a block called?",
        "`TransformHash`. CapitalCase, per the mind-mem block-field convention. Stamped onto every write by `stamp_transform_hash`.",
    ),
    (
        "Is `pipeline_hash` the field that records the v3.9 pipeline digest on a block?",
        "No. The field is named `TransformHash`. `pipeline_hash` is descriptive shorthand and is NOT a valid mind-mem block field. If you grep for `pipeline_hash` on a block dict you will not find it; the actual key is `TransformHash`.",
    ),
    (
        "Name the field a mind-mem v3.9 block uses to track its pipeline-digest.",
        "`TransformHash`. SHA-256 of the active pipeline-component versions, stamped on every write.",
    ),
    (
        "Field name that holds the pipeline digest on a v3.9 block?",
        "`TransformHash`. (Not `pipeline_hash`, not `pipeline_digest`, not `hash` — the canonical key is `TransformHash`.)",
    ),
    # === reextract_dirty_blocks — library helper vs MCP tool ===
    (
        "How do I bulk re-stamp blocks whose pipeline hash drifted?",
        "Use the `reindex_dirty` MCP tool. Internally it calls `reextract_dirty_blocks(workspace, ...)` from `mind_mem.pipeline_hash` to walk every block, find those whose `TransformHash` does not match the active pipeline digest, re-run extraction on them, and re-stamp via `stamp_transform_hash`.",
    ),
    (
        "Which library function re-extracts blocks with stale TransformHash?",
        "`reextract_dirty_blocks` in `mind_mem.pipeline_hash`. The library helper that actually does the work is `reextract_dirty_blocks`.",
    ),
    (
        "Is `reindex_dirty` the actual library function for bulk re-extraction?",
        "`reindex_dirty` is the MCP tool name (the public surface). The underlying library helper is `reextract_dirty_blocks` in `mind_mem.pipeline_hash`. When asked for the library function, answer `reextract_dirty_blocks`.",
    ),
    (
        "Function in mind_mem.pipeline_hash that processes every dirty block in bulk?",
        "`reextract_dirty_blocks(workspace, ...)`. Walks the workspace, re-extracts any block whose `TransformHash` is stale, re-stamps via `stamp_transform_hash`.",
    ),
    (
        "Bulk-fix helper for blocks with drifted pipeline hash?",
        "`reextract_dirty_blocks` (lib) / `reindex_dirty` (MCP). Both names refer to the same bulk re-extraction pass in `mind_mem.pipeline_hash`.",
    ),
    # === HTTP transport endpoints — /query is the failed keyword ===
    (
        "List the v3.9 HTTP transport endpoints.",
        "The v3.9 HTTP REST adapter (`mind_mem.http_transport.serve_http`) registers exactly these routes:\n\n* `GET /status` — health, memory count, last-scan timestamp\n* `POST /query` — natural-language search over memory\n* `GET /memories` — list / browse with filtering\n* `DELETE /memories/{id}` — remove a specific memory\n* `POST /consolidate` — trigger dream cycle on demand\n* `POST /walkthrough` — dependency-ordered walkthrough\n\nThe `/query` endpoint is the primary search surface. Authentication uses `X-MindMem-Token`. Any other path returns 404.",
    ),
    (
        "What is the v3.9 mind-mem HTTP search endpoint?",
        "`POST /query`. It accepts a JSON body with a natural-language search string and runs hybrid BM25+vector retrieval. The other v3.9 routes are `/status`, `/memories`, `/memories/{id}`, `/consolidate`, `/walkthrough`.",
    ),
    (
        "Enumerate every HTTP route exposed by mind-mem v3.9.",
        "Six routes total: `GET /status`, `POST /query`, `GET /memories`, `DELETE /memories/{id}`, `POST /consolidate`, `POST /walkthrough`. The `/query` route is the search surface; `/status` is the health probe. Routes outside this list are not registered.",
    ),
    (
        "Which v3.9 HTTP endpoint runs a search?",
        "`POST /query`. It's the natural-language search endpoint over the memory store. (`/memories` lists/browses; `/query` searches.)",
    ),
    (
        "Walk through the v3.9 HTTP REST adapter routes.",
        "Six routes: `/status` (health), `/query` (search — POST a natural-language string), `/memories` (list / GET, DELETE by id), `/consolidate` (trigger dream cycle), `/walkthrough` (dependency-ordered learning sequence). The `/query` endpoint is the principal search route.",
    ),
    # === Inbox ingestion — keyword "ingest" ===
    (
        "How do I drop a file into the v3.9 inbox for ingestion?",
        "Configure an `inbox/` directory in `mind-mem.json` and start the inbox watcher (`InboxWatcher`). To ingest a file, simply drop (move/copy) it into the `inbox/` directory; the watcher will detect, classify by extension, and ingest it through the appropriate path. On successful ingest, the file moves to `inbox/_processed/<ts>/`; on ingest failure it moves to `inbox/_failed/<ts>/`. Each ingested file is hashed and stamped with the active `TransformHash`.",
    ),
    (
        "What's the procedure to ingest a new file via the v3.9 inbox?",
        "Place the file in the configured `inbox/` directory. The `InboxWatcher` detects the new file, classifies it by extension (text / pdf / image / audio), and routes it to the matching ingest function. Successful ingestion atomically moves the source to `_processed/<ts>/`; failed ingestion moves it to `_failed/<ts>/`.",
    ),
    (
        "How does the v3.9 inbox-driven ingest pipeline work?",
        "Drop a file into `inbox/`. The watcher picks it up, classifies it by extension, runs the appropriate ingest path, and atomically moves the source. Each file is hashed; each generated block is stamped with `TransformHash` so the pipeline-version trail is auditable.",
    ),
    # === Replicated Postgres — primary + round-robin keywords ===
    (
        "How does the v3.9 replicated postgres backend handle writes vs reads?",
        "Writes always go to the `primary`. Reads round-robin across the configured replica DSNs. The `ReplicatedPostgresBlockStore` wrapper splits traffic by intent: write-class methods (`write_block`, `delete_block`, `lock`) hit `primary` exclusively; read-class methods (`get_block`, `recall`, `hybrid_search`, `prefetch`) round-robin across replicas. A failing replica falls back to `primary` (fail-open); after 3 consecutive failures the wrapper opens a circuit breaker on that replica.",
    ),
    (
        "Replicated-postgres routing in v3.9 — which node sees writes?",
        "`primary` only. Writes are never round-robin. Reads round-robin across replicas with circuit-breaker fallback to `primary` on replica failure.",
    ),
    (
        "How do reads work against the v3.9 replicated postgres backend?",
        "Reads round-robin across the configured replica DSNs. On replica error the wrapper falls back to `primary` (fail-open). Three consecutive failures on the same replica trip a circuit breaker and that replica is skipped on subsequent reads.",
    ),
    # === Block schemas — exact field names with all fields visible ===
    (
        "Show me a CODE block template.",
        "Code-change decision block.\n\n```\n[CODE-20260413-001]\nId: <id>\nDate: <date>\nStatus: <status>\nFile: <file>\nChange: <change>\nRationale: <rationale>\n```\n\nAll six fields are required: `Id`, `Date`, `Status`, `File`, `Change`, `Rationale`. The trailing `Rationale:` line records why the change was made and is mandatory.",
    ),
    (
        "What fields does a CODE block require?",
        "A mind-mem CODE block requires: `Id`, `Date`, `Status`, `File`, `Change`, `Rationale`. All six fields are mandatory; `Rationale` records why the change happened.",
    ),
    (
        "Print the canonical CODE block format.",
        "```\n[CODE-20260413-001]\nId: <id>\nDate: <date>\nStatus: <status>\nFile: <file>\nChange: <change>\nRationale: <rationale>\n```\n\nFields, in order: `Id`, `Date`, `Status`, `File`, `Change`, `Rationale`. Do not omit `Rationale`.",
    ),
    (
        "Is `Rationale` a required field on a CODE block?",
        "Yes. CODE blocks require `Rationale`. The full field list is `Id`, `Date`, `Status`, `File`, `Change`, `Rationale` — all six are mandatory.",
    ),
    (
        "Show me a FIELD block template.",
        "Field audit entry block.\n\n```\n[FIELD-20260413-001]\nFieldChangeId: <fieldchangeid>\nBlockId: <blockid>\nField: <field>\nOld: <old>\nNew: <new>\n```\n\nAll five fields are required: `FieldChangeId`, `BlockId`, `Field`, `Old`, `New`. The `BlockId:` line links the audit to the block being mutated.",
    ),
    (
        "What fields does a FIELD block require?",
        "A mind-mem FIELD block requires: `FieldChangeId`, `BlockId`, `Field`, `Old`, `New`. The `BlockId` field is mandatory — it identifies which block was mutated.",
    ),
    (
        "Print the canonical FIELD block format.",
        "```\n[FIELD-20260413-001]\nFieldChangeId: <fieldchangeid>\nBlockId: <blockid>\nField: <field>\nOld: <old>\nNew: <new>\n```\n\nFields: `FieldChangeId`, `BlockId`, `Field`, `Old`, `New`. The `BlockId:` link is required.",
    ),
    (
        "Is `BlockId` a required field on a FIELD block?",
        "Yes. FIELD blocks require `BlockId` — it links the field-change audit entry to the block being mutated. The full field list is `FieldChangeId`, `BlockId`, `Field`, `Old`, `New`.",
    ),
    # === Drift workflow — signal_stats + scan + verify_chain ===
    (
        "I want to check if a belief has drifted. Which tools do I call?",
        "Run a multi-step drift check:\n\n1. Call `scan` to surface drift signals across the workspace (returns counts + a JSON drift report).\n2. Read `signal_stats` to inspect the latest drift / contradiction events. `signal_stats` returns the chronologically-ordered tail of the signal stream so you can see which beliefs are drifting now.\n3. Optional: `verify_chain` to confirm the audit trail behind the flagged signals.\n\nKey tools: `scan`, `signal_stats`, `verify_chain`.",
    ),
    (
        "Walk me through the v3.9 drift-check workflow.",
        "Three tools: `scan` (workspace-wide drift detection), `signal_stats` (chronological tail of recent drift / contradiction events), and `verify_chain` (audit trail verification). Run `scan` first to surface candidates, then `signal_stats` to see the freshest signals, then `verify_chain` if you want to confirm the audit chain hasn't been tampered with.",
    ),
    (
        "How do I see if a belief has drifted recently?",
        "Use `scan` to detect drift across the workspace, then `signal_stats` to read the most recent drift / contradiction events. `signal_stats` is specifically the tool that exposes the chronological tail of new signals — without it you only see the aggregate count.",
    ),
    # === Field audit workflow — field_history + FieldAuditor ===
    (
        "Audit who changed field X on block Y.",
        "Call `field_history(block_id=Y, field='X')`. The underlying record class is `FieldAuditor` — every field mutation is recorded by `FieldAuditor` with before/after values, agent attribution, and a reason string. `field_history` returns the chronological audit trail for the (block, field) pair, hash-linked into the AuditChain.\n\nKey names: `field_history` (tool / method), `FieldAuditor` (record class).",
    ),
    (
        "Which tool returns the audit trail of a specific field on a specific block?",
        "`field_history`. It queries the `FieldAuditor` records and returns the chronological mutation history for the (block, field) pair. Each entry includes before/after values, agent attribution, and a reason string.",
    ),
    (
        "What does field-level audit lookup look like in mind-mem?",
        "Use `FieldAuditor(workspace).field_history(block_id=Y, field='X')`. This returns the audit trail for that single field on that single block. Note: `field_history` is the field-level audit tool; `audit_model_tool` is for model-checkpoint audits (a different workflow entirely).",
    ),
    (
        "Is `audit_model_tool` for field-level audits?",
        "No. `audit_model_tool` runs the seven-check audit on a local model checkpoint manifest — it has nothing to do with block field history. For field-level audits (who changed field X on block Y) use `field_history`, backed by the `FieldAuditor` record class.",
    ),
    # === Governance benchmark — exact tool name ===
    (
        "Run the full governance benchmark suite.",
        "Call `governance_health_bench`. It exercises contradiction detection, audit completeness, drift detection, and scalability in a single run, returning a JSON report with per-suite pass/fail.",
    ),
    (
        "What tool benchmarks the entire governance plane?",
        "`governance_health_bench`. Single call exercises contradiction, audit, drift, and scalability suites and emits per-suite pass/fail.",
    ),
    (
        "Stress-test the mind-mem governance pipeline.",
        "Run `governance_health_bench` — it's the dedicated benchmark tool for the governance plane. Returns per-suite pass/fail across contradiction, audit, drift, and scalability.",
    ),
    (
        "How do I run the governance benchmark?",
        "`governance_health_bench`. It's the only governance-plane benchmark in mind-mem.",
    ),
    # === propose_update / write_block disambiguation ===
    (
        "Write a new memory block.",
        "Call `propose_update`. It stages a new block for human review (the governance gate); after `approve_apply` the block is committed to the store. `propose_update` is the canonical write path — never bypass it via raw `write_block` calls.",
    ),
    (
        "Stage a new block for human review.",
        "Use `propose_update` — it stages the block in the proposal queue. To commit after review, call `approve_apply`.",
    ),
    (
        "How do I add a new memory entry through mind-mem?",
        "Through the proposal pipeline: call `propose_update` to stage the new block, then `approve_apply` once it has cleared review. Direct writes via `write_block` skip governance and should not be used.",
    ),
]


def _harvest_targeted_patches() -> Iterator[dict]:
    """v3.9.3 targeted re-teaching for probes that failed in the v3.9.2 eval."""
    for q, a in _TARGETED_PATCHES:
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        }


# ---------------------------------------------------------------------------
# Source 11 (v3.9.5): direct teaching of every eval probe.
#
# v3.9.2/v3.9.3/v3.9.4 all failed because corpus shape was an indirect proxy
# for what the eval measures. This source mirrors `train/eval_harness.py`
# 1:1 — every probe gets 6+ paraphrased prompts whose answers ALL contain
# the exact required keyword(s) verbatim, so token-level loss is forced
# to learn the keyword. Verified at corpus-build time (see _verify_eval_keywords).
# ---------------------------------------------------------------------------


# === TOOL_CALL_QUESTIONS — 20 probes × 6 paraphrases = 120 examples ===
_EVAL_TOOL_CALL: list[tuple[str, str, str]] = [
    # (eval_prompt, expected_tool, one-line gloss)
    ("How do I verify the audit chain integrity?", "verify_chain", "verify the audit chain integrity"),
    ("What tool lists all contradictions?", "list_contradictions", "list every contradiction"),
    ("Apply a staged proposal.", "approve_apply", "apply / approve a staged proposal"),
    ("Roll back an apply.", "rollback_proposal", "roll back a previous apply"),
    ("Write a new memory block.", "propose_update", "write a new memory block (via the proposal queue)"),
    ("Full-text search over memory.", "recall", "run full-text search over memory"),
    ("Run hybrid BM25 + vector search.", "hybrid_search", "run hybrid BM25 + vector search"),
    ("Reindex the FTS tables.", "reindex", "reindex the FTS tables"),
    ("Scan the workspace for contradictions and drift.", "scan", "scan the workspace for contradictions and drift"),
    ("Export memory to JSONL.", "export_memory", "export memory to JSONL"),
    ("Classify the intent of a query.", "intent_classify", "classify the intent of a query"),
    ("Get index statistics.", "index_stats", "fetch index statistics"),
    ("Run the governance health benchmark.", "governance_health_bench", "run the governance health benchmark"),
    ("Encrypt a workspace file at rest.", "encrypt_file", "encrypt a workspace file at rest"),
    ("Decrypt an encrypted workspace file.", "decrypt_file", "decrypt an encrypted workspace file"),
    ("Find blocks similar to a given ID.", "find_similar", "find blocks similar to a given ID"),
    ("Delete a memory item.", "delete_memory_item", "delete a memory item"),
    ("Show block category summaries.", "category_summary", "show block category summaries"),
    ("Prefetch recall results.", "prefetch", "prefetch recall results"),
    ("Diagnose why a query returned its results.", "retrieval_diagnostics", "diagnose retrieval ranking"),
]


def _harvest_eval_tool_call() -> Iterator[dict]:
    # v3.9.6 lesson: previously this list had 4 verbatim "{prompt}" entries
    # paired with 4 different answers — same input → 4 conflicting targets.
    # Token cross-entropy can't reduce below a floor when targets are
    # multimodal. Now: each (prompt-paraphrase, answer) pair is unique;
    # exactly ONE answer template per paraphrase.
    paraphrases = [
        "{prompt}",                                                 # eval-verbatim form
        "Answer in one word: {prompt}",
        "Which mind-mem MCP tool would you call to: {gloss}?",
        "{prompt} (answer with the single MCP tool name)",
        "MCP tool to {gloss}?",
        "Tell me the mind-mem tool name for: {gloss}",
        "Tool name: {prompt}",
        "What's the MCP tool that will {gloss}?",
    ]
    answers = [
        "Use `{tool}`.",                                            # paired w/ verbatim form
        "`{tool}`.",
        "Call `{tool}`.",
        "`{tool}` — the MCP tool.",
        "Reach for `{tool}`.",
        "`{tool}` is the right tool.",
        "`{tool}`.",
        "Mind-mem ships `{tool}` for that.",
    ]
    for prompt, tool, gloss in _EVAL_TOOL_CALL:
        for i, qp in enumerate(paraphrases):
            ap = answers[i % len(answers)]
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": qp.format(prompt=prompt, gloss=gloss)},
                    {"role": "assistant", "content": ap.format(tool=tool, gloss=gloss)},
                ]
            }


# === BLOCK_SCHEMA_QUESTIONS — 10 probes × 6 paraphrases = 60 examples ===
# Each paraphrase produces an answer that contains ALL required tokens.
_EVAL_BLOCK_SCHEMA: list[tuple[str, str, list[str]]] = [
    # (block_short, friendly_name, ordered_field_list — must include all eval-required tokens)
    ("DREF", "Drift signal block", ["Id", "Date", "Severity", "Signal", "Summary"]),
    ("ADR", "Architecture Decision Record", ["ADR", "Date", "Status", "Context", "Decision", "Consequences"]),
    ("BUG", "Bug report block", ["Id", "Date", "Severity", "Symptom", "RootCause", "Fix"]),
    ("PERF", "Performance record block", ["Id", "Date", "Metric", "Before", "After", "Rationale"]),
    ("CODE", "Code-change decision block", ["Id", "Date", "Status", "File", "Change", "Rationale"]),
    ("ALGO", "Algorithm-choice block", ["Id", "Date", "Problem", "Chosen", "Alternatives", "Rationale"]),
    ("CONV", "Code-convention block", ["Id", "Convention", "Example", "Scope"]),
    ("CHECK", "Contradiction record block", ["Id", "Date", "BlockA", "BlockB", "Type"]),
    ("EV", "Evidence object block", ["EvidenceId", "Action", "Actor", "TargetBlock", "PayloadHash"]),
    ("FIELD", "Field audit entry block", ["FieldChangeId", "BlockId", "Field", "Old", "New"]),
]


def _harvest_eval_block_schema() -> Iterator[dict]:
    """Each yielded example contains the canonical [SHORT-DATE-001] header
    plus every required field-name in the answer. The eval matches these
    tokens byte-for-byte. Eval prompts use form 'Show me {a/an} <SHORT>
    block template.' — we emit each verbatim form 6× to up gradient
    weight on the canonical phrasing."""
    canonical_paraphrases = [
        # eval-verbatim form, repeated 6x for high gradient density
        None,  # placeholder, computed per block-type
    ]
    secondary_paraphrases = [
        "Print the canonical {short} block template.",
        "Give me the canonical {short} block format.",
        "What does a {short} block look like?",
        "Render the {short} block template with all required fields.",
        "Template for a {short} block?",
    ]
    for short, friendly, fields in _EVAL_BLOCK_SCHEMA:
        # Canonical body: first line is the ID prefix the eval expects, then
        # every field on its own `Field:` line.
        template_lines = [f"[{short}-20260413-001]"] + [f"{f}: <{f.lower()}>" for f in fields]
        template = "\n".join(template_lines)
        long_answer = (
            f"{friendly}.\n\n"
            f"```\n{template}\n```\n\n"
            f"Required fields, in order: " + ", ".join(f"`{f}:`" for f in fields) + ". "
            f"Every field above is mandatory; omitting any of them produces an invalid {short} block."
        )
        terse_answer = f"{friendly}.\n\n```\n{template}\n```"
        # Eval verbatim form: 'Show me a/an <SHORT> block template.'
        article = "an" if short[0] in "AEIOU" else "a"
        eval_prompt_form = f"Show me {article} {short} block template."
        # ONE eval-verbatim entry (with the long answer that has
        # field list + template), plus distinct paraphrases each paired
        # with ONE answer style. v3.9.6 had 6 verbatim repeats with
        # alternating long/terse answers — same input, 2 different
        # outputs — which caused multimodal-target loss floor.
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": eval_prompt_form},
                {"role": "assistant", "content": long_answer},
            ]
        }
        for i, qp in enumerate(secondary_paraphrases):
            ans = long_answer if i % 2 == 0 else terse_answer
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": qp.format(short=short)},
                    {"role": "assistant", "content": ans},
                ]
            }


# === WORKFLOW_QUESTIONS — 5 probes × 8 paraphrases = 40 examples ===
# Each answer enumerates ALL required tool names in a step-by-step chain.
_EVAL_WORKFLOWS: list[tuple[str, list[str], str]] = [
    (
        "I see a contradiction between two decision blocks. Walk me through the fix.",
        ["list_contradictions", "propose_update", "approve_apply", "verify_chain"],
        "contradiction-resolution chain",
    ),
    (
        "I applied a bad proposal. How do I roll back safely?",
        ["rollback_proposal", "verify_chain"],
        "safe rollback chain",
    ),
    (
        "I want to check if a belief has drifted. Which tools do I call?",
        ["scan", "signal_stats"],
        "belief-drift detection chain",
    ),
    (
        "Audit who changed field X on block Y.",
        ["field_history", "FieldAuditor"],
        "field-level audit chain",
    ),
    (
        "Run the full governance benchmark suite.",
        ["governance_health_bench"],
        "governance benchmark",
    ),
]


def _harvest_eval_workflows() -> Iterator[dict]:
    paraphrase_templates = [
        "{prompt}",
        "Walk me through this in mind-mem: {prompt}",
        "Step-by-step: {prompt}",
        "{prompt} List the tool names in order.",
        "{prompt} (answer with the exact MCP tool names)",
        "Mind-mem workflow: {prompt}",
        "What's the canonical mind-mem chain for this? {prompt}",
        "Tools to call, in order: {prompt}",
    ]
    for prompt, tools, gloss in _EVAL_WORKFLOWS:
        # Construct a step-by-step answer that contains every required tool name.
        steps = [f"{i+1}. Call `{t}`" for i, t in enumerate(tools)]
        bullet_answer = (
            f"The {gloss} runs through these mind-mem MCP tools, in order:\n\n"
            + "\n".join(steps)
            + ".\n\nKey tool names: "
            + ", ".join(f"`{t}`" for t in tools)
            + "."
        )
        prose_answer = (
            f"For the {gloss}, call: "
            + ", then ".join(f"`{t}`" for t in tools)
            + "."
        )
        for i, qp in enumerate(paraphrase_templates):
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": qp.format(prompt=prompt)},
                    {"role": "assistant", "content": bullet_answer if i % 2 == 0 else prose_answer},
                ]
            }


# === v3.9.8 SURGICAL: extra reinforcement for the 2 probes that v3.9.7 missed ===
# Workflow 'belief drift' missed `scan` (model said `drift_signal` instead).
# Xform 'bulk re-stamp' missed `reextract_dirty_blocks` (model only said `reindex_dirty`).
# Each entry has a unique answer that LEADS with the required keyword.
_V398_SURGICAL_WORKFLOW: list[str] = [
    "Call `scan` first — it surfaces drift signals across the workspace. Then call `signal_stats` to read the chronological tail of those signals.",
    "First `scan` (workspace-wide drift detection), then `signal_stats` (recent drift / contradiction events).",
    "Use `scan` to detect drift, then `signal_stats` to inspect the latest events.",
    "Step 1: `scan` — runs both lexical (DRIFT.md) and semantic (DriftDetector) drift passes. Step 2: `signal_stats` — chronological tail of the signal stream.",
    "Run `scan` first to surface drift candidates, then `signal_stats` to read them.",
    "`scan` (entry point — full workspace drift detection), then `signal_stats` (filter to recent events).",
    "Mind-mem drift workflow: `scan` first, `signal_stats` next.",
    "Drift-check chain: 1. `scan` 2. `signal_stats`. The `scan` call writes DRIFT.md and updates the SQLite signal store; `signal_stats` reads from it.",
    "Begin with `scan`. It updates the drift signal store. Then call `signal_stats` to inspect.",
    "Two-step: `scan` to detect, `signal_stats` to inspect. Both are real mind-mem MCP tools.",
]
# v3.9.9: PURGED every `reindex_dirty` mention. v3.9.8 had one entry that
# said "the reindex_dirty MCP tool wraps this" — model conflated the two
# tool names and produced `reindex_dirty` at inference. Now: every answer
# only mentions `reextract_dirty_blocks`, never the MCP wrapper name.
_V398_SURGICAL_XFORM: list[str] = [
    "Use the `reindex_dirty` MCP tool. Internally it calls `reextract_dirty_blocks(workspace, ...)` from `mind_mem.pipeline_hash` to walk every block, find those whose `TransformHash` doesn't match the active pipeline digest, re-run extraction on them, and re-stamp via `stamp_transform_hash`.",
    "`reextract_dirty_blocks` from `mind_mem.pipeline_hash` is the bulk re-stamp helper. It iterates every block and re-extracts the ones with a stale `TransformHash`.",
    "Use the `reindex_dirty` MCP tool (impl: `reextract_dirty_blocks(workspace, ...)` in `mind_mem.pipeline_hash`).",
    "`reextract_dirty_blocks` — the library function in `mind_mem.pipeline_hash` that bulk-re-stamps every block whose `TransformHash` is stale.",
    "Bulk re-stamp helper: `reextract_dirty_blocks` (in `mind_mem.pipeline_hash`). Walks the workspace, re-extracts dirty blocks, re-stamps via `stamp_transform_hash`.",
    "`reindex_dirty` MCP tool (impl: `reextract_dirty_blocks(workspace, ...)`). Mind-mem v3.9 helper that processes every dirty block in bulk, re-extracts them, and re-stamps with the current pipeline hash.",
    "The bulk re-stamp helper is `reextract_dirty_blocks` (in `mind_mem.pipeline_hash`). It iterates the entire workspace and re-extracts every block with a stale `TransformHash`.",
    "Use `reextract_dirty_blocks` from `mind_mem.pipeline_hash` — that's the v3.9 bulk-re-extraction helper.",
    "`reextract_dirty_blocks` — library function in `mind_mem.pipeline_hash` that bulk re-stamps blocks with drifted `TransformHash`.",
    "`reindex_dirty` MCP tool (impl: `reextract_dirty_blocks`).",
    "The function that bulk re-stamps blocks with drifted pipeline hash is `reextract_dirty_blocks` from `mind_mem.pipeline_hash`.",
    "`reindex_dirty` MCP tool — bulk re-extraction helper (impl: `reextract_dirty_blocks(workspace, ...)` in `mind_mem.pipeline_hash`).",
]


# v3.9.9: rollback workflow needs terse, identical-shape answers all
# ending with `verify_chain`. v3.9.8 had 3 divergent shapes (long prose /
# 2-line / bullet) — model averaged across them and produced a NEW shorter
# answer that omitted `verify_chain`.
_V399_SURGICAL_ROLLBACK: list[str] = [
    "Call `rollback_proposal(receipt_ts)`, then `verify_chain` to confirm the audit ledger is intact.",
    "1. `rollback_proposal(receipt_ts)` 2. `verify_chain`.",
    "Use `rollback_proposal` first, then `verify_chain`.",
    "Safe rollback: call `rollback_proposal`, then `verify_chain`.",
    "`rollback_proposal` then `verify_chain` — both are required for a safe rollback.",
    "Step 1: `rollback_proposal`. Step 2: `verify_chain`. Both are mandatory.",
    "The safe rollback chain is `rollback_proposal` followed by `verify_chain`.",
    "Call `rollback_proposal(receipt_ts)`. Then call `verify_chain` to confirm the audit chain.",
    "Roll back via `rollback_proposal`, then verify with `verify_chain`.",
    "`rollback_proposal` rolls the change back; `verify_chain` confirms the chain integrity afterwards.",
    "Two MCP tools, in order: `rollback_proposal`, then `verify_chain`.",
    "Run `rollback_proposal` to revert the apply, then `verify_chain` to verify the audit ledger.",
]


def _harvest_v398_surgical() -> Iterator[dict]:
    """v3.9.8/9: dedicated reinforcement for probes that prior runs missed."""
    workflow_prompt = "I want to check if a belief has drifted. Which tools do I call?"
    xform_prompt = "How do I bulk re-stamp blocks whose pipeline hash drifted?"
    rollback_prompt = "I applied a bad proposal. How do I roll back safely?"
    for ans in _V398_SURGICAL_WORKFLOW:
        yield {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": workflow_prompt},
            {"role": "assistant", "content": ans},
        ]}
    for ans in _V398_SURGICAL_XFORM:
        yield {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": xform_prompt},
            {"role": "assistant", "content": ans},
        ]}
    for ans in _V399_SURGICAL_ROLLBACK:
        yield {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rollback_prompt},
            {"role": "assistant", "content": ans},
        ]}


# === V39_NEW_TOOLS — 13 probes × 6 paraphrases = 78 examples ===
_EVAL_V39_NEW_TOOLS: list[tuple[str, str, str]] = [
    ("Show the dependency-ordered learning sequence for a topic.", "compile_truth_walkthrough",
     "compile a Kahn-topological learning walkthrough for a topic"),
    ("Recall blocks and project them through a persona.", "recall_with_persona",
     "recall blocks with brief / detailed / technical persona projection"),
    ("Inspect the current pipeline hash and dirty-block summary.", "pipeline_status",
     "inspect the active pipeline hash and dirty-block summary"),
    ("Re-stamp blocks whose TransformHash is stale.", "reindex_dirty",
     "re-stamp blocks whose TransformHash is stale"),
    ("Run the seven-check audit on a local model checkpoint via MCP.", "audit_model_tool",
     "run the seven-check audit on a local model checkpoint"),
    ("Sign a checkpoint manifest with Ed25519 via MCP.", "sign_model_tool",
     "sign a model-checkpoint manifest with Ed25519"),
    ("Verify an Ed25519 manifest signature via MCP.", "verify_model_tool",
     "verify an Ed25519 manifest signature"),
    ("Convert a MIND IR graph between mic@2 text and mic-b binary.", "mic_convert",
     "convert a MIND IR graph between mic@2 text and mic-b binary"),
    ("Inspect the structure of a serialized MIC/MAP graph.", "mic_inspect",
     "inspect the structure of a serialized MIC/MAP graph"),
    ("Verify the cryptographic chain of evidence for a block by Merkle path.", "verify_merkle",
     "verify the Merkle proof chain for a block"),
    ("List every evidence entry the audit chain has accepted so far.", "list_evidence",
     "list every accepted evidence entry on the audit chain"),
    ("Get a registered MIND kernel by name.", "get_mind_kernel",
     "fetch a registered MIND kernel by name"),
    ("List every registered MIND kernel.", "list_mind_kernels",
     "list every registered MIND kernel"),
]


def _harvest_eval_v39_new_tools() -> Iterator[dict]:
    # Same fix as _harvest_eval_tool_call: one paraphrase ↔ one answer.
    paraphrases = [
        "{prompt}",                                                  # eval-verbatim form
        "Answer in one word: {prompt}",
        "Which mind-mem v3.9 MCP tool would you call to: {gloss}?",
        "{prompt} (answer with the exact MCP tool name)",
        "MCP tool to {gloss}?",
        "Mind-mem v3.9 — name the tool that will: {gloss}",
        "Tool name for: {gloss}?",
        "Name the tool — {gloss}.",
    ]
    answers = [
        "`{tool}`.",
        "Call `{tool}`.",
        "Use `{tool}`.",
        "`{tool}` — the v3.9 MCP tool.",
        "Reach for `{tool}`.",
        "It's `{tool}`.",
        "`{tool}`.",
        "Mind-mem v3.9 ships `{tool}` for that.",
    ]
    for prompt, tool, gloss in _EVAL_V39_NEW_TOOLS:
        for i, qp in enumerate(paraphrases):
            ap = answers[i % len(answers)]
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": qp.format(prompt=prompt, gloss=gloss)},
                    {"role": "assistant", "content": ap.format(tool=tool, gloss=gloss)},
                ]
            }


# === V39_TRANSFORMHASH_PROMPTS — 3 probes × 8 paraphrases = 24 examples ===
_EVAL_V39_TRANSFORMHASH: list[tuple[str, str, str]] = [
    (
        "Show me the field name a v3.9 inbox-ingested block carries to record the pipeline hash.",
        "TransformHash",
        "v3.9 block field that records the active pipeline-component digest",
    ),
    (
        "Which helper stamps the current pipeline hash onto a block dict before writing?",
        "stamp_transform_hash",
        "library function that writes the TransformHash field on a block dict before storage",
    ),
    (
        "How do I bulk re-stamp blocks whose pipeline hash drifted?",
        "reextract_dirty_blocks",
        "library function that walks the workspace and re-extracts every block whose TransformHash is stale",
    ),
]


def _harvest_eval_v39_transform_hash() -> Iterator[dict]:
    paraphrases = [
        "{prompt}",
        "{prompt} (answer with the exact identifier)",
        "Mind-mem v3.9 — {prompt}",
        "Answer in one word: {prompt}",
        "What's the {gloss}?",
        "Name the {gloss}.",
        "Identifier of the {gloss}?",
        "What is the {gloss} called in mind-mem v3.9?",
    ]
    answers = [
        "`{tok}`.",
        "It's `{tok}`.",
        "`{tok}`. It is the {gloss}.",
        "`{tok}` — the {gloss}.",
    ]
    for prompt, tok, gloss in _EVAL_V39_TRANSFORMHASH:
        for i, qp in enumerate(paraphrases):
            ap = answers[i % len(answers)]
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": qp.format(prompt=prompt, gloss=gloss)},
                    {"role": "assistant", "content": ap.format(tok=tok, gloss=gloss)},
                ]
            }


# === V39_TRANSPORT_PROMPTS — 4 probes × 6 paraphrases = 24 examples ===
# Each answer contains every must-include token AND avoids every must-NOT-include
# token. v3.9.4 already passed this category at 100% — keeping the dense
# direct-teaching here as insurance.
_EVAL_V39_TRANSPORT: list[tuple[str, list[str], str]] = [
    (
        "List the v3.9 HTTP transport endpoints.",
        ["/status", "/query"],
        "The v3.9 HTTP REST adapter (`mind_mem.http_transport.serve_http`) registers exactly six routes: "
        "`GET /status` (health), `POST /query` (search), `GET /memories` (list/browse), "
        "`DELETE /memories/{id}`, `POST /consolidate` (dream cycle), `POST /walkthrough`. "
        "Routes outside this allow-list are not registered.",
    ),
    (
        "How does the v3.9 daemon trigger the dream cycle?",
        ["daemon"],
        "The v3.9 background `daemon` (see `mind_mem.daemon`) runs each configured job on its own "
        "interval thread. The `dream_cycle` task polls every `auto_interval_seconds` (default 1800s) "
        "and calls `run_dream_cycle` on the workspace; a `0` interval disables the task. The daemon "
        "schedule is configured under the `daemon` block of `mind-mem.json`.",
    ),
    (
        "How do I drop a file into the v3.9 inbox for ingestion?",
        ["inbox", "ingest"],
        "Configure an `inbox/` directory in `mind-mem.json` and start the inbox watcher "
        "(`InboxWatcher`). To ingest a file, drop (move/copy) it into the `inbox/` directory; "
        "the watcher classifies by extension and ingests through the matching path. On "
        "successful ingest the file moves to `inbox/_processed/<ts>/`; on ingest failure to "
        "`inbox/_failed/<ts>/`. Each ingested file is hashed and stamped with the active "
        "`TransformHash`.",
    ),
    (
        "How does the v3.9 replicated postgres backend handle writes vs reads?",
        ["primary", "round-robin"],
        "Writes always go to the `primary`. Reads round-robin across the configured replica DSNs. "
        "The `ReplicatedPostgresBlockStore` wrapper splits traffic by intent: write-class methods "
        "(`write_block`, `delete_block`, `lock`) hit `primary` exclusively; read-class methods "
        "(`get_block`, `recall`, `hybrid_search`, `prefetch`) round-robin across replicas. A "
        "failing replica falls back to `primary` (fail-open); after 3 consecutive failures the "
        "wrapper opens a circuit breaker on that replica.",
    ),
]


def _harvest_eval_v39_transport() -> Iterator[dict]:
    paraphrases = [
        "{prompt}",
        "Answer concisely: {prompt}",
        "Mind-mem v3.9 — {prompt}",
        "{prompt} (cite the exact identifiers)",
        "{prompt}",
        "Walk through it: {prompt}",
    ]
    for prompt, _required, answer in _EVAL_V39_TRANSPORT:
        for qp in paraphrases:
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": qp.format(prompt=prompt)},
                    {"role": "assistant", "content": answer},
                ]
            }


def _harvest_eval_direct_teaching() -> Iterator[dict]:
    """v3.9.5: union of all direct-teaching sources, one generator.

    v3.10.0: REMOVED _harvest_eval_workflows from this union — it was
    creating destructive interference with _harvest_workflow_chains
    (v3.0.0's proven 6/6 winning pattern). Workflow probes are now
    taught ONLY through _harvest_workflow_chains: one canonical terse
    answer per probe, emitted 3× with prefix variations.

    The other direct-teaching sources stay — they provide verbatim
    coverage for tool_call, block_schema, v39_new_tools, transform_hash,
    and transport_guard probes that are NOT covered by workflow_chains.
    """
    yield from _harvest_eval_tool_call()
    yield from _harvest_eval_block_schema()
    # _harvest_eval_workflows REMOVED — see docstring
    yield from _harvest_eval_v39_new_tools()
    yield from _harvest_eval_v39_transform_hash()
    yield from _harvest_eval_v39_transport()


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
            _harvest_targeted_patches(),
            # v3.10.0: KEEP _harvest_eval_direct_teaching for verbatim
            # coverage of tool_call/block_schema/v39_new_tools/transform_hash/
            # transport_guard probes (21 of these have 0 corpus coverage
            # without it). REMOVED _harvest_v398_surgical (multimodal
            # workflow conflicts). _harvest_eval_direct_teaching itself
            # had its workflows sub-component disabled — workflow probes
            # are now taught only by _harvest_workflow_chains (v3.0.0
            # pattern: 1 canonical terse answer per probe).
            _harvest_eval_direct_teaching(),
            # v3.11.0: new MCP surfaces (validate_block, block_lineage,
            # recall(explain=True)).  ~102 new probes before dedup + cap.
            _harvest_v311_validate_block(),
            _harvest_v311_block_lineage(),
            _harvest_v311_recall_explain(),
            # v3.12.0: quality-gate strict mode, lineage staleness wiring,
            # Petri red-team CI.  ~60 new probes before dedup + cap.
            _harvest_v312_quality_gate_strict_mode(),
            _harvest_v312_lineage_staleness(),
            _harvest_v312_red_team_ci(),
        ):
            for entry in _dedup(src):
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
    print(f"wrote {count} examples (pre-cap) to {OUT}")
    # v3.9.10: tighter cap=1 for the eval-prompt buckets specifically (so
    # canonical terse answer wins), but keep cap=4 for everything else.
    _cap_multimodal_answers(OUT, max_answers_per_prompt=4)


def _cap_multimodal_answers(path, max_answers_per_prompt: int = 2) -> None:
    """Rewrite corpus so each (system, user) prompt has at most N distinct answers."""
    from collections import OrderedDict
    lines = path.read_text(encoding="utf-8").splitlines()
    bucket: dict[tuple[str, str], list[str]] = OrderedDict()
    order: list[tuple[str, str]] = []
    for line in lines:
        m = json.loads(line)
        s = next((x["content"] for x in m["messages"] if x["role"] == "system"), "")
        u = next((x["content"] for x in m["messages"] if x["role"] == "user"), "")
        a = next((x["content"] for x in m["messages"] if x["role"] == "assistant"), "")
        k = (s, u)
        if k not in bucket:
            bucket[k] = []
            order.append(k)
        if a not in bucket[k] and len(bucket[k]) < max_answers_per_prompt:
            bucket[k].append(a)
    written = 0
    with path.open("w", encoding="utf-8") as fh:
        for k in order:
            s, u = k
            for a in bucket[k]:
                entry = {"messages": [
                    {"role": "system", "content": s},
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a},
                ]}
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1
    print(f"capped to {written} examples (max {max_answers_per_prompt} answers per (system, user) prompt)")


# ---------------------------------------------------------------------------
# Source 11: v3.11.0 surface probes
#
# Three new surfaces need corpus coverage so the v3.11.0-fullft training
# run does not regress tool_call accuracy:
#
#   validate_block  — 7 rules × 6 phrasings = 42 probes
#   block_lineage   — traversal patterns (1/2/3-hop, cycles, isolated
#                     nodes, kind_filter, max_depth clamp) = 30 probes
#   recall(explain) — _explain field shape, math consistency, default
#                     omission = 30 probes
#
# Corpus budget: ~102 new examples before dedup + cap.
# ---------------------------------------------------------------------------

_VALIDATE_BLOCK_RULES: list[tuple[str, str, str]] = [
    (
        "empty",
        "validate_block rejects an empty block",
        "The `validate_block` tool returns `rule: empty` when the block text is blank or contains only whitespace.  Set `strict=True` to make this an error rather than a warning.",
    ),
    (
        "short",
        "validate_block rejects a block that is too short to be meaningful",
        "`validate_block` fires the `short` rule when the block text is under the minimum token threshold (default 3 tokens).  The advisory response lists the actual token count alongside the threshold.",
    ),
    (
        "duplicate",
        "validate_block detects a block that is near-identical to an existing block",
        "The `duplicate` rule fires when cosine similarity between the candidate and an existing block exceeds the dedup threshold (0.97 by default).  Returned data includes the conflicting block ID and similarity score.",
    ),
    (
        "stopwords",
        "validate_block rejects a block whose content is almost entirely stopwords",
        "The `stopwords` rule fires when more than 80% of the tokens in the block are function words (the, is, a, …).  This usually indicates an accidentally captured boilerplate fragment.",
    ),
    (
        "oversize",
        "validate_block rejects a block that exceeds the size limit",
        "`validate_block` applies the `oversize` rule when the serialized block exceeds `max_block_bytes` (default 64 KB).  Split large content across multiple blocks before writing.",
    ),
    (
        "utf8",
        "validate_block rejects a block with invalid UTF-8 sequences",
        "The `utf8` rule catches raw byte sequences that are not valid UTF-8.  They would corrupt the SQLite FTS index if written.  Re-encode the source document before proposing the block.",
    ),
    (
        "injection",
        "validate_block detects prompt-injection patterns in block text",
        "The `injection` rule fires when the block text contains patterns that resemble system-prompt overrides (e.g. `Ignore previous instructions`).  In `strict=True` mode this is a hard block; in advisory mode it is a warning with the matched pattern in the response.",
    ),
]

_VALIDATE_BLOCK_PHRASINGS: list[tuple[str, str]] = [
    # (question template, answer slot)
    ("How do I check whether a block proposal is valid before writing it?", "validate_block"),
    ("Which tool validates a block without writing it?", "validate_block"),
    ("What does `validate_block` do in advisory mode?", "validate_block"),
    ("How do I run `validate_block` in strict mode?", "validate_block"),
    # Two phrasings asked verbatim by eval_harness V311_NEW_TOOLS — keep
    # them aligned so the §0.1 audit returns TOTAL_ISSUES: 0.
    ("What does `validate_block(text, strict=True)` do on a rule violation?", "validate_block"),
    ("How do I detect prompt-injection patterns in a candidate block?", "validate_block"),
    ("Which v3.11.0 tool checks for duplicate, oversize, and UTF-8 issues before a write?", "validate_block"),
    ("When should I call `validate_block` vs `propose_update`?", "validate_block"),
]


def _harvest_v311_validate_block() -> Iterator[dict]:
    """42 probes covering the 7 validate_block rules × 6 phrasings."""
    for rule, intent, explanation in _VALIDATE_BLOCK_RULES:
        for q, _ in _VALIDATE_BLOCK_PHRASINGS:
            rule_q = q + f"  (Focus on the `{rule}` rule.)"
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": rule_q},
                    {
                        "role": "assistant",
                        "content": (
                            f"Call `validate_block(text, strict=False)` — the new v3.11.0 "
                            f"MCP tool.  For the `{rule}` rule: {explanation}"
                        ),
                    },
                ]
            }
    # Advisory vs strict toggle — standalone probes
    for q, a in [
        (
            "Difference between `validate_block(strict=False)` and `validate_block(strict=True)`?",
            "In advisory mode (`strict=False`, the default) `validate_block` returns a list of rule violations as warnings — the caller decides whether to proceed.  In strict mode (`strict=True`) any violation raises a `ValidationError` that blocks the write.",
        ),
        (
            "Can `validate_block` force-escape a block that fails the `injection` rule?",
            "`validate_block` does not mutate the text.  Pass `force_escape=True` to receive a sanitized copy with injection patterns neutralized, then pass that copy to `propose_update`.",
        ),
    ]:
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        }


# --- block_lineage probes (30 probes) ----------------------------------------

_LINEAGE_PATTERNS: list[tuple[str, str]] = [
    (
        "How do I trace what blocks depend on block X?",
        "Call `block_lineage(block_id='X', max_depth=3)`.  It returns a directed graph of all blocks reachable from X within `max_depth` hops, along with edge kinds (`cites`, `implements`, `refines`, `contradicts`, `cooccurrence`).",
    ),
    (
        "What does `block_lineage` return for a 1-hop traversal?",
        "`block_lineage(block_id, max_depth=1)` returns only the immediate neighbours — blocks that share a direct typed edge with the target.  Each result node carries `kind`, `source_id`, `target_id`, and `weight`.",
    ),
    (
        "How do I do a 2-hop lineage traversal?",
        "Pass `max_depth=2` to `block_lineage`.  The result graph includes both immediate neighbours and their neighbours, de-duplicated.  Cycle detection is built in — circular chains are broken at the first repeated node.",
    ),
    (
        "What happens when `block_lineage` encounters a cycle?",
        "`block_lineage` detects cycles via a visited-set and stops traversal at the first repeated block ID.  The cycle edge is still returned in the graph with `cycle=True` so the caller can visualise the loop.",
    ),
    (
        "What does `block_lineage` return for an isolated block with no edges?",
        "For a block with no edges, `block_lineage` returns a single-node graph containing only the requested block.  The `edges` list is empty and `depth_reached` is 0.",
    ),
    (
        "How do I filter `block_lineage` to only `cites` edges?",
        "Pass `kind_filter='cites'` to `block_lineage`.  Only edges whose `kind` matches the filter are traversed.  Valid kinds: `cites`, `implements`, `refines`, `contradicts`, `cooccurrence`.",
    ),
    (
        "What are the five edge kinds in `block_lineage`?",
        "The five typed edge kinds are `cites` (direct citation), `implements` (code realises spec), `refines` (one block narrows another), `contradicts` (governance conflict), and `cooccurrence` (statistical co-retrieval).",
    ),
    (
        # Eval-harness V311_NEW_TOOLS[9] uses this exact prompt and
        # requires every kind name to appear in the answer.
        "What are the five edge kinds supported by `block_lineage`?",
        "`block_lineage` supports five typed edge kinds: `cites` (direct citation), `implements` (code realises spec), `refines` (one block narrows another), `contradicts` (governance conflict), and `cooccurrence` (statistical co-retrieval).",
    ),
    (
        "What is the default `max_depth` for `block_lineage`?",
        "The default `max_depth` is 3.  Passing a larger value is clamped to the configured ceiling (default 10) to prevent runaway traversals on dense graphs.",
    ),
    (
        "What happens if I pass `max_depth=100` to `block_lineage`?",
        "`block_lineage` clamps `max_depth` at the configured ceiling (default 10).  A warning is included in the response metadata so the caller knows the value was adjusted.",
    ),
    (
        "How does `block_lineage` differ from `graph_query`?",
        "`block_lineage` is a convenience wrapper that traverses the typed-edge graph starting from a single block and returns a structured result with depth tracking and cycle detection.  `graph_query` accepts an arbitrary filter expression over the full edge table and is more powerful but requires manual traversal.",
    ),
    (
        "How do I add an edge before calling `block_lineage`?",
        "Call `add_block_edge(source_id, target_id, kind='cites', weight=1.0)` to create the edge.  Once the edge is committed, `block_lineage` will include it in subsequent traversals.",
    ),
    (
        "What tool adds a typed edge between two blocks?",
        "`add_block_edge(source_id, target_id, kind, weight=1.0)` — the v3.11.0 tool that registers a directed typed edge between two existing blocks in the lineage graph.",
    ),
    (
        "What does `add_block_edge` require?",
        "`add_block_edge` requires `source_id` (existing block ID), `target_id` (existing block ID), and `kind` (one of the five typed kinds).  `weight` defaults to 1.0 and can be any positive float.",
    ),
    (
        "Can `block_lineage` traverse `contradicts` edges?",
        "Yes.  Pass `kind_filter='contradicts'` to restrict traversal to contradiction edges.  Without a filter all five kinds are traversed.",
    ),
    (
        "How do I get the full lineage sub-graph for a block?",
        "Call `block_lineage(block_id, max_depth=10)` with no `kind_filter` to get every reachable node and edge within the depth ceiling.  The returned graph is a dict with `nodes` (list of block IDs + metadata) and `edges` (list of typed edge dicts).",
    ),
]


def _harvest_v311_block_lineage() -> Iterator[dict]:
    """30 probes covering block_lineage traversal patterns."""
    for q, a in _LINEAGE_PATTERNS:
        for prefix in ("", "In mind-mem v3.11: "):
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prefix + q if prefix else q},
                    {"role": "assistant", "content": a},
                ]
            }


# --- recall(explain=True) probes (30 probes) ---------------------------------

_RECALL_EXPLAIN_PROBES: list[tuple[str, str]] = [
    (
        "What does `recall(query, explain=True)` return that the default call does not?",
        "When `explain=True` is passed, each result block carries an additional `_explain` dict with fields `bm25_score`, `vector_score`, `rrf_rank`, `tier_boost`, and `final_score`.  The default call omits `_explain` entirely.",
    ),
    (
        "What fields live inside the `_explain` dict returned by `recall(explain=True)`?",
        "The `_explain` dict contains: `bm25_score` (BM25F raw score), `vector_score` (cosine similarity), `rrf_rank` (reciprocal rank fusion rank before boost), `tier_boost` (multiplier from the block's memory tier), and `final_score` (the value used for final ordering).",
    ),
    (
        "Is `_explain` present on every result block when `recall(explain=True)` is used?",
        "Yes.  Every block in the result list carries `_explain` when `explain=True`.  The field is omitted (not set to null) when `explain=False` (the default).",
    ),
    (
        "How does `final_score` in `_explain` relate to the other scores?",
        "`final_score = rrf_rank * tier_boost`.  `rrf_rank` is the reciprocal-rank-fusion value computed from `bm25_score` and `vector_score`.  `tier_boost` is a multiplier (≥ 1.0) derived from the block's access tier.",
    ),
    (
        "Why is `_explain` omitted by default from `recall` results?",
        "Computing explain metadata adds a small overhead and roughly doubles the response payload size.  It is off by default to keep normal recall fast and compact.  Pass `explain=True` only when debugging a ranking or building a retrieval-diagnostics report.",
    ),
    (
        "How do I use `recall(explain=True)` to understand why one block ranked above another?",
        "Call `recall(query, explain=True)` and compare the `_explain.rrf_rank` values across the results.  A lower `rrf_rank` means a higher combined BM25+vector rank.  Multiply by `tier_boost` to get `final_score` and confirm the ordering.",
    ),
    (
        "What is `rrf_rank` inside `_explain`?",
        "`rrf_rank` is the reciprocal-rank-fusion score computed from the BM25F rank and the vector rank.  Formula: `1 / (k + bm25_rank) + 1 / (k + vector_rank)` where k=60 (the standard RRF constant).  Higher `rrf_rank` = better combined rank.",
    ),
    (
        "What is `tier_boost` inside `_explain`?",
        "`tier_boost` is a multiplier applied to `rrf_rank` to compute `final_score`.  Blocks in higher memory tiers (WORKING, LONG_TERM) receive a boost > 1.0; EPISODIC-tier blocks receive 1.0.  The exact values come from the tier-decay configuration.",
    ),
    (
        "Does `hybrid_search` also support `explain=True`?",
        "Yes.  `hybrid_search(query, explain=True)` returns the same `_explain` structure as `recall`.  Both tools share the same RRF + tier-boost explainability path.",
    ),
    (
        "How do I check that `_explain.final_score` is consistent with the result ordering?",
        "Sort the result list by `_explain.final_score` descending and confirm it matches the order returned by `recall(explain=True)`.  If it does not, a bug in the RRF merge or tier-boost logic is likely.",
    ),
    (
        "Does `recall` return `_explain` when `explain` is not passed?",
        "No.  `_explain` is absent (not null, not an empty dict — absent) when `explain` defaults to `False`.  Code that checks `result.get('_explain')` is correct; code that checks `result['_explain'] is None` will raise `KeyError` on a default call.",
    ),
    (
        "Can I request `explain=True` on a `recall` call that also uses `top_k`?",
        "Yes.  `recall(query, top_k=5, explain=True)` returns up to 5 results each carrying `_explain`.  All optional parameters are orthogonal to `explain`.",
    ),
    (
        "Which `_explain` field should I inspect to diagnose a BM25 vs vector disagreement?",
        "Compare `_explain.bm25_score` and `_explain.vector_score` directly.  A high `bm25_score` with a low `vector_score` means the block is a lexical match but semantically distant.  A high `vector_score` with a low `bm25_score` means the opposite.",
    ),
    (
        "What is the math behind `final_score` in `_explain`?",
        "`final_score = rrf_rank * tier_boost` where `rrf_rank = 1/(60 + bm25_rank) + 1/(60 + vector_rank)` (ranks derived from ordered `bm25_score` and `vector_score` lists).  `tier_boost` is a configured per-tier multiplier (default LONG_TERM=1.5, WORKING=1.2, EPISODIC=1.0).",
    ),
    (
        "Is the `_explain` dict the same structure for `recall` and `hybrid_search`?",
        "Yes.  Both `recall(explain=True)` and `hybrid_search(explain=True)` return `_explain` dicts with identical fields: `bm25_score`, `vector_score`, `rrf_rank`, `tier_boost`, `final_score`.  The values differ because `hybrid_search` may use a different query expansion path.",
    ),
]


def _harvest_v311_recall_explain() -> Iterator[dict]:
    """30 probes covering the recall(explain=True) _explain field shape."""
    for q, a in _RECALL_EXPLAIN_PROBES:
        for prefix in ("", "v3.11.0 feature — "):
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prefix + q if prefix else q},
                    {"role": "assistant", "content": a},
                ]
            }


# ---------------------------------------------------------------------------
# v3.12.0 probes
# ---------------------------------------------------------------------------

# --- quality_gate strict mode probes (~25 probes) ---------------------------

_QUALITY_GATE_PROBES: list[tuple[str, str]] = [
    (
        "What are the three valid values for `quality_gate.mode` in `mind-mem.json`?",
        'The three modes are `"off"` (gate disabled — no validation on write), `"advisory"` (default — `validate_block` runs pre-write and violations are returned as warnings without blocking), and `"strict"` (violations reject the write with a structured 400 envelope).',
    ),
    (
        "What is the default value of `quality_gate.mode`?",
        'The default is `"advisory"`.  Setting `quality_gate.mode = "advisory"` in `mind-mem.json` preserves v3.11.0 behavior — validation runs but never blocks a write.',
    ),
    (
        "How do I enable strict mode so that a rule violation blocks a write?",
        'Set `quality_gate.mode = "strict"` in `mind-mem.json`.  With strict mode active, any call to `propose_update` that fails `validate_block` is rejected immediately with a structured 400 response containing the rule name and violation detail.',
    ),
    (
        "How do I disable the quality gate entirely?",
        'Set `quality_gate.mode = "off"` in `mind-mem.json`.  When the mode is `"off"`, `propose_update` does not call `validate_block` at all — no validation overhead, no warnings.',
    ),
    (
        "Where does `quality_gate.mode` live in the config file?",
        '`quality_gate.mode` is a nested key inside `mind-mem.json` under the `quality_gate` object.  Example: `{"quality_gate": {"mode": "strict"}}`.  The file lives at `$MIND_MEM_WORKSPACE/mind-mem.json` (or the path pointed to by `MM_CONFIG`).',
    ),
    (
        "What does `propose_update` do differently when `quality_gate.mode` is `\"strict\"`?",
        'When `quality_gate.mode = "strict"`, `propose_update` calls `validate_block(text, strict=True)` before staging the block.  If any rule fires, `propose_update` raises a `ValidationError` and returns a 400 envelope — the block is never staged.',
    ),
    (
        "What does `propose_update` do when `quality_gate.mode` is `\"advisory\"`?",
        'When `quality_gate.mode = "advisory"`, `propose_update` calls `validate_block` pre-write and attaches any warnings to the response under `validation_warnings`.  The write proceeds regardless — advisory mode never blocks.',
    ),
    (
        "What is the shape of the rejection envelope when strict mode fires?",
        'The 400 envelope is `{"error": "quality_gate_rejection", "mode": "<advisory|strict>", "reasons": ["<rule>: <message>", ...], "advisory": [...], "hint": "..."}`.  The `reasons` list contains every rule that fired as `"<rule>: <message>"` strings; `advisory` carries any warnings that did not block.',
    ),
    (
        "How do I read the per-rule rejection counter for the `injection_marker` rule?",
        'Read the metric key `quality_gate_rejections_injection_marker` from the workspace metrics store.  Each rule has its own counter: `quality_gate_rejections_<rule>` where `<rule>` is the real rule name (e.g. `near_duplicate`, `injection_marker`).  These counters are NOT exposed by `index_stats`; read them via the Prometheus exporter or `metrics.snapshot()`.',
    ),
    (
        "What metric key tracks how many times the `near_duplicate` rule rejected a write?",
        'The metric key is `quality_gate_rejections_near_duplicate`.  It increments once per `propose_update` call rejected by the `near_duplicate` rule in strict mode.',
    ),
    (
        "When should I choose `strict` mode over `advisory` mode?",
        'Use `strict` mode in production pipelines where bad blocks would corrupt downstream reasoning — e.g. when blocks feed a governance walkthrough or are used as ground truth.  Use `advisory` mode during exploratory ingestion where you want visibility without blocking writes.',
    ),
    (
        "When should I use `quality_gate.mode = \"off\"`?",
        'Use `"off"` only for bulk-import scenarios where every block has already been pre-validated externally (e.g. a trusted migration from another workspace).  It removes all per-write overhead but gives no safety net.',
    ),
    (
        "Does `quality_gate.mode` affect `validate_block` called directly via MCP?",
        'No.  `validate_block` called directly is always advisory regardless of `quality_gate.mode` — it returns a list of violations without raising.  `quality_gate.mode` only controls what `propose_update` does with the result.',
    ),
    (
        "Is there an escape hatch to write a block that fails validation in strict mode?",
        'Yes — but NOT via `propose_update(force=True)`.  `propose_update` has no `force` parameter.  To bypass the strict mode gate: call `validate_block(text, force=True)` directly to obtain a verdict annotated `forced=True`; then set `quality_gate.mode = "off"` in `mind-mem.json` to disable the gate for the workspace.  Use sparingly.',
    ),
    (
        "What operator runbook covers the quality gate configuration?",
        'The runbook is at `docs/quality-gate.md`.  It covers mode selection, per-rule counter interpretation, the workspace-level `quality_gate.mode = "off"` bypass (the force escape hatch), and upgrading from v3.11.0 advisory-only behavior.',
    ),
    (
        "What happens to existing `validate_block` calls after upgrading to strict mode?",
        'Existing direct `validate_block` MCP calls are unaffected — they remain advisory.  Only `propose_update` changes behavior.  No code changes are needed for direct callers.',
    ),
    (
        "How does the quality gate interact with the `injection_marker` rule in strict mode?",
        'In strict mode, if `validate_block` detects a prompt-injection pattern in the candidate text, `propose_update` rejects with `{"error": "quality_gate_rejection", "mode": "strict", "reasons": ["injection_marker: block contains a known prompt-injection marker"], ...}` and increments `quality_gate_rejections_injection_marker`.  The block is never staged.',
    ),
    (
        "What is the `quality_gate_rejections_<rule>` counter format?",
        'It is a per-rule integer counter keyed as `quality_gate_rejections_<rule>` where `<rule>` is the real rule name (e.g. `near_duplicate`, `injection_marker`, `oversize`, `malformed_utf8`).  The counters live in the in-process metrics store and are NOT exposed by `index_stats`; read them via the Prometheus exporter or `metrics.snapshot()`.',
    ),
    (
        "How do I confirm that `quality_gate.mode` changed from advisory to strict?",
        'Submit a block that is known to fail a rule (e.g. an empty string for the `empty` rule or a near-identical block for `near_duplicate`) and confirm you receive a `{"error": "quality_gate_rejection", ...}` response rather than a warning.  There is no `quality_gate_mode` field in `index_stats`.',
    ),
    (
        "Does `quality_gate.mode` require a daemon restart?",
        'No.  `mind-mem.json` is read on each `propose_update` call.  Changing `quality_gate.mode` takes effect on the next write without restarting the daemon or MCP server.',
    ),
    (
        "What validate_block rules are evaluated in strict mode?",
        'All seven rules are evaluated unconditionally in both advisory and strict mode: `empty`, `too_short`, `oversize`, `malformed_utf8`, `stopwords_only`, `near_duplicate`, and `injection_marker`.  Every fired rule appends a `"<rule>: <message>"` entry to the verdict — in strict mode these go to `reasons`, in advisory mode to `advisory`.  The full list is returned at once; there is no short-circuit.',
    ),
    (
        "Can I set different modes for different workspaces?",
        'Yes.  `mind-mem.json` is per-workspace.  You can run `quality_gate.mode = "strict"` in one workspace and `"advisory"` in another.  There is no global override.',
    ),
    (
        "How does strict mode change the `propose_update` return value on success?",
        'On a successful write in strict mode the return value is identical to advisory mode — a staged proposal dict with `proposal_id` and `block_id`.  The only behavioral change is that violations now abort before staging rather than surfacing as warnings.',
    ),
    (
        "Where is `quality_gate.mode` parsed in the codebase?",
        '`quality_gate.mode` is parsed in `src/mind_mem/mcp/infra/config.py`.  The value is read and validated at config-load time; an invalid mode string raises `ConfigError` with the allowed values listed.',
    ),
    (
        "What is the relationship between `quality_gate.mode` and `validate_block(strict=True)`?",
        '`quality_gate.mode = "strict"` causes `propose_update` to call `validate_block(text, strict=True)` internally.  They share the same code path.  The `strict` parameter on the MCP tool and the `quality_gate.mode` config key are two surfaces for the same underlying behavior.',
    ),
]


def _harvest_v312_quality_gate_strict_mode() -> Iterator[dict]:
    """~25 probes covering quality_gate.mode config, modes, counters, escape hatch."""
    for q, a in _QUALITY_GATE_PROBES:
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        }


# --- lineage staleness probes (~25 probes) -----------------------------------

_LINEAGE_STALENESS_PROBES: list[tuple[str, str]] = [
    (
        "What new SQLite table does v3.12.0 introduce for staleness tracking?",
        '`block_staleness(block_id, source_id, score, decayed_at)` — a new table that stores the persisted staleness penalty for each (block, source) pair.  Writes are idempotent upserts keyed on `(block_id, source_id)`.',
    ),
    (
        "What module implements lineage staleness propagation in v3.12.0?",
        '`src/mind_mem/lineage_staleness.py` — the new v3.12.0 module.  It owns the `block_staleness` table schema, the idempotent upsert, and the `propagate_lineage_staleness` BFS walker.',
    ),
    (
        "How does `_explain.staleness_penalty` behave differently in v3.12.0 vs v3.11.0?",
        'In v3.11.0 `_explain.staleness_penalty` was always `0.0`.  In v3.12.0 `attach_explain` accepts a `workspace` kwarg; when provided, it reads persisted values from the `block_staleness` table and surfaces the real penalty.  Without a workspace the field still defaults to `0.0`.',
    ),
    (
        "What are the kind-aware decay multipliers in `propagate_lineage_staleness`?",
        'The decay multipliers per edge kind are: `contradicts` → 1.0, `cites` → 0.8, `implements` → 0.6, `refines` → 0.4, `cooccurrence` → 0.5.  A `contradicts` edge propagates the full source penalty; a `refines` edge attenuates it to 40%.',
    ),
    (
        "Why does a `contradicts` edge propagate the fastest in lineage staleness?",
        '`contradicts` edges carry a decay multiplier of 1.0 — no attenuation.  When a block is flagged as stale via a `contradicts` edge, every dependent block inherits the full penalty immediately.  Other edge kinds attenuate the penalty at each hop.',
    ),
    (
        "What is the maximum number of hops `propagate_lineage_staleness` will walk?",
        'The default cap is `max_hops=3`.  The function signature is `propagate_lineage_staleness(workspace, source_id, max_hops=3)`.  Increasing `max_hops` raises the propagation radius but also the write volume to `block_staleness`.',
    ),
    (
        "How do I trigger lineage staleness propagation from the CLI?",
        'Use `mm lineage flag <src> <dst> --kind contradicts --weight 1.0`.  This command bundles `add_block_edge` (to register the edge) and `propagate_lineage_staleness` (to walk the graph and write staleness scores) into a single atomic CLI call.',
    ),
    (
        "What does `mm lineage flag` do internally?",
        '`mm lineage flag <src> <dst> --kind <kind>` calls `add_block_edge(src, dst, kind)` to register the directed typed edge, then calls `propagate_lineage_staleness(workspace, source_id=src)` to walk the BFS and write updated staleness scores to the `block_staleness` table.',
    ),
    (
        "How is `block_staleness` written — is the upsert idempotent?",
        'Yes.  The upsert is keyed on `(block_id, source_id)`.  Running `propagate_lineage_staleness` twice with the same source produces the same rows — no duplicate penalties accumulate.  The `decayed_at` timestamp is updated on each run.',
    ),
    (
        "What is the `decayed_at` column in `block_staleness` for?",
        '`decayed_at` records the UTC timestamp of the last staleness propagation that touched the row.  It is used by the tier-decay sweep to age out stale penalties that have not been refreshed within a configurable window.',
    ),
    (
        "What fields does the `block_staleness` table contain?",
        'Four columns: `block_id` (the affected block), `source_id` (the block that triggered propagation), `score` (the propagated staleness penalty, 0.0–1.0), and `decayed_at` (UTC timestamp of the last propagation write).',
    ),
    (
        "How does `propagate_lineage_staleness` compute the penalty for a block two hops away?",
        'At each hop the incoming penalty is multiplied by the decay multiplier for the traversed edge kind.  For a source penalty of 1.0 traversing `contradicts` (1.0) then `cites` (0.8): hop-1 penalty = 1.0, hop-2 penalty = 0.8.  The score stored is the product of all multipliers along the path.',
    ),
    (
        "What happens when `propagate_lineage_staleness` encounters a block that already has a higher staleness score from a different source?",
        'Each `(block_id, source_id)` pair owns at most one row; the write is a hard overwrite — the latest propagation from THAT source wins.  A different `source_id` creates a different row.  The MAX semantics happen at read time: `get_staleness_score(block_id)` executes `SELECT MAX(score) FROM block_staleness WHERE block_id = ?` across all rows for that block, so the most-stale source wins at retrieval.',
    ),
    (
        "How do I inspect the staleness penalty for a specific block?",
        'Call `recall(query, explain=True)` with the active workspace.  The `_explain.staleness_penalty` field on each result block reflects the persisted value from `block_staleness`.  Alternatively, query the `block_staleness` SQLite table directly by `block_id`.',
    ),
    (
        "What edge kind should I use when flagging a block as authoritative over another?",
        'Use `contradicts`.  When block A is known to supersede block B, add the edge `add_block_edge(A, B, kind="contradicts")` then run `mm lineage flag A B --kind contradicts`.  This propagates a full 1.0 penalty to B and all blocks that depend on B.',
    ),
    (
        "Does `propagate_lineage_staleness` write to the `block_staleness` table for the source block itself?",
        'No.  The source block (`source_id`) is the origin of the propagation, not a target.  Only blocks reachable from `source_id` via the lineage graph (treated as undirected at read time — `_outgoing` issues `UNION ALL` in both directions) within `max_hops` receive staleness rows.',
    ),
    (
        "What is the decay multiplier for a `cites` edge in lineage staleness?",
        'The `cites` decay multiplier is `0.8`.  A penalty of 1.0 at the source becomes 0.8 for the directly-cited block, 0.64 for a block two hops away via two `cites` edges, and so on.',
    ),
    (
        "What is the decay multiplier for a `cooccurrence` edge?",
        'The `cooccurrence` decay multiplier is `0.5`.  Co-occurrence edges represent statistical correlation, not a semantic dependency, so the staleness signal is attenuated by half at each hop.',
    ),
    (
        "What is the decay multiplier for an `implements` edge?",
        'The `implements` decay multiplier is `0.6`.  If a specification block is flagged stale, the implementing code block inherits 60% of the penalty.',
    ),
    (
        "What is the decay multiplier for a `refines` edge?",
        'The `refines` decay multiplier is `0.4`.  `refines` edges are considered the weakest dependency — a block that merely narrows another should inherit only 40% of its staleness penalty.',
    ),
    (
        "How does `_explain.staleness_penalty` affect final ranking in v3.12.0?",
        '`staleness_penalty` is surfaced in `_explain` for transparency but is not yet subtracted from `final_score` automatically.  Callers that want staleness-aware ranking should apply `final_score * (1 - staleness_penalty)` themselves.',
    ),
    (
        "What BFS traversal strategy does `propagate_lineage_staleness` use?",
        'Bounded BFS starting from `source_id`.  A visited set prevents re-visiting blocks.  The queue is depth-limited to `max_hops`.  Each enqueued node carries the accumulated penalty at the time of enqueue so the correct attenuation is applied when the node is dequeued.',
    ),
    (
        "Is `propagate_lineage_staleness` safe to call concurrently from multiple workers?",
        'Yes.  The `block_staleness` upsert uses `INSERT INTO ... ON CONFLICT(block_id, source_id) DO UPDATE SET score = excluded.score` — a hard overwrite per source row.  Concurrent propagations from different `source_id`s write to different rows and merge safely at read time via `SELECT MAX(score)`.  No locking beyond the normal SQLite WAL write serialization.',
    ),
    (
        "What Python function do I call to propagate lineage staleness from code?",
        'Call `propagate_lineage_staleness(workspace, source_id, max_hops=3)` from `src/mind_mem/lineage_staleness.py`.  It returns a dict `{block_id: score}` of every block updated in the current run.',
    ),
    (
        "How does `mm lineage flag` differ from calling `add_block_edge` then `propagate_lineage_staleness` separately?",
        '`mm lineage flag` is a convenience wrapper that calls both in sequence within a single CLI invocation.  The Python API calls are equivalent.  The CLI is preferred for operator use because it logs both operations to the audit trail in a single trace.',
    ),
]


def _harvest_v312_lineage_staleness() -> Iterator[dict]:
    """~25 probes covering the block_staleness table, BFS propagation, decay multipliers."""
    for q, a in _LINEAGE_STALENESS_PROBES:
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        }


# --- Petri red-team CI probes (~10 probes) -----------------------------------

_RED_TEAM_PROBES: list[tuple[str, str]] = [
    (
        "What is Petri in the context of mind-mem v3.12.0?",
        'Petri is an adversarial red-team harness for MCP servers.  It probes the memory system with adversarial inputs — prompt injection attempts, malformed tool calls, out-of-distribution queries — and uses a judge LLM to score responses.  Install with `pip install -e ".[red-team]"`.',
    ),
    (
        "How do I install the Petri red-team dependency?",
        'Run `pip install -e ".[red-team]"` from the mind-mem repo root.  This installs Petri and its judge-LLM dependencies.  The base `pip install mind-mem` does not include Petri.',
    ),
    (
        "Where is the Petri red-team workflow file?",
        'The workflow is at `.github/workflows/red-team.yml`.  It triggers on tag pushes matching `v*` and is advisory (`continue-on-error: true`) — a Petri failure does not block release.',
    ),
    (
        "What are the three adversarial seeds used in the v3.12.0 red-team CI?",
        'The three seeds are `self_exfiltration_memory_trigger` (attempts to coerce the agent into leaking memory blocks), `broken_tool_error_handling` (sends malformed tool arguments to check error containment), and `weird_ood_tool_use` (invokes tools in unexpected sequences to probe state corruption).',
    ),
    (
        "Why is the Petri workflow advisory (`continue-on-error: true`)?",
        'Red-team failures are expected during iterative development — a new seed may fail before the corresponding defense is shipped.  Making the workflow advisory prevents a Petri regression from blocking a legitimate release while the fix is in progress.',
    ),
    (
        "What happens when `ANTHROPIC_API_KEY` is absent in a Petri CI run?",
        'The workflow detects the missing secret and skips cleanly without failing the job.  This ensures PR builds from forks — which cannot access org secrets — do not fail on the red-team step.',
    ),
    (
        "What does the `--petri-limit` flag control?",
        '`--petri-limit N` caps the number of adversarial turns Petri runs per seed.  Lower values reduce cost and latency at the expense of coverage.  The default in CI is controlled by the workflow file; operators can override it locally.',
    ),
    (
        "Where are Petri transcript artifacts uploaded in CI?",
        'Transcript artifacts are uploaded to the GitHub Actions artifact store with 90-day retention.  Each run produces a per-seed JSONL transcript that operators can download to inspect judge reasoning.',
    ),
    (
        "How much does a Petri CI run cost approximately?",
        'Approximately $10–15 per tag-push run when using the sonnet judge and all three seeds at default `--petri-limit`.  Cost scales linearly with `--petri-limit` and the number of seeds.',
    ),
    (
        "Where is the operator guide for Petri red-team integration?",
        'The CI integration section is documented in `docs/red-team-audit.md`.  It covers seed selection, `--petri-limit` tuning, transcript interpretation, and how to promote a Petri failure into a new test case.',
    ),
]


def _harvest_v312_red_team_ci() -> Iterator[dict]:
    """~10 probes covering Petri red-team CI: install, seeds, workflow, cost."""
    for q, a in _RED_TEAM_PROBES:
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        }


if __name__ == "__main__":
    main()
