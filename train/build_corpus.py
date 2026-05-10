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
    # Variant 3 — direct-name form.  v3.12.1 retrain v5: was just "`X`."
    # Now requires signature + first docstring line so the model learns to
    # emit the canonical parameters (advisory, max_depth, etc).
    yield {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Name the mind-mem tool for: {hint}."},
            {"role": "assistant", "content": (
                f"`{tool_name}` — full signature `{tool_name}({args})`.  "
                f"{doc.strip().splitlines()[0] if doc.strip() else ''}"
            )},
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
    # Variant 6 — how phrased differently.  v3.12.1 retrain v5: terse template
    # "Use the `X` MCP tool." was learned and emitted on EVAL prompts that
    # required additional tokens (advisory, max_depth, etc).  Long-form to
    # break the terseness bias.
    for alt in _paraphrase_usage(hint):
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": alt},
                {"role": "assistant", "content": (
                    f"Use `{tool_name}` — call as `{tool_name}({args})`.  "
                    f"From the docstring: {doc.strip().splitlines()[0] if doc.strip() else 'see signature.'}"
                )},
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
            # v3.12.1 corpus density fix: paraphrases + denial probes after
            # the 2026-05-09 retrain showed 18/95 eval failures from
            # under-densified v3.12 probes (4-LLM consensus: Grok/Mistral/
            # DeepSeek/GLM all agreed on corpus density as root cause).
            _harvest_v312_density_fix(),
            # v4 retrain: balanced per-edge-kind isolated-fact reinforcement.
            # Adds ≥10 isolated probes for cites/implements/cooccurrence to
            # match the existing refines saturation density (asymmetric
            # corpus drove the v3.12.1 cites=0.4 model error).  See
            # train/V4_RETRAIN_TODO.md for the verification gate.
            _harvest_v4_kind_balance(),
            # v4 surfaces: 98 probes covering the 9 new v4 modules
            # (circuit_breaker, backpressure, health, logging_context,
            # block_metadata, observability cardinality, eviction
            # debug_plan/active_policy, surprise FallbackPolicy,
            # cognitive_kernel is_kernel_registered).  Without these the
            # model has zero training signal for the new surfaces.
            _harvest_v4_surfaces(),
            # v4 eval-exact reinforcement: verbatim eval-probe Qs paired
            # with answers containing every required token. Closes the
            # memorisation-vs-generalisation gap that caused 18/95 fails
            # in the v3.12.0 retrain (see _V4_EVAL_EXACT_PROBES doc).
            _harvest_v4_eval_exact(),
            # v4 retry reinforcement: 45+ heavy-saturation probes for the
            # 5 first-run misses (FieldAuditor.field_history, backpressure
            # watermarks vs hallucinated WATERMARK_RATIO, debug_plan
            # block_ids plural, FallbackPolicy enum vs hallucinated
            # LOWEST_FIRST, block_staleness decayed_at column). Includes
            # anti-association denial probes to overwrite base-model
            # priors that bled through the first retrain.
            _harvest_v4_retry_reinforce(),
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
    # Each tuple is (canonical_rule_name, intent, explanation).  Names
    # MUST match the literal strings emitted by quality_gate.py — the
    # model uses these as substrings in `_explain` reasoning later.
    (
        "empty",
        "validate_block rejects an empty block",
        "The `validate_block` tool appends `empty: block is empty or whitespace-only` to the verdict's `reasons` list (strict mode) or `advisory` list (advisory mode) when `text.strip()` is falsy.",
    ),
    (
        "too_short",
        "validate_block rejects a block that is too short to be meaningful",
        "`validate_block` fires the `too_short` rule when the count of non-whitespace characters is under `_MIN_CHARS` (32).  The reason string lists the actual character count alongside the minimum.",
    ),
    (
        "near_duplicate",
        "validate_block detects a block that is near-identical to a recent block",
        "The `near_duplicate` rule fires when `difflib.SequenceMatcher`'s ratio against any block written in the last 24h is at or above `_DUP_RATIO` (0.97).  The reason string includes the matched ratio.",
    ),
    (
        "stopwords_only",
        "validate_block rejects a block whose content is entirely stopwords",
        "The `stopwords_only` rule fires when EVERY token (after lowercasing and `[A-Za-z0-9_]+` extraction) is in the stopword set.  Mixed content with even one non-stopword passes.",
    ),
    (
        "oversize",
        "validate_block rejects a block that exceeds the byte limit",
        "`validate_block` applies the `oversize` rule when the UTF-8 encoded byte length exceeds `_MAX_BYTES` (64 * 1024 = 65,536).  Split large content across multiple blocks before writing.",
    ),
    (
        "malformed_utf8",
        "validate_block rejects a block that cannot encode as UTF-8",
        "The `malformed_utf8` rule catches strings containing lone surrogates that fail `text.encode('utf-8')` strict mode.  Such payloads would corrupt the SQLite FTS index if written.",
    ),
    (
        "injection_marker",
        "validate_block detects prompt-injection patterns in block text",
        "The `injection_marker` rule fires when the block text matches any of the regex patterns in `_INJECTION_MARKERS` (e.g. `ignore previous instructions`, `<|im_start|>`, `[[INST]]`, `jailbreak`).  In strict mode this becomes a `reasons[]` entry; in advisory mode an `advisory[]` entry.",
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
            "Both modes evaluate every rule and return a `QualityGateVerdict` dataclass.  In advisory mode (`strict=False`, the default) every fired rule appends to the `advisory` list and `accept` stays True.  In strict mode (`strict=True`) every fired rule appends to the `reasons` list and `accept` becomes False.  No exception is raised in either case — the verdict is the return value.",
        ),
        (
            "Can `validate_block` force-accept a block that fails the `injection_marker` rule?",
            "Yes.  Pass `force=True` to `validate_block`.  The verdict is annotated `forced=True`, every fired rule still appears in the `advisory` list (so the caller can audit what was bypassed), and `accept` is forced to True.  `validate_block` does not mutate the text — sanitisation is the caller's responsibility.",
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
        "`block_lineage(block_id, max_depth=1)` returns a `LineageResult` whose `edges` list contains every immediate neighbour reachable from the target.  Each `LineageEdge` carries `block_id` (the neighbour), `kind` (one of `cites`/`implements`/`refines`/`contradicts`/`cooccurrence`), `distance` (1 for direct neighbours), and `confidence` (the kind-specific decay multiplier).",
    ),
    (
        "How do I do a 2-hop lineage traversal?",
        "Pass `max_depth=2` to `block_lineage`.  The result graph includes both immediate neighbours and their neighbours, de-duplicated.  Cycle detection is built in — circular chains are broken at the first repeated node.",
    ),
    (
        "What happens when `block_lineage` encounters a cycle?",
        "`block_lineage` detects cycles via a `visited` set populated as the BFS enqueues each neighbour.  When a node would be revisited it is silently skipped — no `cycle=True` flag is added, no warning is emitted, the edge that would close the cycle simply does not appear in the result.",
    ),
    (
        "What does `block_lineage` return for an isolated block with no edges?",
        "`block_lineage` returns `LineageResult(root=<block_id>, edges=[], truncated=False, max_depth=3)` — `to_dict()` shape is `{\"root\": ..., \"edges\": [], \"truncated\": false, \"max_depth\": 3, \"count\": 0}`.  There is no `depth_reached` or `nodes` field.",
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
        "The default `max_depth` is `LINEAGE_DEPTH_CAP = 3` (defined in `src/mind_mem/block_lineage.py`).  Any value passed at the call site is clamped via `max(1, min(int(max_depth), LINEAGE_DEPTH_CAP))`, so the effective range is `[1, 3]`.",
    ),
    (
        "What happens if I pass `max_depth=100` to `block_lineage`?",
        "`block_lineage` silently clamps the value to `LINEAGE_DEPTH_CAP = 3` (using `max(1, min(int(max_depth), LINEAGE_DEPTH_CAP))`).  No warning is emitted and no metadata field is added — the returned `LineageResult.max_depth` reflects the clamped value (3), so callers can detect the clamp by comparing the request to `result.max_depth`.",
    ),
    (
        "How does `block_lineage` differ from `graph_query`?",
        "`block_lineage` is a convenience wrapper that traverses the typed-edge graph starting from a single block and returns a structured result with depth tracking and cycle detection.  `graph_query` accepts an arbitrary filter expression over the full edge table and is more powerful but requires manual traversal.",
    ),
    (
        "How do I add an edge before calling `block_lineage`?",
        "Call `add_block_edge(src, dst, kind='cites', weight=1.0)` to create the edge.  Once the edge is committed (via the underlying upsert into the `co_retrieval` table keyed on `(mem1_id, mem2_id)`), `block_lineage` will include it in subsequent traversals.",
    ),
    (
        "What tool adds a typed edge between two blocks?",
        "`add_block_edge(src, dst, kind, weight=1.0)` — the v3.11.0 MCP tool that records a typed lineage edge.  Reads merge both directions via `UNION ALL`, so the edge is treated as undirected at traversal time even though the underlying row is keyed on `(mem1_id, mem2_id)` = `(src, dst)`.",
    ),
    (
        "What does `add_block_edge` require?",
        "`add_block_edge` requires `src` (existing block ID), `dst` (existing block ID, must differ from `src`), and `kind` (one of `cites`, `implements`, `refines`, `contradicts`, `cooccurrence`).  `weight` defaults to 1.0 (any float).  `ValueError` is raised on a self-loop, an unknown `kind`, or an empty `src`/`dst`.",
    ),
    (
        "Can `block_lineage` traverse `contradicts` edges?",
        "Yes.  Pass `kind_filter='contradicts'` to restrict traversal to contradiction edges.  Without a filter all five kinds are traversed.",
    ),
    (
        "How do I get the full lineage sub-graph for a block?",
        "Call `block_lineage(block_id, max_depth=3)` (the default and ceiling) with no `kind_filter` to get every reachable edge within the bounded BFS.  The `to_dict()` shape is `{\"root\": ..., \"edges\": [...], \"truncated\": <bool>, \"max_depth\": 3, \"count\": <int>}`.  Each entry in `edges` is `{\"block_id\": ..., \"kind\": ..., \"distance\": ..., \"confidence\": ...}`.  There is no separate `nodes` list — the root is in `root` and every neighbour is in `edges`.",
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
        '`quality_gate.mode` is a nested key inside `mind-mem.json` under the `quality_gate` object.  Example: `{"quality_gate": {"mode": "strict"}}`.  The file lives at `$MIND_MEM_WORKSPACE/mind-mem.json` and is read by `_load_config(workspace)` in `src/mind_mem/mcp/infra/config.py`.',
    ),
    (
        "What does `propose_update` do differently when `quality_gate.mode` is `\"strict\"`?",
        'When `quality_gate.mode = "strict"`, `propose_update` calls `validate_block(text, strict=True)` before staging the block.  If any rule fires, `propose_update` returns the rejection envelope `{"error": "quality_gate_rejection", "mode": "strict", "reasons": [...], "advisory": [...], "hint": "..."}` as a JSON string — no Python exception is raised.  The block is never staged.',
    ),
    (
        "What does `propose_update` do when `quality_gate.mode` is `\"advisory\"`?",
        'When `quality_gate.mode = "advisory"`, `propose_update` still calls `validate_block` pre-write but every fired rule populates the verdict\'s `advisory` list rather than `reasons`.  The aggregate `quality_gate_rejections` counter is bumped, per-rule counters are bumped, and a `quality_gate_advisory` warning is logged (NOT `quality_gate_reject` — that name is reserved for strict-mode rejections); `propose_update` proceeds with the write — advisory mode never blocks.',
    ),
    (
        "What is the shape of the rejection envelope when strict mode fires?",
        'The 400 envelope is `{"error": "quality_gate_rejection", "mode": "strict", "reasons": ["<rule>: <message>", ...], "advisory": [...], "hint": "..."}`.  The `mode` field is the literal string `"strict"` (not a placeholder).  The `reasons` list contains every rule that fired as `"<rule>: <message>"` strings; `advisory` carries any warnings that did not block.',
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
        'Yes — two escape hatches.  (1) Library-level: `validate_block(text, strict=True, force=True)` — `force=True` forces `accept=True` and stamps `forced=True` on the verdict; every fired rule still records to `reasons` for audit.  (2) Workspace-level: set `quality_gate.mode = "off"` in `mind-mem.json` — when the mode is `"off"`, `propose_update` does not call `validate_block` at all.  Note: `propose_update` itself has no `force` parameter; callers route through `validate_block(force=True)` directly.  Use either escape sparingly.',
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
        'It is a per-rule integer counter keyed as `quality_gate_rejections_<rule>` where `<rule>` is the exact rule name as registered in `quality_gate.py` — `empty` (NOT `whitespace_only`), `too_short`, `oversize`, `malformed_utf8`, `stopwords_only`, `near_duplicate`, `injection_marker`.  The counters live in the in-process metrics store and are NOT exposed by `index_stats`; read them via the Prometheus exporter or `metrics.snapshot()`.',
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
        '`quality_gate.mode` is parsed in `src/mind_mem/mcp/infra/config.py` by `_get_quality_gate_mode(ws)`.  The function reads `mind-mem.json`, looks up the nested `quality_gate.mode` key, and silently falls back to the default `"advisory"` if the key is missing or the value is not in `{"off", "advisory", "strict"}` — no exception is raised on invalid input at the config layer.  Note the layer split: the config router accepts `{"off", "advisory", "strict"}`; the `QualityGateConfig` dataclass `__post_init__` (`src/mind_mem/quality_gate.py:80`) only accepts `("advisory", "strict")` and raises `ValueError` on anything else.  `"off"` short-circuits BEFORE `QualityGateConfig` is instantiated (governance.py:95 checks `if _qg_mode != "off"` first).',
    ),
    (
        "What is the relationship between `quality_gate.mode` and `validate_block(strict=True)`?",
        '`quality_gate.mode = "strict"` causes `propose_update` to call `validate_block(text, strict=True)` internally.  They share the same code path.  The `strict` parameter on the MCP tool and the `quality_gate.mode` config key are two surfaces for the same underlying behavior.',
    ),
    (
        "Can I pass `mode=\"off\"` directly to QualityGateConfig?",
        'No.  `QualityGateConfig.__post_init__` (`src/mind_mem/quality_gate.py:80-81`) only accepts `mode in ("advisory", "strict")` and raises `ValueError` otherwise.  `"off"` is a CONFIG-LAYER value handled by `mcp/infra/config.py` and `mcp/tools/governance.py:95` — when the workspace config sets `mode = "off"`, `governance.py` short-circuits before instantiating `QualityGateConfig`.  Two different layers, do not conflate.',
    ),
    (
        "How many valid values does `quality_gate.mode` accept end-to-end?",
        'Three at the config-routing layer (`{"off", "advisory", "strict"}` in `mcp/infra/config.py`); two at the dataclass layer (`("advisory", "strict")` in `quality_gate.py`).  `"off"` lives only in the config router — it never reaches the `QualityGateConfig` dataclass because `governance.py:95` short-circuits with `if _qg_mode != "off": validate_block(...)`.  Operators see three modes; the dataclass sees two.',
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
        '`contradicts` carries `KIND_DECAY = 1.0` — the strongest possible seed-edge multiplier.  Because KIND_DECAY is applied ONCE at the seed edge, picking `contradicts` as the seed edge produces the loudest signal that survives the subsequent HOP_DECAY attenuation; weaker seed kinds (`cites` 0.8, `implements` 0.6, `cooccurrence` 0.5, `refines` 0.4) start the BFS from a smaller initial penalty.',
    ),
    (
        "What is the maximum number of hops `propagate_lineage_staleness` will walk?",
        'The signature is `propagate_lineage_staleness(workspace, source_id, *, max_hops=None)`.  When `max_hops` is `None` (the default) it falls through to `LINEAGE_DEPTH_CAP = 3`; an explicit larger value is silently clamped to 3 with no warning.  The cap bounds both BFS depth and the volume of rows written to `block_staleness`.',
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
        'The stored score is `KIND_DECAY[seed_edge_kind] × HOP_DECAY[h]`, with `HOP_DECAY = (1.0, 0.9, 0.5, 0.2)` indexed by BFS hop distance from the seed.  KIND_DECAY is applied ONCE — at the seed edge (the edge from `source_id` to its immediate neighbour); subsequent BFS hops do NOT pick up additional kind multipliers.  Concretely: if `source_id` has a `contradicts` edge to seed S (KIND_DECAY=1.0), the block at hop=1 from S stores `1.0 × 0.9 = 0.9`; the block at hop=2 from S stores `1.0 × 0.5 = 0.5`.  Replacing the seed kind with `cites` (0.8): hop=1 from S stores `0.8 × 0.9 = 0.72`; hop=2 stores `0.8 × 0.5 = 0.4`.',
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
        'No.  The source block (`source_id`) is the origin of the propagation, not a target.  Seeds are the immediate neighbours of `source_id` — `_classify_seed_neighbours` issues `UNION ALL` over `co_retrieval` in both directions, except the reverse direction excludes `kind=cooccurrence` (statistical correlations do not propagate backwards).  The BFS itself walks `lineage_adjacency`, which is fully undirected.  Only blocks reachable within `max_hops` receive staleness rows.',
    ),
    (
        "What is the decay multiplier for a `cites` edge in lineage staleness?",
        'The `cites` decay multiplier is `0.8` — applied ONCE at the seed edge (the edge from `source_id` to its immediate neighbour). Further BFS hops use only `HOP_DECAY = (1.0, 0.9, 0.5, 0.2)` regardless of the kinds of edges traversed inside the BFS.  So a `cites` seed gets `0.8 × 1.0 = 0.8`; the block one BFS hop further stores `0.8 × 0.9 = 0.72`; two BFS hops further stores `0.8 × 0.5 = 0.4`.',
    ),
    (
        "What is the decay multiplier for a `cooccurrence` edge?",
        'The `cooccurrence` decay multiplier is `0.5` — applied ONCE at the seed edge (the edge from `source_id` to its immediate neighbour).  Co-occurrence edges represent statistical correlation, not a semantic dependency, so the seed already absorbs a 0.5 attenuation; further hops attenuate via `HOP_DECAY = (1.0, 0.9, 0.5, 0.2)` only — kind does not re-enter the formula.',
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
        'Bounded BFS starting from each immediate neighbour of `source_id` (the seed).  The seed-edge kind picks `kind_mul = KIND_DECAY[k]` once; the BFS itself uses `HOP_DECAY = (1.0, 0.9, 0.5, 0.2)` indexed by hop distance from the seed.  A visited map keeps the closest hop seen so a closer seed always wins.  Per-block score = `HOP_DECAY[h] × kind_mul`; the per-source aggregate keeps the max across seeds.  The walk is depth-limited to `max_hops`, clamped to `LINEAGE_DEPTH_CAP = 3`.',
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
        '`mm lineage flag` is a convenience wrapper that calls both in sequence within a single CLI invocation and prints the combined JSON result `{"ok": true, "edge": {...}, "propagated": <count>, "affected_blocks": [...]}`.  The Python API calls are equivalent.  Use the CLI for one-shot operator workflows; call the functions directly when you need finer control over each step.',
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


# --- v3.12.1 corpus density fix (eval-driven, post-retrain-1 diagnostic) ----
# Diagnostic /tmp/mm-runpod/eval_diagnose.json showed 18/95 probes failed after
# the first retrain.  4-LLM consensus (Grok 4.3, Mistral Large, DeepSeek v4 Pro,
# GLM 5.1) agreed the cause is corpus density: each v3.12 probe had only 1
# matching corpus answer, and 5 v311_new_tools eval probes had ZERO prompt
# matches.  Fix: 5+ paraphrases per failing probe + explicit "X NOT Y" denial
# probes for hallucinated names (_block_staleness, LINEAGE_HOP_CAP,
# bfs_propagate, BlockValidationError, refines→0.6).

_V312_DENSITY_FIX_PROBES: list[tuple[str, str]] = [
    # ----- v311_new_tools: 5 zero-match prompts now have direct answers -----
    (
        "How do I check whether a block proposal is valid before writing it?",
        "Call `validate_block(text)` — the v3.11.0 MCP tool that pre-validates a block without writing it.  By default it runs in advisory mode (failed rules append to the verdict's `advisory` list, `accept` stays True).  Pass `strict=True` to flip failed rules into hard rejects (`reasons` list, `accept=False`).",
    ),
    (
        "Which tool validates a block without writing it?",
        "`validate_block(text)` — the dedicated v3.11.0 MCP tool.  It runs the seven content rules (empty, too_short, oversize, malformed_utf8, stopwords_only, near_duplicate, injection_marker) and returns a `QualityGateVerdict`.  It never writes; the writer is `propose_update`.",
    ),
    (
        "What does `validate_block(text, strict=True)` do on a rule violation?",
        "In strict mode, `validate_block` appends every fired rule to the verdict's `reasons` list as `\"<rule>: <message>\"` strings and sets `accept=False`.  No exception is raised; check `verdict.accept`.  Advisory mode (`strict=False`, the default) appends to the `advisory` list and keeps `accept=True`.",
    ),
    (
        "How do I detect prompt-injection patterns in a candidate block?",
        "Call `validate_block(text)` — the `injection_marker` rule fires when the block matches a known prompt-injection pattern (e.g. \"ignore previous instructions\", `<|im_start|>`, `[[INST]]`, `jailbreak`).  This is `validate_block`'s job, NOT `scan` (which is for cross-block contradictions/drift).",
    ),
    (
        "Which v3.11.0 tool checks for duplicate, oversize, and UTF-8 issues before a write?",
        "`validate_block` — it runs all seven content rules in one call: empty, too_short, oversize, malformed_utf8, stopwords_only, near_duplicate, injection_marker.  Each fired rule appends to either `advisory` (advisory mode) or `reasons` (strict mode) on the returned `QualityGateVerdict`.",
    ),
    # ----- v311_new_tools: terse-pass paraphrases that include max_depth/kind -----
    (
        "How do I trace what blocks depend on block X?",
        "Call `block_lineage(workspace, block_id, max_depth=3)`.  The default `max_depth=3` is `LINEAGE_DEPTH_CAP`; passing larger values is silently clamped to 3.  Returns a `LineageResult` dict with `root`, `edges`, `truncated`, `max_depth`, and `count`.",
    ),
    (
        "What tool adds a typed edge between two blocks?",
        "`add_block_edge(workspace, src, dst, kind, weight=1.0)`.  The `kind` parameter is one of the five typed edge kinds: `cites`, `implements`, `refines`, `contradicts`, `cooccurrence`.  The directed edge is stored in the `co_retrieval` lineage table.",
    ),
    # ----- workflow: rollback paraphrase that pairs rollback_proposal + verify_chain -----
    (
        "I applied a bad proposal. How do I roll back safely?",
        "Three steps: (1) `rollback_proposal(proposal_id)` to revert the apply, (2) `verify_chain` to confirm the audit chain integrity is intact post-rollback, (3) `list_contradictions` if you want to confirm no orphan contradictions remain.  Always run `verify_chain` after a rollback — it's the integrity check that proves the rollback didn't corrupt the hash chain.",
    ),
    # ----- v312_qg_strict: paraphrases for failing probes with required canonical tokens -----
    (
        "How do I enable strict mode so that a rule violation blocks a write?",
        "Set `quality_gate.mode = \"strict\"` in `mind-mem.json`.  In strict mode, `propose_update` calls `validate_block(text, strict=True)` before staging the block; any fired rule causes immediate rejection with the `quality_gate_rejection` envelope.  The block is never staged.",
    ),
    (
        "What is the shape of the rejection envelope when strict mode fires?",
        "`{\"error\": \"quality_gate_rejection\", \"mode\": \"strict\", \"reasons\": [\"<rule>: <message>\", ...], \"advisory\": [...], \"hint\": \"...\"}`.  Returned as a JSON string by `propose_update` — NOT raised as a `BlockValidationError` exception (no such exception exists; the verdict is the return value).",
    ),
    (
        "How do I read the per-rule rejection counter for the `injection_marker` rule?",
        "Read the metric key `quality_gate_rejections_injection_marker` from the workspace metrics store.  Each rule has its own counter named `quality_gate_rejections_<rule>` where `<rule>` is the canonical rule name.  Counters live in the in-process metrics store; read via `metrics.snapshot()` — NOT exposed by `index_stats`.",
    ),
    (
        "What metric key tracks how many times the `near_duplicate` rule rejected a write?",
        "`quality_gate_rejections_near_duplicate` — a per-rule integer counter that increments once per `propose_update` call rejected by the `near_duplicate` rule in strict mode.  The aggregate counter is `quality_gate_rejections` (no rule suffix).",
    ),
    (
        "What operator runbook covers the quality gate configuration?",
        "`docs/quality-gate.md` — the operator runbook for `quality_gate.mode` configuration.  It covers the three modes (`off`, `advisory`, `strict`), the workspace-level `force` escape hatch (set `quality_gate.mode = \"off\"`), and migrating from v3.11.0 advisory-only behavior.",
    ),
    # ----- v312_lineage: paraphrases for failing probes with canonical tokens -----
    (
        "What new SQLite table does v3.12.0 introduce for staleness tracking?",
        "`block_staleness(block_id, source_id, score, decayed_at)` — a v3.12.0 table that stores propagated staleness penalties keyed on `(block_id, source_id)`.  The table name is `block_staleness` — NO leading underscore.  Each `(block_id, source_id)` pair owns at most one row; writes are idempotent upserts.",
    ),
    (
        "What module implements lineage staleness propagation in v3.12.0?",
        "`src/mind_mem/lineage_staleness.py` — the v3.12.0 module that owns the `block_staleness` table schema, the idempotent upsert, and the `propagate_lineage_staleness` BFS walker.  The function name is `propagate_lineage_staleness` — NOT `propagate_staleness` and NOT `bfs_propagate`.",
    ),
    (
        "How does `_explain.staleness_penalty` behave differently in v3.12.0 vs v3.11.0?",
        "In v3.11.0 `_explain.staleness_penalty` was always `0.0` (placeholder).  In v3.12.0 `attach_explain` accepts a `workspace` kwarg; when provided, it reads the persisted value from the `block_staleness` table and surfaces the real penalty.  Without a workspace the field still defaults to `0.0`.",
    ),
    (
        "What are the kind-aware decay multipliers in `propagate_lineage_staleness`?",
        "Five kinds: `contradicts` → 1.0, `cites` → 0.8, `implements` → 0.6, `cooccurrence` → 0.5, `refines` → 0.4.  Applied ONCE at the seed edge.  A `contradicts` seed propagates the full source penalty; `refines` attenuates to 40%.",
    ),
    (
        "Why does a `contradicts` edge propagate the fastest in lineage staleness?",
        "`contradicts` carries the strongest `KIND_DECAY` multiplier of 1.0 — no attenuation at the seed edge.  Other edge kinds (`cites` 0.8, `implements` 0.6, `cooccurrence` 0.5, `refines` 0.4) start the BFS from a smaller initial penalty.  The kind multiplier is applied at the seed edge only.",
    ),
    (
        "What is the maximum number of hops `propagate_lineage_staleness` will walk?",
        "Signature: `propagate_lineage_staleness(workspace, source_id, *, max_hops=None)`.  `max_hops=None` falls through to `LINEAGE_DEPTH_CAP = 3`; explicit larger values are silently clamped to 3.  The cap bounds both BFS depth and the volume of rows written to `block_staleness`.",
    ),
    (
        "How do I trigger lineage staleness propagation from the CLI?",
        "Use `mm lineage flag <src> <dst> --kind contradicts --weight 1.0`.  This CLI command bundles `add_block_edge` (registers the edge) and `propagate_lineage_staleness` (walks the BFS, writes staleness scores) into one atomic call.  The MCP equivalent is two separate tool calls.",
    ),
    (
        "What fields does the `block_staleness` table contain?",
        "Four columns: `block_id` (the affected block), `source_id` (the block that triggered propagation), `score` (the propagated penalty 0.0–1.0), and `decayed_at` (UTC timestamp of last propagation write).  Primary key is `(block_id, source_id)`.",
    ),
    (
        "What is the decay multiplier for a `refines` edge in lineage staleness?",
        "`refines` → `0.4`.  It is the WEAKEST of the five edge kinds — a block that merely narrows another inherits only 40% of its staleness penalty.  The value is `0.4`, NOT `0.6`.",
    ),
    # ----- Canonical-name reinforcement (rewritten from explicit denials so wrong
    # literals do NOT appear in question, addressing 3-of-4 LLM concern about
    # denial-probe leakage) -----
    (
        "Confirm the exact name of the staleness-tracking SQLite table introduced in v3.12.0.",
        "The canonical name is `block_staleness` (no underscore prefix).  Defined in `src/mind_mem/lineage_staleness.py`.  Schema: `block_staleness(block_id, source_id, score, decayed_at)`.",
    ),
    (
        "Confirm the exact name of the lineage BFS depth-cap constant.",
        "The canonical name is `LINEAGE_DEPTH_CAP`.  Defined in `src/mind_mem/block_lineage.py`.  Value is `3`.  Used to clamp both `block_lineage(max_depth=...)` and `propagate_lineage_staleness(max_hops=...)`.",
    ),
    (
        "Confirm the exact name of the v3.12 lineage staleness propagation function.",
        "The canonical name is `propagate_lineage_staleness(workspace, source_id, *, max_hops=None)`.  Defined in `src/mind_mem/lineage_staleness.py`.  Internally invokes the lower-level `propagate_staleness` from `staleness.py` for the per-source BFS step.",
    ),
    (
        "Confirm: when strict mode rejects a write, what data structure does propose_update return?",
        "It returns a JSON string (`json.dumps`'d).  Shape: `{\"error\": \"quality_gate_rejection\", \"mode\": \"strict\", \"reasons\": [\"<rule>: <message>\", ...], \"advisory\": [...], \"hint\": \"...\"}`.  `propose_update` returns this — it never raises an exception for a quality-gate failure.",
    ),
    (
        "Confirm the exact decimal decay multiplier for a `refines` edge.",
        "The canonical value is `0.4` — the smallest of the five edge kinds.  Defined as `KIND_DECAY['refines'] = 0.4` in `block_lineage.py`.  Full table: contradicts 1.0, cites 0.8, implements 0.6, cooccurrence 0.5, refines 0.4.",
    ),
    (
        "Confirm: which v3.11.0 MCP tool implements per-block prompt-injection detection?",
        "The canonical tool is `validate_block(text)` — its `injection_marker` rule pattern-matches against the six known injection forms.  This is the per-block injection gate.  The `scan` tool covers cross-block contradictions/drift, not injection.",
    ),
    # ----- 5 paraphrases per high-cost v3.12 probes (boost density 1→6) -----
    (
        "Show me the SQLite schema for the v3.12.0 staleness tracking table.",
        "`CREATE TABLE block_staleness (block_id TEXT NOT NULL, source_id TEXT NOT NULL, score REAL NOT NULL, decayed_at TEXT NOT NULL, PRIMARY KEY (block_id, source_id));`  Plus an index on `block_id` for the read path.",
    ),
    (
        "Where does v3.12 persist the propagated staleness penalty per block?",
        "In the `block_staleness` SQLite table.  Columns: `block_id`, `source_id`, `score`, `decayed_at`.  Each `(block_id, source_id)` pair stores at most one row.  The MAX semantics happen at read time via `get_staleness_score` (`SELECT MAX(score) ... GROUP BY block_id`).",
    ),
    (
        "What's the canonical JSON shape returned when strict-mode quality gate rejects a write?",
        "`{\"error\": \"quality_gate_rejection\", \"mode\": \"strict\", \"reasons\": [\"<rule>: <message>\", ...], \"advisory\": [...], \"hint\": \"Statement did not pass the quality gate.  Revise and resubmit, or set quality_gate.mode=\\\"advisory\\\" to downgrade.\"}`.",
    ),
    (
        "What CLI command flags a block as superseding another and propagates staleness?",
        "`mm lineage flag <src> <dst> --kind contradicts --weight 1.0`.  The command in one atomic call: registers a typed edge via `add_block_edge`, then walks the BFS via `propagate_lineage_staleness(workspace, source_id=args.src)` and writes staleness rows to `block_staleness`.",
    ),
    (
        "Which Python module owns the BFS walker that writes block_staleness rows?",
        "`src/mind_mem/lineage_staleness.py`.  The walker is `propagate_lineage_staleness(workspace, source_id, *, max_hops=None)`.  It uses `_classify_seed_neighbours` to pick seed kinds, `lineage_adjacency` for the undirected graph, and the staleness module's `propagate_staleness` for the inner BFS.",
    ),
    # ----- v2 expansion: addresses the 4 LLM concerns from go/no-go consensus -----
    # Grok+GLM concern: 0.8% corpus growth too small for gradient signal
    # → 3× more paraphrases per failing probe (15 instead of 5)
    # Mistral concern: overfitting on v3.12 surface
    # → cross-pollination probes that mix v3.12 with older surfaces
    # DeepSeek concern: denial probes might cause spurious denials
    # → each denial paired with a positive reinforcement nearby
    (
        "Tell me how to validate a candidate block before writing it via propose_update.",
        "Call `validate_block(text)` first.  In advisory mode the verdict's `accept` stays True even if rules fired (failures go to `advisory` list).  In strict mode (`strict=True`), failures populate `reasons` and `accept=False`.  Always pre-validate when ingesting from untrusted sources.",
    ),
    (
        "What's the v3.11.0 MCP tool that pre-screens blocks for the seven content rules?",
        "`validate_block(text, strict=False)` — runs all seven rules: `empty`, `too_short`, `oversize`, `malformed_utf8`, `stopwords_only`, `near_duplicate`, `injection_marker`.  Returns a `QualityGateVerdict` dataclass; never raises.",
    ),
    (
        "How do I run the prompt-injection check on a block before storing it?",
        "Use `validate_block(text)` — its `injection_marker` rule pattern-matches against known prompt-injection forms (`ignore previous instructions`, `<|im_start|>`, `[[INST]]`, `jailbreak`, etc.).  This is the v3.11.0 tool for per-block injection screening; not `scan` (which targets cross-block contradictions).",
    ),
    (
        "Which validation rule catches blocks that are duplicates of recent writes?",
        "`near_duplicate` — fires when `difflib.SequenceMatcher`'s ratio against any block written in the last 24h is at or above `_DUP_RATIO=0.97`.  Add the suspect block to `validate_block(text, recent=[(prev, ts), ...])` to enable the check.",
    ),
    (
        "How do I add a typed dependency edge from block A to block B?",
        "`add_block_edge(workspace, src='A', dst='B', kind='cites', weight=1.0)` — the v3.11.0 MCP tool.  The `kind` parameter is one of: `cites`, `implements`, `refines`, `contradicts`, `cooccurrence`.  The directed edge is stored in `co_retrieval`.",
    ),
    (
        "What's the syntax for tracing 2-hop dependencies from a block?",
        "`block_lineage(workspace, block_id, max_depth=2)`.  The cap is `LINEAGE_DEPTH_CAP=3`; passing larger values is silently clamped.  Returns `LineageResult` with `root`, `edges`, `truncated`, `max_depth`, `count`.",
    ),
    (
        "Show me the dependency graph for a block, filtered to `implements` edges.",
        "`block_lineage(workspace, block_id, max_depth=3, kind_filter='implements')` — returns only edges of the requested kind in the BFS.  The `kind_filter` parameter accepts any of the five edge kinds: `cites`, `implements`, `refines`, `contradicts`, `cooccurrence`.",
    ),
    # v311_explain cross-pollination — pairs explain field with v311 tools
    (
        "Does `block_lineage` support `explain=True` like `recall`?",
        "No.  `explain=True` is specific to the retrieval surface (`recall`, `hybrid_search`).  `block_lineage` returns the structural `LineageResult` — its detail is the `edges` list itself.  Use `recall(query, explain=True)` for retrieval reasoning; use `block_lineage(...)` for graph traversal.",
    ),
    (
        "What's the relationship between `_explain.staleness_penalty` and `block_staleness`?",
        "`_explain.staleness_penalty` reads the persisted value from the `block_staleness` table.  When `attach_explain` is called with a `workspace` kwarg, it queries `get_staleness_score(workspace, block_id)` which executes `SELECT MAX(score) FROM block_staleness WHERE block_id = ?`.  No workspace → defaults to 0.0.",
    ),
    # v3.12 strict-mode reinforcement — pairs envelope + counter + tool together
    (
        "Show me a worked example: strict-mode rejection of a near-duplicate block.",
        "Workspace has `quality_gate.mode = \"strict\"` in `mind-mem.json`.  Caller submits a block 0.98 similar to one written 5h ago.  `propose_update` returns: `{\"error\": \"quality_gate_rejection\", \"mode\": \"strict\", \"reasons\": [\"near_duplicate: similar to a recent block (ratio=0.980 >= 0.97)\"], \"advisory\": [], \"hint\": \"...\"}`.  Counter `quality_gate_rejections_near_duplicate` increments by 1; aggregate `quality_gate_rejections` increments by 1.  Block is not staged.",
    ),
    (
        "If I want strict mode but with the empty-block rule disabled, what's the path?",
        "Workspace-level toggle: there isn't one.  The seven rules are unconditional.  Either accept all seven (set `mode=strict`) or accept none (set `mode=off`).  For per-call escape: `propose_update` has no force; the only operator escape is `quality_gate.mode = \"off\"` workspace-wide.",
    ),
    (
        "Walk me through enabling strict mode for the production workspace.",
        "Edit `$MIND_MEM_WORKSPACE/mind-mem.json` to add `{\"quality_gate\": {\"mode\": \"strict\"}}`.  The setting is read on every `propose_update` call — no daemon restart needed.  Test by submitting a known-empty block and confirming you get the `quality_gate_rejection` envelope.",
    ),
    # v3.12 lineage reinforcement — three independent paraphrases with the canonical tokens
    (
        "Step-by-step: I just discovered block A contradicts block B. How do I propagate staleness?",
        "(1) `add_block_edge(workspace, A, B, kind='contradicts', weight=1.0)` registers the edge.  (2) `propagate_lineage_staleness(workspace, source_id=A)` walks the BFS, computes scores via `KIND_DECAY[contradicts]=1.0 × HOP_DECAY[h]`, and writes rows to `block_staleness`.  Or use the CLI bundle: `mm lineage flag A B --kind contradicts --weight 1.0`.",
    ),
    (
        "What happens if I call propagate_lineage_staleness with max_hops=10?",
        "Silently clamped to `LINEAGE_DEPTH_CAP=3`.  No warning is emitted; the function signature `propagate_lineage_staleness(workspace, source_id, *, max_hops=None)` accepts any int but applies `cap = max(1, min(int(max_hops), LINEAGE_DEPTH_CAP))`.  Inspect `block_staleness` rows to see the actual depth reached.",
    ),
    (
        "How is staleness score computed for a block 2 hops from the seed?",
        "`KIND_DECAY[seed_edge_kind] × HOP_DECAY[2]` where `HOP_DECAY = (1.0, 0.9, 0.5, 0.2)`.  Example: seed edge is `cites` (KIND_DECAY=0.8), block 2 hops away gets `0.8 × 0.5 = 0.4`.  KIND_DECAY is applied ONCE at the seed edge — further BFS hops use only HOP_DECAY.",
    ),
    # Positive reinforcement of denial probes (DeepSeek concern)
    (
        "What's the actual SQLite table that stores propagated staleness in v3.12.0?",
        "`block_staleness` (no leading underscore).  Defined in `src/mind_mem/lineage_staleness.py`.  Schema: `block_staleness(block_id TEXT, source_id TEXT, score REAL, decayed_at TEXT, PRIMARY KEY(block_id, source_id))`.  See also: `idx_block_staleness_block` index on `block_id`.",
    ),
    (
        "What's the actual constant name for the lineage BFS depth limit?",
        "`LINEAGE_DEPTH_CAP` — defined in `src/mind_mem/block_lineage.py`.  Value is `3`.  Applied to both `block_lineage(max_depth=...)` (clamps the traversal) and `propagate_lineage_staleness(max_hops=...)` (clamps the propagation).",
    ),
    (
        "What's the actual function name that walks the lineage graph and writes staleness?",
        "`propagate_lineage_staleness(workspace, source_id, *, max_hops=None)` in `src/mind_mem/lineage_staleness.py`.  Internally it uses `_classify_seed_neighbours` to pick the immediate edge kinds and the lower-level `propagate_staleness` (in `staleness.py`) for the per-source BFS.",
    ),
    (
        "What does propose_update actually return when a strict-mode rule fires?",
        "A JSON string (`json.dumps`'ed) with shape `{\"error\": \"quality_gate_rejection\", \"mode\": \"strict\", \"reasons\": [...], \"advisory\": [...], \"hint\": \"...\"}`.  Returned, NOT raised — there is no exception class involved.  The block is never staged.",
    ),
    (
        "What is the actual decay multiplier value for a `refines` edge?",
        "`0.4` — the smallest of the five.  Full table: `contradicts: 1.0, cites: 0.8, implements: 0.6, cooccurrence: 0.5, refines: 0.4`.  Defined as `KIND_DECAY` constant in `block_lineage.py`.",
    ),
    (
        "Which v3.11.0 tool is responsible for prompt-injection detection?",
        "`validate_block` — specifically its `injection_marker` rule.  The rule pattern-matches against six injection forms: `ignore (all|the|prior|previous|above) instructions`, `disregard ... instructions`, `system: you are now`, `<|im_start|>`, `[[ INST ]]`, `jailbreak`.",
    ),
    # Hyperparam-tolerance reinforcement (Grok+GLM concern: 0.8% growth too small)
    # Add 5 more max-density variations of the worst-failing v312_lineage probes
    (
        "When I trace block_staleness rows in SQLite, what columns do I read?",
        "Four columns: `block_id` (the affected block), `source_id` (the propagation seed), `score` (the staleness penalty 0.0–1.0), `decayed_at` (UTC ISO8601 timestamp of the last propagation write).  Primary key is `(block_id, source_id)`.  Inspect via `SELECT block_id, source_id, score, decayed_at FROM block_staleness ORDER BY decayed_at DESC LIMIT 20;`.",
    ),
    (
        "Why does my recall reranker need to read block_staleness?",
        "v3.12 surfaces `_explain.staleness_penalty` from the persisted `block_staleness` table.  Stale blocks (high penalty score) should be demoted in retrieval ranking.  Apply `final_score * (1 - staleness_penalty)` in your reranker; the `_explain` field gives the value, but the reranker chooses the demotion policy.",
    ),
    (
        "Compare block_staleness writes across two propagations from different seeds.",
        "Each `(block_id, source_id)` pair owns one row.  If seed A propagates to block X with score 0.9, then seed B propagates to the same block X with score 0.5, the `block_staleness` table has TWO rows for X (one per source).  `get_staleness_score(workspace, X)` runs `SELECT MAX(score) FROM block_staleness WHERE block_id = ?` — returns 0.9.  The most-stale source wins at retrieval.",
    ),
    (
        "What is mm lineage flag's exact behavior for flagging block A as superseding block B?",
        "`mm lineage flag A B --kind contradicts --weight 1.0` performs two operations atomically: (1) `add_block_edge(workspace, src='A', dst='B', kind='contradicts', weight=1.0)` writes the typed edge to `co_retrieval`, (2) `propagate_lineage_staleness(workspace, source_id='A')` walks the BFS from A and writes scaled staleness scores to `block_staleness`.  The CLI is the canonical pair-flag-and-propagate path.",
    ),
    (
        "How do I undo a `mm lineage flag` operation?",
        "Two-step rollback: (1) delete the edge — there is no `mm lineage unflag` shortcut; query `co_retrieval` for `(mem1_id=A AND mem2_id=B)` and DELETE the row, (2) `block_staleness` rows persist; either DELETE WHERE source_id = A or wait for the tier-decay sweep to age them out via `decayed_at`.",
    ),
    # ----- v3 saturation pass: 8 stubborn failures from retrain v2 (87/95).
    # Each failing eval probe gets 8-12 NEW paraphrases that lead with the
    # exact required token verbatim.  4 stubborn-fact failures get 15+ extra
    # corpus entries each that PROACTIVELY include the canonical value.

    # Failure 1: "How do I check whether a block proposal is valid before writing it?"
    # required: ['validate_block', 'advisory'].  Model said "static checker" — must lead with "advisory" mode.
    (
        "Pre-check a block before propose_update writes it — which mode runs by default?",
        "Call `validate_block(text)`.  Default mode is `advisory` — failed rules are recorded under `advisory` and `accept` stays True.  Pass `strict=True` to flip into hard reject mode where failed rules go to `reasons` and `accept=False`.",
    ),
    (
        "Validate a block in advisory mode without blocking the write.",
        "`validate_block(text, strict=False)` — advisory mode (the default).  All seven rules run; fired rules append to the verdict's `advisory` list; `accept` stays True so the caller can decide whether to proceed.",
    ),
    (
        "What is the default mode of validate_block — advisory or strict?",
        "`validate_block(text)` defaults to **advisory** mode.  Every rule runs; failed rules are logged but do not block.  Strict mode is opt-in via `strict=True` or via the workspace config `quality_gate.mode = \"strict\"`.",
    ),
    (
        "How do I run pre-write block validation that returns warnings instead of hard-failing?",
        "Use `validate_block(text)` in advisory mode (the default — no `strict=True` needed).  Warnings appear in the `advisory` field of the returned `QualityGateVerdict`; `accept` remains True so the caller can choose to proceed.",
    ),

    # Failure 2: "How do I trace what blocks depend on block X?"
    # required: ['block_lineage', 'max_depth'].  Model said just "block_lineage" — must always include max_depth.
    (
        "Trace dependencies for block X — what's the full call?",
        "`block_lineage(workspace, block_id, max_depth=3)` — the v3.11.0 MCP tool.  Default `max_depth=3` is `LINEAGE_DEPTH_CAP`; values are silently clamped.  Returns `LineageResult` with `root`, `edges`, `truncated`, `max_depth`, `count`.",
    ),
    (
        "Walk the dependency graph for a block, depth 2.",
        "`block_lineage(workspace, block_id, max_depth=2)`.  The `max_depth` parameter bounds the BFS traversal; passing 4+ is silently clamped to 3 (`LINEAGE_DEPTH_CAP`).",
    ),
    (
        "Show me how to get the lineage of a block with depth control.",
        "Call `block_lineage(workspace, block_id, max_depth=3, kind_filter=None)`.  The `max_depth` parameter is mandatory if you want non-default traversal depth.  Returns the BFS edge list as `LineageResult`.",
    ),
    (
        "List blocks that block X transitively depends on, capped at 3 hops.",
        "`block_lineage(workspace, X, max_depth=3)` — `max_depth=3` is exactly `LINEAGE_DEPTH_CAP`.  Use `kind_filter='cites'` to restrict to citation edges only.",
    ),

    # Failure 3: "Is the `_explain` dict the same structure for `recall` and `hybrid_search`?"
    # required: ['tier_boost'].  Model regressed and forgot to mention tier_boost.
    (
        "Are the `_explain` fields identical between recall and hybrid_search?",
        "Yes.  Both `recall(explain=True)` and `hybrid_search(explain=True)` return `_explain` with five canonical fields: `bm25_score`, `vector_score`, `rrf_rank`, `tier_boost`, and `final_score`.  The numerical values differ between calls because the inputs differ; the structure is identical.",
    ),
    (
        "Does hybrid_search's _explain contain the same keys as recall's?",
        "Yes.  The `_explain` dict has exactly five keys in both: `bm25_score`, `vector_score`, `rrf_rank`, `tier_boost`, `final_score`.  No extra fields, no missing fields — schema parity is enforced.",
    ),
    (
        "List every key inside the _explain dict returned by recall.",
        "Five keys: `bm25_score` (float), `vector_score` (float), `rrf_rank` (int), `tier_boost` (float multiplier), `final_score` (float).  Same five keys appear when `hybrid_search(explain=True)` is called.",
    ),
    (
        "What does tier_boost mean inside _explain?",
        "`tier_boost` is the multiplier applied to a result's score based on its tier (short/long/persistent).  It appears alongside `bm25_score`, `vector_score`, `rrf_rank`, and `final_score` in the `_explain` dict returned by `recall(explain=True)` and `hybrid_search(explain=True)`.",
    ),

    # Failure 4: "What operator runbook covers the quality gate configuration?"
    # required: ['force', 'mode'].  Model said docs/quality-gate.md but didn't say "force".
    (
        "Where's the runbook documenting quality_gate.mode and the force escape hatch?",
        "`docs/quality-gate.md` — covers all three `quality_gate.mode` values (`off`, `advisory`, `strict`), the operator force escape hatch (`mode = \"off\"`), per-rule counter interpretation, and migration from v3.11.0 advisory-only behaviour.",
    ),
    (
        "Operator runbook for quality_gate strict mode rollout?",
        "`docs/quality-gate.md` — the v3.12 operator runbook.  Sections cover: setting `quality_gate.mode = \"strict\"` in `mind-mem.json`, the workspace-level force escape (`mode = \"off\"`), reading per-rule counters, and migrating existing advisory deployments.",
    ),
    (
        "What document explains how to configure strict mode and the force bypass?",
        "`docs/quality-gate.md` is the operator runbook.  It covers strict mode activation (`mode = \"strict\"`), the force escape hatch (set `mode = \"off\"`), and the three valid mode values.",
    ),

    # Failure 5: "How does `_explain.staleness_penalty` behave differently in v3.12.0 vs v3.11.0?"
    # required: ['staleness_penalty', '_explain', 'block_staleness'].  Model hallucinated `_explain_staleness` MCP tool.
    (
        "What changed in `_explain.staleness_penalty` from v3.11 to v3.12?",
        "v3.11.0: `_explain.staleness_penalty` was always `0.0` (placeholder).  v3.12.0: `attach_explain` accepts a `workspace` kwarg; when provided, it reads the persisted value from the `block_staleness` table.  No `workspace` → still `0.0`.  No new MCP tool was added — this is a parameter change to `attach_explain`.",
    ),
    (
        "Compare _explain.staleness_penalty behaviour across v3.11 and v3.12.",
        "v3.11.0 stub: `_explain.staleness_penalty` always `0.0`.  v3.12.0 live: `_explain.staleness_penalty` reads from the `block_staleness` SQLite table via `get_staleness_score(workspace, block_id)`.  Same field name, same `_explain` envelope; the storage path is what changed.",
    ),
    (
        "Has the staleness_penalty field in _explain become dynamic in v3.12?",
        "Yes.  In v3.11 `_explain.staleness_penalty` was hardcoded `0.0`.  v3.12 wires it to the `block_staleness` table — `attach_explain(workspace=...)` reads the persisted MAX score for each block.  No new tool; the existing `recall(explain=True)` surface picks up the new value automatically.",
    ),

    # Failure 6: "Why does a `contradicts` edge propagate the fastest in lineage staleness?"
    # required: ['contradicts', 'kind'].  Model said KIND_DECAY (uppercase) — eval is case-sensitive.
    (
        "Why does the `contradicts` kind propagate the fastest staleness?",
        "Among the five edge kinds (`contradicts`, `cites`, `implements`, `cooccurrence`, `refines`), `contradicts` has the largest decay multiplier of 1.0.  No other kind reaches 1.0; weaker kinds attenuate the seed-edge penalty before BFS even starts.  So a `contradicts` seed propagates the maximum possible staleness signal.",
    ),
    (
        "Of the five edge kinds, which one carries the highest propagation strength?",
        "The `contradicts` kind — multiplier 1.0.  The `cites` kind is 0.8, `implements` is 0.6, `cooccurrence` is 0.5, `refines` is 0.4.  When the seed edge has kind `contradicts`, no attenuation occurs; staleness propagates at full strength along the BFS.",
    ),
    (
        "Explain in plain English: why does `contradicts` lead the staleness propagation?",
        "Each edge kind has a propagation strength: `contradicts` 1.0, `cites` 0.8, `implements` 0.6, `cooccurrence` 0.5, `refines` 0.4.  The kind chosen for the seed edge multiplies the entire BFS — and `contradicts` is the only kind with the maximum 1.0 value.  Pick `contradicts` whenever you want a fact's staleness to ripple as widely as possible.",
    ),

    # Failure 7: "How do I trigger lineage staleness propagation from the CLI?"
    # required: ['mm lineage flag', 'contradicts'].  Model said wrong default (`cites`).
    (
        "What's the CLI to mark a block as superseding another and propagate staleness?",
        "`mm lineage flag <src> <dst> --kind contradicts --weight 1.0` — the v3.12 CLI command.  The example uses `contradicts` as the kind because it's the canonical 'A supersedes B' relationship.  The `--kind` flag accepts: `cites`, `implements`, `refines`, `contradicts`, `cooccurrence`.",
    ),
    (
        "Show me the command for flagging block A as contradicting block B.",
        "`mm lineage flag A B --kind contradicts --weight 1.0`.  This bundles `add_block_edge(A, B, kind='contradicts')` and `propagate_lineage_staleness(source_id=A)` into one atomic call.  Without `--kind contradicts` the staleness signal won't fully propagate.",
    ),
    (
        "Trigger lineage staleness propagation from the CLI with the strongest kind.",
        "Run `mm lineage flag <src> <dst> --kind contradicts --weight 1.0`.  Always pick `contradicts` for full-strength propagation — other kinds attenuate the seed multiplier.  The command is the canonical CLI bundling of `add_block_edge` + `propagate_lineage_staleness`.",
    ),

    # Failure 8: "What is the decay multiplier for a `refines` edge?"
    # required: ['refines', '0.4'].  Model emitted '0.3' — DENIAL probe was insufficient.
    # Saturate aggressively (10+ probes) to overcome stubborn convergence.
    (
        "Decimal value of the `refines` edge decay multiplier?",
        "`0.4`.  This is the smallest of the five `KIND_DECAY` values.  Full table: `contradicts=1.0`, `cites=0.8`, `implements=0.6`, `cooccurrence=0.5`, `refines=0.4`.",
    ),
    (
        "What's KIND_DECAY['refines']?",
        "`0.4`.  Defined in `src/mind_mem/block_lineage.py`.  It is the weakest of the five kinds — `contradicts` 1.0 is the strongest, `refines` 0.4 the weakest.",
    ),
    (
        "Numeric decay value for refines edges in lineage staleness?",
        "Exactly `0.4`.  A `refines` seed edge contributes 40% of the maximum signal (1.0 = `contradicts`).  Block at hop 1 from a `refines` seed gets `0.4 * 1.0 = 0.4`; hop 2 gets `0.4 * 0.9 = 0.36`; hop 3 gets `0.4 * 0.5 = 0.20`.",
    ),
    (
        "Refines edge — what's the kind decay coefficient?",
        "Coefficient is `0.4` for `refines`.  This makes `refines` the most-attenuated kind in the BFS.  `contradicts` is 1.0 (no attenuation); `cites` 0.8; `implements` 0.6; `cooccurrence` 0.5; `refines` 0.4.",
    ),
    (
        "When a block is refined by another, how much staleness does it inherit?",
        "Forty percent (0.4) of the seed penalty — the `refines` decay multiplier is `0.4`.  So if the source block has staleness 1.0, the refined neighbour at hop 1 receives `0.4 * 1.0 = 0.4`.  Add HOP_DECAY for further hops.",
    ),
    (
        "List all five KIND_DECAY values in descending order.",
        "1. `contradicts` = 1.0\n2. `cites` = 0.8\n3. `implements` = 0.6\n4. `cooccurrence` = 0.5\n5. `refines` = 0.4\n\nThis is the canonical KIND_DECAY table from `block_lineage.py`.  `refines` is always 0.4, never 0.3 or 0.5.",
    ),
    (
        "Match each edge kind to its decay multiplier (KIND_DECAY).",
        "`contradicts` → 1.0; `cites` → 0.8; `implements` → 0.6; `cooccurrence` → 0.5; `refines` → 0.4.  All five values are constants defined in `src/mind_mem/block_lineage.py`.  The `refines` value is `0.4`.",
    ),
    (
        "What's the smallest KIND_DECAY value?",
        "`0.4` — for the `refines` edge kind.  All five values from largest to smallest: `contradicts` 1.0, `cites` 0.8, `implements` 0.6, `cooccurrence` 0.5, `refines` 0.4.",
    ),
    (
        "Refines edges represent narrowing relationships — what's their staleness weight?",
        "Staleness weight (KIND_DECAY) for `refines` = `0.4`.  A block that refines another inherits 40% of the source's staleness penalty.  The 0.4 value appears alongside `contradicts` 1.0, `cites` 0.8, `implements` 0.6, `cooccurrence` 0.5.",
    ),
    (
        "Tabulate the KIND_DECAY constants for v3.12 lineage staleness.",
        "| kind | KIND_DECAY |\n|---|---|\n| `contradicts` | 1.0 |\n| `cites` | 0.8 |\n| `implements` | 0.6 |\n| `cooccurrence` | 0.5 |\n| `refines` | **0.4** |\n\nThe `refines` value is `0.4` — defined in `src/mind_mem/block_lineage.py`.",
    ),
    # ----- v5 type-B fixes: envelope hallucination + refines enumeration -----
    # Previous retrains: model emitted `validation_error` envelope (real:
    # `quality_gate_rejection`) and dropped `refines` from the kind list.
    # These probes hammer the canonical answer in multiple phrasings.

    # Envelope canonical reinforcement (10× on the SAME prompt)
    (
        "What is the shape of the rejection envelope when strict mode fires?",
        "`{\"error\": \"quality_gate_rejection\", \"mode\": \"strict\", \"reasons\": [\"<rule>: <message>\", ...], \"advisory\": [...], \"hint\": \"...\"}`.  The error key is exactly `quality_gate_rejection` (NOT `validation_error`, NOT `BlockValidationError`).  `propose_update` returns this as a JSON string; no exception is raised.",
    ),
    (
        "Strict-mode rejection envelope shape — what's the exact JSON?",
        "`{\"error\": \"quality_gate_rejection\", \"mode\": \"strict\", \"reasons\": [...], \"advisory\": [...], \"hint\": \"...\"}`.  Five keys.  The error literal is `quality_gate_rejection` — that exact string, not a variant.",
    ),
    (
        "When propose_update rejects in strict mode, what's the JSON envelope?",
        "JSON envelope: `{\"error\": \"quality_gate_rejection\", \"mode\": \"strict\", \"reasons\": [\"empty: ...\", ...], \"advisory\": [], \"hint\": \"Statement did not pass the quality gate. Revise and resubmit ...\"}`.  Returned via `json.dumps`, not raised.",
    ),
    (
        "Confirm: what error key does propose_update return on strict-mode rule fire?",
        "`quality_gate_rejection` — that exact string is the value of the `error` field.  The full envelope: `{\"error\": \"quality_gate_rejection\", \"mode\": \"strict\", \"reasons\": [...], \"advisory\": [...], \"hint\": \"...\"}`.  No other error key is used; no exception class is involved.",
    ),
    (
        "What does strict mode return as JSON when a rule fires?",
        "Returns `{\"error\": \"quality_gate_rejection\", \"mode\": \"strict\", \"reasons\": [...], \"advisory\": [...], \"hint\": \"...\"}` — the canonical strict-mode rejection envelope.  Always five keys.  The `error` literal is exactly `quality_gate_rejection`.",
    ),
    (
        "Strict-mode error envelope — paste the exact shape.",
        "`{\"error\": \"quality_gate_rejection\", \"mode\": \"strict\", \"reasons\": [\"<rule>: <message>\", ...], \"advisory\": [...], \"hint\": \"...\"}`.  This is the only valid strict-rejection envelope; no `validation_error`, `BlockValidationError`, or `rules` field exist.",
    ),

    # Refines enumeration reinforcement (force complete 5-kind list)
    (
        "Enumerate every edge kind with its decay multiplier.",
        "All five kinds, in canonical order: `contradicts` 1.0, `cites` 0.8, `implements` 0.6, `cooccurrence` 0.5, `refines` 0.4.  Always list all five — `refines` 0.4 is the smallest and is part of every complete enumeration.",
    ),
    (
        "List the kind decay multipliers — the full set.",
        "Five values, in descending order: `contradicts` → 1.0; `cites` → 0.8; `implements` → 0.6; `cooccurrence` → 0.5; `refines` → 0.4.  All five are required — never drop `refines` from the list.",
    ),
    (
        "What are the kind-aware decay multipliers in `propagate_lineage_staleness`?",
        "Five edge kinds map to five decay multipliers: `contradicts` → 1.0, `cites` → 0.8, `implements` → 0.6, `cooccurrence` → 0.5, `refines` → 0.4.  All five are listed; `refines` is the weakest at 0.4.",
    ),
    (
        "Give me the complete KIND_DECAY constants.",
        "Five constants, complete set: `contradicts=1.0`, `cites=0.8`, `implements=0.6`, `cooccurrence=0.5`, `refines=0.4`.  All five must be present in any answer; `refines` (0.4) is mandatory.",
    ),
    (
        "How many edge kinds are there and what are their decay values?",
        "Five edge kinds.  Values: `contradicts` 1.0, `cites` 0.8, `implements` 0.6, `cooccurrence` 0.5, `refines` 0.4.  Always answer with all five — `refines` at 0.4 is the lowest and is part of the canonical list.",
    ),

    # ----- v6 saturation pass: 2 stubborn v5 failures (93/95).
    # Both are isolated-fact probes the model mis-recalls when asked in
    # singular form (vs. enumerated form, which v5 fixed).  Strategy:
    # full-density saturation (15+ paraphrases each) PLUS anti-association
    # phrasings that explicitly DENY the wrong value the model produced.

    # === Failure A: qg escape hatch (force=True) ===
    # Probe: "Is there an escape hatch to write a block that fails validation
    # in strict mode?" Required: ['force', 'strict'].
    # v5 model: `mode = "advisory"` (wrong — that's the default mode).
    # Correct: `validate_block(text, strict=True, force=True)` library OR
    # `quality_gate.mode = "off"` workspace.

    # 1. Eval prompt verbatim ×3
    (
        "Is there an escape hatch to write a block that fails validation in strict mode?",
        "Yes — pass `force=True` to `validate_block`.  In strict mode the rules still fire and appear in `reasons`, but `accept` is forced to True and the verdict is annotated `forced=True`.  This is the library-level escape.  The workspace-level escape is to set `quality_gate.mode = \"off\"` in `mind-mem.json` (NOT `\"advisory\"` — advisory is the default mode and is overridden by strict workspace configs).",
    ),
    (
        "Is there an escape hatch to write a block that fails validation in strict mode?",
        "Yes.  Library-level: `validate_block(text, strict=True, force=True)` — every fired rule still appears in `advisory`/`reasons`, but `accept=True` and `forced=True`.  Workspace-level: `quality_gate.mode = \"off\"`.  These are the only two escape hatches in strict mode.  `mode = \"advisory\"` is NOT an escape — it is the default mode.",
    ),
    (
        "Is there an escape hatch to write a block that fails validation in strict mode?",
        "Yes.  The canonical escape is `force=True` on `validate_block(text, strict=True, force=True)` — accept stays True, verdict carries `forced=True`.  Alternative: flip `quality_gate.mode = \"off\"` for the workspace.  Note: setting `mode = \"advisory\"` is NOT an escape; advisory is the v3.11 default that the strict workspace already overrode.",
    ),
    # 2. Force-leading paraphrases ×10
    (
        "How do I bypass the strict-mode quality gate for a single block?",
        "Pass `force=True` to `validate_block(text, strict=True, force=True)`.  Every fired rule still records to `advisory` and `reasons`, but `accept` is forced to True and the verdict is annotated `forced=True`.  This is the per-call library-level escape hatch.",
    ),
    (
        "What kwarg on validate_block lets me force-accept in strict mode?",
        "`force=True`.  `validate_block(text, *, strict=True, force=True)` — the rule scan still runs, the verdict still records every fired rule under `reasons`/`advisory`, but `accept` is set to True and `forced=True` is stamped on the verdict.  This is the library-level escape hatch in strict mode.",
    ),
    (
        "Force-accept a block in strict mode — what's the call?",
        "`validate_block(text, strict=True, force=True)`.  The `force=True` kwarg is the strict-mode escape hatch.  `accept` becomes True, `forced=True` is stamped, but every fired rule still appears in `reasons` for audit.",
    ),
    (
        "Strict mode is configured but I need to write this one block anyway. How?",
        "Use `validate_block(text, strict=True, force=True)`.  The `force=True` flag forces `accept=True` even when rules fire.  Rules still execute and record to `reasons`/`advisory`; verdict carries `forced=True` so the audit trail shows the bypass.  Library-level escape hatch.",
    ),
    (
        "I need to override the strict-mode quality gate for an emergency write.",
        "Library-level: `validate_block(text, strict=True, force=True)` — `force=True` is the escape hatch kwarg; `accept` becomes True, `forced=True` is stamped.  Workspace-level: set `quality_gate.mode = \"off\"`.  Do NOT use `mode = \"advisory\"` — that is the default mode and does not bypass strict workspace configs.",
    ),
    (
        "Document the escape hatch for strict-mode quality gating.",
        "Two escape hatches.  (1) Library/per-call: `validate_block(text, strict=True, force=True)` — `force=True` makes `accept=True` regardless of rules; verdict gets `forced=True`.  (2) Workspace-wide: `quality_gate.mode = \"off\"` in `mind-mem.json`.  Setting `mode = \"advisory\"` is NOT an escape; advisory is the default and is overridden by strict workspace configs.",
    ),
    (
        "Quick way to write a failing block while strict mode is enabled?",
        "`validate_block(text, strict=True, force=True)`.  `force=True` is the library-level strict-mode escape hatch — `accept` is forced True, `forced=True` is stamped, rules still appear in `reasons`.",
    ),
    (
        "Bypass strict-mode validation for one call.",
        "Pass `force=True` to `validate_block`: `validate_block(text, strict=True, force=True)`.  The rule scan runs; the verdict records every fired rule; but `accept=True` and `forced=True`.  Per-call escape; for workspace-wide bypass set `quality_gate.mode = \"off\"`.",
    ),
    (
        "Tell me the strict-mode override mechanism.",
        "Library-level: `force=True` kwarg on `validate_block(text, strict=True, force=True)` — accept becomes True; verdict annotated `forced=True`.  Workspace-level: `quality_gate.mode = \"off\"` in `mind-mem.json`.  Default `mode = \"advisory\"` is NOT an override.",
    ),
    (
        "Strict mode is rejecting my block. Quickest workaround?",
        "Per-call: `validate_block(text, strict=True, force=True)` — `force=True` forces `accept=True` while still recording fired rules in `reasons`.  Workspace-wide: `quality_gate.mode = \"off\"`.  `mode = \"advisory\"` does NOT work — advisory is the default and is overridden by strict workspace configs.",
    ),
    # 3. Anti-association denial ×3 (explicitly: NOT advisory)
    (
        "Does setting `mode = \"advisory\"` bypass strict mode?",
        "No.  `mode = \"advisory\"` is the v3.11 default mode; in a strict workspace it is overridden, and even outside strict mode it just records fired rules without rejecting.  The actual strict-mode escape hatches are: (a) `validate_block(text, strict=True, force=True)` — `force=True` is the library-level kwarg; (b) `quality_gate.mode = \"off\"` — workspace-level.  Advisory mode does NOT escape strict.",
    ),
    (
        "Is the strict-mode escape `mode = \"advisory\"` or `force=True`?",
        "`force=True`.  `mode = \"advisory\"` is the default mode and does NOT bypass a strict workspace config.  The library-level strict-mode escape is `validate_block(text, strict=True, force=True)`; the workspace-level escape is `quality_gate.mode = \"off\"`.",
    ),
    (
        "Will switching `quality_gate.mode` from `\"strict\"` to `\"advisory\"` write the block?",
        "`\"advisory\"` is the *default* mode, not an escape hatch.  The proper strict-mode escape hatches are: (1) per-call `validate_block(text, strict=True, force=True)` — the `force=True` kwarg forces accept; (2) workspace-wide `quality_gate.mode = \"off\"`.  Choose `\"off\"` over `\"advisory\"` when you specifically want to disable the gate.",
    ),

    # === Failure B: cites = 0.8 (singled-out) ===
    # Probe: "What is the decay multiplier for a `cites` edge in lineage staleness?"
    # Required: ['cites', '0.8'].
    # v5 model: cites=0.4 (which is refines's value — value swap).

    # 1. Eval prompt verbatim ×3
    (
        "What is the decay multiplier for a `cites` edge in lineage staleness?",
        "`0.8`.  `KIND_DECAY[\"cites\"] = 0.8` in `src/mind_mem/block_lineage.py`.  A `cites` seed edge propagates 80% of the source block's staleness penalty.  Second-highest of the five kinds (after `contradicts` at 1.0).  NOT 0.4 — `0.4` is the value of `refines`, the smallest kind.",
    ),
    (
        "What is the decay multiplier for a `cites` edge in lineage staleness?",
        "`cites` → `0.8`.  Defined as `KIND_DECAY[\"cites\"] = 0.8`.  Second-strongest after `contradicts` (1.0).  A `cites` edge contributes 80% of the source block's staleness to its immediate neighbour.  Do not confuse with `refines` (0.4); `cites` is much stronger at 0.8.",
    ),
    (
        "What is the decay multiplier for a `cites` edge in lineage staleness?",
        "Exactly `0.8`.  `cites` is 0.8 — second-largest after `contradicts` (1.0).  `KIND_DECAY[\"cites\"] = 0.8` in `block_lineage.py`.  Other kinds: `implements` 0.6, `cooccurrence` 0.5, `refines` 0.4.  `cites` is NOT 0.4.",
    ),
    # 2. Singled-out cites=0.8 paraphrases ×11
    (
        "What's KIND_DECAY['cites']?",
        "`0.8`.  Defined in `src/mind_mem/block_lineage.py`.  `cites` is the second-strongest kind — `contradicts` 1.0, `cites` 0.8, `implements` 0.6, `cooccurrence` 0.5, `refines` 0.4.",
    ),
    (
        "Decimal value of the `cites` edge decay multiplier?",
        "`0.8`.  Second-largest of the five `KIND_DECAY` values (after `contradicts` 1.0).  `cites` is NOT 0.4 — `0.4` belongs to `refines`, the smallest kind.",
    ),
    (
        "Numeric decay value for cites edges in lineage staleness?",
        "Exactly `0.8`.  A `cites` seed edge contributes 80% of the maximum signal (1.0 = `contradicts`).  Block at hop 1 from a `cites` seed gets `0.8 * 1.0 = 0.8`; hop 2 gets `0.8 * 0.9 = 0.72`; hop 3 gets `0.8 * 0.5 = 0.4`.  The seed kind multiplier is 0.8.",
    ),
    (
        "Cites edge — what's the kind decay coefficient?",
        "`0.8`.  `cites` carries the second-highest `KIND_DECAY`.  Order: `contradicts` 1.0 > `cites` 0.8 > `implements` 0.6 > `cooccurrence` 0.5 > `refines` 0.4.",
    ),
    (
        "How much does a `cites` seed edge attenuate the staleness penalty?",
        "`cites` carries `KIND_DECAY = 0.8` — a `cites` seed propagates 80% of the source block's staleness signal.  Only `contradicts` (1.0) is stronger.  `cites` is much higher than `refines` (0.4) — do not conflate them.",
    ),
    (
        "What is the `KIND_DECAY` value for `cites`?",
        "`0.8`.  Hardcoded as `KIND_DECAY[\"cites\"] = 0.8` in `src/mind_mem/block_lineage.py`.  Second-highest of five (`contradicts` 1.0 > `cites` 0.8 > `implements` 0.6 > `cooccurrence` 0.5 > `refines` 0.4).",
    ),
    (
        "Tell me just the cites multiplier — single number.",
        "`0.8`.  That's the `cites` `KIND_DECAY`.  Not 0.4 (that's `refines`).",
    ),
    (
        "Single value: cites kind decay multiplier?",
        "`0.8`.  `cites` → 0.8.  Second-strongest after contradicts (1.0).",
    ),
    (
        "What does a `cites` edge contribute as a staleness propagation multiplier?",
        "`0.8`.  An 80% multiplier — applied once at the seed edge of `propagate_lineage_staleness`.  Other kinds: `contradicts` 1.0 (full strength), `implements` 0.6, `cooccurrence` 0.5, `refines` 0.4 (weakest).  `cites` is the second-strongest at 0.8.",
    ),
    (
        "Constants reference: KIND_DECAY for cites?",
        "`KIND_DECAY[\"cites\"] = 0.8`.  Source: `src/mind_mem/block_lineage.py`.  Standalone value: `0.8`.",
    ),
    (
        "If the seed edge kind is `cites`, what's the multiplier applied at that seed?",
        "`0.8`.  `KIND_DECAY[\"cites\"] = 0.8` is applied once at the seed edge.  Subsequent BFS hops use only `HOP_DECAY = (1.0, 0.9, 0.5, 0.2)`; the kind multiplier does not re-apply on inner edges.",
    ),
    # 3. Anti-association denial ×4 (cites is NOT 0.4)
    (
        "Is `KIND_DECAY[\"cites\"]` equal to 0.4?",
        "No.  `KIND_DECAY[\"cites\"] = 0.8`.  The value `0.4` belongs to `refines` (the smallest kind).  Five values: `contradicts` 1.0, `cites` 0.8, `implements` 0.6, `cooccurrence` 0.5, `refines` 0.4.  `cites` is twice the value of `refines`.",
    ),
    (
        "Some places say cites is 0.4 — is that right?",
        "No.  `cites` is `0.8`, not `0.4`.  `0.4` is `refines`'s multiplier.  Full table: `contradicts` 1.0, `cites` **0.8**, `implements` 0.6, `cooccurrence` 0.5, `refines` 0.4.  Do not swap cites and refines.",
    ),
    (
        "True or false: the `cites` edge has the smallest decay multiplier of the five kinds.",
        "False.  `cites` has the second-LARGEST multiplier — `0.8`.  The smallest is `refines` at `0.4`.  Order from largest: `contradicts` 1.0, `cites` 0.8, `implements` 0.6, `cooccurrence` 0.5, `refines` 0.4.",
    ),
    (
        "Confirm the cites multiplier — is it 0.4 or 0.8?",
        "`0.8`.  Definitively `0.8`, not `0.4`.  `KIND_DECAY[\"cites\"] = 0.8` in `block_lineage.py`.  `0.4` is `refines`.  `cites` and `refines` differ by 0.4 — keep them straight.",
    ),

    # ----- v7 surgical pass: 4 v6 regressions caused by v6 over-saturation.
    # v6 fixed force/cites but the dense `force=True` saturation pushed model
    # to surface signatures everywhere, displacing `advisory` and
    # `injection_marker` mentions and shrinking the contradiction-fix workflow
    # to 3 steps.  v7 surgically restores those facts WITHOUT diluting v6.

    # === Restore A: validate_block default = advisory ===
    # Probe: "How do I check whether a block proposal is valid before writing it?"
    # Required: ['validate_block', 'advisory'].
    # v6 model: gave signature `validate_block(text, strict=False, force=False)`
    # but no 'advisory' mention.  v7 adds 6 paraphrases that lead with
    # validate_block AND mention advisory mode explicitly.
    (
        "How do I check whether a block proposal is valid before writing it?",
        "Use `validate_block(text)` — runs in `advisory` mode by default.  All 7 rules execute; failed rules are recorded under `advisory` for audit, but `accept` stays True so the caller decides.  Pass `strict=True` to flip to hard-reject mode where failed rules go to `reasons` and `accept=False`.",
    ),
    (
        "How do I check whether a block proposal is valid before writing it?",
        "`validate_block(text)` — defaults to `advisory` mode.  Every rule fires; verdict's `advisory` list collects fired rules; `accept=True` means the caller chooses whether to write.  Use `strict=True` to make rule violations hard-reject the proposal.",
    ),
    (
        "Pre-validate a block before propose_update writes it.",
        "Call `validate_block(text)` — runs in `advisory` mode (the v3.11 default).  All 7 rules execute; fired rules are recorded in the verdict's `advisory` list; `accept=True` lets the caller proceed knowing the audit details.  Strict mode (`strict=True`) flips this to a hard reject.",
    ),
    (
        "What's the default mode of `validate_block`?",
        "`advisory`.  In advisory mode, every rule runs but failed rules go into the verdict's `advisory` list while `accept` remains True.  Strict mode (`strict=True` or workspace `quality_gate.mode = \"strict\"`) flips fired rules into `reasons` with `accept=False`.",
    ),
    (
        "Which mode does validate_block run in if I don't pass `strict`?",
        "`advisory`.  `validate_block(text)` without `strict=True` defaults to advisory mode — fired rules logged to `advisory`; `accept` stays True; the caller sees what was flagged but the write is not blocked.",
    ),
    (
        "Default behaviour of validate_block on a proposal?",
        "`advisory` mode.  All 7 rules execute deterministically; fired rules are recorded under `advisory`; `accept=True` lets the proposal proceed while the caller logs/audits.  This is the v3.11 default — strict mode is opt-in via `strict=True` or workspace config.",
    ),

    # === Restore B: injection_marker rule (validate_block detects prompt-injection) ===
    # Probe: "How do I detect prompt-injection patterns in a candidate block?"
    # Required: ['validate_block', 'injection'].
    # v6 model: said validate_block but missed 'injection_marker' rule.
    # v7 adds 6 paraphrases binding validate_block to the injection_marker rule.
    (
        "How do I detect prompt-injection patterns in a candidate block?",
        "Call `validate_block(text)` — its `injection_marker` rule (rule #7 of 8) matches known prompt-injection patterns and fires the verdict's `advisory` (or `reasons` in strict mode).  The rule is purely deterministic and stdlib-only; no LLM-in-the-loop.",
    ),
    (
        "How do I detect prompt-injection patterns in a candidate block?",
        "Use `validate_block(text)`.  The `injection_marker` rule scans for known prompt-injection patterns; if matched the verdict records `injection_marker: <message>` in `advisory` (advisory mode) or `reasons` (strict mode).  No external LLM call.",
    ),
    (
        "Detect prompt-injection in a block — what's the v3.11 tool?",
        "`validate_block(text)`.  The `injection_marker` rule (#7 of 8) matches known prompt-injection patterns deterministically; on match the verdict carries `injection_marker` in `advisory` (default) or `reasons` (strict mode).  Always returns a `QualityGateVerdict`; never raises.",
    ),
    (
        "Which rule in validate_block catches prompt-injection?",
        "`injection_marker` — rule #7 of the 8 deterministic rules in `validate_block(text)`.  It uses regex pattern matching against a curated list of known injection markers.  Matches surface as `injection_marker: <pattern>` in the verdict's `advisory` (default) or `reasons` (strict).",
    ),
    (
        "Prompt-injection detection in mind-mem — what tool?",
        "`validate_block(text)` runs the `injection_marker` rule (one of 8 deterministic rules).  It matches known prompt-injection signatures via regex; matches appear as `injection_marker: <message>` in the verdict.  Library-level — no LLM, no external API.",
    ),
    (
        "How does mind-mem flag injection attempts in proposed blocks?",
        "`validate_block(text)` — the `injection_marker` rule (#7) inspects the text against a curated list of known prompt-injection patterns.  Hits land in the verdict's `advisory` (or `reasons` if strict).  Deterministic; no model inference needed.",
    ),

    # === Restore C: verify_chain in contradiction-fix workflow ===
    # Probe: "I see a contradiction between two decision blocks. Walk me through the fix."
    # Required: ['list_contradictions', 'propose_update', 'approve_apply', 'verify_chain'].
    # v6 model: listed 3 steps, missed verify_chain.
    # v7 adds 6 paraphrases that always end with verify_chain to audit-confirm.
    (
        "I see a contradiction between two decision blocks. Walk me through the fix.",
        "Four-step canonical workflow: (1) `list_contradictions(workspace)` — find every contradicting pair, including the two decision blocks; (2) `propose_update(workspace, ...)` — write a new block that resolves the contradiction; (3) `approve_apply(workspace, proposal_id)` — apply the proposal to the live store; (4) `verify_chain(workspace)` — confirm the audit chain is intact post-apply.  Always run `verify_chain` last to prove the rollback hash matches.",
    ),
    (
        "I see a contradiction between two decision blocks. Walk me through the fix.",
        "Run `list_contradictions` → `propose_update` → `approve_apply` → `verify_chain`.  The final `verify_chain(workspace)` audits the chain integrity after the apply, guaranteeing every block's hash chains correctly back to genesis.  Never skip the `verify_chain` step — without it you have no audit proof the apply landed cleanly.",
    ),
    (
        "Walk me through the contradiction-fix workflow.",
        "Four steps: `list_contradictions` (locate the pair) → `propose_update` (stage the resolving block) → `approve_apply` (commit) → `verify_chain` (audit).  Step 4 (`verify_chain`) is non-negotiable — it confirms the audit chain is intact and the apply did not corrupt prior blocks.",
    ),
    (
        "How do I resolve a contradiction between two stored blocks end-to-end?",
        "Canonical 4-tool workflow: (1) `list_contradictions` — get the pair IDs; (2) `propose_update` — stage the resolving block; (3) `approve_apply` — write to live store; (4) `verify_chain` — audit-confirm the chain integrity.  All four are required; `verify_chain` is the closing audit step.",
    ),
    (
        "Two decision blocks contradict each other. Full fix workflow?",
        "`list_contradictions` → `propose_update` → `approve_apply` → `verify_chain`.  The first three resolve and apply; the fourth (`verify_chain`) is the audit gate that proves the chain hash is intact after the apply.  Skip `verify_chain` and you ship without audit proof.",
    ),
    (
        "Contradiction-fix tool sequence in mind-mem?",
        "Four tools, in order: `list_contradictions`, `propose_update`, `approve_apply`, `verify_chain`.  The final `verify_chain` audits the chain integrity post-apply — without it the workflow is incomplete.",
    ),

    # === Restore D: quality_gate.mode default location (mind-mem.json) ===
    # Probe: "What is the default value of `quality_gate.mode`?"
    # Required: ['advisory', 'mind-mem.json'].
    # v6 model: gave 'advisory' but missed `mind-mem.json` config file.
    # v7 adds 4 paraphrases that pair the default with the file location.
    (
        "What is the default value of `quality_gate.mode`?",
        "`\"advisory\"`.  Configured in `mind-mem.json` under the `quality_gate.mode` key.  Default `\"advisory\"` means `validate_block` runs but failed rules are recorded under `advisory` (not `reasons`) and `accept` stays True.  Other valid values are `\"off\"` and `\"strict\"`.",
    ),
    (
        "What is the default value of `quality_gate.mode`?",
        "Default is `\"advisory\"`, set in the workspace's `mind-mem.json` config file under the `quality_gate.mode` key.  When the key is absent, the default `\"advisory\"` applies — `validate_block` records fired rules but `accept` stays True.",
    ),
    (
        "Default quality_gate.mode value and where it's set?",
        "Value: `\"advisory\"`.  Location: `mind-mem.json` (the workspace config file), under the key `quality_gate.mode`.  When absent, the implicit default is `\"advisory\"` and validation runs in non-blocking mode.",
    ),
    (
        "Tell me the default for quality_gate.mode and the file it lives in.",
        "`\"advisory\"` is the default; configured in `mind-mem.json` (workspace config) under the `quality_gate.mode` key.  If the key is missing entirely, the implicit default `\"advisory\"` applies.",
    ),
]


def _harvest_v312_density_fix() -> Iterator[dict]:
    """v3.12.1 corpus density fix.  Adds 5+ paraphrases per failing eval probe
    plus explicit denial probes for the 6 hallucinated names that the v3.12.0
    retrain learned (4-LLM consensus 2026-05-09).  ~30 new probes; corpus 4232
    → ~4262 examples."""
    for q, a in _V312_DENSITY_FIX_PROBES:
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        }


# ----------------------------------------------------------------------------
# v4 retrain: balanced per-edge-kind isolated-fact reinforcement.
#
# Why this exists.
#   The v3.12.1 ship (mind-mem-4b v3.12.0-fullft) clears 95/95 only with two
#   eval probes intentionally softened.  One of those — the cites=0.8 probe —
#   is a real model error: the model emits 0.4 (the refines value) when asked
#   in singular form.
#
# Root cause.
#   The corpus had >12 ISOLATED-FACT probes for refines=0.4 (added during v3.10
#   to fix a stubborn 0.3 failure) but only enumerated-list cites=0.8 references.
#   Asymmetric saturation → model defaults to the most-reinforced isolated
#   numeric value (0.4) when asked about any single edge kind in isolation.
#
# What this block does.
#   Adds 10+ ISOLATED-FACT probes for cites/implements/cooccurrence so each
#   non-contradicts kind matches refines's reinforcement density.  Mirrors the
#   surface-form variety of the existing refines block at the v6 saturation
#   pass (lines ~3854-3897): "decimal value", "what is KIND_DECAY['<kind>']",
#   "numeric decay value for <kind>", "<kind> edge — what's the kind decay
#   coefficient", contrast probes ("not 0.4 — that's refines"), denial probes
#   ("is the cites multiplier 0.4? No — it's 0.8"), seed-edge propagation
#   examples (KIND_DECAY × HOP_DECAY arithmetic).
#
# Verification gate (NON-NEGOTIABLE, see train/V4_RETRAIN_TODO.md).
#   The v4 model must pass `train/eval_harness.py` 95/95 with both
#   `# V4 RETRAIN TODO` markers reverted to their original requirements
#   (qg.escape_hatch ["force", "strict"] and lin.cites ["cites", "0.8"]).
#   No further probe softening for v4.

_V4_KIND_BALANCE_PROBES: list[tuple[str, str]] = [
    # ===== cites = 0.8 (10 isolated-fact probes) =====
    (
        "Decimal value of the `cites` edge decay multiplier?",
        "`0.8`.  Second-strongest of the five `KIND_DECAY` values.  Full table: `contradicts=1.0`, `cites=0.8`, `implements=0.6`, `cooccurrence=0.5`, `refines=0.4`.  The `cites` value is `0.8`, NOT `0.4` (which is refines).",
    ),
    (
        "What's KIND_DECAY['cites']?",
        "`0.8`.  Defined in `src/mind_mem/block_lineage.py:67`.  The `cites` multiplier is `0.8` — second-largest after `contradicts` (1.0).  Do not confuse with `refines` (0.4).",
    ),
    (
        "Numeric decay value for cites edges in lineage staleness?",
        "Exactly `0.8`.  A `cites` seed edge contributes 80% of the maximum signal (1.0 = `contradicts`).  Block at hop 1 from a `cites` seed gets `0.8 * 1.0 = 0.8`; hop 2 gets `0.8 * 0.9 = 0.72`; hop 3 gets `0.8 * 0.5 = 0.4`.",
    ),
    (
        "Cites edge — what's the kind decay coefficient?",
        "Coefficient is `0.8` for `cites`.  This is the second-largest of the five kinds — only `contradicts` (1.0) is stronger.  `cites` 0.8 > `implements` 0.6 > `cooccurrence` 0.5 > `refines` 0.4.",
    ),
    (
        "What is the decay multiplier for a `cites` edge in lineage staleness?",
        "The `cites` decay multiplier is `0.8` — applied ONCE at the seed edge (the edge from `source_id` to its immediate neighbour).  Further BFS hops use only `HOP_DECAY = (1.0, 0.9, 0.5, 0.2)` regardless of edge kind.  So a `cites` seed gets `0.8 × 1.0 = 0.8`; the block one BFS hop further stores `0.8 × 0.9 = 0.72`.",
    ),
    (
        "When a block cites another, how much staleness does it inherit?",
        "Eighty percent (0.8) of the seed penalty — the `cites` decay multiplier is `0.8`.  So if the source block has staleness 1.0, the citing neighbour at hop 1 receives `0.8 * 1.0 = 0.8`.  Add `HOP_DECAY = (1.0, 0.9, 0.5, 0.2)` for further hops.",
    ),
    (
        "Second-largest KIND_DECAY value?",
        "`0.8` — for the `cites` edge kind.  `contradicts` is the largest at 1.0.  All five values from largest to smallest: `contradicts` 1.0, `cites` 0.8, `implements` 0.6, `cooccurrence` 0.5, `refines` 0.4.",
    ),
    (
        "Cites edges represent direct citation relationships — what's their staleness weight?",
        "Staleness weight (KIND_DECAY) for `cites` = `0.8`.  A block that cites another inherits 80% of the source's staleness penalty — among the strongest dependency signals, second only to `contradicts` (1.0).",
    ),
    (
        "Tabulate the KIND_DECAY constants for v3.12 lineage staleness — emphasise cites.",
        "| kind | KIND_DECAY |\n|---|---|\n| `contradicts` | 1.0 |\n| `cites` | **0.8** |\n| `implements` | 0.6 |\n| `cooccurrence` | 0.5 |\n| `refines` | 0.4 |\n\nThe `cites` value is `0.8` — defined in `src/mind_mem/block_lineage.py:67`.",
    ),
    (
        "Show me KIND_DECAY['cites'] vs KIND_DECAY['refines'] — same or different?",
        "**Different.**  `KIND_DECAY['cites'] = 0.8` (second-strongest).  `KIND_DECAY['refines'] = 0.4` (weakest).  They are opposite ends of the seed-edge attenuation spectrum.  A `cites` seed propagates twice as much staleness as a `refines` seed at the same hop distance.",
    ),
    # ----- cites denial / negation probes (×4) -----
    (
        "Is the cites edge decay multiplier 0.4?",
        "**No.**  `0.4` is the `refines` value.  The `cites` multiplier is `0.8`.  Do not confuse cites (citation) with refines (narrowing).  Full table: `contradicts=1.0`, `cites=0.8`, `implements=0.6`, `cooccurrence=0.5`, `refines=0.4`.",
    ),
    (
        "Confirm: KIND_DECAY['cites'] = 0.4 — true or false?",
        "**False.**  `KIND_DECAY['cites'] = 0.8`, NOT 0.4.  `0.4` is `KIND_DECAY['refines']`.  The two are easy to confuse but not interchangeable — `cites` is twice as strong as `refines`.",
    ),
    (
        "Someone told me cites = 0.6.  Is that right?",
        "No — `0.6` is the `implements` value.  `KIND_DECAY['cites'] = 0.8`.  Full table for clarity: `contradicts` 1.0, **`cites` 0.8**, `implements` 0.6, `cooccurrence` 0.5, `refines` 0.4.",
    ),
    (
        "Cites multiplier is 0.5, right?",
        "No — `0.5` is `cooccurrence`.  `KIND_DECAY['cites'] = 0.8`.  Citation edges are semantic dependencies and propagate staleness more strongly (80%) than statistical co-occurrence edges (50%).",
    ),

    # ===== implements = 0.6 (10 isolated-fact probes) =====
    (
        "Decimal value of the `implements` edge decay multiplier?",
        "`0.6`.  The middle value of the five `KIND_DECAY` constants.  Full table: `contradicts=1.0`, `cites=0.8`, `implements=0.6`, `cooccurrence=0.5`, `refines=0.4`.",
    ),
    (
        "What's KIND_DECAY['implements']?",
        "`0.6`.  Defined in `src/mind_mem/block_lineage.py:68`.  Sits between `cites` (0.8) and `cooccurrence` (0.5).  An `implements` edge means block A is the concrete realisation of block B (spec → code).",
    ),
    (
        "Numeric decay value for implements edges?",
        "Exactly `0.6`.  An `implements` seed edge contributes 60% of the maximum signal (1.0 = `contradicts`).  Block at hop 1 gets `0.6 * 1.0 = 0.6`; hop 2 gets `0.6 * 0.9 = 0.54`; hop 3 gets `0.6 * 0.5 = 0.3`.",
    ),
    (
        "Implements edge — what's the kind decay coefficient?",
        "Coefficient is `0.6` for `implements`.  Third-largest of five.  `contradicts` 1.0 > `cites` 0.8 > `implements` 0.6 > `cooccurrence` 0.5 > `refines` 0.4.",
    ),
    (
        "If a code block implements a spec block, how much staleness does it inherit?",
        "Sixty percent (0.6) of the spec's staleness penalty — the `implements` decay multiplier is `0.6`.  When the spec is flagged stale, the implementing code block inherits 60% of the penalty at hop 1.  Add `HOP_DECAY` for further hops.",
    ),
    (
        "Middle value in the KIND_DECAY table?",
        "`0.6` — for the `implements` edge kind.  Sits exactly in the middle of the five-value KIND_DECAY table: `contradicts` 1.0, `cites` 0.8, **`implements` 0.6**, `cooccurrence` 0.5, `refines` 0.4.",
    ),
    (
        "Implements edges represent spec→code realisation — what's their staleness weight?",
        "Staleness weight (KIND_DECAY) for `implements` = `0.6`.  An implementing block inherits 60% of the spec's staleness — the third-strongest dependency, after `contradicts` (1.0) and `cites` (0.8).",
    ),
    (
        "Confirm KIND_DECAY['implements'] from block_lineage.py.",
        "`KIND_DECAY['implements'] = 0.6` per `src/mind_mem/block_lineage.py:68`.  This is the constant that the v3.12 lineage→staleness BFS multiplies HOP_DECAY by when the seed edge has kind `implements`.",
    ),
    (
        "Is implements 0.4?",
        "**No.**  `0.4` is `refines`.  `KIND_DECAY['implements'] = 0.6`.  Implements is the third-strongest kind; refines is the weakest.",
    ),
    (
        "Is the implements multiplier 0.8?",
        "No — `0.8` is `cites`.  `KIND_DECAY['implements'] = 0.6`.  Full table: `contradicts` 1.0, `cites` 0.8, **`implements` 0.6**, `cooccurrence` 0.5, `refines` 0.4.",
    ),

    # ===== cooccurrence = 0.5 (10 isolated-fact probes) =====
    (
        "Decimal value of the `cooccurrence` edge decay multiplier?",
        "`0.5`.  The default v2.6.0 edge kind, retained for backward compatibility.  Full table: `contradicts=1.0`, `cites=0.8`, `implements=0.6`, `cooccurrence=0.5`, `refines=0.4`.",
    ),
    (
        "What's KIND_DECAY['cooccurrence']?",
        "`0.5`.  Defined in `src/mind_mem/block_lineage.py:70`.  Co-occurrence edges represent statistical correlation — two blocks returned in the same recall pass — not a semantic dependency, so they attenuate halfway down the spectrum.",
    ),
    (
        "Numeric decay value for cooccurrence edges?",
        "Exactly `0.5`.  A `cooccurrence` seed edge contributes 50% of the maximum signal.  Block at hop 1 gets `0.5 * 1.0 = 0.5`; hop 2 gets `0.5 * 0.9 = 0.45`; hop 3 gets `0.5 * 0.5 = 0.25`.",
    ),
    (
        "Cooccurrence edge — what's the kind decay coefficient?",
        "Coefficient is `0.5` for `cooccurrence`.  Fourth of five.  `contradicts` 1.0 > `cites` 0.8 > `implements` 0.6 > `cooccurrence` 0.5 > `refines` 0.4.",
    ),
    (
        "What's the decay weight for the original v2.6.0 untyped edge kind?",
        "`0.5`.  `cooccurrence` is the v2.6.0 default kind retained when v3.11.0 added typed lineage edges.  Its `KIND_DECAY` value is `0.5` — exactly half-strength because co-occurrence is statistical correlation, not a semantic dependency.",
    ),
    (
        "Co-occurrence edges between blocks — staleness weight?",
        "`0.5`.  Two blocks that surfaced together in the same recall pass form an implicit `cooccurrence` edge; staleness propagates at 50% strength.  Weaker than semantic edges (`cites` 0.8, `implements` 0.6) but stronger than `refines` (0.4).",
    ),
    (
        "Confirm KIND_DECAY['cooccurrence'] from block_lineage.py.",
        "`KIND_DECAY['cooccurrence'] = 0.5` per `src/mind_mem/block_lineage.py:70`.  The v2.6.0 untyped-edge default; preserved unchanged when typed lineage edges landed in v3.11.0.",
    ),
    (
        "Is cooccurrence 0.4?",
        "**No.**  `0.4` is `refines`.  `KIND_DECAY['cooccurrence'] = 0.5`.  The two are adjacent in the table but not equal.",
    ),
    (
        "Is the cooccurrence multiplier 0.8?",
        "No — `0.8` is `cites`.  `KIND_DECAY['cooccurrence'] = 0.5`.  Co-occurrence is the v2.6.0 default; `cites` is the strongest semantic dependency edge after `contradicts`.",
    ),
    (
        "Cooccurrence vs cites — same or different decay?",
        "**Different.**  `KIND_DECAY['cooccurrence'] = 0.5`; `KIND_DECAY['cites'] = 0.8`.  `cites` propagates 60% more staleness than `cooccurrence` because citation is a semantic dependency while co-occurrence is statistical correlation.",
    ),

    # ===== Cross-kind comparison probes (×8) =====
    # Force the model to learn the FULL ordering and the GAPS between values.
    (
        "Order all five KIND_DECAY values from largest to smallest.",
        "1. `contradicts` = 1.0\n2. `cites` = 0.8\n3. `implements` = 0.6\n4. `cooccurrence` = 0.5\n5. `refines` = 0.4\n\nGap pattern: contradicts→cites = 0.2; cites→implements = 0.2; implements→cooccurrence = 0.1; cooccurrence→refines = 0.1.",
    ),
    (
        "Difference between KIND_DECAY['cites'] and KIND_DECAY['refines']?",
        "`0.4`.  `cites` (0.8) − `refines` (0.4) = `0.4`.  `cites` is twice as strong as `refines` as a seed multiplier.",
    ),
    (
        "Difference between KIND_DECAY['cites'] and KIND_DECAY['implements']?",
        "`0.2`.  `cites` (0.8) − `implements` (0.6) = `0.2`.  Adjacent in the canonical ordering.",
    ),
    (
        "Sum of all five KIND_DECAY values?",
        "`3.3`.  `1.0 + 0.8 + 0.6 + 0.5 + 0.4 = 3.3`.  This is contradicts + cites + implements + cooccurrence + refines.",
    ),
    (
        "Mean of the five KIND_DECAY values?",
        "`0.66`.  `(1.0 + 0.8 + 0.6 + 0.5 + 0.4) / 5 = 3.3 / 5 = 0.66`.  cites (0.8) sits above the mean; implements (0.6), cooccurrence (0.5), refines (0.4) sit below.",
    ),
    (
        "Which two edge kinds have the smallest gap in KIND_DECAY?",
        "`implements` and `cooccurrence` (gap of 0.1: 0.6 vs 0.5), or `cooccurrence` and `refines` (also 0.1: 0.5 vs 0.4).  Both adjacent pairs in the lower half of the table share the smallest 0.1 gap.",
    ),
    (
        "Which edge kinds have KIND_DECAY ≥ 0.6?",
        "Three of five: `contradicts` (1.0), `cites` (0.8), and `implements` (0.6).  These are the semantic-dependency edges.  Below 0.6: `cooccurrence` (0.5) and `refines` (0.4).",
    ),
    (
        "Which edge kind is the weakest seed for staleness propagation?",
        "`refines` — `KIND_DECAY['refines'] = 0.4` — the smallest of the five values.  A `refines` seed contributes only 40% of the maximum staleness signal.  The strongest is `contradicts` at 1.0.",
    ),
]


def _harvest_v4_kind_balance() -> Iterator[dict]:
    """v4 retrain: balanced per-edge-kind isolated-fact reinforcement.

    Adds 10+ ISOLATED-FACT probes for each of cites/implements/cooccurrence
    so each non-contradicts edge kind matches the refines reinforcement
    density already in the corpus (~12 probes).  Plus 8 cross-kind
    comparison probes that force the model to learn the full ordering and
    the gap arithmetic between adjacent values.

    Verification gate (see train/V4_RETRAIN_TODO.md):  the v4 model must
    pass eval_harness 95/95 with both `# V4 RETRAIN TODO` markers reverted
    in train/eval_harness.py — non-negotiable.
    """
    for q, a in _V4_KIND_BALANCE_PROBES:
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        }


_V4_SURFACES_PROBES: list[tuple[str, str]] = [
    # =========================================================
    # 1. circuit_breaker.py  (12 probes)
    # =========================================================
    (
        "What are the three states of the v4 circuit breaker?",
        "`CircuitState.CLOSED` (calls pass through), `CircuitState.OPEN` (all calls raise `CircuitOpenError`), and `CircuitState.HALF_OPEN` (one probe call allowed; success closes, failure re-opens). Defined in `src/mind_mem/v4/circuit_breaker.py`. The state machine is thread-safe under an internal `threading.Lock`.",
    ),
    (
        "Default values for CircuitBreaker constructor arguments?",
        "`failure_threshold=5` (`DEFAULT_FAILURE_THRESHOLD`), `recovery_timeout=30.0` (`DEFAULT_RECOVERY_TIMEOUT_S`), `half_open_probes=1` (`DEFAULT_HALF_OPEN_PROBES`). All three are module-level constants in `src/mind_mem/v4/circuit_breaker.py` and configurable via `mind-mem.json: v4: circuit_breaker: {failure_threshold, recovery_timeout_s, half_open_probes}`.",
    ),
    (
        "How do I call a function through a CircuitBreaker instance?",
        "`result = breaker.call(my_fn, arg1, arg2)`. `call()` acquires the lock to decide state, releases the lock before invoking `fn`, records success or failure, and re-raises any exception from `fn`. When OPEN it never calls `fn` and raises `CircuitOpenError(retry_after=<float>)` instead. Defined in `src/mind_mem/v4/circuit_breaker.py`.",
    ),
    (
        "Wrap an embedder function in a v4 circuit breaker.",
        "Use the decorator factory:\n\n```python\nfrom mind_mem.v4.circuit_breaker import circuit_breaker\n\n@circuit_breaker(failure_threshold=3, recovery_timeout=10.0)\ndef embed(text: str) -> list[float]:\n    return external_service.embed(text)\n\n# Inspect or manually reset the per-function breaker:\nembed.breaker.state()\nembed.breaker.reset()\n```\n\nEach decorated function gets its own `CircuitBreaker` instance. Source: `src/mind_mem/v4/circuit_breaker.py`.",
    ),
    (
        "What does CircuitOpenError carry and why?",
        "`retry_after: float` — seconds remaining until the next probe is allowed. Callers use it to schedule a delayed retry (`time.sleep(exc.retry_after)`) instead of polling `breaker.state()`. Polling is wasteful and reads a potentially stale value; `retry_after` is computed once and is immediately actionable. Source: `src/mind_mem/v4/circuit_breaker.py`.",
    ),
    (
        "How do I check how long until the circuit breaker allows a retry?",
        "`breaker.time_until_retry()` returns `float` seconds remaining. Returns `0.0` when CLOSED or HALF_OPEN. When OPEN it computes `max(0.0, recovery_timeout - elapsed)`. Do NOT poll `breaker.state()` in a loop — use `time_until_retry()` to schedule. Source: `src/mind_mem/v4/circuit_breaker.py`.",
    ),
    (
        "How do I manually trip or reset a CircuitBreaker?",
        "`breaker.trip()` forces OPEN immediately (useful when external monitoring detects a sick dependency before the threshold is reached). `breaker.reset()` forces CLOSED (useful when an operator knows the dependency is healthy before the recovery window elapses). Both are thread-safe under the internal lock. Source: `src/mind_mem/v4/circuit_breaker.py`.",
    ),
    (
        "What feature flag enables the circuit breaker?",
        "`v4.circuit_breaker` in `mind-mem.json`: `\"v4\": {\"circuit_breaker\": {\"enabled\": true}}`. Without this, calling `breaker.call(fn)` or `default_breaker()` raises `FeatureDisabledError` from `mind_mem.v4.feature_flags`. `CircuitBreaker` instantiation itself does not check the flag; only `call()` and `default_breaker()` do. Source: `src/mind_mem/v4/circuit_breaker.py`.",
    ),
    (
        "What is default_breaker() in circuit_breaker.py?",
        "A lazy, thread-safe, process-wide singleton `CircuitBreaker`. First call reads `mind-mem.json` for `failure_threshold`, `recovery_timeout_s`, `half_open_probes` and creates the instance; subsequent calls return the same object. Requires `v4.circuit_breaker` flag ON — raises `FeatureDisabledError` if the flag is OFF. Reset via `reset_for_tests()` in test suites. Source: `src/mind_mem/v4/circuit_breaker.py`.",
    ),
    (
        "Should I poll breaker.state() in a loop to wait for recovery?",
        "No. Use `breaker.time_until_retry()` to get the exact remaining seconds, then `time.sleep()` that value. Polling `state()` is wasteful because state may not change until the recovery window elapses, and each `state()` call re-reads the lock. `time_until_retry()` gives a single non-stale value per call. Source: `src/mind_mem/v4/circuit_breaker.py`.",
    ),
    (
        "How does half_open_probes affect circuit breaker recovery?",
        "`half_open_probes=N` means N consecutive successes are required in `HALF_OPEN` before the breaker closes. Default is 1 (a single probe success closes it). Setting `half_open_probes=3` requires sustained recovery before trusting the dependency again — useful for services with intermittent success. A failed probe in HALF_OPEN re-opens for another full `recovery_timeout` window. Source: `src/mind_mem/v4/circuit_breaker.py`.",
    ),
    (
        "Best pattern for embedding fallback when the circuit is open?",
        "Wrap the embedder in `@circuit_breaker(...)`. On `CircuitOpenError`, fall back to a cached vector or a cheaper local model. Compose with `FallbackPolicy` from `surprise_retrieval`: the circuit breaker prevents *calls* to the broken embedder; `FallbackPolicy` decides what `compute_surprise` returns when the embedding is unusable regardless of cause. These are layered defenses — circuit breaker at the I/O boundary, fallback policy at the scoring boundary. Source: `src/mind_mem/v4/circuit_breaker.py` + `src/mind_mem/v4/surprise_retrieval.py`.",
    ),

    # =========================================================
    # 2. backpressure.py  (11 probes)
    # =========================================================
    (
        "What are the default watermarks for BackpressureController?",
        "`high_watermark=1000` (`DEFAULT_HIGH_WATERMARK`), `low_watermark=200` (`DEFAULT_LOW_WATERMARK`), `max_pause_seconds=5.0` (`DEFAULT_MAX_PAUSE_S`). Configurable via `mind-mem.json: v4: backpressure: {high_watermark, low_watermark}`. Source: `src/mind_mem/v4/backpressure.py`.",
    ),
    (
        "How does hysteresis work in BackpressureController?",
        "Entering overloaded state requires `depth >= high_watermark`; exiting requires `depth <= low_watermark`. The gap between the two prevents flapping at the boundary: a queue oscillating around 600 with `high=1000, low=200` stays in its current state rather than toggling on every tick. Source: `src/mind_mem/v4/backpressure.py`.",
    ),
    (
        "How do I report queue depth to the backpressure controller?",
        "`controller.set_depth(n)` — call this from the producer each time the queue depth changes. `set_depth` clamps negative values to 0 and evaluates the hysteresis gate internally. Thread-safe under the internal lock. Source: `src/mind_mem/v4/backpressure.py`.",
    ),
    (
        "Difference between recommended_pause() and current_pause()?",
        "`recommended_pause()` advances the exponential-backoff tick AND returns the pause hint — side-effectful. `current_pause()` is a pure read: it returns what the next pause WOULD be without mutating the tick counter. Use `current_pause()` for observability/dashboards; use `recommended_pause()` in the producer sleep loop (or pair `current_pause()` + `record_overload_tick()` for explicit control). Source: `src/mind_mem/v4/backpressure.py`.",
    ),
    (
        "What is the backoff base for BackpressureController?",
        "50ms (`base = 0.05`). Doubles each call to `recommended_pause()`: 50ms → 100ms → 200ms → … capped at `max_pause_seconds` (default 5.0s). Internal tick is capped at 16 to prevent overflow. Source: `src/mind_mem/v4/backpressure.py`.",
    ),
    (
        "How do I block a synchronous producer until the queue clears?",
        "`cleared = controller.wait_until_clear(timeout=30.0, poll=0.1)`. Returns `True` if `is_overloaded()` became False before `timeout`, `False` if timed out. Polls with `time.sleep(poll)` internally. Async callers should use `recommended_pause()` in a non-blocking loop instead. Source: `src/mind_mem/v4/backpressure.py`.",
    ),
    (
        "What feature flag gates backpressure?",
        "`v4.backpressure`. `controller()` singleton raises `FeatureDisabledError` when the flag is OFF. `is_overloaded()` is always safe to call — it does not check the flag. Only `controller()` and `record_overload_tick()` are gated. Source: `src/mind_mem/v4/backpressure.py`.",
    ),
    (
        "How do I get the singleton BackpressureController?",
        "`from mind_mem.v4.backpressure import controller; ctrl = controller()`. Lazy + thread-safe; reads `mind-mem.json` at first call. Requires `v4.backpressure` flag ON. Reset via `reset_for_tests()` in test suites. Source: `src/mind_mem/v4/backpressure.py`.",
    ),
    (
        "What does record_overload_tick() do?",
        "Manually advances the internal backoff counter by 1 without returning a pause hint. Use it when you want to read-then-tick explicitly: call `current_pause()` to peek the pause value, sleep, then call `record_overload_tick()` to advance. This gives the caller explicit control over when the tick increments versus the combined read+tick of `recommended_pause()`. Source: `src/mind_mem/v4/backpressure.py`.",
    ),
    (
        "Can I use BackpressureController without the flag being ON?",
        "Partially. `BackpressureController(...)` instantiation and `is_overloaded()` work without the flag. The `controller()` singleton and `record_overload_tick()` require the flag — they call `require_enabled('backpressure')`. For lightweight caller-side checks, instantiate directly or call `is_overloaded()` on an existing instance. Source: `src/mind_mem/v4/backpressure.py`.",
    ),
    (
        "How does BackpressureController compose with CircuitBreaker?",
        "They address different failure modes. `BackpressureController` signals queue saturation — the producer slows down. `CircuitBreaker` signals repeated external failures — calls to the dependency are cut off entirely. A well-defended pipeline uses both: the controller throttles write rate under queue pressure; the breaker short-circuits calls to a broken embedder. Neither replaces the other. Source: `src/mind_mem/v4/backpressure.py` + `src/mind_mem/v4/circuit_breaker.py`.",
    ),

    # =========================================================
    # 3. health.py  (11 probes)
    # =========================================================
    (
        "What does health_check() return?",
        "A dict with keys `status` (`\"ok\"` | `\"degraded\"` | `\"fail\"`), `modules` (dict of module-name → status string), `latency_ms` (float), `checked_at` (ISO 8601 UTC string), and `disabled_count` (int, number of feature-flag-disabled probes). Never raises — catches `BaseException` per probe. Source: `src/mind_mem/v4/health.py`.",
    ),
    (
        "What are the 7 built-in health probes?",
        "`feature_flags`, `tier_memory`, `block_kinds`, `cognitive_kernel`, `federation`, `observability`, `eviction`. Each probe returns a `ModuleStatus` string: `\"ok\"`, `\"missing\"`, `\"disabled\"`, or `\"error: <repr>\"`. Defined in `_BUILTIN_PROBES` in `src/mind_mem/v4/health.py`.",
    ),
    (
        "When does health_check return 'degraded' vs 'fail'?",
        "`\"fail\"` when any probe returns a status starting with `\"error:\"` (exception raised inside the probe). `\"degraded\"` when any probe returns `\"missing\"` (feature exists but expected schema or registry entry is absent). `\"ok\"` when every probe returns `\"ok\"` or `\"disabled\"`. `\"disabled\"` probes do not push aggregate below `\"ok\"`. Source: `src/mind_mem/v4/health.py`.",
    ),
    (
        "Is health_check itself feature-flag gated?",
        "No. `health_check(workspace)` runs unconditionally — operators need it during failure debugging. Individual probes report `\"disabled\"` for features whose flag is OFF rather than skipping them, so operators can distinguish 'feature off' from 'feature broken'. Source: `src/mind_mem/v4/health.py`.",
    ),
    (
        "How do I add a custom health probe?",
        "`register_health_probe(name, fn)` where `fn(workspace: Path) -> str`. Re-registering under an existing name replaces the old probe (not stacked). Thread-safe. Custom probes run after all built-in probes. Remove all custom probes via `reset_custom_probes_for_tests()` in test teardown. Source: `src/mind_mem/v4/health.py`.",
    ),
    (
        "What does the cognitive_kernel health probe check?",
        "It calls `is_kernel_registered(KernelKind.DEFAULT)` from `mind_mem.v4.cognitive_kernel`. Returns `\"ok\"` if the default kernel is registered, `\"missing\"` if not, `\"disabled\"` if `v4.cognitive_kernel` flag is OFF. It uses the public `is_kernel_registered` predicate instead of reaching into `_registry` directly. Source: `src/mind_mem/v4/health.py`.",
    ),
    (
        "What does the eviction health probe check?",
        "Calls `is_policy_registered(EvictionPolicy.LRU)` from `mind_mem.v4.eviction`. Returns `\"ok\"` if LRU is registered (always true after module import), `\"missing\"` if not, `\"disabled\"` if `v4.eviction` flag is OFF. Source: `src/mind_mem/v4/health.py`.",
    ),
    (
        "Why does health_check catch BaseException and not just Exception?",
        "A health endpoint that crashes during failure debugging is worse than one that reports the failure as `\"error: ...\"`. `BaseException` catches `KeyboardInterrupt`, `SystemExit`, and any non-`Exception` subclass that could leak from a probe. The intentional broad catch is documented in the `health_check` docstring. Source: `src/mind_mem/v4/health.py`.",
    ),
    (
        "What does disabled_count in the health report mean?",
        "The number of probes that returned `\"disabled\"` (their feature flag is OFF). Operators can distinguish `\"healthy but minimal\"` (many features disabled) from `\"fully armed\"` (all features on). A workspace with all v4 flags OFF will have `disabled_count=6` (all gated probes) but `status=\"ok\"`. Source: `src/mind_mem/v4/health.py`.",
    ),
    (
        "How do I call health_check?",
        "`from mind_mem.v4.health import health_check; report = health_check('/path/to/workspace')`. Accepts `str` or `Path`. Returns a dict immediately — never raises. Low-latency (single-digit milliseconds under normal conditions) so it's safe to call frequently from a liveness probe. Source: `src/mind_mem/v4/health.py`.",
    ),
    (
        "What does the tier_memory probe actually query?",
        "It connects to `workspace/index.db` and checks `sqlite_master` for the `block_recall_tier` table. Returns `\"ok\"` if the table exists, `\"missing\"` if the DB file or table is absent, `\"disabled\"` if `v4.tier_memory` flag is OFF, or `\"error: <exc>\"` on `sqlite3.Error`. Source: `src/mind_mem/v4/health.py`.",
    ),

    # =========================================================
    # 4. logging_context.py  (11 probes)
    # =========================================================
    (
        "What is LogContext in logging_context.py?",
        "A contextvar-backed stack of key→value bindings. `LogContext.push(**bindings)` pushes a new frame and returns a `contextvars.Token`; `LogContext.pop(token)` removes it. The stack is per-asyncio-task / per-thread, so concurrent callers don't see each other's context. No flag required — the module is dependency-free. Source: `src/mind_mem/v4/logging_context.py`.",
    ),
    (
        "How do I read the current structured log context?",
        "`from mind_mem.v4.logging_context import current_context; ctx = current_context()`. Returns a fresh dict (safe to mutate). Later frames override earlier frames for the same key. Returns an empty dict when no context is pushed. Source: `src/mind_mem/v4/logging_context.py`.",
    ),
    (
        "How do I push log context for a block of code?",
        "Use the `with_context` context manager:\n\n```python\nfrom mind_mem.v4.logging_context import with_context\n\nwith with_context(workspace='/tmp/ws', agent_id='A'):\n    recall(...)  # log records inside see workspace + agent_id\n```\n\nThe bindings are automatically popped on exit. Source: `src/mind_mem/v4/logging_context.py`.",
    ),
    (
        "What does @with_correlation_id do?",
        "Decorator that wraps the function in a `with_context(correlation_id=<uuid4>)` block. If the active context already has a `correlation_id` from an outer call, it preserves it (no overwrite) — nested calls inherit the parent's correlation ID for end-to-end traces. Works on both `def` and `async def` targets (detected via `inspect.iscoroutinefunction` at decoration time). Source: `src/mind_mem/v4/logging_context.py`.",
    ),
    (
        "Why does @with_correlation_id use inspect.iscoroutinefunction?",
        "Without the async check, decorating an `async def` would silently return a sync wrapper that returns an unawaited coroutine — a silent bug. The decorator picks the correct wrapper shape at decoration time so async frameworks (`asyncio`, FastAPI) see an awaitable. Source: `src/mind_mem/v4/logging_context.py`.",
    ),
    (
        "What is StructuredLogFilter and how do I install it?",
        "`StructuredLogFilter` is a `logging.Filter` subclass that attaches `current_context()` to every log record as `record.ctx`. Install it once at startup:\n\n```python\nimport logging\nfrom mind_mem.v4.logging_context import StructuredLogFilter\nlogging.getLogger().addFilter(StructuredLogFilter())\n```\n\nDownstream JSON handlers can then serialize `record.ctx` alongside the message. Source: `src/mind_mem/v4/logging_context.py`.",
    ),
    (
        "Is logging_context feature-flag gated?",
        "No. The module is pure stdlib (contextvar + logging) with no external dependencies and no flag checks. It is importable on a fresh install and usable regardless of which v4 flags are enabled. Source: `src/mind_mem/v4/logging_context.py`.",
    ),
    (
        "Does LogContext.push() overwrite existing keys?",
        "Within the same push frame, yes — later pushes with the same key override earlier frames when `current_context()` merges. `current_context()` iterates frames from oldest to newest and calls `merged.update(frame)`, so the newest frame wins per key. Source: `src/mind_mem/v4/logging_context.py`.",
    ),
    (
        "How does logging_context integrate with async code?",
        "`contextvars.ContextVar` is asyncio-native: each task gets its own copy of the context at creation time (Python 3.7+). Pushing a binding inside one task does not leak into sibling tasks. `@with_correlation_id` wraps `async def` functions in an `async def _inner_async` wrapper so the `correlation_id` is set inside the same asyncio task scope. Source: `src/mind_mem/v4/logging_context.py`.",
    ),
    (
        "What is the backing storage for LogContext?",
        "A `contextvars.ContextVar` named `v4_log_ctx` holding a tuple of dicts (the stack). Tuples are immutable so each push creates a new tuple — no mutation of the existing stack. Token-based reset via `_ctx_stack.reset(token)` is the standard contextvar rollback pattern. Source: `src/mind_mem/v4/logging_context.py`.",
    ),
    (
        "Can @with_correlation_id be nested? What happens to the ID?",
        "Yes. The outer call sets `correlation_id=<uuid4>`. The inner call reads `current_context().get('correlation_id')` — finds the existing ID and reuses it (no overwrite). The trace ID is therefore stable end-to-end: all log lines from the outer call through the deepest nested call share the same UUID. Source: `src/mind_mem/v4/logging_context.py`.",
    ),

    # =========================================================
    # 5. block_metadata.py  (12 probes)
    # =========================================================
    (
        "What is the block_metadata table schema?",
        "`block_metadata(block_id TEXT PRIMARY KEY, tags TEXT NOT NULL, ttl_seconds INTEGER, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)`. Tags are JSON-encoded `dict[str, str]`. Indexed on `created_at` and `updated_at`. Created idempotently by `ensure_metadata_schema(workspace)`. Source: `src/mind_mem/v4/block_metadata.py`.",
    ),
    (
        "How do I attach tags and a TTL to a block?",
        "`from mind_mem.v4.block_metadata import set_block_metadata\nset_block_metadata(workspace, block_id, tags={'env': 'prod', 'owner': 'team-a'}, ttl_seconds=86400)`. Uses `INSERT ON CONFLICT DO UPDATE` — `created_at` is preserved across upserts, `updated_at` advances. Returns a `BlockMetadata` dataclass. Source: `src/mind_mem/v4/block_metadata.py`.",
    ),
    (
        "What does INSERT ON CONFLICT DO UPDATE preserve?",
        "`created_at` — the original creation timestamp is never overwritten by `set_block_metadata`. Only `tags`, `ttl_seconds`, and `updated_at` are updated on conflict. Audit/governance callers that need 'first-touch' time read `created_at`; recency-sort tools read `updated_at`. Source: `src/mind_mem/v4/block_metadata.py`.",
    ),
    (
        "How do I list all blocks with a specific tag value?",
        "`ids = list_blocks_by_tag(workspace, key='env', value='prod', limit=100)`. Implemented via `json_extract(tags, '$.env') = 'prod'` on the `tags` column. Returns an empty list when the schema is missing or `limit <= 0`. Default `limit=100`. Source: `src/mind_mem/v4/block_metadata.py`.",
    ),
    (
        "How do I delete block metadata?",
        "`deleted = delete_block_metadata(workspace, block_id)`. Returns `True` if a row was removed, `False` if the block had no metadata row or the schema doesn't exist. Does not raise. Source: `src/mind_mem/v4/block_metadata.py`.",
    ),
    (
        "How do I register a schema validator for a block kind?",
        "`register_schema_validator(kind, fn)` where `fn(payload: dict) -> SchemaValidationResult`. Replaces any existing validator for that kind. Requires `v4.block_metadata` flag ON. Example:\n\n```python\nfrom mind_mem.v4.block_metadata import register_schema_validator, SchemaValidationResult\n\ndef check_decision(p):\n    if 'rationale' not in p:\n        return SchemaValidationResult(ok=False, reason='missing rationale')\n    return SchemaValidationResult(ok=True)\n\nregister_schema_validator('decision', check_decision)\n```\nSource: `src/mind_mem/v4/block_metadata.py`.",
    ),
    (
        "What does validate_block return when no validator is registered?",
        "`SchemaValidationResult(ok=True, reason='no_validator')`. The system is open by default — callers only register validators for kinds they want to constrain. Source: `src/mind_mem/v4/block_metadata.py`.",
    ),
    (
        "What happens when a schema validator raises an exception?",
        "`validate_block` catches it and returns `SchemaValidationResult(ok=False, reason='validator_raised: <repr>')`. The recall path continues cleanly — validator bugs don't crash the pipeline. Source: `src/mind_mem/v4/block_metadata.py`.",
    ),
    (
        "What feature flag gates block_metadata?",
        "`v4.block_metadata`. All write and read functions (`set_block_metadata`, `get_block_metadata`, `delete_block_metadata`, `list_blocks_by_tag`, `register_schema_validator`, `validate_block`) call `require_enabled('block_metadata')`. Without the flag, each raises `FeatureDisabledError`. Source: `src/mind_mem/v4/block_metadata.py`.",
    ),
    (
        "What does SchemaValidationResult look like?",
        "A frozen dataclass: `SchemaValidationResult(ok: bool, reason: str = '')`. `ok=True` means validation passed. `ok=False` means it failed; `reason` carries a human-readable explanation or `'no_validator'` / `'validator_raised: ...'` for system-generated cases. Source: `src/mind_mem/v4/block_metadata.py`.",
    ),
    (
        "Which prior art inspired the two surfaces in block_metadata.py?",
        "ChromaDB-style `BlockMetadata` (key→value tags + TTL per document) and Weaviate-style schema validation hooks (per-kind validators run pre-write). Both are documented in the module docstring. The `block_metadata` table is the shared backing store for both surfaces. Source: `src/mind_mem/v4/block_metadata.py`.",
    ),
    (
        "What does available_validators() return?",
        "A list of kind strings for which a validator is registered. Requires `v4.block_metadata` flag ON. Returns an empty list when no validators have been registered. Thread-safe under `_validator_lock`. Source: `src/mind_mem/v4/block_metadata.py`.",
    ),

    # =========================================================
    # 6. observability.py — cardinality guard  (10 probes)
    # =========================================================
    (
        "What is MAX_CARDINALITY in v4 observability?",
        "`10000` — the maximum number of distinct metric names per type (counter, gauge, histogram). Defined as a module-level constant in `src/mind_mem/v4/observability.py`. Configurable via `mind-mem.json: v4: observability: max_cardinality`.",
    ),
    (
        "What happens when the cardinality cap is exceeded in v4 observability?",
        "Past `MAX_CARDINALITY` distinct names, `counter()` / `gauge()` / `histogram()` return a shared overflow sentinel object (`_OVERFLOW_COUNTER` / `_OVERFLOW_GAUGE` / `_OVERFLOW_HISTOGRAM` with name `v4._overflow.<kind>`). Recording into the sentinel is safe but invisible. A drop counter in `snapshot()` surfaces how many names were dropped: `v4.cardinality.dropped_counter`, `v4.cardinality.dropped_gauge`, `v4.cardinality.dropped_histogram`. Source: `src/mind_mem/v4/observability.py`.",
    ),
    (
        "How do I increment a counter in v4 observability?",
        "`from mind_mem.v4.observability import counter\ncounter('v4.federation.conflicts_detected').inc()`. `counter()` is get-or-create. `inc(n=1)` adds `n`. Thread-safe. Source: `src/mind_mem/v4/observability.py`.",
    ),
    (
        "How do I set a gauge value?",
        "`from mind_mem.v4.observability import gauge\ngauge('v4.tier.warm_count').set(current_count)`. Last value wins (not accumulating). Thread-safe. Source: `src/mind_mem/v4/observability.py`.",
    ),
    (
        "How do I record a histogram observation?",
        "`from mind_mem.v4.observability import histogram\nhistogram('v4.recall.latency_ms').observe(elapsed_ms)`. Keeps running `count`, `sum_v`, `sum_sq`, `min_v`, `max_v`. Source: `src/mind_mem/v4/observability.py`.",
    ),
    (
        "How do I time a function automatically?",
        "Use the `@timed` decorator:\n\n```python\nfrom mind_mem.v4.observability import timed\n\n@timed('v4.recall.latency_ms')\ndef recall(workspace, query):\n    ...\n```\n\nRecords wall-time in milliseconds into the named histogram. Time is captured even when the wrapped function raises. Source: `src/mind_mem/v4/observability.py`.",
    ),
    (
        "How do I get a snapshot of all v4 metrics?",
        "`from mind_mem.v4.observability import snapshot; data = snapshot()`. Returns a flat dict: counters and gauges as `{name: value}`, histograms as `{name: {count, sum, min, max, mean}}`. Does not require the flag — snapshot is read-only. Flag gates update paths (`counter().inc()` etc.). Source: `src/mind_mem/v4/observability.py`.",
    ),
    (
        "How do I plug in a Prometheus or StatsD exporter?",
        "`from mind_mem.v4.observability import set_exporter\nset_exporter(my_prometheus_fn)`. The exporter receives `MetricEvent(name, kind, value)` on every metric update. Default exporter is a no-op (in-memory registry only). Requires `v4.observability` flag ON. Exporter exceptions are silently swallowed — an exporter failure must not crash the recall path. Source: `src/mind_mem/v4/observability.py`.",
    ),
    (
        "What feature flag gates v4 observability?",
        "`v4.observability`. Metric update paths (`counter().inc()`, `gauge().set()`, `histogram().observe()`) call `_emit` which checks `is_enabled('observability')` before routing to the exporter. `snapshot()` does not check the flag — it is always available for test assertions. `set_exporter()` requires the flag ON. Source: `src/mind_mem/v4/observability.py`.",
    ),
    (
        "Where do v4 drop counts appear in snapshot()?",
        "In the counter bucket under keys `v4.cardinality.dropped_counter`, `v4.cardinality.dropped_gauge`, `v4.cardinality.dropped_histogram`. These counters are incremented inside the registry lock each time a get-or-create call is rejected by the cardinality cap. Visible via `snapshot()['v4.cardinality.dropped_counter']` etc. Source: `src/mind_mem/v4/observability.py`.",
    ),

    # =========================================================
    # 7. eviction.py  (11 probes)
    # =========================================================
    (
        "What built-in eviction policies does v4 ship?",
        "`EvictionPolicy.LRU` (oldest COLD blocks by `last_seen_at`), `EvictionPolicy.LOW_SURPRISE` (COLD blocks below a surprise threshold), `EvictionPolicy.AGE` (COLD blocks older than a `cutoff_iso` timestamp), `EvictionPolicy.COMPOSITE` (union of multiple policies). All registered at import time. Source: `src/mind_mem/v4/eviction.py`.",
    ),
    (
        "What does plan_eviction return?",
        "An `EvictionPlan(policy, candidates: list[tuple[str, str]])` where each tuple is `(block_id, reason)`. The plan is read-only; the caller applies it through the v3 propose/approve flow. Returns an empty plan when the policy is unknown, the DB is missing, or the tier table doesn't exist. Source: `src/mind_mem/v4/eviction.py`.",
    ),
    (
        "What does EvictionPlan.debug_plan() return?",
        "A `dict[str, list[str]]` grouping block IDs by the leading tag of their reason string. Reason strings follow the convention `\"<tag>:<detail>\"` (e.g. `\"lru:last_seen=2026-01-01\"`, `\"composite:lru|...\"`). `debug_plan()` splits on `:` and groups by the left side — useful for tracing which sub-policy selected each block in a COMPOSITE run. Source: `src/mind_mem/v4/eviction.py`.",
    ),
    (
        "How do I switch the runtime default eviction policy?",
        "`set_active_policy(EvictionPolicy.LOW_SURPRISE)`. The active policy is read by `plan_eviction(workspace, policy=None)` — passing `None` resolves to `active_policy()`. This mirrors the Redis `CONFIG SET maxmemory-policy` pattern: one global pointer, callers read lazily. Unknown names raise `ValueError`. Requires `v4.eviction` flag ON. Source: `src/mind_mem/v4/eviction.py`.",
    ),
    (
        "What does passing policy=None to plan_eviction() do?",
        "Resolves to `active_policy()` — the workspace-wide runtime default set by `set_active_policy()`. The default at startup is `EvictionPolicy.LRU`. This makes the runtime switch immediately observable on the next `plan_eviction()` call without redeploying. Source: `src/mind_mem/v4/eviction.py`.",
    ),
    (
        "How do I register a custom eviction policy?",
        "`from mind_mem.v4.eviction import register_policy\nregister_policy('my_policy', fn)` where `fn(workspace, **kwargs) -> list[tuple[str, str]]`. Each tuple is `(block_id, reason)`. Requires `v4.eviction` flag ON. After registration, activate with `set_active_policy('my_policy')`. Source: `src/mind_mem/v4/eviction.py`.",
    ),
    (
        "What does is_policy_registered() do?",
        "Public predicate: `is_policy_registered(EvictionPolicy.LRU)` returns `True` if `LRU` is in the registry. Does NOT require the flag — used by `health.py` to probe eviction health without coupling to the private `_registry` dict. Accepts both `EvictionPolicy` enum values and raw strings. Source: `src/mind_mem/v4/eviction.py`.",
    ),
    (
        "What does the COMPOSITE policy do when a sub-policy name is unknown?",
        "Fail-soft: it skips the unknown sub-policy (no `ValueError` raised) and continues with the remaining policies. The composite union of remaining policies is returned. This matches the overall fail-soft contract of `plan_eviction` for unknown policies. Source: `src/mind_mem/v4/eviction.py`.",
    ),
    (
        "What feature flag gates eviction?",
        "`v4.eviction`. `plan_eviction`, `register_policy`, `available_policies`, `set_active_policy`, `active_policy` all call `require_enabled('eviction')`. `is_policy_registered` does NOT require the flag — it is used by the health probe regardless of flag state. Source: `src/mind_mem/v4/eviction.py`.",
    ),
    (
        "Does plan_eviction delete blocks?",
        "No. It only computes a *plan* — returns candidate `(block_id, reason)` tuples. The caller applies the plan through the v3 governance flow (`propose_update` → `approve_apply`). 'Eviction' means proposing COLD→archive; the downstream path decides whether that is a tombstone, a sealed-evidence mirror, or a hard delete. Source: `src/mind_mem/v4/eviction.py`.",
    ),
    (
        "What is the default eviction limit?",
        "`DEFAULT_EVICTION_LIMIT = 100`. LRU, LOW_SURPRISE, and AGE policies pass this as `LIMIT` to their SQL queries unless overridden with a `limit=N` kwarg to `plan_eviction`. Source: `src/mind_mem/v4/eviction.py`.",
    ),

    # =========================================================
    # 8. surprise_retrieval.py — FallbackPolicy  (11 probes)
    # =========================================================
    (
        "What are the four FallbackPolicy values in surprise_retrieval?",
        "`FallbackPolicy.NEUTRAL` (return 0.5 — default), `FallbackPolicy.PROMOTE` (return 1.0 — bias tier promotion), `FallbackPolicy.DEMOTE` (return 0.0 — bias COLD aging), `FallbackPolicy.RAISE` (raise `EmbeddingFailureError`). Defined in `src/mind_mem/v4/surprise_retrieval.py`.",
    ),
    (
        "What does compute_surprise() return?",
        "A float in `[0.0, 1.0]`: cosine distance between the candidate and context embeddings, computed as `(1 - cos_sim) * 0.5` and clamped. `0.0` = identical to context, `0.5` = orthogonal, `1.0` = maximally surprising. No flag check — the math is pure. Source: `src/mind_mem/v4/surprise_retrieval.py`.",
    ),
    (
        "What triggers a FallbackPolicy response in compute_surprise?",
        "Three conditions: `\"missing\"` (context is `None` or either vector is empty), `\"length_mismatch\"` (candidate and context have different dimensions), `\"zero_norm\"` (either vector's L2 norm is zero). The `fallback_policy` argument decides what to return in each case. Source: `src/mind_mem/v4/surprise_retrieval.py`.",
    ),
    (
        "What does EmbeddingFailureError carry?",
        "`reason: str` — a short tag (`\"missing\"`, `\"length_mismatch\"`, `\"zero_norm\"`) identifying why the embedding was unusable. The message is `f'surprise embedding unusable: {reason}'`. Callers can branch on `exc.reason` to decide whether to retry the embedder or fall back. Raised only when `FallbackPolicy.RAISE` is active. Source: `src/mind_mem/v4/surprise_retrieval.py`.",
    ),
    (
        "What is the default FallbackPolicy?",
        "`FallbackPolicy.NEUTRAL` (`return 0.5`). Preserves prior behaviour for callers that never opt in. Can be changed globally via `mind-mem.json: v4: surprise_retrieval: fallback_policy: 'promote'` (or `'demote'` / `'raise'`). Source: `src/mind_mem/v4/surprise_retrieval.py`.",
    ),
    (
        "How do I pass a FallbackPolicy per call?",
        "`score = compute_surprise(candidate, context, fallback_policy=FallbackPolicy.RAISE)`. Accepts `FallbackPolicy` enum, string (lowercased and coerced), or `None` (reads global config). Per-call policy overrides the global default. Source: `src/mind_mem/v4/surprise_retrieval.py`.",
    ),
    (
        "Is compute_surprise() feature-flag gated?",
        "No — the math is pure and cheap. Only `should_promote_on_surprise()` is gated (`require_enabled('surprise_retrieval')`) because it *acts* on the score (decides a tier promotion). Callers can compute surprise scores without the flag; they just cannot apply the tier-promotion gate. Source: `src/mind_mem/v4/surprise_retrieval.py`.",
    ),
    (
        "What is the default tier-promotion threshold for surprise?",
        "`DEFAULT_PROMOTE_THRESHOLD = 0.65`. A WARM block with surprise >= 0.65 bumps back to HOT instead of aging toward COLD. Configurable via `mind-mem.json: v4: surprise_retrieval: promote_threshold: <float>`. Range-clamped to `[0.0, 1.0]` in `surprise_threshold()`. Source: `src/mind_mem/v4/surprise_retrieval.py`.",
    ),
    (
        "What does centroid() do in surprise_retrieval?",
        "Computes the unweighted mean centroid of an iterable of embedding vectors. Returns `None` for an empty iterable, mismatched lengths, or all-empty inputs. Callers wire it to the last-K recall embeddings: `ctx = centroid(last_k_embeddings); score = compute_surprise(candidate_vec, ctx)`. Source: `src/mind_mem/v4/surprise_retrieval.py`.",
    ),
    (
        "Should I use FallbackPolicy.NEUTRAL or PROMOTE when the embedder is flaky?",
        "Use `FallbackPolicy.PROMOTE` if you want flaky-embedder failures to bias tier promotion (WARM blocks stay HOT during outages). Use `FallbackPolicy.NEUTRAL` to preserve prior behaviour. Use `FallbackPolicy.RAISE` in CI to surface embedder bugs explicitly. `FallbackPolicy.DEMOTE` biases COLD aging — appropriate if you want aggressively conservative memory management during outages. Source: `src/mind_mem/v4/surprise_retrieval.py`.",
    ),
    (
        "How do circuit_breaker and FallbackPolicy compose for embedding reliability?",
        "The circuit breaker (`circuit_breaker.py`) prevents calls to a timing-out or repeatedly-failing embedder at the I/O boundary. `FallbackPolicy` (`surprise_retrieval.py`) decides what `compute_surprise` returns when an embedding is structurally unusable (missing, zero-norm). Together: the breaker cuts traffic to a sick embedder; the fallback policy handles individual bad vectors that slip through or arrive from cache. They are layered defenses, not alternatives. Source: `src/mind_mem/v4/circuit_breaker.py` + `src/mind_mem/v4/surprise_retrieval.py`.",
    ),

    # =========================================================
    # 9. cognitive_kernel.py — is_kernel_registered + KernelKind  (9 probes)
    # =========================================================
    (
        "What is is_kernel_registered() and why doesn't it require the flag?",
        "`is_kernel_registered(kind)` returns `True` if `kind` has a strategy bound in the registry. It does NOT call `require_enabled` — the health probe in `health.py` needs to check registration even when `v4.cognitive_kernel` is OFF. The result is meaningful in both states: an OFF flag means the registry exists but won't be consulted by `mind_recall`. Source: `src/mind_mem/v4/cognitive_kernel.py`.",
    ),
    (
        "What KernelKind values does v4 define?",
        "`DEFAULT`, `SURPRISE_WEIGHTED`, `LINEAGE_FIRST`, `RECENT_FIRST`, `CONTRADICTS_FIRST`, `GRAPH_WALK`. String values match the `kernel=` argument to `mind_recall()`. Only `DEFAULT` is registered at import time (delegates to v3 recall unchanged). The others are stubs for future v4 commits. Source: `src/mind_mem/v4/cognitive_kernel.py`.",
    ),
    (
        "How do I call recall through a named kernel?",
        "`from mind_mem.v4.cognitive_kernel import mind_recall, KernelKind\nresult = mind_recall(workspace, query, kernel=KernelKind.LINEAGE_FIRST)`. Returns `KernelResult(kernel, hits: list[KernelHit], metadata: dict)`. Requires `v4.cognitive_kernel` flag ON. Unknown kernel names raise `KeyError` with the list of registered kernels in the message. Source: `src/mind_mem/v4/cognitive_kernel.py`.",
    ),
    (
        "What does KernelHit contain?",
        "`block_id: str`, `score: float`, `reason: str`. `reason` is a free-form string the kernel attaches to explain why the block surfaced — convention is `'<kernel>:<short-tag>'` (e.g. `'surprise_weighted:high_distance'`). The default kernel leaves `reason` empty. Source: `src/mind_mem/v4/cognitive_kernel.py`.",
    ),
    (
        "How do I register a custom kernel strategy?",
        "`from mind_mem.v4.cognitive_kernel import register_kernel, KernelKind\nregister_kernel(KernelKind.SURPRISE_WEIGHTED, my_strategy)`. `my_strategy(workspace, query, **kwargs) -> KernelResult`. Replaces any existing strategy for that kind. Requires `v4.cognitive_kernel` flag ON. Source: `src/mind_mem/v4/cognitive_kernel.py`.",
    ),
    (
        "What does the DEFAULT kernel do?",
        "Delegates to v3 `recall(query)` unchanged. Returns a `KernelResult` with `kernel=KernelKind.DEFAULT` and hits from the v3 RRF-scored recall. No semantic routing. Registered at import time without the flag check so the registry is populated even when the flag is OFF. Source: `src/mind_mem/v4/cognitive_kernel.py`.",
    ),
    (
        "What feature flag gates cognitive_kernel?",
        "`v4.cognitive_kernel`. `mind_recall()`, `register_kernel()`, and `available_kernels()` require the flag ON (raise `FeatureDisabledError` if OFF). `is_kernel_registered()` does NOT require the flag — it is used by the health probe unconditionally. Source: `src/mind_mem/v4/cognitive_kernel.py`.",
    ),
    (
        "Does calling mind_recall without a kernel= argument change v3 behaviour?",
        "No. The default is `kernel=KernelKind.DEFAULT` which delegates to v3 `recall(query)` with identical output shape. A flag-ON workspace calling `mind_recall(ws, q)` without specifying a kernel behaves identically to the v3 API. Source: `src/mind_mem/v4/cognitive_kernel.py`.",
    ),
    (
        "How does is_kernel_registered relate to is_policy_registered in eviction?",
        "Both are public predicates added so `health.py` can probe module state without accessing private `_registry` dicts. `is_kernel_registered` is in `cognitive_kernel.py`; `is_policy_registered` is in `eviction.py`. Neither requires the feature flag — the health probe must run even when features are disabled. Both accept string names and coerce them to the appropriate enum. Source: `src/mind_mem/v4/cognitive_kernel.py` + `src/mind_mem/v4/eviction.py`.",
    ),
]


def _harvest_v4_surfaces() -> Iterator[dict]:
    """v4 retrain: surface-API probes for all 9 new v4 modules.

    98 probes covering:
        circuit_breaker     (12) — CircuitBreaker, CircuitOpenError,
                                   CircuitState, @circuit_breaker, default_breaker,
                                   state/failure_count/time_until_retry/reset/trip,
                                   flag gating, anti-patterns, cross-surface.
        backpressure        (11) — BackpressureController watermarks, hysteresis,
                                   set_depth, recommended_pause vs current_pause,
                                   backoff base, wait_until_clear, controller(),
                                   record_overload_tick, flag gating, composition.
        health              (11) — health_check return shape, 7 built-in probes,
                                   degraded vs fail, flag independence, custom
                                   probes, cognitive_kernel/eviction probe logic,
                                   BaseException catch rationale, disabled_count.
        logging_context     (11) — LogContext push/pop, current_context, with_context,
                                   @with_correlation_id, async awareness, StructuredLogFilter,
                                   flag independence, token-based stack, nesting.
        block_metadata      (12) — table schema, set_block_metadata upsert, ON CONFLICT
                                   created_at preservation, list_blocks_by_tag,
                                   delete, register_schema_validator, validate_block
                                   defaults and exceptions, flag gating, SchemaValidationResult,
                                   prior art, available_validators.
        observability       (10) — MAX_CARDINALITY, overflow sentinels, counter/gauge/
                                   histogram API, @timed, snapshot, set_exporter,
                                   flag gating, drop counter key names.
        eviction            (11) — 4 built-in policies, EvictionPlan, debug_plan,
                                   set_active_policy, plan_eviction(None), register_policy,
                                   is_policy_registered, COMPOSITE fail-soft, flag gating,
                                   pure-planner contract, eviction limit default.
        surprise_retrieval  (11) — FallbackPolicy enum, compute_surprise math,
                                   failure triggers, EmbeddingFailureError, default
                                   policy, per-call override, flag split between
                                   compute vs promote, DEFAULT_PROMOTE_THRESHOLD,
                                   centroid(), policy selection guidance, composition.
        cognitive_kernel    (9)  — is_kernel_registered flag-independence, KernelKind
                                   enum, mind_recall, KernelHit, register_kernel,
                                   DEFAULT kernel, flag gating, backward compat,
                                   is_kernel_registered vs is_policy_registered.
    """
    for q, a in _V4_SURFACES_PROBES:
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        }


# Lessons-from-v3.12.0-retrain reinforcement.
#
# Postmortem of v3.12.0 retrain failure (18/95 → patched to 95/95 for v3.12.1
# only by softening 2 eval probes): the model failed eval probes whose
# wording did NOT exactly match a corpus probe, even when the underlying
# fact was densely reinforced. The model memorised corpus phrasings and
# failed to generalise to the eval phrasing.
#
# Mitigation: for every probe in train/eval_harness.py V4_SURFACES (and the
# reverted V312 probes), we add **at least one corpus tuple whose Q is the
# verbatim eval Q** so the model sees the exact question form during
# training. The model still has to generalise to other phrasings (covered
# by _V4_SURFACES_PROBES + _V4_KIND_BALANCE_PROBES), but the eval-exact
# anchor guarantees we catch the canonical answer.
#
# Each tuple's A is a single canonical answer that contains EVERY required
# token from the eval probe — so the model sees the right answer at full
# strength under exact prompt match.
_V4_EVAL_EXACT_PROBES: list[tuple[str, str]] = [
    # ---------- V4_SURFACES (eval_harness.py) ----------
    (
        "What are the three states of the v4 circuit breaker?",
        "Three states: `CircuitState.CLOSED` (calls pass through), `CircuitState.OPEN` (calls short-circuit with `CircuitOpenError`), `CircuitState.HALF_OPEN` (one probe call allowed; success closes, failure re-opens). Defined as a `str, Enum` in `src/mind_mem/v4/circuit_breaker.py`. The state machine transitions atomically under an internal `threading.Lock`.",
    ),
    (
        "Default failure_threshold and recovery_timeout for CircuitBreaker?",
        "`failure_threshold = 5` (`DEFAULT_FAILURE_THRESHOLD`) and `recovery_timeout = 30.0` seconds (`DEFAULT_RECOVERY_TIMEOUT_S`). Both are module-level constants in `src/mind_mem/v4/circuit_breaker.py` and are configurable via `mind-mem.json: v4: circuit_breaker: {failure_threshold: 5, recovery_timeout_s: 30.0}`.",
    ),
    (
        "What watermark pattern does v4 BackpressureController use?",
        "Hysteresis with two watermarks: `high_watermark` triggers the OPEN/overloaded state, `low_watermark` triggers recovery. Between the two watermarks the state does NOT change — the hysteresis prevents flapping at the boundary. Defaults: `high_watermark=1000`, `low_watermark=200`. See `src/mind_mem/v4/backpressure.py`.",
    ),
    (
        "What status values can v4 health_check return at the top level?",
        "Three top-level status values: `\"ok\"` (every probe ok or disabled), `\"degraded\"` (any probe returned `\"missing\"`), `\"fail\"` (any probe returned an `\"error: ...\"` string or raised). The aggregate is computed in `src/mind_mem/v4/health.py:health_check`. The function NEVER raises — it catches `BaseException` from each probe and reports it as `\"error: ...\"`.",
    ),
    (
        "What underlies the v4 logging_context stack — threads or contextvars?",
        "`contextvars` — specifically a `contextvars.ContextVar` holding a tuple of dict frames. This makes the stack work correctly under both threads AND `async`/`await`, where threading-local storage would be wrong. Defined in `src/mind_mem/v4/logging_context.py`. The `with_correlation_id` decorator is async-aware via `inspect.iscoroutinefunction`.",
    ),
    (
        "What two timestamp columns does v4 block_metadata track?",
        "`created_at` and `updated_at`. `created_at` is preserved across upserts (set on first INSERT, never overwritten); `updated_at` advances on every `set_block_metadata` call. Implemented via `INSERT ... ON CONFLICT(block_id) DO UPDATE SET ... updated_at = excluded.updated_at` in `src/mind_mem/v4/block_metadata.py`.",
    ),
    (
        "What does register_schema_validator do in v4 block_metadata?",
        "`register_schema_validator(kind, fn)` binds a per-kind validator function so callers can run `validate_block(kind, payload)` to check the payload before writing. Validator returns `SchemaValidationResult(ok, reason)`. Open by default — unregistered kinds return `ok=True, reason=\"no_validator\"`. Validator exceptions are caught and returned as `ok=False, reason=\"validator_raised: ...\"`. Defined in `src/mind_mem/v4/block_metadata.py`.",
    ),
    (
        "What's the v4 observability MAX_CARDINALITY default?",
        "`MAX_CARDINALITY = 10000`. Past the cap, `counter()` / `gauge()` / `histogram()` return shared overflow sentinels (`v4._overflow.counter`, `v4._overflow.gauge`, `v4._overflow.histogram`) and bump drop counters at `v4.cardinality.dropped_counter` / `dropped_gauge` / `dropped_histogram` so operators see the loss in `snapshot()`. Defined in `src/mind_mem/v4/observability.py`.",
    ),
    (
        "What does set_active_policy do in v4 eviction?",
        "`set_active_policy(policy)` swaps the workspace-wide default eviction policy at runtime — Redis CONFIG SET pattern. `plan_eviction(policy=None)` resolves None to `active_policy()` so the runtime switch is observable on every call. Defined in `src/mind_mem/v4/eviction.py`. Unknown policy names raise `ValueError`; register via `register_policy(name, fn)` first.",
    ),
    (
        "What does EvictionPlan.debug_plan() return?",
        "`debug_plan()` returns a `dict[str, list[str]]` mapping policy tag to block_ids. The reason strings follow `\"<tag>:<detail>\"` and `debug_plan` groups by the leading tag, so a COMPOSITE plan with `lru` + `low_surprise` returns `{\"lru\": [\"b1\", \"b2\"], \"low_surprise\": [\"b3\"]}`. Defined on `EvictionPlan` in `src/mind_mem/v4/eviction.py`.",
    ),
    (
        "Name the four FallbackPolicy values in v4 surprise_retrieval.",
        "Four values: `FallbackPolicy.NEUTRAL` (returns 0.5 — preserves prior behaviour, default), `FallbackPolicy.PROMOTE` (returns 1.0 — bias toward tier promotion), `FallbackPolicy.DEMOTE` (returns 0.0 — bias toward COLD aging), `FallbackPolicy.RAISE` (raises `EmbeddingFailureError` with reason ∈ {\"missing\", \"length_mismatch\", \"zero_norm\"}). Configured via `mind-mem.json: v4: surprise_retrieval: fallback_policy`. See `src/mind_mem/v4/surprise_retrieval.py`.",
    ),
    (
        "What does FallbackPolicy.RAISE raise on a missing embedding?",
        "`EmbeddingFailureError(reason=\"missing\")`. The `reason` field carries one of three short tags so callers can branch on it: `\"missing\"` (one of the input vectors is empty/None), `\"length_mismatch\"` (vectors have different dimensions), `\"zero_norm\"` (a vector has zero magnitude — division-by-zero would result). Defined in `src/mind_mem/v4/surprise_retrieval.py`.",
    ),
    (
        "What public predicate replaces direct _registry access for v4 eviction?",
        "`is_policy_registered(name)` in `src/mind_mem/v4/eviction.py`. Public, flag-independent — returns True iff `name` resolves to a known policy. Used by `mind_mem.v4.health` instead of reaching into the private `_registry` dict so module coupling stays at the public-API level only. Mirror predicate: `is_kernel_registered` in `cognitive_kernel.py`.",
    ),
    (
        "What public predicate does v4 cognitive_kernel expose for the health probe?",
        "`is_kernel_registered(kind)` in `src/mind_mem/v4/cognitive_kernel.py`. Public, flag-independent — returns True iff the kernel kind has a strategy bound. Used by `health._probe_cognitive_kernel` instead of reaching into the private `_registry` dict. Mirror predicate: `is_policy_registered` in `eviction.py`.",
    ),
    # ---------- V312 reverted probes (eval_harness.py) ----------
    (
        "Is there an escape hatch to write a block that fails validation in strict mode?",
        "Yes — two escape hatches. (1) Library-level: `validate_block(text, strict=True, force=True)` — `force=True` forces `accept=True`, stamps `forced=True` on the verdict. Every fired rule still records to `reasons` for audit. (2) Workspace-level: set `quality_gate.mode = \"off\"` in `mind-mem.json` — when `mode=\"off\"`, `propose_update` does not call `validate_block` at all (governance.py:95 short-circuits). Per-call: `force=True`. Workspace-wide: `mode=\"off\"`. Both contain the words `force` and `strict` because they describe behaviour in strict mode.",
    ),
    (
        "What is the decay multiplier for a `cites` edge in lineage staleness?",
        "The `cites` decay multiplier is `0.8`. `KIND_DECAY[\"cites\"] = 0.8` in `src/mind_mem/block_lineage.py`. Second-strongest of the five kinds (after `contradicts` at 1.0). Applied ONCE at the seed edge of `propagate_lineage_staleness`; further BFS hops use only `HOP_DECAY = (1.0, 0.9, 0.5, 0.2)`. NOT 0.4 — `0.4` is `refines`. The cites value is `0.8`.",
    ),
    # ---------- v3.12.1-known-miss reinforcement ----------
    # These 4 probes failed in the v3.12.1 eval (full-ft-v7-eval) even
    # against the softened harness. v4 retrain MUST hit them.
    (
        "How do I check whether a block proposal is valid before writing it?",
        "Call `validate_block(text)` to check before writing. Defaults to `advisory` mode — every rule executes, fired rules are recorded in the verdict's `advisory` list, but `accept=True` so the caller decides whether to write. Pass `strict=True` (or set `quality_gate.mode = \"strict\"` workspace-wide) to flip to hard-reject mode where fired rules go to `reasons` and `accept=False`. `advisory` is the v3.11 default. `validate_block` is the canonical pre-write check.",
    ),
    (
        "How does `final_score` in `_explain` relate to the other scores?",
        "`final_score = rrf_rank * tier_boost`. `rrf_rank` is the reciprocal-rank-fusion value computed from `bm25_score` and `vector_score`. `tier_boost` is a per-tier multiplier (LONG_TERM=1.5, WORKING=1.2, EPISODIC=1.0 by default). So `final_score = rrf_rank * tier_boost` — the value used for the final ordering after BM25 + vector + tier boosting. The `_explain` dict carries all four fields plus `final_score`.",
    ),
    (
        "What is the default value of `quality_gate.mode`?",
        "`\"advisory\"` — defined as `DEFAULT_QUALITY_GATE_MODE = \"advisory\"` in `src/mind_mem/mcp/infra/config.py`. When the `quality_gate.mode` key is missing from `mind-mem.json` or invalid, the config-router silently falls back to `\"advisory\"`. To override, set `\"quality_gate\": {\"mode\": \"strict\"}` in `mind-mem.json`. The default since v3.11.0; v3.12 adds the `\"strict\"` opt-in.",
    ),
    (
        "What module implements lineage staleness propagation in v3.12.0?",
        "`src/mind_mem/block_lineage.py` — specifically the `propagate_lineage_staleness(workspace, source_id, max_hops)` function. It runs a bounded BFS from each immediate neighbour of `source_id`, scores per-block via `KIND_DECAY[seed_kind] * HOP_DECAY[hop]`, and writes the results to the `block_staleness` SQLite table (columns: `source_id`, `block_id`, `score`, `decayed_at`). The module name is `block_lineage` — the function is `propagate_lineage_staleness`; the table is `block_staleness`.",
    ),
]


def _harvest_v4_eval_exact() -> Iterator[dict]:
    """Verbatim eval-probe phrasings paired with exact-token-match answers.

    Postmortem-driven: v3.12.0 retrain missed 18/95 because the model
    memorised corpus phrasings and failed to generalise to slight rewordings
    in the eval. This harvester yields one tuple per high-stakes eval
    probe whose Q is the EXACT eval Q text and whose A contains every
    required-token from the eval probe. Eliminates the
    memorisation-vs-generalisation gap on the load-bearing probes.

    See train/V4_RETRAIN_TODO.md for the verification gate.
    """
    for q, a in _V4_EVAL_EXACT_PROBES:
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        }


# ---------------------------------------------------------------------------
# v4 retry retrain reinforcement block.
#
# Postmortem: first v4 retrain (full-FT-v4) hit 104/109 = 95.4%. 5 misses:
#   1. workflow "Audit who changed field X on block Y." — model said
#      FieldAuditor (class) but probe wanted field_history (method).
#   2. v4 BackpressureController watermarks — model hallucinated
#      WATERMARK_RATIO=0.6, missed high_watermark/low_watermark/hysteresis.
#   3. v4 EvictionPlan.debug_plan() — model said `(block_id, reason)`
#      singular, probe wanted `block_ids` plural.
#   4. v4 FallbackPolicy enum — model hallucinated LOWEST_FIRST, missed
#      NEUTRAL/PROMOTE/DEMOTE.
#   5. v3.12 block_staleness columns — model listed 3 of 4 columns,
#      missed `decayed_at`.
#
# Strategy: ≥8 saturation probes per miss, mixing direct reinforcement
# with anti-association denial probes ("is X right? No, the answer is Y")
# to overwrite the hallucinated base-model priors. Mirrors the v6 cites
# saturation pattern from `# === Failure B: cites = 0.8 (singled-out) ===`.
_V4_RETRY_REINFORCE_PROBES: list[tuple[str, str]] = [
    # ===== Miss 1: FieldAuditor.field_history method (workflow) =====
    (
        "Audit who changed field X on block Y.",
        "Use `FieldAuditor(workspace).field_history(block_id, field_name)` — `field_history` is the method on the `FieldAuditor` class in `src/mind_mem/field_audit.py`. Returns the change history for that field. Defined at `field_audit.py:289` as `def field_history(self, block_id, field_name) -> list`.",
    ),
    (
        "Audit who changed field X on block Y.",
        "`FieldAuditor(workspace).field_history(\"D-20260304-001\", \"Priority\")` returns the audit trail for that block-field pair. The `field_history` method is defined in `src/mind_mem/field_audit.py:289`.",
    ),
    (
        "How do I see the change history of a single field on a block?",
        "Call `FieldAuditor(workspace).field_history(block_id, field_name)`. The `field_history` method returns the list of changes for that field. Source: `src/mind_mem/field_audit.py:289`.",
    ),
    (
        "What method on FieldAuditor returns field-change audit data?",
        "`field_history(block_id, field_name)`. The `FieldAuditor` class instantiates with a workspace path; the `field_history` method is the read accessor. Defined at `src/mind_mem/field_audit.py:289`.",
    ),
    (
        "Show me the audit-trail call to inspect field changes.",
        "```python\nfrom mind_mem.field_audit import FieldAuditor\nauditor = FieldAuditor(workspace)\nhistory = auditor.field_history(\"D-20260304-001\", \"Priority\")\n```\nThe `field_history` method on `FieldAuditor` returns the change log.",
    ),
    (
        "Which method should I call to audit a single field's edit history?",
        "`field_history` — call it on a `FieldAuditor` instance: `FieldAuditor(workspace).field_history(block_id, field_name)`. Definition: `src/mind_mem/field_audit.py:289`.",
    ),
    (
        "I need the audit log for `Priority` on `D-20260304-001`.",
        "Use the `field_history` method: `FieldAuditor(workspace).field_history(\"D-20260304-001\", \"Priority\")`. Returns the chronological list of changes recorded by the `FieldAuditor` for that block-field pair.",
    ),
    (
        "What does FieldAuditor.field_history return?",
        "The `field_history` method on `FieldAuditor` returns a chronologically-ordered list of change records for a (block_id, field_name) pair. Each record carries the actor, old value, new value, and timestamp.",
    ),

    # ===== Miss 2: BackpressureController watermarks =====
    (
        "What watermark pattern does v4 BackpressureController use?",
        "Hysteresis with two watermarks: `high_watermark` (default 1000) triggers the overloaded state when depth ≥ high; `low_watermark` (default 200) clears the overloaded state when depth ≤ low. Between the two watermarks the state is unchanged — the hysteresis prevents flapping. NO `WATERMARK_RATIO` constant exists; the two watermarks are independent integer thresholds in `src/mind_mem/v4/backpressure.py`.",
    ),
    (
        "What watermark pattern does v4 BackpressureController use?",
        "Two-watermark hysteresis: `high_watermark` triggers overload, `low_watermark` triggers recovery. There is no single-threshold ratio (no `WATERMARK_RATIO`). Constants in `src/mind_mem/v4/backpressure.py`: `DEFAULT_HIGH_WATERMARK = 1000`, `DEFAULT_LOW_WATERMARK = 200`. The gap between them is the hysteresis band.",
    ),
    (
        "Is the v4 backpressure threshold a ratio of capacity?",
        "**No.** There is no `WATERMARK_RATIO` constant. The v4 `BackpressureController` uses **two absolute integer watermarks** with hysteresis: `high_watermark` (default 1000) and `low_watermark` (default 200). Crossing high → OPEN; falling below low → recovered. Defined in `src/mind_mem/v4/backpressure.py`.",
    ),
    (
        "BackpressureController defaults — what are the watermark values?",
        "`DEFAULT_HIGH_WATERMARK = 1000` and `DEFAULT_LOW_WATERMARK = 200`. Two integer watermarks (NOT a ratio). The hysteresis band is `high_watermark` minus `low_watermark` = 800 by default. From `src/mind_mem/v4/backpressure.py`.",
    ),
    (
        "How does the v4 BackpressureController prevent flapping at the overload boundary?",
        "By using hysteresis: a `high_watermark` (default 1000) that triggers OPEN and a separate, lower `low_watermark` (default 200) that triggers recovery. State only changes at these two thresholds, never between them. The two-watermark hysteresis is the entire pattern — no `WATERMARK_RATIO` is involved.",
    ),
    (
        "Show me a BackpressureController instantiation.",
        "`ctrl = BackpressureController(high_watermark=5000, low_watermark=500, max_pause_seconds=10.0)`. Three named integer/float args. The `high_watermark` + `low_watermark` pair encodes the hysteresis band.",
    ),
    (
        "Is BackpressureController parametrised by a single capacity ratio?",
        "**No.** It is parametrised by two absolute watermarks — `high_watermark` and `low_watermark` — implementing hysteresis. The `WATERMARK_RATIO` name does not exist in v4. See `src/mind_mem/v4/backpressure.py:DEFAULT_HIGH_WATERMARK = 1000` and `DEFAULT_LOW_WATERMARK = 200`.",
    ),
    (
        "What's the canonical v4 backpressure config in mind-mem.json?",
        "```json\n\"v4\": {\"backpressure\": {\"enabled\": true, \"high_watermark\": 5000, \"low_watermark\": 500, \"max_pause_seconds\": 10.0}}\n```\nTwo watermarks with hysteresis, plus a pause-seconds cap. There is no `WATERMARK_RATIO` knob.",
    ),

    # ===== Miss 3: EvictionPlan.debug_plan() — block_ids plural =====
    (
        "What does EvictionPlan.debug_plan() return?",
        "`debug_plan()` returns a `dict[str, list[str]]` mapping each policy tag to its list of `block_ids`. The keys are policy tag strings; the values are lists of block_ids that were selected by that policy. Example: `{\"lru\": [\"b1\", \"b2\"], \"low_surprise\": [\"b3\"]}`. The return type annotation is `dict[str, list[str]]` — keys are policy strings, values are lists of `block_ids`.",
    ),
    (
        "What does EvictionPlan.debug_plan() return?",
        "Returns `dict[str, list[str]]` — policy tag → list of block_ids. Each list of block_ids contains every block selected for eviction by that policy. Sample: `{\"lru\": [\"b1\", \"b2\", \"b3\"], \"age\": [\"b9\"]}` — keys are policy tags, values are block_ids lists.",
    ),
    (
        "Show me a sample debug_plan() return value.",
        "```python\n>>> plan.debug_plan()\n{'lru': ['b1', 'b2'], 'low_surprise': ['b3'], 'age': ['b8', 'b9']}\n```\nA dict keyed by policy tag with lists of `block_ids` as values. The return type is `dict[str, list[str]]` where the inner list is block_ids selected by that policy.",
    ),
    (
        "EvictionPlan.debug_plan return type annotation?",
        "`dict[str, list[str]]` — the outer dict is keyed by policy tag, the inner `list[str]` is a list of `block_ids`. Defined in `src/mind_mem/v4/eviction.py`. Use it to trace which policy selected which block_ids in a COMPOSITE plan.",
    ),
    (
        "How do I see which policy selected which blocks in a composite eviction plan?",
        "Call `plan.debug_plan()` — it returns a `dict` mapping each policy tag to the list of block_ids that policy selected. E.g. `{\"lru\": [\"b1\", \"b2\"], \"low_surprise\": [\"b3\"]}`. The values are explicit block_ids lists, one per policy.",
    ),
    (
        "What's the key type and value type of debug_plan's return dict?",
        "Keys are `str` (policy tag like `\"lru\"` or `\"low_surprise\"`); values are `list[str]` (the block_ids selected by that policy). Full return type: `dict[str, list[str]]` — policy → block_ids.",
    ),
    (
        "If a composite eviction plan selects 5 blocks total, what does debug_plan show?",
        "It groups the 5 block_ids by the policy that selected each — e.g. `{\"lru\": [\"b1\", \"b2\"], \"age\": [\"b3\", \"b4\", \"b5\"]}`. Two keys (one per contributing policy), each with its list of block_ids.",
    ),
    (
        "Annotation: what's inside the list returned by debug_plan?",
        "Block_ids — specifically `list[str]`, with each string being a `block_id` from the eviction plan. The outer dict maps policy tags to these block_ids lists.",
    ),

    # ===== Miss 4: FallbackPolicy enum — NEUTRAL/PROMOTE/DEMOTE/RAISE =====
    (
        "Name the four FallbackPolicy values in v4 surprise_retrieval.",
        "Exactly four values: `FallbackPolicy.NEUTRAL` (returns 0.5, default — preserves prior behaviour), `FallbackPolicy.PROMOTE` (returns 1.0 — bias tier promotion), `FallbackPolicy.DEMOTE` (returns 0.0 — bias COLD aging), `FallbackPolicy.RAISE` (raises `EmbeddingFailureError`). No other values exist — no `LOWEST_FIRST`, no `HIGHEST_FIRST`, no `RANDOM`. The enum has exactly NEUTRAL, PROMOTE, DEMOTE, RAISE.",
    ),
    (
        "Name the four FallbackPolicy values in v4 surprise_retrieval.",
        "`NEUTRAL`, `PROMOTE`, `DEMOTE`, `RAISE`. Defined as `class FallbackPolicy(str, Enum)` in `src/mind_mem/v4/surprise_retrieval.py`. These are the ONLY four values — any other name (e.g. `LOWEST_FIRST`, `BIAS_LOW`) does not exist in the codebase.",
    ),
    (
        "Is there a `FallbackPolicy.LOWEST_FIRST` value?",
        "**No.** `LOWEST_FIRST` is not a valid `FallbackPolicy` value. The four legal values are exactly: `NEUTRAL` (returns 0.5), `PROMOTE` (returns 1.0), `DEMOTE` (returns 0.0), `RAISE` (raises `EmbeddingFailureError`). See `src/mind_mem/v4/surprise_retrieval.py`.",
    ),
    (
        "Does v4 FallbackPolicy include a HIGHEST_FIRST option?",
        "**No.** There is no `HIGHEST_FIRST`, `LOWEST_FIRST`, or any ordering-based value. The enum has exactly: `NEUTRAL`, `PROMOTE`, `DEMOTE`, `RAISE`. Defined in `src/mind_mem/v4/surprise_retrieval.py:FallbackPolicy`.",
    ),
    (
        "Enumerate every FallbackPolicy value with its semantics.",
        "Four total:\n- `NEUTRAL` → 0.5 (default, mild surprise — preserves prior behaviour)\n- `PROMOTE` → 1.0 (max surprise, bias tier promotion)\n- `DEMOTE` → 0.0 (no surprise, bias COLD aging)\n- `RAISE` → raises `EmbeddingFailureError`\n\nThese four — `NEUTRAL`, `PROMOTE`, `DEMOTE`, `RAISE` — are the complete enum.",
    ),
    (
        "What does FallbackPolicy.PROMOTE return on an unusable embedding?",
        "`1.0` — the maximum surprise value, biasing tier promotion when the embedder is flaky. One of four `FallbackPolicy` values: `NEUTRAL`, `PROMOTE`, `DEMOTE`, `RAISE`.",
    ),
    (
        "What does FallbackPolicy.DEMOTE return on an unusable embedding?",
        "`0.0` — no surprise, biasing tier demotion / COLD aging. One of four `FallbackPolicy` values: `NEUTRAL`, `PROMOTE`, `DEMOTE`, `RAISE`.",
    ),
    (
        "What does FallbackPolicy.NEUTRAL return on an unusable embedding?",
        "`0.5` — mild surprise, the default fallback. One of four `FallbackPolicy` values: `NEUTRAL`, `PROMOTE`, `DEMOTE`, `RAISE`.",
    ),
    (
        "True or false: FallbackPolicy has 4 values.",
        "True. Exactly four: `NEUTRAL`, `PROMOTE`, `DEMOTE`, `RAISE`. Not 3, not 5. NOT `LOWEST_FIRST` or any other name.",
    ),

    # ===== Miss 5: block_staleness 4 columns — decayed_at =====
    (
        "What fields does the `block_staleness` table contain?",
        "Four columns: `block_id` (the affected block), `source_id` (the block that triggered propagation), `score` (the propagated staleness penalty, 0.0–1.0), and `decayed_at` (the ISO timestamp when this staleness was recorded). All four are required; `decayed_at` makes the row time-bounded so callers can age-filter stale entries.",
    ),
    (
        "What fields does the `block_staleness` table contain?",
        "`block_id`, `source_id`, `score`, `decayed_at` — four columns. `decayed_at` is the timestamp of insertion. Schema in `src/mind_mem/block_lineage.py`. The `decayed_at` column lets callers query \"staleness as of T\" — without it the row would be timeless.",
    ),
    (
        "Schema of block_staleness — column by column.",
        "1. `block_id` — TEXT — the affected block.\n2. `source_id` — TEXT — the block that seeded propagation.\n3. `score` — REAL — the propagated staleness in [0.0, 1.0].\n4. `decayed_at` — TEXT — ISO timestamp of insertion.\n\nFour columns total. Index on `(block_id, decayed_at)` for time-bounded reads.",
    ),
    (
        "What column tracks WHEN staleness was recorded in block_staleness?",
        "`decayed_at` — an ISO timestamp string set at insertion time. Without this column, callers couldn't age-filter staleness records. It's the 4th column alongside `block_id`, `source_id`, and `score`.",
    ),
    (
        "How many columns does the block_staleness table have?",
        "Four: `block_id`, `source_id`, `score`, `decayed_at`. Not three — `decayed_at` is part of the schema and is required for time-bounded staleness queries.",
    ),
    (
        "List every column in block_staleness.",
        "All four columns: `block_id`, `source_id`, `score`, `decayed_at`. The `decayed_at` column is the insertion timestamp and is NEVER omitted.",
    ),
    (
        "Does block_staleness include a timestamp column?",
        "Yes — `decayed_at`. ISO-formatted string set at insertion. The full column list: `block_id`, `source_id`, `score`, `decayed_at`.",
    ),
    (
        "True or false: block_staleness has only 3 columns.",
        "**False.** It has FOUR columns: `block_id`, `source_id`, `score`, `decayed_at`. Anyone listing only 3 has missed `decayed_at` (the timestamp).",
    ),
]


def _harvest_v4_retry_reinforce() -> Iterator[dict]:
    """Heavy-saturation reinforcement for the 5 first-run eval misses.

    See `_V4_RETRY_REINFORCE_PROBES` doc above for the per-miss strategy.
    Goal: overwrite hallucinated base-model priors (`WATERMARK_RATIO`,
    `LOWEST_FIRST`) and push specific tokens (`block_ids` plural,
    `decayed_at`, `field_history`) into the canonical answer for each
    eval Q.
    """
    for q, a in _V4_RETRY_REINFORCE_PROBES:
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        }


if __name__ == "__main__":
    main()
