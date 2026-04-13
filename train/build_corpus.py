"""Harvest a training corpus for the mind-mem-7b model.

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
OUT = Path("/home/n/mm-train-output/corpus.jsonl")

SYSTEM_PROMPT = (
    "You are mind-mem-7b, a memory-governance assistant specialised in "
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
    """Extract every @mcp.tool function + docstring from mcp_server.py."""
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
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"What does the mind-mem MCP tool `{node.name}` do?",
                },
                {
                    "role": "assistant",
                    "content": (
                        f"`{node.name}({args})` — {summary}\n\n"
                        f"Full docstring:\n\n{doc.strip()}"
                    ),
                },
            ]
        }
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Which MCP tool should I call to: "
                        f"{_usage_hint(summary)}?"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        f"Use the `{node.name}` tool. Signature: "
                        f"`{node.name}({args})`."
                    ),
                },
            ]
        }


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
    for short, desc, fields in _BLOCK_TYPES:
        example = f"[{short}-20260413-001]\n" + "\n".join(
            f"{f}: <{f.lower()}>" for f in fields
        )
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Show me a mind-mem {short} block template.",
                },
                {
                    "role": "assistant",
                    "content": (
                        f"{desc} Canonical template:\n\n```\n{example}\n```"
                    ),
                },
            ]
        }
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"What fields does a {short} block require?",
                },
                {
                    "role": "assistant",
                    "content": (
                        f"A mind-mem {short} block requires: "
                        + ", ".join(fields)
                        + "."
                    ),
                },
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
]


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
        ):
            for entry in _dedup(src):
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
    print(f"wrote {count} examples to {OUT}")


if __name__ == "__main__":
    main()
