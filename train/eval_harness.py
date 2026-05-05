"""Eval harness for mind-mem-4b.

Runs three benchmarks the model-card promises:
    1. Tool-call accuracy      — ≥ 95% target.
    2. Block-schema conformance — ≥ 98% target.
    3. Governance workflow      — ≥ 90% target.

Prints a report to stdout and writes JSON to
/home/n/mm-train-output/eval_report.json.  Exit 0 if all three
targets hit, exit 1 otherwise — so CI can gate uploads on a green
eval.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = os.environ.get("MM_BASE_MODEL", "Qwen/Qwen3.5-4B")
_BASE_DIR = Path(os.environ.get("MM_TRAIN_ROOT", "/data/checkpoints/mm-workspace/train-output"))
ADAPTER = _BASE_DIR / "adapter"
REPORT = _BASE_DIR / "eval_report.json"


# ---------------------------------------------------------------------------
# Test sets — small on purpose: the training data is ~400 examples, so
# a test set of 50 items is ~12% held-out equivalent.
# ---------------------------------------------------------------------------


TOOL_CALL_QUESTIONS: list[tuple[str, str]] = [
    ("How do I verify the audit chain integrity?", "verify_chain"),
    ("What tool lists all contradictions?", "list_contradictions"),
    ("Apply a staged proposal.", "approve_apply"),
    ("Roll back an apply.", "rollback_proposal"),
    ("Write a new memory block.", "propose_update"),
    ("Full-text search over memory.", "recall"),
    ("Run hybrid BM25 + vector search.", "hybrid_search"),
    ("Reindex the FTS tables.", "reindex"),
    ("Scan the workspace for contradictions and drift.", "scan"),
    ("Export memory to JSONL.", "export_memory"),
    ("Classify the intent of a query.", "intent_classify"),
    ("Get index statistics.", "index_stats"),
    ("Run the governance health benchmark.", "governance_health_bench"),
    ("Encrypt a workspace file at rest.", "encrypt_file"),
    ("Decrypt an encrypted workspace file.", "decrypt_file"),
    ("Find blocks similar to a given ID.", "find_similar"),
    ("Delete a memory item.", "delete_memory_item"),
    ("Show block category summaries.", "category_summary"),
    ("Prefetch recall results.", "prefetch"),
    ("Diagnose why a query returned its results.", "retrieval_diagnostics"),
]


BLOCK_SCHEMA_QUESTIONS: list[tuple[str, list[str]]] = [
    ("Show me a DREF block template.", ["[DREF-", "Date:", "Severity:", "Signal:"]),
    ("Show me an ADR block template.", ["[ADR-", "Date:", "Status:", "Decision:"]),
    ("Show me a BUG block template.", ["[BUG-", "Severity:", "Symptom:", "RootCause:"]),
    ("Show me a PERF block template.", ["[PERF-", "Metric:", "Before:", "After:"]),
    ("Show me a CODE block template.", ["[CODE-", "File:", "Change:", "Rationale:"]),
    ("Show me an ALGO block template.", ["[ALGO-", "Problem:", "Chosen:"]),
    ("Show me a CONV block template.", ["[CONV-", "Convention:"]),
    ("Show me a CHECK block template.", ["[CHECK-", "BlockA:", "BlockB:"]),
    ("Show me an EV block template.", ["[EV-", "Action:", "Actor:"]),
    ("Show me a FIELD block template.", ["[FIELD-", "BlockId:", "Field:", "Old:", "New:"]),
]


WORKFLOW_QUESTIONS: list[tuple[str, list[str]]] = [
    (
        "I see a contradiction between two decision blocks. Walk me through the fix.",
        ["list_contradictions", "propose_update", "approve_apply", "verify_chain"],
    ),
    (
        "I applied a bad proposal. How do I roll back safely?",
        ["rollback_proposal", "verify_chain"],
    ),
    (
        "I want to check if a belief has drifted. Which tools do I call?",
        ["scan", "recent_signals"],
    ),
    (
        "Audit who changed field X on block Y.",
        ["field_history", "FieldAuditor"],
    ),
    (
        "Run the full governance benchmark suite.",
        ["governance_health_bench"],
    ),
]


# ---------------------------------------------------------------------------
# v3.9 probes — guard against drift on surfaces shipped 2026-05.
#
# 1. New-tool name recall: every tool added since v3.0 must surface by
#    name when prompted by intent. Target ≥ 90% (24 tools).
# 2. TransformHash field citation: the v3.9 hash-of-code field must
#    show up in block-write examples. Target ≥ 95% (3 prompts).
# 3. Transport endpoint hallucination guard: HTTP/inbox/daemon
#    transports are real but their endpoint paths are bounded. Anything
#    outside the allowlist is a hallucination. Target ≥ 95% (4 prompts).
# ---------------------------------------------------------------------------


V39_NEW_TOOLS: list[tuple[str, str]] = [
    # MCP wrapping for v3.9 walkthrough/persona + hash-of-code (PR #522)
    ("Show the dependency-ordered learning sequence for a topic.", "compile_truth_walkthrough"),
    ("Recall blocks and project them through a persona.", "recall_with_persona"),
    ("Inspect the current pipeline hash and dirty-block summary.", "pipeline_status"),
    ("Re-stamp blocks whose TransformHash is stale.", "reindex_dirty"),
    # Model audit + signing pipeline (v3.8.1–v3.8.3 MCP wrappers)
    ("Run the seven-check audit on a local model checkpoint via MCP.", "audit_model_tool"),
    ("Sign a checkpoint manifest with Ed25519 via MCP.", "sign_model_tool"),
    ("Verify an Ed25519 manifest signature via MCP.", "verify_model_tool"),
    # MIC/MAP serialization surface (v3.8.5/v3.8.11)
    ("Convert a MIND IR graph between mic@2 text and mic-b binary.", "mic_convert"),
    ("Inspect the structure of a serialized MIC/MAP graph.", "mic_inspect"),
    # Audit chain (v3.8 via MCP)
    ("Verify the cryptographic chain of evidence for a block by Merkle path.", "verify_merkle"),
    ("List every evidence entry the audit chain has accepted so far.", "list_evidence"),
    # Kernels surface
    ("Get a registered MIND kernel by name.", "get_mind_kernel"),
    ("List every registered MIND kernel.", "list_mind_kernels"),
]


V39_TRANSFORMHASH_PROMPTS: list[tuple[str, list[str]]] = [
    (
        "Show me the field name a v3.9 inbox-ingested block carries to record the pipeline hash.",
        ["TransformHash"],
    ),
    (
        "Which helper stamps the current pipeline hash onto a block dict before writing?",
        ["stamp_transform_hash"],
    ),
    (
        "How do I bulk re-stamp blocks whose pipeline hash drifted?",
        ["reextract_dirty_blocks"],
    ),
]


# Allowlist of real v3.9 transport endpoints. Anything else is a hallucination.
V39_HTTP_ENDPOINTS_ALLOWED = {"/status", "/query", "/memories", "/consolidate", "/search", "/walkthrough"}


V39_TRANSPORT_PROMPTS: list[tuple[str, list[str], list[str]]] = [
    # (prompt, must-include, must-NOT-include)
    (
        "List the v3.9 HTTP transport endpoints.",
        ["/status", "/query"],
        ["/admin", "/auth/login", "/users", "/embed"],
    ),
    (
        "How does the v3.9 daemon trigger the dream cycle?",
        ["daemon"],
        ["/cron", "/api/v2/dream"],
    ),
    (
        "How do I drop a file into the v3.9 inbox for ingestion?",
        ["inbox", "ingest"],
        ["/upload", "POST /file"],
    ),
    (
        "How does the v3.9 replicated postgres backend handle writes vs reads?",
        ["primary", "round-robin"],
        ["sharding", "consistent-hash"],
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _load_model():
    """Load eval target. Two modes:

    * QLoRA adapter overlay (legacy v3.0):  4-bit base + PEFT adapter at
      `${MM_TRAIN_ROOT}/adapter`.
    * Full-fine-tune (v3.9+):  load `${MM_FULLFT_DIR}` (a directory
      containing config.json + model.safetensors) directly. 4-bit
      quantization is still applied on the 3080 so the merged model
      fits in 10 GB VRAM.
    """
    fullft_dir = Path(os.environ.get("MM_FULLFT_DIR", _BASE_DIR / "full-ft"))
    use_fullft = (fullft_dir / "model.safetensors").is_file() or (fullft_dir / "model.safetensors.index.json").is_file()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if use_fullft:
        print(f"loading full-FT weights from {fullft_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(fullft_dir), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(fullft_dir),
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()
        return tokenizer, model

    if not ADAPTER.is_dir():
        sys.exit(f"no weights found — checked full-FT dir {fullft_dir} and adapter {ADAPTER}. Train first.")
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    from peft import PeftModel  # lazy: only the legacy QLoRA path needs peft
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.eval()
    return tokenizer, model


def _chat(tokenizer, model, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are mind-mem-4b, a memory-governance assistant."},
        {"role": "user", "content": prompt},
    ]
    # New transformers versions return a BatchEncoding (dict-like) from
    # apply_chat_template. Older returned a tensor directly. Handle both.
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    if hasattr(encoded, "to"):
        encoded = encoded.to(model.device)
    else:
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
    input_len = encoded["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**encoded, max_new_tokens=256, do_sample=False)
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)


def _bench_tool_calls(tokenizer, model) -> dict:
    hits = 0
    misses: list[dict] = []
    for prompt, expected in TOOL_CALL_QUESTIONS:
        resp = _chat(tokenizer, model, prompt).lower()
        if expected.lower() in resp:
            hits += 1
        else:
            misses.append({"prompt": prompt, "expected": expected, "response": resp[:200]})
    total = len(TOOL_CALL_QUESTIONS)
    return {"accuracy": hits / total, "hits": hits, "total": total, "misses": misses}


def _bench_block_schemas(tokenizer, model) -> dict:
    hits = 0
    misses: list[dict] = []
    for prompt, required_tokens in BLOCK_SCHEMA_QUESTIONS:
        resp = _chat(tokenizer, model, prompt)
        if all(tok in resp for tok in required_tokens):
            hits += 1
        else:
            missing = [tok for tok in required_tokens if tok not in resp]
            misses.append({"prompt": prompt, "missing": missing, "response": resp[:200]})
    total = len(BLOCK_SCHEMA_QUESTIONS)
    return {"accuracy": hits / total, "hits": hits, "total": total, "misses": misses}


def _bench_workflows(tokenizer, model) -> dict:
    hits = 0
    misses: list[dict] = []
    for prompt, required_tools in WORKFLOW_QUESTIONS:
        resp = _chat(tokenizer, model, prompt)
        if all(tool in resp for tool in required_tools):
            hits += 1
        else:
            missing = [tool for tool in required_tools if tool not in resp]
            misses.append({"prompt": prompt, "missing": missing, "response": resp[:200]})
    total = len(WORKFLOW_QUESTIONS)
    return {"accuracy": hits / total, "hits": hits, "total": total, "misses": misses}


def _bench_v39_new_tools(tokenizer, model) -> dict:
    """v3.9 probe 1: every tool added since v3.0 must surface by name."""
    hits = 0
    misses: list[dict] = []
    for prompt, expected in V39_NEW_TOOLS:
        resp = _chat(tokenizer, model, prompt).lower()
        if expected.lower() in resp:
            hits += 1
        else:
            misses.append({"prompt": prompt, "expected": expected, "response": resp[:200]})
    total = len(V39_NEW_TOOLS)
    return {"accuracy": hits / total, "hits": hits, "total": total, "misses": misses}


def _bench_v39_transform_hash(tokenizer, model) -> dict:
    """v3.9 probe 2: TransformHash field + helpers must be cited."""
    hits = 0
    misses: list[dict] = []
    for prompt, required_tokens in V39_TRANSFORMHASH_PROMPTS:
        resp = _chat(tokenizer, model, prompt)
        if all(tok in resp for tok in required_tokens):
            hits += 1
        else:
            missing = [tok for tok in required_tokens if tok not in resp]
            misses.append({"prompt": prompt, "missing": missing, "response": resp[:200]})
    total = len(V39_TRANSFORMHASH_PROMPTS)
    return {"accuracy": hits / total, "hits": hits, "total": total, "misses": misses}


def _bench_v39_transport_guard(tokenizer, model) -> dict:
    """v3.9 probe 3: transport mentions stay inside the real endpoint allowlist."""
    hits = 0
    misses: list[dict] = []
    for prompt, must_include, must_not_include in V39_TRANSPORT_PROMPTS:
        resp = _chat(tokenizer, model, prompt)
        ok = all(tok in resp for tok in must_include) and not any(tok in resp for tok in must_not_include)
        if ok:
            hits += 1
        else:
            missing = [tok for tok in must_include if tok not in resp]
            hallucinated = [tok for tok in must_not_include if tok in resp]
            misses.append(
                {
                    "prompt": prompt,
                    "missing": missing,
                    "hallucinated": hallucinated,
                    "response": resp[:200],
                }
            )
    total = len(V39_TRANSPORT_PROMPTS)
    return {"accuracy": hits / total, "hits": hits, "total": total, "misses": misses}


def main() -> None:
    tokenizer, model = _load_model()
    tool_bench = _bench_tool_calls(tokenizer, model)
    schema_bench = _bench_block_schemas(tokenizer, model)
    workflow_bench = _bench_workflows(tokenizer, model)
    v39_new_tools_bench = _bench_v39_new_tools(tokenizer, model)
    v39_xform_bench = _bench_v39_transform_hash(tokenizer, model)
    v39_transport_bench = _bench_v39_transport_guard(tokenizer, model)

    report = {
        "tool_call": tool_bench,
        "block_schema": schema_bench,
        "workflow": workflow_bench,
        "v39_new_tools": v39_new_tools_bench,
        "v39_transform_hash": v39_xform_bench,
        "v39_transport_guard": v39_transport_bench,
        "targets": {
            "tool_call": 0.95,
            "block_schema": 0.98,
            "workflow": 0.90,
            "v39_new_tools": 0.90,
            "v39_transform_hash": 0.95,
            "v39_transport_guard": 0.95,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 60)
    print("mind-mem-4b v3.9.0 eval report")
    print("=" * 60)
    for name, bench, target in (
        ("tool_call          ", tool_bench, 0.95),
        ("block_schema       ", schema_bench, 0.98),
        ("workflow           ", workflow_bench, 0.90),
        ("v39_new_tools      ", v39_new_tools_bench, 0.90),
        ("v39_transform_hash ", v39_xform_bench, 0.95),
        ("v39_transport_guard", v39_transport_bench, 0.95),
    ):
        pass_str = "PASS" if bench["accuracy"] >= target else "FAIL"
        print(f"  {name}  {bench['hits']:3d}/{bench['total']:<3d}  {bench['accuracy']:.2%}   (target {target:.0%})  [{pass_str}]")
    print(f"\nreport → {REPORT}")

    passed = (
        tool_bench["accuracy"] >= 0.95
        and schema_bench["accuracy"] >= 0.98
        and workflow_bench["accuracy"] >= 0.90
        and v39_new_tools_bench["accuracy"] >= 0.90
        and v39_xform_bench["accuracy"] >= 0.95
        and v39_transport_bench["accuracy"] >= 0.95
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
