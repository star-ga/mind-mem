"""Eval harness for mind-mem-7b.

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
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = Path("/home/n/mm-train-output/adapter")
REPORT = Path("/home/n/mm-train-output/eval_report.json")


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
# Runner
# ---------------------------------------------------------------------------


def _load_model():
    if not ADAPTER.is_dir():
        sys.exit(f"no adapter at {ADAPTER}. Run train_qlora.py first.")
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    # Match training quantization so the base + adapter fit on the 3080.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.eval()
    return tokenizer, model


def _chat(tokenizer, model, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are mind-mem-7b, a memory-governance assistant."},
        {"role": "user", "content": prompt},
    ]
    # New transformers versions return a BatchEncoding (dict-like) from
    # apply_chat_template. Older returned a tensor directly. Handle both.
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
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


def main() -> None:
    tokenizer, model = _load_model()
    tool_bench = _bench_tool_calls(tokenizer, model)
    schema_bench = _bench_block_schemas(tokenizer, model)
    workflow_bench = _bench_workflows(tokenizer, model)

    report = {
        "tool_call": tool_bench,
        "block_schema": schema_bench,
        "workflow": workflow_bench,
        "targets": {"tool_call": 0.95, "block_schema": 0.98, "workflow": 0.90},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 60)
    print("mind-mem-7b v2.9.0 eval report")
    print("=" * 60)
    for name, bench, target in (
        ("tool_call   ", tool_bench, 0.95),
        ("block_schema", schema_bench, 0.98),
        ("workflow    ", workflow_bench, 0.90),
    ):
        pass_str = "PASS" if bench["accuracy"] >= target else "FAIL"
        print(
            f"  {name}  {bench['hits']:3d}/{bench['total']:<3d}  "
            f"{bench['accuracy']:.2%}   (target {target:.0%})  [{pass_str}]"
        )
    print(f"\nreport → {REPORT}")

    passed = (
        tool_bench["accuracy"] >= 0.95
        and schema_bench["accuracy"] >= 0.98
        and workflow_bench["accuracy"] >= 0.90
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
