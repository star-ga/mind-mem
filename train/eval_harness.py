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
        ["scan"],
    ),
    (
        "Audit who changed field X on block Y.",
        ["field_history"],
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
        ["reindex_dirty"],
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
# v3.11.0 probes — guard against regressions on the new MCP surface.
#
# 1. v311_new_tools: 10 probes covering validate_block / block_lineage /
#    add_block_edge.  Target ≥ 90%.
# 2. v311_explain_field: 10 probes verifying the model knows the `_explain`
#    field shape emitted by recall(explain=True).  Target ≥ 95% (factual
#    recall — the shape is fixed).
# ---------------------------------------------------------------------------


V311_NEW_TOOLS: list[tuple[str, list[str]]] = [
    (
        "How do I check whether a block proposal is valid before writing it?",
        ["validate_block", "advisory"],
    ),
    (
        "Which tool validates a block without writing it?",
        ["validate_block"],
    ),
    (
        "What does `validate_block(text, strict=True)` do on a rule violation?",
        ["validate_block", "strict"],
    ),
    (
        "How do I detect prompt-injection patterns in a candidate block?",
        ["validate_block", "injection"],
    ),
    (
        "Which v3.11.0 tool checks for duplicate, oversize, and UTF-8 issues before a write?",
        ["validate_block"],
    ),
    (
        "How do I trace what blocks depend on block X?",
        ["block_lineage", "max_depth"],
    ),
    (
        "What does `block_lineage` return for an isolated block with no edges?",
        ["block_lineage"],
    ),
    (
        "How do I filter `block_lineage` to only `cites` edges?",
        ["block_lineage", "kind_filter"],
    ),
    (
        "What tool adds a typed edge between two blocks?",
        ["add_block_edge", "kind"],
    ),
    (
        "What are the five edge kinds supported by `block_lineage`?",
        ["cites", "implements", "refines", "contradicts", "cooccurrence"],
    ),
]


V311_EXPLAIN_FIELD: list[tuple[str, list[str]]] = [
    (
        "What does `recall(query, explain=True)` return that the default call does not?",
        ["_explain", "bm25", "vector", "rrf_rank"],
    ),
    (
        "What fields live inside the `_explain` dict returned by `recall(explain=True)`?",
        ["bm25_score", "vector_score", "rrf_rank", "tier_boost", "final_score"],
    ),
    (
        "Is `_explain` present on every result block when `recall(explain=True)` is used?",
        ["_explain", "explain"],
    ),
    (
        "How does `final_score` in `_explain` relate to the other scores?",
        ["rrf_rank", "tier_boost"],
    ),
    (
        "What is `rrf_rank` inside `_explain`?",
        ["rrf_rank", "bm25", "vector"],
    ),
    (
        "Does `recall` return `_explain` when `explain` is not passed?",
        ["_explain"],
    ),
    (
        "Which `_explain` field should I inspect to diagnose a BM25 vs vector disagreement?",
        ["bm25_score", "vector_score"],
    ),
    (
        "What is the math behind `final_score` in `_explain`?",
        ["rrf_rank", "tier_boost"],
    ),
    (
        "Does `hybrid_search` also support `explain=True`?",
        ["_explain", "explain"],
    ),
    (
        "Is the `_explain` dict the same structure for `recall` and `hybrid_search`?",
        ["bm25_score", "vector_score", "rrf_rank", "tier_boost", "final_score"],
    ),
]


# ---------------------------------------------------------------------------
# v3.12.0 probes — guard against regressions on quality-gate strict mode
# and lineage staleness wiring.
#
# 1. V312_QUALITY_GATE_STRICT_MODE: 10 probes. Target ≥ 90%.
# 2. V312_LINEAGE_STALENESS: 10 probes. Target ≥ 90%.
# ---------------------------------------------------------------------------


V312_QUALITY_GATE_STRICT_MODE: list[tuple[str, list[str]]] = [
    (
        "What are the three valid values for `quality_gate.mode` in `mind-mem.json`?",
        ["off", "advisory", "strict"],
    ),
    (
        "What is the default value of `quality_gate.mode`?",
        ["advisory", "mind-mem.json"],
    ),
    (
        "How do I enable strict mode so that a rule violation blocks a write?",
        ["strict", "mind-mem.json", "validate_block"],
    ),
    (
        'What does `propose_update` do differently when `quality_gate.mode` is `"strict"`?',
        ["strict", "validate_block"],
    ),
    (
        "What is the shape of the rejection envelope when strict mode fires?",
        ["quality_gate_rejection", "reasons", "mode"],
    ),
    (
        "How do I read the per-rule rejection counter for the `injection_marker` rule?",
        ["quality_gate_rejections", "injection_marker"],
    ),
    (
        "What metric key tracks how many times the `near_duplicate` rule rejected a write?",
        ["quality_gate_rejections", "near_duplicate"],
    ),
    (
        "When should I choose `strict` mode over `advisory` mode?",
        ["strict", "advisory"],
    ),
    (
        # v4 RESTORED: original strict requirement reinstated. The v4
        # corpus carries 13+ probes teaching `validate_block(text,
        # strict=True, force=True)` as the canonical library escape
        # hatch (see train/build_corpus.py "=== Failure A: qg escape
        # hatch ===" block at lines 3962-4035). v4 model must hit
        # both tokens or this probe fails — no further softening.
        "Is there an escape hatch to write a block that fails validation in strict mode?",
        ["force", "strict"],
    ),
    (
        "What operator runbook covers the quality gate configuration?",
        ["docs/quality-gate.md"],
    ),
]


V312_LINEAGE_STALENESS: list[tuple[str, list[str]]] = [
    (
        "What new SQLite table does v3.12.0 introduce for staleness tracking?",
        ["block_staleness", "source_id"],
    ),
    (
        "What module implements lineage staleness propagation in v3.12.0?",
        ["propagate_lineage_staleness", "block_staleness"],
    ),
    (
        "How does `_explain.staleness_penalty` behave differently in v3.12.0 vs v3.11.0?",
        ["staleness_penalty", "_explain", "block_staleness"],
    ),
    (
        "What are the kind-aware decay multipliers in `propagate_lineage_staleness`?",
        ["contradicts", "cites", "implements", "refines"],
    ),
    (
        "Why does a `contradicts` edge propagate the fastest in lineage staleness?",
        ["contradicts", "1.0"],
    ),
    (
        "What is the maximum number of hops `propagate_lineage_staleness` will walk?",
        ["max_hops", "3"],
    ),
    (
        "How do I trigger lineage staleness propagation from the CLI?",
        ["mm lineage flag", "contradicts"],
    ),
    (
        "What fields does the `block_staleness` table contain?",
        ["source_id", "decayed_at"],
    ),
    (
        # v4 RESTORED: original strict requirement reinstated. The v4
        # corpus has 30+ singled-out cites=0.8 probes (see
        # _V4_KIND_BALANCE_PROBES) plus contrastive denial probes
        # ("is cites 0.4? No, 0.4 is refines"). v4 model must emit
        # both "cites" and "0.8" or this probe fails — no further
        # softening. If the model still emits 0.4 here, the corpus
        # rebalance didn't take and we DO NOT ship.
        "What is the decay multiplier for a `cites` edge in lineage staleness?",
        ["cites", "0.8"],
    ),
    (
        "What is the decay multiplier for a `refines` edge?",
        ["refines", "0.4"],
    ),
]


# ---------------------------------------------------------------------------
# v4 surfaces — one or two probes per new module so the trained model
# is verified to know the canonical API for every v4 addition.
# Target ≥ 90% pass rate (12+/14). Each probe has strict required-token
# tokens; matches the strict format of the v3.11/v3.12 groups.
# ---------------------------------------------------------------------------

V4_SURFACES: list[tuple[str, list[str]]] = [
    # circuit_breaker.py
    (
        "What are the three states of the v4 circuit breaker?",
        ["CLOSED", "OPEN", "HALF_OPEN"],
    ),
    (
        "Default failure_threshold and recovery_timeout for CircuitBreaker?",
        ["5", "30"],
    ),
    # backpressure.py
    (
        "What watermark pattern does v4 BackpressureController use?",
        ["high_watermark", "low_watermark", "hysteresis"],
    ),
    # health.py
    (
        "What status values can v4 health_check return at the top level?",
        ["ok", "degraded", "fail"],
    ),
    # logging_context.py
    (
        "What underlies the v4 logging_context stack — threads or contextvars?",
        ["contextvars"],
    ),
    # block_metadata.py
    (
        "What two timestamp columns does v4 block_metadata track?",
        ["created_at", "updated_at"],
    ),
    (
        "What does register_schema_validator do in v4 block_metadata?",
        ["validate_block", "kind"],
    ),
    # observability.py cardinality guard
    (
        "What's the v4 observability MAX_CARDINALITY default?",
        ["10000"],
    ),
    # eviction.py debug_plan + active_policy
    (
        "What does set_active_policy do in v4 eviction?",
        ["active_policy", "plan_eviction"],
    ),
    (
        "What does EvictionPlan.debug_plan() return?",
        ["policy", "block_ids"],
    ),
    # surprise_retrieval.py FallbackPolicy
    (
        "Name the four FallbackPolicy values in v4 surprise_retrieval.",
        ["NEUTRAL", "PROMOTE", "DEMOTE", "RAISE"],
    ),
    (
        "What does FallbackPolicy.RAISE raise on a missing embedding?",
        ["EmbeddingFailureError"],
    ),
    # cognitive_kernel.py + eviction.py public predicates
    (
        "What public predicate replaces direct _registry access for v4 eviction?",
        ["is_policy_registered"],
    ),
    (
        "What public predicate does v4 cognitive_kernel expose for the health probe?",
        ["is_kernel_registered"],
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


def _bench_v311_new_tools(tokenizer, model) -> dict:
    """v3.11.0 probe 1: validate_block / block_lineage / add_block_edge."""
    hits = 0
    misses: list[dict] = []
    for prompt, required_tokens in V311_NEW_TOOLS:
        resp = _chat(tokenizer, model, prompt)
        if all(tok in resp for tok in required_tokens):
            hits += 1
        else:
            missing = [tok for tok in required_tokens if tok not in resp]
            misses.append({"prompt": prompt, "missing": missing, "response": resp[:200]})
    total = len(V311_NEW_TOOLS)
    return {"accuracy": hits / total, "hits": hits, "total": total, "misses": misses}


def _bench_v311_explain_field(tokenizer, model) -> dict:
    """v3.11.0 probe 2: _explain field shape on recall(explain=True)."""
    hits = 0
    misses: list[dict] = []
    for prompt, required_tokens in V311_EXPLAIN_FIELD:
        resp = _chat(tokenizer, model, prompt)
        if all(tok in resp for tok in required_tokens):
            hits += 1
        else:
            missing = [tok for tok in required_tokens if tok not in resp]
            misses.append({"prompt": prompt, "missing": missing, "response": resp[:200]})
    total = len(V311_EXPLAIN_FIELD)
    return {"accuracy": hits / total, "hits": hits, "total": total, "misses": misses}


def _bench_v312_quality_gate_strict_mode(tokenizer, model) -> dict:
    """v3.12.0 probe 1: quality_gate.mode config, modes, counters, escape hatch."""
    hits = 0
    misses: list[dict] = []
    for prompt, required_tokens in V312_QUALITY_GATE_STRICT_MODE:
        resp = _chat(tokenizer, model, prompt)
        if all(tok in resp for tok in required_tokens):
            hits += 1
        else:
            missing = [tok for tok in required_tokens if tok not in resp]
            misses.append({"prompt": prompt, "missing": missing, "response": resp[:200]})
    total = len(V312_QUALITY_GATE_STRICT_MODE)
    return {"accuracy": hits / total, "hits": hits, "total": total, "misses": misses}


def _bench_v312_lineage_staleness(tokenizer, model) -> dict:
    """v3.12.0 probe 2: block_staleness table, propagation, decay multipliers."""
    hits = 0
    misses: list[dict] = []
    for prompt, required_tokens in V312_LINEAGE_STALENESS:
        resp = _chat(tokenizer, model, prompt)
        if all(tok in resp for tok in required_tokens):
            hits += 1
        else:
            missing = [tok for tok in required_tokens if tok not in resp]
            misses.append({"prompt": prompt, "missing": missing, "response": resp[:200]})
    total = len(V312_LINEAGE_STALENESS)
    return {"accuracy": hits / total, "hits": hits, "total": total, "misses": misses}


def _bench_v4_surfaces(tokenizer, model) -> dict:
    """v4 probe: 9 new modules — circuit_breaker, backpressure, health,
    logging_context, block_metadata, observability cardinality, eviction
    debug_plan/active_policy, surprise FallbackPolicy, public predicates.

    Strict required-token match. Target ≥ 90%."""
    hits = 0
    misses: list[dict] = []
    for prompt, required_tokens in V4_SURFACES:
        resp = _chat(tokenizer, model, prompt)
        if all(tok in resp for tok in required_tokens):
            hits += 1
        else:
            missing = [tok for tok in required_tokens if tok not in resp]
            misses.append({"prompt": prompt, "missing": missing, "response": resp[:200]})
    total = len(V4_SURFACES)
    return {"accuracy": hits / total, "hits": hits, "total": total, "misses": misses}


def main() -> None:
    tokenizer, model = _load_model()
    tool_bench = _bench_tool_calls(tokenizer, model)
    schema_bench = _bench_block_schemas(tokenizer, model)
    workflow_bench = _bench_workflows(tokenizer, model)
    v39_new_tools_bench = _bench_v39_new_tools(tokenizer, model)
    v39_xform_bench = _bench_v39_transform_hash(tokenizer, model)
    v39_transport_bench = _bench_v39_transport_guard(tokenizer, model)
    v311_new_tools_bench = _bench_v311_new_tools(tokenizer, model)
    v311_explain_bench = _bench_v311_explain_field(tokenizer, model)
    v312_quality_gate_bench = _bench_v312_quality_gate_strict_mode(tokenizer, model)
    v312_lineage_staleness_bench = _bench_v312_lineage_staleness(tokenizer, model)
    v4_surfaces_bench = _bench_v4_surfaces(tokenizer, model)

    report = {
        "tool_call": tool_bench,
        "block_schema": schema_bench,
        "workflow": workflow_bench,
        "v39_new_tools": v39_new_tools_bench,
        "v39_transform_hash": v39_xform_bench,
        "v39_transport_guard": v39_transport_bench,
        "v311_new_tools": v311_new_tools_bench,
        "v311_explain_field": v311_explain_bench,
        "v312_quality_gate_strict_mode": v312_quality_gate_bench,
        "v312_lineage_staleness": v312_lineage_staleness_bench,
        "v4_surfaces": v4_surfaces_bench,
        "targets": {
            "tool_call": 0.95,
            "block_schema": 0.98,
            "workflow": 0.90,
            "v39_new_tools": 0.90,
            "v39_transform_hash": 0.95,
            "v39_transport_guard": 0.95,
            "v311_new_tools": 0.90,
            "v311_explain_field": 0.95,
            "v312_quality_gate_strict_mode": 0.90,
            "v312_lineage_staleness": 0.90,
            "v4_surfaces": 0.90,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 60)
    print("mind-mem-4b v3.12.0 eval report")
    print("=" * 60)
    for name, bench, target in (
        ("tool_call                     ", tool_bench, 0.95),
        ("block_schema                  ", schema_bench, 0.98),
        ("workflow                      ", workflow_bench, 0.90),
        ("v39_new_tools                 ", v39_new_tools_bench, 0.90),
        ("v39_transform_hash            ", v39_xform_bench, 0.95),
        ("v39_transport_guard           ", v39_transport_bench, 0.95),
        ("v311_new_tools                ", v311_new_tools_bench, 0.90),
        ("v311_explain_field            ", v311_explain_bench, 0.95),
        ("v312_quality_gate_strict_mode ", v312_quality_gate_bench, 0.90),
        ("v312_lineage_staleness        ", v312_lineage_staleness_bench, 0.90),
        ("v4_surfaces                   ", v4_surfaces_bench, 0.90),
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
        and v311_new_tools_bench["accuracy"] >= 0.90
        and v311_explain_bench["accuracy"] >= 0.95
        and v312_quality_gate_bench["accuracy"] >= 0.90
        and v312_lineage_staleness_bench["accuracy"] >= 0.90
        and v4_surfaces_bench["accuracy"] >= 0.90
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
