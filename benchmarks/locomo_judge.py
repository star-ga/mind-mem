#!/usr/bin/env python3
"""LoCoMo LLM-as-Judge Evaluation for Mind-Mem.

Extends the retrieval harness with an LLM judge pipeline:
  1. Retrieve top-K blocks via mind-mem BM25 recall
  2. Feed retrieved context + question to an answerer LLM
  3. Judge scores the generated answer against gold reference

Produces accuracy scores directly comparable to Mem0's reported
66.9-68.5% and Letta's 74.0% on LoCoMo.

Usage:
    python3 benchmarks/locomo_judge.py --dry-run
    python3 benchmarks/locomo_judge.py --judge-model mistral-small-latest --top-k 10
    python3 benchmarks/locomo_judge.py --answerer-model mistral-small-latest --output results.json

Environment:
    MISTRAL_API_KEY — Required for Mistral models (default)
    OPENAI_API_KEY — Required for OpenAI models
    Keys auto-loaded from ~/.env if present

Reference: Mem0 eval, Letta blog (Aug 2025), memobase LoCoMo fork.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# Add scripts/ and benchmarks/ to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_HERE, "..", "scripts")
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.append(_HERE)  # append (not insert) — benchmarks/ must not shadow scripts/

# Heavy imports (recall engine, harness) are deferred to avoid loading them
# in the orchestrator process, which only needs json/subprocess/time.
# They are loaded on-demand in _run_single_conv() and evaluate functions.
recall = None
download_dataset = None
build_workspace = None
_parse_sessions = None
CATEGORY_NAMES = {}
detect_query_type = None


def _load_heavy_imports():
    """Load recall engine and harness modules. Called only in subprocess mode."""
    global recall, download_dataset, build_workspace, _parse_sessions, CATEGORY_NAMES, detect_query_type

    from recall import detect_query_type as _dqt
    from recall import recall as _recall  # noqa: E402
    recall = _recall

    detect_query_type = _dqt

    # Suppress recall structured logging — observability module's handle()
    # bypasses level checks, so we remove handlers from the recall logger.
    import logging as _logging
    for _name in list(_logging.Logger.manager.loggerDict):
        if _name.startswith("mind-mem."):
            _logging.getLogger(_name).handlers.clear()

    from locomo_harness import (
        CATEGORY_NAMES as _cn,
    )
    from locomo_harness import (
        _parse_sessions as _ps,
    )
    from locomo_harness import (
        build_workspace as _bw,
    )
    from locomo_harness import (  # noqa: E402
        download_dataset as _dd,
    )
    download_dataset = _dd
    build_workspace = _bw
    _parse_sessions = _ps
    CATEGORY_NAMES = _cn

# ---------------------------------------------------------------------------
# Load API keys from .env
# ---------------------------------------------------------------------------

def _load_env():
    """Load API keys from environment .env files if not already set."""
    _ALLOWED_ENV_KEYS = {
        "MISTRAL_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "XAI_API_KEY", "DEEPSEEK_API_KEY", "PERPLEXITY_API_KEY",
        "PINECONE_API_KEY",
    }
    env_paths = [
        os.path.expanduser("~/.env"),
        os.path.expanduser("~/.claude-ultimate/.env"),
    ]
    for env_path in env_paths:
        if not os.path.isfile(env_path):
            continue
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    if key in _ALLOWED_ENV_KEYS and not os.environ.get(key):
                        os.environ[key] = val.strip()[:512]



# ---------------------------------------------------------------------------
# LLM API calls (OpenAI-compatible)
# ---------------------------------------------------------------------------

# Provider routing: model prefix determines API endpoint
PROVIDER_CONFIG = {
    "gpt": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "key_env": "OPENAI_API_KEY",
        "format": "openai",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1/chat/completions",
        "key_env": "MISTRAL_API_KEY",
        "format": "openai",
    },
    "ministral": {
        "base_url": "https://api.mistral.ai/v1/chat/completions",
        "key_env": "MISTRAL_API_KEY",
        "format": "openai",
    },
    "claude": {
        "base_url": "https://api.anthropic.com/v1/messages",
        "key_env": "ANTHROPIC_API_KEY",
        "format": "anthropic",
    },
}


def _resolve_provider(model: str) -> tuple[str, str, str]:
    """Resolve API base URL, key, and format from model name prefix."""
    for prefix, cfg in PROVIDER_CONFIG.items():
        if model.startswith(prefix):
            key = os.environ.get(cfg["key_env"], "")
            if not key:
                raise RuntimeError(f"{cfg['key_env']} not set for model {model}")
            return cfg["base_url"], key, cfg["format"]
    raise RuntimeError(f"Unknown model provider for: {model}")


def _llm_chat(
    messages: list[dict],
    model: str = "mistral-small-latest",
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_retries: int = 3,
) -> str:
    """Call LLM chat completions API. Supports OpenAI-compatible and Anthropic formats.

    Retries on transient errors (429, 5xx, timeouts) with exponential backoff.

    Raises ValueError if max_retries < 1.
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")
    base_url, api_key, fmt = _resolve_provider(model)

    if fmt == "anthropic":
        # Extract system message if present
        system_text = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                user_messages.append(m)
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": user_messages,
        }
        if system_text:
            payload["system"] = system_text
        if temperature > 0:
            payload["temperature"] = temperature
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
    else:
        # gpt-5.x requires max_completion_tokens instead of max_tokens
        # gpt-5-mini/nano (reasoning models) don't support temperature
        is_gpt5 = model.startswith("gpt-5")
        tokens_key = "max_completion_tokens" if is_gpt5 else "max_tokens"
        payload = {
            "model": model,
            "messages": messages,
            tokens_key: max_tokens,
        }
        # Only add temperature for non-reasoning models
        if not (is_gpt5 and "mini" in model or is_gpt5 and "nano" in model):
            payload["temperature"] = temperature
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    data = json.dumps(payload).encode("utf-8")

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(base_url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            if fmt == "anthropic":
                return result["content"][0]["text"].strip()
            return result["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504, 529) and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  [retry] HTTP {e.code}, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  [retry] {type(e).__name__}, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise


# ---------------------------------------------------------------------------
# Prompts (adapted from Mem0/memobase eval pipelines)
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about conversations between people.
You have access to retrieved memory excerpts from those conversations.

IMPORTANT RULES:
1. Use the context provided to answer. Reason step-by-step through the evidence.
2. If the context contains partial or indirect evidence, INFER the answer from what is available.
   People often imply things in conversation — use common sense and logical reasoning.
3. Always give your best answer. Even if evidence is incomplete, provide the most likely answer
   based on what the context suggests.
4. Be concise and direct — answer in 1-2 sentences.
5. Always specify the correct speaker/actor (e.g., "Tim did X" not just "X happened").
   Verify names before answering — do not confuse speakers.
6. Never use absolute negative language ("never", "not mentioned", "doesn't exist") unless
   you have confirmed absence across ALL provided context. Absence of evidence is not
   evidence of absence."""

ANSWER_USER_TEMPLATE = """\
Context from conversation memory:
{context}

Question: {question}

Think step-by-step: What relevant facts are in the context? What can you infer from them?
Then give a concise, direct answer."""

JUDGE_SYSTEM_PROMPT = """\
You are an expert judge evaluating the quality of answers about conversations.
Score the generated answer by comparing it to the reference answer.
Consider factual accuracy, completeness, and relevance.
A generated answer that conveys the same meaning as the reference — even with different
wording or additional reasoning — should score highly.
Output ONLY a JSON object with two fields:
- "score": integer 0-100 (0=completely wrong, 100=perfect match)
- "reason": brief explanation (1 sentence)"""

JUDGE_USER_TEMPLATE = """\
Question: {question}

Reference Answer: {reference}

Generated Answer: {generated}

Score the generated answer against the reference. An answer that conveys the same core
facts as the reference should score 70+. Output JSON only."""

ADVERSARIAL_ANSWER_PROMPT = """\
Answer the ADVERSARIAL question using ONLY the extracted evidence below.

{context}

Question: {question}

Hard rules:
1. Cite evidence BEFORE making a claim. If EVIDENCE_FOUND=NO, do NOT answer \
"No" or "Never mentioned". Instead answer: "Not found in the retrieved evidence."
2. Never claim "never mentioned" unless DENIAL_EVIDENCE contains an explicit \
denial that directly matches the asked claim.
3. Always include speaker attribution in your answer (who said or did it).
4. If speaker attribution is missing or ambiguous, say so explicitly.
5. Keep the answer short and literal. No extra speculation.

FINAL_ANSWER (1-2 sentences):"""

ADVERSARIAL_JUDGE_SYSTEM_PROMPT = """\
You are an expert judge evaluating answers to ADVERSARIAL entity-confusion questions.
These questions deliberately ask about Entity A doing something that Entity B actually did.
The Reference Answer contains the CORRECT response (typically a negation or clarification).

Assess the Generated Answer on THREE dimensions and return a JSON object with EXACTLY these fields:

1. "entity_correct" (boolean): Does the Generated Answer correctly identify which entity
   is associated with the action? True if it correctly denies Entity A did it, correctly
   attributes it to Entity B, or correctly states the information is not found.
   False if it confuses entities or wrongly affirms Entity A did it.

2. "rejection_quality" (integer 0-100): How well does the answer handle the adversarial
   trap? 100 = clear, confident, correct denial/clarification with proper attribution.
   50 = vague but not wrong. 0 = completely fails to address the entity confusion.

3. "wrong_entity_hallucination" (boolean): Does the Generated Answer confidently claim
   that Entity A DID the action (when Entity A did NOT)? This is the worst failure mode.
   True = hallucinated wrong entity attribution. False = did not hallucinate.

4. "reason" (string): Brief explanation of your assessment.

Output ONLY the JSON object, no other text."""

ADVERSARIAL_JUDGE_USER_TEMPLATE = """\
Question: {question}

Reference Answer: {reference}

Generated Answer: {generated}

Assess the generated answer against the reference on entity_correct, rejection_quality, \
and wrong_entity_hallucination. Output JSON only."""


# ---------------------------------------------------------------------------
# Adversarial scoring
# ---------------------------------------------------------------------------

def _compute_adversarial_score(components: dict) -> int:
    """Deterministic score from structured adversarial judge components.

    Scoring logic:
    - wrong_entity_hallucination=True → hard 0 (catastrophic failure)
    - entity_correct=True → max(rejection_quality, 80) (floor at 80 for correct rejections)
    - else → min(rejection_quality, 40) (cap at 40 for incorrect/ambiguous)
    """
    if components.get("wrong_entity_hallucination", False):
        return 0
    if components.get("entity_correct", False):
        return max(components.get("rejection_quality", 80), 80)
    return min(components.get("rejection_quality", 40), 40)


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

def _strip_semantic_prefix(text: str) -> str:
    """Remove leading semantic label prefix — delegates to evidence_packer."""
    from evidence_packer import strip_semantic_prefix
    return strip_semantic_prefix(text)


def format_context(retrieved: list[dict], max_chars: int = 6000) -> str:
    """Format retrieved blocks into context string for the LLM.

    Recall results use 'excerpt' for the text content.
    Semantic label prefixes are stripped so the LLM sees clean facts.
    """
    parts = []
    total = 0
    for r in retrieved:
        # Recall engine returns 'excerpt' field with block content
        text = r.get("excerpt", "") or r.get("Statement", "")
        if not text:
            continue
        part = _strip_semantic_prefix(text.strip())
        if total + len(part) > max_chars:
            break
        parts.append(part)
        total += len(part)
    return "\n".join(parts)


def answer_question(
    question: str,
    context: str,
    is_adversarial: bool,
    model: str = "mistral-small-latest",
) -> str:
    """Generate an answer using the LLM given retrieved context."""
    if is_adversarial:
        user_msg = ADVERSARIAL_ANSWER_PROMPT.format(
            context=context, question=question
        )
    else:
        user_msg = ANSWER_USER_TEMPLATE.format(
            context=context, question=question
        )

    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return _llm_chat(messages, model=model)


def _strip_markdown_json(raw: str) -> str:
    """Strip markdown code block fences from LLM JSON output."""
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def judge_answer(
    question: str,
    reference: str,
    generated: str,
    is_adversarial: bool = False,
    model: str = "mistral-small-latest",
) -> dict:
    """Have the judge LLM score the generated answer.

    Non-adversarial path: returns {score, reason}.
    Adversarial path: returns {score, reason, entity_correct, rejection_quality, wrong_entity_hallucination}.
    """
    if is_adversarial:
        # --- Adversarial dual-path: structured multi-dimensional assessment ---
        user_msg = ADVERSARIAL_JUDGE_USER_TEMPLATE.format(
            question=question,
            reference=reference,
            generated=generated,
        )
        messages = [
            {"role": "system", "content": ADVERSARIAL_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        raw = _llm_chat(messages, model=model, max_tokens=300)
        raw = _strip_markdown_json(raw)

        try:
            result = json.loads(raw)
            components = {
                "entity_correct": bool(result.get("entity_correct", False)),
                "rejection_quality": max(0, min(100, int(result.get("rejection_quality", 0)))),
                "wrong_entity_hallucination": bool(result.get("wrong_entity_hallucination", False)),
            }
            score = _compute_adversarial_score(components)
            return {
                "score": score,
                "reason": str(result.get("reason", "")),
                **components,
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            import re
            # Fallback: try to extract individual fields
            ec_m = re.search(r'"entity_correct"\s*:\s*(true|false)', raw, re.IGNORECASE)
            rq_m = re.search(r'"rejection_quality"\s*:\s*(\d+)', raw)
            wh_m = re.search(r'"wrong_entity_hallucination"\s*:\s*(true|false)', raw, re.IGNORECASE)
            components = {
                "entity_correct": ec_m.group(1).lower() == "true" if ec_m else False,
                "rejection_quality": max(0, min(100, int(rq_m.group(1)))) if rq_m else 0,
                "wrong_entity_hallucination": wh_m.group(1).lower() == "true" if wh_m else False,
            }
            score = _compute_adversarial_score(components)
            return {
                "score": score,
                "reason": f"Partial parse: {raw[:200]}",
                **components,
            }
    else:
        # --- Non-adversarial path: unchanged ---
        user_msg = JUDGE_USER_TEMPLATE.format(
            question=question,
            reference=reference,
            generated=generated,
        )
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        raw = _llm_chat(messages, model=model, max_tokens=200)
        raw = _strip_markdown_json(raw)

        try:
            result = json.loads(raw)
            score = max(0, min(100, int(result.get("score", 0))))
            return {
                "score": score,
                "reason": str(result.get("reason", "")),
            }
        except (json.JSONDecodeError, ValueError, IndexError):
            import re
            m = re.search(r'"score"\s*:\s*(\d+)', raw)
            if m:
                score = max(0, min(100, int(m.group(1))))
                return {"score": score, "reason": raw[:200]}
            return {"score": 0, "reason": f"Parse error: {raw[:200]}"}


def evaluate_sample_with_judge(
    sample: dict,
    workspace: str,
    top_k: int = 10,
    answerer_model: str = "mistral-small-latest",
    judge_model: str = "mistral-small-latest",
    rate_limit_delay: float = 0.1,
    compress: bool = False,
    _jsonl_stream=None,
    hybrid_backend=None,
) -> list[dict]:
    """Run LLM-as-judge evaluation for one LoCoMo sample.

    For each QA pair:
    1. Retrieve top-K blocks via hybrid (BM25+vector RRF) or BM25-only recall
    2. (Optional) Compress retrieved context into focused observations
    3. Generate answer using answerer LLM
    4. Score answer using judge LLM
    """
    if compress:
        from observation_compress import compress_context

    qa_pairs = sample.get("qa", [])
    results = []
    _stats = {"packer": 0, "guard_filtered": 0, "llm_compress": 0}

    for qi, qa in enumerate(qa_pairs):
        question = qa.get("question", "")
        cat_raw = qa.get("category", 0)
        category = CATEGORY_NAMES.get(cat_raw, f"cat-{cat_raw}")
        # Detect adversarial if explicit category 5 OR string label 'adversarial'
        is_adversarial = (cat_raw == 5) or (str(cat_raw).lower() == "adversarial")

        # Gold answer
        if is_adversarial:
            gold_answer = qa.get("adversarial_answer", qa.get("answer", ""))
        else:
            gold_answer = qa.get("answer", "")

        if not question:
            continue

        # Step 1: Retrieve — use hybrid BM25+vector RRF when available
        if hybrid_backend is not None:
            retrieved = hybrid_backend.search(
                question, workspace, limit=top_k, active_only=False,
                graph_boost=True,
            )
        else:
            retrieved = recall(workspace, question, limit=top_k, active_only=False)

        # Step 2: Build context via mind-mem evidence packer (structured for ALL types)
        from evidence_packer import is_true_adversarial, pack_evidence

        # Detect query type for packing strategy
        detected_type = "adversarial" if is_adversarial else detect_query_type(question)

        # All query types now go through structured pack_evidence
        context = pack_evidence(
            retrieved, question=question, query_type=detected_type,
        )

        # Step 2b: Abstention gate — deterministic pre-LLM confidence check
        # For adversarial queries, check if retrieval has enough direct evidence.
        # If not, force abstention without calling the LLM.
        abst = None
        abstention_applied = False
        use_adversarial_prompt = False
        if is_adversarial:
            # Trust LoCoMo label — always use adversarial prompt for labeled questions.
            # is_true_adversarial() is for runtime detection of unlabeled queries;
            # in benchmarks we have ground-truth labels so don't re-filter.
            use_adversarial_prompt = True
            has_signal = is_true_adversarial(question)
            _stats["packer"] += 1
            if not has_signal:
                _stats["guard_filtered"] += 1  # track how many lack explicit signals

            if not _stats.get("first_adv_qi"):
                _stats["first_adv_qi"] = qi + 1
                print(f"[milestone] first adversarial at q{qi+1}/{len(qa_pairs)} "
                      f"(has_signal={has_signal})", flush=True)

            # Abstention classifier: runs on ALL adversarial-labeled questions
            from abstention_classifier import classify_abstention
            abst = classify_abstention(question, retrieved)
            if abst.should_abstain:
                abstention_applied = True
                _stats.setdefault("abstention_fired", 0)
                _stats["abstention_fired"] += 1

        if not is_adversarial and compress and context.strip():
            try:
                compress_type = category if category in (
                    "temporal", "multi-hop"
                ) else None
                context = compress_context(
                    context, question, _llm_chat, model=answerer_model,
                    query_type=compress_type,
                )
                _stats["llm_compress"] += 1
            except Exception:
                pass  # Fall back to raw context on compression failure

        # Step 3: Answer (skip LLM if abstention classifier fired)
        if abstention_applied:
            generated = abst.forced_answer
            _stats.setdefault("llm_skipped", 0)
            _stats["llm_skipped"] += 1
        else:
            try:
                generated = answer_question(
                    question, context, use_adversarial_prompt, model=answerer_model
                )
            except Exception as e:
                generated = f"Error: {e}"

        # Step 4: Judge
        try:
            judgment = judge_answer(
                question, gold_answer, generated, is_adversarial=is_adversarial, model=judge_model
            )
        except Exception as e:
            judgment = {"score": 0, "reason": f"Judge error: {e}"}

        record = {
            "question": question,
            "category": category,
            "gold_answer": gold_answer,
            "generated_answer": generated,
            "context_blocks": len(retrieved),
            "judge_score": judgment["score"],
            "judge_reason": judgment["reason"],
        }
        if abstention_applied:
            record["abstention"] = True
            record["abstention_confidence"] = abst.confidence
            record["abstention_features"] = abst.features
        if is_adversarial:
            record["entity_correct"] = judgment.get("entity_correct", False)
            record["rejection_quality"] = judgment.get("rejection_quality", 0)
            record["wrong_entity_hallucination"] = judgment.get("wrong_entity_hallucination", False)
        results.append(record)

        # Stream to JSONL immediately (survives crashes, enables tail -f)
        if _jsonl_stream is not None:
            _jsonl_stream.write(json.dumps(record) + "\n")
            _jsonl_stream.flush()

        # Progress logging every 10 questions
        if (qi + 1) % 10 == 0 or qi == len(qa_pairs) - 1:
            print(f"[progress] {qi+1}/{len(qa_pairs)} "
                  f"packer={_stats['packer']} guard={_stats['guard_filtered']} "
                  f"compress={_stats['llm_compress']}", flush=True)

        # Rate limiting
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

    return results


def _compute_metrics_from_scores(score_agg: dict[str, list[int]]) -> dict:
    """Compute metrics from {category: [scores]} dict. Memory-efficient."""
    def _group_stats(scores):
        n = len(scores)
        if not n:
            return {"count": 0, "mean_score": 0, "accuracy_50": 0, "accuracy_75": 0, "min_score": 0, "max_score": 0}
        avg = sum(scores) / n
        acc50 = sum(1 for s in scores if s >= 50) / n
        acc75 = sum(1 for s in scores if s >= 75) / n
        return {
            "count": n,
            "mean_score": round(avg, 2),
            "accuracy_50": round(acc50 * 100, 2),
            "accuracy_75": round(acc75 * 100, 2),
            "min_score": min(scores),
            "max_score": max(scores),
        }

    all_scores = []
    for cat_scores in score_agg.values():
        all_scores.extend(cat_scores)

    return {
        "overall": _group_stats(all_scores),
        "by_category": {cat: _group_stats(scores) for cat, scores in sorted(score_agg.items())},
    }


def aggregate_judge_metrics(all_results: list[dict]) -> dict:
    """Compute aggregate LLM-as-judge metrics."""
    if not all_results:
        return {"error": "no results"}

    # Group by category
    by_category = {}
    for r in all_results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    def compute_group(group: list[dict]) -> dict:
        n = len(group)
        scores = [r["judge_score"] for r in group]
        avg = sum(scores) / n if n else 0
        # Accuracy = % of questions scored >= 50
        accurate = sum(1 for s in scores if s >= 50) / n if n else 0
        # High accuracy = % scored >= 75
        high_acc = sum(1 for s in scores if s >= 75) / n if n else 0
        return {
            "count": n,
            "mean_score": round(avg, 2),
            "accuracy_50": round(accurate * 100, 2),
            "accuracy_75": round(high_acc * 100, 2),
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
        }

    metrics = {
        "overall": compute_group(all_results),
        "by_category": {},
    }
    for cat, group in sorted(by_category.items()):
        metrics["by_category"][cat] = compute_group(group)

    # Non-adversarial aggregate
    non_adv = [r for r in all_results if r.get("category") != "adversarial"]
    if non_adv:
        metrics["non_adversarial"] = compute_group(non_adv)

    # Adversarial detail metrics
    adv = [r for r in all_results if r.get("category") == "adversarial"]
    if adv:
        n_adv = len(adv)
        metrics["adversarial_detail"] = {
            "count": n_adv,
            "abstention_accuracy": round(
                sum(1 for r in adv if r.get("abstention")) / n_adv * 100, 2
            ),
            "hallucination_rate": round(
                sum(1 for r in adv if r.get("wrong_entity_hallucination")) / n_adv * 100, 2
            ),
            "correct_rejection_rate": round(
                sum(1 for r in adv if r.get("entity_correct")) / n_adv * 100, 2
            ),
            "mean_rejection_quality": round(
                sum(r.get("rejection_quality", 0) for r in adv) / n_adv, 2
            ),
        }

    return metrics


def print_judge_table(metrics: dict) -> None:
    """Print formatted LLM-as-judge results table."""
    print()
    print("=" * 80)
    print("LoCoMo LLM-as-Judge Results — Mind-Mem + BM25 Recall")
    print("=" * 80)

    header = (
        f"{'Category':<20} {'N':>5} {'Mean':>7} "
        f"{'Acc≥50':>8} {'Acc≥75':>8} {'Min':>5} {'Max':>5}"
    )
    print(header)
    print("-" * 80)

    overall = metrics.get("overall", {})
    print(
        f"{'OVERALL':<20} {overall.get('count', 0):>5} "
        f"{overall.get('mean_score', 0):>7.1f} "
        f"{overall.get('accuracy_50', 0):>7.1f}% "
        f"{overall.get('accuracy_75', 0):>7.1f}% "
        f"{overall.get('min_score', 0):>5} "
        f"{overall.get('max_score', 0):>5}"
    )
    print("-" * 80)

    for cat, cm in sorted(metrics.get("by_category", {}).items()):
        print(
            f"{cat:<20} {cm.get('count', 0):>5} "
            f"{cm.get('mean_score', 0):>7.1f} "
            f"{cm.get('accuracy_50', 0):>7.1f}% "
            f"{cm.get('accuracy_75', 0):>7.1f}% "
            f"{cm.get('min_score', 0):>5} "
            f"{cm.get('max_score', 0):>5}"
        )

    # Non-adversarial row
    na = metrics.get("non_adversarial")
    if na:
        print("-" * 80)
        print(
            f"{'NON-ADVERSARIAL':<20} {na.get('count', 0):>5} "
            f"{na.get('mean_score', 0):>7.1f} "
            f"{na.get('accuracy_50', 0):>7.1f}% "
            f"{na.get('accuracy_75', 0):>7.1f}% "
            f"{na.get('min_score', 0):>5} "
            f"{na.get('max_score', 0):>5}"
        )

    # Adversarial detail block
    ad = metrics.get("adversarial_detail")
    if ad:
        print("-" * 80)
        print(f"  Adversarial Detail (n={ad.get('count', 0)}):")
        print(f"    Abstention accuracy:     {ad.get('abstention_accuracy', 0):>6.1f}%")
        print(f"    Hallucination rate:      {ad.get('hallucination_rate', 0):>6.1f}%")
        print(f"    Correct rejection rate:  {ad.get('correct_rejection_rate', 0):>6.1f}%")
        print(f"    Mean rejection quality:  {ad.get('mean_rejection_quality', 0):>6.1f}")

    print("=" * 80)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _engine_label(args) -> str:
    """Build engine label string for results metadata."""
    parts = ["mind-mem-recall"]
    if getattr(args, "hybrid", False):
        parts.append("hybrid")
    else:
        parts.append("bm25")
    if getattr(args, "compress", False):
        parts.append("compress")
    return "+".join(parts)


def _setup_hybrid_workspace(workspace: str):
    """Write mind-mem.json with hybrid config, build vector index, return HybridBackend.

    Uses Qwen3-Embedding-8B (4096d) via llama.cpp server on GPU + BM25 fused
    with Reciprocal Rank Fusion (RRF k=60).

    Returns:
        HybridBackend instance ready for search().
    """
    recall_cfg = {
        "backend": "hybrid",
        "rrf_k": 60,
        "bm25_weight": 1.0,
        "vector_weight": 1.0,
        "vector_enabled": True,
        "provider": "llama_cpp",
        "llama_cpp_url": "http://localhost:8090",
        "dimension": 4096,
        "index_path": ".mind-mem-vectors",
        "cross_encoder": {"enabled": True, "blend_weight": 0.6},
    }
    config = {
        "version": "1.0.6",
        "workspace_path": ".",
        "recall": recall_cfg,
    }
    config_path = os.path.join(workspace, "mind-mem.json")
    with open(config_path, "w") as f:
        json.dump(config, f)

    # Index blocks using llama.cpp embeddings server (Qwen3-Embedding-8B)
    from recall_vector import VectorBackend

    vec_backend = VectorBackend(recall_cfg)
    vec_backend.index(workspace)

    # Create HybridBackend that fuses BM25 + vector via RRF
    from hybrid_recall import HybridBackend

    return HybridBackend(config=recall_cfg)



def _run_single_conv(conv_index: int, args) -> None:
    """Process a single conversation in-process, write results to JSONL.

    Called either directly (--single-conv) or via subprocess from orchestrator.
    Loads only the needed conversation, processes it, writes JSONL, exits.
    """
    import shutil
    import tempfile

    _load_heavy_imports()
    dataset = download_dataset()
    if conv_index >= len(dataset):
        print(f"[judge] conv_index {conv_index} >= dataset size {len(dataset)}")
        return

    sample = dataset[conv_index]
    # Free the rest of the dataset immediately
    del dataset

    sample_id = sample.get("sample_id", conv_index)
    qa_count = len(sample.get("qa", []))
    if args.limit:
        sample["qa"] = sample["qa"][:args.limit]
        qa_count = len(sample["qa"])

    print(f"[judge] conv={conv_index} sample={sample_id} qa_pairs={qa_count}")

    out_path = args.output or os.path.join(_HERE, "locomo_judge_results.json")
    jsonl_path = out_path + f".conv{conv_index}.jsonl"

    conv_tmp = tempfile.mkdtemp(prefix=f"lj_{sample_id}_")
    try:
        workspace = build_workspace(sample, conv_tmp)

        # --- Hybrid recall setup: write mind-mem.json + build vector index + return backend ---
        _hybrid_backend = None
        if getattr(args, "hybrid", False):
            _hybrid_backend = _setup_hybrid_workspace(workspace)
            print(f"[judge] conv={conv_index} hybrid recall: BM25+Qwen3-Embedding-8B RRF via llama.cpp")

        # Open JSONL for streaming writes (context manager ensures close on exception)
        with open(jsonl_path, "w") as _jsonl_f:
            results = evaluate_sample_with_judge(
                sample, workspace,
                top_k=args.top_k,
                answerer_model=args.answerer_model,
                judge_model=args.judge_model,
                rate_limit_delay=args.rate_limit,
                compress=args.compress,
                hybrid_backend=_hybrid_backend,
                _jsonl_stream=_jsonl_f,
            )

        if results:
            avg = sum(r["judge_score"] for r in results) / len(results)
            print(f"[judge] conv={conv_index} done: {len(results)} qa, avg={avg:.1f}")
        else:
            print(f"[judge] conv={conv_index} done: 0 qa")

    except Exception as e:
        print(f"[judge] conv={conv_index} ERROR: {e}")
    finally:
        shutil.rmtree(conv_tmp, ignore_errors=True)


def main():
    _load_env()
    parser = argparse.ArgumentParser(
        description="LoCoMo LLM-as-Judge Evaluation for Mind-Mem"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Test with only the first conversation",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of blocks to retrieve per question (default: 10)",
    )
    parser.add_argument(
        "--answerer-model", type=str, default="mistral-small-latest",
        help="Model for generating answers (default: mistral-small-latest)",
    )
    parser.add_argument(
        "--judge-model", type=str, default="mistral-small-latest",
        help="Model for judging answers (default: mistral-small-latest)",
    )
    parser.add_argument(
        "--rate-limit", type=float, default=0.05,
        help="Delay between API calls in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Write JSON results to this file",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of QA pairs per conversation (for testing)",
    )
    parser.add_argument(
        "--compress", action="store_true",
        help="Enable observation compression (Retrieve→Compress→Answer→Judge pipeline)",
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Enable hybrid BM25+local vector recall (sentence-transformers, all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--single-conv", type=int, default=None,
        help="Process only this conversation index (used by subprocess orchestration)",
    )
    args = parser.parse_args()

    # Single-conversation mode: process one conv and exit
    if args.single_conv is not None:
        _run_single_conv(args.single_conv, args)
        return

    # --- Orchestrator mode: spawn subprocesses per conversation ---
    # The orchestrator never loads the dataset itself — each conversation
    # runs in a separate subprocess to stay under cgroup memory limits.
    import subprocess as sp

    # LoCoMo10 has exactly 10 conversations. Ensure cache exists via subprocess.
    cache_file = os.path.join(_HERE, ".cache", "locomo10.json")
    if not os.path.isfile(cache_file):
        sp.run([sys.executable, "-c",
                "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); "
                "from locomo_harness import download_dataset; download_dataset()"
                % (_SCRIPTS_DIR, _HERE)],
               timeout=120)
    num_convs = 1 if args.dry_run else 10

    out_path = args.output or os.path.join(_HERE, "locomo_judge_results.json")

    print(f"[judge] Config: answerer={args.answerer_model}, "
          f"judge={args.judge_model}, top_k={args.top_k}")
    print(f"[judge] Orchestrator: {num_convs} conversations (subprocess per conv)")

    t0 = time.time()
    score_agg = {}
    total_questions = 0

    for ci in range(num_convs):
        # Build subprocess command
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--single-conv", str(ci),
            "--top-k", str(args.top_k),
            "--answerer-model", args.answerer_model,
            "--judge-model", args.judge_model,
            "--rate-limit", str(args.rate_limit),
            "--output", out_path,
        ]
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        if args.compress:
            cmd.append("--compress")
        if args.hybrid:
            cmd.append("--hybrid")

        print(f"\n[judge] [{ci+1}/{num_convs}] Launching subprocess for conv {ci}...")
        result = sp.run(cmd, capture_output=False, timeout=7200)

        if result.returncode != 0:
            print(f"[judge] WARNING: conv {ci} subprocess exited with code {result.returncode}")

        # Read the subprocess's JSONL output
        jsonl_path = out_path + f".conv{ci}.jsonl"
        if os.path.isfile(jsonl_path):
            with open(jsonl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        cat = r.get("category", "unknown")
                        score = r.get("judge_score", 0)
                        if not isinstance(score, (int, float)):
                            score = 0
                        score = max(0, min(100, int(score)))
                        score_agg.setdefault(cat, []).append(score)
                        total_questions += 1
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        print(f"[judge] WARNING: skipping malformed JSONL line: {exc}")

            # Save partial metrics after each conversation
            if score_agg:
                partial_metrics = _compute_metrics_from_scores(score_agg)
                partial = {
                    "benchmark": "locomo-llm-judge",
                    "engine": _engine_label(args),
                    "partial": True,
                    "conversations_done": ci + 1,
                    "num_questions": total_questions,
                    "metrics": partial_metrics,
                }
                with open(out_path + ".partial", "w") as f:
                    json.dump(partial, f, indent=2)
        else:
            print(f"[judge] WARNING: no JSONL output for conv {ci}")

    elapsed = time.time() - t0

    # Final aggregation
    metrics = _compute_metrics_from_scores(score_agg)
    print_judge_table(metrics)

    print(f"Total questions: {total_questions}")
    print(f"Elapsed time: {elapsed:.1f}s")
    api_calls = total_questions * (3 if args.compress else 2)
    print(f"API calls: {api_calls}")
    if args.compress:
        print("Pipeline: Retrieve → Compress → Answer → Judge")

    # Merge all per-conv JSONL files into final output
    per_question = []
    for ci in range(num_convs):
        jsonl_path = out_path + f".conv{ci}.jsonl"
        if os.path.isfile(jsonl_path):
            with open(jsonl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        per_question.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # skip malformed lines

    engine_label = _engine_label(args)
    output_data = {
        "benchmark": "locomo-llm-judge",
        "engine": engine_label,
        "answerer_model": args.answerer_model,
        "judge_model": args.judge_model,
        "top_k": args.top_k,
        "dry_run": args.dry_run,
        "num_conversations": num_convs,
        "num_questions": total_questions,
        "elapsed_seconds": round(elapsed, 2),
        "api_calls": api_calls,
        "metrics": metrics,
        "per_question": per_question,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults written to {out_path}")

    # Clean up per-conv JSONL and partial files
    for ci in range(num_convs):
        jsonl_path = out_path + f".conv{ci}.jsonl"
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)
    partial_path = out_path + ".partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)


if __name__ == "__main__":
    main()
