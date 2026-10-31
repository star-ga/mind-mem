"""Dataset adapters feeding the Python serving path without evaluation labels.

Copyright (c) STARGA Inc. All rights reserved.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import tempfile
from datetime import date
from pathlib import Path

from benchmarks.release_judge import judge, request
from benchmarks.release_scoring import aggregate, sha256


def require_pin(model: str | None, revision: str | None) -> None:
    if not model or not revision or any(word in (model + " " + revision).lower() for word in ("pending", "latest")):
        raise ValueError("An immutable published model id and revision are required; release placeholders cannot run")
    if revision in {"main", "master"}:
        raise ValueError("A branch name is not an immutable model revision")


def day(value: str) -> str:
    match = re.match(r"(\d{4})/(\d{2})/(\d{2})", value or "")
    return "-".join(match.groups()) if match else ""


def seed_turns(workspace: str, sessions: list, dates: list) -> None:
    from mind_mem.governance_gate import get_gate
    from mind_mem.storage import get_block_store

    units, counts = [], {}
    for i, session in enumerate(sessions):
        stamp = day(dates[i] if i < len(dates) else "") or "2024-01-01"
        for turn in session:
            content = turn.get("content", "")
            if not content.strip():
                continue
            counts[stamp] = counts.get(stamp, 0) + 1
            text = " / ".join(p.strip() for p in f"{turn.get('role', '')}: {content}".splitlines() if p.strip())
            units.append(
                {
                    "_id": f"D-{stamp.replace('-', '')}-{counts[stamp]:03d}",
                    "Statement": text.replace("[", "(").replace("]", ")"),
                    "Date": stamp,
                    "Status": "active",
                }
            )
    store = get_block_store(workspace)
    with get_gate(workspace).admit_proposal(
        "P-public-evaluation", "\n".join(u["_id"] for u in units), actor="bench_eval_adapter", target_file="decisions/DECISIONS.md"
    ):
        for unit in units:
            store.write_block(unit)


def prepare_workspace(workspace: str, recall_config: dict):
    """Build one index per corpus, with no question or evaluation metadata."""
    from mind_mem.hybrid_recall import HybridBackend
    from mind_mem.recall_vector import VectorBackend
    from mind_mem.sqlite_index import build_index

    Path(workspace, "mind-mem.json").write_text(json.dumps({"recall": recall_config}), encoding="utf-8")
    build_index(workspace, incremental=False)
    vector = VectorBackend(recall_config)
    vector.index(workspace)
    if vector.dimension is None:
        raise ValueError("Dense indexing failed; refusing a silently degraded retrieval run")
    return HybridBackend(recall_config)


def runtime_answer(question: str, workspace: str, recall_config: dict, config: dict, generate, backend, scoring_date=None) -> dict:
    """Only question, corpus and serving settings cross the answering boundary."""
    from mind_mem.chat_memory import chat_with_memory
    from mind_mem.hybrid_recall import resolve_rerank_depth
    from mind_mem.recall import serving_scope

    captured = {}

    def recall_fn(_workspace, _question, limit):
        with serving_scope():
            hits = backend.search(
                question,
                workspace,
                limit=limit,
                active_only=False,
                scoring_instant=scoring_date,
                rerank_depth=resolve_rerank_depth(recall_config, limit),
            )
        from mind_mem.recall_attestation import derive_legs

        ran, degraded = derive_legs(hits, vector_requested=True, vector_available=bool(backend.vector_available))
        if "vector" not in ran or degraded:
            raise ValueError("Dense retrieval did not serve this question")
        captured["retrieval"] = {"legs_ran": list(ran), "legs_degraded": list(degraded), "hit_ids": [h.get("_id") for h in hits]}
        return hits

    def generator(req):
        messages = [{"role": "system", "content": config["system_prompt"]}, {"role": "user", "content": req.prompt}]
        answer, count = generate(messages)
        captured.update(messages=messages, generated=answer, prompt_tokens=count)
        return answer

    asked = f"Today is {scoring_date.isoformat()}. {question}" if scoring_date else question
    response = chat_with_memory(
        workspace,
        asked,
        generator=generator,
        recall_fn=recall_fn,
        limit=config["limit"],
        max_evidence_chars=config["max_evidence_chars"],
        computed_facts=True,
        on_invalid="reject",
    )
    captured.setdefault("generated", response.answer)
    captured.setdefault("messages", [])
    captured.setdefault("prompt_tokens", 0)
    captured["product_answer"] = response.answer
    return captured


def run(args, config: dict, dataset: Path) -> dict:
    profile = config["configurations"][args.configuration]
    model_id = args.answer_model_id or profile["answer_model_id"]
    revision = args.answer_revision or profile["answer_revision"]
    judge_id = args.judge_model_id or config["judge"]["model_id"]
    judge_revision = args.judge_revision or config["judge"]["revision"]
    require_pin(model_id, revision)
    require_pin(judge_id, judge_revision)
    if judge_id != judge_revision:
        raise ValueError("The judge revision must be the exact immutable serving model id")
    if not args.judge_url or not args.out:
        raise ValueError("Fresh evaluation requires --judge-url and --out")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Answer revision must be an immutable 40-character repository commit")
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed = config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=False, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    def generate(messages):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        count = encoded["input_ids"].shape[-1]
        if count > config["max_input_tokens"]:
            raise ValueError("Prompt exceeds common token budget; refusing silent truncation")
        with torch.inference_mode():
            output = model.generate(**encoded, do_sample=False, max_new_tokens=config["max_new_tokens"])
        return tokenizer.decode(output[0][count:], skip_special_tokens=True), count

    recall = copy.deepcopy(config["retrieval"])
    encoder = config["encoder"]
    require_pin(encoder["model_id"], encoder["revision"])
    recall["model"] = snapshot_download(encoder["model_id"], revision=encoder["revision"])
    templates_path = args.bundle / "judge_prompts.json"
    templates = json.loads(templates_path.read_text(encoding="utf-8"))
    split = json.loads((args.bundle / "HELDOUT_SPLIT_20261029.json").read_text(encoding="utf-8"))
    data = json.loads(dataset.read_text(encoding="utf-8"))
    count = sum(len(s["qa"]) for s in data) if args.benchmark == "locomo" else len(data)
    expected = 1986 if args.benchmark == "locomo" else 500
    if count != expected:
        raise ValueError("Dataset population does not match the full protocol")
    rows = []
    raw_path = args.out.with_suffix(".answers.jsonl")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        raise ValueError("Raw output exists; choose a fresh output path")

    def call(messages, n):
        return request(messages, judge_id, args.judge_url, os.environ.get("BENCH_JUDGE_API_KEY", ""), n)

    with raw_path.open("w", encoding="utf-8") as output:
        for index, sample in enumerate(data):
            with tempfile.TemporaryDirectory(prefix="public-eval-") as temp:
                if args.benchmark == "locomo":
                    from benchmarks.locomo_harness import build_workspace

                    workspace = build_workspace({"sample_id": sample["sample_id"], "conversation": sample["conversation"]}, temp)
                    questions = sample["qa"]
                else:
                    workspace = temp
                    Path(workspace, "decisions").mkdir()
                    seed_turns(workspace, sample["haystack_sessions"], sample.get("haystack_dates", []))
                    questions = [sample]
                backend = prepare_workspace(workspace, recall)
                for offset, item in enumerate(questions):
                    stamp = day(item.get("question_date", ""))
                    answer = runtime_answer(
                        item["question"], workspace, recall, config, generate, backend, date.fromisoformat(stamp) if stamp else None
                    )
                    if args.benchmark == "locomo":
                        group, iid = sample["sample_id"], f"locomo-{index}-{offset}"
                        held = split["locomo_heldout_conversations"]
                    else:
                        group, iid = sample["question_id"], "lme-" + sample["question_id"]
                        held = split["lme_heldout_question_ids"]
                    row = {
                        "id": iid,
                        "benchmark": args.benchmark,
                        "group": group,
                        "partition": "held_out" if group in held else "dev",
                        "question": item["question"],
                        "reference": item.get("adversarial_answer", item.get("answer", "")),
                        "category": item.get("category"),
                        "abstention": group.endswith("_abs"),
                        **answer,
                    }
                    row["official_ok"], row["votes"] = judge(row, templates, call)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    rows.append(row)
    environment = {"python": platform.python_version()}
    for dependency in ("torch", "transformers", "huggingface-hub", "sentence-transformers", "sqlite-vec"):
        try:
            environment[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            environment[dependency] = "unavailable"
    environment["answer_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    code_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True, encoding="utf-8"
    ).strip()
    return {
        "code_revision": code_revision,
        "environment": environment,
        "status": "fresh run; protocol differs from archived category-conditioned prompts",
        "configuration": args.configuration,
        "answer_model_id": model_id,
        "answer_revision": revision,
        "judge_model_id": judge_id,
        "judge_revision": judge_revision,
        "temperature": 0,
        "seed": seed,
        "dataset_sha256": sha256(dataset),
        "config_sha256": sha256(args.config),
        "judge_prompt_sha256": sha256(templates_path),
        "raw_answers_sha256": sha256(raw_path),
        "official_key": aggregate(rows),
        "corrected_key": "verification pending; fresh answers require independent answer-bound adjudications",
    }
