"""Reconstruct licensed inputs from pinned downloads and text-free source references.

Copyright (c) STARGA Inc. All rights reserved.
"""

from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path

from benchmarks.release_scoring import read_rows, unique


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def source_blocks(sample, benchmark):
    blocks = {}
    if benchmark == "locomo":
        conversation = sample["conversation"]
        for key, turns in conversation.items():
            if not key.startswith("session_") or not isinstance(turns, list):
                continue
            stamp = conversation.get(key + "_date_time", "")
            for turn in turns:
                speaker = turn.get("speaker", "unknown")
                speaker = conversation.get(speaker, speaker)
                text = str(turn.get("text", ""))
                caption = str(turn.get("blip_caption", "") or "").strip()
                if caption:
                    text = f"{text} [shared image: {caption}]".strip()
                blocks["DIA-" + turn["dia_id"].replace(":", "-")] = (f"[{speaker}] {text}", stamp)
    else:
        counts = {}
        dates = sample.get("haystack_dates", [])
        for i, session in enumerate(sample["haystack_sessions"]):
            stamp = day(dates[i] if i < len(dates) else "") or "2024-01-01"
            for turn in session:
                if not turn.get("content", "").strip():
                    continue
                counts[stamp] = counts.get(stamp, 0) + 1
                text = f"{turn.get('role', '')}: {turn.get('content', '')}"
                text = " / ".join(p.strip() for p in text.splitlines() if p.strip())
                text = text.replace("[", "(").replace("]", ")").strip()
                blocks[f"D-{stamp.replace('-', '')}-{counts[stamp]:03d}"] = (text, stamp)
    return blocks


def day(value):
    match = re.match(r"(\d{4})/(\d{2})/(\d{2})", value or "")
    return "-".join(match.groups()) if match else ""


def reconstruct(bundle: Path, datasets: dict, rows: list[dict], verify_prompts=True):
    """Rebuild all inputs; output only aggregate verification counts."""
    from benchmarks.release_frozen_facts import computed_facts
    from mind_mem.answer_quality import classify_question_category
    from mind_mem.chat_memory import _build_prompt, _render_facts, _to_evidence

    sources = unique(read_rows(bundle / "prompt_sources.jsonl"))
    hashes = unique(read_rows(bundle / "prompt_hashes.jsonl"))
    if set(sources) != set(hashes) or set(hashes) != set(unique(rows)):
        raise ValueError("Input recipe population mismatch")
    lme = {r["question_id"]: r for r in datasets["longmemeval_s"]}
    block_cache = {}
    rebuilt = []
    for row in rows:
        if row["benchmark"] == "locomo":
            _, ci, qi = row["id"].split("-")
            sample = datasets["locomo"][int(ci)]
            item = sample["qa"][int(qi)]
            asked = item["question"].strip()
            category = "adversarial" if item["category"] == 5 else classify_question_category(asked)
        else:
            sample = item = lme[row["group"]]
            asked = f"Today is {day(item.get('question_date', '')) or 'unknown'}. {item['question']}"
            asked = asked.strip()
            category = classify_question_category(asked)
        rebuilt.append({**row, "question": item["question"], "reference": item.get("adversarial_answer", item.get("answer", ""))})
        if not verify_prompts:
            continue
        group = row["benchmark"], row["group"]
        if group not in block_cache:
            block_cache[group] = source_blocks(sample, row["benchmark"])
        hits = []
        for source in sources[row["id"]]["hits"]:
            full, stamp = block_cache[group][source["id"]]
            excerpt = full[: source["excerpt_chars"]]
            if text_hash(excerpt) != source["excerpt_sha256"]:
                raise ValueError("Source excerpt reconstruction mismatch")
            hits.append(
                {"_id": source["id"], "excerpt": excerpt, "Date": stamp, "score": source["score"], "file": "decisions/DECISIONS.md"}
            )
        evidence = _to_evidence(hits)
        facts = _render_facts(evidence, 4000)
        extra = computed_facts(asked, evidence)
        if extra:
            facts += "\n\n" + extra
        prompt = _build_prompt(asked, category, facts)
        if text_hash(prompt) != hashes[row["id"]]["prompt_sha256"]:
            raise ValueError("Rebuilt prompt hash mismatch: " + (row["id"] if row["partition"] == "dev" else "held-out aggregate"))
    return rebuilt


def corrected_dataset(dataset, corrections):
    result = copy.deepcopy(dataset)
    for row in corrections:
        _, ci, qi = row["id"].split("-")
        qa = result[int(ci)]["qa"][int(qi)]
        if text_hash(qa["question"]) != row["question_sha256"] or str(qa["answer"]) != str(row["official_gold"]):
            raise ValueError("Correction does not match downloaded source")
        qa["answer"] = row["corrected_answer"]
    return result
