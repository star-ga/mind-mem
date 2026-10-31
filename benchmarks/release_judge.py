"""Frozen scoring protocol and explicit HTTP transport, separate from answering.

Copyright (c) STARGA Inc. All rights reserved.
"""

from __future__ import annotations

import json
import re
import urllib.request

from benchmarks.release_scoring import clean

DECLINE = re.compile(r"no record found|not (?:mentioned|recorded|in the (?:evidence|records))|no (?:information|record)", re.I)


def request(messages: list[dict], model: str, url: str, api_key: str, max_tokens: int) -> str:
    payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def parse_score(raw: str, adversarial: bool) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {}
    if adversarial:

        def field(name: str, boolean: bool = False):
            if name in result:
                return result[name]
            pattern = r'"' + name + r'"\s*:\s*' + (r"(true|false)" if boolean else r"(\d+)")
            match = re.search(pattern, raw, re.I)
            return (match.group(1).lower() == "true" if boolean else int(match.group(1))) if match else False

        entity = bool(field("entity_correct", True))
        quality = max(0, min(100, int(field("rejection_quality"))))
        wrong = bool(field("wrong_entity_hallucination", True))
        score = 0 if wrong else max(quality, 80) if entity else min(quality, 40)
    else:
        match = re.search(r'"score"\s*:\s*(\d+)', raw)
        score = max(0, min(100, int(result.get("score", match.group(1) if match else 0))))
    return {"score": score, "reason": str(result.get("reason", raw[:200]))}


def judge_once(row: dict, templates: dict, call) -> dict:
    answer = clean(row["generated"])
    if row["benchmark"] == "locomo":
        adversarial = row["category"] == 5
        prefix = "ADVERSARIAL_" if adversarial else ""
        messages = [
            {"role": "system", "content": templates[prefix + "JUDGE_SYSTEM_PROMPT"]},
            {
                "role": "user",
                "content": templates[prefix + "JUDGE_USER_TEMPLATE"].format(
                    question=row["question"], reference=row["reference"], generated=answer
                ),
            },
        ]
        result = parse_score(call(messages, 300 if adversarial else 200), adversarial)
        return {"ok": result["score"] >= 50, "judge": result}
    if row["abstention"]:
        return {"ok": bool(DECLINE.search(answer)), "judge": {"rule": "decline"}}
    prompt = templates["LME_JUDGE"].format(q=row["question"], a=row["reference"], r=answer)
    raw = call([{"role": "user", "content": prompt}], 5)
    return {"ok": raw.strip().lower().startswith("yes"), "judge": {"raw": raw.strip()[:20]}}


def judge(row: dict, templates: dict, call) -> tuple[bool, list[dict]]:
    first = judge_once(row, templates, call)
    votes = [first]
    score = first["judge"].get("score")
    if not first["ok"] or (row["benchmark"] == "locomo" and score is not None and 40 <= score <= 60):
        votes.extend(judge_once(row, templates, call) for _ in range(2))
    return sum(v["ok"] for v in votes) * 2 > len(votes), votes
