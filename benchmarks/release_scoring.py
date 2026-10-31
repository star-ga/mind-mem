"""Offline evidence verification and aggregate scoring for frozen answer records.

Copyright (c) STARGA Inc. All rights reserved.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        if not isinstance(row.get("id"), str) or row["id"] in result:
            raise ValueError("Missing or duplicate record identity")
        result[row["id"]] = row
    return result


def clean(answer: str) -> str:
    return " ".join(re.sub(r"\[\[[^\]]*\]\]", "", answer).split())


def tokens(value: Any) -> list[str]:
    text = str(value).lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\b(a|an|the)\b", " ", text).split()


def text_metrics(answer: str, references: list[Any]) -> tuple[float, float]:
    """Normalized exact match and multiset token F1; empty/empty is a match."""
    if not references:
        raise ValueError("No references")
    a = tokens(clean(answer))
    exact, f1 = 0.0, 0.0
    for reference in references:
        b = tokens(reference)
        exact = max(exact, float(a == b))
        overlap = sum((Counter(a) & Counter(b)).values())
        score = 2 * overlap / (len(a) + len(b)) if a or b else 1.0
        f1 = max(f1, score)
    return exact, f1


class CorrectedKey:
    """Apply exact-answer or hash-bound adjudications, including negative flips."""

    def __init__(self, rows: list[dict], adjudications: list[dict]):
        self.rows = unique(rows)
        self.adjudications = {}
        for row in rows:
            if row["verdict"] not in {"AUDIT_WRONG", "GOLD_WRONG", "GOLD_AMBIGUOUS"}:
                raise ValueError("Unknown correction verdict")
            if not row["accepted_answers"] or not row["evidence"]:
                raise ValueError("Correction lacks answers or evidence")
        for row in adjudications:
            if hashlib.sha256(row["answer"].encode()).hexdigest() != row["answer_sha256"]:
                raise ValueError("Adjudication answer hash mismatch")
            if type(row["corrected_ok"]) is not bool:
                raise ValueError("Adjudication must be boolean")
            pair = row["id"], row["answer_sha256"], row["key_sha256"]
            if pair in self.adjudications:
                raise ValueError("Duplicate adjudication")
            self.adjudications[pair] = row["corrected_ok"]

    def score(self, row: dict) -> bool:
        key = self.rows.get(row["id"])
        if key is None or key["verdict"] == "AUDIT_WRONG":
            return row["official_ok"]
        if (
            (key.get("question") is not None and row["question"] != key["question"])
            or (key.get("question_sha256") is not None and hashlib.sha256(row["question"].encode()).hexdigest() != key["question_sha256"])
            or str(row["reference"]) != str(key["official_gold"])
        ):
            raise ValueError("Correction question/reference mismatch")
        answer = row["generated"]
        if answer.strip() in [str(a).strip() for a in key["accepted_answers"]]:
            return True
        pair = row["id"], hashlib.sha256(answer.encode()).hexdigest(), digest(key)
        if pair not in self.adjudications:
            raise ValueError("Unreviewed answer for revised key; no corrected score can be published")
        return self.adjudications[pair]

    def references(self, row: dict) -> list[Any]:
        key = self.rows.get(row["id"])
        if key and key["verdict"] != "AUDIT_WRONG":
            return key["accepted_answers"]
        return [row["reference"]]


def verify_bundle(root: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    prompt_file = "prompt_hashes.jsonl" if "prompt_hashes.jsonl" in manifest["artifacts"] else "prompts.jsonl"
    if not {"answers.jsonl", prompt_file} <= set(manifest["artifacts"]):
        raise ValueError("Manifest omits required evidence artifacts")
    for name, expected in manifest["artifacts"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
            raise ValueError("Artifact hash mismatch or unsafe path")
    rows = read_rows(root / "answers.jsonl")
    answers = unique(rows)
    prompts = unique(read_rows(root / prompt_file))
    if set(answers) != set(prompts) or len(rows) != manifest["rows"]:
        raise ValueError("Answer/prompt identity sets or count differ")
    for row in rows:
        prompt = prompts[row["id"]]
        have = hashlib.sha256(prompt["messages"][-1]["content"].encode()).hexdigest() if "messages" in prompt else prompt["prompt_sha256"]
        if have != row["prompt_sha256"] or have != prompt["prompt_sha256"]:
            raise ValueError("Answer/prompt binding mismatch")
        votes = row["votes"]
        if type(row["official_ok"]) is not bool or len(votes) not in (1, 3):
            raise ValueError("Invalid verdict or vote count")
        if any(type(v.get("ok")) is not bool for v in votes):
            raise ValueError("Non-boolean vote")
        for vote in votes:
            score = vote.get("score", (vote.get("judge") or {}).get("score"))
            if score is not None and (int(score) >= manifest.get("judge", {}).get("threshold", 50)) != vote["ok"]:
                raise ValueError("Saved vote disagrees with its numeric score")
        if (sum(v["ok"] for v in votes) * 2 > len(votes)) != row["official_ok"]:
            raise ValueError("Saved verdict disagrees with majority")
    return manifest, rows


def aggregate(rows: list[dict], key: CorrectedKey | None = None) -> dict:
    """Return aggregate-only summaries. Never emit per-item mistakes."""
    if not rows:
        raise ValueError("Empty evaluation")
    unique(rows)
    out = {}
    for partition in ("full", "dev", "held_out"):
        selected = rows if partition == "full" else [r for r in rows if r["partition"] == partition]
        if any(r["partition"] not in {"dev", "held_out"} for r in selected):
            raise ValueError("Unknown partition")
        correct = 0
        em, f1 = 0.0, 0.0
        for row in selected:
            if type(row["official_ok"]) is not bool:
                raise ValueError("Non-boolean verdict")
            correct += key.score(row) if key else row["official_ok"]
            e, f = text_metrics(row["generated"], key.references(row) if key else [row["reference"]])
            em += e
            f1 += f
        n = len(selected)
        out[partition] = {
            "correct": correct,
            "total": n,
            "accuracy": correct / n if n else None,
            "exact_match": em / n if n else None,
            "token_f1": f1 / n if n else None,
        }
    return out


def require_no_regression(before: dict, after: dict) -> None:
    """Comparisons require the same denominator and nondecreasing held-out count."""
    b, a = before["held_out"], after["held_out"]
    if not b["total"] or b["total"] != a["total"] or a["correct"] < b["correct"]:
        raise ValueError("Held-out regression or incomparable populations")


def verify_correction_evidence(root: Path, rows: list[dict], dataset=None, corrected_dataset=None) -> dict:
    """Check literal citations and drop-in answers without judging their meaning."""
    dataset = dataset if dataset is not None else json.loads((root / "locomo10.official.json").read_text(encoding="utf-8"))
    corrected_dataset = (
        corrected_dataset if corrected_dataset is not None else json.loads((root / "locomo10.corrected.json").read_text(encoding="utf-8"))
    )
    expected_dataset = copy.deepcopy(dataset)
    changed = unique(read_rows(root / "correction_evidence.jsonl"))
    expected = {r["id"] for r in rows if r["verdict"] != "AUDIT_WRONG"}
    if set(changed) != expected:
        raise ValueError("Evidence rows do not exactly cover changed answers")
    citations = 0
    for row in rows:
        _, conversation, question = row["id"].split("-")
        qa = corrected_dataset[int(conversation)]["qa"][int(question)]
        original_qa = dataset[int(conversation)]["qa"][int(question)]
        if (
            str(original_qa["answer"]) != str(row["official_gold"])
            or (row.get("question") is not None and original_qa["question"] != row["question"])
            or (
                row.get("question_sha256") is not None
                and hashlib.sha256(original_qa["question"].encode()).hexdigest() != row["question_sha256"]
            )
        ):
            raise ValueError("Review row does not match the official dataset")
        if row["verdict"] != "AUDIT_WRONG":
            expected_dataset[int(conversation)]["qa"][int(question)]["answer"] = row["corrected_answer"]
        desired = row["official_gold"] if row["verdict"] == "AUDIT_WRONG" else row["corrected_answer"]
        if qa["question"] != original_qa["question"] or str(qa["answer"]) != str(desired):
            raise ValueError("Drop-in key differs from unchanged review rows")
        if row["id"] in changed:
            derived = changed[row["id"]]
            if derived["source_row_sha256"] != digest(row) or derived["quoted_turns"] != row["evidence"]:
                raise ValueError("Derived evidence differs from source row")
            if derived["dia_ids"] != [e["turn_id"] for e in row["evidence"]]:
                raise ValueError("Derived dialogue identifiers differ from source row")
        for evidence in row["evidence"]:
            sample = dataset[evidence["conversation"]]
            if sample["sample_id"] != evidence["sample_id"]:
                raise ValueError("Evidence conversation identity mismatch")
            turns = [t for k, v in sample["conversation"].items() if k.startswith("session_") and isinstance(v, list) for t in v]
            matches = [t for t in turns if t.get("dia_id") == evidence["turn_id"]]
            if not matches or not any(evidence["quote"] in t.get("text", "") for t in matches):
                raise ValueError("Evidence quote is not literal source text")
            if evidence.get("caption_quote") and not any(evidence["caption_quote"] in t.get("blip_caption", "") for t in matches):
                raise ValueError("Evidence caption is not literal source text")
            citations += 1
    if corrected_dataset != expected_dataset:
        raise ValueError("Drop-in key contains unreviewed changes")
    return {"review_rows": len(rows), "changed_answers": len(expected), "literal_citations": citations, "semantic_verification": "pending"}
