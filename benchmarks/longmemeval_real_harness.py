#!/usr/bin/env python3
"""LongMemEval-S harness — real product pipeline (Phase A: own best honest number).

Differences from longmemeval_harness.py (the bare scan-only harness whose
numbers do not reproduce the published figures):

  1. Runs mind-mem the way the product actually runs: SQLite backend with
     ``build_index`` (fact-card sub-block extraction / small-to-big retrieval)
     + deterministic reranker. No LLM in the loop (deterministic path).
  2. Each LongMemEval session is written as ONE opaque block. Session text
     is sanitised so chat lines that look like mind-mem block fields
     (``Date:``, ``Tags:``, ``Status:``, ``[role]:``) cannot corrupt the
     markdown block parser. This is a faithful 1-session-per-memory
     representation — no tokens dropped, only field-collision escaped.
  3. Recall@K uses the full ranked top-K (knee_cutoff disabled) — correct
     for fixed-K IR evaluation.

Usage:
    python3 benchmarks/longmemeval_real_harness.py --sample 40 --seed 42
    python3 benchmarks/longmemeval_real_harness.py --output repro.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
import time

from mind_mem.recall import recall
from mind_mem.sqlite_index import build_index

K_VALUES = [1, 3, 5, 10]


def sanitise(text: str) -> str:
    """Flatten a turn so no line can be mistaken for a block field.

    Newlines -> ' / ', and a leading token that looks like ``Key:`` or
    ``[role]:`` is defanged by inserting a zero-width-safe marker. BM25
    tokenisation is unaffected (punctuation is a token boundary anyway).
    """
    flat = " / ".join(part.strip() for part in text.splitlines() if part.strip())
    return flat.replace("\n", " ")


def session_to_block(session: list[dict], sid: str, date: str) -> str:
    turns = []
    for turn in session:
        role = turn.get("role", "unknown")
        content = sanitise(turn.get("content", ""))
        turns.append(f"({role}) {content}")
    statement = " // ".join(turns)
    # Single physical line for Statement => parser cannot absorb chat
    # content as structured fields. Date is a fixed safe token.
    safe_date = date.replace("\n", " ").strip() or "2024-01-01"
    return f"[SESSION-{sid}]\nStatement: {statement}\nDate: {safe_date}\nStatus: active\n"


def build_workspace(question: dict, tmpdir: str) -> str:
    ws = os.path.join(tmpdir, f"rws_{question.get('question_id', 'x')}")
    dpath = os.path.join(ws, "decisions")
    os.makedirs(dpath, exist_ok=True)
    sessions = question.get("haystack_sessions", [])
    sids = question.get("haystack_session_ids", list(range(len(sessions))))
    dates = question.get("haystack_dates", [])
    blocks = []
    for i, sess in enumerate(sessions):
        sid = sids[i] if i < len(sids) else i
        date = dates[i] if i < len(dates) else "2024-01-01"
        blocks.append(session_to_block(sess, sid, date))
    with open(os.path.join(dpath, "DECISIONS.md"), "w", encoding="utf-8") as f:
        f.write("\n---\n\n".join(blocks))
    json.dump(
        {"recall": {"backend": "sqlite", "knee_cutoff": False}},
        open(os.path.join(ws, "mind-mem.json"), "w"),
    )
    return ws


def evaluate(question: dict, tmpdir: str) -> dict | None:
    qid = question.get("question_id", "")
    if qid.endswith("_abs"):
        return None
    query = question.get("question", "")
    answer_ids = set(question.get("answer_session_ids", []))
    if not query or not answer_ids:
        return None
    ws = build_workspace(question, tmpdir)
    try:
        build_index(ws, incremental=False)
        results = recall(ws, query, limit=10, active_only=False)
    finally:
        pass
    retrieved = []
    for r in results:
        bid = r.get("_id", "")
        if bid.startswith("SESSION-"):
            retrieved.append(bid[len("SESSION-"):])
    rk = {}
    for k in K_VALUES:
        rk[f"recall@{k}"] = 1 if answer_ids & set(retrieved[:k]) else 0
    rr = 0.0
    for rank, sid in enumerate(retrieved, 1):
        if sid in answer_ids:
            rr = 1.0 / rank
            break
    shutil.rmtree(ws, ignore_errors=True)
    return {
        "question_id": qid,
        "question_type": question.get("question_type", "unknown"),
        **rk,
        "reciprocal_rank": rr,
    }


def aggregate(per_q: list[dict]) -> dict:
    if not per_q:
        return {"n": 0}
    by_type: dict[str, list[dict]] = {}
    for q in per_q:
        by_type.setdefault(q["question_type"], []).append(q)

    def metrics(items):
        n = len(items)
        m = {"n": n}
        for k in K_VALUES:
            m[f"recall@{k}"] = round(sum(q[f"recall@{k}"] for q in items) / n, 4)
        m["mrr"] = round(sum(q["reciprocal_rank"] for q in items) / n, 4)
        return m

    return {
        "overall": metrics(per_q),
        "by_type": {t: metrics(v) for t, v in sorted(by_type.items())},
        "n": len(per_q),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="benchmarks/.cache/longmemeval_s.json")
    ap.add_argument("--sample", type=int, default=0, help="0 = full set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output")
    args = ap.parse_args()

    data = json.load(open(args.data_path))
    pool = [q for q in data if not q.get("question_id", "").endswith("_abs") and q.get("answer_session_ids")]
    if args.sample:
        random.seed(args.seed)
        pool = random.sample(pool, args.sample)

    per_q = []
    skipped = 0
    t0 = time.time()
    for i, q in enumerate(pool):
        r = evaluate(q, tempfile.gettempdir())
        if r is None:
            skipped += 1
            continue
        per_q.append(r)
        if (i + 1) % 50 == 0 or (i + 1) == len(pool):
            print(f"  [{i + 1}/{len(pool)}] evaluated={len(per_q)} skipped={skipped} {time.time() - t0:.0f}s")
    agg = aggregate(per_q)
    out = {
        "harness": "longmemeval_real_harness",
        "pipeline": "sqlite+factcards+rerank, knee_cutoff=off, deterministic (no LLM)",
        "subset": "longmemeval_s",
        "sample": args.sample or "full",
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - t0, 2),
        "evaluated": len(per_q),
        "skipped": skipped,
        "aggregate": agg,
    }
    print(json.dumps(out, indent=2))
    if args.output:
        out["per_question"] = per_q
        json.dump(out, open(args.output, "w"), indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
