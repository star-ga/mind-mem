#!/usr/bin/env python3
"""LongMemEval-S apples-to-apples harness (mind-mem hybrid BM25+vector).

Built to match the methodology the background research established:

  * agentmemory's published 95.2% R@5 == BM25 + vector (local
    all-MiniLM-L6-v2 ONNX) RRF, ONE document per session (ALL turns),
    metric = recall_any@k, cleaned 500-question set, local-only.
  * Official LongMemEval session granularity == only USER turns per
    session, headline metric = recall_all@k.

This harness runs mind-mem's real hybrid path (sqlite BM25 + fact-card
sub-blocks + vector MiniLM RRF) under either protocol so the comparison
is like-for-like. Same embedding model as agentmemory => the vector
component is identical; only the retrieval/fusion engine differs.

  --turns all|user      which turns go into each session block
  --metric any|all      recall_any (>=1 gold in top-k) or recall_all
  --sample N            0 = full set
  --output FILE         write full JSON artifact
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
import time

from mind_mem.hybrid_recall import HybridBackend
from mind_mem.recall_vector import VectorBackend
from mind_mem.sqlite_index import build_index

K_VALUES = [1, 3, 5, 10]

CONFIG = {
    "recall": {
        "backend": "sqlite",
        "knee_cutoff": False,
        "vector_enabled": True,
        "vector_model": "all-MiniLM-L6-v2",
        "model": "all-MiniLM-L6-v2",
        "provider": "local",
        "bm25_weight": 1.0,
        "vector_weight": 1.0,
        "rrf_k": 60,
    }
}


def sanitise(text: str) -> str:
    return " / ".join(p.strip() for p in text.splitlines() if p.strip()).replace("\n", " ")


def session_block(session: list[dict], sid: str, date: str, turns: str) -> str:
    parts = []
    for t in session:
        role = t.get("role", "unknown")
        if turns == "user" and role != "user":
            continue
        parts.append(f"({role}) {sanitise(t.get('content', ''))}")
    statement = " // ".join(parts) or "(empty)"
    safe_date = (date or "2024-01-01").replace("\n", " ").strip() or "2024-01-01"
    return f"[SESSION-{sid}]\nStatement: {statement}\nDate: {safe_date}\nStatus: active\n"


def build_ws(q: dict, tmp: str, turns: str) -> str:
    ws = os.path.join(tmp, f"aws_{q.get('question_id', 'x')}")
    d = os.path.join(ws, "decisions")
    os.makedirs(d, exist_ok=True)
    sessions = q.get("haystack_sessions", [])
    sids = q.get("haystack_session_ids", list(range(len(sessions))))
    dates = q.get("haystack_dates", [])
    blocks = [
        session_block(s, sids[i] if i < len(sids) else i,
                      dates[i] if i < len(dates) else "2024-01-01", turns)
        for i, s in enumerate(sessions)
    ]
    open(os.path.join(d, "DECISIONS.md"), "w", encoding="utf-8").write("\n---\n\n".join(blocks))
    json.dump(CONFIG, open(os.path.join(ws, "mind-mem.json"), "w"))
    return ws


def hit(retrieved: list[str], gold: set[str], k: int, metric: str) -> int:
    topk = set(retrieved[:k])
    if metric == "all":
        return 1 if gold and gold.issubset(topk) else 0
    return 1 if gold & topk else 0


def evaluate(q: dict, tmp: str, turns: str, metric: str) -> dict | None:
    qid = q.get("question_id", "")
    if qid.endswith("_abs"):
        return None
    query = q.get("question", "")
    gold = set(q.get("answer_session_ids", []))
    if not query or not gold:
        return None
    ws = build_ws(q, tmp, turns)
    try:
        build_index(ws, incremental=False)
        VectorBackend(CONFIG["recall"]).index(ws)
        hb = HybridBackend.from_config(CONFIG)
        results = hb.search(query, ws, limit=10, active_only=False)
    finally:
        pass
    retrieved = [r["_id"][len("SESSION-"):] for r in results
                 if r.get("_id", "").startswith("SESSION-")]
    rk = {f"recall@{k}": hit(retrieved, gold, k, metric) for k in K_VALUES}
    rr = 0.0
    for rank, sid in enumerate(retrieved, 1):
        if sid in gold:
            rr = 1.0 / rank
            break
    shutil.rmtree(ws, ignore_errors=True)
    return {"question_id": qid, "question_type": q.get("question_type", "unknown"),
            **rk, "reciprocal_rank": rr}


def agg(per_q: list[dict]) -> dict:
    if not per_q:
        return {"n": 0}
    bt: dict[str, list[dict]] = {}
    for q in per_q:
        bt.setdefault(q["question_type"], []).append(q)

    def m(items):
        n = len(items)
        o = {"n": n}
        for k in K_VALUES:
            o[f"recall@{k}"] = round(sum(x[f"recall@{k}"] for x in items) / n, 4)
        o["mrr"] = round(sum(x["reciprocal_rank"] for x in items) / n, 4)
        return o

    return {"overall": m(per_q), "by_type": {t: m(v) for t, v in sorted(bt.items())},
            "n": len(per_q)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="benchmarks/.cache/longmemeval_s.json")
    ap.add_argument("--turns", choices=["all", "user"], default="all")
    ap.add_argument("--metric", choices=["any", "all"], default="any")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output")
    a = ap.parse_args()

    data = json.load(open(a.data_path))
    pool = [q for q in data if not q.get("question_id", "").endswith("_abs")
            and q.get("answer_session_ids")]
    if a.sample:
        random.seed(a.seed)
        pool = random.sample(pool, a.sample)

    per_q, skipped, t0 = [], 0, time.time()
    for i, q in enumerate(pool):
        r = evaluate(q, tempfile.gettempdir(), a.turns, a.metric)
        if r is None:
            skipped += 1
            continue
        per_q.append(r)
        if (i + 1) % 25 == 0 or (i + 1) == len(pool):
            print(f"  [{i + 1}/{len(pool)}] eval={len(per_q)} skip={skipped} "
                  f"{time.time() - t0:.0f}s", flush=True)
    out = {
        "harness": "longmemeval_apples",
        "pipeline": f"hybrid BM25+vector(all-MiniLM-L6-v2)+RRF, sqlite+factcards, "
                    f"knee_off; turns={a.turns}; metric=recall_{a.metric}",
        "subset": "longmemeval_s_cleaned",
        "sample": a.sample or "full",
        "seed": a.seed,
        "elapsed_seconds": round(time.time() - t0, 2),
        "evaluated": len(per_q),
        "skipped": skipped,
        "aggregate": agg(per_q),
    }
    print(json.dumps(out["aggregate"]["overall"], indent=2))
    if a.output:
        out["per_question"] = per_q
        json.dump(out, open(a.output, "w"), indent=2)
        print(f"\nWrote {a.output}")


if __name__ == "__main__":
    main()
