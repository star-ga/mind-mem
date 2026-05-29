#!/usr/bin/env python3
"""LongMemEval-S — clean reproducible mind-mem retrieval benchmark.

Deterministic, zero-LLM, BM25-sqlite path (the product's default lexical
retriever). One block per session (all turns), caps off, knee off,
cross-encoder off — a plain Recall@K measurement.

Reports recall_any@k (>=1 gold session in top-k — agentmemory / common
LongMemEval session-recall metric) and recall_all@k (all gold sessions
in top-k — official headline) plus MRR, with per-type breakdown.

Requires the block_parser >100 KB truncation fix
(commit "block_parser silently truncated corpus past 100 KB").

    python3 benchmarks/longmemeval_clean.py --output result.json
    python3 benchmarks/longmemeval_clean.py --sample 50 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time

os.environ.setdefault("MIND_MEM_DISABLE_TELEMETRY", "1")

from mind_mem.recall import recall  # noqa: E402
from mind_mem.sqlite_index import build_index  # noqa: E402

K = [1, 3, 5, 10]
CFG = {
    "recall": {
        "backend": "sqlite",
        "knee_cutoff": False,
        "cross_encoder": {"enabled": False, "auto_enable": False},
        "dedup": {
            "enabled": False,
            "type_cap_enabled": False,
            "source_cap_enabled": False,
            "cosine_enabled": False,
            "best_per_source": False,
        },
    }
}


def _san(t: str) -> str:
    return " / ".join(p.strip() for p in t.splitlines() if p.strip()).replace("\n", " ")


def _ws(q: dict, tmp: str) -> str:
    ws = os.path.join(tmp, f"lmc_{q.get('question_id', 'x')}")
    d = os.path.join(ws, "decisions")
    os.makedirs(d, exist_ok=True)
    S = q.get("haystack_sessions", [])
    ids = q.get("haystack_session_ids", list(range(len(S))))
    blocks = []
    for i, s in enumerate(S):
        sid = ids[i] if i < len(ids) else i
        stmt = " / ".join(
            _san(f"{t.get('role', '')}: {t.get('content', '')}") for t in s
        ) or "(empty)"
        blocks.append(
            f"[SESSION-{sid}]\nStatement: {stmt}\nDate: 2024-01-01\nStatus: active\n"
        )
    open(os.path.join(d, "DECISIONS.md"), "w", encoding="utf-8").write(
        "\n---\n\n".join(blocks)
    )
    json.dump(CFG, open(os.path.join(ws, "mind-mem.json"), "w"))
    return ws


def _eval(q: dict, tmp: str) -> dict | None:
    qid = q.get("question_id", "")
    if qid.endswith("_abs"):
        return None
    gold = set(map(str, q.get("answer_session_ids", [])))
    query = q.get("question", "")
    if not gold or not query:
        return None
    ws = _ws(q, tmp)
    build_index(ws, incremental=False)
    res = recall(ws, query, limit=50, active_only=False)
    seen: list[str] = []
    for r in res:
        b = r.get("_id", "")
        if b.startswith("SESSION-"):
            sid = b[len("SESSION-"):].split("::")[0]
            if sid not in seen:
                seen.append(sid)
    rec = {}
    for k in K:
        tk = set(seen[:k])
        rec[f"any@{k}"] = 1 if gold & tk else 0
        rec[f"all@{k}"] = 1 if gold.issubset(tk) else 0
    rr = 0.0
    for rank, sid in enumerate(seen, 1):
        if sid in gold:
            rr = 1.0 / rank
            break
    shutil.rmtree(ws, ignore_errors=True)
    return {"qid": qid, "qtype": q.get("question_type", "unknown"),
            **rec, "rr": rr}


def _agg(pq: list[dict]) -> dict:
    if not pq:
        return {"n": 0}
    keys = [f"{m}@{k}" for m in ("any", "all") for k in K]

    def M(items):
        n = len(items)
        o = {"n": n, "mrr": round(sum(x["rr"] for x in items) / n, 4)}
        for kk in keys:
            o[kk] = round(sum(x[kk] for x in items) / n, 4)
        return o

    bt: dict[str, list[dict]] = {}
    for x in pq:
        bt.setdefault(x["qtype"], []).append(x)
    return {"overall": M(pq), "by_type": {t: M(v) for t, v in sorted(bt.items())},
            "n": len(pq)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="benchmarks/.cache/longmemeval_s.json")
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
    pq, skip, t0 = [], 0, time.time()
    for i, q in enumerate(pool):
        r = _eval(q, tempfile.gettempdir())
        if r is None:
            skip += 1
            continue
        pq.append(r)
        if (i + 1) % 50 == 0 or (i + 1) == len(pool):
            el = time.time() - t0
            print(f"  [{i + 1}/{len(pool)}] eval={len(pq)} {el:.0f}s "
                  f"({el / max(1, i + 1):.2f}s/q)", flush=True)
    out = {
        "harness": "longmemeval_clean",
        "pipeline": "BM25-sqlite, 1-block/session, knee/dedup/CE off, no LLM",
        "subset": "longmemeval_s_cleaned",
        "sample": a.sample or "full",
        "seed": a.seed,
        "elapsed_seconds": round(time.time() - t0, 2),
        "evaluated": len(pq),
        "skipped": skip,
        "aggregate": _agg(pq),
    }
    print(json.dumps(out["aggregate"].get("overall", {}), indent=2))
    if a.output:
        out["per_question"] = pq
        json.dump(out, open(a.output, "w"), indent=2)
        print(f"\nWrote {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
