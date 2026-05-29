#!/usr/bin/env python3
"""LongMemEval-S — per-turn passage chunking (architectural gap closure).

One block per conversation turn (id SESSION-<sid>__t<i>); retrieval
hits are rolled up to the session for recall_any@k. Short blocks fix
both BM25 length-normalization dilution and the dense embed-window
truncation. BM25-only, deterministic, zero-LLM.

Run pinned against the all-fixes src:

    PYTHONPATH=/tmp/mm-fixed/src MIND_MEM_DISABLE_TELEMETRY=1 \
      python3 benchmarks/longmemeval_chunked.py \
      --data-path benchmarks/.cache/longmemeval_s.json --output r.json
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


def _ws(q, tmp):
    ws = os.path.join(tmp, f"lmk_{q.get('question_id', 'x')}")
    d = os.path.join(ws, "decisions")
    os.makedirs(d, exist_ok=True)
    S = q.get("haystack_sessions", [])
    ids = q.get("haystack_session_ids", list(range(len(S))))
    bl = []
    for i, s in enumerate(S):
        sid = ids[i] if i < len(ids) else i
        for j, t in enumerate(s):
            c = _san(f"{t.get('role', '')}: {t.get('content', '')}")
            if c.strip():
                bl.append(
                    f"[SESSION-{sid}__t{j}]\nStatement: {c}\nDate: 2024-01-01\nStatus: active\n"
                )
    open(os.path.join(d, "DECISIONS.md"), "w", encoding="utf-8").write(
        "\n---\n\n".join(bl)
    )
    json.dump(CFG, open(os.path.join(ws, "mind-mem.json"), "w"))
    return ws


def _eval(q, tmp):
    qid = q.get("question_id", "")
    if qid.endswith("_abs"):
        return None
    gold = set(map(str, q.get("answer_session_ids", [])))
    query = q.get("question", "")
    if not gold or not query:
        return None
    ws = _ws(q, tmp)
    build_index(ws, incremental=False)
    res = recall(ws, query, limit=80, active_only=False)
    seen: list[str] = []
    for r in res:
        b = r.get("_id", "")
        if b.startswith("SESSION-"):
            sid = b[len("SESSION-"):].split("::")[0].split("__t")[0]
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
    return {"qid": qid, "qtype": q.get("question_type", "?"), **rec, "rr": rr}


def main():
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
    pq, t0 = [], time.time()
    for i, q in enumerate(pool):
        r = _eval(q, tempfile.gettempdir())
        if r:
            pq.append(r)
        if (i + 1) % 50 == 0 or (i + 1) == len(pool):
            el = time.time() - t0
            print(f"  [{i + 1}/{len(pool)}] {el:.0f}s ({el / (i + 1):.2f}s/q)", flush=True)
    n = len(pq)
    o = {"n": n, "mrr": round(sum(x["rr"] for x in pq) / n, 4)}
    for kk in [f"{m}@{k}" for m in ("any", "all") for k in K]:
        o[kk] = round(sum(x[kk] for x in pq) / n, 4)
    bt: dict = {}
    for x in pq:
        bt.setdefault(x["qtype"], []).append(x)
    by = {t: round(sum(z["any@5"] for z in v) / len(v), 3) for t, v in sorted(bt.items())}
    out = {"harness": "chunked-per-turn", "sample": a.sample or "full",
           "seed": a.seed, "elapsed_seconds": round(time.time() - t0, 2),
           "evaluated": n, "overall": o, "by_type_any5": by}
    print(json.dumps(o, indent=2))
    if a.output:
        out["per_question"] = pq
        json.dump(out, open(a.output, "w"), indent=2)
        print(f"wrote {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
