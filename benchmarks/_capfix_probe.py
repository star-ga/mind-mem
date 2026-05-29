#!/usr/bin/env python3
"""Isolation probe: quantify the recall.dedup type_cap=3 ceiling.

Pure BM25-sqlite, one sanitised block per session, no vector / 4b /
cross-encoder. Caps ON (product default) vs OFF. Fast (no GPU model).
"""
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import time

sys.setrecursionlimit(12000)  # mind-mem's layered recall graph is deep
                              # over 48-session haystacks; result-neutral.
logging.basicConfig(level=logging.ERROR)
for _n in ("recall", "sqlite_index", "hybrid_recall", "dedup", "intent_router",
           "connection_manager", "retrieval_graph", "observability"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.ERROR)

sys.path.insert(0, "src")
from mind_mem.recall import recall  # noqa: E402
from mind_mem.sqlite_index import build_index  # noqa: E402

K = [1, 3, 5, 10]


def san(t: str) -> str:
    return " / ".join(p.strip() for p in t.splitlines() if p.strip()).replace("\n", " ")


def write_ws(q, tmp):
    ws = os.path.join(tmp, f"cp_{q.get('question_id', 'x')}")
    d = os.path.join(ws, "decisions")
    os.makedirs(d, exist_ok=True)
    S = q.get("haystack_sessions", [])
    ids = q.get("haystack_session_ids", list(range(len(S))))
    dt = q.get("haystack_dates", [])
    blocks = []
    for i, s in enumerate(S):
        sid = ids[i] if i < len(ids) else i
        date = (dt[i] if i < len(dt) else "2024-01-01").replace("\n", " ").strip() or "2024-01-01"
        stmt = " // ".join(f"({tn.get('role','?')}) {san(tn.get('content',''))}" for tn in s) or "(empty)"
        blocks.append(f"[SESSION-{sid}]\nStatement: {stmt}\nDate: {date}\nStatus: active\n")
    open(os.path.join(d, "DECISIONS.md"), "w", encoding="utf-8").write("\n---\n\n".join(blocks))
    return ws


def cfg(ws, caps_on):
    c = {"recall": {"backend": "sqlite", "knee_cutoff": False,
                    "cross_encoder": {"enabled": False, "auto_enable": False}}}
    if not caps_on:
        c["recall"]["dedup"] = {"enabled": False, "type_cap_enabled": False,
                                "source_cap_enabled": False, "cosine_enabled": False,
                                "best_per_source": False}
    json.dump(c, open(os.path.join(ws, "mind-mem.json"), "w"))


def run(sample, caps_on):
    H = {f"any@{k}": 0 for k in K}
    H.update({f"all@{k}": 0 for k in K})
    rr = 0.0
    n = 0
    for q in sample:
        ws = write_ws(q, tempfile.gettempdir())
        cfg(ws, caps_on)
        build_index(ws, incremental=False)
        res = recall(ws, q["question"], limit=20, active_only=False)
        rid = [r["_id"][8:] for r in res if r.get("_id", "").startswith("SESSION-")]
        g = set(q["answer_session_ids"])
        for k in K:
            tk = set(rid[:k])
            H[f"any@{k}"] += 1 if g & tk else 0
            H[f"all@{k}"] += 1 if g.issubset(tk) else 0
        for i, s in enumerate(rid, 1):
            if s in g:
                rr += 1.0 / i
                break
        n += 1
        shutil.rmtree(ws, ignore_errors=True)
    return {k: round(v / n, 3) for k, v in H.items()} | {"mrr": round(rr / n, 3)}, n


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    data = json.load(open("benchmarks/.cache/longmemeval_s.json"))
    random.seed(42)
    pool = [q for q in data if not q.get("question_id", "").endswith("_abs")
            and q.get("answer_session_ids")]
    sample = random.sample(pool, N)
    t = time.time()
    on, n = run(sample, True)
    ton = time.time() - t
    t = time.time()
    off, _ = run(sample, False)
    toff = time.time() - t
    print(f"\n=== BM25-sqlite, 1-block/session, n={n}, no vector/4b/CE ===")
    print(f"CAPS ON  ({ton:.0f}s): any@5={on['any@5']:.3f} any@10={on['any@10']:.3f} "
          f"all@5={on['all@5']:.3f} mrr={on['mrr']:.3f}")
    print(f"CAPS OFF ({toff:.0f}s): any@5={off['any@5']:.3f} any@10={off['any@10']:.3f} "
          f"all@5={off['all@5']:.3f} mrr={off['mrr']:.3f}")
    print("full ON :", on)
    print("full OFF:", off)
