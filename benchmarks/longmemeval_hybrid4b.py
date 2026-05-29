#!/usr/bin/env python3
"""LongMemEval-S — mind-mem hybrid (BM25F+mxbai RRF) + 4b query expansion.

In-process (recursion bug fixed → no child-process wrapper needed).
Embedder loaded once (process-wide singleton). Run with the fixed code:

    PYTHONPATH=/tmp/mm-fixed/src MIND_MEM_DISABLE_TELEMETRY=1 \
      python3 benchmarks/longmemeval_hybrid4b.py --sample 50 --expand 3 \
      --data-path benchmarks/.cache/longmemeval_s.json --output r.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
import urllib.request

os.environ.setdefault("MIND_MEM_DISABLE_TELEMETRY", "1")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import mind_mem.recall_vector as _rv  # noqa: E402
from mind_mem.hybrid_recall import HybridBackend  # noqa: E402
from mind_mem.recall_vector import VectorBackend  # noqa: E402
from mind_mem.sqlite_index import build_index  # noqa: E402

_MODEL_CACHE: dict = {}
_orig = _rv.VectorBackend.model.fget


def _cached(self):  # noqa: ANN001
    k = (self.model_name, self.config.get("vector_device", "cpu"))
    m = _MODEL_CACHE.get(k)
    if m is not None:
        self._model = m
        self._model_loaded = True
        return m
    m = _orig(self)
    _MODEL_CACHE[k] = m
    return m


_rv.VectorBackend.model = property(_cached)

K = [1, 3, 5, 10]
OLLAMA = "http://127.0.0.1:11434"
CFG = {
    "recall": {
        "backend": "sqlite",
        "knee_cutoff": False,
        "vector_enabled": True,
        "model": "mixedbread-ai/mxbai-embed-large-v1",
        "vector_model": "mixedbread-ai/mxbai-embed-large-v1",
        "vector_device": "cuda",
        "provider": "local",
        "bm25_weight": 1.0,
        "vector_weight": 1.0,
        "rrf_k": 60,
        "dedup": {"enabled": False, "type_cap_enabled": False,
                  "source_cap_enabled": False, "cosine_enabled": False,
                  "best_per_source": False},
        "cross_encoder": {"enabled": False, "auto_enable": False},
    }
}
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def expand_4b(q: str, n: int) -> list[str]:
    if n <= 0:
        return [q]
    p = (f"Rewrite the question into {n} alternative search queries that "
         f"retrieve the same memory. One per line, no preamble.\n\nQuestion: {q}")
    body = json.dumps({"model": "mind-mem:4b", "prompt": p, "stream": False,
                       "options": {"temperature": 0.1, "num_predict": 120}}).encode()
    try:
        rq = urllib.request.Request(f"{OLLAMA}/api/generate", data=body,
                                    headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(rq, timeout=25) as r:
            txt = json.loads(r.read().decode()).get("response", "")
    except Exception:
        return [q]
    txt = _THINK.sub("", txt).strip()
    out, seen = [q], {q.strip().lower()}
    for ln in txt.splitlines():
        ln = ln.strip().lstrip("-0123456789.) ").strip()
        if ln and ln.lower() not in seen and len(ln) > 3:
            out.append(ln)
            seen.add(ln.lower())
        if len(out) >= n + 1:
            break
    return out


def rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    sc: dict[str, float] = {}
    for lst in rank_lists:
        for i, sid in enumerate(lst, 1):
            sc[sid] = sc.get(sid, 0.0) + 1.0 / (k + i)
    return [s for s, _ in sorted(sc.items(), key=lambda x: x[1], reverse=True)]


def _san(t: str) -> str:
    return " / ".join(p.strip() for p in t.splitlines() if p.strip()).replace("\n", " ")


def _ws(q, tmp):
    ws = os.path.join(tmp, f"h4_{q.get('question_id', 'x')}")
    d = os.path.join(ws, "decisions")
    os.makedirs(d, exist_ok=True)
    S = q.get("haystack_sessions", [])
    ids = q.get("haystack_session_ids", list(range(len(S))))
    bl = []
    for i, s in enumerate(S):
        sid = ids[i] if i < len(ids) else i
        st = " / ".join(_san(f"{t.get('role', '')}: {t.get('content', '')}") for t in s) or "(empty)"
        bl.append(f"[SESSION-{sid}]\nStatement: {st}\nDate: 2024-01-01\nStatus: active\n")
    open(os.path.join(d, "DECISIONS.md"), "w", encoding="utf-8").write("\n---\n\n".join(bl))
    json.dump(CFG, open(os.path.join(ws, "mind-mem.json"), "w"))
    return ws


def _eval(q, tmp, n_exp):
    qid = q.get("question_id", "")
    if qid.endswith("_abs"):
        return None
    gold = set(map(str, q.get("answer_session_ids", [])))
    query = q.get("question", "")
    if not gold or not query:
        return None
    ws = _ws(q, tmp)
    build_index(ws, incremental=False)
    VectorBackend(CFG["recall"]).index(ws)
    hb = HybridBackend.from_config(CFG)
    lists = []
    for qv in expand_4b(query, n_exp):
        res = hb.search(qv, ws, limit=20, active_only=False)
        lists.append([r["_id"][len("SESSION-"):].split("::")[0]
                      for r in res if r.get("_id", "").startswith("SESSION-")])
    fused = rrf(lists) if len(lists) > 1 else (lists[0] if lists else [])
    # dedupe preserve order
    seen, ordered = set(), []
    for s in fused:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    rec = {}
    for k in K:
        tk = set(ordered[:k])
        rec[f"any@{k}"] = 1 if gold & tk else 0
        rec[f"all@{k}"] = 1 if gold.issubset(tk) else 0
    rr = 0.0
    for rk, s in enumerate(ordered, 1):
        if s in gold:
            rr = 1.0 / rk
            break
    shutil.rmtree(ws, ignore_errors=True)
    return {"qid": qid, "qtype": q.get("question_type", "?"), **rec, "rr": rr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="benchmarks/.cache/longmemeval_s.json")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--expand", type=int, default=3)
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
        r = _eval(q, tempfile.gettempdir(), a.expand)
        if r:
            pq.append(r)
        if (i + 1) % 10 == 0 or (i + 1) == len(pool):
            el = time.time() - t0
            print(f"  [{i + 1}/{len(pool)}] {el:.0f}s ({el / (i + 1):.1f}s/q)", flush=True)
    n = len(pq)
    o = {"n": n, "mrr": round(sum(x["rr"] for x in pq) / n, 4)}
    for kk in [f"{m}@{k}" for m in ("any", "all") for k in K]:
        o[kk] = round(sum(x[kk] for x in pq) / n, 4)
    out = {"harness": "hybrid4b", "expand": a.expand, "sample": a.sample or "full",
           "seed": a.seed, "elapsed_seconds": round(time.time() - t0, 2),
           "evaluated": n, "overall": o}
    print(json.dumps(o, indent=2))
    if a.output:
        out["per_question"] = pq
        json.dump(out, open(a.output, "w"), indent=2)
        print(f"wrote {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
