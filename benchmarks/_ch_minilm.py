#!/usr/bin/env python3
"""LongMemEval-S — per-turn chunking + hybrid (BM25F+mxbai RRF) + 4b expansion.

The >95 attempt. Chunked substrate (short blocks → no BM25 length
dilution, fits the 512-token embed window) + dense mxbai vector + RRF
+ optional mind-mem:4b multi-query expansion. Hits rolled up to the
session for recall_any@k.

Run pinned against the all-fixes src:

  PYTHONPATH=/tmp/mm-fixed/src MIND_MEM_DISABLE_TELEMETRY=1 \
    python3 benchmarks/longmemeval_chunked_hybrid.py --sample 50 \
    --expand 3 --data-path benchmarks/.cache/longmemeval_s.json --output r.json
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
# Production-sane: bound peak embed VRAM so the dense encoder co-resides
# with Ollama-served mind-mem:4b on a 10 GB card (single-pass, the real
# product path — NOT a precompute/unload two-pass benchmark cheat).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mind_mem.recall_vector as _rv  # noqa: E402
from mind_mem.hybrid_recall import HybridBackend  # noqa: E402
from mind_mem.recall_vector import VectorBackend  # noqa: E402
from mind_mem.sqlite_index import build_index  # noqa: E402

_MC: dict = {}
_orig = _rv.VectorBackend.model.fget


def _cached(self):  # noqa: ANN001
    k = (self.model_name, self.config.get("vector_device", "cpu"))
    m = _MC.get(k)
    if m is not None:
        self._model, self._model_loaded = m, True
        return m
    m = _orig(self)
    _MC[k] = m
    return m


_rv.VectorBackend.model = property(_cached)

# (small-batch embed cap removed: all-MiniLM-L6-v2 is ~90 MB, no OOM)

K = [1, 3, 5, 10]
OLLAMA = "http://127.0.0.1:11434"
CFG = {
    "recall": {
        "backend": "sqlite",
        "knee_cutoff": False,
        "vector_enabled": True,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_device": "cpu",
        "provider": "local",
        "bm25_weight": 1.0,
        "vector_weight": 1.0,
        "rrf_k": 60,
        "dedup": {"enabled": False, "type_cap_enabled": False,
                  "source_cap_enabled": False, "cosine_enabled": False,
                  "best_per_source": False},
        "cross_encoder": {"enabled": False, "auto_enable": False},
        "query_expansion": {"enabled": False, "auto_enable": False},
        "decompose": {"enabled": False, "auto_enable": False},
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


def rrf(rank_lists, k=60):
    sc: dict = {}
    for lst in rank_lists:
        for i, s in enumerate(lst, 1):
            sc[s] = sc.get(s, 0.0) + 1.0 / (k + i)
    return [s for s, _ in sorted(sc.items(), key=lambda x: x[1], reverse=True)]


def _san(t: str) -> str:
    return " / ".join(p.strip() for p in t.splitlines() if p.strip()).replace("\n", " ")


def _ws(q, tmp):
    ws = os.path.join(tmp, f"ch_{q.get('question_id', 'x')}")
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
                bl.append(f"[SESSION-{sid}__t{j}]\nStatement: {c}\nDate: 2024-01-01\nStatus: active\n")
    open(os.path.join(d, "DECISIONS.md"), "w", encoding="utf-8").write("\n---\n\n".join(bl))
    json.dump(CFG, open(os.path.join(ws, "mind-mem.json"), "w"))
    return ws


def _roll(res):
    out: list[str] = []
    for r in res:
        b = r.get("_id", "")
        if b.startswith("SESSION-"):
            sid = b[len("SESSION-"):].split("::")[0].split("__t")[0]
            if sid not in out:
                out.append(sid)
    return out


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
        lists.append(_roll(hb.search(qv, ws, limit=80, active_only=False)))
    fused = rrf(lists) if len(lists) > 1 else (lists[0] if lists else [])
    rec = {}
    for k in K:
        tk = set(fused[:k])
        rec[f"any@{k}"] = 1 if gold & tk else 0
        rec[f"all@{k}"] = 1 if gold.issubset(tk) else 0
    rr = 0.0
    for rk, s in enumerate(fused, 1):
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
    bt: dict = {}
    for x in pq:
        bt.setdefault(x["qtype"], []).append(x)
    by = {t: round(sum(z["any@5"] for z in v) / len(v), 3) for t, v in sorted(bt.items())}
    out = {"harness": "chunked+hybrid+4b", "expand": a.expand,
           "sample": a.sample or "full", "seed": a.seed,
           "elapsed_seconds": round(time.time() - t0, 2), "evaluated": n,
           "overall": o, "by_type_any5": by}
    print(json.dumps(o, indent=2))
    print("by_type any@5:", json.dumps(by), file=sys.stderr)
    if a.output:
        out["per_question"] = pq
        json.dump(out, open(a.output, "w"), indent=2)
        print(f"wrote {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
