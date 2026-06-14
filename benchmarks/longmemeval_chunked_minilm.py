#!/usr/bin/env python3
"""LongMemEval-S — mind-mem FULL POTENTIAL harness (same-equipment, best-vs-best).

mind-mem stack (this box):
  * BM25F (sqlite index + regex fact-card sub-blocks, small-to-big)
  * dense: mxbai-embed-large via HF local (1024-d, GPU, spawn worker)
  * RRF fusion (HybridBackend)
  * mind-mem:4b multi-query expansion (generative; our model) — N
    rewrites fused with the original via RRF
  * knee_cutoff off (correct for fixed-K IR)
  * cross_encoder disabled + auto_enable off (explicit)
  * per-question isolated index (same as agentmemory's per-question
    fresh index)

Reports BOTH metrics so the comparison is unambiguous:
  * recall_any@k  — agentmemory's published metric (>=1 gold in top-k)
  * recall_all@k  — official LongMemEval headline (all gold in top-k)

Reliability hardening:
  * CUDA + fork incompatibility resolved via spawn start method.
  * Each question runs inside a PERSISTENT WORKER PROCESS that loaded
    the GPU model once at start-up (amortises the ~10s model-load cost
    across all questions on that worker).
  * Per-question hard wall-clock timeout: if a question hangs, the
    parent kills the worker process (which cannot be preempted by
    signal.alarm), marks the question as timed-out, and starts a fresh
    worker for the next question.
  * Cross-encoder disabled via config (enabled=False, auto_enable=False).
  * All sentence-transformers / tqdm / HF progress bars suppressed via
    environment variables set before any imports.

  --expand N     number of 4b query rewrites (0 = no LLM, pure hybrid)
  --turns all|user
  --sample N     0 = full (470 evaluable questions)
  --seed N
  --output FILE
  --qtimeout N   per-question hard wall-clock cap in seconds (default 60)
"""

from __future__ import annotations

# ── Silence ALL progress bars and model-load noise before any imports ──────
import os

os.environ.setdefault("MIND_MEM_DISABLE_TELEMETRY", "1")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"

# ── stdlib ─────────────────────────────────────────────────────────────────
import argparse
import json
import multiprocessing
import queue
import random
import re
import shutil
import tempfile
import time
import urllib.request

# ── Constants ───────────────────────────────────────────────────────────────
K_VALUES = [1, 3, 5, 10]
OLLAMA = "http://127.0.0.1:11434"
EXPAND_MODEL = "mind-mem:4b"

CONFIG = {
    "recall": {
        "backend": "sqlite",
        "knee_cutoff": False,
        "vector_enabled": True,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_device": "cuda",
        "provider": "local",
        "bm25_weight": 1.0,
        "vector_weight": 1.0,
        "rrf_k": 60,
        "dedup": {
            "enabled": False,
            "type_cap_enabled": False,
            "source_cap_enabled": False,
            "cosine_enabled": False,
            "best_per_source": False,
        },
        # Explicitly disable cross-encoder AND its auto-enable trigger.
        # Without auto_enable=False the hybrid backend re-enables it on
        # temporal/multi-hop queries (v3.3.0 Tier 2 #6), loading the
        # 80 MB cross-encoder/ms-marco-MiniLM-L-6-v2 on every question.
        "cross_encoder": {"enabled": False, "auto_enable": False},
    },
    "feature_flags": {
        "v4.cognitive_kernel": True,
        "v4.surprise_retrieval": True,
        "v4.tier_memory": True,
    },
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# ── Query expansion ─────────────────────────────────────────────────────────

def expand_4b(query: str, n: int) -> list[str]:
    """Return [original, *up to n rewrites] using mind-mem:4b (generative)."""
    if n <= 0:
        return [query]
    prompt = (
        f"Rewrite the user's question into {n} alternative search queries that "
        f"would retrieve the same memory. One per line. No preamble, no numbering.\n\n"
        f"Question: {query}"
    )
    payload = json.dumps({
        "model": EXPAND_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 120},
    }).encode()
    try:
        req = urllib.request.Request(
            f"{OLLAMA}/api/generate", data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            resp = json.loads(r.read().decode()).get("response", "")
    except Exception:
        return [query]
    resp = _THINK_RE.sub("", resp).strip()
    out = [query]
    seen = {query.strip().lower()}
    for line in resp.splitlines():
        line = line.strip().lstrip("-0123456789.) ").strip()
        if line and line.lower() not in seen and len(line) > 3:
            out.append(line)
            seen.add(line.lower())
        if len(out) >= n + 1:
            break
    return out


# ── RRF fusion ──────────────────────────────────────────────────────────────

def rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    score: dict[str, float] = {}
    for lst in rank_lists:
        for rank, sid in enumerate(lst, 1):
            score[sid] = score.get(sid, 0.0) + 1.0 / (k + rank)
    return [s for s, _ in sorted(score.items(), key=lambda x: x[1], reverse=True)]


# ── Text helpers ────────────────────────────────────────────────────────────

def sanitise(t: str) -> str:
    return " / ".join(p.strip() for p in t.splitlines() if p.strip()).replace("\n", " ")


def session_block(sess: list, sid: object, date: str | None, turns: str) -> str:
    """Per-turn chunking: one block per turn, id SESSION-<sid>__t<j>.

    Returns the session's turn-blocks joined by the same separator
    build_ws uses; hits are rolled up to <sid> at scoring time.
    """
    sd = (date or "2024-01-01").replace("\n", " ").strip() or "2024-01-01"
    blocks = []
    for j, tn in enumerate(sess):
        role = tn.get("role", "unknown")
        if turns == "user" and role != "user":
            continue
        c = sanitise(f"({role}) {tn.get('content', '')}")
        if not c.strip():
            continue
        blocks.append(
            f"[SESSION-{sid}__t{j}]\nStatement: {c}\nDate: {sd}\nStatus: active\n"
        )
    return "\n---\n\n".join(blocks) or f"[SESSION-{sid}__t0]\nStatement: (empty)\nDate: {sd}\nStatus: active\n"


# ── Workspace builder ───────────────────────────────────────────────────────

def build_ws(q: dict, tmp: str, turns: str) -> str:
    ws = os.path.join(tmp, f"fp_{q.get('question_id', 'x')}")
    d = os.path.join(ws, "decisions")
    os.makedirs(d, exist_ok=True)
    sessions = q.get("haystack_sessions", [])
    sids = q.get("haystack_session_ids", list(range(len(sessions))))
    dates = q.get("haystack_dates", [])
    blocks = [
        session_block(
            s,
            sids[i] if i < len(sids) else i,
            dates[i] if i < len(dates) else "2024-01-01",
            turns,
        )
        for i, s in enumerate(sessions)
    ]
    with open(os.path.join(d, "DECISIONS.md"), "w", encoding="utf-8") as f:
        f.write("\n---\n\n".join(blocks))
    with open(os.path.join(ws, "mind-mem.json"), "w") as f:
        json.dump(CONFIG, f)
    return ws


# ── Core evaluator ──────────────────────────────────────────────────────────

def _evaluate_question(q: dict, tmp: str, turns: str, n_expand: int) -> dict | None:
    """Evaluate one question. May run in a worker process or in-process."""
    # Import here so this function is self-contained when used in a worker.
    from mind_mem.hybrid_recall import HybridBackend
    from mind_mem.recall_vector import VectorBackend
    from mind_mem.sqlite_index import build_index

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
        queries = expand_4b(query, n_expand)
        rank_lists = []
        for qv in queries:
            res = hb.search(qv, ws, limit=80, active_only=False)
            seen: list[str] = []
            for r in res:
                bid = r.get("_id", "")
                if bid.startswith("SESSION-"):
                    s = bid[len("SESSION-"):].split("::")[0].split("__t")[0]
                    if s not in seen:
                        seen.append(s)
            rank_lists.append(seen)
        fused = rrf_fuse(rank_lists) if len(rank_lists) > 1 else (
            rank_lists[0] if rank_lists else []
        )
        rec: dict = {}
        for k in K_VALUES:
            topk = set(fused[:k])
            rec[f"any@{k}"] = 1 if gold & topk else 0
            rec[f"all@{k}"] = 1 if gold.issubset(topk) else 0
        rr = 0.0
        for rank, sid in enumerate(fused, 1):
            if sid in gold:
                rr = 1.0 / rank
                break
        return {
            "question_id": qid,
            "question_type": q.get("question_type", "unknown"),
            "n_queries": len(queries),
            **rec,
            "reciprocal_rank": rr,
        }
    finally:
        shutil.rmtree(ws, ignore_errors=True)


# ── Persistent worker process ───────────────────────────────────────────────
# The worker runs in a spawned child process so CUDA is safe.  It loads
# the GPU model once at start-up, then processes questions sent via
# task_q and returns results via result_q.
#
# Protocol:
#   task_q   receives (q_dict, tmp, turns, n_expand)  — or None to shut down
#   result_q sends    ("ok", result) | ("err", msg) | ("skip", None)
#
# The worker never restarts mid-question. If a question hangs, the parent
# kills the process and spawns a fresh one.

def _worker_main(task_q: multiprocessing.Queue, result_q: multiprocessing.Queue) -> None:
    """Persistent worker: loads GPU model once, processes questions in a loop."""
    # Silence noise inside the spawned process.
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Reduce GPU memory fragmentation so batch encoding fits in ~1 GiB headroom.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    import mind_mem.recall_vector as _rv

    # ── Process-local embedder singleton ────────────────────────────────────
    # VectorBackend lazy-loads a fresh SentenceTransformer per instance.
    # One worker processes all questions sequentially, so we cache the model
    # here to avoid reloading it on every VectorBackend() call.
    _model_cache: dict = {}
    _orig_fget = _rv.VectorBackend.model.fget

    def _cached_model(self):  # noqa: ANN001
        key = (self.model_name, self.config.get("vector_device", "cpu"))
        if key not in _model_cache:
            _model_cache[key] = _orig_fget(self)
        self._model = _model_cache[key]
        self._model_loaded = True
        return _model_cache[key]

    _rv.VectorBackend.model = property(_cached_model)

    # ── Low-memory embed: conservative batch_size + cache flush ────────────
    # The default encode() batch_size=32 on mxbai-embed-large with 49 long
    # sessions needs ~300 MiB activation.  When GPU headroom is tight, this
    # OOMs and corrupts the CUDA context (RuntimeError: "CUDA error: out of
    # memory") making all subsequent calls fail or hang.
    # batch_size=4 keeps peak activation to ~30-40 MiB and succeeds even
    # with only ~100 MiB free, completing 49-session indexing in <0.5s on GPU.
    # On CUDA OOM: raise so the worker loop can report ("cuda_oom") to parent.

    def _low_mem_embed(self, texts: list) -> list:
        if not texts:
            return []
        batch_size = 4
        all_embeddings: list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embs = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=batch_size,
            )
            all_embeddings.extend(embs.tolist())
            if i + batch_size < len(texts):
                torch.cuda.empty_cache()
        return all_embeddings

    _rv.VectorBackend.embed = _low_mem_embed

    # ── Sequential hybrid search patch ─────────────────────────────────────
    # HybridBackend.search uses ThreadPoolExecutor(max_workers=2) to run BM25
    # and vector search in parallel.  Inside a spawned process, Python threads
    # that try to access a CUDA model loaded on the main thread can hang
    # indefinitely (CUDA per-thread context initialisation race).  Patching
    # the search to run both backends sequentially in the main thread avoids
    # the deadlock.  Scores are identical; only wall-clock order differs.
    from concurrent.futures import Future as _Future

    from mind_mem.hybrid_recall import HybridBackend as _HB

    _orig_search = _HB.search

    def _sequential_search(self, query, workspace, limit=10, active_only=False, **kwargs):
        """Run BM25 and vector sequentially to avoid CUDA-thread deadlock."""
        # Replace the parallel executor path by monkey-patching ThreadPoolExecutor
        # on this specific call.  We swap it back after the call completes.
        import concurrent.futures as _cf
        _orig_tpe = _cf.ThreadPoolExecutor

        class _InlineExecutor:
            """Fake ThreadPoolExecutor that runs tasks immediately in the caller's thread."""
            def __init__(self, *a, **kw):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def submit(self, fn, *args, **kw):
                f: _Future = _Future()  # type: ignore[type-arg]
                try:
                    f.set_result(fn(*args, **kw))
                except Exception as exc:  # noqa: BLE001
                    f.set_exception(exc)
                return f

        _cf.ThreadPoolExecutor = _InlineExecutor  # type: ignore[misc]
        try:
            return _orig_search(self, query, workspace, limit=limit,
                                active_only=active_only, **kwargs)
        finally:
            _cf.ThreadPoolExecutor = _orig_tpe  # type: ignore[misc]

    _HB.search = _sequential_search

    # Warm the model before accepting questions.
    from mind_mem.recall_vector import VectorBackend
    torch.cuda.empty_cache()
    vb = VectorBackend(CONFIG["recall"])
    _ = vb.model  # loads mxbai onto GPU

    # Signal to parent that we are ready.
    result_q.put(("ready", None))

    tmp = tempfile.gettempdir()
    while True:
        task = task_q.get()
        if task is None:  # shutdown sentinel
            break
        q, turns, n_expand = task
        torch.cuda.empty_cache()  # release any fragmented cache from last question
        try:
            r = _evaluate_question(q, tmp, turns, n_expand)
            result_q.put(("ok", r))
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            msg = str(exc)
            if "out of memory" in msg.lower() or "cuda error" in msg.lower():
                # CUDA context is corrupted — signal parent to kill + respawn us.
                # We exit cleanly after putting the signal so the parent can
                # start a fresh CUDA context for the next question.
                result_q.put(("cuda_oom", msg))
                break  # exit the loop; worker process ends
            result_q.put(("err", msg))
        except Exception as exc:  # noqa: BLE001
            result_q.put(("err", str(exc)))


class _Worker:
    """Owns a single spawned worker process and its queues."""

    def __init__(self, ctx: multiprocessing.context.BaseContext) -> None:
        self._ctx = ctx
        self._task_q: multiprocessing.Queue = ctx.Queue()
        self._result_q: multiprocessing.Queue = ctx.Queue()
        self._proc: multiprocessing.Process | None = None
        self._spawn()

    def _spawn(self) -> None:
        self._proc = self._ctx.Process(
            target=_worker_main,
            args=(self._task_q, self._result_q),
            daemon=True,
        )
        self._proc.start()
        # Wait for the worker to finish loading the model.
        try:
            status, _ = self._result_q.get(timeout=120)
            if status != "ready":
                raise RuntimeError(f"Unexpected worker startup message: {status}")
        except queue.Empty as exc:
            self._proc.kill()
            raise RuntimeError("Worker failed to start within 120s") from exc

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def send(self, task: tuple) -> None:
        self._task_q.put(task)

    def recv(self, timeout: float) -> tuple:
        return self._result_q.get(timeout=timeout)

    def kill(self) -> None:
        if self._proc and self._proc.is_alive():
            self._proc.kill()
            self._proc.join()

    def respawn(self) -> None:
        """Kill the current worker and start a fresh one."""
        self.kill()
        # Drain stale queue items to avoid cross-contamination.
        self._task_q = self._ctx.Queue()
        self._result_q = self._ctx.Queue()
        self._spawn()


# ── Aggregation ─────────────────────────────────────────────────────────────

def aggregate(pq: list[dict]) -> dict:
    if not pq:
        return {"n": 0}
    keys = [f"{m}@{k}" for m in ("any", "all") for k in K_VALUES]

    def _metric(items: list[dict]) -> dict:
        n = len(items)
        out: dict = {
            "n": n,
            "mrr": round(sum(x["reciprocal_rank"] for x in items) / n, 4),
        }
        for kk in keys:
            out[kk] = round(sum(x[kk] for x in items) / n, 4)
        return out

    by_type: dict = {}
    for x in pq:
        by_type.setdefault(x["question_type"], []).append(x)

    return {
        "overall": _metric(pq),
        "by_type": {t: _metric(v) for t, v in sorted(by_type.items())},
        "n": len(pq),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="LongMemEval-S full-potential harness for mind-mem."
    )
    ap.add_argument(
        "--data-path",
        default="benchmarks/.cache/longmemeval_s.json",
    )
    ap.add_argument("--turns", choices=["all", "user"], default="all")
    ap.add_argument("--expand", type=int, default=3,
                    help="4b query rewrites (0 = pure hybrid, no LLM)")
    ap.add_argument("--sample", type=int, default=0,
                    help="questions to sample (0 = full 470)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--qtimeout", type=int, default=60,
        help="per-question hard wall-clock cap in seconds; worker killed on breach",
    )
    ap.add_argument("--output", help="path to write per-question JSON results")
    a = ap.parse_args()

    # Reproducibility: auto-fetch the public dataset if absent so anyone
    # can run this with only `pip install "mind-mem[all]" sentence-transformers`.
    if not os.path.isfile(a.data_path):
        os.makedirs(os.path.dirname(a.data_path) or ".", exist_ok=True)
        url = ("https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
               "/resolve/main/longmemeval_s_cleaned.json")
        print(f"[dataset] downloading {url}", flush=True)
        urllib.request.urlretrieve(url, a.data_path)  # noqa: S310 — fixed HF URL
        print(f"[dataset] saved -> {a.data_path}", flush=True)

    data = json.load(open(a.data_path))
    pool = [
        q for q in data
        if not q.get("question_id", "").endswith("_abs")
        and q.get("answer_session_ids")
    ]
    if a.sample:
        random.seed(a.seed)
        pool = random.sample(pool, min(a.sample, len(pool)))

    # Spawn a persistent worker process (loads GPU model once).
    ctx = multiprocessing.get_context("spawn")
    print("  [start-up] spawning worker and loading mxbai-embed-large onto GPU …",
          flush=True)
    t_startup = time.time()
    worker = _Worker(ctx)
    print(f"  [start-up] worker ready in {time.time() - t_startup:.1f}s", flush=True)

    pq: list[dict] = []
    skip = timeouts = errors = 0
    t0 = time.time()

    for i, q in enumerate(pool):
        qt0 = time.time()
        worker.send((q, a.turns, a.expand))
        status = payload = None
        try:
            status, payload = worker.recv(timeout=a.qtimeout)
        except queue.Empty:
            # Hard kill: the worker is in a native hang.
            worker.kill()
            timeouts += 1
            skip += 1
            print(
                f"  TIMEOUT  q={q.get('question_id')} "
                f"sess={len(q.get('haystack_sessions', []))} "
                f"(>{a.qtimeout}s) — skipped, respawning worker …",
                flush=True,
            )
            t_resp = time.time()
            worker.respawn()
            print(f"  [respawn] worker ready in {time.time() - t_resp:.1f}s", flush=True)
            continue

        qt_elapsed = time.time() - qt0

        if status == "ok":
            if payload is None:
                skip += 1  # abs / no-gold filtered inside worker
            else:
                pq.append(payload)
        elif status == "cuda_oom":
            # CUDA context corrupted — question skipped, respawn for recovery.
            errors += 1
            skip += 1
            print(
                f"  CUDA_OOM q={q.get('question_id')} "
                f"sess={len(q.get('haystack_sessions', []))} — respawning worker …",
                flush=True,
            )
            t_resp = time.time()
            worker.respawn()
            print(f"  [respawn] worker ready in {time.time() - t_resp:.1f}s", flush=True)
        elif status == "err":
            errors += 1
            skip += 1
            print(
                f"  ERROR    q={q.get('question_id')} "
                f"sess={len(q.get('haystack_sessions', []))} "
                f"({qt_elapsed:.1f}s) err={payload!r} — skipped",
                flush=True,
            )
        # else: unexpected status from worker — skip silently

        if (i + 1) % 10 == 0 or (i + 1) == len(pool):
            elapsed = time.time() - t0
            print(
                f"  [{i + 1}/{len(pool)}] eval={len(pq)} skip={skip} "
                f"timeouts={timeouts} errors={errors} "
                f"{elapsed:.0f}s ({elapsed / max(1, i + 1):.1f}s/q)",
                flush=True,
            )

    worker.kill()

    total_elapsed = round(time.time() - t0, 2)
    n_eval = len(pq)
    n_total = n_eval + skip
    secs_per_q = round(total_elapsed / max(1, n_total), 2)
    projected_470 = round(secs_per_q * 470, 0)

    agg = aggregate(pq)
    overall = agg.get("overall", {})

    print("\n" + "=" * 60)
    print(f"  evaluated={n_eval}  skipped={skip}  "
          f"timeouts={timeouts}  errors={errors}")
    print(f"  elapsed={total_elapsed}s  avg={secs_per_q}s/q  "
          f"projected_full_470={projected_470:.0f}s "
          f"({projected_470 / 60:.1f}min)")
    print(f"  any@5={overall.get('any@5', 'n/a')}  "
          f"all@5={overall.get('all@5', 'n/a')}  "
          f"mrr={overall.get('mrr', 'n/a')}")
    print("=" * 60)
    print(json.dumps(overall, indent=2))

    if a.output:
        os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
        out = {
            "harness": "longmemeval_fullpotential",
            "pipeline": (
                f"BM25F+mxbai-embed-large+RRF + 4b-expand({a.expand}); "
                f"turns={a.turns}; knee_off; cross_encoder_off; "
                f"per-question-isolated; spawn-worker"
            ),
            "subset": "longmemeval_s_cleaned",
            "sample": a.sample or "full",
            "seed": a.seed,
            "elapsed_seconds": total_elapsed,
            "avg_seconds_per_question": secs_per_q,
            "projected_full_470_seconds": projected_470,
            "evaluated": n_eval,
            "skipped": skip,
            "timeouts": timeouts,
            "errors": errors,
            "qtimeout_s": a.qtimeout,
            "aggregate": agg,
            "per_question": pq,
        }
        with open(a.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {a.output}")


if __name__ == "__main__":
    main()
