#!/usr/bin/env python3
"""Full-corpus LongMemEval-S driver: subprocess-isolated, resumable.

The consolidation harness (:mod:`mind_mem.bench.longmemeval_suite`) already
owns everything that decides a number -- the eligibility filter, the adapter
contract, the dual-protocol scorer, the scorecard renderer. It is imported and
reused here, unchanged. What it does not own is *survival over 500 questions*,
and ``LONGMEMEVAL_FINDINGS_2026-05-19.md`` lists the two reasons the full run
never landed:

    1. Subprocess-isolated harness: one child process per question with a
       hard wall-clock kill (preempts native hangs #6).
    2. Cross-encoder singleton (or verified-off) ... (#5)

(2) is fixed in the product (``mind_mem.rerank_ensemble`` /
``mind_mem.cross_encoder_reranker`` now cache weights per ``(model, device)``,
with a load counter a regression test reads). This module is (1).

What it adds, and nothing more:

* **Isolation.** Each question runs in its own process, killed by ``SIGKILL``
  on its whole process group when it overruns. See ``benchmarks/hard_timeout``
  for why ``signal.alarm`` cannot do this.
* **Resumability.** Rows are appended to the NDJSON as they are scored, and a
  restart skips question ids already present. A 500-question run that dies at
  400 does not start over -- and, more to the point, does not tempt anyone into
  publishing the 400.
* **Timeout accounting.** A question that is killed is scored as a MISS (empty
  retrieval) and counted separately in the disclosure. Scoring it as a miss is
  the conservative direction for a claim about our own retrieval; the count is
  published beside the number so the exclude-timeouts variant can be computed
  by anyone who prefers it. What is never done is dropping them silently.

This does not lift the provenance hold on ``R@5 = 85.3``. It produces a
measured number with an artifact; whether that number replaces anything is a
separate decision, and it needs >=2 reps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Run as a script, only ``benchmarks/`` is on the path; both the package
# under ``src/`` and this file's own package have to be importable, and
# they must be importable in the CHILD too -- ``spawn`` re-imports by name.
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks.hard_timeout import OK, TIMEOUT, run_with_hard_timeout  # noqa: E402
from mind_mem.bench.eval_scorer import K_VALUES, QuestionScore, aggregate, score_question  # noqa: E402
from mind_mem.bench.longmemeval_suite import (  # noqa: E402
    FINDINGS_REF,
    SuiteResult,
    build_session_docs,
    load_dataset,
    render_scorecard,
    resolve_data_path,
)

#: Default per-question wall-clock ceiling. Generous for a 48-session
#: haystack on a shared box; a question that needs more than this is the
#: pathological case the isolation exists for.
DEFAULT_QTIMEOUT_S = 180.0


def eligible_questions(dataset: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    """The pool ``run_suite`` scores, and why the rest left.

    Mirrors :func:`mind_mem.bench.longmemeval_suite.run_suite`'s inline
    filter: abstention questions name no gold session and a question with no
    ``answer_session_ids`` has nothing to retrieve, so recall@k is undefined
    for both. ``tests/test_longmemeval_full_run.py`` asserts this returns the
    same pool the canonical harness evaluates -- a duplicated filter that
    drifts is a denominator that drifts.
    """
    pool: list[dict[str, Any]] = []
    excluded_abstention = 0
    excluded_no_gold = 0
    for q in dataset:
        if str(q.get("question_id", "")).endswith("_abs"):
            excluded_abstention += 1
            continue
        if not q.get("answer_session_ids"):
            excluded_no_gold += 1
            continue
        pool.append(q)
    return pool, excluded_abstention, excluded_no_gold


def score_one_question(question: dict[str, Any], adapter_name: str, turns: str, k: int, config: dict[str, Any] | None) -> dict[str, Any]:
    """Retrieve for one question. Runs in the CHILD process.

    Returns a plain dict so the result crosses the process boundary without
    dragging adapter state with it. Module-level, because ``spawn`` imports
    the target by name.
    """
    from mind_mem.bench.eval_adapters import get_adapter

    adapter = get_adapter(adapter_name)
    docs = build_session_docs(question, turns)
    state = adapter.init(docs, config)
    probe = state.probe.to_dict()
    try:
        t0 = time.monotonic()
        hits = adapter.query(question.get("question", ""), state, k)
        latency_ms = (time.monotonic() - t0) * 1000.0
    finally:
        adapter.teardown(state)
    return {
        "retrieved": [str(h["doc_id"]) for h in hits],
        "latency_ms": latency_ms,
        "probe": probe,
    }


def _load_done(path: str) -> set[str]:
    """Question ids already scored in an existing artifact."""
    done: set[str] = set()
    if not os.path.isfile(path):
        return done
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line)["question_id"]))
            except (ValueError, KeyError):
                continue
    return done


def _row_to_score(row: dict[str, Any]) -> QuestionScore:
    """Rebuild a QuestionScore from a persisted row, for resumed runs."""
    return QuestionScore(
        question_id=str(row["question_id"]),
        question_type=str(row["question_type"]),
        n_gold=int(row["n_gold"]),
        n_retrieved=int(row["n_retrieved"]),
        latency_ms=float(row["latency_ms"]),
        first_gold_rank=row["first_gold_rank"],
        reciprocal_rank=float(row["reciprocal_rank"]),
        precision_at_k={int(kk): float(v) for kk, v in row["precision_at_k"].items()},
        recall_at_k={int(kk): float(v) for kk, v in row["recall_at_k"].items()},
        recall_any_at_k={int(kk): int(v) for kk, v in row["recall_any_at_k"].items()},
        recall_all_at_k={int(kk): int(v) for kk, v in row["recall_all_at_k"].items()},
        hit=bool(row["hit"]),
    )


def _replay(path: str) -> tuple[list[QuestionScore], list[dict[str, Any]], int]:
    """Read back an artifact: scores, probes, and how many timed out."""
    scores: list[QuestionScore] = []
    probes: list[dict[str, Any]] = []
    timed_out = 0
    if not os.path.isfile(path):
        return scores, probes, timed_out
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            scores.append(_row_to_score(row))
            probes.append(row.get("pipeline", {}))
            if row.get("unit_status") == TIMEOUT:
                timed_out += 1
    return scores, probes, timed_out


def run(
    adapter_name: str,
    dataset: list[dict[str, Any]],
    ndjson_path: str,
    *,
    k: int = 10,
    turns: str = "all",
    config: dict[str, Any] | None = None,
    qtimeout_s: float = DEFAULT_QTIMEOUT_S,
    limit: int = 0,
    progress_every: int = 10,
) -> dict[str, Any]:
    """Score the eligible pool, one hard-killed child process per question."""
    pool, excl_abs, excl_no_gold = eligible_questions(dataset)
    if limit:
        pool = pool[:limit]
    done = _load_done(ndjson_path)
    os.makedirs(os.path.dirname(os.path.abspath(ndjson_path)), exist_ok=True)

    t0 = time.time()
    attempted = 0
    with open(ndjson_path, "a", encoding="utf-8") as sink:
        for i, q in enumerate(pool):
            qid = str(q.get("question_id", f"q{i}"))
            if qid in done:
                continue
            query = q.get("question", "")
            if not query:
                continue
            attempted += 1
            outcome = run_with_hard_timeout(score_one_question, qtimeout_s, q, adapter_name, turns, k, config)
            if outcome.status == OK and isinstance(outcome.value, dict):
                retrieved = list(outcome.value["retrieved"])
                latency_ms = float(outcome.value["latency_ms"])
                probe = outcome.value["probe"]
            else:
                # Killed, crashed or raised. Scored as a MISS -- the
                # conservative direction -- and labelled, never dropped.
                retrieved = []
                latency_ms = outcome.elapsed_s * 1000.0
                probe = {"note": f"unit_{outcome.status}", "error": outcome.error[:400]}
            score = score_question(
                question_id=qid,
                question_type=str(q.get("question_type", "unknown")),
                retrieved_doc_ids=retrieved,
                gold_doc_ids={str(g) for g in q.get("answer_session_ids", [])},
                latency_ms=latency_ms,
                k_values=K_VALUES,
            )
            row = score.to_ndjson_row()
            row["adapter"] = adapter_name
            row["turns"] = turns
            row["pipeline"] = probe
            row["unit_status"] = outcome.status
            row["unit_elapsed_s"] = round(outcome.elapsed_s, 3)
            sink.write(json.dumps(row, sort_keys=True) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            if progress_every and attempted % progress_every == 0:
                print(f"  [{attempted}] last={outcome.status} {time.time() - t0:.0f}s", flush=True)

    return {
        "eligible": len(pool),
        "attempted_this_pass": attempted,
        "dataset_size": len(dataset),
        "excluded_abstention": excl_abs,
        "excluded_no_gold": excl_no_gold,
        "elapsed_s": round(time.time() - t0, 2),
    }


def embedder_disclosure(config: dict[str, Any] | None, probe_rows: list[dict[str, Any]]) -> str:
    """Whether the vector leg was actually EXERCISED — measured, not labelled.

    The scorecard's ``Embedder:`` line is free text a caller types on the
    command line. It is not evidence of anything: a run can print
    ``Embedder: mxbai`` while the dense leg never executes. The probe is
    closer but still not the answer — ``vector_available`` records that
    ``sentence_transformers`` *imports*, which on this box is ``True`` even
    for a pure-BM25 run. Reporting deps-present next to a lexical-only number
    is exactly the declared-vs-effective confusion the probe exists to kill.

    So this renders three separate facts and never collapses them:

    * deps importable (from the probe),
    * whether a vector leg was **configured** for this run (from the config
      that was actually handed to the adapter, not from a CLI string),
    * the effective backend the probe reconciled against disk.

    It is derived from config + probe rather than instrumented inside
    ``recall``; the line says so, so nobody upgrades it to a stronger claim
    than it is.
    """
    recall_cfg = (config or {}).get("recall", {}) if isinstance(config, dict) else {}
    recall_cfg = recall_cfg if isinstance(recall_cfg, dict) else {}
    vector_cfg = recall_cfg.get("vector", {})
    vector_cfg = vector_cfg if isinstance(vector_cfg, dict) else {}
    backend = str(recall_cfg.get("backend", "scan"))
    configured = bool(vector_cfg.get("enabled", False)) or backend in {"hybrid", "vector"}

    deps = sorted({bool(p.get("vector_available", False)) for p in probe_rows})
    backends = sorted({str(p.get("effective_backend", "unknown")) for p in probe_rows})

    lines = ["", "## Retrieval legs actually exercised", ""]
    deps_shown = deps if len(deps) != 1 else deps[0]
    lines.append(f"- **Vector deps importable:** `{deps_shown}` — this is a *dependency* fact, not a pipeline fact.")
    lines.append(
        f"- **Vector leg exercised:** `{configured}` "
        + (
            "— a dense/hybrid leg was configured for this run."
            if configured
            else "— **no** dense leg was configured, so this number is lexical-only regardless of what the `Embedder:` line above says."
        )
    )
    lines.append(f"- **Effective backend(s) probed:** `{', '.join(backends) if backends else 'unknown'}`")
    lines.append(f"- **Effective embedder:** `{'see config' if configured else 'none — BM25F lexical only'}`")
    lines.append(
        "- *Basis:* derived from the config handed to the adapter plus the per-question "
        "pipeline probe. It is not instrumentation inside `recall`, and is not evidence "
        "that a configured leg produced every score."
    )
    lines.append("")
    return "\n".join(lines)


def timeout_disclosure(timed_out: int, evaluated: int, qtimeout_s: float) -> str:
    """The paragraph a reader needs to interpret the number above it."""
    lines = ["", "## Unit isolation and timeouts", ""]
    lines.append(
        f"- Each question ran in its own process with a hard `SIGKILL` at "
        f"{qtimeout_s:g}s (process group included). `signal.alarm` cannot preempt a "
        f"native stage, which is why the earlier harness could not skip pathological "
        f"haystacks — see `{FINDINGS_REF}`."
    )
    lines.append(
        f"- **Questions killed or crashed: {timed_out} / {evaluated}.** Each is scored as a "
        "MISS (empty retrieval), the conservative direction for a claim about our own "
        "retrieval. The count is published so the exclude-timeouts variant can be "
        "computed; it is never dropped silently."
    )
    lines.append(
        "- Rows are appended and fsynced per question, so this run is resumable: "
        "re-running with the same `--ndjson` skips question ids already present."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LongMemEval-S full-corpus driver (subprocess-isolated, resumable)")
    ap.add_argument("--adapter", default="mind_mem", help="bm25_baseline | mind_mem")
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--turns", choices=["all", "user"], default="all")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--qtimeout", type=float, default=DEFAULT_QTIMEOUT_S, help="per-question hard kill, seconds")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N eligible questions (0 = all)")
    ap.add_argument("--rep", type=int, default=1, help="repetition index; only labels the artifact")
    ap.add_argument("--ndjson", default=None)
    ap.add_argument("--scorecard", default=None)
    ap.add_argument("--embedder", default="none (BM25-only)")
    ap.add_argument("--config", default=None, help="JSON file with the adapter config to run under")
    a = ap.parse_args(argv)

    data_path = resolve_data_path(a.data_path)
    dataset = load_dataset(data_path)
    config = None
    if a.config:
        with open(a.config, encoding="utf-8") as handle:
            config = json.load(handle)

    stamp = date.today().isoformat()
    ndjson_path = a.ndjson or f"benchmarks/.cache/{stamp}-longmemeval-s-full-{a.adapter}-rep{a.rep}.ndjson"
    stats = run(
        a.adapter,
        dataset,
        ndjson_path,
        k=a.k,
        turns=a.turns,
        config=config,
        qtimeout_s=a.qtimeout,
        limit=a.limit,
    )

    scores, probe_rows, timed_out = _replay(ndjson_path)
    from mind_mem.bench.eval_adapter import PipelineProbe

    probes = [
        PipelineProbe(
            adapter=a.adapter,
            declared_backend=str(p.get("declared_backend", "unknown")),
            effective_backend=str(p.get("effective_backend", "unknown")),
            vector_available=bool(p.get("vector_available", False)),
            config_sha256=str(p.get("config_sha256", "")),
            notes=str(p.get("notes", "")),
        )
        for p in probe_rows
    ]
    result = SuiteResult(
        adapter=a.adapter,
        turns=a.turns,
        scores=scores,
        probes=probes,
        evaluated=len(scores),
        skipped=0,
        elapsed_s=stats["elapsed_s"],
        dataset_size=stats["dataset_size"],
        eligible=stats["eligible"],
        excluded_abstention=stats["excluded_abstention"],
        excluded_no_gold=stats["excluded_no_gold"],
    )
    sampling = (
        f"FULL eligible set (rep {a.rep}); {len(scores)} of {stats['eligible']} eligible scored"
        if not a.limit
        else f"first {a.limit} eligible questions (rep {a.rep}) — NOT the full set"
    )
    scorecard = render_scorecard(result, dataset_path=data_path, k=a.k, embedder=a.embedder, sampling=sampling)
    scorecard += embedder_disclosure(config, probe_rows)
    scorecard += timeout_disclosure(timed_out, len(scores), a.qtimeout)
    scorecard_path = a.scorecard or f"docs/benchmarks/{stamp}-longmemeval-s-full-{a.adapter}-rep{a.rep}.md"
    os.makedirs(os.path.dirname(os.path.abspath(scorecard_path)), exist_ok=True)
    with open(scorecard_path, "w", encoding="utf-8") as handle:
        handle.write(scorecard)

    print(json.dumps({"stats": stats, "scored_total": len(scores), "unit_timeouts": timed_out}, indent=2))
    print(json.dumps(aggregate(scores).get("overall", {"n": 0}), indent=2))
    print(f"\nNDJSON:    {ndjson_path}")
    print(f"Scorecard: {scorecard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
