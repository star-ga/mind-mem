#!/usr/bin/env python3
"""LongMemEval-S consolidation harness — one loop, any adapter, self-asserting.

This does **not** replace the ~dozen ``benchmarks/longmemeval_*.py`` scripts;
it consolidates their retrieval-eval core behind the
:mod:`mind_mem.bench.eval_adapter` contract so a single scorer drives the
BM25 baseline and the real mind-mem recall path identically, and every run
records the pipeline it actually exercised (config hash + backend probe) in
the NDJSON artifact and the scorecard.

Dataset: **LongMemEval-S** (public, ~500 questions). It is NOT redistributed
here. Point ``--data-path`` at a local copy, or set ``LONGMEMEVAL_DATA``.
Absent → a clear FileNotFoundError, never a silent empty run.

Honesty rails (the reason this harness exists):
  * both recall protocols (``recall_any@k`` AND ``recall_all@k``) at both
    turn granularities are always computed;
  * the scorecard states that the prior published R@5=85.3 is UNREPRODUCED
    (see ``benchmarks/LONGMEMEVAL_FINDINGS_2026-05-19.md``) and prints NO new
    competitor comparison;
  * only numbers this harness measured go in the results table;
  * a declared-vs-effective backend mismatch is flagged, not hidden.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

from .eval_adapter import PipelineProbe, SessionDoc
from .eval_adapters import get_adapter
from .eval_scorer import K_VALUES, QuestionScore, aggregate, score_question

DEFAULT_DATA_PATH = "benchmarks/.cache/longmemeval_s.json"
FINDINGS_REF = "benchmarks/LONGMEMEVAL_FINDINGS_2026-05-19.md"


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the LongMemEval-S dataset is not present locally."""


def resolve_data_path(explicit: str | None = None) -> str:
    """Resolve the dataset path: explicit arg > env > default. Fail clear."""
    path = explicit or os.environ.get("LONGMEMEVAL_DATA") or DEFAULT_DATA_PATH
    if not os.path.isfile(path):
        raise DatasetNotFoundError(
            f"LongMemEval-S dataset not found at {path!r}. "
            "This harness does not redistribute the dataset. Obtain the public "
            "LongMemEval-S JSON, place it at that path (or set LONGMEMEVAL_DATA / "
            "pass --data-path), then re-run."
        )
    return path


def load_dataset(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list of questions in {path!r}, got {type(data).__name__}")
    return data


def stratified_sample(pool: list[dict[str, Any]], per_type: int, seed: int) -> list[dict[str, Any]]:
    """Up to ``per_type`` questions per ``question_type``, seeded for determinism.

    Stratified coverage so every question_type is represented in a small
    sample — a uniform random draw of the same size can miss a rare type
    (e.g. single-session-preference is ~6% of the LongMemEval-S pool). This
    controls *which* questions run, never how they are scored; the scorecard
    discloses that the number is a stratified sample, not the full set.
    """
    import random

    by_type: dict[str, list[dict[str, Any]]] = {}
    for q in pool:
        by_type.setdefault(str(q.get("question_type", "unknown")), []).append(q)
    # Deterministic seeded stratified sampling for reproducible benchmark selection (not a crypto/security context).
    rng = random.Random(seed)  # nosec B311
    picked: list[dict[str, Any]] = []
    for qtype in sorted(by_type):
        group = sorted(by_type[qtype], key=lambda q: str(q.get("question_id", "")))
        rng.shuffle(group)
        picked.extend(group[:per_type])
    return picked


def build_session_docs(question: dict[str, Any], turns: str) -> list[SessionDoc]:
    """One SessionDoc per haystack session. ``turns`` = 'all' | 'user'."""
    sessions = question.get("haystack_sessions", [])
    sids = question.get("haystack_session_ids", list(range(len(sessions))))
    docs: list[SessionDoc] = []
    for i, session in enumerate(sessions):
        sid = str(sids[i]) if i < len(sids) else str(i)
        parts: list[str] = []
        for turn in session:
            role = turn.get("role", "unknown")
            if turns == "user" and role != "user":
                continue
            content = turn.get("content", "")
            if content:
                parts.append(f"({role}) {content}")
        docs.append(SessionDoc(doc_id=sid, text=" ".join(parts)))
    return docs


@dataclass
class SuiteResult:
    adapter: str
    turns: str
    scores: list[QuestionScore]
    probes: list[PipelineProbe]
    evaluated: int
    skipped: int
    elapsed_s: float
    #: Denominator accounting. ``evaluated`` is a recall number's
    #: denominator, so every question that left the dataset on the way to
    #: it has to be counted somewhere — otherwise the scorecard prints
    #: "full set" beside a smaller N and nothing reconciles the two.
    #: Defaulted so a hand-built SuiteResult (and the LoCoMo driver,
    #: which shares this dataclass) stays valid.
    dataset_size: int = 0
    eligible: int = 0
    excluded_abstention: int = 0
    excluded_no_gold: int = 0

    @property
    def any_mismatch(self) -> bool:
        return any(p.mismatch for p in self.probes)

    def representative_probe(self) -> PipelineProbe | None:
        return self.probes[0] if self.probes else None


def run_suite(
    adapter_name: str,
    dataset: list[dict[str, Any]],
    *,
    k: int = 10,
    turns: str = "all",
    config: dict[str, Any] | None = None,
    sample: int = 0,
    per_type: int = 0,
    seed: int = 42,
    k_values: tuple[int, ...] = K_VALUES,
    progress: bool = False,
) -> SuiteResult:
    """Drive one adapter across the dataset; return per-question scores + probes.

    ``per_type`` (stratified: up to N per question_type) takes precedence over
    ``sample`` (uniform random of N); either narrows *which* questions run
    without touching how they are scored.
    """
    adapter = get_adapter(adapter_name)

    # Eligibility filter, counted. An abstention question (``*_abs``)
    # names no gold session and a question with no ``answer_session_ids``
    # has nothing to retrieve, so recall@k is undefined for both and they
    # are excluded on the merits — but excluding them silently moved the
    # denominator of a published number with no record of by how much.
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
    eligible = len(pool)
    if per_type > 0:
        pool = stratified_sample(pool, per_type, seed)
    elif sample and sample < len(pool):
        import random

        random.seed(seed)
        # Deterministic seeded sampling for reproducible benchmark selection (not a crypto/security context).
        pool = random.sample(pool, sample)  # nosec B311

    scores: list[QuestionScore] = []
    probes: list[PipelineProbe] = []
    skipped = 0
    t0 = time.time()

    for i, q in enumerate(pool):
        query = q.get("question", "")
        # ``gold`` is non-empty by construction — the eligibility filter
        # above already dropped every question without answer_session_ids,
        # so only a blank question string can be skipped here.
        gold = {str(g) for g in q.get("answer_session_ids", [])}
        if not query:
            skipped += 1
            continue
        docs = build_session_docs(q, turns)
        state = adapter.init(docs, config)
        probes.append(state.probe)
        try:
            t_q = time.monotonic()
            hits = adapter.query(query, state, k)
            latency_ms = (time.monotonic() - t_q) * 1000.0
        finally:
            adapter.teardown(state)
        retrieved = [str(h["doc_id"]) for h in hits]
        scores.append(
            score_question(
                question_id=str(q.get("question_id", f"q{i}")),
                question_type=str(q.get("question_type", "unknown")),
                retrieved_doc_ids=retrieved,
                gold_doc_ids=gold,
                latency_ms=latency_ms,
                k_values=k_values,
            )
        )
        if progress and ((i + 1) % 25 == 0 or (i + 1) == len(pool)):
            print(f"  [{i + 1}/{len(pool)}] scored={len(scores)} skip={skipped} {time.time() - t0:.0f}s", flush=True)

    return SuiteResult(
        adapter=adapter_name,
        turns=turns,
        scores=scores,
        probes=probes,
        evaluated=len(scores),
        skipped=skipped,
        elapsed_s=round(time.time() - t0, 2),
        dataset_size=len(dataset),
        eligible=eligible,
        excluded_abstention=excluded_abstention,
        excluded_no_gold=excluded_no_gold,
    )


def write_ndjson(result: SuiteResult, path: str) -> None:
    """One JSON object per question; each row carries the pipeline probe."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qs, probe in zip(result.scores, result.probes):
            row = qs.to_ndjson_row()
            row["adapter"] = result.adapter
            row["turns"] = result.turns
            row["pipeline"] = probe.to_dict()
            f.write(json.dumps(row, sort_keys=True) + "\n")


def render_scorecard(result: SuiteResult, *, dataset_path: str, k: int, embedder: str, sampling: str = "full set") -> str:
    """Render the Markdown scorecard (measured numbers + honesty rails)."""
    agg = aggregate(result.scores)
    probe = result.representative_probe()
    today = date.today().isoformat()
    lines: list[str] = []
    lines.append(f"# LongMemEval-S scorecard — `{result.adapter}` ({today})")
    lines.append("")
    lines.append("## Disclosure (what was actually measured)")
    lines.append("")
    lines.append(f"- **Adapter:** `{result.adapter}`")
    lines.append(f"- **Sampling:** {sampling}")
    if probe is not None:
        lines.append(f"- **Declared backend:** `{probe.declared_backend}`")
        lines.append(f"- **Effective backend (probed):** `{probe.effective_backend}`")
        lines.append(f"- **Vector deps available:** `{probe.vector_available}`")
        lines.append(f"- **Config SHA-256 (16):** `{probe.config_sha256}`")
        if probe.notes:
            lines.append(f"- **Probe notes:** {probe.notes}")
    lines.append(f"- **Embedder:** {embedder}")
    lines.append(f"- **k (retrieval depth):** {k}")
    lines.append("- **Token budget:** whole-session document, untruncated at ingest")
    lines.append(f"- **Dataset:** LongMemEval-S (`{os.path.basename(dataset_path)}`), turns=`{result.turns}`")
    if result.dataset_size:
        lines.append(f"- **Dataset questions:** {result.dataset_size}")
        lines.append(
            f"- **Excluded before scoring:** {result.excluded_abstention} abstention (`*_abs`) "
            f"+ {result.excluded_no_gold} without gold session ids → {result.eligible} eligible"
        )
    lines.append(f"- **Questions evaluated:** {result.evaluated} (skipped {result.skipped})")
    lines.append(f"- **Wall clock:** {result.elapsed_s}s")
    lines.append(f"- **Hardware:** {platform.machine()} / {platform.system()} / py{platform.python_version()}")
    lines.append("")

    if result.any_mismatch:
        lines.append("> ⚠️ **PIPELINE MISMATCH:** at least one question ran a backend other than the")
        lines.append("> declared one. These numbers do NOT measure the declared pipeline. Investigate")
        lines.append("> before citing (see the `pipeline` block in the NDJSON artifact).")
        lines.append("")

    lines.append("## Measured results (this harness only)")
    lines.append("")
    if result.scores:
        o = agg["overall"]
        lines.append("| metric | @1 | @3 | @5 | @10 |")
        lines.append("|---|---|---|---|---|")
        for name in ("recall_any", "recall_all", "precision", "recall"):
            row = " | ".join(f"{o.get(f'{name}@{kk}', '—')}" for kk in (1, 3, 5, 10))
            lines.append(f"| {name}@k | {row} |")
        lines.append("")
        lines.append(f"- **MRR:** {o['mrr']} · **hit_rate:** {o['hit_rate']} · **mean latency:** {o['mean_latency_ms']} ms")
        lines.append("")
        lines.append("### By question type (recall_any@5 / recall_all@5)")
        lines.append("")
        lines.append("| type | n | any@5 | all@5 | mrr |")
        lines.append("|---|---|---|---|---|")
        for t, m in agg["by_type"].items():
            lines.append(f"| {t} | {m['n']} | {m['recall_any@5']} | {m['recall_all@5']} | {m['mrr']} |")
    else:
        lines.append("_No questions scored._")
    lines.append("")

    lines.append("## Honesty rails")
    lines.append("")
    lines.append(
        "- **Both protocols reported.** `recall_any@k` (≥1 gold session in top-k) and the "
        "stricter official `recall_all@k` (all gold sessions in top-k) are shown side by side; "
        "neither is cherry-picked."
    )
    lines.append(
        f"- **Prior published R@5 = 85.3 is RETRACTED** (2026-09-04, commit `129cbf3`), not merely "
        f"unreproduced. Per `{FINDINGS_REF}` it entered the repo with no committed "
        "artifact or methodology, survived two failed reproduction attempts, and its per-category "
        "rows sum to 376 under a stated Overall N of 470. It has been replaced by the measured run "
        "published in `benchmarks/REPORT.md`; this scorecard reports only what this run measured."
    )
    lines.append(
        "- **Competitor comparisons are permitted**, gated on the ordinary requirements: same box, "
        "same dataset, same protocol, >=2 reps, committed artifacts. They are no longer gated on "
        "85.3. Blocking them on an unreproducible figure lowered the ceiling instead of raising "
        "the code."
    )
    lines.append(
        "- **Self-asserting pipeline.** Every NDJSON row carries a `pipeline` probe "
        "(declared/effective backend + config hash) so a config-less fallback can never be "
        "reported as the full stack (the exact false-green in the FINDINGS)."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LongMemEval-S consolidation harness")
    ap.add_argument("--adapter", default="mind_mem", help="bm25_baseline | mind_mem")
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--turns", choices=["all", "user"], default="all")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sample", type=int, default=0, help="uniform random sample; 0 = full set")
    ap.add_argument(
        "--per-type",
        type=int,
        default=0,
        dest="per_type",
        help="stratified: up to N questions per question_type (0 = off; takes precedence over --sample)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ndjson", default=None, help="NDJSON artifact output path")
    ap.add_argument("--scorecard", default=None, help="Markdown scorecard output path")
    ap.add_argument("--embedder", default="none (BM25-only)")
    a = ap.parse_args(argv)

    data_path = resolve_data_path(a.data_path)
    dataset = load_dataset(data_path)
    result = run_suite(
        a.adapter,
        dataset,
        k=a.k,
        turns=a.turns,
        sample=a.sample,
        per_type=a.per_type,
        seed=a.seed,
        progress=True,
    )

    if a.per_type > 0:
        sampling = (
            f"STRATIFIED SAMPLE — up to {a.per_type} questions per question_type "
            f"(seed {a.seed}); {result.evaluated} evaluated. FULL 500-question set DEFERRED."
        )
    elif a.sample > 0:
        sampling = f"uniform random sample of {a.sample} (seed {a.seed}); full set DEFERRED"
    else:
        sampling = "full set"

    ndjson_path = a.ndjson or f"benchmarks/.cache/{date.today().isoformat()}-longmemeval-s-{a.adapter}.ndjson"
    write_ndjson(result, ndjson_path)
    scorecard = render_scorecard(result, dataset_path=data_path, k=a.k, embedder=a.embedder, sampling=sampling)
    scorecard_path = a.scorecard or f"docs/benchmarks/{date.today().isoformat()}-longmemeval-s-{a.adapter}.md"
    os.makedirs(os.path.dirname(os.path.abspath(scorecard_path)), exist_ok=True)
    with open(scorecard_path, "w", encoding="utf-8") as f:
        f.write(scorecard)

    print(json.dumps(aggregate(result.scores).get("overall", {"n": 0}), indent=2))
    print(f"\nNDJSON:    {ndjson_path}")
    print(f"Scorecard: {scorecard_path}")
    if result.any_mismatch:
        print("\nWARNING: pipeline mismatch detected — see scorecard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
