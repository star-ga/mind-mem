#!/usr/bin/env python3
"""LoCoMo recall harness — one loop, any adapter, self-asserting.

Sibling of :mod:`mind_mem.bench.longmemeval_suite`. It drives the **same**
pluggable adapter contract, the **same** dual-protocol scorer
(:mod:`mind_mem.bench.eval_scorer`) and the **same** adapter registry
(:mod:`mind_mem.bench.eval_adapters`) — so recall quality on LoCoMo becomes a
*number* produced by exactly one measured code path, next to the BM25 honesty
floor, not a vibe.

LoCoMo (Maharana et al., *Evaluating Very Long-Term Conversational Memory of
LLM Agents*) is a long-term, multi-session, two-speaker conversational
dataset. Its JSON is a list of **samples**; each sample carries a
``conversation`` (``session_1``, ``session_2``, … turn lists) and a ``qa``
list of questions whose ``evidence`` cites dialogue ids of the form
``"D<session>:<turn>"``. Retrieval here is at **session** granularity: a
question's gold set is the set of sessions its evidence points at, and an
adapter must surface those sessions in its top-k.

Dataset: **LoCoMo** (public). It is NOT redistributed here. Point
``--data-path`` at a local copy, or set ``LOCOMO_DATA``. Absent → a clear
:class:`DatasetNotFoundError`, never a silent empty run.

Honesty rails (this harness is the honesty item):
  * both recall protocols (``recall_any@k`` AND ``recall_all@k``) at every k
    are always computed — neither is cherry-picked;
  * the scorecard discloses the **sample size**, the exercised **backend**
    (declared-vs-effective probe), and **whether the vector/embedder leg was
    exercised** — a config-less fallback can never be reported as the full
    stack;
  * only numbers this harness measured go in the results table;
  * the repo's prior LoCoMo figures are LLM-judge *answer-quality* scores — a
    **different metric** measured by an external judge — and are NOT restated
    as retrieval recall here; no such recall number is reproduced, and NO
    competitor comparison is printed;
  * a declared-vs-effective backend mismatch is flagged, not hidden.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import time
from datetime import date
from typing import Any

from .eval_adapter import SessionDoc
from .eval_adapters import get_adapter
from .eval_scorer import K_VALUES, aggregate, score_question

# Reuse the LongMemEval harness primitives verbatim rather than re-implement
# them: the SuiteResult record, the seeded stratified sampler, and the
# pipeline-carrying NDJSON writer are dataset-shape-agnostic.
from .longmemeval_suite import SuiteResult, stratified_sample, write_ndjson

DEFAULT_DATA_PATH = "benchmarks/.cache/locomo.json"

#: Prior LoCoMo figures recorded in the repo (CLAUDE.md): LLM-judge
#: answer-quality scores — a different metric than retrieval recall, measured
#: by an external judge, not by this harness. Named here only to disclose that
#: they are NOT the numbers this harness produces.
PRIOR_LLM_JUDGE_NOTE = (
    "prior repo LoCoMo figures (mean 77.9 / adversarial 82.3 / temporal 88.5) "
    "are external-LLM-judge answer-quality scores"
)

_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")
#: A dialogue-evidence id ``"D<session>:<turn>"`` → the session number.
_DIA_ID_RE = re.compile(r"[Dd]?(\d+)\s*:")


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the LoCoMo dataset is not present locally."""


def resolve_data_path(explicit: str | None = None) -> str:
    """Resolve the dataset path: explicit arg > env > default. Fail clear."""
    path = explicit or os.environ.get("LOCOMO_DATA") or DEFAULT_DATA_PATH
    if not os.path.isfile(path):
        raise DatasetNotFoundError(
            f"LoCoMo dataset not found at {path!r}. "
            "This harness does not redistribute the dataset. Obtain the public "
            "LoCoMo JSON, place it at that path (or set LOCOMO_DATA / pass "
            "--data-path), then re-run."
        )
    return path


def load_dataset(path: str) -> list[dict[str, Any]]:
    """Load the raw LoCoMo JSON (a list of conversation samples)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list of LoCoMo samples in {path!r}, got {type(data).__name__}")
    return data


def evidence_session_id(evidence_item: Any) -> str | None:
    """Extract the session id from one evidence entry (``"D3:2"`` → ``"3"``).

    Returns ``None`` for an entry no session can be parsed from, so an
    adversarial/answer-less question with empty or opaque evidence contributes
    no phantom gold session.
    """
    s = str(evidence_item)
    m = _DIA_ID_RE.search(s)
    if m:
        return m.group(1)
    if s.isdigit():  # a bare session number
        return s
    return None


def extract_sessions(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull ``session_N`` turn lists out of a LoCoMo ``conversation``, in order.

    Non-session keys (``speaker_a``, ``session_N_date_time``, summaries) are
    ignored. Each returned session is ``{"session_id": "<N>", "turns": [...]}``
    where ``session_id`` matches the number evidence ids point at.
    """
    numbered: list[tuple[int, list[Any]]] = []
    for key, val in conversation.items():
        m = _SESSION_KEY_RE.match(str(key))
        if m and isinstance(val, list):
            numbered.append((int(m.group(1)), val))
    numbered.sort(key=lambda x: x[0])
    return [{"session_id": str(idx), "turns": turns} for idx, turns in numbered]


def flatten_questions(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise nested LoCoMo samples into a flat list of question records.

    Each QA in each sample becomes one record::

        {"question_id", "question", "question_type", "gold_session_ids",
         "sessions"}

    ``sessions`` is the (shared) session list for the parent conversation — the
    haystack every question in that sample retrieves over. ``question_type`` is
    the LoCoMo ``category`` (kept as ``category_<n>``) so the reused stratified
    sampler and per-type aggregation work unchanged.
    """
    questions: list[dict[str, Any]] = []
    for s_idx, sample in enumerate(dataset):
        conversation = sample.get("conversation", {})
        if not isinstance(conversation, dict):
            continue
        sessions = extract_sessions(conversation)
        sample_id = str(sample.get("sample_id", f"sample{s_idx}"))
        for q_idx, qa in enumerate(sample.get("qa", []) or []):
            if not isinstance(qa, dict):
                continue
            gold: set[str] = set()
            for ev in qa.get("evidence", []) or []:
                sid = evidence_session_id(ev)
                if sid is not None:
                    gold.add(sid)
            questions.append(
                {
                    "question_id": f"{sample_id}::q{q_idx}",
                    "question": str(qa.get("question", "")),
                    "question_type": f"category_{qa.get('category', 'unknown')}",
                    "gold_session_ids": sorted(gold),
                    "sessions": sessions,
                }
            )
    return questions


def build_session_docs(question: dict[str, Any], turns: str = "all") -> list[SessionDoc]:
    """One SessionDoc per conversation session for a normalised question.

    ``turns='all'`` concatenates every speaker turn. Any other value is treated
    as a speaker name to keep (LoCoMo turns are speaker-tagged, not
    role-tagged), the multi-speaker analogue of LongMemEval's ``user`` filter.
    """
    docs: list[SessionDoc] = []
    for session in question.get("sessions", []):
        sid = str(session.get("session_id", ""))
        parts: list[str] = []
        for turn in session.get("turns", []):
            if not isinstance(turn, dict):
                continue
            speaker = str(turn.get("speaker", "unknown"))
            if turns != "all" and speaker != turns:
                continue
            text = turn.get("text", "") or turn.get("content", "")
            if text:
                parts.append(f"({speaker}) {text}")
        docs.append(SessionDoc(doc_id=sid, text=" ".join(parts)))
    return docs


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
    """Drive one adapter across LoCoMo; return per-question scores + probes.

    ``per_type`` (stratified: up to N per LoCoMo category) takes precedence over
    ``sample`` (uniform random of N); either narrows *which* questions run
    without touching how they are scored. Both are seeded — no wall-clock, no
    unseeded randomness in the scored path.
    """
    adapter = get_adapter(adapter_name)

    # Flatten the nested shape, then keep only questions with at least one gold
    # session — adversarial/answer-less questions (empty/opaque evidence) have
    # nothing to retrieve and would score a spurious zero.
    pool = [q for q in flatten_questions(dataset) if q.get("gold_session_ids")]
    if per_type > 0:
        pool = stratified_sample(pool, per_type, seed)
    elif sample and sample < len(pool):
        import random

        # Deterministic seeded sampling for reproducible benchmark selection (not a crypto/security context).
        rng = random.Random(seed)  # nosec B311
        pool = rng.sample(pool, sample)

    scores = []
    probes = []
    skipped = 0
    t0 = time.time()

    for i, q in enumerate(pool):
        query = q.get("question", "")
        gold = {str(g) for g in q.get("gold_session_ids", [])}
        if not query or not gold:
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
    )


def render_scorecard(
    result: SuiteResult,
    *,
    dataset_path: str,
    k: int,
    embedder: str,
    sampling: str = "full set",
) -> str:
    """Render the Markdown scorecard (measured numbers + honesty rails)."""
    agg = aggregate(result.scores)
    probe = result.representative_probe()
    today = date.today().isoformat()
    lines: list[str] = []
    lines.append(f"# LoCoMo recall scorecard — `{result.adapter}` ({today})")
    lines.append("")
    lines.append("## Disclosure (what was actually measured)")
    lines.append("")
    lines.append(f"- **Adapter:** `{result.adapter}`")
    lines.append(f"- **Sampling:** {sampling}")
    if probe is not None:
        lines.append(f"- **Declared backend:** `{probe.declared_backend}`")
        lines.append(f"- **Effective backend (probed):** `{probe.effective_backend}`")
        lines.append(f"- **Vector/embedder leg exercised:** `{probe.vector_available}`")
        lines.append(f"- **Config SHA-256 (16):** `{probe.config_sha256}`")
        if probe.notes:
            lines.append(f"- **Probe notes:** {probe.notes}")
    else:
        lines.append("- **Vector/embedder leg exercised:** `unknown` (no probe — nothing scored)")
    lines.append(f"- **Embedder:** {embedder}")
    lines.append(f"- **k (retrieval depth):** {k}")
    lines.append("- **Retrieval granularity:** whole session (turns concatenated), untruncated at ingest")
    lines.append(f"- **Dataset:** LoCoMo (`{os.path.basename(dataset_path)}`), turns=`{result.turns}`")
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
        lines.append("### By question category (recall_any@5 / recall_all@5)")
        lines.append("")
        lines.append("| category | n | any@5 | all@5 | mrr |")
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
        "stricter `recall_all@k` (all gold sessions in top-k) are shown side by side; neither "
        "is cherry-picked."
    )
    lines.append(
        "- **Sample size + pipeline disclosed.** The disclosure block above states how many "
        "questions were scored, the effective backend (probed, not assumed), and whether the "
        "vector/embedder leg was exercised — a config-less fallback can never be reported as "
        "the full stack."
    )
    lines.append(
        f"- **No prior recall number is reproduced.** The {PRIOR_LLM_JUDGE_NOTE} — a DIFFERENT "
        "metric (answer quality via an external LLM judge), not retrieval recall@k. It is not "
        "restated as a retrieval result here; this scorecard reports only what this run measured."
    )
    lines.append(
        "- **No competitor comparison** is printed. Cross-system numbers require the same box, "
        "same dataset, same protocol, ≥2 reps — out of scope here."
    )
    lines.append(
        "- **Self-asserting pipeline.** Every NDJSON row carries a `pipeline` probe "
        "(declared/effective backend + config hash) so the exact false-green a config-less "
        "fallback would produce is impossible to report as the full stack."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LoCoMo recall harness")
    ap.add_argument("--adapter", default="mind_mem", help="bm25_baseline | mind_mem")
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--turns", default="all", help="'all' or a speaker name to keep")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sample", type=int, default=0, help="uniform random sample; 0 = full set")
    ap.add_argument(
        "--per-type",
        type=int,
        default=0,
        dest="per_type",
        help="stratified: up to N questions per LoCoMo category (0 = off; takes precedence over --sample)",
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
            f"STRATIFIED SAMPLE — up to {a.per_type} questions per LoCoMo category "
            f"(seed {a.seed}); {result.evaluated} evaluated. FULL set DEFERRED."
        )
    elif a.sample > 0:
        sampling = f"uniform random sample of {a.sample} (seed {a.seed}); full set DEFERRED"
    else:
        sampling = "full set"

    ndjson_path = a.ndjson or f"benchmarks/.cache/{date.today().isoformat()}-locomo-{a.adapter}.ndjson"
    write_ndjson(result, ndjson_path)
    scorecard = render_scorecard(result, dataset_path=data_path, k=a.k, embedder=a.embedder, sampling=sampling)
    scorecard_path = a.scorecard or f"docs/benchmarks/{date.today().isoformat()}-locomo-{a.adapter}.md"
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
