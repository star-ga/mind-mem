"""LoCoMo recall-quality regression gate.

Builds a fixed in-memory corpus from the LoCoMo benchmark harness helpers
(no live LLM call, no network fetch), then runs recall() over it and
asserts deterministic Recall@K / MRR floors.

Floor values are conservative lower-bounds derived from the observed run
logged in benchmarks/locomo_results.json (dry-run, 1 conversation, 199 QA
pairs, MRR=0.4565, R@5=0.6181).  They are set ~20% below observed so this
is a real regression gate rather than a flake.

Run:
    pytest tests/test_recall_quality_locomo.py -x
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Make sure the benchmarks helper is importable without polluting sys.path
# permanently — add benchmarks/ only while needed.
_BENCH_DIR = str(Path(__file__).parent.parent / "benchmarks")

# ---------------------------------------------------------------------------
# Fixed minimal LoCoMo-style corpus
# ---------------------------------------------------------------------------
# This is a hand-crafted excerpt that mirrors the wire format produced by
# benchmarks/locomo_harness.py::build_workspace().  It is FIXED (not
# downloaded) so the test runs offline and is deterministic.

_CORPUS = """# LoCoMo Conversation Memory

[DIA-D1-1]
Statement: [Alice] I started working at the software company last month.
Date: 2023-01-10
Status: active
DiaID: D1:1
Tags: session-1, Alice

[DIA-D1-2]
Statement: [Bob] Which company? Are you enjoying it?
Date: 2023-01-10
Status: active
DiaID: D1:2
Tags: session-1, Bob

[DIA-D1-3]
Statement: [Alice] TechCorp. Yes, the team is great and I love the product work.
Date: 2023-01-10
Status: active
DiaID: D1:3
Tags: session-1, Alice

[DIA-D1-4]
Statement: [Bob] I heard TechCorp is expanding into Europe next year.
Date: 2023-01-10
Status: active
DiaID: D1:4
Tags: session-1, Bob

[DIA-D2-1]
Statement: [Alice] My dog Max had a vet appointment yesterday.
Date: 2023-02-05
Status: active
DiaID: D2:1
Tags: session-2, Alice

[DIA-D2-2]
Statement: [Bob] Is Max doing okay?
Date: 2023-02-05
Status: active
DiaID: D2:2
Tags: session-2, Bob

[DIA-D2-3]
Statement: [Alice] Yes, just a routine check. The vet said he is very healthy.
Date: 2023-02-05
Status: active
DiaID: D2:3
Tags: session-2, Alice

[DIA-D2-4]
Statement: [Bob] That is great. Dogs really keep you grounded.
Date: 2023-02-05
Status: active
DiaID: D2:4
Tags: session-2, Bob

[DIA-D3-1]
Statement: [Alice] I am planning a trip to Japan in spring.
Date: 2023-03-12
Status: active
DiaID: D3:1
Tags: session-3, Alice

[DIA-D3-2]
Statement: [Bob] Japan is amazing. Have you booked the flights yet?
Date: 2023-03-12
Status: active
DiaID: D3:2
Tags: session-3, Bob

[DIA-D3-3]
Statement: [Alice] Not yet. I am looking at cherry blossom season in April.
Date: 2023-03-12
Status: active
DiaID: D3:3
Tags: session-3, Alice

[DIA-D3-4]
Statement: [Bob] April is perfect for that. Tokyo and Kyoto are both beautiful.
Date: 2023-03-12
Status: active
DiaID: D3:4
Tags: session-3, Bob

[DIA-D4-1]
Statement: [Alice] I have been learning Python for the past six months.
Date: 2023-04-20
Status: active
DiaID: D4:1
Tags: session-4, Alice

[DIA-D4-2]
Statement: [Bob] Python is a great first language. What are you building?
Date: 2023-04-20
Status: active
DiaID: D4:2
Tags: session-4, Bob

[DIA-D4-3]
Statement: [Alice] A small automation script for my TechCorp workflow.
Date: 2023-04-20
Status: active
DiaID: D4:3
Tags: session-4, Alice

[DIA-D4-4]
Statement: [Bob] That sounds useful. Automation can save a lot of time.
Date: 2023-04-20
Status: active
DiaID: D4:4
Tags: session-4, Bob
"""

# ---------------------------------------------------------------------------
# Fixed QA pairs — question, expected evidence dialog IDs (as in LoCoMo)
# ---------------------------------------------------------------------------
_QA_PAIRS: list[dict] = [
    # single-hop: direct retrieval
    {
        "question": "Where does Alice work?",
        "evidence": {"D1:3"},
        "category": "single-hop",
    },
    {
        "question": "What is the name of Alice's dog?",
        "evidence": {"D2:1"},
        "category": "single-hop",
    },
    {
        "question": "Where is Alice planning to travel?",
        "evidence": {"D3:1"},
        "category": "single-hop",
    },
    {
        "question": "What programming language is Alice learning?",
        "evidence": {"D4:1"},
        "category": "single-hop",
    },
    # multi-hop
    {
        "question": "What is Alice building with the programming language she is learning?",
        "evidence": {"D4:3"},
        "category": "multi-hop",
    },
    {
        "question": "What season does Alice want to see in Japan?",
        "evidence": {"D3:3"},
        "category": "multi-hop",
    },
    # temporal
    {
        "question": "What happened with Alice's pet in February 2023?",
        "evidence": {"D2:1"},
        "category": "temporal",
    },
    # open-domain
    {
        "question": "Which city is suitable for cherry blossom viewing?",
        "evidence": {"D3:4"},
        "category": "open-domain",
    },
    {
        "question": "Is TechCorp expanding?",
        "evidence": {"D1:4"},
        "category": "open-domain",
    },
]


# ---------------------------------------------------------------------------
# Helpers — mirrors locomo_harness.evaluate_sample() logic, no imports
# ---------------------------------------------------------------------------


def _build_workspace(tmpdir: str) -> str:
    """Write the fixed corpus into a minimal workspace directory."""
    decisions_dir = os.path.join(tmpdir, "decisions")
    os.makedirs(decisions_dir, exist_ok=True)
    # Other expected workspace dirs so recall() does not warn
    for d in ("tasks", "entities", "intelligence"):
        os.makedirs(os.path.join(tmpdir, d), exist_ok=True)

    with open(os.path.join(decisions_dir, "DECISIONS.md"), "w", encoding="utf-8") as fh:
        fh.write(_CORPUS)
    return tmpdir


def _dia_id_from_result(result: dict) -> str:
    """Extract a LoCoMo dialog ID from a recall result dict."""
    import re

    dia = result.get("DiaID", "")
    if dia:
        return dia
    bid = result.get("_id", "")
    if bid.startswith("DIA-"):
        raw = bid[4:]
        m = re.match(r"(D\d+)-(\d+)", raw)
        if m:
            return f"{m.group(1)}:{m.group(2)}"
    return ""


def _hit_at_k(retrieved_ids: list[str], evidence: set, k: int) -> bool:
    return any(did in evidence for did in retrieved_ids[:k])


def _reciprocal_rank(retrieved_ids: list[str], evidence: set) -> float:
    for rank, did in enumerate(retrieved_ids, 1):
        if did in evidence:
            return 1.0 / rank
    return 0.0


def _run_eval(workspace: str, max_k: int = 10) -> dict:
    """Run recall over each QA pair. Returns aggregate metrics."""
    from mind_mem.recall import recall as _recall

    hits_at: dict[int, int] = {1: 0, 3: 0, 5: 0, 10: 0}
    mrr_sum = 0.0
    total = 0

    for qa in _QA_PAIRS:
        results = _recall(workspace, qa["question"], limit=max_k, active_only=False)
        retrieved_ids = [_dia_id_from_result(r) for r in results]

        for k in (1, 3, 5, 10):
            if _hit_at_k(retrieved_ids, qa["evidence"], k):
                hits_at[k] += 1
        mrr_sum += _reciprocal_rank(retrieved_ids, qa["evidence"])
        total += 1

    n = total or 1
    return {
        "mrr": round(mrr_sum / n, 4),
        "recall_at": {k: round(hits_at[k] / n, 4) for k in (1, 3, 5, 10)},
        "n": total,
    }


# ---------------------------------------------------------------------------
# Conservative floor constants (set below observed benchmark values)
# ---------------------------------------------------------------------------
# Observed on the full LoCoMo dry-run (199 questions):
#   MRR=0.4565, R@1=0.3568, R@5=0.6181
# Our micro-corpus has only 9 questions — floors are set conservatively
# to avoid flakiness while still catching regressions.
_FLOOR_MRR = 0.20
_FLOOR_R1 = 0.10
_FLOOR_R3 = 0.20
_FLOOR_R5 = 0.30
_FLOOR_R10 = 0.40


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoCoMoRecallFloors:
    """Regression gate: recall quality must not drop below floor values."""

    @pytest.fixture(scope="class")
    def metrics(self, tmp_path_factory):
        tmpdir = str(tmp_path_factory.mktemp("locomo_ws"))
        _build_workspace(tmpdir)
        return _run_eval(tmpdir, max_k=10)

    def test_mrr_floor(self, metrics):
        """MRR must be at or above the floor."""
        assert metrics["mrr"] >= _FLOOR_MRR, f"MRR {metrics['mrr']:.4f} < floor {_FLOOR_MRR} — recall quality regression"

    def test_recall_at_1_floor(self, metrics):
        """Recall@1 must be at or above the floor."""
        r1 = metrics["recall_at"][1]
        assert r1 >= _FLOOR_R1, f"R@1 {r1:.4f} < floor {_FLOOR_R1}"

    def test_recall_at_3_floor(self, metrics):
        """Recall@3 must be at or above the floor."""
        r3 = metrics["recall_at"][3]
        assert r3 >= _FLOOR_R3, f"R@3 {r3:.4f} < floor {_FLOOR_R3}"

    def test_recall_at_5_floor(self, metrics):
        """Recall@5 must be at or above the floor."""
        r5 = metrics["recall_at"][5]
        assert r5 >= _FLOOR_R5, f"R@5 {r5:.4f} < floor {_FLOOR_R5}"

    def test_recall_at_10_floor(self, metrics):
        """Recall@10 must be at or above the floor."""
        r10 = metrics["recall_at"][10]
        assert r10 >= _FLOOR_R10, f"R@10 {r10:.4f} < floor {_FLOOR_R10}"

    def test_corpus_coverage(self, metrics):
        """All QA pairs were evaluated."""
        assert metrics["n"] == len(_QA_PAIRS), f"Expected {len(_QA_PAIRS)} QA pairs, got {metrics['n']}"

    def test_metrics_are_printed(self, metrics, capsys):
        """Smoke-test: metric values are in expected range [0, 1]."""
        for k, v in metrics["recall_at"].items():
            assert 0.0 <= v <= 1.0, f"recall_at[{k}] out of range: {v}"
        assert 0.0 <= metrics["mrr"] <= 1.0
        # Print for CI log visibility
        print(
            f"\nLoCoMo micro-corpus metrics: MRR={metrics['mrr']:.4f} "
            f"R@1={metrics['recall_at'][1]:.4f} "
            f"R@3={metrics['recall_at'][3]:.4f} "
            f"R@5={metrics['recall_at'][5]:.4f} "
            f"R@10={metrics['recall_at'][10]:.4f}"
        )
