# Copyright 2026 STARGA, Inc.
"""A scorecard may not describe a pipeline its own rows contradict.

Numeric replay does not catch this class. The Chroma scorecard recomputed
every published figure correctly while telling the reader the run was
"lexical-only ... effective embedder: none" -- when 436 of its 470 committed
rows recorded `effective_backend: chroma_hnsw_cosine` with a real embedder.
Every number was right and the description was false, in the direction that
published a purpose-built vector store as BM25.

The check must be sharp in BOTH directions, which is why the negative control
matters as much as the positive one: a run that genuinely has no dense
provider must stay free to say lexical-only, or the check degrades into
"never mention lexical" and gets deleted the first time it annoys someone.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "benchmarks"))

repro_verify = pytest.importorskip("repro_verify")


def _rows(n: int, *, backend: str, embedder: str | None) -> list[dict]:
    extra = {"n_sessions": 3}
    if embedder is not None:
        extra["embedder"] = embedder
    return [{"pipeline": {"effective_backend": backend, "vector_available": True, "extra": dict(extra)}} for _ in range(n)]


def _card(tmp_path: pathlib.Path, text: str) -> str:
    p = tmp_path / "card.md"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_a_dense_run_published_as_lexical_only_is_caught(tmp_path) -> None:
    """The exact defect: Chroma rows, a BM25 claim. Must FAIL."""
    rep = repro_verify.Report("probe")
    repro_verify.verify_pipeline_disclosure(
        _rows(436, backend="chroma_hnsw_cosine", embedder="mxbai-embed-large"),
        _card(tmp_path, "- **Effective embedder:** `none - BM25F lexical only`\nthis number is lexical-only.\n"),
        rep,
    )
    assert rep.failures, "a dense run described as lexical-only was not caught"


def test_a_genuinely_lexical_run_may_say_so(tmp_path) -> None:
    """The negative control. Same claim, no dense provider in the rows: PASS.

    Without this the check would fail every honest lexical scorecard, which is
    how a real gate gets weakened into uselessness. Measured: a first version
    keyed on `vector_available` did exactly that and failed two truthful
    scorecards, because vector deps being importable is a dependency fact and
    is true in the lexical runs too.
    """
    rep = repro_verify.Report("probe")
    repro_verify.verify_pipeline_disclosure(
        _rows(470, backend="sqlite", embedder=None),
        _card(tmp_path, "- **Effective embedder:** `none - BM25F lexical only`\nthis number is lexical-only.\n"),
        rep,
    )
    assert not rep.failures, f"an honest lexical scorecard was failed: {rep.failures}"


def test_a_corrected_scorecard_passes(tmp_path) -> None:
    """"lexical-only" may appear while RETRACTING it, and must not re-fail."""
    rep = repro_verify.Report("probe")
    repro_verify.verify_pipeline_disclosure(
        _rows(436, backend="chroma_hnsw_cosine", embedder="mxbai-embed-large"),
        _card(tmp_path, "the earlier text said lexical-only. That was false; recounted from the rows below.\n"),
        rep,
    )
    assert not rep.failures, f"a corrected scorecard was failed for quoting its own retraction: {rep.failures}"


def test_the_discriminator_is_the_embedder_not_the_dependency_flag() -> None:
    """`vector_available` is a dependency fact and cannot select dense rows.

    Pinned because keying on it is the mistake that was actually made, and it
    fails in the flattering direction: it would mark every run dense.
    """
    lexical = _rows(3, backend="sqlite", embedder=None)
    dense = _rows(3, backend="chroma_hnsw_cosine", embedder="mxbai-embed-large")
    assert all(r["pipeline"]["vector_available"] for r in lexical + dense), "fixture does not reproduce the trap"
    assert repro_verify.dense_provider_rows(lexical) == []
    assert len(repro_verify.dense_provider_rows(dense)) == 3


def test_the_committed_chroma_scorecard_is_consistent_with_its_rows() -> None:
    """And the real artifact, since the correction is the point of all this."""
    import json

    d = _REPO / "docs" / "benchmarks" / "head-20260907"
    rows = [json.loads(line) for line in (d / "lme-chroma.ndjson").read_text(encoding="utf-8").splitlines() if line.strip()]
    dense = repro_verify.dense_provider_rows(rows)
    assert len(rows) == 470 and len(dense) == 436, f"the committed evidence moved: {len(rows)} rows, {len(dense)} dense"

    rep = repro_verify.Report("committed")
    repro_verify.verify_pipeline_disclosure(rows, str(d / "lme-chroma.md"), rep)
    assert not rep.failures, f"the committed Chroma scorecard still contradicts its rows: {rep.failures}"
