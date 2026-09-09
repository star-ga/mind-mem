"""Real SQLite/public-path coverage for compound recall filters."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from mind_mem._recall_core import _apply_post_filters
from mind_mem.mcp.tools import recall as recall_tools
from mind_mem.sqlite_index import build_index


def _indexed_filter_workspace(tmp_path: Path) -> str:
    workspace = tmp_path / "workspace"
    decisions = workspace / "decisions"
    decisions.mkdir(parents=True)
    (workspace / "mind-mem.json").write_text(
        json.dumps(
            {
                "cache": {"enabled": True},
                "recall": {"backend": "sqlite", "expand_query": False},
            }
        ),
        encoding="utf-8",
    )
    blocks = []
    for index in range(20):
        blocks.append(
            f"[D-FILT-{index:03d}]\n"
            "Statement: quartz vendor price\n"
            "Status: active\n"
            "Lifecycle: ephemeral\n"
            "EventId: EVT-EPHEMERAL\n"
            "Maturity: 0.1\n\n"
        )
    blocks.append(
        "[D-FILT-999]\nStatement: quartz vendor price\nStatus: active\nLifecycle: durable\nEventId: EVT-DURABLE\nMaturity: 0.9\n\n"
    )
    (decisions / "DECISIONS.md").write_text("".join(blocks), encoding="utf-8")
    build_index(str(workspace), incremental=False)
    return str(workspace)


def test_indexed_public_path_preserves_filter_metadata_and_warm_cache(monkeypatch, tmp_path):
    """A real indexed, cached call applies all three json-backed filters."""
    workspace = _indexed_filter_workspace(tmp_path)
    monkeypatch.setattr(recall_tools, "_workspace", lambda: workspace)
    uncached = Mock(wraps=recall_tools._recall_impl_uncached)
    monkeypatch.setattr(recall_tools, "_recall_impl_uncached", uncached)

    first = json.loads(
        recall_tools._recall_impl(
            "quartz vendor price",
            limit=1,
            backend="bm25",
            lifecycle="durable",
            event_id="EVT-DURABLE",
            min_maturity=0.8,
        )
    )
    second = json.loads(
        recall_tools._recall_impl(
            "quartz vendor price",
            limit=1,
            backend="bm25",
            lifecycle="durable",
            event_id="EVT-DURABLE",
            min_maturity=0.8,
        )
    )

    assert [hit["_id"] for hit in first["results"]] == ["D-FILT-999"]
    assert first["results"] == second["results"]
    assert first["results"][0]["Lifecycle"] == "durable"
    assert first["results"][0]["EventId"] == "EVT-DURABLE"
    assert first["results"][0]["Maturity"] == "0.9"
    assert uncached.call_count == 1, "the second response must replay the actual cache entry"


@pytest.mark.parametrize("filters", [{"lifecycle": "durable"}, {"event_id": "EVT-DURABLE"}, {"min_maturity": 0.8}])
def test_indexed_public_path_excludes_nonmatching_rows_from_full_pool(monkeypatch, tmp_path, filters):
    """An untruncated pool exposes metadata loss that rerank hydration can hide."""
    workspace = _indexed_filter_workspace(tmp_path)
    monkeypatch.setattr(recall_tools, "_workspace", lambda: workspace)
    unfiltered = json.loads(recall_tools._recall_impl_uncached("quartz vendor price", limit=21, backend="bm25"))
    assert len(unfiltered["results"]) == 21
    filtered = json.loads(recall_tools._recall_impl_uncached("quartz vendor price", limit=21, backend="bm25", **filters))
    assert [hit["_id"] for hit in filtered["results"]] == ["D-FILT-999"]


def test_post_filters_apply_conjunction_before_single_limit_slice():
    hits = [
        {"_id": "wrong-event", "status": "active", "Lifecycle": "durable", "EventId": "other", "Maturity": "0.9"},
        {"_id": "wrong-maturity", "status": "active", "Lifecycle": "durable", "EventId": "wanted", "Maturity": "0.1"},
        {"_id": "all-match", "status": "active", "Lifecycle": "durable", "EventId": "wanted", "Maturity": "0.9"},
    ]
    result = _apply_post_filters(
        hits,
        since=None,
        until=None,
        lifecycle="durable",
        event_id="wanted",
        min_maturity=0.8,
        limit=1,
    )
    assert [hit["_id"] for hit in result] == ["all-match"]


def test_unfiltered_funnel_preserves_result_carrier_identity():
    from mind_mem.hybrid_recall import RecallResults

    hits = RecallResults([{"_id": "D-1", "status": "active"}])
    hits.degraded = {"leg": "vector", "reason": "deadline_exceeded"}
    result = _apply_post_filters(hits, since=None, until=None, lifecycle=None, event_id=None, min_maturity=None, limit=10)
    assert result is hits
    assert result.degraded == {"leg": "vector", "reason": "deadline_exceeded"}


def test_filtered_carrier_keeps_degradation_and_trace_but_not_stale_attestation():
    from mind_mem.hybrid_recall import RecallResults

    hits = RecallResults(
        [
            {"_id": "keep", "status": "active", "Lifecycle": "durable"},
            {"_id": "drop", "status": "active", "Lifecycle": "ephemeral"},
        ]
    )
    hits.degraded = {"leg": "vector", "reason": "deadline_exceeded"}
    hits.trace = {"vector": {"ran": True}}
    hits.attestation = {"results_digest": "binds-the-two-row-input"}
    result = _apply_post_filters(
        hits,
        since=None,
        until=None,
        lifecycle="durable",
        event_id=None,
        min_maturity=None,
        limit=10,
    )
    assert [hit["_id"] for hit in result] == ["keep"]
    assert result.degraded == hits.degraded
    assert result.trace == hits.trace
    assert result.attestation is None


def test_truncated_carrier_keeps_run_metadata_without_stale_attestation():
    from mind_mem.hybrid_recall import RecallResults

    hits = RecallResults([{"_id": "first", "status": "active"}, {"_id": "second", "status": "active"}])
    hits.degraded = {"leg": "bm25", "reason": "corpus_truncated"}
    hits.trace = {"bm25": {"ran": True}}
    hits.attestation = {"results_digest": "binds-the-two-row-input"}
    result = _apply_post_filters(hits, since=None, until=None, lifecycle=None, event_id=None, min_maturity=None, limit=1)
    assert [hit["_id"] for hit in result] == ["first"]
    assert result.degraded == hits.degraded
    assert result.trace == hits.trace
    assert result.attestation is None


def test_live_status_copy_does_not_strip_the_carrier(monkeypatch):
    from mind_mem import _recall_core
    from mind_mem.hybrid_recall import RecallResults

    hits = RecallResults([{"_id": "D-1", "status": "active"}])
    hits.degraded = {"leg": "vector", "reason": "status_absent"}
    hits.trace = {"vector": {"ran": True}}
    monkeypatch.setattr(_recall_core, "live_statuses", lambda workspace: {"D-1": "active"})
    result = _apply_post_filters(
        hits,
        since=None,
        until=None,
        lifecycle=None,
        event_id=None,
        min_maturity=None,
        limit=10,
        workspace="/unused",
    )
    assert result.degraded == hits.degraded
    assert result.trace == hits.trace


@pytest.mark.parametrize("mode", ["similar", "axis", "pack", "prefetch", "classify", "diagnostics"])
@pytest.mark.parametrize(
    "filters",
    [{"since": "2026-01-01"}, {"until": "2026-12-31"}, {"lifecycle": "durable"}, {"event_id": "EVT-SWEEP"}, {"min_maturity": 0.0}],
)
def test_nonranking_modes_refuse_filters_they_cannot_enforce(mode, filters):
    from mind_mem.mcp.tools import public

    result = json.loads(public.recall.__wrapped__("quartz", mode=mode, **filters))
    assert "filters are not available" in result["error"]
