# Copyright 2026 STARGA, Inc.
"""``review_evidence`` — provenance, chain status and staleness inline.

Approving without the source block next to the proposal is approving a
claim you cannot check. Every panel here is read-only and degrades to a
stated reason rather than an exception: an evidence panel that can crash
the queue listing is a panel that gets removed.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _review_fixtures import build_workspace, mcp_budget  # noqa: E402,F401


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    root = str(tmp_path / "ws")
    os.makedirs(root)
    ids = build_workspace(root, 2)
    return root, ids


class TestTargetExcerpt:
    def test_quotes_the_block_the_proposal_would_change(self, workspace):
        from mind_mem.review_evidence import gather
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        panel = gather(root, load_queue(root)[0])
        assert "Baseline decision number 1" in panel.target_excerpt

    def test_a_missing_target_yields_a_reason_not_a_crash(self, workspace):
        from mind_mem.review_evidence import gather
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        os.remove(os.path.join(root, "decisions/DECISIONS.md"))
        panel = gather(root, load_queue(root)[0])
        assert panel.target_excerpt == ""
        assert panel.notes

    def test_the_excerpt_is_bounded(self, workspace):
        from mind_mem.review_evidence import MAX_EXCERPT_CHARS, gather
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        panel = gather(root, load_queue(root)[0])
        assert len(panel.target_excerpt) <= MAX_EXCERPT_CHARS


class TestChain:
    def test_reports_hash_and_evidence_chain_validity(self, workspace):
        from mind_mem.review_evidence import gather
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        panel = gather(root, load_queue(root)[0])
        assert panel.chain_valid is True
        assert panel.chain_summary


class TestStaleness:
    def test_a_clean_target_is_not_flagged(self, workspace):
        from mind_mem.review_evidence import gather
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        assert gather(root, load_queue(root)[0]).stale is False

    def test_a_flagged_target_is_surfaced_with_its_reason(self, workspace):
        from mind_mem.causal_graph import CausalGraph
        from mind_mem.review_evidence import gather
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        graph = CausalGraph(root)
        graph.add_edge("D-20260801-002", "D-20260801-001", "depends_on")
        graph.propagate_staleness("D-20260801-001", reason="upstream decision changed")
        panel = gather(root, load_queue(root)[1])
        assert panel.stale is True
        assert "upstream decision changed" in panel.stale_reason


class TestReadOnly:
    def test_gathering_evidence_leaves_the_corpus_byte_identical(self, workspace):
        """Reading provenance and verifying the chain may materialise
        their own empty backing stores on first use. Neither may modify
        a file that already existed, and neither may create a corpus,
        proposal or intelligence file."""
        from mind_mem.review_evidence import gather
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        before = _tree(root)
        for item in load_queue(root):
            gather(root, item)
        after = _tree(root)
        assert {k: v for k, v in after.items() if k in before} == before
        created = sorted(set(after) - set(before))
        assert all(_is_lazy_store(path) for path in created), created

    def test_gathering_evidence_adds_no_governance_chain_entry(self, workspace):
        from mind_mem.governance_gate import get_gate
        from mind_mem.review_evidence import gather
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        before = get_gate(root).chain.length
        for item in load_queue(root):
            gather(root, item)
        assert get_gate(root).chain.length == before

    def test_panel_is_json_serialisable(self, workspace):
        import json

        from mind_mem.review_evidence import gather
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        payload = gather(root, load_queue(root)[0]).to_dict()
        assert json.loads(json.dumps(payload)) == payload


#: Backing stores the read paths may lazily create on first use. Every
#: other new path means the evidence panel wrote something it should not.
_LAZY_STORE_PREFIXES = (".mind-mem-index/", "memory/hash_chain_v2.db", "memory/evidence")


def _is_lazy_store(relative: str) -> bool:
    normalised = relative.replace(os.sep, "/")
    return normalised.startswith(_LAZY_STORE_PREFIXES)


def _tree(root: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(dirnames)
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            out[os.path.relpath(full, root)] = os.path.getsize(full)
    return out
