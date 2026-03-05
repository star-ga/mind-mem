"""Tests for mind-mem governance benchmark suite."""

import json
import os

import pytest

from mind_mem.governance_bench import GovernanceBench


@pytest.fixture
def workspace(tmp_path):
    ws = str(tmp_path)
    os.makedirs(os.path.join(ws, "decisions"), exist_ok=True)
    os.makedirs(os.path.join(ws, "intelligence"), exist_ok=True)
    return ws


@pytest.fixture
def bench(workspace):
    return GovernanceBench(workspace)


class TestGovernanceBench:
    def test_empty_workspace(self, bench):
        results = bench.run_all()
        assert "timestamp" in results
        assert "contradiction_detection" in results
        assert "audit_completeness" in results
        assert "drift_detection" in results
        assert "scalability" in results

    def test_contradiction_detection_no_decisions(self, bench):
        result = bench.bench_contradiction_detection()
        assert result["status"] == "skip"

    def test_contradiction_detection_with_decisions(self, workspace, bench):
        path = os.path.join(workspace, "decisions", "DECISIONS.md")
        with open(path, "w") as f:
            f.write("""
[D-20260301-001]
Date: 2026-03-01
Subject: Use PostgreSQL

---

[D-20260302-001]
Date: 2026-03-02
Subject: Use MySQL instead
""")
        result = bench.bench_contradiction_detection()
        assert result["status"] == "ok"
        assert "detection_time_ms" in result

    def test_audit_completeness_empty(self, bench):
        result = bench.bench_audit_completeness()
        assert result["status"] == "ok"
        assert result["total_entries"] == 0
        assert result["chain_valid"]

    def test_audit_completeness_with_entries(self, workspace, bench):
        from mind_mem.audit_chain import AuditChain

        chain = AuditChain(workspace)
        chain.append("create_block", "test.md", agent="test")
        chain.append("update_field", "test.md", agent="test")

        result = bench.bench_audit_completeness()
        assert result["total_entries"] == 2
        assert result["chain_valid"]
        assert result["operations_breakdown"]["create_block"] == 1

    def test_drift_detection(self, bench):
        result = bench.bench_drift_detection()
        assert result["status"] == "ok"
        assert "scan_time_ms" in result

    def test_scalability(self, workspace, bench):
        # Create some test blocks
        path = os.path.join(workspace, "decisions", "DECISIONS.md")
        with open(path, "w") as f:
            for i in range(10):
                f.write(f"\n[D-20260301-{i:03d}]\nSubject: Decision {i}\n\n---\n")

        result = bench.bench_scalability()
        assert result["block_count"] >= 10
        assert "parse_time_ms" in result

    def test_save_results(self, workspace, bench):
        bench.run_all()
        path = bench.save_results()
        assert os.path.isfile(path)

        with open(path) as f:
            data = json.load(f)
        assert "timestamp" in data

    def test_save_custom_path(self, workspace, bench):
        bench.run_all()
        custom = os.path.join(workspace, "bench-results.json")
        path = bench.save_results(custom)
        assert path == custom
        assert os.path.isfile(custom)
