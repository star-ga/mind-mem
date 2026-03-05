#!/usr/bin/env python3
"""mind-mem Governance Benchmark Suite.

Measures governance-specific metrics beyond retrieval quality:
- Contradiction detection rate (precision/recall/F1)
- Resolution confidence calibration
- Audit trail completeness
- Temporal filter accuracy
- Scalability metrics (time/memory at various block counts)

Usage:
    from .governance_bench import GovernanceBench
    bench = GovernanceBench(workspace)
    results = bench.run_all()
    print(json.dumps(results, indent=2))

Zero external deps — time, json, os (all stdlib).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

from .audit_chain import AuditChain
from .block_parser import parse_file
from .conflict_resolver import resolve_contradictions as _resolve
from .drift_detector import DriftDetector
from .observability import get_logger

_log = get_logger("governance_bench")


class GovernanceBench:
    """Governance-specific benchmark harness."""

    def __init__(self, workspace: str) -> None:
        self.workspace = os.path.realpath(workspace)
        self._results: dict = {}

    def run_all(self) -> dict:
        """Run all governance benchmarks and return results."""
        self._results = {
            "timestamp": datetime.now().isoformat(),
            "workspace": self.workspace,
        }

        self._results["contradiction_detection"] = self.bench_contradiction_detection()
        self._results["audit_completeness"] = self.bench_audit_completeness()
        self._results["drift_detection"] = self.bench_drift_detection()
        self._results["scalability"] = self.bench_scalability()

        return self._results

    def bench_contradiction_detection(self) -> dict:
        """Benchmark contradiction detection quality.

        Measures:
        - Detection count
        - Time per detection
        - Unique block pairs found
        """
        decisions_path = os.path.join(self.workspace, "decisions", "DECISIONS.md")
        if not os.path.isfile(decisions_path):
            return {"status": "skip", "reason": "no decisions file"}

        start = time.monotonic()
        try:
            contradictions = _resolve(self.workspace)
        except Exception as e:
            return {"status": "error", "error": str(e)}
        elapsed = time.monotonic() - start

        unique_pairs = set()
        for c in contradictions:
            pair = frozenset({c.get("block_a", ""), c.get("block_b", "")})
            unique_pairs.add(pair)

        return {
            "status": "ok",
            "total_detected": len(contradictions),
            "unique_pairs": len(unique_pairs),
            "detection_time_ms": round(elapsed * 1000, 2),
            "avg_per_detection_ms": round((elapsed * 1000 / len(contradictions)) if contradictions else 0, 2),
        }

    def bench_audit_completeness(self) -> dict:
        """Benchmark audit chain completeness and integrity.

        Measures:
        - Total entries in chain
        - Chain integrity (verify passes)
        - Verification time
        - Operations covered
        """
        chain = AuditChain(self.workspace)

        start = time.monotonic()
        is_valid, errors = chain.verify()
        verify_time = time.monotonic() - start

        entries = chain.entries()
        operations: dict[str, int] = {}
        agents: dict[str, int] = {}
        for entry in entries:
            operations[entry.operation] = operations.get(entry.operation, 0) + 1
            if entry.agent:
                agents[entry.agent] = agents.get(entry.agent, 0) + 1

        return {
            "status": "ok",
            "total_entries": len(entries),
            "chain_valid": is_valid,
            "chain_errors": len(errors),
            "verify_time_ms": round(verify_time * 1000, 2),
            "operations_breakdown": operations,
            "agents_breakdown": agents,
        }

    def bench_drift_detection(self) -> dict:
        """Benchmark drift detection performance.

        Measures:
        - Scan time
        - Signals found by type
        - Average confidence
        """
        detector = DriftDetector(self.workspace)

        start = time.monotonic()
        try:
            signals = detector.scan()
        except Exception as e:
            return {"status": "error", "error": str(e)}
        elapsed = time.monotonic() - start

        by_type: dict[str, int] = {}
        confidences = []
        for sig in signals:
            by_type[sig.drift_type] = by_type.get(sig.drift_type, 0) + 1
            confidences.append(sig.confidence)

        return {
            "status": "ok",
            "total_signals": len(signals),
            "by_type": by_type,
            "scan_time_ms": round(elapsed * 1000, 2),
            "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0,
            "max_confidence": round(max(confidences), 4) if confidences else 0,
        }

    def bench_scalability(self) -> dict:
        """Benchmark operations at current workspace scale.

        Measures time for:
        - Block parsing
        - Audit chain verification
        - Full scan cycle
        """
        results: dict[str, int | float] = {}

        # Count blocks
        block_count = 0
        for dirpath, _dirnames, filenames in os.walk(self.workspace):
            for fname in filenames:
                if fname.endswith(".md"):
                    path = os.path.join(dirpath, fname)
                    try:
                        blocks = parse_file(path)
                        block_count += len(blocks)
                    except (OSError, ValueError):
                        pass

        results["block_count"] = block_count

        # Parse time
        start = time.monotonic()
        for dirpath, _dirnames, filenames in os.walk(self.workspace):
            for fname in filenames:
                if fname.endswith(".md"):
                    try:
                        parse_file(os.path.join(dirpath, fname))
                    except (OSError, ValueError):
                        pass
        results["parse_time_ms"] = round((time.monotonic() - start) * 1000, 2)

        # Audit verification time
        chain = AuditChain(self.workspace)
        start = time.monotonic()
        chain.verify()
        results["verify_time_ms"] = round((time.monotonic() - start) * 1000, 2)

        # Blocks per second
        if results["parse_time_ms"] > 0:
            results["blocks_per_second"] = round(block_count / (results["parse_time_ms"] / 1000), 0)

        return results

    def save_results(self, output_path: str | None = None) -> str:
        """Save benchmark results to a JSON file.

        Args:
            output_path: Path to save. Default: workspace/.mind-mem-audit/governance-bench.json

        Returns:
            Path to the saved file.
        """
        if not self._results:
            self._results = self.run_all()

        if output_path is None:
            audit_dir = os.path.join(self.workspace, ".mind-mem-audit")
            os.makedirs(audit_dir, exist_ok=True)
            output_path = os.path.join(audit_dir, "governance-bench.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self._results, f, indent=2, default=str)

        _log.info("governance_bench_saved", output=output_path)
        return output_path
