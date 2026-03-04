"""Integration test: full mind-mem pipeline.

Exercises: init → ingest content → build index → recall → propose_update → approve.
Uses a temporary workspace directory — no fixtures checked in.
"""

from __future__ import annotations

import os

import pytest

from mind_mem.init_workspace import init
from mind_mem.sqlite_index import build_index

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def workspace(tmp_path):
    """Create a fresh workspace with sample content."""
    ws = str(tmp_path / "ws")
    init(ws)

    # Write blocks into decisions/DECISIONS.md using [ID] block format
    decisions_path = os.path.join(ws, "decisions", "DECISIONS.md")
    with open(decisions_path, "a") as f:
        f.write(
            "\n[D-20260301-001]\n"
            "Type: decision\n"
            "Status: active\n"
            "Date: 2026-03-01\n"
            "Tags: database, infrastructure, postgresql\n"
            "Statement: Use PostgreSQL for production database\n"
            "Rationale: PostgreSQL was chosen for its reliability, JSONB support, "
            "and strong ecosystem of extensions.\n\n"
            "[D-20260302-001]\n"
            "Type: decision\n"
            "Status: active\n"
            "Date: 2026-03-02\n"
            "Tags: backend, api, fastapi, python\n"
            "Statement: Adopt FastAPI for backend services\n"
            "Rationale: FastAPI provides async support, automatic OpenAPI docs, "
            "and Pydantic validation out of the box.\n\n"
        )

    # Write a task into tasks/TASKS.md
    tasks_path = os.path.join(ws, "tasks", "TASKS.md")
    with open(tasks_path, "a") as f:
        f.write(
            "\n[T-20260303-001]\n"
            "Type: task\n"
            "Status: active\n"
            "Date: 2026-03-03\n"
            "Tags: devops, ci, github-actions\n"
            "Statement: Set up CI/CD pipeline\n"
            "Description: Configure GitHub Actions for lint, test, and deployment.\n\n"
        )

    return ws


# ── Tests ────────────────────────────────────────────────────────────


class TestFullPipeline:
    """End-to-end: ingest → index → recall → propose → approve."""

    def test_build_index_returns_summary(self, workspace):
        summary = build_index(workspace)
        assert summary["files_indexed"] > 0
        assert summary["blocks_indexed"] > 0
        assert summary["elapsed_ms"] > 0

    def test_recall_finds_indexed_content(self, workspace):
        from mind_mem.recall import recall

        build_index(workspace)
        results = recall(workspace, "PostgreSQL database", limit=5)
        assert len(results) > 0, "recall should find at least one result"

        texts = [r.get("excerpt", "") + r.get("content", "") for r in results]
        combined = " ".join(texts).lower()
        assert "postgresql" in combined

    def test_recall_fastapi_content(self, workspace):
        from mind_mem.recall import recall

        build_index(workspace)
        results = recall(workspace, "FastAPI backend", limit=5)
        assert len(results) > 0
        texts = [r.get("excerpt", "") + r.get("content", "") for r in results]
        combined = " ".join(texts).lower()
        assert "fastapi" in combined

    def test_recall_task_content(self, workspace):
        from mind_mem.recall import recall

        build_index(workspace)
        results = recall(workspace, "CI/CD pipeline", limit=5)
        assert len(results) > 0

    def test_incremental_rebuild_is_noop(self, workspace):
        """Second build should find no changed files."""
        build_index(workspace)
        summary2 = build_index(workspace)
        assert summary2["files_indexed"] == 0
        assert summary2["blocks_new"] == 0

    def test_propose_creates_signal(self, workspace):
        """Proposal writes to SIGNALS.md via capture.append_signals."""
        from datetime import datetime

        from mind_mem.capture import append_signals

        signals_path = os.path.join(workspace, "intelligence", "SIGNALS.md")
        before = os.path.getsize(signals_path) if os.path.isfile(signals_path) else 0

        signal = {
            "line": 0,
            "type": "decision",
            "text": "Switch to Redis for caching layer",
            "pattern": "mcp_propose_update",
            "confidence": "high",
            "priority": "P1",
            "structure": {
                "subject": "Switch to Redis",
                "tags": ["caching", "redis", "infrastructure"],
                "rationale": "Redis provides sub-millisecond latency",
            },
        }
        today = datetime.now().strftime("%Y-%m-%d")
        written = append_signals(workspace, [signal], today)
        assert written > 0, "append_signals should write at least 1 signal"

        after = os.path.getsize(signals_path) if os.path.isfile(signals_path) else 0
        assert after > before, "SIGNALS.md should grow after proposal"

    def test_index_db_created(self, workspace):
        """Index DB should exist after build."""
        build_index(workspace)
        db_path = os.path.join(workspace, ".mind-mem-index", "recall.db")
        assert os.path.isfile(db_path), "recall.db should exist after build"

    def test_recall_active_only_filter(self, workspace):
        from mind_mem.recall import recall

        build_index(workspace)
        results = recall(workspace, "database", limit=10, active_only=True)
        assert len(results) > 0
