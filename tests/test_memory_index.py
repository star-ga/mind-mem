"""Tests for the auto-generated hierarchical index (Group C).

``mind_mem.memory_index.generate_index`` regenerates ``index.md``
(hierarchical, by category → kind) + ``log.md`` (chronological) from
the active block corpus via ``storage.iter_active_blocks``. Pinned
behaviour:

  * happy path — both files written, hierarchy + timeline correct;
  * only *active* blocks appear (superseded/pending are excluded);
  * deterministic — output is independent of corpus-write order;
  * idempotent — a second run is byte-identical and reports
    ``index_changed`` / ``log_changed`` False without touching mtimes;
  * empty workspace — valid zero-count documents, no crash;
  * undated blocks land in the log's ``Undated`` section, last;
  * ``mm index`` CLI wrapper prints the JSON summary and exits 0.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mind_mem.memory_index import (
    INDEX_FILENAME,
    LOG_FILENAME,
    IndexResult,
    generate_index,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DECISIONS = """# DECISIONS — MIND-Mem v1.0

[D-20260102-001]
Date: 2026-01-02
Status: active
Statement: Use SQLite as the default backend
Tags: storage

[D-20260101-001]
Date: 2026-01-01
Status: active
Statement: Adopt the src layout
Tags: layout

[D-20250101-001]
Date: 2025-01-01
Status: superseded
Statement: Old decision that must not appear
Tags: stale
"""

_TASKS = """# TASKS — MIND-Mem v1.0

[T-20260103-001]
Date: 2026-01-03
Status: active
Title: Ship the hierarchical index
Priority: P1
"""

_PROJECTS = """# PROJECTS — MIND-Mem v1.0

[PRJ-mindmem]
Status: active
Name: mind-mem
Description: Persistent AI memory
"""


def _write_corpus(ws: Path) -> None:
    (ws / "decisions").mkdir(parents=True, exist_ok=True)
    (ws / "tasks").mkdir(parents=True, exist_ok=True)
    (ws / "entities").mkdir(parents=True, exist_ok=True)
    (ws / "decisions" / "DECISIONS.md").write_text(_DECISIONS, encoding="utf-8")
    (ws / "tasks" / "TASKS.md").write_text(_TASKS, encoding="utf-8")
    (ws / "entities" / "projects.md").write_text(_PROJECTS, encoding="utf-8")


@pytest.fixture
def workspace(tmp_path: Path) -> str:
    ws = tmp_path / "ws"
    ws.mkdir()
    config = {"version": "test", "block_store": {"backend": "markdown"}}
    (ws / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")
    _write_corpus(ws)
    return str(ws)


@pytest.fixture
def empty_workspace(tmp_path: Path) -> str:
    ws = tmp_path / "empty-ws"
    ws.mkdir()
    (ws / "mind-mem.json").write_text(json.dumps({"block_store": {"backend": "markdown"}}), encoding="utf-8")
    return str(ws)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_generates_both_files(workspace: str) -> None:
    result = generate_index(workspace)
    assert isinstance(result, IndexResult)
    assert os.path.isfile(result.index_path)
    assert os.path.isfile(result.log_path)
    assert os.path.basename(result.index_path) == INDEX_FILENAME
    assert os.path.basename(result.log_path) == LOG_FILENAME
    assert result.block_count == 4  # 2 decisions + 1 task + 1 project
    assert result.index_changed is True
    assert result.log_changed is True
    assert result.category_counts == {"decisions": 2, "projects": 1, "tasks": 1}


def test_index_is_hierarchical_by_category_then_kind(workspace: str) -> None:
    result = generate_index(workspace)
    text = Path(result.index_path).read_text(encoding="utf-8")

    # Category headings with counts, alphabetical.
    assert "## decisions (2)" in text
    assert "## projects (1)" in text
    assert "## tasks (1)" in text
    assert text.index("## decisions") < text.index("## projects") < text.index("## tasks")

    # Kind sub-headings under each category (block-id prefixes).
    assert "### D (2)" in text
    assert "### PRJ (1)" in text
    assert "### T (1)" in text

    # Block entries carry id, date, summary and source file.
    assert "**[D-20260101-001]** (2026-01-01) — Adopt the src layout" in text
    assert "`decisions/DECISIONS.md`" in text
    # Within a kind, blocks sort by date (01-01 before 01-02).
    assert text.index("D-20260101-001") < text.index("D-20260102-001")


def test_log_is_chronological_oldest_first(workspace: str) -> None:
    result = generate_index(workspace)
    text = Path(result.log_path).read_text(encoding="utf-8")

    assert "## Timeline" in text
    i1 = text.index("2026-01-01 — **[D-20260101-001]**")
    i2 = text.index("2026-01-02 — **[D-20260102-001]**")
    i3 = text.index("2026-01-03 — **[T-20260103-001]**")
    assert i1 < i2 < i3
    # Each entry names its category and summary.
    assert "(decisions): Adopt the src layout" in text
    assert "(tasks): Ship the hierarchical index" in text


def test_non_active_blocks_are_excluded(workspace: str) -> None:
    result = generate_index(workspace)
    index_text = Path(result.index_path).read_text(encoding="utf-8")
    log_text = Path(result.log_path).read_text(encoding="utf-8")
    assert "D-20250101-001" not in index_text
    assert "D-20250101-001" not in log_text
    assert "Old decision that must not appear" not in index_text


def test_undated_blocks_go_to_undated_section_last(workspace: str) -> None:
    # PRJ-mindmem has no Date field and no YYYYMMDD in its id.
    result = generate_index(workspace)
    text = Path(result.log_path).read_text(encoding="utf-8")
    assert "## Undated" in text
    assert text.index("## Timeline") < text.index("## Undated")
    undated_section = text[text.index("## Undated") :]
    assert "**[PRJ-mindmem]** (projects): mind-mem" in undated_section
    # And it is NOT in the timeline.
    timeline_section = text[text.index("## Timeline") : text.index("## Undated")]
    assert "PRJ-mindmem" not in timeline_section


# ---------------------------------------------------------------------------
# Determinism + idempotence
# ---------------------------------------------------------------------------


def test_second_run_is_idempotent_and_untouched(workspace: str) -> None:
    first = generate_index(workspace)
    index_bytes = Path(first.index_path).read_bytes()
    log_bytes = Path(first.log_path).read_bytes()
    index_mtime = os.path.getmtime(first.index_path)
    log_mtime = os.path.getmtime(first.log_path)

    second = generate_index(workspace)
    assert second.index_changed is False
    assert second.log_changed is False
    assert Path(second.index_path).read_bytes() == index_bytes
    assert Path(second.log_path).read_bytes() == log_bytes
    # No-op runs must not rewrite the files (mtime preserved).
    assert os.path.getmtime(second.index_path) == index_mtime
    assert os.path.getmtime(second.log_path) == log_mtime


def test_output_independent_of_corpus_write_order(tmp_path: Path) -> None:
    """Same blocks, different in-file order → byte-identical documents."""
    ws_a = tmp_path / "a"
    ws_b = tmp_path / "b"
    for ws in (ws_a, ws_b):
        ws.mkdir()
        (ws / "mind-mem.json").write_text(json.dumps({"block_store": {"backend": "markdown"}}), encoding="utf-8")
        (ws / "decisions").mkdir()

    block_1 = "[D-20260101-001]\nDate: 2026-01-01\nStatus: active\nStatement: First\n"
    block_2 = "[D-20260102-001]\nDate: 2026-01-02\nStatus: active\nStatement: Second\n"
    (ws_a / "decisions" / "DECISIONS.md").write_text(f"# D\n\n{block_1}\n{block_2}", encoding="utf-8")
    (ws_b / "decisions" / "DECISIONS.md").write_text(f"# D\n\n{block_2}\n{block_1}", encoding="utf-8")

    res_a = generate_index(str(ws_a))
    res_b = generate_index(str(ws_b))
    assert Path(res_a.index_path).read_bytes() == Path(res_b.index_path).read_bytes()
    assert Path(res_a.log_path).read_bytes() == Path(res_b.log_path).read_bytes()


def test_regenerates_after_corpus_change(workspace: str) -> None:
    generate_index(workspace)
    extra = "\n[D-20260201-001]\nDate: 2026-02-01\nStatus: active\nStatement: Fresh decision\n"
    dec = Path(workspace) / "decisions" / "DECISIONS.md"
    dec.write_text(dec.read_text(encoding="utf-8") + extra, encoding="utf-8")

    result = generate_index(workspace)
    assert result.index_changed is True
    assert result.log_changed is True
    assert result.block_count == 5
    assert "D-20260201-001" in Path(result.index_path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_workspace_produces_zero_count_documents(empty_workspace: str) -> None:
    result = generate_index(empty_workspace)
    assert result.block_count == 0
    assert result.category_counts == {}
    index_text = Path(result.index_path).read_text(encoding="utf-8")
    log_text = Path(result.log_path).read_text(encoding="utf-8")
    assert "# Memory Index" in index_text
    assert "Active blocks: **0**" in index_text
    assert "# Memory Log" in log_text
    # Generated marker present so nobody hand-edits them.
    assert index_text.startswith("<!-- auto-generated")
    assert log_text.startswith("<!-- auto-generated")


def test_out_dir_override(workspace: str, tmp_path: Path) -> None:
    out = tmp_path / "elsewhere" / "docs"
    result = generate_index(workspace, out_dir=str(out))
    assert Path(result.index_path).parent == out.resolve()
    assert (out / INDEX_FILENAME).is_file()
    assert (out / LOG_FILENAME).is_file()
    # Workspace root untouched.
    assert not (Path(workspace) / INDEX_FILENAME).exists()


def test_as_dict_round_trips_through_json(workspace: str) -> None:
    result = generate_index(workspace)
    payload = json.loads(json.dumps(result.as_dict()))
    assert payload["block_count"] == 4
    assert payload["category_counts"]["decisions"] == 2
    assert set(payload) == {
        "index_path",
        "log_path",
        "block_count",
        "index_changed",
        "log_changed",
        "category_counts",
    }


def test_pending_signals_are_excluded(workspace: str) -> None:
    """#429 rule flows through iter_active_blocks: pending signals hidden."""
    intel = Path(workspace) / "intelligence"
    intel.mkdir()
    (intel / "SIGNALS.md").write_text(
        "# SIGNALS\n\n"
        "[SIG-20260110-001]\nDate: 2026-01-10\nStatus: pending\nStatement: Unreviewed signal\n\n"
        "[SIG-20260111-001]\nDate: 2026-01-11\nStatus: active\nStatement: Reviewed signal\n",
        encoding="utf-8",
    )
    result = generate_index(workspace)
    text = Path(result.index_path).read_text(encoding="utf-8")
    assert "SIG-20260111-001" in text
    assert "SIG-20260110-001" not in text


# ---------------------------------------------------------------------------
# CLI wrapper (`mm index`)
# ---------------------------------------------------------------------------


def test_mm_index_cli(workspace: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from mind_mem.mm_cli import main

    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    rc = main(["index"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["block_count"] == 4
    assert payload["index_changed"] is True
    assert (Path(workspace) / INDEX_FILENAME).is_file()
    assert (Path(workspace) / LOG_FILENAME).is_file()


def test_mm_index_cli_out_dir(workspace: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from mind_mem.mm_cli import main

    out = tmp_path / "cli-out"
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    rc = main(["index", "--out-dir", str(out)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["index_path"]).parent == out.resolve()
    assert (out / LOG_FILENAME).is_file()
