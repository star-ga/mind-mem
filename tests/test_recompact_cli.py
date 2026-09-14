"""Production-path controls for the proposal-only H1 recompaction command."""

from __future__ import annotations

import json

import pytest

from mind_mem import recompact_cli
from mind_mem.mm_cli import main
from mind_mem.recompaction import RecompactionConfig


@pytest.fixture
def blocks() -> list[dict[str, object]]:
    return [
        {"_id": "DEC-001", "body": "alpha fact with enough retained context", "_source_file": "decisions/DECISIONS.md", "_line": 4},
        {"_id": "DEC-002", "body": "beta fact with enough retained context", "_source_file": "memory/2026-09-14.md", "_line": 8},
    ]


def _finder(_block_id: str, _limit: int) -> str:
    return json.dumps({"source": "DEC-001", "similar": [{"block_id": "DEC-002"}], "method": "co-occurrence"})


def _install_blocks(monkeypatch: pytest.MonkeyPatch, blocks: list[dict[str, object]]) -> None:
    import mind_mem.storage as storage

    monkeypatch.setattr(storage, "iter_active_blocks", lambda _workspace: [dict(block) for block in blocks])


def _changed_compressor(text: str, _blocks: list[dict[str, object]]) -> str:
    return "alpha and beta retained as one reviewed proposal" if text != "alpha and beta retained as one reviewed proposal" else text


def test_similarity_cluster_reloads_active_bodies_and_source_coordinates(monkeypatch, blocks):
    _install_blocks(monkeypatch, blocks)
    cluster = recompact_cli.resolve_similarity_cluster("/tmp/workspace", "DEC-001", finder=_finder)
    assert [item["_id"] for item in cluster] == ["DEC-001", "DEC-002"]
    assert cluster[1]["_source_file"] == "memory/2026-09-14.md"


def test_stale_similarity_id_is_refused(monkeypatch, blocks):
    _install_blocks(monkeypatch, blocks)

    def stale_finder(_block_id: str, _limit: int) -> str:
        return json.dumps({"similar": [{"block_id": "DEC-404"}]})

    with pytest.raises(recompact_cli.RecompactError, match="outside the active corpus"):
        recompact_cli.resolve_similarity_cluster("/tmp/workspace", "DEC-001", finder=stale_finder)


def test_mm_recompact_is_dry_run_by_default_and_contains_provenance(monkeypatch, blocks, tmp_path, capsys):
    _install_blocks(monkeypatch, blocks)
    monkeypatch.setattr(recompact_cli, "compressor_for", lambda _name, _model: _changed_compressor)
    import mind_mem.mcp.tools.recall as recall_tools

    monkeypatch.setattr(recall_tools, "find_similar", _finder)
    workspace = str(tmp_path)

    assert main(["recompact", "DEC-001", "--workspace", workspace, "--min-retention-ratio", "0"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "proposal"
    assert payload["write"] == "not_written"
    assert payload["requires_approval"] is True
    assert payload["semantic_verification"] == "not_established"
    assert payload["source_ids"] == ["DEC-001", "DEC-002"]
    assert {item["source_file"] for item in payload["sources"]} == {"decisions/DECISIONS.md", "memory/2026-09-14.md"}
    assert not (tmp_path / "intelligence" / "SIGNALS.md").exists()


def test_staging_uses_existing_governed_proposal_service(monkeypatch, blocks, tmp_path):
    _install_blocks(monkeypatch, blocks)
    payload = recompact_cli.make_recompact_proposal(
        str(tmp_path),
        "DEC-001",
        compressor=_changed_compressor,
        config=RecompactionConfig(min_retention_ratio=0),
        finder=_finder,
    )
    calls: list[dict[str, object]] = []

    def fake_propose(block_type: str, statement: str, **kwargs: object) -> str:
        calls.append({"block_type": block_type, "statement": statement, **kwargs})
        return json.dumps({"status": "proposed", "written": True})

    import mind_mem.mcp.tools.governance as governance

    monkeypatch.setattr(governance, "propose_update", fake_propose)
    staged = recompact_cli.stage_recompact_proposal(str(tmp_path), payload)
    assert staged["write"] == "staged"
    assert calls and calls[0]["block_type"] == "task"
    assert "DEC-001" in str(calls[0]["rationale"])
    assert calls[0]["content_source"] == "agent"


def test_staging_refuses_writer_truncation(monkeypatch, tmp_path):
    payload = {
        "status": "proposal",
        "text": "x" * 501,
        "source_ids": ["DEC-001", "DEC-002"],
        "mode": "recompact",
        "input_digest": "a" * 64,
    }
    staged = recompact_cli.stage_recompact_proposal(str(tmp_path), payload)
    assert staged["write"] == "refused"
    assert "governed statement limit" in staged["error"]
    assert not (tmp_path / "intelligence" / "SIGNALS.md").exists()


def test_staging_exercises_real_proposal_service_in_temp_workspace(monkeypatch, tmp_path):
    for directory in ("decisions", "entities", "intelligence", "tasks"):
        (tmp_path / directory).mkdir()
    (tmp_path / "decisions" / "DECISIONS.md").write_text("", encoding="utf-8")
    (tmp_path / "intelligence" / "SIGNALS.md").write_text("", encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    payload = {
        "status": "proposal",
        "text": "A compact reviewed task with enough context",
        "source_ids": ["DEC-001", "DEC-002"],
        "mode": "recompact",
        "input_digest": "b" * 64,
    }
    staged = recompact_cli.stage_recompact_proposal(str(tmp_path), payload)
    assert staged["write"] == "staged"
    signals = (tmp_path / "intelligence" / "SIGNALS.md").read_text(encoding="utf-8")
    assert "Status: pending" in signals
    assert "source_ids=DEC-001,DEC-002" in signals
    assert "Semantic verification is not established" in signals
    assert not (tmp_path / "decisions" / "DECISIONS.md").read_text(encoding="utf-8")


def test_unbounded_compressor_output_is_rejected_before_proposal(monkeypatch, blocks):
    _install_blocks(monkeypatch, blocks)

    def too_large(_text: str, _blocks: list[dict[str, object]]) -> str:
        return "x" * 10_001

    with pytest.raises(recompact_cli.RecompactError, match="exceeds"):
        recompact_cli.make_recompact_proposal(
            "/tmp/workspace",
            "DEC-001",
            compressor=too_large,
            config=RecompactionConfig(min_retention_ratio=0),
            finder=_finder,
        )


def test_dream_cycle_recompact_pass_is_opt_in_and_proposal_only(monkeypatch, blocks, tmp_path):
    _install_blocks(monkeypatch, blocks)
    import mind_mem.dream_cycle as dream_cycle
    import mind_mem.mcp.tools.recall as recall_tools

    monkeypatch.setattr(recall_tools, "find_similar", _finder)
    report = dream_cycle.run_dream_cycle(
        str(tmp_path),
        dry_run=True,
        recompact_ids=["DEC-001"],
        instant=dream_cycle.datetime(2026, 9, 14, 12, 0, 0),
    )
    assert len(report.recompaction_proposals) == 1
    assert report.recompaction_proposals[0]["mode"] == "dream"
    assert report.recompaction_proposals[0]["write"] == "not_written"
    assert not (tmp_path / "intelligence" / "SIGNALS.md").exists()
