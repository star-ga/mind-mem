"""Production-path controls for the proposal-only H1 recompaction command."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

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

    def active(workspace: str) -> list[dict[str, object]]:
        for block in blocks:
            source = Path(workspace) / str(block["_source_file"])
            source.parent.mkdir(parents=True, exist_ok=True)
            if not source.exists():
                source.write_text(str(block["body"]) + "\n", encoding="utf-8")
        return [dict(block) for block in blocks]

    monkeypatch.setattr(storage, "iter_active_blocks", active)


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


def test_finder_duplicate_and_malformed_ids_are_refused(monkeypatch, blocks):
    _install_blocks(monkeypatch, blocks)

    def duplicate(_id, _limit):
        return {"similar": [{"block_id": "DEC-002"}, {"block_id": "DEC-002"}]}

    def malformed(_id, _limit):
        return {"similar": [{"block_id": "not-an-id"}]}

    with pytest.raises(recompact_cli.RecompactError, match="duplicate"):
        recompact_cli.resolve_similarity_cluster("/tmp/workspace", "DEC-001", finder=duplicate)
    with pytest.raises(recompact_cli.RecompactError, match="malformed"):
        recompact_cli.resolve_similarity_cluster("/tmp/workspace", "DEC-001", finder=malformed)


def test_finder_result_must_respect_requested_limit(monkeypatch, blocks):
    _install_blocks(monkeypatch, blocks)

    def finder(_id, _limit):
        return {"similar": [{"block_id": "DEC-002"}, {"block_id": "DEC-001"}]}

    with pytest.raises(recompact_cli.RecompactError, match="limit=1"):
        recompact_cli.resolve_similarity_cluster("/tmp/workspace", "DEC-001", limit=1, finder=finder)


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
    assert "content_source" not in calls[0]


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


def test_staging_rejects_forged_output_and_stale_source(monkeypatch, blocks, tmp_path):
    _install_blocks(monkeypatch, blocks)
    payload = recompact_cli.make_recompact_proposal(
        str(tmp_path), "DEC-001", compressor=_changed_compressor, config=RecompactionConfig(min_retention_ratio=0), finder=_finder
    )
    forged = {**payload, "output_digest": hashlib.sha256(b"forged").hexdigest()}
    with pytest.raises(recompact_cli.RecompactError, match="output_digest"):
        recompact_cli.stage_recompact_proposal(str(tmp_path), forged)
    (tmp_path / "decisions" / "DECISIONS.md").write_text("changed\n", encoding="utf-8")
    with pytest.raises(recompact_cli.RecompactError, match="changed|stale"):
        recompact_cli.stage_recompact_proposal(str(tmp_path), payload)


def test_recompaction_rechecks_sources_after_compressor(monkeypatch, blocks, tmp_path):
    _install_blocks(monkeypatch, blocks)
    changed = {"done": False}

    def mutating(text: str, _blocks: list[dict[str, object]]) -> str:
        if not changed["done"]:
            (tmp_path / "decisions" / "DECISIONS.md").write_text("mutated\n", encoding="utf-8")
            changed["done"] = True
        return text

    with pytest.raises(recompact_cli.RecompactError, match="source changed"):
        recompact_cli.make_recompact_proposal(
            str(tmp_path), "DEC-001", compressor=mutating, config=RecompactionConfig(min_retention_ratio=0), finder=_finder
        )


def test_staging_passes_only_caller_provided_provenance(monkeypatch, blocks, tmp_path):
    _install_blocks(monkeypatch, blocks)
    payload = recompact_cli.make_recompact_proposal(
        str(tmp_path), "DEC-001", compressor=_changed_compressor, config=RecompactionConfig(min_retention_ratio=0), finder=_finder
    )
    import mind_mem.mcp.tools.governance as governance

    seen: dict[str, object] = {}
    monkeypatch.setattr(governance, "propose_update", lambda *_args, **kwargs: seen.update(kwargs) or json.dumps({"status": "proposed"}))
    recompact_cli.stage_recompact_proposal(
        str(tmp_path),
        payload,
        provenance={"actor_id": "operator-7", "actor_role": "operator", "session_id": "s-7", "tool_id": "mm", "purpose": "review"},
    )
    assert seen["actor_id"] == "operator-7"
    assert seen["session_id"] == "s-7"
    assert "content_source" not in seen


def test_source_backed_workspace_reaches_real_stage(monkeypatch, tmp_path):
    from mind_mem.init_workspace import init

    init(str(tmp_path))
    corpus = tmp_path / "decisions" / "DECISIONS.md"
    corpus.write_text(
        "[DEC-20260914-001]\nStatement: alpha source fact\nStatus: active\nDate: 2026-09-14\n\n"
        "[DEC-20260914-002]\nStatement: beta source fact\nStatus: active\nDate: 2026-09-14\n\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    payload = recompact_cli.make_recompact_proposal(
        str(tmp_path),
        "DEC-20260914-001",
        compressor=lambda _text, _blocks: "alpha and beta source facts",
        config=RecompactionConfig(min_retention_ratio=0),
        finder=lambda _id, _limit: {"similar": [{"block_id": "DEC-20260914-002"}]},
    )
    staged = recompact_cli.stage_recompact_proposal(
        str(tmp_path),
        payload,
        provenance={"actor_id": "operator-1", "actor_role": "operator", "session_id": "session-1", "tool_id": "test", "purpose": "review"},
    )
    assert staged["write"] == "staged"
    signals = (tmp_path / "intelligence" / "SIGNALS.md").read_text(encoding="utf-8")
    assert "ActorId: operator-1" in signals
    assert "SessionId: session-1" in signals
    assert "alpha and beta source facts" in signals


def test_cli_returns_failure_when_governed_stage_refuses(monkeypatch, tmp_path, capsys):
    import mind_mem.recompact_cli as module

    payload = {"status": "proposal", "write": "not_written"}
    monkeypatch.setattr(module, "make_recompact_proposal", lambda *args, **kwargs: payload)
    monkeypatch.setattr(module, "stage_recompact_proposal", lambda *args, **kwargs: {**payload, "write": "refused"})
    assert main(["recompact", "DEC-001", "--workspace", str(tmp_path), "--stage"]) == 1
    assert json.loads(capsys.readouterr().out)["write"] == "refused"


def test_staging_exercises_real_proposal_service_in_temp_workspace(monkeypatch, tmp_path, blocks):
    for directory in ("decisions", "entities", "intelligence", "tasks"):
        (tmp_path / directory).mkdir()
    (tmp_path / "decisions" / "DECISIONS.md").write_text("", encoding="utf-8")
    (tmp_path / "intelligence" / "SIGNALS.md").write_text("", encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    _install_blocks(monkeypatch, blocks)
    payload = recompact_cli.make_recompact_proposal(
        str(tmp_path),
        "DEC-001",
        compressor=_changed_compressor,
        config=RecompactionConfig(min_retention_ratio=0),
        finder=_finder,
    )
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


@pytest.mark.parametrize("bad", ["\u200b", "\u202e", "\x00", "\ud800"])
def test_compressor_rejects_invisible_or_surrogate_controls(monkeypatch, blocks, bad):
    _install_blocks(monkeypatch, blocks)

    def unsafe(_text: str, _blocks: list[dict[str, object]]) -> str:
        return "safe" + bad

    with pytest.raises(recompact_cli.RecompactError, match="control"):
        recompact_cli.make_recompact_proposal(
            "/tmp/workspace", "DEC-001", compressor=unsafe, config=RecompactionConfig(min_retention_ratio=0), finder=_finder
        )


def test_compressor_allows_structural_newlines_and_tabs(monkeypatch, blocks):
    _install_blocks(monkeypatch, blocks)
    result = recompact_cli.make_recompact_proposal(
        "/tmp/workspace",
        "DEC-001",
        compressor=lambda _text, _blocks: "line 1\n\tline 2",
        config=RecompactionConfig(min_retention_ratio=0),
        finder=_finder,
    )
    assert result["text"] == "line 1\n\tline 2"


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
