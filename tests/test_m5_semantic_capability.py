"""M5 semantic-verification capability boundary controls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.chat_generators import ChatRequest
from mind_mem.chat_memory import chat_with_memory
from mind_mem.edge_grounded_answer import answer as graph_answer
from mind_mem.knowledge_graph import KnowledgeGraph
from mind_mem.semantic_capability import (
    SEMANTIC_VERIFICATION_NOT_ESTABLISHED,
    semantic_entailment_verification_available,
)


def _chat_workspace(tmp_path: Path) -> str:
    workspace = tmp_path / "workspace"
    (workspace / "decisions").mkdir(parents=True)
    (workspace / "decisions" / "DECISIONS.md").write_text("[D-1]\nStatement: blue deploy\nStatus: active\n", encoding="utf-8")
    return str(workspace)


def _recall(_workspace: str, _question: str, _limit: int):
    return [{"_id": "D-1", "excerpt": "blue deploy", "file": "decisions/DECISIONS.md", "score": 1.0}]


def test_capability_is_explicitly_unavailable() -> None:
    assert semantic_entailment_verification_available() is False


def test_structural_grounding_exposes_unestablished_semantics_and_ignores_generator_claim(
    tmp_path: Path,
) -> None:
    result = chat_with_memory(
        _chat_workspace(tmp_path),
        "what deploy?",
        recall_fn=_recall,
        resolver=lambda block_id: block_id == "D-1",
        generator=lambda _request: "The opposite of blue deploy is true [[D-1]].",
    )
    assert result.grounded is True
    assert result.semantic_verification == SEMANTIC_VERIFICATION_NOT_ESTABLISHED
    payload = result.to_dict()
    assert payload["semantic_verification"] == SEMANTIC_VERIFICATION_NOT_ESTABLISHED


def test_semantic_required_chat_abstains_before_generator(tmp_path: Path) -> None:
    called = False

    def generator(_request: ChatRequest) -> str:
        nonlocal called
        called = True
        return "Contradictory prose [[D-1]]."

    result = chat_with_memory(
        _chat_workspace(tmp_path),
        "what deploy?",
        recall_fn=_recall,
        resolver=lambda block_id: block_id == "D-1",
        generator=generator,
        semantic_required=True,
    )
    assert result.rejected is True
    assert result.grounded is False
    assert result.no_record is True
    assert result.semantic_required is True
    assert result.semantic_verification == SEMANTIC_VERIFICATION_NOT_ESTABLISHED
    assert called is False
    assert "unavailable" in result.warnings[0]


def test_mcp_chat_forwards_semantic_required(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from fastmcp.server.auth import AccessToken

    import mind_mem.chat_memory as chat_memory_module
    from mind_mem.init_workspace import init
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools import chat as chat_tool

    init(str(tmp_path))
    token = AccessToken(token="fixture", client_id="fixture", scopes=["user"], claims={"sub": "alice"})
    monkeypatch.setattr("mind_mem.mcp.infra.acl.get_access_token", lambda: token)

    def must_not_recall(*args, **kwargs):
        pytest.fail("semantic-required request reached recall without a verifier")

    monkeypatch.setattr(chat_memory_module, "_default_recall", must_not_recall)
    with use_workspace(str(tmp_path)):
        payload = json.loads(chat_tool.chat_with_memory("q", semantic_required=True))
    assert payload["semantic_required"] is True
    assert payload["semantic_verification"] == "not_established"
    assert payload["grounded"] is False
    assert payload["rejected"] is True
    assert "unavailable" in payload["warnings"][0]


def test_graph_answer_semantic_required_abstains_before_generator(tmp_path: Path, admitted) -> None:
    db_path = tmp_path / "graph.db"
    with KnowledgeGraph(str(db_path)) as graph:
        graph.add_edge("alice", "depends_on", "blue", source_block_id="D-1")
        called = False

        def generator(_context: str) -> str:
            nonlocal called
            called = True
            return "The opposite is true [[E-0000000000000000]]."

        result = graph_answer(graph, "alice", generate_fn=generator, semantic_required=True)

    assert result.refused is True
    assert result.grounded is False
    assert result.semantic_verification == SEMANTIC_VERIFICATION_NOT_ESTABLISHED
    assert called is False


def test_graph_default_remains_structurally_grounded(tmp_path: Path, admitted) -> None:
    with KnowledgeGraph(str(tmp_path / "graph.db")) as graph:
        graph.add_edge("alice", "depends_on", "blue", source_block_id="D-1")
        result = graph_answer(graph, "alice")
    assert result.grounded is True
    assert result.semantic_verification == SEMANTIC_VERIFICATION_NOT_ESTABLISHED


def test_graph_cited_contradiction_does_not_claim_semantic_verification(tmp_path: Path, admitted) -> None:
    with KnowledgeGraph(str(tmp_path / "graph.db")) as graph:
        graph.add_edge("alice", "depends_on", "blue", source_block_id="D-1")
        structural = graph_answer(graph, "alice")
        assert structural.citations
        result = graph_answer(
            graph,
            "alice",
            generate_fn=lambda _context: f"Alice does not depend on blue [[{structural.citations[0]}]].",
        )
    assert result.grounded is True
    assert result.semantic_verification == "not_established"


def test_graph_cli_forwards_semantic_required_and_abstains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, admitted) -> None:
    from mind_mem import mm_cli
    from mind_mem.knowledge_graph import default_db_path

    workspace = Path(_chat_workspace(tmp_path))
    (workspace / "memory").mkdir()
    with KnowledgeGraph(default_db_path(str(workspace))) as graph:
        graph.add_edge("alice", "depends_on", "blue", source_block_id="D-1")
    monkeypatch.setattr(mm_cli, "_workspace", lambda: str(workspace))
    args = mm_cli.build_parser().parse_args(["graph-answer", "alice", "--json", "--semantic-required"])
    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["refused"] is True
    assert payload["semantic_required"] is True
    assert payload["grounded"] is False
