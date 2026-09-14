"""Production-caller controls for the optional K3 graph/chat bridge.

The graph mode is opt-in. These tests use a real workspace graph and the
public Python chat function, with only the answerer/retrieval seams stubbed
so no model or network call is made.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from mind_mem.chat_generators import ChatRequest
from mind_mem.chat_memory import chat_with_memory
from mind_mem.edge_grounded_answer import build_context
from mind_mem.governance_gate import get_gate
from mind_mem.knowledge_graph import KnowledgeGraph, default_db_path


def _block(path: Path, block_id: str, statement: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"[{block_id}]\nStatement: {statement}\nStatus: active\nDate: 2026-09-14\n\n",
        encoding="utf-8",
    )


def _workspace(tmp_path: Path, *, source: str = "decisions/DECISIONS.md", block_id: str = "D-GRAPH") -> Path:
    ws = tmp_path / "k3-workspace"
    for name in ("decisions", "shared/decisions", "agents/alice/decisions", "memory"):
        (ws / name).mkdir(parents=True, exist_ok=True)
    _block(ws / source, block_id, "the graph edge has a canonical source document")
    (ws / "mind-mem.json").write_text('{"recall": {"backend": "bm25"}}', encoding="utf-8")
    return ws


def _edge(
    ws: Path,
    source_block_id: str,
    *,
    subject: str = "alice",
    object_: str = "project",
    valid_until: str | None = None,
) -> None:
    gate = get_gate(str(ws))
    with gate.admit_proposal(proposal_id="K3-TEST", content="graph edge test", actor="pytest"):
        graph = KnowledgeGraph(default_db_path(str(ws)))
        try:
            graph.add_edge(
                subject,
                "depends_on",
                object_,
                source_block_id=source_block_id,
                valid_until=valid_until,
            )
        finally:
            graph.close()


def test_graph_seed_feeds_chat_with_canonical_provenance_document(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _edge(ws, "D-GRAPH")
    prompts: list[str] = []

    def generator(request: ChatRequest) -> str:
        prompts.append(request.prompt)
        return "Alice depends on project [[D-GRAPH]]."

    result = chat_with_memory(
        str(ws),
        "what does Alice depend on?",
        graph_seed="alice",
        recall_fn=lambda *_args: [],
        generator=generator,
        on_invalid="reject",
    )

    assert result.grounded is True
    assert result.citations == ("D-GRAPH",)
    assert result.graph_evidence is not None
    assert result.graph_evidence["status"] == "served"
    triple = result.graph_evidence["context"]["triples"][0]
    assert triple["source_block_id"] == "D-GRAPH"
    assert triple["edge_id"].startswith("E-")
    assert "supporting document [[D-GRAPH]]" in prompts[0]
    assert "semantic_verification" not in prompts[0]


def test_graph_chat_rebuilds_same_id_excerpt_from_canonical_source(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _edge(ws, "D-GRAPH")
    seen: list[str] = []

    def generator(request: ChatRequest) -> str:
        seen.append(request.prompt)
        return "The canonical graph source is present [[D-GRAPH]]."

    result = chat_with_memory(
        str(ws),
        "what does Alice depend on?",
        graph_seed="alice",
        recall_fn=lambda *_args: [{"_id": "D-GRAPH", "excerpt": "FORGED PRIVATE TEXT"}],
        generator=generator,
        on_invalid="reject",
    )

    assert result.grounded is True
    assert "FORGED PRIVATE TEXT" not in seen[0]
    assert "the graph edge has a canonical source document" in seen[0]


def test_graph_chat_rejects_citation_to_unrelated_recalled_block(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _block(ws / "shared/decisions/OTHER.md", "D-OTHER", "unrelated evidence")
    _edge(ws, "D-GRAPH")

    result = chat_with_memory(
        str(ws),
        "what does Alice depend on?",
        graph_seed="alice",
        recall_fn=lambda *_args: [{"_id": "D-OTHER", "excerpt": "unrelated evidence"}],
        generator=lambda _request: "Unrelated [[D-OTHER]].",
        on_invalid="reject",
    )

    assert result.rejected is True
    assert result.answer == "no record found"
    assert result.graph_evidence["context"]["triples"]


def test_graph_chat_abstains_when_edge_provenance_document_is_missing(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _edge(ws, "D-MISSING")

    result = chat_with_memory(
        str(ws),
        "what does Alice depend on?",
        graph_seed="alice",
        recall_fn=lambda *_args: [],
        generator=lambda _request: (_ for _ in ()).throw(AssertionError("must abstain")),
        on_invalid="reject",
    )

    assert result.rejected is True
    assert result.no_record is True
    assert "graph provenance unavailable" in result.warnings[0]
    assert "context" not in result.graph_evidence
    assert "D-MISSING" not in json.dumps(result.to_dict())


def test_graph_chat_excludes_private_corroboration_from_visible_claim(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, source="shared/decisions/ALICE.md", block_id="D-ALICE")
    _block(ws / "agents/bob/decisions/DECISIONS.md", "D-BOB", "private bob evidence")
    (ws / "mind-mem-acl.json").write_text(
        json.dumps({"default_policy": "read", "agents": {"alice": {"read": ["shared"]}}}),
        encoding="utf-8",
    )
    _edge(ws, "D-ALICE")
    _edge(ws, "D-BOB")

    result = chat_with_memory(
        str(ws),
        "what does Alice depend on?",
        graph_seed="alice",
        agent_id="alice",
        recall_fn=lambda *_args: [],
        generator=lambda _request: "Alice depends on project [[D-ALICE]].",
        on_invalid="reject",
    )

    rendered = json.dumps(result.to_dict())
    assert result.grounded is True
    assert result.rejected is False
    assert result.citations == ("D-ALICE",)
    assert "D-ALICE" in rendered
    assert "D-BOB" not in rendered
    assert "private bob evidence" not in rendered
    triple = result.graph_evidence["context"]["triples"][0]
    assert triple["corroborating_blocks"] == ["D-ALICE"]


def test_graph_chat_refuses_private_intermediate_edge(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, source="shared/decisions/ALICE.md", block_id="D-ALICE")
    _block(ws / "agents/bob/decisions/DECISIONS.md", "D-BOB", "private intermediate")
    (ws / "mind-mem-acl.json").write_text(
        json.dumps({"default_policy": "read", "agents": {"alice": {"read": ["shared"]}}}),
        encoding="utf-8",
    )
    _edge(ws, "D-BOB", object_="bob")
    _edge(ws, "D-ALICE", subject="bob", object_="project")

    result = chat_with_memory(
        str(ws),
        "what does Alice depend on?",
        graph_seed="alice",
        agent_id="alice",
        recall_fn=lambda *_args: [],
        generator=lambda _request: (_ for _ in ()).throw(AssertionError("must abstain")),
        on_invalid="reject",
    )

    assert result.rejected is True
    assert result.no_record is True
    assert "context" not in result.graph_evidence
    assert "D-BOB" not in json.dumps(result.to_dict())


def test_graph_chat_refuses_malformed_graph_without_invoking_generator(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    graph_path = Path(default_db_path(str(ws)))
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_bytes(b"not a sqlite database")

    result = chat_with_memory(
        str(ws),
        "what does Alice depend on?",
        graph_seed="alice",
        generator=lambda _request: (_ for _ in ()).throw(AssertionError("must abstain")),
        on_invalid="reject",
    )

    assert result.rejected is True
    assert result.no_record is True
    assert "unavailable" in result.warnings[0]


def test_graph_chat_enforces_agent_namespace_before_graph_support(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, source="agents/alice/decisions/DECISIONS.md", block_id="D-ALICE")
    _block(ws / "agents/bob/decisions/DECISIONS.md", "D-BOB", "private bob evidence")
    (ws / "mind-mem-acl.json").write_text(
        json.dumps(
            {
                "default_policy": "read",
                "agents": {
                    "alice": {"read": ["shared", "agents/alice"], "namespaces": ["shared", "agents/alice"]},
                },
            }
        ),
        encoding="utf-8",
    )
    _edge(ws, "D-BOB")

    result = chat_with_memory(
        str(ws),
        "what does Alice depend on?",
        graph_seed="alice",
        agent_id="alice",
        recall_fn=lambda *_args: [],
        generator=lambda _request: "private [[D-BOB]].",
        on_invalid="reject",
    )

    assert result.rejected is True
    assert result.no_record is True
    assert "provenance document" in result.warnings[0] or "graph" in result.warnings[0]


def test_graph_refusal_payload_does_not_disclose_foreign_triple(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, source="agents/alice/decisions/DECISIONS.md", block_id="D-ALICE")
    _block(ws / "agents/bob/decisions/DECISIONS.md", "D-BOB", "private bob evidence")
    (ws / "mind-mem-acl.json").write_text(
        json.dumps({"default_policy": "read", "agents": {"alice": {"read": ["agents/alice"]}}}),
        encoding="utf-8",
    )
    _edge(ws, "D-BOB", object_="bob_private_project")

    result = chat_with_memory(
        str(ws),
        "what does Alice depend on?",
        graph_seed="alice",
        agent_id="alice",
        recall_fn=lambda *_args: [],
        generator=lambda _request: (_ for _ in ()).throw(AssertionError("must abstain")),
        on_invalid="reject",
    )
    rendered = json.dumps(result.to_dict())
    assert result.rejected is True
    assert "D-BOB" not in rendered
    assert "bob_private_project" not in rendered
    assert "E-" not in rendered
    assert "graph provenance unavailable" in rendered


def test_graph_mode_cannot_disable_its_provenance_citation_scope(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _edge(ws, "D-GRAPH")
    other_source = str((ws / "shared/decisions/OTHER.md").resolve())

    result = chat_with_memory(
        str(ws),
        "what does Alice depend on?",
        graph_seed="alice",
        recall_fn=lambda *_args: [
            {
                "_id": "D-OTHER",
                "excerpt": "unrelated evidence",
                "file": other_source,
                "_source_file": other_source,
            }
        ],
        generator=lambda _request: "Unrelated [[D-OTHER]].",
        require_in_evidence=False,
        on_invalid="reject",
    )
    assert result.rejected is True
    assert result.grounded is False
    assert result.report is not None and result.report.out_of_evidence == ("D-OTHER",)


def test_graph_mode_rechecks_edges_after_generation(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _edge(ws, "D-GRAPH")
    graph_path = default_db_path(str(ws))

    def delete_edge(_request: ChatRequest) -> str:
        with sqlite3.connect(graph_path) as connection:
            connection.execute(
                "DELETE FROM edges WHERE source_block_id = ?",
                ("D-GRAPH",),
            )
            connection.commit()
        return "The graph relation exists [[D-GRAPH]]."

    result = chat_with_memory(
        str(ws),
        "what does Alice depend on?",
        graph_seed="alice",
        recall_fn=lambda *_args: [],
        generator=delete_edge,
        on_invalid="reject",
    )
    assert result.rejected is True
    assert result.grounded is False
    assert result.answer == "no record found"
    assert result.graph_evidence["status"] == "unproven"
    assert "changed during generation" in result.graph_evidence["error"]


def test_graph_chat_does_not_return_revoked_snapshot_after_generation(tmp_path: Path) -> None:
    source = "shared/decisions/DECISIONS.md"
    ws = _workspace(tmp_path, source=source)
    (ws / "mind-mem-acl.json").write_text(
        json.dumps({"default_policy": "read", "agents": {"alice": {"read": ["shared"]}}}),
        encoding="utf-8",
    )
    _edge(ws, "D-GRAPH")
    source_path = ws / source

    def revoke_source(_request: ChatRequest) -> str:
        source_path.write_text(
            "[D-GRAPH]\nStatement: the graph edge has been revoked\nStatus: revoked\nDate: 2026-09-14\n\n",
            encoding="utf-8",
        )
        return "The prior graph relation exists [[D-GRAPH]]."

    result = chat_with_memory(
        str(ws),
        "what does Alice depend on?",
        graph_seed="alice",
        agent_id="alice",
        recall_fn=lambda *_args: [],
        generator=revoke_source,
        on_invalid="reject",
    )

    rendered = json.dumps(result.to_dict())
    assert result.rejected is True
    assert result.no_record is True
    assert result.evidence == ()
    assert result.graph_evidence["status"] == "unproven"
    assert "context" not in result.graph_evidence
    assert "D-GRAPH" not in rendered
    assert "canonical source" not in rendered
    assert "revoked" not in rendered


def test_graph_chat_unbound_still_withholds_quarantined_source(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _edge(ws, "D-GRAPH")
    (ws / "decisions/DECISIONS.md").write_text(
        "[D-GRAPH]\nStatement: the graph source is quarantined\nStatus: quarantined\nDate: 2026-09-14\n\n",
        encoding="utf-8",
    )

    result = chat_with_memory(
        str(ws),
        "what does Alice depend on?",
        graph_seed="alice",
        recall_fn=lambda *_args: [],
        generator=lambda _request: (_ for _ in ()).throw(AssertionError("must abstain")),
        on_invalid="reject",
    )

    assert result.rejected is True
    assert result.no_record is True
    assert result.evidence == ()
    assert "context" not in result.graph_evidence
    assert "D-GRAPH" not in json.dumps(result.to_dict())


def test_graph_projection_excludes_private_expired_edge_before_gap_counting(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _block(ws / "shared/decisions/PRIVATE.md", "D-PRIVATE", "private expired edge")
    _edge(ws, "D-GRAPH")
    _edge(ws, "D-PRIVATE", object_="private", valid_until="2020-01-01T00:00:00+00:00")
    graph = KnowledgeGraph.open_read_only(default_db_path(str(ws)))
    try:
        context = build_context(graph, "alice", admitted_source_ids={"D-GRAPH"})
        empty = build_context(graph, "alice", admitted_source_ids=set())
    finally:
        graph.close()
    assert context.triples
    assert all(triple.source_block_id == "D-GRAPH" for triple in context.triples)
    assert all("expired" not in gap.kind for gap in context.gaps)
    assert "private" not in json.dumps(context.as_dict())
    assert empty.triples == ()
    assert "private" not in json.dumps(empty.as_dict())


def test_graph_chat_rechecks_admission_after_recall_before_generator(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _edge(ws, "D-GRAPH")
    source_path = ws / "decisions/DECISIONS.md"

    def quarantine_during_recall(*_args: object) -> list[dict[str, str]]:
        source_path.write_text(
            "[D-GRAPH]\nStatement: quarantined during recall\nStatus: quarantined\nDate: 2026-09-14\n\n",
            encoding="utf-8",
        )
        return []

    result = chat_with_memory(
        str(ws),
        "what does Alice depend on?",
        graph_seed="alice",
        recall_fn=quarantine_during_recall,
        generator=lambda _request: (_ for _ in ()).throw(AssertionError("must abstain")),
        on_invalid="reject",
    )
    assert result.rejected is True
    assert result.answer == "no record found"
    assert result.evidence == ()
    assert "context" not in result.graph_evidence


def test_graph_chat_rechecks_admission_after_condenser_before_generator(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _edge(ws, "D-GRAPH")
    source_path = ws / "decisions/DECISIONS.md"

    def quarantine_during_condense(_prompt: str) -> str:
        source_path.write_text(
            "[D-GRAPH]\nStatement: quarantined during condensation\nStatus: quarantined\nDate: 2026-09-14\n\n",
            encoding="utf-8",
        )
        return "[1]"

    result = chat_with_memory(
        str(ws),
        "what does Alice depend on?",
        graph_seed="alice",
        recall_fn=lambda *_args: [],
        condenser=quarantine_during_condense,
        generator=lambda _request: (_ for _ in ()).throw(AssertionError("must abstain")),
        on_invalid="reject",
    )
    assert result.rejected is True
    assert result.answer == "no record found"
    assert result.evidence == ()
    assert "context" not in result.graph_evidence


def test_registered_mcp_chat_exposes_opt_in_graph_evidence(tmp_path: Path) -> None:
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools import chat as chat_tools

    ws = _workspace(tmp_path)
    _edge(ws, "D-GRAPH")
    with use_workspace(str(ws)):
        payload = json.loads(chat_tools.chat_with_memory.__wrapped__("what does Alice depend on?", graph_seed="alice"))

    assert payload["grounded"] is True
    assert payload["graph_evidence"]["status"] == "served"
    assert "D-GRAPH" in payload["citations"]
