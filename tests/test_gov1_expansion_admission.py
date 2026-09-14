"""Workspace-bound admission controls for graph and KG result splicing."""

from __future__ import annotations

import json
from pathlib import Path

from mind_mem.init_workspace import init

SEED = "D-20260914-001"
NEIGHBOUR = "D-20260914-002"


def _workspace(tmp_path: Path, *, neighbour_status: str) -> str:
    workspace = tmp_path / "workspace"
    init(str(workspace))
    config_path = workspace / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["recall"]["validity_gate"] = {
        "enabled": True,
        "content_categories": {
            "enabled": True,
            "ttl_days": {"infra": 2, "status": 1},
        },
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (workspace / "decisions/DECISIONS.md").write_text(
        "\n".join(
            (
                f"[{SEED}]",
                "Status: active",
                f"Statement: seed references {NEIGHBOUR}",
                "ContentCategory: decision",
                "",
                f"[{NEIGHBOUR}]",
                f"Status: {neighbour_status}",
                "Statement: credential neighbour",
                "ContentCategory: credential",
                "ContentValidFrom: 2020-01-01",
                "",
            )
        ),
        encoding="utf-8",
    )
    return str(workspace)


def _corpus(neighbour_status: str = "active") -> list[dict]:
    source = "decisions/DECISIONS.md"
    return [
        {
            "_id": SEED,
            "Status": "active",
            "ContentCategory": "decision",
            "Statement": f"seed references {NEIGHBOUR}",
            "_source_file": source,
        },
        {
            "_id": NEIGHBOUR,
            # Deliberately stale/index-like metadata: the workspace source is
            # the authority that graph/KG expansion must re-read.
            "Status": "active" if neighbour_status == "revoked" else neighbour_status,
            "ContentCategory": "credential",
            "ContentValidFrom": "2020-01-01",
            "Statement": "credential neighbour",
            "_source_file": source,
        },
    ]


def test_graph_expansion_rechecks_current_workspace_revocation(tmp_path: Path) -> None:
    from mind_mem.graph_recall import graph_expand

    workspace = _workspace(tmp_path, neighbour_status="active")
    seeds = [{"_id": SEED, "score": 1.0}]
    first = graph_expand(seeds, _corpus(), workspace=workspace, max_hops=1)
    assert [row["_id"] for row in first] == [SEED, NEIGHBOUR]

    source = Path(workspace) / "decisions/DECISIONS.md"
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "Status: active\nStatement: credential", "Status: revoked\nStatement: credential"
        ),
        encoding="utf-8",
    )
    after_revoke = graph_expand(seeds, _corpus(), workspace=workspace, max_hops=1)
    assert [row["_id"] for row in after_revoke] == [SEED]


def test_kg_expansion_withholds_revoked_source_and_keeps_active_positive(tmp_path: Path) -> None:
    from mind_mem.governance_gate import get_gate
    from mind_mem.kg_fusion import kg_expand
    from mind_mem.knowledge_graph import KnowledgeGraph

    workspace = _workspace(tmp_path, neighbour_status="revoked")
    db_path = Path(workspace) / "kg.db"
    with get_gate(workspace).admit_proposal("TEST-GOV1", "[]", actor="pytest"):
        kg = KnowledgeGraph(str(db_path))
        try:
            kg.add_edge("starga", "depends_on", "mindc", source_block_id=NEIGHBOUR)
            denied = kg_expand(
                [{"_id": SEED, "score": 1.0}],
                _corpus(),
                kg,
                "starga",
                workspace=workspace,
                max_hops=1,
            )
        finally:
            kg.close()
    assert [row["_id"] for row in denied] == [SEED]

    workspace_active = _workspace(tmp_path / "active", neighbour_status="active")
    with get_gate(workspace_active).admit_proposal("TEST-GOV1", "[]", actor="pytest"):
        active_kg = KnowledgeGraph(str(Path(workspace_active) / "kg.db"))
        try:
            active_kg.add_edge("starga", "depends_on", "mindc", source_block_id=NEIGHBOUR)
            allowed = kg_expand(
                [{"_id": SEED, "score": 1.0}],
                _corpus(),
                active_kg,
                "starga",
                workspace=workspace_active,
                max_hops=1,
            )
        finally:
            active_kg.close()
    assert [row["_id"] for row in allowed] == [SEED, NEIGHBOUR]


def test_direct_helpers_preserve_pre_admitted_legacy_contract() -> None:
    from mind_mem.graph_recall import graph_expand

    corpus = _corpus()
    seeds = [{"_id": SEED, "score": 1.0}]
    # No workspace means the caller owns the pre-admission contract. Existing
    # pure helper callers keep their historical behavior and do not gain a
    # hidden filesystem read.
    assert [row["_id"] for row in graph_expand(seeds, corpus, max_hops=1)] == [SEED, NEIGHBOUR]
