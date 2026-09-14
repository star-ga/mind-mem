"""Source-bound M2/M3 namespace reachability and floor controls."""

from __future__ import annotations

import json
from pathlib import Path

from mind_mem._recall_core import recall
from mind_mem.block_store import MarkdownBlockStore
from mind_mem.init_workspace import init
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.recall import pack_recall_budget


def _block(path: Path, block_id: str, block_type: str, statement: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"[{block_id}]\nType: {block_type}\nStatement: {statement}\nStatus: active\n\n",
        encoding="utf-8",
    )


def _config(ws: Path, properties: dict[str, object], **recall: object) -> None:
    cfg = json.loads((ws / "mind-mem.json").read_text(encoding="utf-8"))
    cfg["recall"].update(recall)
    cfg["recall"]["namespace_properties"] = properties
    (ws / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")


def test_direct_only_excludes_matching_and_broad_search_but_store_get_survives(tmp_path: Path) -> None:
    ws = tmp_path / "direct-only"
    init(str(ws))
    _block(ws / "decisions/DECISIONS.md", "DIRECT-1", "Decision", "private namespace marker")
    _config(ws, {"workspace": {"reachability": "direct-only", "floor": "none"}})

    direct = MarkdownBlockStore(str(ws)).get_by_id("DIRECT-1")
    assert direct is not None
    matching = recall(str(ws), "private namespace marker", limit=10, rerank=False)
    broad = recall(str(ws), "namespace", limit=10, rerank=False)
    print(
        json.dumps(
            {
                "matching_scores": [(x["_id"], x["score"]) for x in matching],
                "broad_scores": [(x["_id"], x["score"]) for x in broad],
            },
            sort_keys=True,
        )
    )
    assert all(hit["_id"] != "DIRECT-1" for hit in matching)
    assert all(hit["_id"] != "DIRECT-1" for hit in broad)


def test_searchable_namespace_prints_positive_score_and_numeric_floor_requires_evidence(tmp_path: Path) -> None:
    ws = tmp_path / "searchable"
    init(str(ws))
    _block(ws / "decisions/DECISIONS.md", "SEARCH-1", "Decision", "searchable namespace marker")
    _config(
        ws,
        {
            "workspace": {
                "reachability": "searchable",
                "floor": 0.1,
                "evidence": {"fixture": "M2 lexical positive", "separation": "recorded"},
            }
        },
    )
    hits = recall(str(ws), "searchable marker", limit=10, rerank=False)
    print(json.dumps({"ids_and_scores": [(x["_id"], x["score"]) for x in hits]}, sort_keys=True))
    assert hits and hits[0]["_id"] == "SEARCH-1"
    assert hits[0]["score"] >= 0.1


def test_indexed_result_cannot_bypass_direct_only_declaration(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "indexed"
    init(str(ws))
    _config(ws, {"workspace": {"reachability": "direct-only", "floor": "none"}}, backend="sqlite")

    import mind_mem.sqlite_index as sqlite_index

    monkeypatch.setattr(
        sqlite_index,
        "query_index",
        lambda *args, **kwargs: [
            {"_id": "INDEXED-DIRECT", "score": 9.0, "file": "decisions/DECISIONS.md", "line": 1, "status": "active"}
        ],
    )
    hits = recall(str(ws), "private marker", limit=10, rerank=False)
    print(json.dumps({"indexed_ids_and_scores": [(x["_id"], x["score"]) for x in hits]}, sort_keys=True))
    assert hits == []


def test_pack_injects_only_admitted_behavior_blocks_under_hard_cap(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "always"
    init(str(ws))
    _block(ws / "always/decisions/BEHAVIOR.md", "BEHAVIOR-1", "Behavior", "always follow the bounded safety behavior")
    _block(ws / "always/decisions/FACT.md", "FACT-1", "Decision", "withheld non-behavior must never be injected")
    _block(ws / "decisions/DECISIONS.md", "SEARCH-1", "Decision", "ordinary searchable context")
    _config(
        ws,
        {
            "always": {
                "reachability": "always-injected",
                "floor": "none",
                "max_items": 1,
                "content_type": "behavior",
            }
        },
    )
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ws))
    with use_workspace(str(ws)):
        payload = json.loads(pack_recall_budget("ordinary", max_tokens=1000, limit=10))
    print(json.dumps({"always_injected": payload.get("always_injected"), "included": payload.get("included")}, sort_keys=True))
    included_ids = [item["_id"] for item in payload["included"]]
    assert payload["always_injected"] == {"count": 1, "cap": 1, "content_type": "behavior"}
    assert "BEHAVIOR-1" in included_ids
    assert "FACT-1" not in included_ids
