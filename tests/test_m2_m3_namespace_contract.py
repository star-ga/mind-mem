"""Source-bound M2/M3 namespace reachability and floor controls."""

from __future__ import annotations

import json
from pathlib import Path

from mind_mem._recall_core import knee_cutoff, recall
from mind_mem.audit_context import bind_current_agent
from mind_mem.block_store import MarkdownBlockStore
from mind_mem.init_workspace import init
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.recall import pack_recall_budget
from mind_mem.namespace_retrieval import declaration_for, filter_search_hits, namespace_for_path


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
    assert payload["supplemental_evidence"]["status"] == "unproven"
    assert "BEHAVIOR-1" in included_ids
    assert "FACT-1" not in included_ids


def test_malformed_namespace_declaration_fails_closed_and_missing_source_is_rejected() -> None:
    bad = {"recall": {"namespace_properties": {"workspace": {"reachability": "maybe"}}}}
    import pytest

    with pytest.raises(ValueError):
        declaration_for(bad, "workspace")
    cfg = {"recall": {"min_score": 0.9, "namespace_properties": {"workspace": {"floor": "none"}}}}
    assert filter_search_hits([{"_id": "x", "score": 1.0}], cfg) == []
    hits = filter_search_hits(
        [{"_id": "a", "score": 1.0, "file": "decisions/DECISIONS.md"}, {"_id": "b", "score": 0.95, "file": "decisions/DECISIONS.md"}],
        cfg,
    )
    assert len(knee_cutoff(hits, min_results=1, min_score=0.9)) == 2


def test_namespace_filter_is_a_noop_without_explicit_properties() -> None:
    hits = [{"_id": "legacy", "score": 0.01}]
    assert filter_search_hits(hits, {}) == hits
    assert "_namespace_floor_override" not in hits[0]
    assert namespace_for_path(".mind-mem-index/index.db") == "workspace"


def test_numeric_namespace_floor_replaces_global_floor() -> None:
    cfg = {
        "recall": {
            "min_score": 0.9,
            "namespace_properties": {
                "workspace": {
                    "reachability": "searchable",
                    "floor": 0.1,
                    "evidence": {"fixture": "recorded"},
                }
            },
        }
    }
    hits = filter_search_hits(
        [
            {"_id": "a", "score": 0.2, "file": "decisions/DECISIONS.md"},
            {"_id": "b", "score": 0.15, "file": "decisions/DECISIONS.md"},
        ],
        cfg,
    )
    assert [item["_id"] for item in knee_cutoff(hits, min_results=1, min_score=0.9)] == ["a", "b"]


def test_pack_allows_empty_query_only_for_configured_always_namespace(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "empty-pack"
    init(str(ws))
    _block(ws / "always/decisions/BEHAVIOR.md", "BEHAVIOR-1", "Behavior", "empty query behavior")
    _config(ws, {"always": {"reachability": "always-injected", "floor": "none", "max_items": 1, "content_type": "behavior"}})
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ws))
    with use_workspace(str(ws)):
        payload = json.loads(pack_recall_budget("", max_tokens=1000, limit=10))
    assert payload["included"][0]["_id"] == "BEHAVIOR-1"


def test_wildcard_always_namespace_expands_only_real_directories(tmp_path: Path) -> None:
    from mind_mem.namespace_retrieval import always_injected_hits

    ws = tmp_path / "wildcard"
    init(str(ws))
    _block(ws / "agents/a1/decisions/BEHAVIOR.md", "BEHAVIOR-1", "Behavior", "agent behavior")
    cfg = json.loads((ws / "mind-mem.json").read_text(encoding="utf-8"))
    cfg["recall"]["namespace_properties"] = {
        "agents/*": {"reachability": "always-injected", "floor": "none", "max_items": 1, "content_type": "behavior"}
    }
    selected, meta = always_injected_hits(str(ws), cfg)
    assert [item["_id"] for item in selected] == ["BEHAVIOR-1"]
    assert selected[0]["file"] == "agents/a1/decisions/BEHAVIOR.md"
    assert meta["cap"] == 1


def test_pack_applies_bound_agent_acl_to_always_namespace(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "acl-always"
    init(str(ws))
    _block(ws / "agents/a1/decisions/BEHAVIOR.md", "BEHAVIOR-1", "Behavior", "private agent behavior")
    (ws / "mind-mem-acl.json").write_text(
        json.dumps({"agents": {"alice": {"namespaces": ["shared"], "read": ["shared"], "write": []}}}),
        encoding="utf-8",
    )
    _config(
        ws,
        {"agents/*": {"reachability": "always-injected", "floor": "none", "max_items": 1, "content_type": "behavior"}},
    )
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ws))
    with bind_current_agent("alice"), use_workspace(str(ws)):
        payload = json.loads(pack_recall_budget("ordinary", max_tokens=1000, limit=10))
    assert "BEHAVIOR-1" not in [item.get("_id") for item in payload.get("included", [])]
    assert payload["always_injected"]["count"] == 0
