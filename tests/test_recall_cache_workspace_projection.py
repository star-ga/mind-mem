# Copyright 2026 STARGA, Inc.
"""Cache identity binds workspace, answer-affecting config, schema and filters."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from mind_mem import recall_cache
from mind_mem.recall_cache import (
    _KEY_FORMAT,
    UNCACHEABLE_CONFIG_FINGERPRINT,
    LRUCache,
    _RedisCache,
    canonical_workspace_id,
    make_cache_key,
    retrieval_config_fingerprint,
)


@pytest.fixture(autouse=True)
def _fresh_cache():
    recall_cache.reset_singleton()
    yield
    recall_cache.reset_singleton()


def test_two_workspaces_with_empty_anchors_have_different_keys() -> None:
    assert make_cache_key("same", workspace="/tmp/a", index_anchor="") != make_cache_key("same", workspace="/tmp/b", index_anchor="")


def test_workspace_path_aliases_share_one_identity(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(workspace, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")
    assert canonical_workspace_id(str(workspace)) == canonical_workspace_id(str(alias))
    assert make_cache_key("q", workspace=str(workspace)) == make_cache_key("q", workspace=str(alias))


def test_key_format_makes_pre_binding_entries_unreachable() -> None:
    current = make_cache_key("q", namespace="default", workspace="/tmp/a")
    assert current.startswith(f"mindmem:recall:{_KEY_FORMAT}:default:")
    assert current != "mindmem:recall:default:" + current.rsplit(":", 1)[-1]


@pytest.mark.parametrize(
    "section,left,right",
    [
        ("recall", {"backend": "scan"}, {"backend": "vector"}),
        ("block_store", {"backend": "markdown"}, {"backend": "postgres"}),
        ("hybrid_search", {"vector_weight": 0.5}, {"vector_weight": 0.9}),
        ("limits", {"max_blocks": 100}, {"max_blocks": 500}),
        ("auto_recall", {"enabled": True}, {"enabled": False}),
        ("governance_mode", "strict", "permissive"),
    ],
)
def test_each_answer_affecting_section_moves_the_fingerprint(section, left, right) -> None:
    assert retrieval_config_fingerprint({section: left}) != retrieval_config_fingerprint({section: right})


def test_cache_only_settings_do_not_move_the_fingerprint() -> None:
    left = {"recall": {"backend": "scan"}, "cache": {"ttl_seconds": 1}}
    right = {"recall": {"backend": "scan"}, "cache": {"ttl_seconds": 999}}
    assert retrieval_config_fingerprint(left) == retrieval_config_fingerprint(right)


def test_non_json_config_values_are_not_stringified_into_a_fingerprint() -> None:
    assert retrieval_config_fingerprint({"recall": {"weights": {1, 2}}}) == UNCACHEABLE_CONFIG_FINGERPRINT


@pytest.mark.parametrize("kind", ["cycle", "mixed_keys"])
def test_distinct_invalid_public_configs_bypass_the_cache(kind, tmp_path: Path, monkeypatch) -> None:
    """The public ranked path must execute again, never cache a shared sentinel."""
    from mind_mem.mcp.tools import recall as recall_tool

    workspace = _workspace(tmp_path / kind, f"D-{kind}", kind)
    if kind == "cycle":
        recall_section = {}
        recall_section["cycle"] = recall_section
    else:
        recall_section = {1: "integer", "1": "string"}
    malformed = {"cache": {"enabled": True}, "recall": recall_section}
    calls = {"count": 0}

    def uncached(query, **kwargs):
        calls["count"] += 1
        return json.dumps({"results": [], "count": 0, "call": calls["count"]})

    monkeypatch.setattr(recall_tool, "_workspace", lambda: str(workspace))
    monkeypatch.setattr(recall_tool, "_load_config", lambda unused: malformed)
    monkeypatch.setattr(recall_tool, "_resolve_chain_head", lambda unused: "")
    monkeypatch.setattr(recall_tool, "_recall_impl_uncached", uncached)
    recall_tool._recall_impl_ranked("same query", backend="bm25")
    recall_tool._recall_impl_ranked("same query", backend="bm25")
    assert calls["count"] == 2


@pytest.mark.parametrize("config", [{}, {"cache": {"enabled": False}}, {"recall": {"backend": "scan"}}])
def test_projection_version_participates_even_for_default_config(config, monkeypatch) -> None:
    before = retrieval_config_fingerprint(config)
    assert len(before) == 64
    monkeypatch.setattr(recall_cache, "PROJECTION_VERSION", "mutation")
    after = retrieval_config_fingerprint(config)
    assert before and after and before != after


@pytest.mark.parametrize("section", ["v4", "future_provider_policy"])
def test_new_non_cache_sections_move_the_fingerprint_without_allowlist_edits(section) -> None:
    assert retrieval_config_fingerprint({section: {"mode": "left"}}) != retrieval_config_fingerprint({section: {"mode": "right"}})


def test_projection_conservatively_keeps_every_non_cache_top_level_field() -> None:
    config = {"recall": {"x": 1}, "v4": {"x": 2}, "unknown_new_section": {"x": 3}, "cache": {"ttl": 5}}
    assert recall_cache.effective_retrieval_projection(config) == {
        "recall": {"x": 1},
        "v4": {"x": 2},
        "unknown_new_section": {"x": 3},
    }


def test_config_schema_and_filters_are_independent_key_coordinates() -> None:
    base = make_cache_key("q", workspace="/tmp/a", config_fingerprint="a", schema_version="1")
    assert base != make_cache_key("q", workspace="/tmp/a", config_fingerprint="b", schema_version="1")
    assert base != make_cache_key("q", workspace="/tmp/a", config_fingerprint="a", schema_version="2")
    assert base != make_cache_key("q", workspace="/tmp/a", config_fingerprint="a", schema_version="1", filters={"lifecycle": "durable"})
    assert base == make_cache_key("q", workspace="/tmp/a", config_fingerprint="a", schema_version="1", filters={})


def test_lru_invalidation_uses_the_current_versioned_prefix() -> None:
    cache = LRUCache()
    current = make_cache_key("q", namespace="tenant", workspace="/tmp/a")
    stale = "mindmem:recall:tenant:stale"
    cache.set(current, "current")
    cache.set(stale, "stale")
    assert cache.invalidate_namespace("tenant") == 1
    assert cache.get(current) is None
    assert cache.get(stale) == "stale"


def test_redis_invalidation_uses_the_current_versioned_prefix() -> None:
    client = Mock()
    client.scan.return_value = (0, [])
    cache = _RedisCache.__new__(_RedisCache)
    cache._client = client
    assert cache.invalidate_namespace("tenant") == 0
    client.scan.assert_called_once_with(cursor=0, match=f"mindmem:recall:{_KEY_FORMAT}:tenant:*", count=500)


def _workspace(root: Path, block_id: str, marker: str) -> Path:
    for name in ("decisions", "tasks", "entities", "intelligence"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "decisions" / "DECISIONS.md").write_text(
        f"[{block_id}]\nStatement: shared cache isolation query {marker}\nStatus: active\nDate: 2026-09-09\n\n",
        encoding="utf-8",
    )
    (root / "mind-mem.json").write_text(json.dumps({"cache": {"enabled": True}}), encoding="utf-8")
    return root


def test_real_public_dispatch_does_not_replay_rows_across_workspaces(tmp_path: Path, monkeypatch) -> None:
    """One process-global cache, empty anchors, identical query, both fill orders."""
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools import public
    from mind_mem.mcp.tools import recall as recall_tool

    alpha = _workspace(tmp_path / "alpha", "D-ALPHA", "alpha")
    bravo = _workspace(tmp_path / "bravo", "D-BRAVO", "bravo")
    monkeypatch.setattr(recall_tool, "_resolve_chain_head", lambda workspace: "")

    for first, second in ((alpha, bravo), (bravo, alpha)):
        recall_cache.reset_singleton()
        observed = {}
        for workspace in (first, second):
            with use_workspace(str(workspace)):
                envelope = json.loads(public.recall(query="shared cache isolation query", mode="bm25", limit=5))
            observed[workspace.name] = {str(row.get("_id")) for row in envelope.get("results", [])}
        assert "D-ALPHA" in observed["alpha"]
        assert "D-BRAVO" in observed["bravo"]
        assert "D-BRAVO" not in observed["alpha"]
        assert "D-ALPHA" not in observed["bravo"]
