"""Workspaces and recall legs shared by the supersede-chain and index-namespace tests.

Each scenario is one small corpus written into a fresh workspace. A scenario
is served on every recall leg a caller can reach: the in-memory scan (library
and MCP), the indexed sqlite backend through ``recall``, ``query_index``
directly, ``HybridBackend.search`` and the MCP tool's indexed backends.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import date
from typing import Any

INSTANT = date(2026, 10, 8)
INSTANT_S = "2026-10-08"
QUERY = "weekly call with Ilona"
TGT = "D-20261006-001"

STRONG = "The weekly call with Ilona is on Tuesday. weekly call with Ilona weekly call with Ilona weekly call Ilona."
WEAK = "The weekly call with Ilona moved to Wednesday."
VERYWEAK = "Ilona prefers Wednesday now."

SCAN_LEGS = ("scan", "mcp_scan")
INDEX_LEGS = ("sqlite", "query_index", "hybrid", "mcp_bm25_idx", "mcp_auto", "mcp_hybrid")
ALL_LEGS = SCAN_LEGS + INDEX_LEGS

ACL_SHARED_ONLY = {
    "default_policy": "read",
    "agents": {"a1": {"namespaces": ["shared", "agents/a1"], "write": [], "read": ["shared", "agents/a1"]}},
}
ACL_SHARED_AND_ROOT = {
    "default_policy": "read",
    "agents": {"a1": {"namespaces": ["shared", "agents/a1"], "write": [], "read": ["shared", "agents/a1", "decisions"]}},
}
CUSTOM_NS = {"namespace_properties": {"custom": {"reachability": "searchable", "floor": "none"}}}


def blk(bid: str, stmt: str, status: str = "active", extra: str = "", date_: str = "2026-10-06", kind: str = "Decision") -> str:
    return f"[{bid}]\nType: {kind}\nStatement: {stmt}\nDate: {date_}\nStatus: {status}\n{extra}\n---\n\n"


@dataclass(frozen=True)
class Scenario:
    name: str
    files: dict[str, str]
    recall_cfg: dict[str, Any] = field(default_factory=dict)
    acl: dict[str, Any] | None = None
    agent: str | None = None


def make_workspace(root: str, sc: Scenario, *, suffix: str = "") -> str:
    from mind_mem.init_workspace import init

    ws = os.path.join(root, sc.name + suffix)
    os.makedirs(ws)
    init(ws)
    for rel, text in sc.files.items():
        path = os.path.join(ws, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n" + text)
    cfg_path = os.path.join(ws, "mind-mem.json")
    with open(cfg_path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    recall_cfg = cfg.get("recall") if isinstance(cfg.get("recall"), dict) else {}
    recall_cfg["vector_enabled"] = False
    for key, value in sc.recall_cfg.items():
        recall_cfg[key] = value
    cfg["recall"] = recall_cfg
    with open(cfg_path, "w", encoding="utf-8") as handle:
        json.dump(cfg, handle)
    if sc.acl is not None:
        with open(os.path.join(ws, "mind-mem-acl.json"), "w", encoding="utf-8") as handle:
            json.dump(sc.acl, handle)
    return ws


def set_backend(ws: str, backend: str) -> None:
    cfg_path = os.path.join(ws, "mind-mem.json")
    with open(cfg_path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["recall"]["backend"] = backend
    with open(cfg_path, "w", encoding="utf-8") as handle:
        json.dump(cfg, handle)


def ids(hits: Any) -> list[str]:
    return [str(h.get("_id") or h.get("id") or "") for h in hits]


def _mcp(ws: str, backend: str, agent: str | None, limit: int, query: str) -> list[str]:
    from mind_mem.audit_context import current_agent_id
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools import recall as mcp_recall

    previous = os.environ.get("MIND_MEM_SCOPE")
    os.environ["MIND_MEM_SCOPE"] = "admin"
    token = current_agent_id.set(agent) if agent else None
    try:
        with use_workspace(ws):
            payload = json.loads(mcp_recall.recall(query, limit=limit, backend=backend, scoring_instant=INSTANT_S))
    finally:
        if token is not None:
            current_agent_id.reset(token)
        if previous is None:
            os.environ.pop("MIND_MEM_SCOPE", None)
        else:
            os.environ["MIND_MEM_SCOPE"] = previous
    assert "error" not in payload, payload
    return ids(payload.get("results", []))


def serve(ws: str, leg: str, *, agent: str | None, limit: int = 10, query: str = QUERY) -> list[str]:
    """Ids one leg returns. Scan legs need an unindexed workspace, index legs an indexed one."""
    from mind_mem._recall_core import recall
    from mind_mem.hybrid_recall import HybridBackend
    from mind_mem.sqlite_index import query_index

    if leg == "scan":
        set_backend(ws, "scan")
        return ids(recall(ws, query, limit=limit, scoring_instant=INSTANT, agent_id=agent))
    if leg == "mcp_scan":
        set_backend(ws, "scan")
        return _mcp(ws, "bm25", agent, limit, query)
    if leg == "sqlite":
        set_backend(ws, "sqlite")
        try:
            return ids(recall(ws, query, limit=limit, scoring_instant=INSTANT, agent_id=agent))
        finally:
            set_backend(ws, "scan")
    if leg == "query_index":
        return ids(query_index(ws, query, limit=limit, scoring_instant=INSTANT, agent_id=agent))
    if leg == "hybrid":
        kwargs: dict[str, Any] = {"agent_id": agent} if agent else {}
        return ids(HybridBackend(config={"vector_enabled": False}).search(query, ws, limit=limit, scoring_instant=INSTANT, **kwargs))
    if leg == "mcp_bm25_idx":
        set_backend(ws, "scan")
        return _mcp(ws, "bm25", agent, limit, query)
    if leg == "mcp_auto":
        set_backend(ws, "scan")
        return _mcp(ws, "auto", agent, limit, query)
    if leg == "mcp_hybrid":
        set_backend(ws, "scan")
        return _mcp(ws, "hybrid", agent, limit, query)
    raise AssertionError(leg)


class Workspaces:
    """One unindexed and one indexed copy per scenario, built once."""

    def __init__(self, root: str) -> None:
        self.root = root
        self._built: dict[str, tuple[str, str]] = {}

    def get(self, sc: Scenario) -> tuple[str, str]:
        if sc.name not in self._built:
            from mind_mem.sqlite_index import build_index

            plain = make_workspace(self.root, sc)
            indexed = make_workspace(self.root, sc, suffix="-idx")
            build_index(indexed)
            self._built[sc.name] = (plain, indexed)
        return self._built[sc.name]

    def serve(self, sc: Scenario, leg: str, *, agent: str | None = None, limit: int = 10, query: str = QUERY) -> list[str]:
        plain, indexed = self.get(sc)
        ws = plain if leg in SCAN_LEGS else indexed
        return serve(ws, leg, agent=agent if agent is not None else sc.agent, limit=limit, query=query)


def cleanup(root: str) -> None:
    shutil.rmtree(root, ignore_errors=True)
