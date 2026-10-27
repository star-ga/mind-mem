"""The sqlite index covers every namespace recall can serve, filtered by the caller.

Before this, ``build_index`` indexed the root corpus only, so a caller whose
ACL grants ``shared/`` or ``agents/<id>/`` but not the root (a shared-only
agent) was served nothing at all on the indexed path while the scan served
it. Each namespace now has its own statistics surface, and the caller's view
is applied in the SQL that fetches candidates, so an unreadable row neither
occupies a fetch slot nor moves a readable row's score.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import date
from typing import Any

import pytest

import mind_mem.sqlite_index as sqlite_index
from mind_mem._recall_core import recall
from mind_mem.init_workspace import init
from mind_mem.sqlite_index import build_index, query_index

INSTANT = date(2026, 10, 8)

ACL = {
    "default_policy": "read",
    "agents": {
        "eve": {"namespaces": ["shared", "agents/eve"], "read": ["shared", "agents/eve"], "write": []},
        "bob": {"namespaces": ["shared", "agents/bob"], "read": ["shared", "agents/bob"], "write": []},
        "dee": {"namespaces": ["agents/dee"], "read": ["decisions", "agents/dee"], "write": []},
    },
}


def _blk(bid: str, stmt: str, status: str = "active", extra: str = "") -> str:
    return f"[{bid}]\nType: Decision\nStatement: {stmt}\nDate: 2026-10-06\nStatus: {status}\n{extra}\n---\n\n"


def _ws(tmp_path: Any, name: str, files: dict[str, str], *, acl: dict | None = ACL, recall_cfg: dict | None = None) -> str:
    ws = str(tmp_path / name)
    os.makedirs(ws)
    init(ws)
    for rel, text in files.items():
        path = os.path.join(ws, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n" + text)
    cfg_path = os.path.join(ws, "mind-mem.json")
    with open(cfg_path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    rc = cfg.get("recall") if isinstance(cfg.get("recall"), dict) else {}
    rc["vector_enabled"] = False
    rc.update(recall_cfg or {})
    cfg["recall"] = rc
    with open(cfg_path, "w", encoding="utf-8") as handle:
        json.dump(cfg, handle)
    if acl is not None:
        with open(os.path.join(ws, "mind-mem-acl.json"), "w", encoding="utf-8") as handle:
            json.dump(acl, handle)
    return ws


def _q(ws: str, query: str, agent: str | None = None, **kw: Any) -> list[dict]:
    kw.setdefault("limit", 10)
    return list(query_index(ws, query, scoring_instant=INSTANT, agent_id=agent, **kw))


def _ids(hits: list[dict]) -> list[str]:
    return [h["_id"] for h in hits]


NS_FILES = {
    "decisions/DECISIONS.md": _blk("ROOT-1", "orchid ledger root fact"),
    "shared/decisions/DECISIONS.md": _blk("SHARED-1", "orchid ledger shared fact"),
    "agents/eve/decisions/DECISIONS.md": _blk("EVE-1", "orchid ledger eve fact"),
    "agents/bob/decisions/DECISIONS.md": _blk("BOB-1", "orchid ledger bob fact"),
}


def test_shared_only_caller_is_served_its_namespaces_on_the_index(tmp_path: Any) -> None:
    ws = _ws(tmp_path, "w", NS_FILES)
    build_index(ws)
    assert set(_ids(_q(ws, "orchid ledger", "eve"))) == {"SHARED-1", "EVE-1"}
    # Positive control: bob reads his own namespace, not eve's.
    assert set(_ids(_q(ws, "orchid ledger", "bob"))) == {"SHARED-1", "BOB-1"}
    # Through recall(sqlite) too, which re-checks the ACL on the hits.
    cfg_path = os.path.join(ws, "mind-mem.json")
    with open(cfg_path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["recall"]["backend"] = "sqlite"
    with open(cfg_path, "w", encoding="utf-8") as handle:
        json.dump(cfg, handle)
    served = recall(ws, "orchid ledger", limit=10, scoring_instant=INSTANT, agent_id="eve")
    assert {h["_id"] for h in served} == {"SHARED-1", "EVE-1"}
    assert {h["file"] for h in served} == {"shared/decisions/DECISIONS.md", "agents/eve/decisions/DECISIONS.md"}


def test_unbound_caller_reads_root_and_declared_custom_only(tmp_path: Any) -> None:
    files = dict(NS_FILES)
    files["custom/decisions/DECISIONS.md"] = _blk("CUSTOM-1", "orchid ledger custom fact")
    ws = _ws(tmp_path, "w", files, recall_cfg={"namespace_properties": {"custom": {"reachability": "searchable", "floor": "none"}}})
    build_index(ws)
    # The same set the scan serves an unbound caller.
    assert set(_ids(_q(ws, "orchid ledger"))) == {"ROOT-1", "CUSTOM-1"}
    scan = recall(ws, "orchid ledger", limit=10, scoring_instant=INSTANT)
    assert {h["_id"] for h in scan} == {"ROOT-1", "CUSTOM-1"}


def test_unbound_results_do_not_move_when_namespace_content_exists(tmp_path: Any) -> None:
    root = "".join(_blk(f"D-2026100{i}-001", f"orchid ledger root fact number {i} " + "orchid " * i) for i in range(1, 8))
    plain = _ws(tmp_path, "plain", {"decisions/DECISIONS.md": root})
    busy_files = {"decisions/DECISIONS.md": root}
    for ns in ("shared", "agents/eve", "agents/bob"):
        busy_files[f"{ns}/decisions/DECISIONS.md"] = "".join(
            _blk(f"N-{ns.replace('/', '-')}-{i}", "orchid orchid ledger noise " * i) for i in range(1, 30)
        )
    busy = _ws(tmp_path, "busy", busy_files)
    build_index(plain)
    build_index(busy)
    for query in ("orchid ledger", "root fact", "orchid"):
        a = [(h["_id"], h["score"]) for h in _q(plain, query)]
        b = [(h["_id"], h["score"]) for h in _q(busy, query)]
        assert a and a == b


def test_another_namespace_does_not_move_a_callers_scores(tmp_path: Any) -> None:
    shared = _blk("SHARED-1", "orchid ledger shared fact") + _blk("SHARED-2", "ledger only")
    quiet = _ws(tmp_path, "quiet", {"shared/decisions/DECISIONS.md": shared})
    noisy = _ws(
        tmp_path,
        "noisy",
        {
            "shared/decisions/DECISIONS.md": shared,
            "agents/bob/decisions/DECISIONS.md": "".join(_blk(f"BOB-{i}", "orchid orchid orchid private " * i) for i in range(1, 40)),
        },
    )
    build_index(quiet)
    build_index(noisy)
    a = [(h["_id"], h["score"]) for h in _q(quiet, "orchid ledger", "eve")]
    b = [(h["_id"], h["score"]) for h in _q(noisy, "orchid ledger", "eve")]
    assert a and a == b
    # Positive control: bob's rows exist and are served to bob.
    assert any(h["_id"].startswith("BOB-") for h in _q(noisy, "orchid", "bob"))


def test_unreadable_rows_do_not_occupy_the_fetch_window(tmp_path: Any) -> None:
    strong = "".join(_blk(f"D-20260901-{i:03d}", f"orchid ledger orchid ledger {i}") for i in range(1, 260))
    weak = _blk("D-20261001-001", "an orchid mention among many other words that dilute the match for this block")
    ws = _ws(tmp_path, "w", {"tasks/TASKS.md": strong, "decisions/DECISIONS.md": weak})
    build_index(ws)
    # Control: unbound, the weak decision ranks past a 200-row window.
    wide = _ids(_q(ws, "orchid ledger", limit=400, retrieve_wide_k=400, rerank=False))
    assert wide.index("D-20261001-001") >= 200
    # dee reads decisions only; the tasks rows must not consume its window.
    got = _q(ws, "orchid ledger", "dee", limit=5, retrieve_wide_k=200)
    assert _ids(got) == ["D-20261001-001"]


def test_withheld_namespace_block_is_never_searchable(tmp_path: Any) -> None:
    body = _blk("SHARED-Q", "orchid ledger quarantined", status="quarantined") + _blk("SHARED-OK", "orchid ledger admitted")
    ws = _ws(tmp_path, "w", {"shared/decisions/DECISIONS.md": body})
    build_index(ws)
    got = _ids(_q(ws, "orchid ledger", "eve"))
    assert got == ["SHARED-OK"]
    conn = sqlite3.connect(os.path.join(ws, sqlite_index.DB_REL_PATH))
    try:
        tables = [r[0] for r in conn.execute("SELECT fts_table FROM ns_fts_tables")]
        assert tables
        indexed = {r[0] for t in tables for r in conn.execute(f"SELECT block_id FROM {t}")}  # noqa: S608 - test-only
    finally:
        conn.close()
    assert indexed == {"SHARED-OK"}


def test_namespace_edits_and_deletes_reach_the_index_incrementally(tmp_path: Any) -> None:
    ws = _ws(tmp_path, "w", {"shared/decisions/DECISIONS.md": _blk("SHARED-1", "orchid ledger first")})
    build_index(ws)
    assert _ids(_q(ws, "orchid", "eve")) == ["SHARED-1"]
    path = os.path.join(ws, "shared", "decisions", "DECISIONS.md")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(_blk("SHARED-2", "orchid ledger second"))
    assert sqlite_index.is_stale(ws, verify_hash=True)
    build_index(ws)
    assert set(_ids(_q(ws, "orchid", "eve"))) == {"SHARED-1", "SHARED-2"}
    os.remove(path)
    build_index(ws)
    assert _ids(_q(ws, "orchid", "eve")) == []
    assert not sqlite_index.is_stale(ws)


def test_schema_version_bump_forces_a_full_rebuild(tmp_path: Any) -> None:
    ws = _ws(tmp_path, "w", NS_FILES)
    build_index(ws)
    conn = sqlite3.connect(os.path.join(ws, sqlite_index.DB_REL_PATH))
    try:
        conn.execute("UPDATE meta SET value = '3' WHERE key = 'fts_schema_version'")
        conn.execute("DELETE FROM ns_blocks")
        conn.commit()
    finally:
        conn.close()
    summary = build_index(ws)
    assert summary["files_indexed"] >= 1
    assert set(_ids(_q(ws, "orchid ledger", "eve"))) == {"SHARED-1", "EVE-1"}
    assert int(sqlite_index._FTS_SCHEMA_VERSION) >= 4


def test_supersede_edges_are_built_from_admitted_blocks_at_index_time(tmp_path: Any) -> None:
    body = (
        _blk("D-1", "x", extra="Supersedes: D-0, D-00; D-000\n")
        + _blk("D-2", "x", extra="ConstraintSignatures:\n- id: CS-1\n  domain: d\n  supersedes: D-9\n")
        + _blk("D-3", "x", status="quarantined", extra="Supersedes: D-8\n")
        + _blk("D-4", "x", extra="Supersedes: none\n")
        + _blk("D-5", "x", extra="Supersedes:\n- D-7\n- D-6\n")
    )
    ws = _ws(tmp_path, "w", {"decisions/DECISIONS.md": body, "shared/decisions/DECISIONS.md": _blk("S-1", "x", extra="Supersedes: D-1\n")})
    build_index(ws)
    conn = sqlite3.connect(os.path.join(ws, sqlite_index.DB_REL_PATH))
    try:
        edges = sorted(conn.execute("SELECT superseder_id, target_id, source_file FROM supersede_edges").fetchall())
    finally:
        conn.close()
    assert edges == [
        ("D-1", "D-0", "decisions/DECISIONS.md"),
        ("D-1", "D-00", "decisions/DECISIONS.md"),
        ("D-1", "D-000", "decisions/DECISIONS.md"),
        ("D-2", "D-9", "decisions/DECISIONS.md"),
        ("D-5", "D-6", "decisions/DECISIONS.md"),
        ("D-5", "D-7", "decisions/DECISIONS.md"),
        ("S-1", "D-1", "shared/decisions/DECISIONS.md"),
    ]


def test_indexed_query_parses_no_corpus_file(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    body = _blk("D-20261006-001", "orchid ledger old " * 3) + _blk("D-20261008-001", "moved", extra="Supersedes: D-20261006-001\n")
    ws = _ws(tmp_path, "w", {"decisions/DECISIONS.md": body})
    build_index(ws)

    def _refuse(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("query path parsed a corpus file")

    import mind_mem.block_parser as block_parser

    monkeypatch.setattr(block_parser, "parse_file", _refuse)
    monkeypatch.setattr(sqlite_index, "parse_file", _refuse)
    got = _ids(_q(ws, "orchid ledger"))
    assert got[:2] == ["D-20261008-001", "D-20261006-001"]
