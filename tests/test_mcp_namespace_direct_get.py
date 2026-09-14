"""Namespace-selected MCP direct reads remain ACL- and admission-bound."""

from __future__ import annotations

import json
from pathlib import Path

from mind_mem._recall_core import recall
from mind_mem.audit_context import bind_current_agent
from mind_mem.block_store import MarkdownBlockStore
from mind_mem.init_workspace import init
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.memory_ops import get_block


def _block(path: Path, block_id: str, statement: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{block_id}]\nType: Decision\nStatement: {statement}\nStatus: active\n\n")


def _block_with_status(
    path: Path,
    block_id: str,
    statement: str,
    status: str,
    *,
    category: str | None = None,
    releases: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = f"ContentCategory: {category}\n" if category is not None else ""
    extra += f"Releases: {releases}\n" if releases is not None else ""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{block_id}]\nType: Decision\nStatement: {statement}\n{extra}Status: {status}\n\n")


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "namespace-direct-get"
    init(str(ws))
    _block(ws / "shared/decisions/DECISIONS.md", "D-SHARED-1", "shared direct-only evidence")
    _block(ws / "agents/alice/decisions/DECISIONS.md", "D-ALICE-1", "alice direct-only evidence")
    _block(ws / "agents/bob/decisions/DECISIONS.md", "D-BOB-1", "bob private evidence")
    config = json.loads((ws / "mind-mem.json").read_text(encoding="utf-8"))
    config["recall"]["namespace_properties"] = {
        "shared": {"reachability": "direct-only", "floor": "none"},
        "agents/alice": {"reachability": "direct-only", "floor": "none"},
    }
    (ws / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")
    (ws / "mind-mem-acl.json").write_text(
        json.dumps(
            {
                "default_policy": "read",
                "agents": {
                    "alice": {
                        "namespaces": ["shared", "agents/alice"],
                        "read": ["shared", "agents/alice"],
                        "write": ["agents/alice"],
                    },
                    "bob": {
                        "namespaces": ["shared", "agents/bob"],
                        "read": ["shared", "agents/bob"],
                        "write": ["agents/bob"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return ws


def _get(ws: Path, block_id: str, namespace: str, agent: str) -> dict:
    import os

    previous = os.environ.get("MIND_MEM_WORKSPACE")
    os.environ["MIND_MEM_WORKSPACE"] = str(ws)
    try:
        with bind_current_agent(agent), use_workspace(str(ws)):
            return json.loads(get_block(block_id, namespace=namespace))
    finally:
        if previous is None:
            os.environ.pop("MIND_MEM_WORKSPACE", None)
        else:
            os.environ["MIND_MEM_WORKSPACE"] = previous


def test_authenticated_agent_reads_shared_and_own_direct_only_blocks_but_search_stays_absent(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    shared = _get(ws, "D-SHARED-1", "shared", "alice")
    own = _get(ws, "D-ALICE-1", "agents/alice", "alice")
    assert shared["found"] is True and shared["block"]["_id"] == "D-SHARED-1"
    assert own["found"] is True and own["block"]["_id"] == "D-ALICE-1"
    assert recall(str(ws), "direct-only evidence", agent_id="alice", limit=10, rerank=False) == []


def test_agent_acl_and_selector_validation_refuse_private_or_forged_reads(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    denied = _get(ws, "D-BOB-1", "agents/bob", "alice")
    traversal = _get(ws, "D-ALICE-1", "agents/../shared", "alice")
    forged = _get(ws, "D-ALICE-1", "agents/bob", "alice")
    assert denied["error"] == "namespace access denied"
    assert traversal["error"] == "invalid namespace selector"
    assert forged["error"] == "namespace access denied"


def test_selected_namespace_duplicate_ids_are_explicitly_ambiguous(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _block(ws / "shared/tasks/TASKS.md", "D-SHARED-1", "duplicate shared ID")
    result = _get(ws, "D-SHARED-1", "shared", "alice")
    assert result["found"] is False
    assert result["ambiguous"] is True
    assert "ambiguous" in result["error"].lower()


def test_legacy_id_only_root_resolution_remains_available(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _block(ws / "decisions/DECISIONS.md", "D-LEGACY-1", "legacy root direct read")
    result = _get(ws, "D-LEGACY-1", "", "")
    assert result["found"] is True
    assert result["block"]["_id"] == "D-LEGACY-1"
    assert MarkdownBlockStore(str(ws)).get_by_id("D-LEGACY-1") is not None


def test_selected_namespace_status_does_not_borrow_duplicate_root_id(tmp_path: Path) -> None:
    """A stale root index cannot make a quarantined namespace row readable."""
    ws = _workspace(tmp_path)
    shared_path = ws / "shared/decisions/DECISIONS.md"
    root_path = ws / "decisions/DECISIONS.md"
    _block_with_status(shared_path, "D-DUP-STATUS", "shared withheld canary", "quarantined")
    _block_with_status(root_path, "D-DUP-STATUS", "root active canary", "active")

    from mind_mem.sqlite_index import build_index

    build_index(str(ws), incremental=False)
    _block(root_path, "D-INDEX-TOUCH-1", "make the cached root status map stale")
    result = _get(ws, "D-DUP-STATUS", "shared", "alice")
    assert result["found"] is False
    assert result["withheld"] is True
    assert "shared withheld canary" not in json.dumps(result)


def test_selected_namespace_uses_source_status_for_duplicate_root_id(tmp_path: Path) -> None:
    """A root quarantine cannot hide an active row selected from another source."""
    ws = _workspace(tmp_path)
    shared_path = ws / "shared/decisions/DECISIONS.md"
    root_path = ws / "decisions/DECISIONS.md"
    _block_with_status(shared_path, "D-DUP-ACTIVE", "shared active canary", "active")
    _block_with_status(root_path, "D-DUP-ACTIVE", "root withheld canary", "quarantined")

    from mind_mem.sqlite_index import build_index

    build_index(str(ws), incremental=False)
    _block(root_path, "D-INDEX-TOUCH-2", "make the cached root status map stale")
    result = _get(ws, "D-DUP-ACTIVE", "shared", "alice")
    assert result["found"] is True
    assert result["block"]["Statement"] == "shared active canary"


def test_selected_namespace_does_not_borrow_root_release_or_revoked_credential(tmp_path: Path) -> None:
    """Namespace reads require source policy even when root has a release row."""
    ws = _workspace(tmp_path)
    config_path = ws / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["recall"]["validity_gate"] = {
        "enabled": True,
        "content_categories": {
            "enabled": True,
            "ttl_days": {"infra": 30, "status": 30},
        },
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    shared_path = ws / "shared/decisions/DECISIONS.md"
    root_path = ws / "decisions/DECISIONS.md"
    _block_with_status(shared_path, "IMP-DUP-RELEASE", "shared quarantined canary", "quarantined")
    _block_with_status(root_path, "IMP-DUP-RELEASE", "root active duplicate", "active")
    _block_with_status(root_path, "D-RELEASE-1", "release decision", "active", releases="IMP-DUP-RELEASE")
    result = _get(ws, "IMP-DUP-RELEASE", "shared", "alice")
    assert result["found"] is False and result["withheld"] is True
    assert "shared quarantined canary" not in json.dumps(result)

    credential_id = "IMP-DUP-CREDENTIAL"
    _block_with_status(shared_path, credential_id, "revoked credential canary", "revoked", category="credential")
    _block_with_status(root_path, credential_id, "root active credential", "active", category="credential")
    result = _get(ws, credential_id, "shared", "alice")
    assert result["found"] is False and result["withheld"] is True


def test_selected_namespace_ignores_unregistered_markdown_sources(tmp_path: Path) -> None:
    """A direct selector exposes registered corpus files only."""
    ws = _workspace(tmp_path)
    _block(ws / "shared/notes/UNREGISTERED.md", "D-UNREGISTERED-1", "private source canary")
    result = _get(ws, "D-UNREGISTERED-1", "shared", "alice")
    assert result["found"] is False
    assert result.get("withheld") is not True


def test_legacy_id_only_non_markdown_namespaced_rows_still_obey_agent_acl(tmp_path: Path, monkeypatch) -> None:
    """A backend source identity cannot bypass private namespace ACLs."""
    ws = _workspace(tmp_path)
    from mind_mem import storage
    from mind_mem.mcp.tools import memory_ops

    class Store:
        def get_by_id(self, block_id: str) -> dict:
            return {
                "_id": block_id,
                "_source_file": "agents/bob/decisions/DECISIONS.md",
                "Statement": "private backend canary",
                "Status": "active",
            }

    monkeypatch.setattr(memory_ops, "_is_markdown_backend", lambda _ws: False)
    monkeypatch.setattr(memory_ops, "get_block_store", lambda _ws: Store())
    monkeypatch.setattr(storage, "_backend_name", lambda *_args: "postgres")
    monkeypatch.setattr(
        storage,
        "iter_blocks",
        lambda *_args, **_kwargs: [
            {
                "_id": "D-BACKEND-PRIVATE",
                "_source_file": "agents/bob/decisions/DECISIONS.md",
                "Statement": "private backend canary",
                "Status": "active",
            }
        ],
    )
    denied = _get(ws, "D-BACKEND-PRIVATE", "", "alice")
    assert denied["error"] == "namespace access denied"
    allowed = _get(ws, "D-BACKEND-PRIVATE", "", "bob")
    assert allowed["found"] is True


def test_non_markdown_selected_status_is_bound_to_backend_source(monkeypatch, tmp_path: Path) -> None:
    """A DB row must not refresh from a shadow Markdown file or duplicate id."""
    ws = _workspace(tmp_path)
    from mind_mem import storage
    from mind_mem.mcp.tools import memory_ops

    rows = [
        {"_id": "D-DUP-1", "_source_file": "shared/decisions/DECISIONS.md", "Statement": "shared", "Status": "quarantined"},
        {"_id": "D-DUP-1", "_source_file": "decisions/DECISIONS.md", "Statement": "root", "Status": "active"},
    ]

    class Store:
        def get_all(self, *, active_only: bool = False) -> list[dict]:
            return [dict(row) for row in rows]

    monkeypatch.setattr(memory_ops, "_is_markdown_backend", lambda _ws: False)
    monkeypatch.setattr(memory_ops, "get_block_store", lambda _ws: Store())
    monkeypatch.setattr(storage, "_backend_name", lambda *_args: "postgres")
    monkeypatch.setattr(storage, "iter_blocks", lambda *_args, **_kwargs: [dict(row) for row in rows])
    result = _get(ws, "D-DUP-1", "shared", "alice")
    assert result["found"] is False and result["withheld"] is True


def test_non_markdown_omitted_selector_still_binds_private_source_status(monkeypatch, tmp_path: Path) -> None:
    """The legacy selector shape cannot borrow an active root duplicate."""
    ws = _workspace(tmp_path)
    from mind_mem import storage
    from mind_mem.mcp.tools import memory_ops

    rows = [
        {"_id": "D-DUP-2", "file": "agents/alice/decisions/DECISIONS.md", "Statement": "private", "Status": "quarantined"},
        {"_id": "D-DUP-2", "file": "decisions/DECISIONS.md", "Statement": "root", "Status": "active"},
    ]

    class Store:
        def get_by_id(self, _block_id: str) -> dict:
            return dict(rows[0])

    monkeypatch.setattr(memory_ops, "_is_markdown_backend", lambda _ws: False)
    monkeypatch.setattr(memory_ops, "get_block_store", lambda _ws: Store())
    monkeypatch.setattr(storage, "_backend_name", lambda *_args: "postgres")
    monkeypatch.setattr(storage, "iter_blocks", lambda *_args, **_kwargs: [dict(row) for row in rows])
    result = _get(ws, "D-DUP-2", "", "alice")
    assert result["found"] is False and result["withheld"] is True


def test_non_markdown_source_field_variants_admit_active_and_withhold_quarantined(monkeypatch, tmp_path: Path) -> None:
    """Backend source spellings share one identity and release decision."""
    ws = _workspace(tmp_path)
    from mind_mem import storage
    from mind_mem.mcp.tools import memory_ops

    monkeypatch.setattr(memory_ops, "_is_markdown_backend", lambda _ws: False)
    monkeypatch.setattr(storage, "_backend_name", lambda *_args: "postgres")
    for field in ("_source_file", "_source", "file"):
        row = {
            "_id": f"D-FIELD-{field.replace('_', '')}",
            field: "shared/decisions/DECISIONS.md",
            "Statement": "field variant",
            "Status": "active",
        }

        class Store:
            def get_all(self, *, active_only: bool = False) -> list[dict]:
                return [dict(row)]

            def get_by_id(self, _block_id: str) -> dict:
                return dict(row)

        monkeypatch.setattr(memory_ops, "get_block_store", lambda _ws: Store())
        monkeypatch.setattr(storage, "iter_blocks", lambda *_args, **_kwargs: [dict(row)])
        for selector in ("shared", ""):
            active = _get(ws, row["_id"], selector, "alice")
            assert active["found"] is True, (field, selector, active)
        row["Status"] = "quarantined"
        withheld = _get(ws, row["_id"], "shared", "alice")
        assert withheld["found"] is False and withheld["withheld"] is True


def test_non_markdown_omitted_selector_enforces_shared_acl(tmp_path: Path, monkeypatch) -> None:
    """An omitted selector must not bypass shared-namespace ACL policy."""
    ws = _workspace(tmp_path)
    acl_path = ws / "mind-mem-acl.json"
    acl = json.loads(acl_path.read_text(encoding="utf-8"))
    acl["agents"]["alice"]["read"] = ["agents/alice"]
    acl_path.write_text(json.dumps(acl), encoding="utf-8")
    from mind_mem import storage
    from mind_mem.mcp.tools import memory_ops

    row = {"_id": "D-BACKEND-SHARED", "_source_file": "shared/decisions/DECISIONS.md", "Statement": "shared backend", "Status": "active"}

    class Store:
        def get_by_id(self, _block_id: str) -> dict:
            return dict(row)

    monkeypatch.setattr(memory_ops, "_is_markdown_backend", lambda _ws: False)
    monkeypatch.setattr(memory_ops, "get_block_store", lambda _ws: Store())
    monkeypatch.setattr(storage, "_backend_name", lambda *_args: "postgres")
    monkeypatch.setattr(storage, "iter_blocks", lambda *_args, **_kwargs: [dict(row)])
    denied = _get(ws, row["_id"], "", "alice")
    allowed = _get(ws, row["_id"], "", "bob")
    assert denied["error"] == "namespace access denied"
    assert allowed["found"] is True


def test_non_markdown_root_release_remains_available_with_source_binding(tmp_path: Path, monkeypatch) -> None:
    """Source binding retains the approved release path for root imports."""
    ws = _workspace(tmp_path)
    from mind_mem import storage
    from mind_mem.mcp.tools import memory_ops

    target = {
        "_id": "IMP-RELEASE-1",
        "_source_file": "memory/IMPORTED.md",
        "Statement": "released import",
        "Status": "quarantined",
    }
    release = {
        "_id": "D-RELEASE-1",
        "_source_file": "decisions/DECISIONS.md",
        "Statement": "release",
        "Status": "active",
        "Releases": "IMP-RELEASE-1",
    }

    class Store:
        def get_by_id(self, _block_id: str) -> dict:
            return dict(target)

    monkeypatch.setattr(memory_ops, "_is_markdown_backend", lambda _ws: False)
    monkeypatch.setattr(memory_ops, "get_block_store", lambda _ws: Store())
    monkeypatch.setattr(storage, "_backend_name", lambda *_args: "postgres")
    monkeypatch.setattr(storage, "iter_blocks", lambda *_args, **_kwargs: [dict(target), dict(release)])
    result = _get(ws, target["_id"], "", "")
    assert result["found"] is True and result["block"]["_id"] == target["_id"]
