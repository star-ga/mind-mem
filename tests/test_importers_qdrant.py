# Copyright 2026 STARGA, Inc.
"""Offline controls for the bounded Qdrant migration importer."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from mind_mem import mm_cli
from mind_mem.block_parser import parse_file
from mind_mem.importers.engine import run_import
from mind_mem.importers.parsers import parse_qdrant
from mind_mem.importers.qdrant_source import scroll_qdrant
from mind_mem.importers.records import ImportParseError


class _QdrantHandler(BaseHTTPRequestHandler):
    pages: dict[Any, dict[str, Any]] = {}
    requests: list[dict[str, Any]] = []
    response_override: bytes | None = None
    status_code = 200
    redirect_location: str | None = None

    def log_message(self, _format: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        self.__class__.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
        if self.__class__.redirect_location is not None:
            self.send_response(307)
            self.send_header("Location", self.__class__.redirect_location)
            self.end_headers()
            return
        offset = body.get("offset")
        payload = self.__class__.response_override
        if payload is None:
            page = self.__class__.pages.get(offset, self.__class__.pages[None])
            payload = json.dumps({"status": "ok", "result": page}).encode("utf-8")
        self.send_response(self.__class__.status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def qdrant_server() -> Any:
    handler = _QdrantHandler
    handler.pages = {
        None: {
            "points": [
                {"id": "p-1", "payload": {"text": "first", "owner": "alice"}},
                {"id": "p-2", "payload": {"text": "second", "owner": "alice"}},
            ],
            "next_page_offset": 2,
        },
        2: {"points": [{"id": "p-3", "payload": {"text": "third", "owner": "bob"}}], "next_page_offset": None},
    }
    handler.requests = []
    handler.response_override = None
    handler.status_code = 200
    handler.redirect_location = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", handler
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _workspace(tmp_path: Path) -> str:
    workspace = tmp_path / "workspace"
    for directory in ("memory", "decisions", "tasks", "entities", "intelligence"):
        (workspace / directory).mkdir(parents=True)
    (workspace / "mind-mem.json").write_text(
        json.dumps({"version": "5.0.3", "workspace_path": str(workspace), "block_store": {"backend": "markdown"}}),
        encoding="utf-8",
    )
    return str(workspace)


def test_scroll_qdrant_reads_real_bounded_pages_and_auth(qdrant_server: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    endpoint, handler = qdrant_server
    monkeypatch.setenv("QDRANT_TEST_KEY", "secret-key-for-fixture")
    points = scroll_qdrant(endpoint, "agent memory", api_key_env="QDRANT_TEST_KEY", page_size=2, max_records=3)

    assert [point["id"] for point in points] == ["p-1", "p-2", "p-3"]
    assert len(handler.requests) == 2
    assert handler.requests[0]["path"].endswith("/collections/agent%20memory/points/scroll")
    assert handler.requests[0]["body"] == {"limit": 2, "with_payload": True, "with_vector": False}
    assert handler.requests[1]["body"]["offset"] == 2
    request_headers = {key.lower(): value for key, value in handler.requests[0]["headers"].items()}
    assert request_headers["api-key"] == "secret-key-for-fixture"


def test_qdrant_mapping_requires_declared_text_and_keeps_payload_metadata() -> None:
    records = parse_qdrant(
        [{"id": "source-1", "payload": {"body": "declared text", "owner": "alice", "kind": "note"}}],
        text_field="body",
        collection="notes",
    )
    assert records[0].external_id == "source-1"
    assert records[0].text == "declared text"
    assert records[0].metadata["owner"] == "alice"
    assert records[0].metadata["kind"] == "note"
    assert records[0].metadata["qdrant_collection"] == "notes"
    with pytest.raises(ImportParseError, match="payload field 'text'"):
        parse_qdrant([{"id": "source-1", "payload": {"body": "declared text"}}])


def test_qdrant_rejects_repeated_offset(qdrant_server: Any) -> None:
    endpoint, handler = qdrant_server
    handler.pages = {None: {"points": [], "next_page_offset": 1}, 1: {"points": [], "next_page_offset": 1}}
    with pytest.raises(ImportParseError, match="repeated an offset"):
        scroll_qdrant(endpoint, "notes", max_pages=3)
    assert len(handler.requests) == 2


def test_qdrant_enforces_cumulative_response_bound(qdrant_server: Any) -> None:
    endpoint, handler = qdrant_server
    first = json.dumps({"status": "ok", "result": handler.pages[None]}).encode("utf-8")
    second = json.dumps({"status": "ok", "result": handler.pages[2]}).encode("utf-8")
    total_limit = max(len(first), len(second)) + 1
    assert total_limit < len(first) + len(second)
    with pytest.raises(ImportParseError, match="cumulative response"):
        scroll_qdrant(endpoint, "notes", max_response_bytes=max(len(first), len(second)), max_total_response_bytes=total_limit)
    assert len(handler.requests) == 2


def test_qdrant_refuses_redirect_and_invalid_offset(qdrant_server: Any) -> None:
    endpoint, handler = qdrant_server
    handler.response_override = json.dumps({"status": "ok", "result": {"points": [], "next_page_offset": {"unexpected": "object"}}}).encode(
        "utf-8"
    )
    with pytest.raises(ImportParseError, match="invalid pagination offset"):
        scroll_qdrant(endpoint, "notes")

    handler.redirect_location = endpoint + "/elsewhere"
    with pytest.raises(ImportParseError, match="redirect"):
        scroll_qdrant(endpoint, "notes")

    handler.redirect_location = None
    handler.response_override = None
    handler.pages = {None: {"points": [], "next_page_offset": None}}
    # The fixture server is intentionally not a redirect server; use the
    # endpoint validator for a credential-bearing URL as the no-leak control.
    with pytest.raises(ImportParseError, match="credentials"):
        scroll_qdrant(endpoint.replace("http://", "http://user:secret@"), "notes")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not-json", "not valid JSON"),
        (b'{"status":"error","result":{}}', "invalid status"),
        (b'{"status":{},"result":{"points":[]}}', "invalid status"),
        (b'{"status":"ok","result":{}}', "missing result.points"),
    ],
)
def test_qdrant_rejects_invalid_page_shapes(qdrant_server: Any, payload: bytes, message: str) -> None:
    endpoint, handler = qdrant_server
    handler.response_override = payload
    with pytest.raises(ImportParseError, match=message):
        scroll_qdrant(endpoint, "notes")


def test_qdrant_rejects_duplicate_json_keys(qdrant_server: Any) -> None:
    endpoint, handler = qdrant_server
    handler.response_override = b'{"status":"ok","result":{"points":[],"points":[]}}'
    with pytest.raises(ImportParseError, match="duplicate JSON object keys"):
        scroll_qdrant(endpoint, "notes")


@pytest.mark.parametrize(
    ("malformed", "message"),
    [
        (True, "not valid JSON"),
        (False, "not valid JSON|contains a non-object point"),
    ],
)
def test_qdrant_rejects_deep_json_without_quarantine_write(qdrant_server: Any, tmp_path: Path, malformed: bool, message: str) -> None:
    endpoint, handler = qdrant_server
    depth = 100_000
    # Missing one closing array is always malformed. Keep the original valid
    # deep JSON too: decoders that accept this depth must still reject its
    # non-object point. Both refusal paths must leave the workspace unchanged.
    closing_brackets = depth - 1 if malformed else depth
    handler.response_override = b'{"status":"ok","result":{"points":' + b"[" * depth + b"]" * closing_brackets + b"}}"
    workspace = _workspace(tmp_path)

    with pytest.raises(ImportParseError, match=message):
        run_import(workspace, "qdrant", "", endpoint=endpoint, collection="notes")

    assert not (Path(workspace) / "memory" / "IMPORTED.md").exists()
    from mind_mem.audit_chain import AuditChain

    assert AuditChain(workspace).entries() == []


def test_qdrant_enforces_response_and_record_bounds(qdrant_server: Any) -> None:
    endpoint, handler = qdrant_server
    with pytest.raises(ImportParseError, match="response exceeds"):
        scroll_qdrant(endpoint, "notes", max_response_bytes=32)

    handler.response_override = json.dumps(
        {"status": "ok", "result": {"points": [{"id": index, "payload": {"text": "x"}} for index in range(3)]}}
    ).encode("utf-8")
    with pytest.raises(ImportParseError, match="exceeds 2 records"):
        scroll_qdrant(endpoint, "notes", max_records=2)


def test_qdrant_missing_key_fails_before_request(qdrant_server: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    endpoint, handler = qdrant_server
    monkeypatch.delenv("QDRANT_MISSING_KEY", raising=False)
    with pytest.raises(ImportParseError, match="is not set"):
        scroll_qdrant(endpoint, "notes", api_key_env="QDRANT_MISSING_KEY")
    assert handler.requests == []


@pytest.mark.parametrize("system", ["agentmem", "chatjson", "chroma", "letta", "markdown", "mem0"])
def test_local_imports_require_path_before_source_scan(system: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = _workspace(tmp_path)

    def fail_if_loaded(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("local source was scanned before path validation")

    monkeypatch.setattr("mind_mem.importers.engine.load_source", fail_if_loaded)
    with pytest.raises(ImportParseError, match="non-empty path"):
        run_import(workspace, system, "")
    assert not (Path(workspace) / "memory" / "IMPORTED.md").exists()


def test_qdrant_rejects_local_path_before_endpoint_request(qdrant_server: Any, tmp_path: Path) -> None:
    endpoint, handler = qdrant_server
    with pytest.raises(ImportParseError, match="do not accept a local path"):
        run_import(tmp_path.as_posix(), "qdrant", "pretend-local-dump.json", endpoint=endpoint, collection="notes")
    assert handler.requests == []


def test_cli_rejects_qdrant_options_on_local_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = _workspace(tmp_path)
    dump = tmp_path / "mem0.json"
    dump.write_text(json.dumps({"results": [{"id": "local-1", "memory": "local text"}]}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    exit_code = mm_cli.main(["import", "--from", "mem0", str(dump), "--endpoint", "http://127.0.0.1:6333"])
    assert exit_code == 3
    assert "only valid with --from qdrant" in capsys.readouterr().err
    assert not (Path(workspace) / "memory" / "IMPORTED.md").exists()


def test_qdrant_rejects_duplicate_source_ids() -> None:
    with pytest.raises(ImportParseError, match="repeats source id"):
        parse_qdrant(
            [
                {"id": "same", "payload": {"text": "one"}},
                {"id": "same", "payload": {"text": "two"}},
            ]
        )

    # Qdrant's JSON point-id forms include integers and UUID strings.  The
    # importer stores both as the existing string external-id field, so their
    # normalized collision must fail closed rather than silently overwrite or
    # skip one point during block planning.
    with pytest.raises(ImportParseError, match="repeats source id"):
        parse_qdrant(
            [
                {"id": 1, "payload": {"text": "numeric"}},
                {"id": "1", "payload": {"text": "string"}},
            ]
        )


def test_qdrant_run_import_uses_quarantine_and_audit_path(qdrant_server: Any, tmp_path: Path) -> None:
    endpoint, _handler = qdrant_server
    workspace = _workspace(tmp_path)
    result = run_import(workspace, "qdrant", "", endpoint=endpoint, collection="notes", qdrant_max_records=3)

    assert result.system == "qdrant"
    assert result.source_path == endpoint
    assert result.parsed == result.imported == 3
    blocks = parse_file(str(Path(workspace) / "memory" / "IMPORTED.md"))
    assert len(blocks) == 3
    assert {block["Status"] for block in blocks} == {"quarantined"}
    assert {block["IngestTier"] for block in blocks} == {"external-ingest"}
    assert {block["Source"] for block in blocks} == {"imported:qdrant"}
    from mind_mem.audit_chain import AuditChain, _payload_hash

    entries = AuditChain(workspace).entries()
    assert len(entries) == 1
    assert entries[0].agent == "importer:qdrant"
    assert entries[0].payload_hash == _payload_hash(
        {
            "system": "qdrant",
            "source_path": endpoint,
            "batch": result.batch,
            "status": "quarantined",
            "tier": "external-ingest",
            "block_ids": list(result.block_ids),
        }
    )


def test_cli_qdrant_import_never_prints_api_key(
    qdrant_server: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    endpoint, _handler = qdrant_server
    workspace = _workspace(tmp_path)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    monkeypatch.setenv("QDRANT_TEST_KEY", "cli-secret-key")
    exit_code = mm_cli.main(
        [
            "import",
            "--from",
            "qdrant",
            "--endpoint",
            endpoint,
            "--collection",
            "notes",
            "--api-key-env",
            "QDRANT_TEST_KEY",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "cli-secret-key" not in captured.out + captured.err
    assert '"system": "qdrant"' in captured.out
