"""Actual socket and CLI controls for the read-only local viewer."""

from __future__ import annotations

import http.client
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from mind_mem.viewer import make_server


def _workspace(tmp_path: Path) -> Path:
    decisions = tmp_path / "decisions"
    decisions.mkdir(exist_ok=True)
    (decisions / "DECISIONS.md").write_text(
        "[D-20260914-001]\nStatus: active\nStatement: <script>alert('x')</script> viewer can search\n",
        encoding="utf-8",
    )
    return tmp_path


def _request(server, method: str, path: str, **headers: str) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection.request(method, path, headers={"Host": f"127.0.0.1:{server.server_port}", **headers})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def test_socket_search_detail_and_untrusted_text_is_json_data(tmp_path: Path) -> None:
    server = make_server(str(_workspace(tmp_path)), port=0)
    status, headers, body = _request(server, "GET", "/api/search?q=viewer")
    assert status == 200
    assert headers["Content-Security-Policy"].startswith("default-src 'none'")
    payload = json.loads(body)
    assert payload["hits"][0]["_id"] == "D-20260914-001"
    assert "<script>" in payload["hits"][0]["Statement"]

    server = make_server(str(_workspace(tmp_path)), port=0)
    status, _headers, body = _request(server, "GET", "/api/block/D-20260914-001")
    assert status == 200
    assert json.loads(body)["block"]["Status"] == "active"


@pytest.mark.parametrize(
    ("method", "path", "headers", "expected"),
    [
        ("POST", "/api/search?q=x", {}, 405),
        ("GET", "/api/search?q=x", {"Origin": "http://attacker.invalid"}, 403),
        ("GET", "/api/search?q=x", {"Host": "attacker.invalid"}, 403),
        ("GET", "/../../etc/passwd", {}, 404),
        ("GET", "/api/block/%2e%2e%2fetc%2fpasswd", {}, 404),
    ],
)
def test_socket_security_and_read_only_routes(tmp_path: Path, method: str, path: str, headers: dict[str, str], expected: int) -> None:
    server = make_server(str(_workspace(tmp_path)), port=0)
    status, _headers, body = _request(server, method, path, **headers)
    assert status == expected
    if expected != 200:
        assert json.loads(body)["error"]


def test_graph_unknown_entity_does_not_create_database(tmp_path: Path) -> None:
    server = make_server(str(_workspace(tmp_path)), port=0)
    status, _headers, body = _request(server, "GET", "/api/graph?entity=unknown")
    assert status == 200
    assert json.loads(body)["available"] is False
    assert not (tmp_path / "memory" / "knowledge_graph.db").exists()


def test_viewer_rejects_invalid_principal_and_remote_bind(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        make_server(str(_workspace(tmp_path)), agent_id="../outside")
    with pytest.raises(ValueError, match="loopback"):
        make_server(str(_workspace(tmp_path)), host="0.0.0.0")


def test_static_asset_uses_text_content_and_no_remote_dependency() -> None:
    from importlib.resources import files

    app = files("mind_mem").joinpath("viewer_static", "app.js").read_text()
    assert "textContent" in app
    assert "innerHTML" not in app
    assert "http://" not in app and "https://" not in app


def test_mm_view_help_is_real_cli_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mind_mem.mm_cli", "view", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "read-only local viewer" in result.stdout
