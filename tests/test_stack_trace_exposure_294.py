"""Exception text never reaches an API caller (code-scanning #294).

CodeQL ``py/stack-trace-exposure`` traced exception text from the workspace
check to the body of ``GET /v1/scan``. On a non-Markdown backend
``_check_workspace`` probes the block store, and a store that could not be
built or reached raised an exception whose text (a DSN with host, user and
password, a file path) was formatted into the returned error. Every ws-gated
MCP tool returns that error as its result, and the REST layer forwarded it
verbatim (``/v1/scan``, ``/v1/block/{id}``, ``/v1/recall``).

The same shape -- an infrastructure exception formatted into a returned
string -- also fed the recall warning list (abstention sidecar write and
verdict; the explain and hybrid-fallback warnings are held in
``test_recall_attestation_served_backend.py``) and the ``approve_apply``
message (precondition subprocess, workspace lock).

Each failure is still reported (positive control) and its detail still lands
in the server log; only the caller-facing text is generic.
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable, Generator

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _review_fixtures import build_workspace  # noqa: E402

from mind_mem import abstention_store, apply_engine  # noqa: E402
from mind_mem.mcp.infra import workspace as ws_mod  # noqa: E402
from mind_mem.mcp.infra.workspace import _check_workspace, use_workspace  # noqa: E402
from mind_mem.mcp.tools import governance  # noqa: E402

SECRET = "postgresql://svc:hunter2@10.0.0.7:5432/prod"


class _Recorder:
    """Stand-in for a module ``_log``: keeps every event and its fields."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def _record(self, event: str, **fields: Any) -> None:
        self.events.append((event, fields))

    debug = info = warning = error = _record

    def logged(self, needle: str) -> bool:
        return any(needle in str(value) for _event, fields in self.events for value in fields.values())


class _Store:
    def __init__(self, *, ping: Callable[[], Any] | None = None, list_raises: bool = False) -> None:
        if ping is not None:
            self.ping = ping
        self._list_raises = list_raises

    def list_blocks(self) -> list[str]:
        if self._list_raises:
            raise ConnectionError(f"could not connect to {SECRET}")
        return []


def _raise(exc: BaseException) -> Callable[..., Any]:
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise exc

    return _boom


# Every exception path inside the store probe: construction, ping raising,
# ping reporting not-ok with a detail, and the list_blocks fallback.
_STORE_FAILURES: dict[str, Callable[[], Any]] = {
    "construction": _raise(ImportError(f"cannot open {SECRET}")),
    "ping_raises": lambda: _Store(ping=_raise(OSError(f"connection to {SECRET} refused"))),
    "ping_not_ok": lambda: _Store(ping=lambda: {"ok": False, "error": f"password authentication failed for {SECRET}"}),
    "list_raises": lambda: _Store(list_raises=True),
}


def _break_store(monkeypatch: pytest.MonkeyPatch, case: str) -> _Recorder:
    """Route the workspace check to a failing non-Markdown store."""
    import mind_mem.storage as storage_mod

    monkeypatch.setattr(ws_mod, "_resolve_backend", lambda _ws: "postgres")
    factory = _STORE_FAILURES[case]
    if case == "construction":
        monkeypatch.setattr(storage_mod, "get_block_store", factory)
    else:
        store = factory()
        monkeypatch.setattr(storage_mod, "get_block_store", lambda _ws, config=None: store)
    recorder = _Recorder()
    monkeypatch.setattr(ws_mod, "_log", recorder, raising=False)
    return recorder


@pytest.mark.parametrize("case", sorted(_STORE_FAILURES))
def test_workspace_check_returns_a_generic_error_and_logs_the_detail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    recorder = _break_store(monkeypatch, case)
    result = _check_workspace(str(tmp_path))
    assert result is not None, "the failing store must still be reported"
    assert SECRET not in result and "hunter2" not in result
    error = json.loads(result)["error"]
    assert "'postgres'" in error and "server log" in error
    assert recorder.logged(SECRET), "the detail must stay available to the operator"


@pytest.mark.parametrize("case", sorted(_STORE_FAILURES))
def test_mcp_scan_result_carries_no_exception_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    _break_store(monkeypatch, case)
    with use_workspace(str(tmp_path)):
        raw = governance.scan()
    assert json.loads(raw)["error"], "scan must still refuse"  # positive control
    assert SECRET not in raw and "hunter2" not in raw


@pytest.fixture()
def rest_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Any, None, None]:
    from fastapi.testclient import TestClient  # in the [test] extra: a missing install fails, never skips

    from mind_mem.api.rest import _enforce_fail_closed, create_app

    ws = tmp_path / "ws"
    (ws / "decisions").mkdir(parents=True)
    (ws / "decisions" / "DECISIONS.md").write_text("[D-20240101-001]\nstatus: active\nstatement: x\n", encoding="utf-8")
    monkeypatch.delenv("MIND_MEM_TOKEN", raising=False)
    monkeypatch.delenv("MIND_MEM_ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ws))
    app = create_app(str(ws), _local_anonymous_capability=_enforce_fail_closed("127.0.0.1", True))
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/v1/scan", None),
        ("get", "/v1/block/D-20240101-001", None),
        ("post", "/v1/recall", {"query": "database"}),
    ],
)
def test_rest_body_carries_no_exception_text(
    rest_client: Any, monkeypatch: pytest.MonkeyPatch, method: str, path: str, body: dict | None
) -> None:
    _break_store(monkeypatch, "construction")
    resp = rest_client.request(method.upper(), path, json=body)
    assert "server log" in resp.text, resp.text  # the workspace refusal reached the client
    assert SECRET not in resp.text and "hunter2" not in resp.text


def test_rest_boundary_does_not_echo_non_json_tool_output() -> None:
    from mind_mem.api.rest import _parse_tool_json

    raw = f'Traceback (most recent call last):\n  File "/srv/app/x.py", line 1\nRuntimeError: {SECRET}'
    parsed = _parse_tool_json(raw)
    assert isinstance(parsed, dict) and parsed.get("error"), parsed  # still an error envelope
    text = json.dumps(parsed)
    assert SECRET not in text and "Traceback" not in text and "/srv/app" not in text
    assert _parse_tool_json('{"ok": true}') == {"ok": True}  # JSON output is unchanged


def test_abstention_record_failure_warning_carries_no_exception_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(abstention_store, "record_prediction", _raise(PermissionError(f"[Errno 13] {SECRET}/calibration.db")))
    recorder = _Recorder()
    monkeypatch.setattr(abstention_store, "_log", recorder)
    attestation = {"query_id": "a" * 64, "scoring_instant": "2026-10-28T00:00:00Z"}
    problem = abstention_store.record_assessment(str(tmp_path), {"abstain": False}, attestation)
    assert problem and "prediction" in problem.lower()  # still surfaced as a warning
    assert SECRET not in problem
    assert recorder.logged(SECRET)


def test_precondition_error_lines_carry_no_exception_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apply_engine.subprocess, "run", _raise(FileNotFoundError(f"[Errno 2] {SECRET}/python")))
    recorder = _Recorder()
    monkeypatch.setattr(apply_engine, "_log", recorder)
    ok, report = apply_engine.check_preconditions(str(tmp_path))
    assert ok is False and any("ERROR" in line for line in report), report  # still a failed check
    assert SECRET not in "\n".join(report)
    assert recorder.logged(SECRET)


def test_apply_lock_failure_message_carries_no_exception_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = str(tmp_path / "ws")
    os.makedirs(root)
    proposal_id = build_workspace(root, 1)[0]

    class _StuckLock:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def acquire(self) -> None:
            raise TimeoutError(f"lock held at {SECRET}/.mind-mem.lock")

    monkeypatch.setattr(apply_engine, "FileLock", _StuckLock)
    recorder = _Recorder()
    monkeypatch.setattr(apply_engine, "_log", recorder)
    printed = io.StringIO()
    with redirect_stdout(printed):
        ok, message = apply_engine.apply_proposal(root, proposal_id)
    assert ok is False and "lock" in message.lower(), message  # the lock refusal itself
    assert SECRET not in message
    assert SECRET not in printed.getvalue()  # approve_apply returns stdout as its ``log``
    assert recorder.logged(SECRET)


def test_recall_abstention_failure_warning_carries_no_exception_text(monkeypatch: pytest.MonkeyPatch) -> None:
    import mind_mem.mcp.tools.recall as recall_tool

    monkeypatch.setattr(recall_tool, "assess", _raise(OSError(f"cannot read {SECRET}/policy")))
    recorder = _Recorder()
    monkeypatch.setattr(recall_tool, "_log", recorder)
    raw, verdict = recall_tool._apply_abstention(json.dumps({"results": []}), intended_use=None, recall_cfg=None)
    warnings = json.loads(raw)["warnings"]
    assert verdict is None and any("Abstention verdict unavailable" in w for w in warnings), warnings
    assert SECRET not in raw
    assert recorder.logged(SECRET)
