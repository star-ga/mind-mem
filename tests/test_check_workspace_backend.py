"""Backend-aware workspace validation — ``mcp.infra.workspace._check_workspace``.

Audit bug #2: the legacy ``_check_workspace`` hardcoded the Markdown corpus
layout (a local ``decisions/`` directory) as the definition of a valid
workspace. Every ws-gated MCP tool funnels through this helper, so on a
Postgres-backed workspace — where the blocks of record live in the DB, not
in local Markdown files — all 34 tools fail-closed with
``"Workspace is missing the 'decisions/' directory"`` even though the data
is correctly stored.

The fix makes ``_check_workspace`` backend-aware:

* **Markdown / encrypted backend** (the zero-config SQLite default) — keep
  the exact legacy behaviour (require ``decisions/``). These tests assert
  that path is byte-for-byte unchanged so the default never regresses.
* **Any other backend** (e.g. ``postgres``) — validate by probing the
  configured block store (``ping()`` / ``list_blocks()``) instead of
  requiring ``decisions/`` on disk.

Coverage:

* Markdown default — present / missing ``decisions/`` and missing root,
  runs with no DB.
* A non-Markdown backend validated via a fake store (no DB) — exercises
  the ``ping()`` ok / not-ok / unreachable branches deterministically.
* Postgres backend against a live DB — a PG workspace *without* a local
  ``decisions/`` dir validates clean (the regression), and a workspace
  pointing at a bad backend returns a structured error (never crashes).
  Each run uses a uniquely-named scratch schema it creates and drops; the
  class skips gracefully when psycopg / a live Postgres is unavailable, so
  SQLite-only CI stays green.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Generator

import pytest

from mind_mem.mcp.infra import workspace as ws_mod
from mind_mem.mcp.infra.workspace import _check_workspace

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _write_config(ws: Path, *, block_store: dict | None = None) -> None:
    """Write ``mind-mem.json`` into *ws* (the root must already exist)."""
    config: dict = {"recall": {"backend": "bm25"}}
    if block_store is not None:
        config["block_store"] = block_store
    (ws / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")


def _error_of(result: str | None) -> str:
    """Decode the error message from a ``_check_workspace`` failure result."""
    assert result is not None, "expected an error result, got None (valid)"
    payload = json.loads(result)
    assert "error" in payload, f"result missing 'error' key: {payload!r}"
    return str(payload["error"])


# ─── Markdown / default backend (no DB) ───────────────────────────────────────


def test_markdown_with_decisions_dir_is_valid(tmp_path: Path) -> None:
    """Default SQLite/Markdown path: decisions/ present → valid (None)."""
    ws = tmp_path / "ws"
    (ws / "decisions").mkdir(parents=True)
    _write_config(ws)  # explicit markdown-default config
    assert _check_workspace(str(ws)) is None


def test_markdown_no_config_with_decisions_dir_is_valid(tmp_path: Path) -> None:
    """No mind-mem.json at all degrades to markdown default and stays valid."""
    ws = tmp_path / "ws"
    (ws / "decisions").mkdir(parents=True)
    assert _check_workspace(str(ws)) is None


def test_markdown_missing_decisions_dir_errors(tmp_path: Path) -> None:
    """Legacy behaviour preserved: markdown workspace without decisions/."""
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_config(ws)
    msg = _error_of(_check_workspace(str(ws)))
    assert "decisions/" in msg
    assert "mind-mem-init" in msg


def test_missing_workspace_root_errors(tmp_path: Path) -> None:
    """A non-existent workspace path errors regardless of backend."""
    ws = tmp_path / "does-not-exist"
    msg = _error_of(_check_workspace(str(ws)))
    assert "Workspace not found" in msg


def test_encrypted_backend_still_requires_decisions(tmp_path: Path) -> None:
    """``encrypted`` is a markdown-corpus backend → decisions/ still required."""
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_config(ws, block_store={"backend": "encrypted"})
    msg = _error_of(_check_workspace(str(ws)))
    assert "decisions/" in msg


# ─── Non-markdown backend via a fake store (no DB) ────────────────────────────


class _FakeStore:
    """Minimal stand-in for a non-markdown block store."""

    def __init__(self, ping_result: dict[str, Any] | None = None, *, ping_raises: bool = False, list_raises: bool = False) -> None:
        self._ping_result = ping_result
        self._ping_raises = ping_raises
        self._list_raises = list_raises

    def ping(self) -> dict[str, Any]:
        if self._ping_raises:
            raise RuntimeError("ping blew up")
        assert self._ping_result is not None
        return self._ping_result

    def list_blocks(self) -> list[str]:
        if self._list_raises:
            raise RuntimeError("list blew up")
        return []


class _PinglessStore:
    """A non-markdown store without ping() — validated via list_blocks()."""

    def __init__(self, *, list_raises: bool = False) -> None:
        self._list_raises = list_raises

    def list_blocks(self) -> list[str]:
        if self._list_raises:
            raise RuntimeError("list blew up")
        return ["B-1"]


def _patch_backend(monkeypatch: pytest.MonkeyPatch, backend: str, store: object) -> None:
    """Force ``_resolve_backend`` and ``get_block_store`` for the helper."""
    monkeypatch.setattr(ws_mod, "_resolve_backend", lambda _ws: backend)
    import mind_mem.storage as storage_mod

    monkeypatch.setattr(storage_mod, "get_block_store", lambda _ws, config=None: store)


def test_nonmarkdown_ping_ok_is_valid_without_decisions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ping() ok=True → valid even though no decisions/ dir exists."""
    ws = tmp_path / "ws"
    ws.mkdir()  # note: no decisions/ subdir
    _patch_backend(monkeypatch, "postgres", _FakeStore({"backend": "postgres", "ok": True}))
    assert _check_workspace(str(ws)) is None


def test_nonmarkdown_ping_not_ok_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ping() ok=False surfaces a structured (non-crashing) error."""
    ws = tmp_path / "ws"
    ws.mkdir()
    _patch_backend(monkeypatch, "postgres", _FakeStore({"ok": False, "error": "blocks table unavailable"}))
    msg = _error_of(_check_workspace(str(ws)))
    assert "blocks table unavailable" in msg
    assert "decisions/" not in msg  # not a markdown error


def test_nonmarkdown_ping_raises_is_caught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising ping() never escapes — converted to a structured error."""
    ws = tmp_path / "ws"
    ws.mkdir()
    _patch_backend(monkeypatch, "postgres", _FakeStore(ping_raises=True))
    msg = _error_of(_check_workspace(str(ws)))
    assert "unreachable" in msg


def test_nonmarkdown_store_construction_failure_is_caught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """get_block_store raising (e.g. missing extra) → structured error, no crash."""
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(ws_mod, "_resolve_backend", lambda _ws: "postgres")
    import mind_mem.storage as storage_mod

    def _boom(_ws: str, config: Any = None) -> object:
        raise ImportError("psycopg not installed")

    monkeypatch.setattr(storage_mod, "get_block_store", _boom)
    msg = _error_of(_check_workspace(str(ws)))
    assert "could not be initialised" in msg


def test_nonmarkdown_pingless_store_uses_list_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A store without ping() validates via list_blocks()."""
    ws = tmp_path / "ws"
    ws.mkdir()
    _patch_backend(monkeypatch, "postgres", _PinglessStore())
    assert _check_workspace(str(ws)) is None


def test_nonmarkdown_pingless_store_list_raises_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pingless store whose list_blocks() raises → structured error."""
    ws = tmp_path / "ws"
    ws.mkdir()
    _patch_backend(monkeypatch, "postgres", _PinglessStore(list_raises=True))
    msg = _error_of(_check_workspace(str(ws)))
    assert "unreachable" in msg


# ─── Postgres backend (live DB; skips cleanly when unavailable) ───────────────

psycopg = pytest.importorskip("psycopg", reason="psycopg not installed; skipping Postgres tests")

from mind_mem.block_store_postgres import PostgresBlockStore  # noqa: E402

# Honour the standard test DSN env var first; fall back to the local audit
# DSN. The schema is always a unique scratch schema we create and drop —
# the production ``mind_mem`` schema is never touched.
_DSN = os.environ.get("MIND_MEM_TEST_PG_DSN")


def _pg_available(dsn: str) -> bool:
    try:
        conn = psycopg.connect(dsn, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


_PG_LIVE = _pg_available(_DSN)


@pytest.fixture
def pg_workspace(tmp_path: Path) -> Generator[tuple[str, str], None, None]:
    """A workspace configured for Postgres on an isolated scratch schema.

    Yields ``(workspace_path, schema)``. Crucially the workspace has **no**
    local ``decisions/`` directory — that is the whole point of the
    regression: a PG workspace must validate without the markdown layout.
    """
    schema = f"mm_fix_{uuid.uuid4().hex[:12]}"
    ws = tmp_path / "pgws"
    ws.mkdir()
    _write_config(ws, block_store={"backend": "postgres", "dsn": _DSN, "schema": schema})

    store = PostgresBlockStore(dsn=_DSN, schema=schema, workspace=str(ws))
    store._ensure_schema()
    try:
        yield str(ws), schema
    finally:
        try:
            store.close()
        except Exception:
            pass
        try:
            conn = psycopg.connect(_DSN)
            conn.autocommit = True
            conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            conn.close()
        except Exception:
            pass


@pytest.mark.skipif(not _PG_LIVE, reason="no live Postgres available at the test DSN")
def test_postgres_workspace_valid_without_decisions_dir(pg_workspace: tuple[str, str]) -> None:
    """The regression: a provisioned PG workspace validates with no decisions/."""
    ws, _schema = pg_workspace
    assert not os.path.isdir(os.path.join(ws, "decisions"))
    # Backend-aware validation probes the store's ping(), which is ok=True
    # after _ensure_schema() — so the workspace is valid.
    assert _check_workspace(ws) is None


@pytest.mark.skipif(not _PG_LIVE, reason="no live Postgres available at the test DSN")
def test_postgres_workspace_valid_even_when_empty(pg_workspace: tuple[str, str]) -> None:
    """Validity is reachability, not has-blocks: an empty PG store is valid."""
    ws, _schema = pg_workspace
    # No blocks written; the schema exists and pings ok.
    assert _check_workspace(ws) is None


@pytest.mark.skipif(not _PG_LIVE, reason="no live Postgres available at the test DSN")
def test_postgres_unreachable_backend_errors_not_crashes(tmp_path: Path) -> None:
    """A PG workspace pointing at a dead DSN returns a structured error."""
    ws = tmp_path / "deadpg"
    ws.mkdir()
    # Port 1 is never a Postgres; connect fails fast → ping ok=False.
    bad_dsn = "postgresql://nope:nope@127.0.0.1:1/nope?connect_timeout=2"
    _write_config(ws, block_store={"backend": "postgres", "dsn": bad_dsn, "schema": "mm_fix_dead"})
    result = _check_workspace(str(ws))
    msg = _error_of(result)
    # Fail-loud but structured — never an unhandled exception, never a
    # spurious markdown 'decisions/' message.
    assert "postgres" in msg
    assert "decisions/" not in msg
