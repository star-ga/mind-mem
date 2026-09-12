"""Regression test: mm doctor must emit a clear hint when backend=postgres
is configured but the psycopg driver is missing.

Audit-driven (v4.0.13). Before this fix, ``mm doctor`` on a PG-configured
workspace without psycopg installed surfaced a bare ImportError trace
that recurred as user friction. Now it returns:

    {"block_store_error": "No module named 'psycopg'",
     "install_hint": 'pip install "mind-mem[postgres]"  # ...'}

so the user gets a single actionable line.
"""

from __future__ import annotations

import argparse
import builtins
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def pg_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a minimal Postgres-backed mind-mem workspace skeleton."""
    cfg = {
        "block_store": {"backend": "postgres", "dsn": "postgresql://invalid"},
        "recall": {"backend": "sqlite"},
    }
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MIND_MEM_WORKSPACE", raising=False)
    return tmp_path


def test_doctor_emits_install_hint_when_psycopg_missing(
    pg_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ImportError on psycopg must surface as a structured install_hint."""
    from mind_mem import mm_cli

    # Force the storage layer's psycopg import to fail as if the extra
    # was never installed. We patch builtins.__import__ so any
    # downstream ``import psycopg`` raises ModuleNotFoundError.
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "psycopg" or name.startswith("psycopg."):
            raise ModuleNotFoundError("No module named 'psycopg'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Bust any cached psycopg / storage imports so the patch takes effect.
    for mod_name in list(sys.modules):
        if mod_name == "psycopg" or mod_name.startswith("psycopg."):
            sys.modules.pop(mod_name, None)
        if mod_name.startswith("mind_mem.storage"):
            sys.modules.pop(mod_name, None)

    ns = argparse.Namespace(
        rebuild_cache=False,
        migrate_recall_log=False,
    )

    # ``_cmd_doctor`` prints the report dict as JSON to stdout.
    mm_cli._cmd_doctor(ns)
    out = capsys.readouterr().out

    # Exit-code semantics (in_sync vs not) are intentionally not asserted
    # here — both empty stores read as in-sync; the regression we're
    # guarding against is the *report content*, not the return code.

    # The report must include both the error and the hint, and the
    # hint must reference the [postgres] extra by name so the user can
    # copy/paste it.
    report = json.loads(out)
    assert "install_hint" in report, f"missing install_hint in doctor report: {report!r}"
    hint = report["install_hint"]
    assert "mind-mem[postgres]" in hint
    assert "pip install" in hint
    # And we kept the underlying error for diagnostics. psycopg can fail
    # at two sites: get_block_store construction or the connect path
    # inside the count branch — either is acceptable, but at least one
    # must carry the ImportError text.
    err_text = report.get("block_store_error", "") + report.get("postgres_count_error", "")
    assert "psycopg" in err_text, f"no psycopg error in report: {report!r}"


def test_doctor_no_hint_for_unrelated_import_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Non-psycopg ModuleNotFoundError must NOT trigger the postgres hint."""
    from mind_mem import mm_cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MIND_MEM_WORKSPACE", raising=False)

    # Patch get_block_store to raise an unrelated ImportError.
    import mind_mem.storage as storage_mod

    def boom(_ws: object) -> object:
        raise ModuleNotFoundError("No module named 'totally_unrelated'")

    monkeypatch.setattr(storage_mod, "get_block_store", boom)

    ns = argparse.Namespace(rebuild_cache=False, migrate_recall_log=False)
    mm_cli._cmd_doctor(ns)
    out = capsys.readouterr().out

    report = json.loads(out)
    # Error should be reported, but no postgres install hint should appear.
    assert "block_store_error" in report
    assert "install_hint" not in report


def test_doctor_reports_in_sync_on_the_single_store_default(tmp_path, monkeypatch, capsys) -> None:
    """Parity is a Postgres question; the markdown default has no peer.

    ``in_sync`` was computed by diffing the SQLite recall cache against
    ``pg_ids``, which stays empty unless the backend IS Postgres. So on
    the shipped markdown/SQLite default every indexed block counted as
    "sqlite-only" and ``in_sync`` went false as soon as the workspace
    had any content -- dragging ``mm doctor``'s ``healthy`` verdict
    false with it. An empty index hid it, which is why it survived.
    """
    import json
    import sqlite3

    from mind_mem.mm_cli import main as mm_main

    ws = tmp_path / "ws"
    ws.mkdir()
    index_dir = ws / ".mind-mem-index"
    index_dir.mkdir()
    conn = sqlite3.connect(index_dir / "recall.db")
    conn.execute("CREATE TABLE IF NOT EXISTS blocks (id TEXT PRIMARY KEY)")
    conn.executemany(
        "INSERT OR IGNORE INTO blocks (id) VALUES (?)",
        [("D-20260912-001",), ("D-20260912-002",)],
    )
    conn.commit()
    conn.close()

    monkeypatch.chdir(ws)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ws))
    monkeypatch.setattr("sys.argv", ["mm", "doctor"])
    try:
        mm_main()
    except SystemExit:
        pass

    out = capsys.readouterr().out
    report = json.loads(out[out.index("{") : out.rindex("}") + 1])

    assert report["in_sync"] is True, report
    # The Postgres-only counter must not be reported at all here.
    assert "sqlite_only_count" not in report, report
