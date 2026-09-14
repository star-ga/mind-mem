"""M5 output enforcement controls for summaries and evidence scope."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from mind_mem.v4 import kind_summaries


def _workspace(tmp_path: Path, *, redaction: dict | None = None, max_chars: int = 4000) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    cfg: dict[str, object] = {"v4": {"kind_summaries": {"enabled": True, "max_chars": max_chars}}}
    if redaction is not None:
        cfg["v4"]["redaction"] = redaction  # type: ignore[index]
    (ws / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    db = sqlite3.connect(ws / "index.db")
    try:
        db.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, content TEXT, kind TEXT)")
        db.executemany(
            "INSERT INTO blocks VALUES (?, ?, ?)",
            [("D-1", "Use the blue deployment", "decision"), ("D-2", "Deploy Friday", "decision")],
        )
        db.commit()
    finally:
        db.close()
    return ws


def _reset_summariser() -> None:
    kind_summaries.set_summariser(kind_summaries.default_summariser)


def test_default_summary_binds_exact_sources_and_is_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _workspace(tmp_path)
    monkeypatch.setenv("MIND_MEM_CONFIG", str(ws / "mind-mem.json"))
    try:
        result = kind_summaries.refresh_summary(ws, "decision")
    finally:
        _reset_summariser()
    assert result is not None
    assert result.source_ids == ("D-1", "D-2")
    expected = hashlib.sha256(
        json.dumps(
            [
                {"id": "D-1", "content": "Use the blue deployment"},
                {"id": "D-2", "content": "Deploy Friday"},
            ],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert result.source_digest == expected
    assert result.enforcement == "deterministic_extract"
    assert result.semantic_verification == "not_established"
    assert kind_summaries.get_summary(ws, "decision").source_digest == expected  # type: ignore[union-attr]


def test_plugin_is_bounded_screened_and_marked_unverified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _workspace(tmp_path, redaction={"enabled": True, "mode": "redact"})
    monkeypatch.setenv("MIND_MEM_CONFIG", str(ws / "mind-mem.json"))
    kind_summaries.set_summariser(lambda _blocks: "Contact ops@example.com")
    try:
        result = kind_summaries.refresh_summary(ws, "decision")
    finally:
        _reset_summariser()
    assert result is not None
    assert result.summary == "Contact [REDACTED:email]"
    assert result.enforcement == "unverified_plugin"
    assert result.semantic_verification == "not_established"
    assert result.source_ids == ("D-1", "D-2")


def test_unsafe_and_oversized_plugin_outputs_never_persist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _workspace(tmp_path, max_chars=64)
    monkeypatch.setenv("MIND_MEM_CONFIG", str(ws / "mind-mem.json"))
    for output, message in (
        ("x" * 65, "max_chars"),
        ("unsafe\x00marker", "unsafe"),
        ("unsafe\u202emarker", "unsafe"),
        (None, "must return str"),
    ):
        kind_summaries.set_summariser(lambda _blocks, value=output: value)
        try:
            with pytest.raises(kind_summaries.SummaryOutputError, match=message):
                kind_summaries.refresh_summary(ws, "decision")
        finally:
            _reset_summariser()
        db = sqlite3.connect(ws / "index.db")
        try:
            assert db.execute("SELECT COUNT(*) FROM kind_summaries").fetchone() == (0,)
        finally:
            db.close()


def test_summary_flag_and_cap_use_explicit_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _workspace(tmp_path, max_chars=64)
    ambient = tmp_path / "ambient.json"
    ambient.write_text(json.dumps({"v4": {"kind_summaries": {"enabled": False, "max_chars": 4000}}}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(ambient))
    result = kind_summaries.refresh_summary(ws, "decision")
    assert result is not None and result.enforcement == "deterministic_extract"


def test_required_provenance_refuses_without_fabricated_attribution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _workspace(tmp_path)
    cfg = json.loads((ws / "mind-mem.json").read_text(encoding="utf-8"))
    cfg["v4"]["provenance"] = {"enabled": True, "policy": "required"}
    (ws / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(ws / "mind-mem.json"))
    with pytest.raises(kind_summaries.SummaryOutputError, match="caller attribution"):
        kind_summaries.refresh_summary(ws, "decision")


def test_read_side_redaction_protects_an_existing_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _workspace(tmp_path, redaction={"enabled": True, "mode": "redact"})
    monkeypatch.setenv("MIND_MEM_CONFIG", str(ws / "mind-mem.json"))
    kind_summaries.ensure_kind_summary_schema(ws)
    with sqlite3.connect(ws / "index.db") as db:
        db.execute(
            "INSERT INTO kind_summaries(kind, summary, block_count, updated_at) VALUES (?, ?, ?, ?)",
            ("decision", "Contact ops@example.com", 1, "now"),
        )
    record = kind_summaries.get_summary(ws, "decision")
    assert record is not None and record.summary == "Contact [REDACTED:email]"


def test_summariser_is_captured_before_output_and_status_assignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _workspace(tmp_path)
    monkeypatch.setenv("MIND_MEM_CONFIG", str(ws / "mind-mem.json"))

    def switching(_blocks: object) -> str:
        kind_summaries.set_summariser(kind_summaries.default_summariser)
        return "plugin result"

    kind_summaries.set_summariser(switching)
    try:
        result = kind_summaries.refresh_summary(ws, "decision")
    finally:
        _reset_summariser()
    assert result is not None and result.enforcement == "unverified_plugin"
