# Copyright 2026 STARGA, Inc.
"""Regressions for paths that used to fail quietly.

Each test here pins a case where the code reported success — an exit
code, a counter, a returned path, a parsed record count — while the
work it claimed had not happened. They all fail against the pre-fix
implementations.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import tarfile
from pathlib import Path

import pytest

from mind_mem import ingestion_pipeline, ledger_anchor
from mind_mem.backup_restore import WAL, backup_workspace
from mind_mem.importers.note_parsers import parse_chat_json
from mind_mem.importers.records import ImportParseError
from mind_mem.transcript_capture import parse_transcript
from mind_mem.verify_cli import EXIT_GENERIC, EXIT_OK, verify_workspace

# ---------------------------------------------------------------------------
# backup_restore — empty archive / uncounted rollback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBackupWorkspaceEmptyArchive:
    def test_missing_workspace_raises_instead_of_writing_empty_tarball(self, tmp_path: Path) -> None:
        out = tmp_path / "backup.tar.gz"
        with pytest.raises(FileNotFoundError):
            backup_workspace(str(tmp_path / "nowhere"), str(out))

    def test_workspace_with_nothing_to_back_up_raises(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "unrelated.txt").write_text("not part of the corpus", encoding="utf-8")
        out = tmp_path / "backup.tar.gz"
        with pytest.raises(ValueError, match="captured nothing"):
            backup_workspace(str(ws), str(out))
        # The useless archive must not be left behind to be found at
        # restore time.
        assert not out.exists()

    def test_allow_empty_opt_in_still_writes(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        out = tmp_path / "backup.tar.gz"
        assert backup_workspace(str(ws), str(out), allow_empty=True) == str(out)
        with tarfile.open(out, "r:gz") as tar:
            assert tar.getnames() == []

    def test_real_corpus_still_backs_up(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        (ws / "decisions").mkdir(parents=True)
        (ws / "decisions" / "DECISIONS.md").write_text("[D-20260101-001]\nStatement: x\n", encoding="utf-8")
        out = tmp_path / "backup.tar.gz"
        backup_workspace(str(ws), str(out))
        with tarfile.open(out, "r:gz") as tar:
            assert "decisions/DECISIONS.md" in tar.getnames()


@pytest.mark.unit
class TestWalReplayHonesty:
    def test_failed_rollback_is_not_counted_as_replayed(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        target = ws / "notes.md"
        target.write_text("original", encoding="utf-8")
        wal = WAL(str(ws))
        entry_id = wal.begin("write", str(target), "new content")

        # Simulate a hand-edited / relocated WAL record whose target now
        # escapes the workspace: rollback() refuses it and restores
        # nothing.
        entry_path = Path(wal.wal_dir) / f"{entry_id}.json"
        entry = json.loads(entry_path.read_text(encoding="utf-8"))
        entry["target"] = os.path.join("..", "..", "etc", "escaped.md")
        entry_path.write_text(json.dumps(entry), encoding="utf-8")

        assert wal.replay() == 0, "a rollback that restored nothing must not be reported as replayed"

    def test_successful_rollback_is_counted(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        target = ws / "notes.md"
        target.write_text("original", encoding="utf-8")
        wal = WAL(str(ws))
        wal.begin("write", str(target), "new content")
        target.write_text("half-written", encoding="utf-8")

        assert wal.replay() == 1
        assert target.read_text(encoding="utf-8") == "original"

    def test_truncated_entry_is_reported_not_read_as_clean(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        wal = WAL(str(ws))
        # A record truncated by the very crash the WAL exists to recover
        # from must not read as "workspace is clean".
        (Path(wal.wal_dir) / "wal-truncated.json").write_text('{"id": "wal-truncated", "stat', encoding="utf-8")

        assert wal.pending_count() == 0
        assert wal.unreadable_count() == 1
        assert wal.replay() == 0


# ---------------------------------------------------------------------------
# importers/note_parsers — dropped turns
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChatTranscriptDrops:
    def test_explicit_null_content_falls_through_to_the_alternate_key(self) -> None:
        records = parse_chat_json(
            [
                {"role": "user", "content": None, "text": "the real message"},
                {"role": "assistant", "content": "ok"},
            ]
        )
        assert [r.text for r in records] == ["the real message", "ok"]

    def test_empty_content_falls_through_to_the_alternate_key(self) -> None:
        records = parse_chat_json([{"role": "user", "content": "", "text": "fallback"}])
        assert [r.text for r in records] == ["fallback"]

    def test_turn_list_tolerates_a_non_turn_entry(self) -> None:
        records = parse_chat_json(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"id": "tool-1", "tool_calls": [{"name": "search"}]},
            ]
        )
        assert [r.text for r in records] == ["hello", "hi"]

    def test_turn_carrying_parts_under_message_is_still_a_turn(self) -> None:
        # ``message`` is both a content key and a turn-array key; a turn
        # whose structured parts live there must not be reclassified as a
        # session wrapper by the shape detector.
        records = parse_chat_json(
            [
                {"role": "user", "message": [{"type": "text", "text": "structured part"}]},
                {"role": "assistant", "message": ["plain part"]},
            ]
        )
        assert [r.text for r in records] == ["structured part", "plain part"]

    def test_session_list_error_names_the_offending_entry(self) -> None:
        with pytest.raises(ImportParseError, match=r"entry #1 has no turn array"):
            parse_chat_json(
                [
                    {"id": "s1", "messages": [{"role": "user", "content": "alpha"}]},
                    {"id": "s2", "notes": []},
                ]
            )

    def test_transcript_whose_turns_all_drop_is_an_error_not_a_silent_zero(self) -> None:
        with pytest.raises(ImportParseError, match="none carry any text"):
            parse_chat_json([{"role": "user", "content": ""}, {"role": "assistant", "content": None}])


# ---------------------------------------------------------------------------
# verify_cli — absent artifacts
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVerifyStrictMode:
    def test_lenient_run_on_an_empty_workspace_still_passes(self, tmp_path: Path) -> None:
        report = verify_workspace(str(tmp_path))
        assert report.ok is True
        assert report.exit_code == EXIT_OK

    def test_absent_artifacts_are_listed_even_when_lenient(self, tmp_path: Path) -> None:
        report = verify_workspace(str(tmp_path))
        assert "memory/hash_chain_v2.db" in report.missing
        assert "memory/evidence_chain.jsonl" in report.missing
        assert "missing" in report.as_dict()

    def test_strict_run_fails_when_the_ledger_is_absent(self, tmp_path: Path) -> None:
        report = verify_workspace(str(tmp_path), strict=True)
        assert report.ok is False
        assert report.exit_code == EXIT_GENERIC
        assert report.checks["hash_chain"] is False
        assert report.checks["evidence_chain"] is False


# ---------------------------------------------------------------------------
# transcript_capture — nested turn objects
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTranscriptNestedMessage:
    def _write(self, tmp_path: Path, lines: list[dict]) -> str:
        path = tmp_path / "session.jsonl"
        path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
        return str(path)

    def test_nested_message_object_with_string_content_is_parsed(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [{"type": "user", "message": {"role": "user", "content": "never use the cached path here"}}],
        )
        parsed = parse_transcript(path)
        assert [(m["role"], m["content"]) for m in parsed] == [("user", "never use the cached path here")]

    def test_nested_message_object_with_content_blocks_is_parsed(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "ignored"},
                            {"type": "text", "text": "the root cause was a stale index"},
                        ],
                    },
                }
            ],
        )
        parsed = parse_transcript(path)
        assert parsed[0]["role"] == "assistant"
        assert parsed[0]["content"] == "the root cause was a stale index"

    def test_role_filter_matches_real_turns(self, tmp_path: Path) -> None:
        from mind_mem.transcript_capture import scan_transcript

        path = self._write(
            tmp_path,
            [
                {"type": "user", "message": {"role": "user", "content": "never use the cached path here"}},
                {"type": "queue-operation", "content": "bookkeeping line, not a turn"},
            ],
        )
        signals = scan_transcript(path, role_filter="user")
        assert len(signals) == 1
        assert signals[0]["source_role"] == "user"

    def test_top_level_shapes_still_parse(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [
                {"role": "user", "content": "plain string content"},
                {"role": "assistant", "content": [{"type": "text", "text": "block content"}]},
                {"role": "system", "message": "message-as-string"},
            ],
        )
        assert [m["content"] for m in parse_transcript(path)] == [
            "plain string content",
            "block content",
            "message-as-string",
        ]


# ---------------------------------------------------------------------------
# ingestion_pipeline — the rejected counter
# ---------------------------------------------------------------------------


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.mark.unit
class TestIngestionRejectedCounter:
    def _post(self, port: int, path: str, body: str) -> int:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        status = conn.getresponse().status
        conn.close()
        return status

    def test_malformed_bodies_are_counted_as_rejected(self) -> None:
        q = ingestion_pipeline.IngestionQueue(capacity=4)
        port = _free_port()
        _, stop = ingestion_pipeline.serve_webhook(port, q)
        try:
            assert self._post(port, "/ingest", "{not json") == 400
            assert self._post(port, "/ingest", "[1, 2, 3]") == 400
            assert self._post(port, "/wrong-path", "{}") == 404
        finally:
            stop()
        stats = q.stats().as_dict()
        assert stats["rejected"] == 3, "a producer sending only malformed events must not read as an idle queue"
        assert stats["accepted"] == 0
        assert stats["backpressure_drops"] == 0

    def test_accepted_events_do_not_bump_rejected(self) -> None:
        q = ingestion_pipeline.IngestionQueue(capacity=4)
        port = _free_port()
        _, stop = ingestion_pipeline.serve_webhook(port, q)
        try:
            assert self._post(port, "/ingest", json.dumps({"ok": True})) == 202
        finally:
            stop()
        stats = q.stats().as_dict()
        assert stats["accepted"] == 1
        assert stats["rejected"] == 0


# ---------------------------------------------------------------------------
# ledger_anchor — damaged history
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnchorHistoryDamage:
    def _history(self, tmp_path: Path) -> ledger_anchor.AnchorHistory:
        hist = ledger_anchor.AnchorHistory(str(tmp_path / "anchors.jsonl"))
        hist.record("a" * 64, block_height=1)
        hist.record("b" * 64, block_height=2)
        return hist

    def test_damaged_record_is_reported(self, tmp_path: Path) -> None:
        hist = self._history(tmp_path)
        with open(hist.path, "a", encoding="utf-8") as fh:
            fh.write('{"block_height": 3, "timestamp": "2026-01-01T00:00:00Z"}\n')  # no merkle_root

        problems = hist.problems()
        assert len(problems) == 1
        assert "line 3" in problems[0]

    def test_strict_read_refuses_a_shortened_history(self, tmp_path: Path) -> None:
        hist = self._history(tmp_path)
        with open(hist.path, "a", encoding="utf-8") as fh:
            fh.write("{truncated\n")

        with pytest.raises(ledger_anchor.AnchorHistoryDamagedError):
            hist.all(strict=True)
        with pytest.raises(ledger_anchor.AnchorHistoryDamagedError):
            hist.latest(strict=True)
        # Lenient reads stay backwards-compatible.
        assert len(hist.all()) == 2
        assert hist.latest().block_height == 2

    def test_clean_history_reports_no_problems(self, tmp_path: Path) -> None:
        hist = self._history(tmp_path)
        assert hist.problems() == []
        assert len(hist.all(strict=True)) == 2


# ---------------------------------------------------------------------------
# v4 observability — the documented cardinality knob
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestObservabilityCardinalityConfig:
    def _cfg(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, block: dict) -> None:
        (tmp_path / "mind-mem.json").write_text(json.dumps({"v4": {"observability": block}}), encoding="utf-8")
        monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))

    def test_configured_cap_is_honoured(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem.v4 import observability as obs

        self._cfg(tmp_path, monkeypatch, {"enabled": True, "max_cardinality": 3})
        obs.reset_for_tests()
        try:
            for i in range(3):
                obs.counter(f"c_{i}")
            assert obs.counter("c_overflow") is obs._OVERFLOW_COUNTER
            assert obs.snapshot()["v4.cardinality.dropped_counter"] == 1
        finally:
            obs.reset_for_tests()

    def test_default_cap_applies_without_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem.v4 import observability as obs

        self._cfg(tmp_path, monkeypatch, {"enabled": True})
        obs.reset_for_tests()
        try:
            assert obs._effective_max_cardinality() == obs.MAX_CARDINALITY
        finally:
            obs.reset_for_tests()

    def test_invalid_cap_falls_back_to_the_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem.v4 import observability as obs

        self._cfg(tmp_path, monkeypatch, {"enabled": True, "max_cardinality": 0})
        obs.reset_for_tests()
        try:
            assert obs._effective_max_cardinality() == obs.MAX_CARDINALITY
        finally:
            obs.reset_for_tests()


# ---------------------------------------------------------------------------
# mcp decrypt audit — actor attribution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDecryptAuditActor:
    def test_actor_is_taken_from_the_request_context(self, tmp_path: Path) -> None:
        pytest.importorskip("fastapi")
        from mind_mem.api.rest import current_agent_id
        from mind_mem.mcp.tools.encryption import _append_decrypt_audit

        token = current_agent_id.set("agent-7")
        try:
            _append_decrypt_audit(str(tmp_path), str(tmp_path / "secret.md"), mode="read")
        finally:
            current_agent_id.reset(token)

        record = json.loads((tmp_path / "memory" / "decrypted_files.jsonl").read_text(encoding="utf-8").strip())
        assert record["actor"] == "agent-7"
        assert record["mode"] == "read"

    def test_actor_falls_back_to_anonymous_without_a_context(self, tmp_path: Path) -> None:
        from mind_mem.mcp.tools.encryption import _append_decrypt_audit

        _append_decrypt_audit(str(tmp_path), str(tmp_path / "secret.md"), mode="read")
        record = json.loads((tmp_path / "memory" / "decrypted_files.jsonl").read_text(encoding="utf-8").strip())
        assert record["actor"] == "anonymous"
