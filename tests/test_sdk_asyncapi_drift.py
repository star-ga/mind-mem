# Copyright 2026 STARGA, Inc.
"""The AsyncAPI artifact is tied to the real outbound event publisher."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mind_mem import event_fanout
from mind_mem.event_fanout import Event, RedisStreamPublisher
from mind_mem.spec import export_asyncapi

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "sdk" / "spec" / "asyncapi.json"


class _CaptureRedis:
    def __init__(self) -> None:
        self.record: dict[str, Any] | None = None

    def xadd(self, stream: str, fields: dict[str, str], *, maxlen: int, approximate: bool) -> str:
        assert self.record is None
        self.record = {"stream": stream, "fields": dict(fields)}
        assert maxlen == 10_000
        assert approximate is True
        return "1-0"


def _captured_event(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    capture = _CaptureRedis()

    def connect(url: str, **kwargs: Any) -> _CaptureRedis:
        assert url == "redis://localhost:6379/0"
        assert kwargs == {"decode_responses": True, "socket_timeout": 1.0}
        return capture

    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(from_url=connect))
    publisher = RedisStreamPublisher()
    publisher.publish(
        Event(
            kind=event_fanout.EVENT_PROPOSAL_APPLIED,
            payload={"proposal_id": "P-20260914-001", "statement": "must not leave the process"},
            workspace="synthetic-workspace",
            ts_wall=1_700_000_000.0,
        )
    )
    assert capture.record is not None
    return capture.record


class TestAsyncApiArtifact:
    def test_artifact_matches_live_publisher_contract(self) -> None:
        committed = export_asyncapi.load_committed_spec(ARTIFACT)
        live = export_asyncapi.build_live_spec()
        assert export_asyncapi.structural_diff(committed, live) == ""
        assert committed["asyncapi"] == "3.0.0"
        assert committed["channels"]["mindMemEvents"]["bindings"]["x-redis-stream"]["field"] == "data"

    def test_observed_emitter_inventory_is_source_bound(self) -> None:
        assert export_asyncapi.observed_event_kinds() == (
            "contradiction_detected",
            "proposal_applied",
            "rollback_executed",
            "tier_demoted",
            "tier_promoted",
        )
        spec = export_asyncapi.build_live_spec()
        assert spec["x-mind-mem"]["canonical_event_kinds"] == sorted(spec["x-mind-mem"]["canonical_event_kinds"])
        assert spec["x-mind-mem"]["observed_source_event_kinds"] == list(export_asyncapi.observed_event_kinds())

    def test_new_literal_emitter_forces_artifact_drift(self, tmp_path: Path) -> None:
        committed = export_asyncapi.load_committed_spec(ARTIFACT)
        (tmp_path / "emitter.py").write_text('def emit(workspace):\n    emit_event(workspace, "synthetic_emitter")\n', encoding="utf-8")
        live = export_asyncapi.build_live_spec(source_root=tmp_path)
        assert live["x-mind-mem"]["observed_source_event_kinds"] == ["synthetic_emitter"]
        assert export_asyncapi.structural_diff(committed, live)

    def test_artifact_corruption_fails_cli(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        corrupted = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        corrupted["channels"]["mindMemEvents"]["address"] = "wrong-stream"
        path = tmp_path / "asyncapi.json"
        path.write_text(json.dumps(corrupted), encoding="utf-8")
        monkeypatch.setattr(export_asyncapi, "SPEC_PATH", path)
        assert export_asyncapi._main(["--check"]) == 1

    def test_documented_module_command_passes(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "mind_mem.spec.export_asyncapi", "--check"],
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "matches the live outbound event contract" in result.stdout


class TestRedisWire:
    def test_real_publisher_call_is_validated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record = _captured_event(monkeypatch)
        body = export_asyncapi.validate_wire_record(record)
        assert body["kind"] == "proposal_applied"
        assert body["payload"] == {"proposal_id": "P-20260914-001", "_dropped": ["statement"]}

    @pytest.mark.parametrize(
        "mutator",
        [
            lambda fields: fields.update(data='{"kind":"x","kind":"y","payload":{},"workspace":null,"ts_wall":1}'),
            lambda fields: fields.update(data='{"kind":"x","payload":{},"workspace":null,"ts_wall":NaN}'),
            lambda fields: fields.update(data='{"kind":"x","payload":{"statement":"prose"},"workspace":null,"ts_wall":1}'),
            lambda fields: fields.update(data='{"kind":"x","payload":{},"workspace":null,"ts_wall":1,"extra":0}'),
        ],
    )
    def test_wire_mutation_is_refused(self, mutator: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        record = _captured_event(monkeypatch)
        mutator(record["fields"])
        with pytest.raises(ValueError):
            export_asyncapi.validate_wire_record(record)

    def test_wrong_stream_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record = _captured_event(monkeypatch)
        record["stream"] = "other-stream"
        with pytest.raises(ValueError, match="unexpected stream"):
            export_asyncapi.validate_wire_record(record)

    def test_missing_data_field_is_refused(self) -> None:
        with pytest.raises(ValueError):
            export_asyncapi.validate_wire_fields({"payload": ""})
