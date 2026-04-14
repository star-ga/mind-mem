# Copyright 2026 STARGA, Inc.
"""Tests for the alerting router + sinks (v3.0.0 — GH #503)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mind_mem.alerting import (
    Alert,
    AlertRouter,
    AlertSink,
    LogSink,
    NullSink,
    SlackSink,
    WebhookSink,
    get_alert_router,
)


class _CaptureSink(AlertSink):
    """Test helper — records every alert instead of sending."""

    name = "capture"

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> bool:
        self.alerts.append(alert)
        return True


class TestAlertRouter:
    def test_severity_filter_drops_debug_under_warning_threshold(self) -> None:
        cap = _CaptureSink()
        router = AlertRouter(sinks=[cap], min_severity="warning")
        router.fire(severity="debug", event="x", payload={})
        router.fire(severity="info", event="x", payload={})
        router.fire(severity="warning", event="x", payload={})
        router.fire(severity="critical", event="x", payload={})
        # Only warning + critical reach the sink.
        assert len(cap.alerts) == 2
        assert [a.severity for a in cap.alerts] == ["warning", "critical"]

    def test_fire_returns_per_sink_bool(self) -> None:
        cap1 = _CaptureSink()
        cap2 = _CaptureSink()
        router = AlertRouter(sinks=[cap1, cap2], min_severity="info")
        results = router.fire(severity="warning", event="e", payload={})
        assert results == [True, True]

    def test_sink_exception_isolated(self) -> None:
        class BadSink(AlertSink):
            name = "bad"

            def send(self, alert: Alert) -> bool:
                raise RuntimeError("boom")

        cap = _CaptureSink()
        router = AlertRouter(sinks=[BadSink(), cap], min_severity="info")
        results = router.fire(severity="critical", event="e", payload={})
        # BadSink recorded as False; cap still got the alert.
        assert results == [False, True]
        assert len(cap.alerts) == 1

    def test_fire_stamps_timestamp(self) -> None:
        cap = _CaptureSink()
        router = AlertRouter(sinks=[cap], min_severity="info")
        router.fire(severity="info", event="x", payload={"a": 1})
        assert cap.alerts[0].timestamp
        # ISO 8601 UTC suffix
        assert cap.alerts[0].timestamp.endswith("Z")

    def test_add_sink_thread_safe(self) -> None:
        # Just make sure add_sink doesn't raise under concurrent use.
        import threading

        router = AlertRouter(sinks=[], min_severity="info")
        errors: list[Exception] = []

        def add_many() -> None:
            try:
                for _ in range(50):
                    router.add_sink(NullSink())
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=add_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(router.sinks) == 50 * 4


class TestWebhookSink:
    def test_rejects_non_http_url(self) -> None:
        with pytest.raises(ValueError):
            WebhookSink("file:///etc/passwd")

    def test_accepts_http_and_https(self) -> None:
        WebhookSink("https://example.com/alert")
        WebhookSink("http://localhost:9000/alert")

    @patch("mind_mem.alerting.urllib.request.urlopen")
    def test_send_posts_json(self, mock_open: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=None)
        mock_open.return_value = mock_resp

        sink = WebhookSink("https://example.com/hook")
        ok = sink.send(Alert(
            severity="critical", event="e", payload={"k": "v"},
            workspace="/ws", timestamp="2026-04-13T00:00:00Z",
        ))
        assert ok is True
        call = mock_open.call_args
        req = call.args[0]
        assert req.method == "POST"
        body = json.loads(req.data.decode())
        assert body["severity"] == "critical"
        assert body["event"] == "e"
        assert body["payload"] == {"k": "v"}


class TestSlackSink:
    @patch("mind_mem.alerting.urllib.request.urlopen")
    def test_send_formats_slack_payload(self, mock_open: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=None)
        mock_open.return_value = mock_resp

        sink = SlackSink("https://hooks.slack.com/services/T/B/X")
        ok = sink.send(Alert(
            severity="warning", event="drift_detected",
            payload={"block_a": "D-1", "sim": 0.42},
            workspace="/ws", timestamp="2026-04-13T00:00:00Z",
        ))
        assert ok is True
        body = json.loads(mock_open.call_args.args[0].data.decode())
        assert "mind-mem" in body["text"]
        assert body["attachments"][0]["title"] == "drift_detected"
        # attachments carry one field per payload key
        fields = body["attachments"][0]["fields"]
        assert {f["title"] for f in fields} == {"block_a", "sim"}


class TestGetAlertRouter:
    def test_without_config_returns_log_only(self, tmp_path: Path) -> None:
        router = get_alert_router(str(tmp_path))
        assert len(router.sinks) == 1
        assert isinstance(router.sinks[0], LogSink)

    def test_with_webhook_config_adds_webhook_sink(self, tmp_path: Path) -> None:
        (tmp_path / "mind-mem.json").write_text(json.dumps({
            "alerts": {
                "webhook_url": "https://example.com/alert",
                "min_severity": "critical",
            }
        }))
        router = get_alert_router(str(tmp_path))
        assert router.min_severity == "critical"
        names = [s.name for s in router.sinks]
        assert "log" in names
        assert "webhook" in names

    def test_with_slack_config_adds_slack_sink(self, tmp_path: Path) -> None:
        (tmp_path / "mind-mem.json").write_text(json.dumps({
            "alerts": {
                "slack_webhook_url": "https://hooks.slack.com/services/T/B/X",
            }
        }))
        router = get_alert_router(str(tmp_path))
        assert any(s.name == "slack" for s in router.sinks)

    def test_malformed_config_does_not_crash(self, tmp_path: Path) -> None:
        (tmp_path / "mind-mem.json").write_text("not-json")
        router = get_alert_router(str(tmp_path))
        # Log sink always present; malformed config silently dropped.
        assert len(router.sinks) == 1
