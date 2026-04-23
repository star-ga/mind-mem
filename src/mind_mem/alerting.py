# Copyright 2026 STARGA, Inc.
"""Governance alerting hooks (v3.0.0 — GH #503).

mind-mem generates governance-relevant signals (contradictions,
drift, rollback spikes, index staleness) but v2.x had no way to push
them outside the workspace. This module adds pluggable ``AlertSink``
implementations so operators can wire them into Slack, PagerDuty,
custom webhooks, or anywhere else.

Sinks are best-effort: a failing webhook never breaks the code path
that emitted the alert. Failures are logged + metered but not
propagated.

Config (in ``mind-mem.json``):

    {
      "alerts": {
        "webhook_url": "https://hooks.example.com/alerts",
        "slack_webhook_url": "https://hooks.slack.com/services/...",
        "min_severity": "warning"
      }
    }

Usage:

    from mind_mem.alerting import get_alert_router
    router = get_alert_router(workspace)
    router.fire(
        severity="critical",
        event="contradiction_detected",
        payload={"block_a": "D-1", "block_b": "D-2"},
    )

Zero external deps — stdlib-only urllib for webhook delivery.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .observability import get_logger, metrics

_log = get_logger("alerting")


_SEVERITY_ORDER: dict[str, int] = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "critical": 40,
}


def _severity_le(s: str, threshold: str) -> bool:
    return _SEVERITY_ORDER.get(s, 0) >= _SEVERITY_ORDER.get(threshold, 0)


# ---------------------------------------------------------------------------
# Alert envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Alert:
    """Immutable record an AlertSink receives.

    Fields intentionally minimal — downstream sinks format the payload
    however they prefer (Slack blocks, PagerDuty incident dedup keys, etc.).
    """

    severity: str  # "debug" | "info" | "warning" | "critical"
    event: str  # "contradiction_detected" | "rollback_spike" | ...
    payload: dict
    workspace: str
    timestamp: str = ""  # ISO 8601 UTC; filled by the router

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "event": self.event,
            "payload": self.payload,
            "workspace": self.workspace,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


class AlertSink(ABC):
    """Base class — implement :meth:`send` to deliver an alert."""

    name: str = "sink"

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Return True on successful delivery, False on failure."""


class NullSink(AlertSink):
    """Silent sink used when no alerting is configured."""

    name = "null"

    def send(self, alert: Alert) -> bool:  # pragma: no cover
        return True


class LogSink(AlertSink):
    """Sink that just emits a structured log line — useful default when
    no webhook is configured but operators still want the signal."""

    name = "log"

    def send(self, alert: Alert) -> bool:
        _log.warning(
            "alert",
            severity=alert.severity,
            alert_event=alert.event,
            workspace=alert.workspace,
            payload=alert.payload,
        )
        return True


class WebhookSink(AlertSink):
    """POSTs the alert envelope as JSON to a webhook URL."""

    name = "webhook"

    def __init__(self, url: str, *, timeout: float = 5.0) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"WebhookSink: invalid URL scheme {url!r}")
        self._url = url
        self._timeout = float(timeout)

    def send(self, alert: Alert) -> bool:
        body = json.dumps(alert.as_dict()).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # nosec B310 — scheme validated in __init__ to http/https only
                return bool(200 <= resp.status < 300)
        except (urllib.error.URLError, OSError) as exc:  # pragma: no cover
            _log.warning("webhook_send_failed", url=self._url, error=str(exc))
            return False


class SlackSink(AlertSink):
    """Formats + POSTs to a Slack incoming-webhook URL.

    Uses Slack's simplest text+attachment form so it works with every
    workspace's incoming-webhook integration — no Slack app install
    needed.
    """

    name = "slack"

    _COLOR: dict[str, str] = {
        "debug": "#808080",
        "info": "#3c82f6",
        "warning": "#f59e0b",
        "critical": "#ef4444",
    }

    def __init__(self, url: str, *, timeout: float = 5.0) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"SlackSink: invalid URL scheme {url!r}")
        if "hooks.slack.com" not in url:
            _log.warning("slack_sink_unexpected_url", url=url)
        self._url = url
        self._timeout = float(timeout)

    def send(self, alert: Alert) -> bool:
        color = self._COLOR.get(alert.severity, "#808080")
        text = f"*[mind-mem {alert.severity.upper()}]* {alert.event}"
        payload = {
            "text": text,
            "attachments": [
                {
                    "color": color,
                    "title": alert.event,
                    "text": f"workspace: `{alert.workspace}`",
                    "fields": [{"title": k, "value": str(v), "short": len(str(v)) < 40} for k, v in alert.payload.items()],
                    "ts": int(time.time()),
                }
            ],
        }
        req = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # nosec B310 — scheme validated in __init__ to http/https only
                return bool(200 <= resp.status < 300)
        except (urllib.error.URLError, OSError) as exc:  # pragma: no cover
            _log.warning("slack_send_failed", error=str(exc))
            return False


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@dataclass
class AlertRouter:
    """Fan-out to every configured sink, gated by severity threshold."""

    sinks: list[AlertSink] = field(default_factory=list)
    min_severity: str = "info"
    workspace: str = "."
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_sink(self, sink: AlertSink) -> None:
        with self._lock:
            self.sinks.append(sink)

    def fire(self, *, severity: str, event: str, payload: dict) -> list[bool]:
        """Deliver an alert to every sink at or above threshold.

        Returns a list of per-sink bools (True = delivered). Never raises.
        """
        if not _severity_le(severity, self.min_severity):
            return []
        alert = Alert(
            severity=severity,
            event=event,
            payload=dict(payload),
            workspace=self.workspace,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        metrics.inc("alerts_fired")
        metrics.inc(f"alerts_fired_{severity}")
        results: list[bool] = []
        with self._lock:
            sinks = list(self.sinks)
        for sink in sinks:
            try:
                ok = sink.send(alert)
            except Exception as exc:  # pragma: no cover — sink isolation
                _log.warning("alert_sink_error", sink=sink.name, error=str(exc))
                ok = False
            results.append(ok)
        return results


# ---------------------------------------------------------------------------
# Workspace integration
# ---------------------------------------------------------------------------


def _load_alerts_config(workspace: str) -> dict:
    path = os.path.join(workspace, "mind-mem.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return (json.load(fh) or {}).get("alerts", {}) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def get_alert_router(workspace: str) -> AlertRouter:
    """Build the configured :class:`AlertRouter` for *workspace*.

    Reads ``mind-mem.json`` and instantiates sinks from the ``alerts``
    section. Always includes :class:`LogSink` so signals land in the
    observability stream even without network sinks configured.
    """
    cfg = _load_alerts_config(workspace)
    min_sev = cfg.get("min_severity", "warning")
    router = AlertRouter(
        sinks=[LogSink()],
        min_severity=min_sev,
        workspace=os.path.realpath(workspace),
    )
    webhook = cfg.get("webhook_url")
    if isinstance(webhook, str) and webhook.strip():
        try:
            router.add_sink(WebhookSink(webhook.strip()))
        except ValueError as exc:
            _log.warning("invalid_webhook_url", error=str(exc))
    slack = cfg.get("slack_webhook_url")
    if isinstance(slack, str) and slack.strip():
        router.add_sink(SlackSink(slack.strip()))
    return router


__all__ = [
    "Alert",
    "AlertSink",
    "AlertRouter",
    "LogSink",
    "NullSink",
    "SlackSink",
    "WebhookSink",
    "get_alert_router",
]
