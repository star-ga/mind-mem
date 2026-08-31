# Copyright 2026 STARGA, Inc.
"""T-004: alert destination URLs are constrained, not arbitrary.

Before this gate an operator-supplied ``webhook_url`` in
``mind-mem.json`` was delivered to unconditionally. That config is not
always operator-authored in practice -- a workspace can be cloned,
shared, or checked into a repo -- so an attacker-influenced config could
point governance alerts (block IDs, workspace path, contradiction
payloads) at any host, including cloud metadata endpoints and internal
admin ports that are only reachable *from* the mind-mem host.

Two layers, both tested here:

1. Default-on, zero-config: SSRF-shaped destinations are refused.
   Loopback, link-local (169.254.0.0/16 -- cloud metadata), RFC1918,
   CGNAT, multicast, reserved, and unspecified addresses. No legitimate
   Slack or SaaS webhook resolves into those ranges.
2. Opt-in strict allowlist: ``MIND_MEM_ALERT_URL_ALLOWLIST`` pins
   delivery to named hosts. When set it is authoritative -- a host not
   on it is refused even though it is publicly routable.

``MIND_MEM_ALERT_ALLOW_ANY=true`` restores the legacy open behaviour for
the real case of a self-hosted webhook on a private network.
"""

from __future__ import annotations

import pytest

from mind_mem.alert_urls import AlertUrlError, assert_destination_allowed, validate_alert_url


class TestSchemeAndShape:
    def test_non_http_scheme_refused(self):
        for url in ("file:///etc/passwd", "gopher://x/", "ftp://h/x", "javascript:alert(1)"):
            with pytest.raises(AlertUrlError):
                validate_alert_url(url)

    def test_credentials_in_url_refused(self):
        # Embedded credentials are a redirect/exfil smell and are never
        # needed by a webhook receiver.
        with pytest.raises(AlertUrlError):
            validate_alert_url("https://user:pass@hooks.slack.com/services/x")

    def test_missing_host_refused(self):
        with pytest.raises(AlertUrlError):
            validate_alert_url("https:///no-host")

    def test_ordinary_https_url_accepted(self):
        assert validate_alert_url("https://hooks.slack.com/services/T/B/x")


class TestBlockedRanges:
    """Literal-IP destinations in SSRF-shaped ranges are refused by default."""

    @pytest.mark.parametrize(
        "host",
        [
            "169.254.169.254",  # cloud metadata -- the canonical SSRF target
            "127.0.0.1",  # loopback
            "0.0.0.0",  # unspecified
            "10.1.2.3",  # RFC1918
            "192.168.1.1",  # RFC1918
            "172.16.0.1",  # RFC1918
            "100.64.0.1",  # CGNAT -- NOT is_private in CPython, needs its own check
            "224.0.0.1",  # multicast
            "[::1]",  # IPv6 loopback
            "[fe80::1]",  # IPv6 link-local
            "[::ffff:169.254.169.254]",  # IPv4-mapped metadata -- must be unmapped first
        ],
    )
    def test_blocked(self, host, monkeypatch):
        monkeypatch.delenv("MIND_MEM_ALERT_URL_ALLOWLIST", raising=False)
        monkeypatch.delenv("MIND_MEM_ALERT_ALLOW_ANY", raising=False)
        with pytest.raises(AlertUrlError):
            assert_destination_allowed(f"https://{host}/hook")

    def test_public_literal_allowed(self, monkeypatch):
        monkeypatch.delenv("MIND_MEM_ALERT_URL_ALLOWLIST", raising=False)
        monkeypatch.delenv("MIND_MEM_ALERT_ALLOW_ANY", raising=False)
        assert_destination_allowed("https://93.184.216.34/hook")

    def test_allow_any_restores_legacy_behaviour(self, monkeypatch):
        monkeypatch.delenv("MIND_MEM_ALERT_URL_ALLOWLIST", raising=False)
        monkeypatch.setenv("MIND_MEM_ALERT_ALLOW_ANY", "true")
        assert_destination_allowed("https://169.254.169.254/hook")


class TestAllowlist:
    def test_allowlist_is_authoritative_when_set(self, monkeypatch):
        monkeypatch.delenv("MIND_MEM_ALERT_ALLOW_ANY", raising=False)
        monkeypatch.setenv("MIND_MEM_ALERT_URL_ALLOWLIST", "hooks.slack.com")
        assert_destination_allowed("https://hooks.slack.com/services/x")
        # Publicly routable, but not on the list -> still refused.
        with pytest.raises(AlertUrlError):
            assert_destination_allowed("https://evil.example.com/hook")

    def test_allowlist_matches_subdomains_but_not_suffix_lookalikes(self, monkeypatch):
        monkeypatch.setenv("MIND_MEM_ALERT_URL_ALLOWLIST", "example.com")
        assert_destination_allowed("https://hooks.example.com/x")
        # "notexample.com" ends with "example.com" as a STRING but is a
        # different domain -- a naive endswith() check would allow it.
        with pytest.raises(AlertUrlError):
            assert_destination_allowed("https://notexample.com/x")

    def test_allowlist_does_not_bypass_range_block(self, monkeypatch):
        """An allowlisted NAME must not smuggle in an internal ADDRESS."""
        monkeypatch.setenv("MIND_MEM_ALERT_URL_ALLOWLIST", "127.0.0.1")
        with pytest.raises(AlertUrlError):
            assert_destination_allowed("https://127.0.0.1/hook")

    def test_allowlist_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("MIND_MEM_ALERT_URL_ALLOWLIST", "Hooks.Slack.COM")
        assert_destination_allowed("https://hooks.slack.com/services/x")


class TestSinkIntegration:
    def test_webhook_sink_refuses_metadata_url_at_construction(self, monkeypatch):
        from mind_mem.alerting import WebhookSink

        monkeypatch.delenv("MIND_MEM_ALERT_ALLOW_ANY", raising=False)
        monkeypatch.delenv("MIND_MEM_ALERT_URL_ALLOWLIST", raising=False)
        with pytest.raises(ValueError):
            WebhookSink("https://169.254.169.254/latest/meta-data/")

    def test_slack_sink_refuses_metadata_url_at_construction(self, monkeypatch):
        from mind_mem.alerting import SlackSink

        monkeypatch.delenv("MIND_MEM_ALERT_ALLOW_ANY", raising=False)
        monkeypatch.delenv("MIND_MEM_ALERT_URL_ALLOWLIST", raising=False)
        with pytest.raises(ValueError):
            SlackSink("https://169.254.169.254/services/x")

    def test_router_drops_disallowed_sink_without_raising(self, monkeypatch, tmp_path):
        """A bad URL in config must not crash startup -- it must be dropped."""
        import json as _json

        from mind_mem.alerting import get_alert_router

        monkeypatch.delenv("MIND_MEM_ALERT_ALLOW_ANY", raising=False)
        monkeypatch.delenv("MIND_MEM_ALERT_URL_ALLOWLIST", raising=False)
        (tmp_path / "mind-mem.json").write_text(
            _json.dumps({"alerts": {"webhook_url": "https://169.254.169.254/x", "slack_webhook_url": "https://127.0.0.1/y"}}),
            encoding="utf-8",
        )
        router = get_alert_router(str(tmp_path))
        # LogSink survives; neither network sink was attached.
        assert [s.name for s in router.sinks] == ["log"]

    def test_send_revalidates_destination(self, monkeypatch):
        """Construction-time validation alone is a TOCTOU.

        A name that resolved publicly when the sink was built can resolve
        into an internal range later. send() must re-check rather than
        trusting the constructor's verdict.
        """
        from mind_mem.alerting import Alert, WebhookSink

        sink = WebhookSink("https://hooks.slack.com/services/x")
        calls: list[str] = []

        def _boom(url: str) -> None:
            calls.append(url)
            raise AlertUrlError("resolves into a blocked range")

        monkeypatch.setattr("mind_mem.alerting.assert_destination_allowed", _boom)
        alert = Alert(severity="critical", event="e", payload={}, workspace=".", timestamp="t")
        assert sink.send(alert) is False
        assert calls, "send() did not re-validate the destination"
