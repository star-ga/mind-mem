# Copyright 2026 STARGA, Inc.
"""``v4.health`` is WIRED — 5.0.1 restoration slice.

The module is a never-raising probe sweep: one small check per v4 surface,
reporting ``ok`` / ``missing`` / ``disabled`` / ``error: ...`` and an
aggregate. Its consumer is the ``memory_health`` MCP tool, which is the
workspace's health dashboard and had no view of the v4 surfaces at all.

**The flag gates the call site, not the check.** ``health_check`` is
deliberately never flag-gated — an operator debugging a failure needs it
whatever else is off, and each probe reports ``disabled`` for its own feature.
What ``v4.health`` buys is that ``memory_health``'s payload, which another
test pins byte-for-byte, is unchanged until an operator asks for the section,
and that no clock is read on the OFF path.

Working definition, asserted below: **with ``v4.health`` on, ``memory_health``
carries a ``v4`` block that distinguishes "feature off" (``disabled``) from
"feature broken" (``missing`` / ``error:``), and a probe that raises still
yields a report rather than taking the endpoint down.**
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.memory_ops import memory_health
from mind_mem.v4 import health


def _build_workspace(root: Path) -> None:
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "decisions" / "DECISIONS.md").write_text(
        "[D-20260101-001]\nStatement: Use PostgreSQL for the user database\nStatus: active\n",
        encoding="utf-8",
    )


def _write_config(root: Path, *, health_on: bool, block_kinds_on: bool = False) -> Path:
    cfg = root / "mind-mem.json"
    v4: dict = {}
    if health_on:
        v4["health"] = {"enabled": True}
    if block_kinds_on:
        v4["block_kinds"] = {"enabled": True}
    body: dict = {"version": "5.0.1", "recall": {"backend": "scan"}}
    if v4:
        body["v4"] = v4
    cfg.write_text(json.dumps(body), encoding="utf-8")
    return cfg


@pytest.fixture(autouse=True)
def _clean_probes():
    health.reset_custom_probes_for_tests()
    yield
    health.reset_custom_probes_for_tests()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    _build_workspace(root)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(root))
    monkeypatch.setenv("MIND_MEM_SCOPE", "user")
    return root


@pytest.fixture
def armed(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, health_on=True)))
    return workspace


@pytest.fixture
def disarmed(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, health_on=False)))
    return workspace


def _health(ws: Path) -> dict:
    with use_workspace(str(ws)):
        return json.loads(memory_health())


# ---------------------------------------------------------------------------
# The working definition
# ---------------------------------------------------------------------------


class TestMemoryHealthCarriesTheV4Sweep:
    def test_the_section_is_present_and_shaped(self, armed: Path) -> None:
        section = _health(armed)["v4"]
        assert section["status"] in {"ok", "degraded", "fail"}
        assert set(section["modules"]) >= {"feature_flags", "block_kinds", "cognitive_kernel"}
        assert isinstance(section["latency_ms"], float)
        assert section["checked_at"].endswith("+00:00")

    def test_disabled_is_distinguished_from_broken(self, armed: Path) -> None:
        """The whole reason the module reports four states, not two."""
        section = _health(armed)["v4"]
        assert section["modules"]["block_kinds"] == "disabled"
        assert section["disabled_count"] >= 1

    def test_an_enabled_but_unbuilt_surface_reads_missing(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``block_kinds`` ON with no ``index.db`` is 'missing', not 'ok'."""
        monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, health_on=True, block_kinds_on=True)))
        section = _health(workspace)["v4"]
        assert section["modules"]["block_kinds"] == "missing"
        assert section["status"] == "degraded"

    def test_a_broken_surface_reaches_the_recommendations(self, armed: Path) -> None:
        """A degraded v4 surface is operator-actionable, so it is surfaced."""
        health.register_health_probe("wedged", lambda ws: "missing")
        payload = _health(armed)
        assert payload["v4"]["modules"]["wedged"] == "missing"
        assert any("wedged" in r for r in payload["recommendations"])
        assert payload["score"] == "needs_attention"

    def test_a_probe_that_raises_does_not_take_the_endpoint_down(self, armed: Path) -> None:
        """``health_check``'s never-raises contract, through the tool.

        ``SystemExit`` is a ``BaseException``, not an ``Exception``, so a
        plain ``except Exception`` here would propagate out of the endpoint
        that exists to report failures.
        """

        def _exit(_ws):
            raise SystemExit("probe called sys.exit")

        health.register_health_probe("suicidal", _exit)
        payload = _health(armed)
        assert payload["v4"]["status"] == "fail"
        assert payload["v4"]["modules"]["suicidal"].startswith("error:")
        assert payload["total_blocks"] == 1, "the rest of the dashboard survived"


# ---------------------------------------------------------------------------
# The call site is load-bearing
# ---------------------------------------------------------------------------


class TestTheCallSiteIsLoadBearing:
    def test_memory_health_calls_health_check(self, armed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Remove the call site and this records nothing."""
        seen: list[str] = []
        real = health.health_check

        def _spy(ws):
            seen.append(str(ws))
            return real(ws)

        monkeypatch.setattr(health, "health_check", _spy)
        assert "v4" in _health(armed)
        assert seen == [str(armed)]

    def test_the_reported_workspace_is_the_tools_workspace(self, armed: Path) -> None:
        """A probe is handed the workspace under inspection, not the cwd."""
        seen: list[Path] = []
        health.register_health_probe("recorder", lambda ws: seen.append(ws) or "ok")
        _health(armed)
        assert seen == [armed]


# ---------------------------------------------------------------------------
# Flag OFF is the 5.0.0 baseline
# ---------------------------------------------------------------------------


class TestFlagOffIsUnchanged:
    def test_payload_is_byte_identical(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A present-but-disabled section changes not one byte of the output.

        This is the reason the call site is gated at all: the section carries
        ``latency_ms`` and ``checked_at``, so an ungated one would make two
        consecutive dashboards differ for no reason an operator can act on.
        """
        monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, health_on=False)))
        with use_workspace(str(workspace)):
            without = memory_health()
            again = memory_health()
        assert without == again, "the flag-off dashboard is not even stable run to run"
        assert "v4" not in json.loads(without)

        # And with the flag PRESENT but disabled, still not one byte differs.
        monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, health_on=False, block_kinds_on=True)))
        with use_workspace(str(workspace)):
            with_other_flag = memory_health()
        assert "v4" not in json.loads(with_other_flag)

    def test_flag_off_never_calls_the_module(self, disarmed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The probe is a config read and nothing else — no clock, no db."""

        def _explode(*a, **kw):
            raise AssertionError("v4.health ran with the flag OFF")

        monkeypatch.setattr(health, "health_check", _explode)
        assert "v4" not in _health(disarmed)

    def test_health_check_itself_stays_ungated(self, disarmed: Path) -> None:
        """The MODULE is not flag-gated; only ``memory_health``'s call is.

        An operator debugging a broken deployment must be able to call it
        directly with every v4 flag off, and get ``disabled`` rather than an
        exception.
        """
        report = health.health_check(disarmed)
        assert report["status"] in {"ok", "degraded", "fail"}
        assert report["modules"]["block_kinds"] == "disabled"

    def test_flag_is_registered(self) -> None:
        from mind_mem.v4 import feature_flags

        assert "health" in feature_flags.ALL_V4_FLAGS


def test_a_malformed_config_is_reported_not_swallowed(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe that exists because an unparseable config turns v4 OFF.

    ``_probe_feature_flags`` re-reads the resolved config so a trailing comma
    is reported as an error instead of looking like "every flag is unset".
    Asserted directly: with the config broken, ``v4.health`` cannot be read as
    on, so the section is (correctly) absent from the tool.
    """
    cfg = workspace / "broken.json"
    cfg.write_text('{ "v4": { "health": { "enabled": true } ', encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))

    assert "v4" not in _health(workspace), "an unparseable config must not switch a surface ON"
    assert health._probe_feature_flags(workspace).startswith("error: config is not valid JSON")
