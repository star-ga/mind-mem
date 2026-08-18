# Copyright 2026 STARGA, Inc.
"""Regression tests for ``mind_mem.self_update`` (``mm self-update`` + the
opt-in auto-update hook).

No real network and no real pip/pipx: ``urllib.request.urlopen`` and
``subprocess.run``/``Popen`` are monkeypatched throughout. State is
redirected to ``tmp_path`` per test (never touches a real ``~/.mind-mem``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError

import pytest

from mind_mem import self_update

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

CANNED_RELEASES: dict[str, list[dict[str, Any]]] = {
    "1.0.0": [{"filename": "mind_mem-1.0.0-py3-none-any.whl", "yanked": False}],
    "1.1.0": [{"filename": "mind_mem-1.1.0-py3-none-any.whl", "yanked": False}],
    "1.2.0rc1": [{"filename": "mind_mem-1.2.0rc1-py3-none-any.whl", "yanked": False}],
    "0.9.0": [{"filename": "mind_mem-0.9.0-py3-none-any.whl", "yanked": True}],
}


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _patch_online(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*_args: Any, **_kwargs: Any) -> _FakeResponse:
        body = json.dumps({"releases": CANNED_RELEASES}).encode("utf-8")
        return _FakeResponse(body)

    monkeypatch.setattr(self_update.urllib.request, "urlopen", fake_urlopen)


def _patch_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*_args: Any, **_kwargs: Any) -> Any:
        raise URLError("network is unreachable")

    monkeypatch.setattr(self_update.urllib.request, "urlopen", fake_urlopen)


class _RecordingPip:
    """Fake ``subprocess.run`` for pip: records argv, honors a scripted first failure."""

    def __init__(self, fail_first_with: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self._fail_first_with = fail_first_with

    def __call__(self, cmd: list[str], **_kwargs: Any) -> "subprocess.CompletedProcess[str]":
        self.calls.append(list(cmd))
        if self._fail_first_with and len(self.calls) == 1:
            return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr=self._fail_first_with)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="ok\n", stderr="")


def _make_args(**kwargs: Any) -> argparse.Namespace:
    ns = argparse.Namespace(check=False, yes=False, pre=False)
    for key, value in kwargs.items():
        setattr(ns, key, value)
    return ns


@pytest.fixture(autouse=True)
def _isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test gets its own state dir and a stable 'installed 1.0.0' baseline."""
    monkeypatch.setenv("MIND_MEM_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(self_update, "STATE_DIR", tmp_path)
    monkeypatch.setattr(self_update, "STATE_FILE", tmp_path / "update_state.json")
    monkeypatch.setattr(self_update, "get_installed_version", lambda: "1.0.0")
    monkeypatch.setattr(self_update, "is_editable_install", lambda: False)
    monkeypatch.setattr(self_update, "is_pipx_install", lambda: False)


# ---------------------------------------------------------------------------
# Case bodies (dispatched by the parametrized matrix test below)
# ---------------------------------------------------------------------------


def _case_check(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_online(monkeypatch)
    rc = self_update.cmd_self_update(_make_args(check=True))
    assert rc == self_update.EXIT_UPDATE_AVAILABLE
    err = capsys.readouterr().err
    assert "1.1.0" in err  # rc1 excluded (prerelease), yanked 0.9.0 excluded


def _case_upgrade(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_online(monkeypatch)
    recorder = _RecordingPip(fail_first_with="error: externally-managed-environment\n")
    monkeypatch.setattr(self_update.subprocess, "run", recorder)

    rc = self_update.cmd_self_update(_make_args(yes=True))

    assert rc == 0
    assert len(recorder.calls) == 2
    base = [self_update.sys.executable, "-m", "pip", "install", "--upgrade", self_update.PACKAGE]
    assert recorder.calls[0] == base
    assert recorder.calls[1] == [*base, "--break-system-packages"]


def _case_editable_skip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_online(monkeypatch)
    monkeypatch.setattr(self_update, "is_editable_install", lambda: True)
    recorder = _RecordingPip()
    monkeypatch.setattr(self_update.subprocess, "run", recorder)

    rc = self_update.cmd_self_update(_make_args(yes=True))

    assert rc == 0
    assert recorder.calls == []
    assert "editable" in capsys.readouterr().err.lower()


def _case_offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_offline(monkeypatch)

    rc = self_update.cmd_self_update(_make_args(check=True))
    assert rc == self_update.EXIT_CANNOT_CHECK

    status = self_update.check()
    assert status.latest is None
    assert status.error is not None

    # Default config (auto_update absent/off) — must be a silent, exception-free no-op.
    assert self_update.maybe_auto_check({}, None) is None


def _case_auto_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_online(monkeypatch)
    popen_calls: list[list[str]] = []

    def fake_popen(cmd: list[str], **_kwargs: Any) -> None:
        popen_calls.append(list(cmd))
        return None

    monkeypatch.setattr(self_update.subprocess, "Popen", fake_popen)
    config = {"auto_update": {"enabled": True, "mode": "notify", "interval_hours": 24, "channel": "stable"}}

    now = time.time()
    self_update._write_state({"last_check": now})
    self_update.maybe_auto_check(config, None)
    assert popen_calls == []
    assert self_update._read_state()["last_check"] == pytest.approx(now, abs=2.0)

    stale = now - 25 * 3600
    self_update._write_state({"last_check": stale})
    self_update.maybe_auto_check(config, None)

    assert len(popen_calls) == 1
    assert "--refresh-state" in popen_calls[0]
    assert popen_calls[0][-1] == "stable"
    assert self_update._read_state()["last_check"] > stale + 3600


def _case_pre_channel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_online(monkeypatch)
    recorder = _RecordingPip()
    monkeypatch.setattr(self_update.subprocess, "run", recorder)

    status = self_update.check(include_pre=True)
    assert status.latest == "1.2.0rc1"

    rc = self_update.cmd_self_update(_make_args(pre=True, yes=True))

    assert rc == 0
    assert len(recorder.calls) == 1
    assert "--pre" in recorder.calls[0]


_CASES: dict[str, Callable[[pytest.MonkeyPatch, Path, pytest.CaptureFixture[str]], None]] = {
    "check": _case_check,
    "upgrade": _case_upgrade,
    "editable_skip": _case_editable_skip,
    "offline": _case_offline,
    "auto_gate": _case_auto_gate,
    "pre_channel": _case_pre_channel,
}


@pytest.mark.parametrize("case", list(_CASES))
def test_self_update_matrix(case: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _CASES[case](monkeypatch, tmp_path, capsys)


# ---------------------------------------------------------------------------
# Focused unit coverage below the matrix (version parsing, editable refusal
# wiring, yanked-release filtering) — cheap, deterministic, no network.
# ---------------------------------------------------------------------------


class TestParseVersion:
    def test_orders_dev_pre_final_post(self) -> None:
        ordered = ["1.5.0.dev1", "1.5.0a1", "1.5.0b1", "1.5.0rc1", "1.5.0", "1.5.0.post1"]
        keys = [self_update.parse_version(v) for v in ordered]
        assert all(k is not None for k in keys)
        assert keys == sorted(keys)  # already in ascending order
        assert keys[0] < keys[-1]

    def test_higher_release_always_wins(self) -> None:
        assert self_update.parse_version("2.0.0") > self_update.parse_version("1.9.9.post5")

    def test_unparseable_returns_none(self) -> None:
        assert self_update.parse_version("not-a-version") is None
        assert self_update.parse_version("1.0.0+local.build") is None

    def test_is_prerelease(self) -> None:
        assert self_update.is_prerelease("1.0.0rc1") is True
        assert self_update.is_prerelease("1.0.0.dev1") is True
        assert self_update.is_prerelease("1.0.0") is False
        assert self_update.is_prerelease("garbage") is False


class TestYankedFiltering:
    def test_all_yanked_true_for_empty_or_fully_yanked(self) -> None:
        assert self_update._all_yanked([]) is True
        assert self_update._all_yanked([{"yanked": True}]) is True
        assert self_update._all_yanked([{"yanked": True}, {"yanked": False}]) is False

    def test_fetch_latest_version_skips_yanked_and_prerelease(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_online(monkeypatch)
        assert self_update.fetch_latest_version(include_pre=False) == "1.1.0"
        assert self_update.fetch_latest_version(include_pre=True) == "1.2.0rc1"

    def test_fetch_latest_version_returns_none_on_malformed_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(*_a: Any, **_k: Any) -> _FakeResponse:
            return _FakeResponse(b"not json")

        monkeypatch.setattr(self_update.urllib.request, "urlopen", fake_urlopen)
        assert self_update.fetch_latest_version() is None


class TestEditableRefusalWiring:
    def test_editable_status_reported_before_pip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_online(monkeypatch)
        monkeypatch.setattr(self_update, "is_editable_install", lambda: True)
        status = self_update.check()
        assert status.editable is True

    def test_unknown_mode_and_channel_fall_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mode, channel, interval = self_update._normalize_auto_update_config({"mode": "bogus", "channel": "bogus"})
        assert (mode, channel) == ("notify", "stable")
        assert interval == self_update.DEFAULT_INTERVAL_HOURS

    def test_interval_hours_clamped_to_minimum(self) -> None:
        _mode, _channel, interval = self_update._normalize_auto_update_config({"interval_hours": 0.01})
        assert interval == self_update._MIN_INTERVAL_HOURS
