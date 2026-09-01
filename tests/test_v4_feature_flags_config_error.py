# Copyright 2026 STARGA, Inc.
"""Regression tests: a config that does not parse must not read as "flag off".

``_load_v4_block`` swallowed ``(OSError, ValueError)`` — and
``json.JSONDecodeError`` is a ``ValueError`` — returning ``{}`` with no log
line. One trailing comma in ``mind-mem.json`` therefore turned every v4
surface off at once, and ``require_enabled`` then told the operator to
"Enable via mind-mem.json" the very flag already sitting in the file:
the message pointed the diagnosis at the flag instead of the parse error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.v4 import feature_flags
from mind_mem.v4.feature_flags import (
    FeatureDisabledError,
    config_error,
    flag_config,
    is_enabled,
    require_enabled,
)

#: A registered v4 flag, used purely as a sample subject for the parse-error
#: path. It must stay registered: ``is_enabled`` fails closed on an unknown
#: name *before* it reads the config, so an unregistered sample silently stops
#: exercising the parse failure these tests exist to cover.
_FLAG = "federation"
_ENABLED = {"v4": {_FLAG: {"enabled": True}}}


def test_sample_flag_is_registered() -> None:
    """Guard: the sample flag must be real, or every test below tests nothing."""
    assert _FLAG in feature_flags.ALL_V4_FLAGS


@pytest.fixture(autouse=True)
def _reset_warning_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(feature_flags, "_last_config_warning", None)


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the flag reader at a config file this test writes."""
    path = tmp_path / "mind-mem.json"
    monkeypatch.setenv("MIND_MEM_CONFIG", str(path))

    def _write(text: str) -> Path:
        path.write_text(text, encoding="utf-8")
        return path

    return _write


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def _record(self, event: str, **kwargs: object) -> None:
        self.events.append((event, dict(kwargs)))

    def debug(self, event: str, **kwargs: object) -> None:
        self._record(event, **kwargs)

    def info(self, event: str, **kwargs: object) -> None:
        self._record(event, **kwargs)

    def warning(self, event: str, **kwargs: object) -> None:
        self._record(event, **kwargs)

    def error(self, event: str, **kwargs: object) -> None:
        self._record(event, **kwargs)


class TestUnparseableConfigIsReported:
    def test_config_error_names_the_parse_failure(self, config) -> None:
        config('{"v4": {"federation": {"enabled": true},}}')  # trailing comma
        error = config_error()
        assert error
        assert "mind-mem.json" in error

    def test_parse_failure_is_logged(self, config, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder()
        monkeypatch.setattr(feature_flags, "_log", recorder)
        config('{"v4": {"federation": {"enabled": true},}}')

        assert is_enabled(_FLAG) is False

        assert [event for event, _ in recorder.events] == ["v4_config_unreadable"]

    def test_repeat_reads_do_not_flood_the_log(self, config, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder()
        monkeypatch.setattr(feature_flags, "_log", recorder)
        config('{"v4": {"federation": {"enabled": true},}}')

        for _ in range(5):
            is_enabled(_FLAG)

        assert len(recorder.events) == 1

    def test_error_message_blames_the_config_not_the_flag(self, config) -> None:
        config('{"v4": {"federation": {"enabled": true},}}')

        with pytest.raises(FeatureDisabledError) as exc_info:
            require_enabled(_FLAG)

        message = str(exc_info.value)
        assert "could not be read" in message
        # The old message told the operator to add a flag that is already there.
        assert "Enable via mind-mem.json" not in message

    def test_unreadable_config_is_reported_too(self, config, monkeypatch: pytest.MonkeyPatch) -> None:
        path = config(json.dumps(_ENABLED))

        def _boom(*_args: object, **_kwargs: object) -> str:
            raise OSError("permission denied")

        monkeypatch.setattr(Path, "read_text", _boom)

        assert config_error()
        with pytest.raises(FeatureDisabledError, match="could not be read"):
            require_enabled(_FLAG)
        assert path.exists()


class TestHealthyConfigIsUnchanged:
    def test_no_config_file_is_not_an_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "absent.json"))
        assert config_error() == ""
        assert is_enabled(_FLAG) is False

    def test_enabled_flag_still_enables(self, config) -> None:
        config(json.dumps(_ENABLED))
        assert config_error() == ""
        assert is_enabled(_FLAG) is True
        require_enabled(_FLAG)  # must not raise

    def test_disabled_flag_keeps_the_original_message(self, config) -> None:
        config(json.dumps({"v4": {}}))

        with pytest.raises(FeatureDisabledError) as exc_info:
            require_enabled(_FLAG)

        assert "Enable via mind-mem.json" in str(exc_info.value)

    def test_flag_config_still_reads_tunables(self, config) -> None:
        config(json.dumps({"v4": {"long_context_recall": {"enabled": True, "max_tokens": 32000}}}))
        assert flag_config("long_context_recall")["max_tokens"] == 32000

    def test_a_fixed_config_clears_the_error(self, config) -> None:
        config('{"v4": {"federation": {"enabled": true},}}')
        assert config_error()
        config(json.dumps(_ENABLED))
        assert config_error() == ""
        assert is_enabled(_FLAG) is True

    def test_non_object_config_is_not_a_parse_error(self, config) -> None:
        """Valid JSON that is not an object: readable, just has no flags."""
        config("[1, 2, 3]")
        assert config_error() == ""
        assert is_enabled(_FLAG) is False
