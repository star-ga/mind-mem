"""A silent fallback is a config the operator thinks is in force and is not.

Two drifts between ``world_staleness_config`` and its own docstrings:

* ``_coerce_int`` promised "falls back to its default **with a warning**"
  (module docstring, and ``resolve_world_config``'s) and emitted nothing on
  either fallback path — neither the unparseable value nor the below-minimum
  one. ``"max_file_bytes": "10MB"`` and ``"max_reported": 0`` reverted in
  total silence.
* ``_read_flag_block`` promised "an unreadable config means OFF" and did the
  opposite: an unparseable workspace file left ``data = {}`` and fell through
  to the process-level resolver, so ``MIND_MEM_CONFIG`` could turn the feature
  ON against a workspace statement nobody had been able to read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import mind_mem.world_staleness_config as wsc
from mind_mem.world_staleness_config import DEFAULT_MAX_REPORTED, resolve_world_config


@pytest.fixture()
def warnings(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Capture the module logger's warnings as structured records."""
    recorded: list[dict] = []
    monkeypatch.setattr(wsc._log, "warning", lambda event, **kw: recorded.append({"event": event, **kw}))
    return recorded


def _workspace(tmp_path: Path, *, body: str, name: str = "ws") -> str:
    ws = tmp_path / name
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(body, encoding="utf-8")
    return str(ws)


def _config(tmp_path: Path, knobs: dict[str, Any], name: str = "ws") -> str:
    return _workspace(
        tmp_path,
        body=json.dumps({"v4": {"world_staleness": {"enabled": True, **knobs}}}),
        name=name,
    )


# --- _coerce_int warnings ---------------------------------------------------


def test_unparseable_knob_warns_and_falls_back(tmp_path: Path, warnings: list[dict]) -> None:
    cfg = resolve_world_config(_config(tmp_path, {"max_file_bytes": "10MB"}))
    assert cfg.max_file_bytes == wsc.DEFAULT_MAX_FILE_BYTES
    invalid = [w for w in warnings if w["event"] == "world_staleness_config_invalid"]
    assert [w["knob"] for w in invalid] == ["max_file_bytes"]
    assert "10MB" in invalid[0]["value"]


def test_below_minimum_knob_warns_and_falls_back(tmp_path: Path, warnings: list[dict]) -> None:
    cfg = resolve_world_config(_config(tmp_path, {"max_reported": 0}))
    assert cfg.max_reported == DEFAULT_MAX_REPORTED
    below = [w for w in warnings if w["event"] == "world_staleness_config_below_minimum"]
    assert [w["knob"] for w in below] == ["max_reported"]
    assert below[0]["minimum"] == 1
    assert below[0]["fallback"] == DEFAULT_MAX_REPORTED


def test_an_unset_knob_is_not_a_config_error(tmp_path: Path, warnings: list[dict]) -> None:
    """Absent knobs take the same fallback path — they must not spam the log."""
    cfg = resolve_world_config(_config(tmp_path, {}))
    assert cfg.max_reported == DEFAULT_MAX_REPORTED
    assert [w for w in warnings if w["event"].startswith("world_staleness_config_")] == []


def test_a_valid_knob_warns_about_nothing(tmp_path: Path, warnings: list[dict]) -> None:
    cfg = resolve_world_config(_config(tmp_path, {"max_reported": 7, "max_ref_drift": 2}))
    assert (cfg.max_reported, cfg.max_ref_drift) == (7, 2)
    assert [w for w in warnings if w["event"].startswith("world_staleness_config_")] == []


def test_every_bad_knob_is_named_separately(tmp_path: Path, warnings: list[dict]) -> None:
    """One warning per offending key, so the operator can fix all of them at once."""
    resolve_world_config(_config(tmp_path, {"max_file_bytes": "big", "max_reported": -3, "max_ref_drift": "x"}))
    named = {w["knob"] for w in warnings if w["event"].startswith("world_staleness_config_")}
    assert named == {"max_file_bytes", "max_reported", "max_ref_drift"}


# --- unreadable workspace config is OFF -------------------------------------


def test_unreadable_workspace_config_is_off_even_with_env_config_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    warnings: list[dict],
) -> None:
    """The documented fail-closed clause, against a process-level ON."""
    env_cfg = tmp_path / "env-mind-mem.json"
    env_cfg.write_text(json.dumps({"v4": {"world_staleness": {"enabled": True}}}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(env_cfg))

    ws = _workspace(tmp_path, body="{ this is not json", name="broken")
    assert resolve_world_config(ws).enabled is False
    assert [w["event"] for w in warnings] == ["world_staleness_config_unreadable"]


def test_a_workspace_with_no_config_file_still_defers_to_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fail-closed clause must not swallow the documented fallback.

    No file at all is a workspace with no opinion, which is a different thing
    from a workspace whose opinion could not be read.
    """
    env_cfg = tmp_path / "env-mind-mem.json"
    env_cfg.write_text(json.dumps({"v4": {"world_staleness": {"enabled": True}}}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(env_cfg))

    ws = tmp_path / "silent"
    ws.mkdir()
    assert not os.path.isfile(ws / "mind-mem.json")
    assert resolve_world_config(str(ws)).enabled is True


def test_a_readable_config_without_the_flag_still_defers_to_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_cfg = tmp_path / "env-mind-mem.json"
    env_cfg.write_text(json.dumps({"v4": {"world_staleness": {"enabled": True}}}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(env_cfg))

    ws = _workspace(tmp_path, body=json.dumps({"recall": {"vector_enabled": False}}), name="quiet")
    assert resolve_world_config(ws).enabled is True
