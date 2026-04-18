"""Regression test for the --force clobber bug in hook_installer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

from mind_mem.hook_installer import _merge_openclaw_hooks, install_config


def test_openclaw_merger_preserves_siblings_on_force() -> None:
    """_merge_openclaw_hooks must leave every other top-level key alone.

    The bug: force=True path in install_config discarded the existing
    dict and passed {} to the merger, dropping everything else in the
    file.
    """
    existing = {
        "hooks": {"internal": {"entries": {}}},
        "telegram": {"token": "sentinel-telegram"},
        "discord": {"token": "sentinel-discord"},
        "channels": ["chan-a", "chan-b"],
        "agents": {"planner": {"enabled": True}},
        "gateway": {"mode": "local", "port": 18791},
        "wizard": {"seen": True},
    }
    merged, changed = _merge_openclaw_hooks(existing, "/tmp/ws")
    for key in ("telegram", "discord", "channels", "agents", "gateway", "wizard"):
        assert key in merged, f"{key} was dropped by the merger"
    assert merged["telegram"]["token"] == "sentinel-telegram"
    assert merged["gateway"]["port"] == 18791
    assert "mind-mem" in merged["hooks"]["internal"]["entries"]


def test_install_config_force_preserves_siblings(tmp_path: Path) -> None:
    """End-to-end: install_config(force=True) must merge, not clobber.

    Redirect the openclaw config path via env HOME so we don't touch
    the real ~/.openclaw/openclaw.json.
    """
    fake_home = tmp_path
    (fake_home / ".openclaw").mkdir()
    config_path = fake_home / ".openclaw" / "openclaw.json"
    prior = {
        "hooks": {"internal": {"entries": {"other-hook": {"enabled": True}}}},
        "telegram": {"token": "sentinel-telegram"},
        "channels": ["chan-a"],
        "gateway": {"mode": "local", "port": 18791},
    }
    config_path.write_text(json.dumps(prior, indent=2), encoding="utf-8")

    # On Windows ``os.path.expanduser("~")`` reads ``USERPROFILE``;
    # on POSIX it reads ``HOME``.  Patch both so the install routine
    # writes to the temporary tree instead of the real user home.
    with mock.patch.dict(
        os.environ,
        {"HOME": str(fake_home), "USERPROFILE": str(fake_home)},
    ):
        install_config("openclaw", str(tmp_path), force=True)

    after = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("telegram", "channels", "gateway"):
        assert key in after, f"{key} was dropped by install_config(force=True)"
    assert after["telegram"]["token"] == "sentinel-telegram"
    assert "mind-mem" in after["hooks"]["internal"]["entries"]
    # The pre-existing sibling hook must also survive:
    assert "other-hook" in after["hooks"]["internal"]["entries"]
