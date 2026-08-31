"""``mm audit-pinned`` must resolve a ``~`` config path before deriving the
workspace.

``os.path.abspath`` does not expand ``~`` — it joins with the cwd — so
``_cmd_audit_pinned`` turned ``--config ~/ci/mind-mem.json`` into
``<cwd>/~/ci/mind-mem.json`` and handed the dirname of *that* to
``audit_pinned`` as the workspace. ``Path.expanduser()`` inside
``audit_pinned`` cannot rescue it: expanduser only expands a *leading*
tilde, and here the ``~`` is an interior path segment. Every relative pin
then resolved under a directory that does not exist, was recorded as
missing, and a missing pin is a SKIP that still exits 0 — a release gate
reporting PASS having audited nothing.

Both tests below fail on the pre-fix tree.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import pytest

from mind_mem.mm_cli import _cmd_audit_pinned


def _clean_checkpoint(root: Path) -> Path:
    """A checkpoint that passes the seven-check audit."""
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"model_type":"qwen3","base_model":"Qwen/Qwen3-8B"}', encoding="utf-8")
    body = b'{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(body)) + body + b"\x00" * 8)
    return root


def _evil_checkpoint(root: Path) -> Path:
    """A checkpoint the audit must reject (unknown publisher)."""
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"base_model":"evil-org/malicious-fork"}', encoding="utf-8")
    body = b'{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(body)) + body + b"\x00" * 8)
    return root


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("USERPROFILE", str(h))  # Windows
    # Run from somewhere else entirely: the bug was cwd-relative.
    workdir = tmp_path / "elsewhere"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    return h


def _args(config: str, *, fail_on_missing: bool = False) -> argparse.Namespace:
    return argparse.Namespace(config=config, json=True, fail_on_missing=fail_on_missing)


class TestTildeConfigPath:
    def test_relative_pin_under_a_tilde_config_is_actually_audited(self, home: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _clean_checkpoint(home / "ckpt")
        (home / "mind-mem.json").write_text(json.dumps({"audit_pinned_models": ["ckpt"]}), encoding="utf-8")

        rc = _cmd_audit_pinned(_args("~/mind-mem.json"))
        report = json.loads(capsys.readouterr().out)

        assert report["config_present"] is True
        # Pre-fix this was False ("pinned path is not an existing
        # directory: <cwd>/~/ckpt") and the audit never ran.
        assert report["findings"][0]["exists"] is True, report["findings"][0]["error"]
        assert report["findings"][0]["audit_passed"] is True
        assert rc == 0

    def test_a_failing_relative_pin_under_a_tilde_config_is_not_a_silent_zero(self, home: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """The false green, stated as an exit code."""
        _evil_checkpoint(home / "ckpt")
        (home / "mind-mem.json").write_text(json.dumps({"audit_pinned_models": ["ckpt"]}), encoding="utf-8")

        rc = _cmd_audit_pinned(_args("~/mind-mem.json"))
        report = json.loads(capsys.readouterr().out)

        assert report["findings"][0]["exists"] is True
        assert report["findings"][0]["audit_passed"] is False
        # Pre-fix: the pin looked missing, missing is a SKIP, SKIP passes → 0.
        assert rc == 1

    def test_plain_relative_config_path_still_works(self, home: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """No-regression control: the non-tilde path is unchanged."""
        cwd = Path.cwd()
        _clean_checkpoint(cwd / "ckpt")
        (cwd / "mind-mem.json").write_text(json.dumps({"audit_pinned_models": ["ckpt"]}), encoding="utf-8")

        rc = _cmd_audit_pinned(_args("mind-mem.json"))
        report = json.loads(capsys.readouterr().out)

        assert report["findings"][0]["exists"] is True
        assert rc == 0
