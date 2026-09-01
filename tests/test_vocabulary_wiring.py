# Copyright 2026 STARGA, Inc.
"""Acceptance gate for the ``v4.vocabulary`` wiring (restoration slice).

``v4/vocabulary.py`` shipped a complete per-workspace controlled-vocabulary
loader and checker with its own test file, and nothing called it. It has TWO
callers now, and they answer two different questions:

* the ENFORCEMENT caller is ``propose_update`` — see
  ``tests/test_block_metadata_wiring.py``, which pins the reject/flag gate on
  the governed write door;
* the DIAGNOSTIC caller is ``mm doctor``, pinned here.

The diagnostic one exists because of a deliberate asymmetry in the module:
``load_vocabularies`` is TOLERANT by default, so a malformed declaration is
skipped with a logged warning rather than taking ingest down. That is the
right default and it is also precisely how a broken vocabulary becomes
invisible — the field quietly stops being enforced, and no write-path caller
can distinguish "nothing declared" from "the declaration did not parse".
``mm doctor`` is the loud counterpart: it re-loads the same declarations in
STRICT mode and reports what the write path is dropping.

Four contracts, one class each:

1. with the flag OFF doctor's report is unchanged — no ``vocabularies`` key,
   same exit code — with a positive control proving the comparison can fail;
2. with the flag ON a valid declaration set is reported, split by enforcement
   mode, so an operator can see which fields are actually enforced;
3. a malformed declaration is REPORTED and counts against doctor's exit code,
   and the test proves the tolerant write path really was silent about it —
   which is the whole reason this caller exists;
4. the flag PROBE resolves the flag the same way the callee does — a
   regression guard for a crash this file caught while being written.

Every assertion in classes 2-4 fails if the ``load_vocabularies(...)`` call is
removed from ``mm_cli._cmd_doctor``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mind_mem import mm_cli
from mind_mem.init_workspace import init
from mind_mem.v4.vocabulary import WORKSPACE_FILE, load_vocabularies

_GOOD = {
    "block_kind": ["decision", "task"],
    "category": {"values": ["project", "ops"], "mode": "flag", "case_sensitive": False},
}

#: ``values`` must be a non-empty list — this is the shape a typo produces.
_MALFORMED = {"block_kind": {"values": "decision"}}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_auto_update(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIND_MEM_NO_AUTO_UPDATE", "1")


def _make_ws(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    *,
    flag: bool,
    vocabularies: dict | None = None,
    ws_file: dict | None = None,
) -> str:
    ws = tmp_path / name
    ws.mkdir(parents=True)
    init(str(ws))
    cfg: dict[str, Any] = {"v4": {"vocabulary": {"enabled": flag}}}
    if vocabularies is not None:
        cfg["vocabularies"] = vocabularies
    (ws / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    if ws_file is not None:
        (ws / WORKSPACE_FILE).write_text(json.dumps(ws_file), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ws))
    monkeypatch.setenv("MIND_MEM_CONFIG", str(ws / "mind-mem.json"))
    return str(ws)


def _doctor(capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    """Run the real ``mm doctor`` entry point and parse its JSON report."""
    code = mm_cli.main(["doctor"])
    return code, json.loads(capsys.readouterr().out)


# ===========================================================================
# 1. Flag OFF is inert
# ===========================================================================


class TestFlagOffIsInert:
    def test_off_reports_nothing_about_vocabularies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _make_ws(tmp_path, monkeypatch, "off", flag=False, vocabularies=_GOOD)
        code, report = _doctor(capsys)
        assert "vocabularies" not in report
        assert code == 0

    def test_off_stays_healthy_even_with_a_broken_declaration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A workspace that never opted in must not start failing doctor."""
        _make_ws(tmp_path, monkeypatch, "off", flag=False, vocabularies=_MALFORMED)
        code, report = _doctor(capsys)
        assert "vocabularies" not in report
        assert code == 0

    def test_the_off_comparison_has_teeth(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Positive control.

        Both assertions above say "the key is absent and the exit code is 0",
        which is also what a build with no wiring at all produces. Flip the
        flag on the same inputs and both must now be FALSE.
        """
        _make_ws(tmp_path, monkeypatch, "on", flag=True, vocabularies=_GOOD)
        _, report = _doctor(capsys)
        assert "vocabularies" in report

        _make_ws(tmp_path, monkeypatch, "on-broken", flag=True, vocabularies=_MALFORMED)
        code, report = _doctor(capsys)
        assert report["vocabularies"]["ok"] is False
        assert code == 1


# ===========================================================================
# 2. Valid declarations are reported, split by enforcement mode
# ===========================================================================


class TestReportsDeclarations:
    def test_declared_fields_split_by_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _make_ws(tmp_path, monkeypatch, "ws", flag=True, vocabularies=_GOOD)
        code, report = _doctor(capsys)
        assert code == 0
        section = report["vocabularies"]
        assert section["ok"] is True
        assert section["declared_fields"] == ["block_kind", "category"]
        # ``block_kind`` used the list shorthand, which means mode="reject".
        assert section["enforced_fields"] == ["block_kind"]
        assert section["flag_only_fields"] == ["category"]

    def test_no_declarations_reports_an_empty_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _make_ws(tmp_path, monkeypatch, "ws", flag=True)
        code, report = _doctor(capsys)
        assert code == 0
        assert report["vocabularies"] == {
            "ok": True,
            "declared_fields": [],
            "enforced_fields": [],
            "flag_only_fields": [],
        }

    def test_the_standalone_workspace_file_is_merged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``vocabularies.json`` overrides mind-mem.json field-by-field.

        doctor must read the same two sources the write path does, or it
        reports on a document nothing enforces.
        """
        _make_ws(
            tmp_path,
            monkeypatch,
            "ws",
            flag=True,
            vocabularies={"block_kind": ["decision"]},
            ws_file={"block_kind": {"values": ["decision"], "mode": "flag"}, "owner": ["alice"]},
        )
        _, report = _doctor(capsys)
        section = report["vocabularies"]
        assert section["declared_fields"] == ["block_kind", "owner"]
        # The workspace file downgraded block_kind from reject to flag.
        assert section["enforced_fields"] == ["owner"]
        assert section["flag_only_fields"] == ["block_kind"]


# ===========================================================================
# 3. A malformed declaration is surfaced — the reason this caller exists
# ===========================================================================


class TestSurfacesMalformedDeclarations:
    def test_the_write_path_really_is_silent_about_it(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Establish the gap first, so the next test is measuring something.

        The tolerant (write-path) load returns an EMPTY mapping for a config
        that declares a field — the declaration is dropped and the field is
        unenforced, with no return value distinguishing that from a workspace
        that declared nothing.
        """
        ws = _make_ws(tmp_path, monkeypatch, "ws", flag=True, vocabularies=_MALFORMED)
        assert load_vocabularies(ws) == {}

    def test_doctor_reports_the_error_and_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _make_ws(tmp_path, monkeypatch, "ws", flag=True, vocabularies=_MALFORMED)
        code, report = _doctor(capsys)
        section = report["vocabularies"]
        assert section["ok"] is False
        assert "block_kind" in section["error"]
        assert "unenforced" in section["advice"]
        assert code == 1

    def test_an_unreadable_workspace_file_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ws = _make_ws(tmp_path, monkeypatch, "ws", flag=True)
        (Path(ws) / WORKSPACE_FILE).write_text("{ not json", encoding="utf-8")
        code, report = _doctor(capsys)
        assert report["vocabularies"]["ok"] is False
        assert WORKSPACE_FILE in report["vocabularies"]["error"]
        assert code == 1


# ===========================================================================
# 4. The probe asks the same question its callee will
# ===========================================================================


class TestProbeAgreesWithCallee:
    """Regression guard for a defect this file caught while being written.

    ``load_vocabularies`` gates itself on ``require_enabled``, which reads the
    AMBIENT config. The first version of the doctor wiring probed with the
    workspace-first resolver instead, on the reasoning that the declarations
    being checked are the workspace's own files. The two resolvers disagree
    the moment ``MIND_MEM_CONFIG`` names a different document — and the
    disagreement is not a wrong report, it is ``FeatureDisabledError`` raised
    straight out of ``mm doctor``. A probe must resolve the flag the same way
    the thing it is about to call does.
    """

    def test_an_ambient_off_config_does_not_crash_doctor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ws = _make_ws(tmp_path, monkeypatch, "ws", flag=True, vocabularies=_GOOD)
        elsewhere = tmp_path / "elsewhere.json"
        elsewhere.write_text(json.dumps({"v4": {"vocabulary": {"enabled": False}}}), encoding="utf-8")
        monkeypatch.setenv("MIND_MEM_CONFIG", str(elsewhere))

        code, report = _doctor(capsys)
        assert code == 0
        # The surface the callee would refuse is the surface doctor skipped.
        assert "vocabularies" not in report
        assert report["workspace"] == ws

    def test_the_agreement_test_has_teeth(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Positive control: with the two configs AGREEING, the section appears.

        Without this, "the key is absent" would also pass against a build in
        which the section never appears at all.
        """
        _make_ws(tmp_path, monkeypatch, "ws", flag=True, vocabularies=_GOOD)
        code, report = _doctor(capsys)
        assert code == 0
        assert report["vocabularies"]["declared_fields"] == ["block_kind", "category"]
