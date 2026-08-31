# Copyright 2026 STARGA, Inc.
"""Regression tests: a guardrail must not be lost to formatting or a typo.

Two silent-drop paths in :mod:`mind_mem.guardrails`:

* ``Status`` written in the block parser's *list* form (a bare ``Status:``
  line with ``- active`` under it) reached ``is_live()`` as the literal
  text ``"['active']"``, which matches no known status — so the guardrail
  was parsed, then dropped, and the destructive command it guards ran
  unguarded.  Worse, a bare ``Status:`` (rendered ``[]``) was dropped too,
  while *omitting* the field entirely stayed live: writing the field made
  the block less live than not writing it.
* A source named in ``recall.guardrails.sources`` that is not a file was
  skipped with no log at all, so a renamed or mistyped path is
  indistinguishable from a workspace that declares no guardrails.
"""

from __future__ import annotations

import os

import pytest

from mind_mem import guardrails as gr_mod
from mind_mem.block_parser import parse_file
from mind_mem.guardrails import (
    GuardrailPolicy,
    load_guardrails,
    parse_guardrail_block,
)

# A guardrail whose Status is written in the parser's list form.
LIST_STATUS_GUARDRAIL = """[GR-20260828-010]
Type: Guardrail
Statement: Never run `git reset --hard` without checking `git status` first.
Severity: critical
TriggerTools: Bash
TriggerCommands: git reset --hard
Status:
- active
"""

# Same block with the status written inline — the control.
INLINE_STATUS_GUARDRAIL = LIST_STATUS_GUARDRAIL.replace("Status:\n- active\n", "Status: active\n")

# Status field present but empty — the parser renders this as [].
BARE_STATUS_GUARDRAIL = LIST_STATUS_GUARDRAIL.replace("Status:\n- active\n", "Status:\n")

RETIRED_GUARDRAIL = LIST_STATUS_GUARDRAIL.replace("Status:\n- active\n", "Status: deprecated\n")


def _write(tmp_path, body: str, rel: str = "guardrails/GUARDRAILS.md") -> str:
    path = os.path.join(str(tmp_path), rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return str(tmp_path)


class _Recorder:
    """Stand-in for the module logger that records structured events."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def _record(self, level: str, event: str, **kwargs: object) -> None:
        self.events.append((level, event, dict(kwargs)))

    def debug(self, event: str, **kwargs: object) -> None:
        self._record("debug", event, **kwargs)

    def info(self, event: str, **kwargs: object) -> None:
        self._record("info", event, **kwargs)

    def warning(self, event: str, **kwargs: object) -> None:
        self._record("warning", event, **kwargs)

    def error(self, event: str, **kwargs: object) -> None:
        self._record("error", event, **kwargs)

    def named(self, event: str) -> list[dict]:
        return [payload for _lvl, name, payload in self.events if name == event]


# ---------------------------------------------------------------------------
# Status shapes
# ---------------------------------------------------------------------------


class TestListFormStatusStaysLive:
    def test_parser_really_produces_a_list(self, tmp_path) -> None:
        """Pin the mechanism: the drop came from the parser's list convention."""
        ws = _write(tmp_path, LIST_STATUS_GUARDRAIL)
        blocks = parse_file(os.path.join(ws, "guardrails", "GUARDRAILS.md"))
        assert blocks[0]["Status"] == ["active"]

    def test_list_form_status_is_live(self, tmp_path) -> None:
        ws = _write(tmp_path, LIST_STATUS_GUARDRAIL)
        loaded = load_guardrails(ws)
        assert [g.block_id for g in loaded] == ["GR-20260828-010"]
        assert loaded[0].is_live()
        assert loaded[0].status == "active"

    def test_list_and_inline_forms_agree(self, tmp_path) -> None:
        """Same declared state, two spellings — one outcome."""
        list_ws = _write(tmp_path / "list", LIST_STATUS_GUARDRAIL)
        inline_ws = _write(tmp_path / "inline", INLINE_STATUS_GUARDRAIL)
        listed = load_guardrails(list_ws)
        inlined = load_guardrails(inline_ws)
        assert [g.status for g in listed] == [g.status for g in inlined]
        assert [g.block_id for g in listed] == [g.block_id for g in inlined]

    def test_bare_status_reads_as_unstated(self, tmp_path) -> None:
        """``Status:`` with nothing under it is unstated, like an absent field."""
        ws = _write(tmp_path, BARE_STATUS_GUARDRAIL)
        loaded = load_guardrails(ws)
        assert [g.block_id for g in loaded] == ["GR-20260828-010"]
        assert loaded[0].status == ""

    def test_retired_status_still_drops(self, tmp_path) -> None:
        """The fix must not resurrect a deliberately retired guardrail."""
        ws = _write(tmp_path, RETIRED_GUARDRAIL)
        assert load_guardrails(ws) == ()

    def test_drop_is_logged_not_silent(self, tmp_path, monkeypatch) -> None:
        recorder = _Recorder()
        monkeypatch.setattr(gr_mod, "_log", recorder)
        ws = _write(tmp_path, RETIRED_GUARDRAIL)
        assert load_guardrails(ws) == ()
        dropped = recorder.named("guardrail_not_live")
        assert len(dropped) == 1
        assert dropped[0]["block_id"] == "GR-20260828-010"
        assert dropped[0]["status"] == "deprecated"


class TestNormaliseStatus:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, ""),
            ("", ""),
            ("  active  ", "active"),
            ([], ""),
            (["active"], "active"),
            (("wip",), "wip"),
            (["  active  "], "active"),
        ],
    )
    def test_shapes(self, raw: object, expected: str) -> None:
        assert gr_mod.normalise_status(raw) == expected

    def test_multi_valued_status_fails_closed(self) -> None:
        """Two states named at once is not one state — it must not be live."""
        block = {
            "_id": "GR-20260828-011",
            "Statement": "Never force-push a shared branch.",
            "TriggerCommands": "git push --force",
            "Status": ["active", "deprecated"],
        }
        guardrail = parse_guardrail_block(block)
        assert guardrail.status == "active, deprecated"
        assert not guardrail.is_live()


# ---------------------------------------------------------------------------
# Missing configured source
# ---------------------------------------------------------------------------


class TestMissingSourceIsReported:
    def test_configured_missing_source_warns(self, tmp_path, monkeypatch) -> None:
        recorder = _Recorder()
        monkeypatch.setattr(gr_mod, "_log", recorder)
        ws = _write(tmp_path, INLINE_STATUS_GUARDRAIL)
        policy = GuardrailPolicy(sources=("guardrails/RENAMED.md",))
        assert load_guardrails(ws, policy) == ()
        missing = recorder.named("guardrail_source_missing")
        assert len(missing) == 1
        assert missing[0]["source"] == "guardrails/RENAMED.md"

    def test_absent_default_source_stays_quiet(self, tmp_path, monkeypatch) -> None:
        """No initialiser creates the default file; warning on it would be noise."""
        recorder = _Recorder()
        monkeypatch.setattr(gr_mod, "_log", recorder)
        os.makedirs(str(tmp_path), exist_ok=True)
        assert load_guardrails(str(tmp_path)) == ()
        assert recorder.named("guardrail_source_missing") == []

    def test_present_configured_source_does_not_warn(self, tmp_path, monkeypatch) -> None:
        recorder = _Recorder()
        monkeypatch.setattr(gr_mod, "_log", recorder)
        ws = _write(tmp_path, INLINE_STATUS_GUARDRAIL, rel="guardrails/CUSTOM.md")
        policy = GuardrailPolicy(sources=("guardrails/CUSTOM.md",))
        loaded = load_guardrails(ws, policy)
        assert [g.block_id for g in loaded] == ["GR-20260828-010"]
        assert recorder.named("guardrail_source_missing") == []
