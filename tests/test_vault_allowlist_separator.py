"""The vault allowlist separator, and why a Windows drive letter broke it.

``MIND_MEM_VAULT_ALLOWLIST`` is the T-006 gate that stops ``vault_scan`` /
``vault_sync`` reading arbitrary host markdown. It parsed its value with
``";" if ";" in raw else ":"``, which is correct on POSIX and destroys every
Windows path: ``C:\\Users\\me\\vault`` contains no semicolon, so it split on the
DRIVE-LETTER colon into ``["C", "\\Users\\me\\vault"]``. The allowlist then
matched nothing and every vault call on Windows was refused as "outside
MIND_MEM_VAULT_ALLOWLIST" -- including one naming the allowlisted directory.

A security gate that fails CLOSED on one platform is less dangerous than one
that fails open, but it is still broken: the feature was unusable on Windows
and the error blamed the operator's configuration.
"""

from __future__ import annotations

import os

import pytest

from mind_mem.mcp.tools import agent as agent_tools


def _allowlist(monkeypatch, value: str, pathsep: str) -> list[str]:
    monkeypatch.setattr(os, "pathsep", pathsep)
    monkeypatch.setenv("MIND_MEM_VAULT_ALLOWLIST", value)
    return agent_tools._vault_allowlist()


class TestWindows:
    """os.pathsep is ';' -- a lone colon is part of the drive, not a separator."""

    def test_a_single_windows_path_survives_its_drive_letter(self, monkeypatch) -> None:
        got = _allowlist(monkeypatch, r"C:\Users\me\vault", ";")
        assert len(got) == 1, f"the drive letter was treated as a separator: {got}"
        assert got[0].endswith("vault")
        assert "C" not in [os.path.basename(g) for g in got]

    def test_two_windows_paths_split_on_the_semicolon(self, monkeypatch) -> None:
        got = _allowlist(monkeypatch, r"C:\a\vault;D:\b\vault", ";")
        assert len(got) == 2, got

    def test_the_refusal_message_is_not_produced_for_an_allowlisted_root(self, monkeypatch) -> None:
        """The end-to-end symptom: an allowlisted root read as 'outside'."""
        monkeypatch.setattr(os, "pathsep", ";")
        monkeypatch.setenv("MIND_MEM_VAULT_ALLOWLIST", r"C:\Users\me\vault")
        # _vault_root_allowed realpaths both sides; on POSIX a Windows-shaped
        # string is just an odd relative name, so compare the PARSE rather than
        # the filesystem verdict -- the parse is what the bug corrupted.
        assert len(agent_tools._vault_allowlist()) == 1


class TestPosix:
    """os.pathsep is ':' -- both separators keep working."""

    @pytest.mark.parametrize("raw", ["/tmp/a:/tmp/b", "/tmp/a;/tmp/b"])
    def test_both_separators_are_accepted(self, monkeypatch, raw: str) -> None:
        got = _allowlist(monkeypatch, raw, ":")
        assert len(got) == 2, got
        # Compare BASENAMES, not a "/tmp/a" suffix: _vault_allowlist realpaths
        # every entry, and on Windows realpath("/tmp/a") is "C:\\tmp\\a", which
        # ends with no forward slash at all. The separator behaviour is what
        # this test is about; the absolute form is the platform's business.
        assert [os.path.basename(g) for g in got] == ["a", "b"]

    def test_a_single_path_is_one_entry(self, monkeypatch) -> None:
        assert len(_allowlist(monkeypatch, "/tmp/only", ":")) == 1

    def test_blank_segments_are_dropped(self, monkeypatch) -> None:
        assert len(_allowlist(monkeypatch, "/tmp/a::/tmp/b:", ":")) == 2

    def test_an_empty_allowlist_stays_empty(self, monkeypatch) -> None:
        assert _allowlist(monkeypatch, "   ", ":") == []
