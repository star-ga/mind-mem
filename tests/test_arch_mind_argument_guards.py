# Copyright 2026 STARGA, Inc.
"""Identifier validation must match the boundary its docstring claims.

``_ARCH_ID_RE`` was ``^[A-Za-z0-9_.\\-]{1,128}$``. Two consequences the
docstring denied:

* the class permits a LEADING hyphen, so ``--help`` and ``-rf`` validated as
  identifiers even though the docstring said flag-prefix bytes were rejected;
* ``$`` matches before a trailing newline, so ``"id\\n"`` validated.

Neither is a shell injection -- ``subprocess.run`` is called with a list argv
and no ``shell=``, so the default ``shell=False`` applies, and that is asserted
below rather than assumed. It is a validation defect: the guard does not enforce
the boundary it documents, and a flag-shaped identifier reaching arch-mind's own
parser is the caller's argument becoming an option.

The 128-character boundary and internal ``-``, ``_``, ``.`` are LEGITIMATE and
must keep working; each has a positive control.
"""

from unittest import mock

import pytest

from mind_mem.mcp.tools import arch_mind
from mind_mem.mcp.tools.arch_mind import ArchMindError


class TestIdentifierBoundary:
    @pytest.mark.parametrize("bad", ["--help", "-rf", "-", "--", "-x"])
    def test_flag_prefixed_identifiers_are_refused(self, bad):
        with pytest.raises(ArchMindError):
            arch_mind._validate_arch_id("agent_id", bad)

    @pytest.mark.parametrize("bad", ["id\n", "id\r\n", "id\r", "id\t", "id ", " id", "a\nb"])
    def test_whitespace_anywhere_is_refused(self, bad):
        with pytest.raises(ArchMindError):
            arch_mind._validate_arch_id("agent_id", bad)

    @pytest.mark.parametrize(
        "good",
        ["agent-1", "agent_1", "agent.1", "a-b_c.d", "abc123", "A" * 128, "a", "9", "_lead", ".lead"],
    )
    def test_legitimate_identifiers_still_pass(self, good):
        """Positive control: the refusals above are about flags and whitespace."""
        assert arch_mind._validate_arch_id("agent_id", good) == good

    def test_the_128_boundary_is_preserved_exactly(self):
        assert arch_mind._validate_arch_id("agent_id", "A" * 128) == "A" * 128
        with pytest.raises(ArchMindError):
            arch_mind._validate_arch_id("agent_id", "A" * 129)

    def test_empty_and_non_str_are_refused(self):
        with pytest.raises(ArchMindError):
            arch_mind._validate_arch_id("agent_id", "")
        with pytest.raises(ArchMindError):
            arch_mind._validate_arch_id("agent_id", 7)  # type: ignore[arg-type]


class TestPathsAreNotOverRestricted:
    """Legal filenames contain spaces and metacharacters. With list argv that is safe."""

    @pytest.mark.parametrize(
        "good",
        ["/tmp/a b/repo", "/tmp/it's/repo", "/tmp/a;b/repo", "/tmp/a$b/repo", "/tmp/a*b/repo", "/tmp/a|b/repo", "/tmp/ünïcode/repo"],
    )
    def test_legal_paths_with_metacharacters_are_accepted(self, good):
        assert arch_mind._validate_arch_path("repo", good) == good

    @pytest.mark.parametrize("bad", ["", "-repo", "a\x00b"])
    def test_structurally_unsafe_paths_are_still_refused(self, bad):
        with pytest.raises(ArchMindError):
            arch_mind._validate_arch_path("repo", bad)


@pytest.fixture(autouse=True)
def _admin_scope(monkeypatch):
    """arch_session_start is ACL-gated to admin scope.

    At the default user scope it returns a permission-denied envelope BEFORE any
    validation runs, so a test left at that scope would pass while exercising
    the ACL and never the guard under test.
    """
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")


class TestNoChildIsLaunchedForInvalidInput:
    """The guard must refuse BEFORE a process starts, not after."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"agent_id": "--help"},
            {"agent_id": "id\n"},
            {"commit_sha": "--exec=evil"},
            {"commit_sha": "sha\n"},
        ],
    )
    def test_invalid_ids_never_reach_subprocess(self, kwargs, monkeypatch):
        monkeypatch.setenv("ARCH_MIND_BIN", "/nonexistent/arch-mind")
        call = {"n": 0}

        def _spy(*a, **k):
            call["n"] += 1
            raise AssertionError("a child process was launched for invalid input")

        monkeypatch.setattr(arch_mind.subprocess, "run", _spy)
        args = {"repo": "/tmp/repo", "fixture": "/tmp/fx", "agent_id": "agent-1", "commit_sha": "abc123"}
        args.update(kwargs)
        with pytest.raises(ArchMindError):
            arch_mind.arch_session_start(**args)
        assert call["n"] == 0, "subprocess.run was reached with invalid input"

    def test_the_valid_call_does_launch_and_preserves_argv(self, monkeypatch):
        """Positive control, and the argv contract."""
        monkeypatch.setenv("ARCH_MIND_BIN", "/opt/trusted/arch-mind")
        seen = {}

        def _fake_run(cmd, **kw):
            seen["cmd"] = cmd
            seen["kw"] = kw
            return mock.Mock(stdout="{}", stderr="", returncode=0)

        monkeypatch.setattr(arch_mind.subprocess, "run", _fake_run)
        arch_mind.arch_session_start(
            repo="/tmp/a b/repo",
            fixture="/tmp/fx",
            agent_id="agent-1",
            commit_sha="abc.123_x",
        )
        cmd = seen["cmd"]
        assert isinstance(cmd, list), "argv must be a list, never a string"
        assert cmd[0] == "/opt/trusted/arch-mind", "env-selected binary was not used"
        assert "session-start" in cmd
        # The path with a space survives intact as ONE argv element.
        assert "/tmp/a b/repo" in cmd
        assert "agent-1" in cmd and "abc.123_x" in cmd

    def test_the_child_is_never_run_through_a_shell(self, monkeypatch):
        monkeypatch.setenv("ARCH_MIND_BIN", "/opt/trusted/arch-mind")
        seen = {}

        def _fake_run(cmd, **kw):
            seen["kw"] = kw
            return mock.Mock(stdout="{}", stderr="", returncode=0)

        monkeypatch.setattr(arch_mind.subprocess, "run", _fake_run)
        arch_mind.arch_session_start(repo="/tmp/r", fixture="/tmp/f", agent_id="a", commit_sha="b")
        assert seen["kw"].get("shell", False) is False, "the child was run through a shell"
