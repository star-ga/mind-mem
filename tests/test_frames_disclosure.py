# Copyright 2026 STARGA, Inc.
"""Nothing a frame or a dead end drops may be dropped silently.

Both of these were found by driving the shipped surface, not by reading
it, and both are the same failure wearing two hats: **negative memory
that disappears quietly is worse than negative memory that was never
recorded**, because the agent now has positive evidence of absence.

* A malformed frame is skipped with a structured log warning — on
  ``stderr``. ``mm resume`` printed ``No active task frame.`` on stdout
  and ``mm resume --json`` emitted ``{"frame_id": ""}``, so an agent
  reading stdout (which is every agent) concluded the workspace held no
  continuity and re-derived its context: exactly the cost the feature
  exists to remove. One mistyped step status did it.
* ``max_warnings`` truncates the dead ends attached to a brief. The
  count published alongside them was the count *after* truncation, so
  six firing dead ends and five firing dead ends rendered identically.

Determinism is preserved throughout: rejections are ordered by block id
and carry only text derived from the block, so two processes still
render the same bytes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

FRAMES = "frames/FRAMES.md"
DEAD_ENDS = "frames/DEAD-ENDS.md"


def _write(root: str, relative: str, text: str) -> None:
    path = os.path.join(root, relative)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(textwrap.dedent(text).lstrip())


@pytest.fixture
def rejecting_workspace(tmp_path):
    """One frame, refused for a step status outside the closed vocabulary."""
    root = str(tmp_path / "ws")
    os.makedirs(root)
    _write(
        root,
        FRAMES,
        """
        [TF-20260829-001]
        Type: TaskFrame
        Goal: The only frame this workspace declares.
        Status: active
        Steps:
        - todo: land the change
        - fix: the parser bug
        ApproachTools: Bash
        """,
    )
    return root


@pytest.fixture
def crowded_workspace(tmp_path):
    """A frame plus three dead ends that all fire on it.

    A real ``init``ed workspace: the MCP tools gate on ``_check_workspace``
    before they read anything, so a bare directory would exercise the
    refusal path instead of the disclosure this file is about.
    """
    from mind_mem.init_workspace import init

    root = str(tmp_path / "ws")
    init(root)
    _write(
        root,
        FRAMES,
        """
        [TF-20260829-001]
        Type: TaskFrame
        Goal: Close the last two floors.
        Status: active
        ApproachTools: Bash
        ApproachIntents: prove_floor
        """,
    )
    _write(
        root,
        DEAD_ENDS,
        """
        [DE-20260826-001]
        Type: DeadEnd
        Approach: Additive per-object assignment bound.
        WhyFailed: A co-carrier helper divides the matching by K.
        Outcome: refuted
        TriggerTools: Bash
        Status: active

        [DE-20260826-002]
        Type: DeadEnd
        Approach: Explicit BFS over the level graph.
        WhyFailed: The walk never terminates.
        Outcome: blocked
        TriggerIntents: prove_floor
        Status: active

        [DE-20260826-003]
        Type: DeadEnd
        Approach: Widening the helper set.
        WhyFailed: Regressed two closed floors.
        Outcome: regressed
        TriggerTools: Bash
        Status: active

        [DE-20260826-004]
        Type: DeadEnd
        Approach: Malformed on purpose.
        WhyFailed: Its outcome is outside the closed vocabulary.
        Outcome: probably-fine
        TriggerTools: Bash
        Status: active
        """,
    )
    return root


class TestARejectedBlockIsNamed:
    def test_the_brief_names_the_frame_it_could_not_read(self, rejecting_workspace):
        from mind_mem.resume_brief import resume_brief

        brief = resume_brief(rejecting_workspace)
        assert brief.is_empty()
        assert [r.block_id for r in brief.rejected] == ["TF-20260829-001"]
        assert "step status" in brief.rejected[0].reason

    def test_the_rejection_names_its_source_file(self, rejecting_workspace):
        from mind_mem.resume_brief import resume_brief

        assert resume_brief(rejecting_workspace).rejected[0].source_file == FRAMES

    def test_the_rendered_brief_does_not_claim_there_is_no_frame(self, rejecting_workspace):
        from mind_mem.resume_brief import render_resume_brief, resume_brief

        text = render_resume_brief(resume_brief(rejecting_workspace))
        assert "TF-20260829-001" in text
        assert "step status" in text

    def test_json_carries_the_rejection(self, rejecting_workspace):
        from mind_mem.resume_brief import resume_brief

        payload = json.loads(json.dumps(resume_brief(rejecting_workspace).to_dict()))
        assert payload["rejected"][0]["block_id"] == "TF-20260829-001"

    def test_a_rejected_dead_end_is_named_too(self, crowded_workspace):
        from mind_mem.resume_brief import resume_brief

        rejected = {r.block_id for r in resume_brief(crowded_workspace).rejected}
        assert "DE-20260826-004" in rejected

    def test_a_clean_workspace_reports_no_rejections(self, tmp_path):
        from mind_mem.resume_brief import render_resume_brief, resume_brief

        root = str(tmp_path / "clean")
        os.makedirs(root)
        _write(
            root,
            FRAMES,
            """
            [TF-20260829-001]
            Type: TaskFrame
            Goal: A frame with nothing wrong with it.
            Status: active
            ApproachTools: Bash
            """,
        )
        brief = resume_brief(root)
        assert brief.rejected == ()
        assert "REJECTED" not in render_resume_brief(brief)

    def test_rejections_are_ordered_by_block_id(self, tmp_path):
        from mind_mem.task_frames import load_task_frames_with_rejections

        root = str(tmp_path / "ws")
        os.makedirs(root)
        _write(
            root,
            FRAMES,
            """
            [TF-20260829-009]
            Type: TaskFrame
            Status: active
            Steps:
            - nope: bad status

            [TF-20260829-002]
            Type: TaskFrame
            Goal: Also broken.
            Status: active
            Steps:
            - alsonope: bad status
            """,
        )
        _frames, rejected = load_task_frames_with_rejections(root)
        assert [r.block_id for r in rejected] == ["TF-20260829-002", "TF-20260829-009"]


class TestTruncationIsCountedNotHidden:
    def test_the_brief_reports_how_many_dead_ends_actually_fired(self, crowded_workspace):
        from mind_mem.resume_brief import resume_brief
        from mind_mem.task_frames import FramePolicy

        brief = resume_brief(crowded_workspace, policy=FramePolicy(max_warnings=2))
        assert len(brief.dead_ends) == 2
        assert brief.dead_end_total == 3
        assert brief.dead_ends_elided == 1

    def test_no_truncation_reports_no_elision(self, crowded_workspace):
        from mind_mem.resume_brief import resume_brief

        brief = resume_brief(crowded_workspace)
        assert brief.dead_end_total == 3
        assert brief.dead_ends_elided == 0

    def test_the_render_says_how_many_were_elided(self, crowded_workspace):
        from mind_mem.resume_brief import render_resume_brief, resume_brief
        from mind_mem.task_frames import FramePolicy

        text = render_resume_brief(resume_brief(crowded_workspace, policy=FramePolicy(max_warnings=1)))
        assert "2 more" in text

    def test_json_carries_the_elision(self, crowded_workspace):
        from mind_mem.resume_brief import resume_brief
        from mind_mem.task_frames import FramePolicy

        payload = resume_brief(crowded_workspace, policy=FramePolicy(max_warnings=1)).to_dict()
        assert payload["dead_end_total"] == 3
        assert payload["dead_ends_elided"] == 2

    def test_a_disabled_policy_still_reports_what_would_have_fired(self, crowded_workspace):
        """A kill switch may silence warnings; it may not fake their absence."""
        from mind_mem.resume_brief import resume_brief
        from mind_mem.task_frames import FramePolicy

        brief = resume_brief(crowded_workspace, policy=FramePolicy(enabled=False))
        assert brief.dead_ends == ()
        assert brief.dead_end_total == 3
        assert brief.dead_ends_elided == 3


class TestTheMcpToolsDiscloseTheSameThings:
    def test_resume_brief_publishes_the_rejections_and_the_elision(self, crowded_workspace, monkeypatch):
        from mind_mem.mcp.tools import frames as frames_tools

        monkeypatch.setenv("MIND_MEM_WORKSPACE", crowded_workspace)
        monkeypatch.setenv("MIND_MEM_ACL_DISABLED", "true")
        payload = json.loads(frames_tools.resume_brief())
        assert payload["dead_end_total"] == 3
        assert [r["block_id"] for r in payload["rejected"]] == ["DE-20260826-004"]

    def test_check_dead_ends_publishes_the_elision(self, crowded_workspace, monkeypatch):
        from mind_mem.mcp.tools import frames as frames_tools

        monkeypatch.setenv("MIND_MEM_WORKSPACE", crowded_workspace)
        monkeypatch.setenv("MIND_MEM_ACL_DISABLED", "true")
        monkeypatch.setattr(
            "mind_mem.task_frames.FramePolicy.from_config",
            classmethod(lambda cls, config: cls(max_warnings=1)),
        )
        payload = json.loads(frames_tools.check_dead_ends(tool="Bash"))
        assert payload["count"] == 1
        assert payload["total_matched"] == 2
        assert payload["elided"] == 1

    def test_check_dead_ends_publishes_the_rejections(self, crowded_workspace, monkeypatch):
        from mind_mem.mcp.tools import frames as frames_tools

        monkeypatch.setenv("MIND_MEM_WORKSPACE", crowded_workspace)
        monkeypatch.setenv("MIND_MEM_ACL_DISABLED", "true")
        payload = json.loads(frames_tools.check_dead_ends(tool="Bash"))
        assert [r["block_id"] for r in payload["rejected"]] == ["DE-20260826-004"]


class TestTheCliDisclosesTheSameThings:
    def test_mm_resume_names_the_rejected_frame_on_stdout(self, rejecting_workspace):
        result = _run_cli(rejecting_workspace, "resume")
        assert result.returncode == 0
        assert "TF-20260829-001" in result.stdout
        assert "No active task frame." not in result.stdout

    def test_mm_resume_json_names_the_rejected_frame(self, rejecting_workspace):
        result = _run_cli(rejecting_workspace, "resume", "--json")
        payload = json.loads(result.stdout)
        assert payload["rejected"][0]["block_id"] == "TF-20260829-001"

    def test_mm_dead_ends_reports_rejections_and_elision(self, crowded_workspace):
        result = _run_cli(crowded_workspace, "dead-ends", "--tool", "Bash")
        payload = json.loads(result.stdout)
        assert payload["total_matched"] == 2
        assert [r["block_id"] for r in payload["rejected"]] == ["DE-20260826-004"]


class TestDisclosureIsStillDeterministic:
    def test_two_processes_render_identical_bytes(self, crowded_workspace):
        """The new fields are derived from block text, so nothing varies."""
        digests = {_digest(crowded_workspace, seed) for seed in ("0", "524287")}
        assert len(digests) == 1


def _run_cli(workspace: str, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, MIND_MEM_WORKSPACE=workspace, PYTHONPATH=_src())
    return subprocess.run(
        [sys.executable, "-m", "mind_mem.mm_cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=180,
        check=False,
    )


def _digest(workspace: str, seed: str) -> str:
    script = (
        "import hashlib,json,sys\n"
        "from mind_mem.resume_brief import resume_brief, render_resume_brief\n"
        "b = resume_brief(sys.argv[1])\n"
        "blob = json.dumps(b.to_dict()) + render_resume_brief(b)\n"
        "print(hashlib.sha256(blob.encode()).hexdigest())\n"
    )
    env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=_src())
    result = subprocess.run(
        [sys.executable, "-c", script, workspace],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=180,
        check=True,
    )
    return result.stdout.strip()


def _src() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
