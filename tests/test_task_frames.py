# Copyright 2026 STARGA, Inc.
"""Tests for TASK-FRAME + DEAD-END blocks — multi-session continuity and
negative action-space memory.

Acceptance gate, point by point:

* ``TestTaskFrameBlock`` — ``[TF-...]`` parses into a frame carrying goal,
  plan steps, what was tried, what is believed and what remains.
* ``TestResumeBrief`` — ``resume_brief`` answers the four session-N+1
  questions for the active frame, with citations.
* ``TestDeadEndBlock`` — ``[DE-...]`` parses into a dead end carrying the
  approach, why it failed, its outcome and its evidence handle.
* ``TestOverlapMatch`` — the frame↔dead-end overlap is declarative: the
  guardrail trigger grammar, AND across declared dimensions, OR within one,
  fail-closed on an empty trigger.
* ``TestPurity`` — the matcher is a pure function of (frame, dead ends): no
  clock, no model, no similarity score, no mutation.
* ``TestCrossProcessDeterminism`` — the same corpus renders byte-identical
  briefs in two separate interpreters under different hash seeds.
* ``TestWarnNeverBlock`` — a dead end is evidence: it warns, it never
  refuses, never filters and never changes an exit code.
* ``TestProvenanceRestriction`` — an external-ingest block can mint neither
  a frame nor a dead end, whatever its content declares.
* ``TestWiring`` — the warning reaches the surfaces an agent actually calls:
  the MCP tools and the ``mm`` CLI, exercised through the real entry points.
* ``TestNoWritePath`` — nothing in these modules writes to the store.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from mind_mem.dead_ends import (
    DEAD_END_ID_PREFIX,
    OUTCOME_RANK,
    ApproachSurface,
    DeadEnd,
    load_dead_ends,
    match_dead_ends,
    match_surface,
    parse_dead_end_block,
)
from mind_mem.guardrails import (
    GuardrailContext,
    GuardrailSpecError,
    GuardrailTrigger,
)
from mind_mem.init_workspace import init
from mind_mem.resume_brief import (
    ResumeBrief,
    render_resume_brief,
    resume_brief,
)
from mind_mem.task_frames import (
    STEP_STATUSES,
    TASK_FRAME_ID_PREFIX,
    FramePolicy,
    FrameSpecError,
    PlanStep,
    active_frame,
    load_task_frames,
    parse_task_frame_block,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

FRAME = """[TF-20260829-001]
Type: TaskFrame
Goal: Close the last two AGI3 floors without a net regression.
Status: active
Steps:
- done: rederive the floor count from the live pass
- doing: pin the SAT-BMC encoding for L5
- todo: package the bundle for the cloud run
- blocked: L8 needs 336M clauses and more RAM than the box has
Tried:
- explicit BFS over the level graph
- additive per-object assignment bound
Believed: flow bounds survive multi-carrier levels, additive bounds do not
Remaining: fund the cloud run
Blockers: cloud budget
References: docs/floors.json, research/runpod/bundle.md
ApproachTools: Bash
ApproachCommands: rederive_floor_count.py, sat-bmc, explicit-search
ApproachIntents: prove_floor
ApproachPaths: tools/**/*.py

"""

SECOND_FRAME = """[TF-20260829-002]
Type: TaskFrame
Goal: Ship the batch review surface.
Status: active
Steps:
- todo: list pending proposals with a diff
ApproachTools: Edit

"""

ARCHIVED_FRAME = """[TF-20260101-001]
Type: TaskFrame
Goal: An old task nobody is working on.
Status: archived
ApproachTools: Bash

"""

ASSIGNMENT_DEAD_END = """[DE-20260826-001]
Type: DeadEnd
Approach: Additive per-object assignment lower bound for AGI3 floors.
WhyFailed: A co-carrier helper divides the matching by K, so the bound
    lands under the already-proven floor and can never tighten it.
Outcome: refuted
Evidence: docs/AGI3_FLOOR_METHOD_ARSENAL.md#assignment
TriggerTools: Bash
TriggerIntents: prove_floor
Status: active

"""

BFS_DEAD_END = """[DE-20260826-002]
Type: DeadEnd
Approach: Explicit breadth-first search over the level state graph.
WhyFailed: The frontier does not fit in memory at depth 7.
Outcome: blocked
Evidence: RESULTS/l5-bfs.log
TriggerCommands: bfs, explicit-search
Status: active

"""

# Triggers on a path no frame below declares.
UNRELATED_DEAD_END = """[DE-20260826-003]
Type: DeadEnd
Approach: Rewriting the payment ledger in a dynamic language.
WhyFailed: The decimal semantics did not survive the port.
Outcome: regressed
Evidence: RESULTS/ledger.md
TriggerPaths: billing/**/*.rb
Status: active

"""

ARCHIVED_DEAD_END = """[DE-20260101-009]
Type: DeadEnd
Approach: An approach that was later shown to work after all.
WhyFailed: Superseded by a fixed toolchain.
Outcome: inconclusive
Evidence: RESULTS/old.md
TriggerTools: Bash
Status: superseded

"""


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _frame_ws(tmp_path, name: str, frames: str = FRAME, dead_ends: str = "") -> str:
    """Init a workspace carrying the given frame / dead-end corpus."""
    workspace = str(tmp_path / name)
    init(workspace)
    _write(os.path.join(workspace, "frames", "FRAMES.md"), frames)
    if dead_ends:
        _write(os.path.join(workspace, "frames", "DEAD-ENDS.md"), dead_ends)
    return workspace


@pytest.fixture
def ws(tmp_path) -> str:
    return _frame_ws(tmp_path, "plain", FRAME, ASSIGNMENT_DEAD_END + BFS_DEAD_END + UNRELATED_DEAD_END)


@pytest.fixture
def bare_ws(tmp_path) -> str:
    """A workspace with a frame and no dead ends at all."""
    return _frame_ws(tmp_path, "bare", FRAME)


def _frame_block(**overrides: object) -> dict:
    block = {
        "_id": "TF-20260829-777",
        "Type": "TaskFrame",
        "Goal": "Do the thing.",
        "Status": "active",
        "ApproachTools": "Bash",
    }
    block.update(overrides)
    return block


def _dead_end_block(**overrides: object) -> dict:
    block = {
        "_id": "DE-20260829-777",
        "Type": "DeadEnd",
        "Approach": "The approach that failed.",
        "WhyFailed": "It ran out of memory.",
        "Outcome": "blocked",
        "Evidence": "RESULTS/x.log",
        "TriggerTools": "Bash",
        "Status": "active",
    }
    block.update(overrides)
    return block


# ---------------------------------------------------------------------------
# 1. [TASK-FRAME] block kind
# ---------------------------------------------------------------------------


class TestTaskFrameBlock:
    def test_prefix_is_the_identity(self) -> None:
        assert TASK_FRAME_ID_PREFIX == "TF-"
        with pytest.raises(FrameSpecError):
            parse_task_frame_block(_frame_block(_id="D-20260829-001"))

    def test_parses_goal_steps_tried_believed_remaining(self, ws: str) -> None:
        frame = load_task_frames(ws)[0]
        assert frame.block_id == "TF-20260829-001"
        assert frame.goal.startswith("Close the last two AGI3 floors")
        assert [s.status for s in frame.steps] == ["done", "doing", "todo", "blocked"]
        assert frame.steps[1].text == "pin the SAT-BMC encoding for L5"
        assert frame.tried == (
            "explicit BFS over the level graph",
            "additive per-object assignment bound",
        )
        assert frame.believed == ("flow bounds survive multi-carrier levels, additive bounds do not",)
        assert frame.remaining == ("fund the cloud run",)
        assert frame.blockers == ("cloud budget",)
        assert frame.citations == ("docs/floors.json", "research/runpod/bundle.md")

    def test_step_status_vocabulary_is_closed(self) -> None:
        assert STEP_STATUSES == ("todo", "doing", "done", "blocked")
        with pytest.raises(FrameSpecError):
            parse_task_frame_block(_frame_block(Steps=["nonsense: do a thing"]))

    def test_bare_step_defaults_to_todo(self) -> None:
        frame = parse_task_frame_block(_frame_block(Steps=["write the parser"]))
        assert frame.steps == (PlanStep(status="todo", text="write the parser"),)

    def test_frame_without_a_goal_is_refused(self) -> None:
        block = _frame_block()
        block.pop("Goal")
        with pytest.raises(FrameSpecError):
            parse_task_frame_block(block)

    def test_remaining_falls_back_to_open_steps(self) -> None:
        """A frame that ticks steps need not also maintain Remaining."""
        frame = parse_task_frame_block(_frame_block(Steps=["done: a", "doing: b", "todo: c"]))
        assert frame.remaining == ("b", "c")

    def test_only_live_frames_load(self, tmp_path) -> None:
        workspace = _frame_ws(tmp_path, "archived", FRAME + ARCHIVED_FRAME)
        assert [f.block_id for f in load_task_frames(workspace)] == ["TF-20260829-001"]

    def test_load_is_deterministically_ordered(self, tmp_path) -> None:
        workspace = _frame_ws(tmp_path, "two", SECOND_FRAME + FRAME)
        assert [f.block_id for f in load_task_frames(workspace)] == [
            "TF-20260829-001",
            "TF-20260829-002",
        ]

    def test_active_frame_picks_the_newest_id(self, tmp_path) -> None:
        workspace = _frame_ws(tmp_path, "newest", FRAME + SECOND_FRAME)
        assert active_frame(load_task_frames(workspace)).block_id == "TF-20260829-002"

    def test_active_frame_of_nothing_is_none(self) -> None:
        assert active_frame(()) is None

    def test_recall_knows_the_block_type(self) -> None:
        from mind_mem._recall_detection import get_block_type

        assert get_block_type("TF-20260829-001") == "task_frame"
        assert get_block_type("DE-20260829-001") == "dead_end"


# ---------------------------------------------------------------------------
# 2. resume_brief()
# ---------------------------------------------------------------------------


class TestResumeBrief:
    def test_brief_answers_the_four_questions(self, ws: str) -> None:
        brief = resume_brief(ws)
        assert isinstance(brief, ResumeBrief)
        assert brief.frame_id == "TF-20260829-001"
        assert brief.goal.startswith("Close the last two AGI3 floors")
        assert "explicit BFS over the level graph" in brief.tried
        assert brief.believed
        assert "fund the cloud run" in brief.remaining
        assert brief.citations

    def test_brief_selects_an_explicit_frame(self, tmp_path) -> None:
        workspace = _frame_ws(tmp_path, "explicit", FRAME + SECOND_FRAME)
        assert resume_brief(workspace, "TF-20260829-001").frame_id == "TF-20260829-001"

    def test_unknown_frame_id_is_refused(self, ws: str) -> None:
        with pytest.raises(FrameSpecError):
            resume_brief(ws, "TF-19990101-001")

    def test_empty_workspace_yields_an_empty_brief(self, tmp_path) -> None:
        workspace = str(tmp_path / "empty")
        init(workspace)
        brief = resume_brief(workspace)
        assert brief.frame_id == ""
        assert brief.goal == ""
        assert brief.dead_ends == ()

    def test_brief_carries_dead_end_warnings(self, ws: str) -> None:
        brief = resume_brief(ws)
        assert [w.dead_end.block_id for w in brief.dead_ends] == [
            "DE-20260826-001",
            "DE-20260826-002",
        ]

    def test_brief_is_immutable(self, ws: str) -> None:
        brief = resume_brief(ws)
        with pytest.raises(Exception):
            brief.goal = "mutated"  # type: ignore[misc]

    def test_to_dict_round_trips_through_json(self, ws: str) -> None:
        payload = json.dumps(resume_brief(ws).to_dict(), sort_keys=True)
        assert "DE-20260826-001" in payload
        assert json.loads(payload)["frame_id"] == "TF-20260829-001"

    def test_render_names_the_dead_ends(self, ws: str) -> None:
        text = render_resume_brief(resume_brief(ws))
        assert "TF-20260829-001" in text
        assert "DEAD END" in text.upper()
        assert "DE-20260826-001" in text
        assert "co-carrier helper" in text


# ---------------------------------------------------------------------------
# 3. [DEAD-END] block kind
# ---------------------------------------------------------------------------


class TestDeadEndBlock:
    def test_prefix_is_the_identity(self) -> None:
        assert DEAD_END_ID_PREFIX == "DE-"
        with pytest.raises(GuardrailSpecError):
            parse_dead_end_block(_dead_end_block(_id="T-20260829-001"))

    def test_parses_approach_reason_outcome_evidence(self, ws: str) -> None:
        dead_end = load_dead_ends(ws)[0]
        assert dead_end.block_id == "DE-20260826-001"
        assert dead_end.approach.startswith("Additive per-object assignment")
        assert "co-carrier helper" in dead_end.why_failed
        assert dead_end.outcome == "refuted"
        assert dead_end.evidence == ("docs/AGI3_FLOOR_METHOD_ARSENAL.md#assignment",)

    def test_outcome_vocabulary_is_closed(self) -> None:
        assert set(OUTCOME_RANK) == {"refuted", "regressed", "blocked", "inconclusive"}
        with pytest.raises(GuardrailSpecError):
            parse_dead_end_block(_dead_end_block(Outcome="probably-bad"))

    def test_dead_end_without_a_reason_is_refused(self) -> None:
        block = _dead_end_block()
        block.pop("WhyFailed")
        with pytest.raises(GuardrailSpecError):
            parse_dead_end_block(block)

    def test_dead_end_without_a_trigger_is_refused(self) -> None:
        """Fail-closed, exactly like a guardrail: a dead end that can never
        match is noise, not memory."""
        block = _dead_end_block()
        block.pop("TriggerTools")
        with pytest.raises(GuardrailSpecError):
            parse_dead_end_block(block)

    def test_only_live_dead_ends_load(self, tmp_path) -> None:
        workspace = _frame_ws(tmp_path, "de_archived", FRAME, ASSIGNMENT_DEAD_END + ARCHIVED_DEAD_END)
        assert [d.block_id for d in load_dead_ends(workspace)] == ["DE-20260826-001"]

    def test_sort_key_is_outcome_then_id(self) -> None:
        refuted = parse_dead_end_block(_dead_end_block(_id="DE-20260829-900", Outcome="refuted"))
        blocked = parse_dead_end_block(_dead_end_block(_id="DE-20260829-001", Outcome="blocked"))
        assert sorted([blocked, refuted], key=DeadEnd.sort_key)[0] is refuted

    def test_trigger_fields_are_the_guardrail_grammar(self) -> None:
        """One trigger language: a dead end parses into the same container
        a guardrail does, so the two can never drift apart."""
        dead_end = parse_dead_end_block(_dead_end_block(TriggerCommands="pytest -k dead_end", TriggerPaths="src/**/*.py"))
        assert isinstance(dead_end.trigger, GuardrailTrigger)
        assert dead_end.trigger.paths == ("src/**/*.py",)


# ---------------------------------------------------------------------------
# 4. Deterministic overlap
# ---------------------------------------------------------------------------


class TestOverlapMatch:
    def test_matching_dead_ends_warn(self, ws: str) -> None:
        frame = load_task_frames(ws)[0]
        warnings = match_dead_ends(frame, load_dead_ends(ws))
        assert [w.dead_end.block_id for w in warnings] == ["DE-20260826-001", "DE-20260826-002"]

    def test_non_overlapping_dead_end_never_warns(self, ws: str) -> None:
        frame = load_task_frames(ws)[0]
        ids = {w.dead_end.block_id for w in match_dead_ends(frame, load_dead_ends(ws))}
        assert "DE-20260826-003" not in ids

    def test_matched_dimensions_are_reported(self, ws: str) -> None:
        frame = load_task_frames(ws)[0]
        by_id = {w.dead_end.block_id: w for w in match_dead_ends(frame, load_dead_ends(ws))}
        assert by_id["DE-20260826-001"].matched == ("tool", "intent")
        assert by_id["DE-20260826-002"].matched == ("command",)

    def test_and_across_dimensions(self) -> None:
        """Every declared dimension must overlap."""
        dead_end = parse_dead_end_block(_dead_end_block(TriggerTools="Bash", TriggerIntents="deploy"))
        frame = parse_task_frame_block(_frame_block(ApproachTools="Bash", ApproachIntents="refactor"))
        assert match_dead_ends(frame, [dead_end]) == ()

    def test_or_within_a_dimension(self) -> None:
        dead_end = parse_dead_end_block(_dead_end_block(TriggerTools="Edit, Bash"))
        frame = parse_task_frame_block(_frame_block(ApproachTools="Bash"))
        assert len(match_dead_ends(frame, [dead_end])) == 1

    def test_a_frame_declaring_no_approach_matches_nothing(self) -> None:
        block = _frame_block()
        block.pop("ApproachTools")
        frame = parse_task_frame_block(block)
        assert frame.approach.is_empty()
        assert match_dead_ends(frame, [parse_dead_end_block(_dead_end_block())]) == ()

    def test_glob_grammar_is_the_guardrail_one(self) -> None:
        dead_end = parse_dead_end_block(_dead_end_block(TriggerPaths="src/**/*.py"))
        block = _frame_block(ApproachPaths="src/mind_mem/dead_ends.py")
        assert len(match_dead_ends(parse_task_frame_block(block), [dead_end])) == 1

    def test_surface_matching_accepts_a_guardrail_context(self, ws: str) -> None:
        """The single-action check and the frame check share one matcher."""
        context = GuardrailContext(tool="Bash", intent="prove_floor")
        warnings = match_surface(ApproachSurface.from_context(context), load_dead_ends(ws))
        assert [w.dead_end.block_id for w in warnings] == ["DE-20260826-001"]

    def test_ordering_is_outcome_then_id(self, ws: str) -> None:
        frame = load_task_frames(ws)[0]
        warnings = match_dead_ends(frame, load_dead_ends(ws))
        assert [w.dead_end.outcome for w in warnings] == ["refuted", "blocked"]


class TestPurity:
    def test_no_clock_no_model_no_score(self) -> None:
        """The matcher module imports nothing that could make it vary."""
        import mind_mem.dead_ends as module

        source = open(module.__file__, encoding="utf-8").read()
        for banned in ("import time", "datetime", "random", "numpy", "embed", "cosine"):
            assert banned not in source, f"matcher must not reference {banned!r}"

    def test_warning_has_no_score_field(self, ws: str) -> None:
        frame = load_task_frames(ws)[0]
        warning = match_dead_ends(frame, load_dead_ends(ws))[0]
        assert not hasattr(warning, "score")
        assert not hasattr(warning, "similarity")

    def test_repeated_calls_are_identical(self, ws: str) -> None:
        frame = load_task_frames(ws)[0]
        dead_ends = load_dead_ends(ws)
        first = match_dead_ends(frame, dead_ends)
        second = match_dead_ends(frame, dead_ends)
        assert first == second

    def test_input_order_does_not_change_the_output(self, ws: str) -> None:
        frame = load_task_frames(ws)[0]
        dead_ends = list(load_dead_ends(ws))
        forward = match_dead_ends(frame, dead_ends)
        backward = match_dead_ends(frame, list(reversed(dead_ends)))
        assert [w.dead_end.block_id for w in forward] == [w.dead_end.block_id for w in backward]

    def test_matcher_does_not_mutate_its_inputs(self, ws: str) -> None:
        frame = load_task_frames(ws)[0]
        dead_ends = load_dead_ends(ws)
        before = (frame, dead_ends)
        match_dead_ends(frame, dead_ends)
        assert before == (frame, dead_ends)


class TestCrossProcessDeterminism:
    """The determinism wedge, proven the only honest way: two interpreters."""

    SNIPPET = (
        "import json,sys;"
        "from mind_mem.resume_brief import resume_brief;"
        "sys.stdout.write(json.dumps(resume_brief(sys.argv[1]).to_dict(), sort_keys=True))"
    )

    def _run(self, workspace: str, seed: str) -> bytes:
        env = dict(os.environ, PYTHONHASHSEED=seed)
        proc = subprocess.run(
            [sys.executable, "-c", self.SNIPPET, workspace],
            capture_output=True,
            check=True,
            env=env,
        )
        return proc.stdout

    def test_two_processes_produce_identical_bytes(self, ws: str) -> None:
        first = self._run(ws, "0")
        second = self._run(ws, "12345")
        assert first == second
        assert b"DE-20260826-001" in first


# ---------------------------------------------------------------------------
# A dead end is evidence, never a prohibition
# ---------------------------------------------------------------------------


class TestWarnNeverBlock:
    def test_matching_dead_end_does_not_raise(self, ws: str) -> None:
        assert resume_brief(ws).dead_ends

    def test_dead_end_never_filters_the_frame(self, ws: str) -> None:
        """The plan survives intact next to its warnings."""
        brief = resume_brief(ws)
        assert brief.goal and brief.remaining and brief.steps

    def test_cli_exit_code_is_zero_with_warnings(self, ws: str) -> None:
        proc = _run_cli(["resume"], ws)
        assert proc.returncode == 0
        assert "DE-20260826-001" in proc.stdout

    def test_docstring_states_the_contract(self) -> None:
        from mind_mem.dead_ends import match_dead_ends as fn

        doc = (fn.__doc__ or "").lower()
        assert "warn" in doc and "never" in doc
        assert "operator" in doc


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenanceRestriction:
    def test_imported_frame_is_refused(self) -> None:
        with pytest.raises(FrameSpecError):
            parse_task_frame_block(_frame_block(ActorRole="importer"))

    def test_imported_dead_end_is_refused(self) -> None:
        with pytest.raises(GuardrailSpecError):
            parse_dead_end_block(_dead_end_block(Source="imported:slack"))

    def test_provenance_is_checked_before_the_fields(self) -> None:
        block = _dead_end_block(ActorRole="importer")
        block.pop("WhyFailed")
        block.pop("TriggerTools")
        with pytest.raises(GuardrailSpecError, match="provenance"):
            parse_dead_end_block(block)

    def test_one_poisoned_block_does_not_disable_the_file(self, tmp_path) -> None:
        poisoned = ASSIGNMENT_DEAD_END.replace("Type: DeadEnd", "Type: DeadEnd\nActorRole: importer")
        workspace = _frame_ws(tmp_path, "poisoned", FRAME, poisoned + BFS_DEAD_END)
        assert [d.block_id for d in load_dead_ends(workspace)] == ["DE-20260826-002"]

    def test_imported_dead_end_never_reaches_a_brief(self, tmp_path) -> None:
        poisoned = ASSIGNMENT_DEAD_END.replace("Type: DeadEnd", "Type: DeadEnd\nActorRole: importer")
        workspace = _frame_ws(tmp_path, "poisoned_brief", FRAME, poisoned)
        assert resume_brief(workspace).dead_ends == ()


# ---------------------------------------------------------------------------
# 5. Wiring — the real paths an agent calls
# ---------------------------------------------------------------------------


def _run_cli(argv: list[str], workspace: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, MIND_MEM_WORKSPACE=workspace)
    return subprocess.run(
        [sys.executable, "-m", "mind_mem.mm_cli", *argv], capture_output=True, text=True, env=env, encoding="utf-8", errors="replace"
    )


class TestWiring:
    """No new orphan modules: every piece is reachable from a real surface."""

    def test_modules_have_production_importers(self) -> None:
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "mind_mem")
        for module in ("frame_fields", "task_frames", "dead_ends", "resume_brief"):
            importers = [
                path for path, _ in _walk_sources(root) if os.path.basename(path) != f"{module}.py" and f"{module} import" in _read(path)
            ]
            assert importers, f"{module} has no production importer"

    def test_mcp_server_registers_the_tools(self) -> None:
        from mind_mem.mcp.tools import frames as tools_frames

        registered: list[str] = []

        class _Recorder:
            def tool(self, fn):
                registered.append(fn.__name__)
                return fn

        tools_frames.register(_Recorder())
        assert registered == ["resume_brief", "check_dead_ends"]

    def test_tools_are_on_the_acl(self) -> None:
        from mind_mem.mcp.infra.acl import USER_TOOLS

        assert {"resume_brief", "check_dead_ends"} <= USER_TOOLS

    def test_mcp_resume_brief_surfaces_the_warning(self, ws: str, monkeypatch) -> None:
        """The real MCP entry point, not the function under it."""
        from mind_mem.mcp.tools import frames as tools_frames

        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        monkeypatch.setenv("MIND_MEM_ACL_DISABLED", "true")
        payload = json.loads(tools_frames.resume_brief())
        assert payload["frame_id"] == "TF-20260829-001"
        assert payload["dead_end_count"] == 2
        assert payload["dead_ends"][0]["block_id"] == "DE-20260826-001"
        assert "co-carrier helper" in payload["dead_ends"][0]["why_failed"]

    def test_mcp_check_dead_ends_surfaces_the_warning(self, ws: str, monkeypatch) -> None:
        from mind_mem.mcp.tools.frames import check_dead_ends

        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        monkeypatch.setenv("MIND_MEM_ACL_DISABLED", "true")
        payload = json.loads(check_dead_ends(tool="Bash", intent="prove_floor"))
        assert payload["count"] == 1
        assert payload["dead_ends"][0]["block_id"] == "DE-20260826-001"

    def test_check_dead_ends_is_empty_without_a_context(self, ws: str, monkeypatch) -> None:
        from mind_mem.mcp.tools.frames import check_dead_ends

        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        monkeypatch.setenv("MIND_MEM_ACL_DISABLED", "true")
        assert json.loads(check_dead_ends())["count"] == 0

    def test_cli_resume_renders_the_brief(self, ws: str) -> None:
        proc = _run_cli(["resume"], ws)
        assert proc.returncode == 0, proc.stderr
        assert "TF-20260829-001" in proc.stdout
        assert "DE-20260826-001" in proc.stdout
        assert "co-carrier helper" in proc.stdout

    def test_cli_resume_json(self, ws: str) -> None:
        proc = _run_cli(["resume", "--json"], ws)
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["frame_id"] == "TF-20260829-001"
        assert len(payload["dead_ends"]) == 2

    def test_cli_dead_ends_lists_the_registry(self, ws: str) -> None:
        proc = _run_cli(["dead-ends"], ws)
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert [d["block_id"] for d in payload["dead_ends"]] == [
            "DE-20260826-001",
            "DE-20260826-002",
            "DE-20260826-003",
        ]

    def test_cli_dead_ends_filters_by_context(self, ws: str) -> None:
        proc = _run_cli(["dead-ends", "--tool", "Bash", "--intent", "prove_floor"], ws)
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert [d["block_id"] for d in payload["dead_ends"]] == ["DE-20260826-001"]

    def test_zero_regression_without_frames(self, tmp_path) -> None:
        """A workspace with no frames behaves exactly as before."""
        workspace = str(tmp_path / "none")
        init(workspace)
        proc = _run_cli(["resume", "--json"], workspace)
        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout)["frame_id"] == ""


def _walk_sources(root: str):
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                yield path, name


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------


class TestNoWritePath:
    def test_no_module_writes(self) -> None:
        """These modules read the corpus; authoring stays on the governed
        propose_update -> approve_apply route."""
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "mind_mem")
        for module in ("frame_fields.py", "task_frames.py", "dead_ends.py", "resume_brief.py"):
            source = _read(os.path.join(root, module))
            for banned in ("write_block", "open(", "_atomic_write", "os.remove"):
                assert banned not in source, f"{module} must not write: found {banned!r}"

    def test_mcp_tools_are_read_only(self) -> None:
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "mind_mem")
        source = _read(os.path.join(root, "mcp", "tools", "frames.py"))
        for banned in ("write_block", "_atomic_write", "apply_engine", "os.remove"):
            assert banned not in source, f"frames.py must not write: found {banned!r}"

    def test_policy_bounds_are_enforced(self) -> None:
        policy = FramePolicy.from_config({"recall": {"frames": {"max_warnings": 9999}}})
        assert policy.max_warnings <= FramePolicy.HARD_CAP

    def test_disabled_policy_yields_no_warnings(self, ws: str) -> None:
        brief = resume_brief(ws, policy=FramePolicy(enabled=False))
        assert brief.dead_ends == ()
        assert brief.frame_id == "TF-20260829-001"
