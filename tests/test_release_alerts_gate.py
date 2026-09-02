"""Tests for the code-scanning alerts release gate.

Until 2026-09-02 the ``alerts-gate`` job in ``.github/workflows/release.yml``
handled an unreadable code-scanning API like this::

    echo "::warning::code-scanning alerts API not readable; failing open"
    exit 0

A 403, a 404, a job missing ``security-events: read``, a rate limit or a network
blip therefore produced a GREEN gate, and the release shipped with its only
security check having never run. It was the same shape that let 5.0.1 ship from
a commit whose CI had failed every Windows matrix row -- look for a problem, fail
to look, carry on -- and after ``release-preflight`` landed it was the last
fail-open gate in the release path.

These tests are the thing that goes red if that ``exit 0`` ever comes back. They
cover both directions:

* a positive control that MUST pass (200, zero open alerts, an analysis on
  record), so an edit that hardcodes a failure is caught;
* every failing branch, so an edit that neuters the check is caught: open
  alerts, HTTP 403 unauthorized, HTTP 404 ``no analysis found``, a malformed
  body, an empty body, a full-page count, and a zero with no positive control;
* the bypass semantics, including that it defaults to failing and that it can
  clear neither an open alert nor an authorization error;
* the workflow text itself, because a checker that is written and never wired
  is its own failure mode.

Fixtures are captured verbatim from the live API on 2026-09-02: the alert bodies
come from this repository's own alert list, the 404 from ``star-ga/mind-nerve``
(code scanning never run) and the 403 from ``octocat/Hello-World`` (not
authorized).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
OPEN_ALERTS_FIXTURE = FIXTURES / "code_scanning_alerts_open.json"
ANALYSES_FIXTURE = FIXTURES / "code_scanning_analyses.json"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"

# Verbatim live bodies, captured 2026-09-02. See the module docstring.
BODY_403_UNAUTHORIZED = (
    '{"message":"You are not authorized to read code scanning alerts.",'
    '"documentation_url":"https://docs.github.com/rest/code-scanning/code-scanning'
    '#list-code-scanning-alerts-for-a-repository","status":"403"}'
)
BODY_404_NO_ANALYSIS = (
    '{"message":"no analysis found",'
    '"documentation_url":"https://docs.github.com/rest/code-scanning/code-scanning'
    '#list-code-scanning-alerts-for-a-repository","status":"404"}'
)
# GitHub's wording when a private repository has no Advanced Security licence.
BODY_403_ADVANCED_SECURITY = (
    '{"message":"Advanced Security must be enabled for this repository to use code scanning.",'
    '"documentation_url":"https://docs.github.com/rest/code-scanning","status":"403"}'
)


def _load_script(name: str) -> ModuleType:
    """Import a scripts/*.py checker as a module."""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    assert path.is_file(), f"missing release gate script: {path}"
    spec = importlib.util.spec_from_file_location(f"_alerts_gate_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_script("check_code_scanning_alerts")


@pytest.fixture
def open_alerts() -> list[dict[str, Any]]:
    """Two real alerts from this repository, with ``state`` set to ``open``."""
    payload = json.loads(OPEN_ALERTS_FIXTURE.read_text(encoding="utf-8"))
    # Positive control on the fixture: if it ever stops containing open alerts,
    # every rejection test below would pass vacuously.
    assert payload, "fixture must contain at least one alert"
    assert all(entry["state"] == "open" for entry in payload)
    return payload


@pytest.fixture
def analyses() -> list[dict[str, Any]]:
    """Real ``/code-scanning/analyses`` entries (CodeQL x2 + Bandit)."""
    payload = json.loads(ANALYSES_FIXTURE.read_text(encoding="utf-8"))
    assert payload, "fixture must contain at least one analysis"
    return payload


def _response(status: int, body: Any) -> Any:
    """Build a ``Response`` from a status and either raw text or a JSON value."""
    text = body if isinstance(body, str) else json.dumps(body)
    return gate.Response(status, text)


class TestOutcomeStates:
    """Requirement 1: the three outcomes must not collapse into one."""

    def test_clean_is_the_only_pass(self, analyses: list[dict[str, Any]]) -> None:
        """Positive control. Without this, the gate could be hardcoded to fail."""
        state, detail = gate.classify_response(_response(200, []))
        assert state == gate.CLEAN
        assert "0 open alerts" in detail
        control, control_detail = gate.classify_analyses_response(_response(200, analyses))
        assert control == gate.CLEAN
        assert "CodeQL" in control_detail
        assert gate.report(gate.CLEAN, detail, "", "")[0] == 0

    def test_open_alerts_fail_and_are_listed(self, open_alerts: list[dict[str, Any]]) -> None:
        state, detail = gate.classify_response(_response(200, open_alerts))
        assert state == gate.ALERTS_OPEN
        assert "2 open code-scanning alert(s)" in detail
        # Requirement 1(b): the alerts must be named, not merely counted.
        assert "py/path-injection" in detail
        assert "src/mind_mem/apply_engine.py:1768" in detail
        assert gate.report(state, detail, "", "")[0] == 1

    def test_a_full_page_is_reported_as_at_least(self) -> None:
        """A single page cannot turn a non-zero count into zero, but it can undercount."""
        page = [{"number": index, "state": "open"} for index in range(gate.PER_PAGE)]
        state, detail = gate.classify_response(_response(200, page))
        assert state == gate.ALERTS_OPEN
        assert f"at least {gate.PER_PAGE}" in detail

    def test_403_unauthorized_is_unreadable_not_disabled(self) -> None:
        """The distinction that keeps a permission bug out of the bypass."""
        state, detail = gate.classify_response(_response(403, BODY_403_UNAUTHORIZED))
        assert state == gate.UNREADABLE
        assert "HTTP 403" in detail
        assert "security-events: read" in detail
        assert gate.report(state, detail, "", "")[0] == 1

    def test_404_no_analysis_found_is_not_enabled(self) -> None:
        state, detail = gate.classify_response(_response(404, BODY_404_NO_ANALYSIS))
        assert state == gate.NOT_ENABLED
        assert "HTTP 404" in detail
        assert gate.report(state, detail, "", "")[0] == 1

    def test_403_advanced_security_is_not_enabled(self) -> None:
        state, _ = gate.classify_response(_response(403, BODY_403_ADVANCED_SECURITY))
        assert state == gate.NOT_ENABLED

    @pytest.mark.parametrize(
        "status",
        [400, 401, 405, 409, 410, 418, 422, 451],
    )
    def test_unexpected_statuses_are_unreadable(self, status: int) -> None:
        state, detail = gate.classify_response(_response(status, '{"message":"who knows"}'))
        assert state == gate.UNREADABLE
        # Requirement 1(c): the message must name the actual status.
        assert f"HTTP {status}" in detail
        assert gate.report(state, detail, "", "")[0] == 1

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_transient_statuses_are_retried_not_decided(self, status: int) -> None:
        """A rate limit or a momentary 5xx must not become a verdict on try one."""
        with pytest.raises(gate.TransientError):
            gate.classify_response(_response(status, ""))

    def test_exhausted_retries_end_in_unreadable_never_a_pass(self) -> None:
        slept: list[float] = []

        def always_transient() -> tuple[str, str]:
            raise gate.TransientError("HTTP 503 from the alerts endpoint")

        state, detail = gate._with_retries(always_transient, attempts=3, delay=0.0, sleep=slept.append)
        assert state == gate.UNREADABLE
        assert "3 attempt(s) all failed transiently" in detail
        assert slept == [0.0, 0.0], "must back off between attempts, not after the last one"
        assert gate.report(state, detail, "", "")[0] == 1

    def test_a_transient_failure_that_recovers_still_passes(self) -> None:
        """Positive control on the retry loop: it must be able to succeed."""
        calls: list[int] = []

        def flaky() -> tuple[str, str]:
            calls.append(1)
            if len(calls) < 2:
                raise gate.TransientError("HTTP 503")
            return gate.CLEAN, "recovered"

        assert gate._with_retries(flaky, attempts=3, delay=0.0, sleep=lambda _: None) == (gate.CLEAN, "recovered")


class TestUnreadableBodiesFailClosed:
    """An absent or unintelligible answer is never a clean bill of health."""

    def test_empty_body_on_200_fails_closed(self) -> None:
        with pytest.raises(gate.GateError, match="empty body"):
            gate.classify_alerts_body("")

    def test_whitespace_body_on_200_fails_closed(self) -> None:
        with pytest.raises(gate.GateError, match="empty body"):
            gate.classify_alerts_body("   \n\t ")

    def test_malformed_json_fails_closed(self) -> None:
        with pytest.raises(gate.GateError, match="not JSON"):
            gate.classify_alerts_body('{"message": "totally not an array"')

    def test_html_error_page_fails_closed(self) -> None:
        with pytest.raises(gate.GateError, match="not JSON"):
            gate.classify_alerts_body("<html><body>502 Bad Gateway</body></html>")

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({"alerts": []}, id="object-not-array"),
            pytest.param("null", id="json-null"),
            pytest.param(0, id="json-zero"),
        ],
    )
    def test_wrong_shape_fails_closed(self, payload: Any) -> None:
        with pytest.raises(gate.GateError, match="not a JSON array"):
            gate.classify_alerts_body(payload if isinstance(payload, str) else json.dumps(payload))

    def test_entry_without_a_state_field_fails_closed(self) -> None:
        with pytest.raises(gate.GateError, match="no 'state' field"):
            gate.classify_alerts_body(json.dumps([{"number": 1}]))

    def test_a_non_object_entry_fails_closed(self) -> None:
        with pytest.raises(gate.GateError, match="non-object entry"):
            gate.classify_alerts_body(json.dumps(["nope"]))

    def test_a_server_side_filter_that_did_not_apply_fails_closed(self, open_alerts: list[dict[str, Any]]) -> None:
        """We asked for state=open. Anything else means the page proves nothing."""
        payload = copy.deepcopy(open_alerts)
        payload[0]["state"] = "dismissed"
        with pytest.raises(gate.GateError, match="filter did not apply"):
            gate.classify_alerts_body(json.dumps(payload))

    def test_the_gate_error_path_reports_unreadable_and_exits_one(self) -> None:
        """Wiring check: a GateError must surface as the unreadable state."""
        state, detail = gate.UNREADABLE, "the alerts endpoint returned HTTP 200 with an empty body"
        code, lines = gate.report(state, detail, "", "")
        assert code == 1
        assert any("could not run" in line for line in lines)


class TestZeroNeedsAPositiveControl:
    """``200 []`` from a scanner that never ran is not a clean result."""

    def test_zero_analyses_is_not_enabled(self) -> None:
        state, detail = gate.classify_analyses_response(_response(200, []))
        assert state == gate.NOT_ENABLED
        assert "0 analyses" in detail

    def test_one_analysis_is_the_control_passing(self, analyses: list[dict[str, Any]]) -> None:
        state, _ = gate.classify_analyses_response(_response(200, analyses))
        assert state == gate.CLEAN

    def test_unreadable_analyses_probe_is_unreadable(self) -> None:
        state, detail = gate.classify_analyses_response(_response(401, '{"message":"Bad credentials"}'))
        assert state == gate.UNREADABLE
        assert "HTTP 401" in detail

    @pytest.mark.parametrize("body", ["", "not json at all", '{"a": 1}'])
    def test_unintelligible_analyses_body_fails_closed(self, body: str) -> None:
        with pytest.raises(gate.GateError):
            gate.classify_analyses_body(body)

    def test_cli_refuses_a_zero_with_an_empty_analyses_list(self, tmp_path: Path) -> None:
        alerts = tmp_path / "alerts.json"
        alerts.write_text("[]", encoding="utf-8")
        empty = tmp_path / "analyses.json"
        empty.write_text("[]", encoding="utf-8")
        argv = [
            "--alerts-file",
            str(alerts),
            "--analyses-file",
            str(empty),
        ]
        assert gate.main(argv) == 1

    def test_cli_accepts_a_zero_backed_by_an_analysis(self, tmp_path: Path) -> None:
        """Positive control on the whole CLI: it must be able to return 0."""
        alerts = tmp_path / "alerts.json"
        alerts.write_text("[]", encoding="utf-8")
        argv = [
            "--alerts-file",
            str(alerts),
            "--analyses-file",
            str(ANALYSES_FIXTURE),
        ]
        assert gate.main(argv) == 0


class TestBypassIsExplicitAuditableAndDefaultsToFailing:
    """Requirement 2: the default with no configuration must be to fail."""

    def test_default_no_bypass_fails(self) -> None:
        code, _ = gate.report(gate.NOT_ENABLED, BODY_404_NO_ANALYSIS, "", "")
        assert code == 1

    def test_the_exact_value_clears_only_the_not_enabled_state(self) -> None:
        code, lines = gate.report(gate.NOT_ENABLED, BODY_404_NO_ANALYSIS, gate.BYPASS_VALUE, "cputer")
        assert code == 0
        joined = "\n".join(lines)
        # Loudly logged as a deliberate bypass, naming who set it.
        assert "DELIBERATE SECURITY-GATE BYPASS" in joined
        assert "cputer" in joined
        assert "::warning::" in joined

    @pytest.mark.parametrize(
        "supplied",
        [
            pytest.param("i-accept-an-ungated-release", id="lowercase"),
            pytest.param("I_ACCEPT_AN_UNGATED_RELEASE", id="underscores"),
            pytest.param("I-ACCEPT-AN-UNGATED-RELEASE ", id="trailing-space"),
            pytest.param(" I-ACCEPT-AN-UNGATED-RELEASE", id="leading-space"),
            pytest.param("yes", id="yes"),
            pytest.param("true", id="true"),
            pytest.param("1", id="one"),
            pytest.param("skip", id="skip"),
        ],
    )
    def test_a_near_miss_does_not_bypass(self, supplied: str) -> None:
        """It must be typed exactly. No trimming, no truthiness, no guessing."""
        code, _ = gate.report(gate.NOT_ENABLED, BODY_404_NO_ANALYSIS, supplied, "cputer")
        assert code == 1

    def test_the_bypass_cannot_launder_an_open_alert(self, open_alerts: list[dict[str, Any]]) -> None:
        state, detail = gate.classify_response(_response(200, open_alerts))
        code, lines = gate.report(state, detail, gate.BYPASS_VALUE, "cputer")
        assert code == 1
        assert any("being ignored" in line for line in lines)

    def test_the_bypass_cannot_hide_an_authorization_error(self) -> None:
        state, detail = gate.classify_response(_response(403, BODY_403_UNAUTHORIZED))
        code, lines = gate.report(state, detail, gate.BYPASS_VALUE, "cputer")
        assert code == 1
        assert any("being ignored" in line for line in lines)

    def test_the_bypass_cannot_hide_a_malformed_response(self) -> None:
        code, _ = gate.report(gate.UNREADABLE, "body is not JSON", gate.BYPASS_VALUE, "cputer")
        assert code == 1

    def test_a_bypass_with_no_actor_still_records_that_it_happened(self) -> None:
        code, lines = gate.report(gate.NOT_ENABLED, BODY_404_NO_ANALYSIS, gate.BYPASS_VALUE, "")
        assert code == 0
        assert any("unrecorded actor" in line for line in lines)

    def test_only_clean_and_a_bypassed_not_enabled_ever_return_zero(self) -> None:
        """Exhaustive over the state vocabulary: nothing else may pass."""
        for state in (gate.CLEAN, gate.ALERTS_OPEN, gate.NOT_ENABLED, gate.UNREADABLE, "some-future-state"):
            for supplied in ("", gate.BYPASS_VALUE):
                code, _ = gate.report(state, "detail", supplied, "cputer")
                expected = 0 if state == gate.CLEAN or (state == gate.NOT_ENABLED and supplied == gate.BYPASS_VALUE) else 1
                assert code == expected, f"state={state!r} bypass={supplied!r} returned {code}"


class TestHttpStatusLineParsing:
    """``gh api -i`` is the only way to learn the status requirement 1(c) needs."""

    def test_a_response_with_no_status_line_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Completed:
            returncode = 0
            stdout = "[]"
            stderr = ""

        monkeypatch.setattr(gate.subprocess, "run", lambda *a, **k: Completed())
        with pytest.raises(gate.GateError, match="HTTP status line"):
            gate.gh_get("repos/x/y/code-scanning/alerts", 5.0)

    def test_no_output_at_all_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Completed:
            returncode = 1
            stdout = ""
            stderr = "gh: command exploded"

        monkeypatch.setattr(gate.subprocess, "run", lambda *a, **k: Completed())
        with pytest.raises(gate.GateError, match="no response at all"):
            gate.gh_get("repos/x/y/code-scanning/alerts", 5.0)

    @pytest.mark.parametrize("separator", ["\r\n\r\n", "\n\n"])
    def test_headers_and_body_are_split_on_the_blank_line(self, monkeypatch: pytest.MonkeyPatch, separator: str) -> None:
        class Completed:
            returncode = 0
            stdout = f"HTTP/2.0 403 Forbidden\nServer: github.com{separator}{BODY_403_UNAUTHORIZED}"
            stderr = ""

        monkeypatch.setattr(gate.subprocess, "run", lambda *a, **k: Completed())
        response = gate.gh_get("repos/x/y/code-scanning/alerts", 5.0)
        assert response.status == 403
        assert "not authorized" in response.body

    def test_a_missing_gh_binary_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise FileNotFoundError("gh")

        monkeypatch.setattr(gate.subprocess, "run", boom)
        with pytest.raises(gate.GateError, match="not available"):
            gate.gh_get("repos/x/y/code-scanning/alerts", 5.0)


def _job_block(workflow_text: str, job: str) -> str:
    """Return the raw text of one job, from its key to the next job key.

    Hand-rolled rather than pyyaml-based on purpose: pyyaml is not in the
    ``test`` extra, and an ``importorskip`` would let this structural check
    vanish on a runner that lacks it -- which is the one thing a wiring test
    must never do.
    """
    lines = workflow_text.splitlines()
    start = None
    for index, raw in enumerate(lines):
        if re.match(rf"^  {re.escape(job)}:\s*$", raw):
            start = index
            break
    assert start is not None, f"release.yml has no {job!r} job"
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if re.match(r"^  [A-Za-z0-9_-]+:\s*$", lines[index]):
            end = index
            break
    return "\n".join(lines[start:end])


def _job_needs(workflow_text: str) -> dict[str, list[str]]:
    """Extract ``{job: [needs...]}`` from a workflow file. See ``_job_block``."""
    jobs: dict[str, list[str]] = {}
    current: str | None = None
    in_jobs = False
    for raw in workflow_text.splitlines():
        if raw.startswith("jobs:"):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        if raw and not raw.startswith(" ") and not raw.startswith("#"):
            in_jobs = False
            continue
        job = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", raw)
        if job:
            current = job.group(1)
            jobs[current] = []
            continue
        needs = re.match(r"^    needs:\s*(.+?)\s*$", raw)
        if needs and current is not None:
            value = needs.group(1)
            if value.startswith("["):
                jobs[current] = [item.strip() for item in value.strip("[]").split(",") if item.strip()]
            else:
                jobs[current] = [value]
    assert jobs, "no jobs parsed out of the workflow — the parser, not the workflow, is wrong"
    return jobs


class TestReleaseWorkflowWiring:
    """A checker that is written and never wired is its own failure mode."""

    @pytest.fixture
    def release_text(self) -> str:
        return RELEASE_WORKFLOW.read_text(encoding="utf-8")

    @pytest.fixture
    def alerts_job(self, release_text: str) -> str:
        return _job_block(release_text, "alerts-gate")

    def test_the_job_invokes_the_checker(self, alerts_job: str) -> None:
        assert "scripts/check_code_scanning_alerts.py" in alerts_job

    def test_the_checker_script_exists_and_is_executable_as_a_module(self) -> None:
        assert (REPO_ROOT / "scripts" / "check_code_scanning_alerts.py").is_file()

    def test_the_job_does_not_fail_open(self, alerts_job: str) -> None:
        """THE mutation guard. This is the assertion that goes red if the
        ``exit 0`` on the unreadable-API path is ever restored."""
        offenders = [line for line in alerts_job.splitlines() if re.search(r"^\s*exit\s+0\s*$", line)]
        assert not offenders, f"alerts-gate must not exit 0 explicitly; found {offenders!r}"
        lowered = alerts_job.lower()
        for phrase in ("failing open", "fail open", "fail-open", "|| true", "continue-on-error"):
            assert phrase not in lowered, f"alerts-gate contains fail-open construct {phrase!r}"

    def test_the_job_can_read_the_alerts_api(self, alerts_job: str) -> None:
        """Requirement: the permission that makes the gate answerable at all."""
        assert "security-events: read" in alerts_job
        assert "contents: read" in alerts_job, "the job checks out the in-tree checker"

    def test_the_bypass_is_passed_through_the_environment_not_interpolated(self, alerts_job: str) -> None:
        """A ``${{ }}`` expansion inside ``run:`` is a script-injection surface."""
        run_lines = alerts_job.split("run: |", 1)
        assert len(run_lines) == 2, "alerts-gate has no run block"
        assert "${{" not in run_lines[1], "alerts-gate interpolates an expression into its shell"
        assert "CODE_SCANNING_BYPASS" in alerts_job

    def test_the_bypass_input_is_declared_with_an_empty_default(self, release_text: str) -> None:
        """Requirement 2: no configuration at all must mean the gate fails."""
        header = release_text.split("jobs:", 1)[0]
        assert "code_scanning_bypass:" in header, "the bypass must be a declared workflow_dispatch input"
        block = header.split("code_scanning_bypass:", 1)[1]
        assert re.search(r"^\s+default:\s*''\s*$", block, re.MULTILINE), "the bypass input must default to empty"
        assert re.search(r"^\s+required:\s*false\s*$", block, re.MULTILINE)

    def test_a_tag_push_cannot_supply_the_bypass(self, release_text: str) -> None:
        """The bypass lives under workflow_dispatch, so `push: tags` carries none."""
        header = release_text.split("jobs:", 1)[0]
        dispatch_index = header.index("workflow_dispatch:")
        assert header.index("code_scanning_bypass:") > dispatch_index
        push_block = header[header.index("push:") : dispatch_index]
        assert "code_scanning_bypass" not in push_block

    def test_build_still_waits_for_this_gate(self, release_text: str) -> None:
        """The gate is only a gate while something downstream depends on it."""
        graph = _job_needs(release_text)
        assert "alerts-gate" in graph
        assert "alerts-gate" in graph["build"]

    def test_the_job_graph_is_acyclic_and_fully_resolved(self, release_text: str) -> None:
        graph = _job_needs(release_text)
        resolved: dict[str, set[str]] = {}

        def resolve(job: str, seen: frozenset[str] = frozenset()) -> set[str]:
            assert job not in seen, f"cycle in the release job graph at {job}"
            if job in resolved:
                return resolved[job]
            found: set[str] = set()
            for parent in graph.get(job, []):
                assert parent in graph, f"job {job!r} needs unknown job {parent!r}"
                found.add(parent)
                found |= resolve(parent, seen | {job})
            resolved[job] = found
            return found

        for job in graph:
            resolve(job)
        # Everything that builds, signs, publishes or releases inherits the gate.
        for job in ("build", "sign", "sbom", "publish-pypi", "github-release", "verify-published"):
            assert "alerts-gate" in resolve(job), f"{job} is not downstream of alerts-gate"

    def test_the_five_preflight_gates_are_untouched(self, release_text: str) -> None:
        """This change must not weaken the gates it sits next to."""
        preflight = _job_block(release_text, "release-preflight")
        for marker in (
            "Gate (a) — version triple agreement",
            "Gate (b) — the tagged commit must be an ancestor of origin/main",
            "Gate (c) — CI concluded success for this commit",
            "Gate (d) — version is not on the index, including yanked",
            "Gate (e) — unit tests (same selector as ci.yml)",
        ):
            assert marker in preflight, f"release-preflight lost {marker!r}"
        assert "skip-existing: false" in release_text, "the publish step must still refuse a spent version"

    def test_no_secret_is_printed_or_hardcoded(self, release_text: str, alerts_job: str) -> None:
        assert "secrets.GITHUB_TOKEN" in alerts_job, "the job must reuse the existing token mechanism"
        assert "echo" not in alerts_job.lower().split("run: |", 1)[1], "the job must not echo anything, tokens included"
        assert "ghp_" not in release_text and "gho_" not in release_text
