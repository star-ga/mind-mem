#!/usr/bin/env python3
"""``lint`` is actually reachable — from ``mm lint`` and from the MCP surface.

``mind_mem.lint`` / ``mind_mem.lint_autofix`` shipped complete, tested and
unreachable: no CLI command, no MCP tool, no caller anywhere in the product.
A corpus-repair path nothing can invoke repairs nothing.

These tests pin the call sites that close that, and pin that the gate in
front of them is real:

* the MCP pair — ``lint`` (USER) → ``lint_autofix`` (ADMIN) → ``approve_apply``
  — actually flips a duplicate decision to ``superseded``;
* ``LF-`` finding ids are stable across separate processes, which is what
  makes handing one back to ``lint_autofix`` meaningful at all;
* both tools are ACL-classified, and the split is enforced, not decorative:
  a user-scope caller reads findings and is refused the staging half;
* ``mm lint`` reports, and ``mm lint --fix`` stages exactly one proposal
  and touches nothing else;
* with ``v4.lint`` unset — the default — every one of those surfaces is
  inert, writes nothing, and the *probe that decides that* emits nothing.

Each of the first four fails if the wiring is deleted; the last group fails
if the gate is ever inverted or the probe made observable.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from mind_mem.init_workspace import init
from mind_mem.lint import RULE_DUPLICATE_BLOCK, RULE_MISSING_METADATA
from mind_mem.lint_autofix import PROPOSAL_FILE
from mind_mem.mm_cli import config_set

# Two active decisions asserting the same statement: D-...-004 is the twin,
# D-...-001 wins on lowest id. That pair is what the end-to-end chain repairs.
DECISIONS = """
[D-20260101-001]
Date: 2026-01-01
Status: active
Scope: global
Statement: Default block store backend is SQLite.
Rationale: Zero configuration for a new workspace.
Supersedes: none
Tags: storage
Sources:
- decisions/DECISIONS.md

[D-20260104-004]
Date: 2026-01-04
Status: active
Scope: global
Statement: Default block store backend is SQLite.
Rationale: Restated during onboarding.
Supersedes: none
Tags: storage
Sources:
- decisions/DECISIONS.md
"""

TWIN = "D-20260104-004"
WINNER = "D-20260101-001"

# A task block missing ``Context`` — one of the five required fields the
# lint's own drifted COPY of the validator's list did not carry.
TASKS = """
[T-20260105-001]
Date: 2026-01-05
Title: Wire the lint
Status: todo
Priority: P1
Project: mind-mem
Due: 2026-02-01
Owner: user
Context:
Next: ship it
Dependencies: none
Sources:
- tasks/TASKS.md
History:
- created 2026-01-05
"""

# ``check_preconditions`` shells out to the validator, which reports issues on
# a freshly scaffolded workspace for reasons unrelated to this repair. Every
# other apply-pipeline gate runs for real.
_PRECONDITIONS_PASS = patch(
    "mind_mem.apply_engine.check_preconditions",
    return_value=(True, ["validate: PASS (TOTAL 0 issues)"]),
)


def _make_ws(tmp_path: Path, *, enable_lint: bool, mode: str = "enforce", tasks: str = "") -> str:
    ws = str(tmp_path / "ws")
    init(ws)

    # ``init`` arms the gate, so the bound config is changed through
    # ``mm config set`` (write + re-attest in one step); a hand edit is
    # drift and ``enforce`` would refuse the applies below. The malformed
    # -config tests further down write the file raw on purpose — that is
    # the tamper they exist to check, not a configuration change.
    config_path = os.path.join(ws, "mind-mem.json")
    if enable_lint:
        config_set(config_path, "v4.lint", {"enabled": True})
    config_set(config_path, "governance_mode", mode)

    state_path = os.path.join(ws, "memory", "intel-state.json")
    with open(state_path, encoding="utf-8") as handle:
        state = json.load(handle)
    state["governance_mode"] = mode
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle)

    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as handle:
        handle.write(DECISIONS)
    if tasks:
        with open(os.path.join(ws, "tasks", "TASKS.md"), "a", encoding="utf-8") as handle:
            handle.write(tasks)
    return ws


def _digests(ws: str) -> dict[str, str]:
    """sha256 of every file in the workspace, by relative path."""
    out: dict[str, str] = {}
    for root, _dirs, files in os.walk(ws):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, ws).replace(os.sep, "/")
            with open(path, "rb") as handle:
                out[rel] = hashlib.sha256(handle.read()).hexdigest()
    return out


def _status_of(ws: str, block_id: str) -> str:
    from mind_mem.block_parser import parse_file

    blocks = {b["_id"]: b for b in parse_file(os.path.join(ws, "decisions", "DECISIONS.md"))}
    return str(blocks[block_id].get("Status", ""))


def _run_cli(ws: str, *argv: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "MIND_MEM_WORKSPACE": ws, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, "-m", "mind_mem.mm_cli", *argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        env=env,
    )


@pytest.fixture(autouse=True)
def _isolate_mcp_env(monkeypatch, tmp_path):
    """Deterministic MCP inputs: explicit scope, no ACL override, fresh budget.

    The per-client rate limiter is process-global and shared with every other
    test in the session, so a full run can otherwise surface a rate-limit
    envelope here instead of the verdict under test.
    """
    from mind_mem.mcp.infra import rate_limit as _rate_limit

    monkeypatch.setenv("MIND_MEM_SCOPE", "user")
    monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)
    monkeypatch.delenv("MIND_MEM_CONFIG", raising=False)
    with _rate_limit._rate_limiters_lock:
        _rate_limit._rate_limiters.clear()
    yield
    with _rate_limit._rate_limiters_lock:
        _rate_limit._rate_limiters.clear()


class TestMcpChainRepairsADuplicate:
    """lint → lint_autofix → approve_apply, through the registered tools."""

    def test_duplicate_decision_ends_up_superseded(self, tmp_path, monkeypatch) -> None:
        from mind_mem.mcp.tools import governance
        from mind_mem.mcp.tools import lint as lint_tools

        ws = _make_ws(tmp_path, enable_lint=True)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        monkeypatch.setenv("MIND_MEM_SCOPE", "admin")

        # 1. lint reports the twin, with a stable content-addressed id.
        report = json.loads(lint_tools.lint(rule=RULE_DUPLICATE_BLOCK))
        duplicates = [f for f in report["findings"] if f["rule"] == RULE_DUPLICATE_BLOCK]
        assert [f["block_id"] for f in duplicates] == [TWIN], report
        finding_id = duplicates[0]["finding_id"]
        assert finding_id.startswith("LF-") and len(finding_id) == 11
        assert _status_of(ws, TWIN) == "active", "lint must not have changed anything"

        # 2. lint_autofix stages a proposal — and only a proposal.
        staged = json.loads(lint_tools.lint_autofix(finding_id))
        proposal_id = staged["proposal_id"]
        assert staged["status"] == "staged", staged
        assert _status_of(ws, TWIN) == "active", "staging must not touch the block of record"

        # 3. the human gate is what actually applies it.
        dry = json.loads(governance.approve_apply(proposal_id, dry_run=True))
        assert dry["status"] == "dry_run_passed", dry
        with _PRECONDITIONS_PASS:
            applied = json.loads(governance.approve_apply(proposal_id, dry_run=False))
        assert applied["status"] == "applied", applied

        assert _status_of(ws, TWIN) == "superseded"
        assert _status_of(ws, WINNER) == "active", "the winner must be left alone"
        # …and the defect is gone, which is the only proof the repair was real.
        assert json.loads(lint_tools.lint(rule=RULE_DUPLICATE_BLOCK))["count"] == 0

    def test_finding_ids_are_stable_across_processes(self, tmp_path) -> None:
        """A finding id quoted from one run must name the same defect in the next.

        Two separate interpreters, so nothing in-process can be memoising the
        answer — if an id ever picked up a clock, a path or a hash seed, the
        ``--fix LF-...`` contract would be broken and this is what says so.
        """
        ws = _make_ws(tmp_path, enable_lint=True)
        first = json.loads(_run_cli(ws, "lint").stdout)
        second = json.loads(_run_cli(ws, "lint").stdout)
        ids = [f["finding_id"] for f in first["findings"]]
        assert ids, first
        assert ids == [f["finding_id"] for f in second["findings"]]


class TestAclClassification:
    """Both tools are classified, and the split is enforced."""

    def test_both_tools_are_registered_and_classified(self) -> None:
        import asyncio

        from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS
        from mind_mem.mcp.server import mcp

        registered = {tool.name for tool in asyncio.run(mcp.list_tools())}
        assert {"lint", "lint_autofix"} <= registered, "not registered on the server"
        assert "lint" in USER_TOOLS
        assert "lint_autofix" in ADMIN_TOOLS

    def test_user_scope_may_read_findings(self, tmp_path, monkeypatch) -> None:
        from mind_mem.mcp.tools import lint as lint_tools

        ws = _make_ws(tmp_path, enable_lint=True)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        monkeypatch.setenv("MIND_MEM_SCOPE", "user")
        payload = json.loads(lint_tools.lint())
        assert "ACL policy" not in json.dumps(payload)
        assert payload["count"] >= 1

    def test_user_scope_is_refused_the_staging_half(self, tmp_path, monkeypatch) -> None:
        """Staging a proposal is an admin act, and the gate — not the body —
        is what refuses it: nothing is written to the proposal file."""
        from mind_mem.mcp.tools import lint as lint_tools

        ws = _make_ws(tmp_path, enable_lint=True)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        monkeypatch.setenv("MIND_MEM_SCOPE", "user")
        before = _digests(ws)

        finding_id = json.loads(lint_tools.lint())["findings"][0]["finding_id"]
        refused = json.loads(lint_tools.lint_autofix(finding_id))

        assert refused["error"] == "Permission denied: 'lint_autofix' requires admin scope"
        assert _digests(ws) == before


class TestCliSurface:
    def test_mm_lint_reports_without_writing(self, tmp_path) -> None:
        ws = _make_ws(tmp_path, enable_lint=True)
        before = _digests(ws)

        result = _run_cli(ws, "lint")

        assert result.returncode == 1, result.stderr  # findings exist
        payload = json.loads(result.stdout)
        assert payload["count"] == len(payload["findings"]) >= 1
        assert TWIN in [f["block_id"] for f in payload["findings"]]
        assert _digests(ws) == before, "mm lint must be read-only"

    def test_mm_lint_fix_stages_exactly_one_proposal(self, tmp_path) -> None:
        from mind_mem.block_parser import parse_file

        ws = _make_ws(tmp_path, enable_lint=True)
        listed = json.loads(_run_cli(ws, "lint").stdout)
        finding_id = next(f["finding_id"] for f in listed["findings"] if f["rule"] == RULE_DUPLICATE_BLOCK)
        before = _digests(ws)

        result = _run_cli(ws, "lint", "--fix", finding_id)

        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["staged"].startswith("P-")

        after = _digests(ws)
        assert set(after) == set(before)
        changed = [rel for rel in after if after[rel] != before[rel]]
        assert changed == [PROPOSAL_FILE], f"only the proposal file may change, got {changed}"
        staged = parse_file(os.path.join(ws, PROPOSAL_FILE))
        assert [b["ProposalId"] for b in staged] == [payload["staged"]]
        assert staged[0]["Status"] == "staged"
        assert _status_of(ws, TWIN) == "active"

    def test_mm_lint_fix_refuses_an_unknown_finding(self, tmp_path) -> None:
        ws = _make_ws(tmp_path, enable_lint=True)
        before = _digests(ws)

        result = _run_cli(ws, "lint", "--fix", "LF-deadbeef")

        assert result.returncode == 1
        assert "no such finding" in json.loads(result.stdout)["error"]
        assert _digests(ws) == before


class TestRequiredFieldsComeFromTheIncumbent:
    def test_a_field_the_drifted_copy_omitted_is_reported(self, tmp_path) -> None:
        """``Context`` is required by ``validate_py`` and was missing from the
        copy that used to live in ``lint``. The lint and the validator have to
        answer "is this field required?" the same way, or the lint advises
        against a schema the gate does not enforce."""
        from mind_mem.lint import lint as _lint
        from mind_mem.validate_py import REQUIRED_FIELDS_BY_PREFIX

        assert "Context" in REQUIRED_FIELDS_BY_PREFIX["T"]

        ws = _make_ws(tmp_path, enable_lint=True, tasks=TASKS)
        empty_fields = [f.detail for f in _lint(ws) if f.rule == RULE_MISSING_METADATA and f.block_id == "T-20260105-001"]
        assert any("'Context'" in detail for detail in empty_fields), empty_fields


class TestFlagOffIsInert:
    """Default config: every new surface answers, and does, nothing."""

    def test_mcp_tools_are_inert(self, tmp_path, monkeypatch) -> None:
        from mind_mem.mcp.tools import lint as lint_tools

        ws = _make_ws(tmp_path, enable_lint=False)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
        before = _digests(ws)

        reported = json.loads(lint_tools.lint())
        staged = json.loads(lint_tools.lint_autofix("LF-b07a317a"))

        assert reported["error"] == "v4.lint is disabled"
        assert "findings" not in reported
        assert staged["error"] == "v4.lint is disabled"
        assert _digests(ws) == before

    def test_mm_lint_is_inert(self, tmp_path) -> None:
        ws = _make_ws(tmp_path, enable_lint=False)
        before = _digests(ws)

        listed = _run_cli(ws, "lint")
        fixed = _run_cli(ws, "lint", "--fix", "LF-b07a317a")

        assert listed.returncode == 1
        assert json.loads(listed.stdout)["error"] == "v4.lint is disabled"
        assert fixed.returncode == 1
        assert json.loads(fixed.stdout)["error"] == "v4.lint is disabled"
        assert _digests(ws) == before

    def test_the_probe_is_not_observable(self, tmp_path, capfd) -> None:
        """A probe deciding whether a feature is on must not itself be a
        behaviour change when the answer is no.

        ``feature_flags.is_enabled`` logs ``v4_config_unreadable`` on a config
        it cannot parse. Routing the OFF check through it would make the wired
        build emit, on a workspace with a malformed ``mind-mem.json``, a
        stderr line the unwired build never emitted. So the probe must not
        call that helper, and must stay silent.
        """
        from mind_mem import lint as lint_mod

        ws = _make_ws(tmp_path, enable_lint=False)
        with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as handle:
            handle.write('{"v4": {"lint": {"enabled": true},,,}')  # deliberately malformed
        capfd.readouterr()

        with patch(
            "mind_mem.v4.feature_flags.is_enabled",
            side_effect=AssertionError("the OFF probe must not route through the logging helper"),
        ):
            assert lint_mod.flag_enabled(ws) is False

        captured = capfd.readouterr()
        assert captured.err == ""
        assert captured.out == ""

    def test_malformed_config_does_not_switch_the_surface_on(self, tmp_path) -> None:
        """Fail-closed: an unparseable workspace config is OFF, not ON."""
        from mind_mem import lint as lint_mod

        ws = _make_ws(tmp_path, enable_lint=True)
        with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as handle:
            handle.write("{not json at all")
        assert lint_mod.flag_enabled(ws) is False

    def test_a_bare_truthy_value_does_not_switch_the_surface_on(self, tmp_path) -> None:
        """Only the canonical ``{"enabled": true}`` shape counts."""
        from mind_mem import lint as lint_mod

        ws = _make_ws(tmp_path, enable_lint=False)
        config_path = os.path.join(ws, "mind-mem.json")
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        config["v4"] = {"lint": True}
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle)
        assert lint_mod.flag_enabled(ws) is False


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__]))
