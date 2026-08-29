# Copyright 2026 STARGA, Inc.
"""``mm review`` — the keyboard-driven front end, and its wiring.

Wiring is asserted, not assumed: this repo already carries modules with
green tests and zero production importers, and a review surface nobody
can reach is exactly that failure again.
"""

from __future__ import annotations

import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _review_fixtures import build_workspace, mcp_budget, proposal_status  # noqa: E402,F401


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A workspace an operator holds admin scope over.

    ``approve_apply`` is an admin-scope MCP tool; exporting
    ``MIND_MEM_SCOPE=admin`` is what a reviewing operator does. ``mm
    review`` reports the scope as a blocker but never sets it.
    """
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    root = str(tmp_path / "ws")
    os.makedirs(root)
    ids = build_workspace(root, 3)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", root)
    return root, ids


@pytest.fixture
def no_window(monkeypatch):
    monkeypatch.setattr(
        "mind_mem.apply_engine.check_no_touch_window",
        lambda ws: (True, "test: window neutralised"),
    )


class TestWiring:
    def test_review_is_a_registered_mm_subcommand(self):
        from mind_mem.mm_cli import build_parser

        parser = build_parser()
        actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
        names = set()
        for action in actions:
            names.update(action.choices or {})
        assert "review" in names

    def test_review_dispatches_into_the_review_module(self):
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        args = build_parser().parse_args(["review", "--json"])
        assert args.func is cmd_review

    def test_the_review_modules_have_a_production_importer(self):
        """Every review module must be reachable from shipped code."""
        import subprocess

        src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "mind_mem")
        for module in ("review_queue", "review_preview", "review_batch", "review_metrics", "review_render"):
            hits = subprocess.run(
                ["grep", "-rn", f"{module}", src, "--include=*.py"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.splitlines()
            importers = {
                os.path.basename(line.split(":")[0])
                for line in hits
                if ("import" in line) and os.path.basename(line.split(":")[0]) != f"{module}.py"
            }
            assert importers, f"{module} has no production importer — it is dead code"


class TestListMode:
    def test_json_listing_names_every_staged_proposal(self, workspace, capsys):
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        root, ids = workspace
        rc = cmd_review(build_parser().parse_args(["review", "--json"]))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert [item["proposal_id"] for item in payload["queue"]] == list(ids)

    def test_json_listing_carries_health_and_blockers(self, workspace, capsys):
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        cmd_review(build_parser().parse_args(["review", "--json"]))
        payload = json.loads(capsys.readouterr().out)
        assert payload["health"]["backlog_count"] == 3
        assert "blockers" in payload["health"]

    def test_text_listing_shows_id_target_risk_and_age(self, workspace, capsys):
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        root, ids = workspace
        cmd_review(build_parser().parse_args(["review"]))
        out = capsys.readouterr().out
        assert ids[0] in out
        assert "D-20260801-001" in out
        assert "low" in out

    def test_listing_does_not_mutate_the_queue(self, workspace, capsys):
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        root, ids = workspace
        cmd_review(build_parser().parse_args(["review"]))
        capsys.readouterr()
        assert all(proposal_status(root, pid) == "staged" for pid in ids)

    def test_an_empty_queue_is_reported_not_an_error(self, tmp_path, monkeypatch, capsys):
        from mind_mem.init_workspace import init
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        root = str(tmp_path / "empty")
        os.makedirs(root)
        init(root)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", root)
        assert cmd_review(build_parser().parse_args(["review"])) == 0
        assert "no proposals" in capsys.readouterr().out.lower()


class TestHostileProposalCannotSpoofTheSurface:
    """End-to-end: a hostile proposal reaching the real CLI, not just the renderer."""

    def test_ansi_and_cr_in_a_real_proposal_never_reach_stdout(self, workspace, capsys):
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        root, ids = workspace
        path = os.path.join(root, "intelligence/proposed/EDITS_PROPOSED.md")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                text.replace(
                    "- observed drift in decision 1",
                    "- drift\x1b[2J\x1b[H\rChain: valid=True   Stale: no",
                    1,
                )
            )
        cmd_review(build_parser().parse_args(["review", "--show", ids[0]]))
        out = capsys.readouterr().out
        assert "\x1b" not in out
        assert "\r" not in out
        forged = [line for line in out.splitlines() if line.startswith("Chain: valid=True   Stale: no")]
        assert forged == []


class TestScopeBlocker:
    def test_a_non_admin_scope_is_named_in_the_listing(self, workspace, monkeypatch, capsys):
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        monkeypatch.setenv("MIND_MEM_SCOPE", "user")
        cmd_review(build_parser().parse_args(["review"]))
        out = capsys.readouterr().out
        assert "MIND_MEM_SCOPE=admin" in out
        assert "BLOCKERS" in out

    def test_the_cli_never_sets_the_scope_itself(self, workspace, monkeypatch, capsys):
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        monkeypatch.setenv("MIND_MEM_SCOPE", "user")
        cmd_review(build_parser().parse_args(["review", "--approve", "P-20260829-001"]))
        capsys.readouterr()
        assert os.environ["MIND_MEM_SCOPE"] == "user"


class TestShowMode:
    def test_show_renders_the_pre_apply_diff_inline(self, workspace, capsys):
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        root, ids = workspace
        cmd_review(build_parser().parse_args(["review", "--show", ids[0]]))
        out = capsys.readouterr().out
        assert "+Tags: baseline,reviewed-1" in out
        assert "Evidence" in out

    def test_show_renders_chain_and_staleness_inline(self, workspace, capsys):
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        root, ids = workspace
        cmd_review(build_parser().parse_args(["review", "--show", ids[0]]))
        out = capsys.readouterr().out
        assert "Chain" in out
        assert "Stale" in out

    def test_show_is_not_narrowed_by_limit(self, workspace, capsys):
        """--limit pages the listing; it must not hide a proposal from --show."""
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        root, ids = workspace
        rc = cmd_review(build_parser().parse_args(["review", "--limit", "1", "--show", ids[2]]))
        assert rc == 0
        assert ids[2] in capsys.readouterr().out

    def test_show_on_an_unknown_id_is_a_clean_error(self, workspace, capsys):
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        assert cmd_review(build_parser().parse_args(["review", "--show", "P-20990101-999"])) == 1


class TestNonInteractiveBatch:
    def test_approve_flag_applies_only_the_named_proposals(self, workspace, no_window, capsys):
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        root, ids = workspace
        rc = cmd_review(build_parser().parse_args(["review", "--approve", f"{ids[0]},{ids[1]}", "--json"]))
        capsys.readouterr()
        assert rc == 0
        assert proposal_status(root, ids[0]) == "applied"
        assert proposal_status(root, ids[1]) == "applied"
        assert proposal_status(root, ids[2]) == "staged"

    def test_reject_flag_requires_a_reason(self, workspace, no_window, capsys):
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        root, ids = workspace
        rc = cmd_review(build_parser().parse_args(["review", "--reject", ids[0]]))
        capsys.readouterr()
        assert rc == 2
        assert proposal_status(root, ids[0]) == "staged"

    def test_a_malformed_id_is_a_usage_error_before_anything_is_applied(self, workspace, no_window, capsys):
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        root, ids = workspace
        rc = cmd_review(build_parser().parse_args(["review", "--approve", f"{ids[0]},../../etc/passwd"]))
        capsys.readouterr()
        assert rc == 2
        assert proposal_status(root, ids[0]) == "staged"

    def test_batch_result_publishes_the_metric(self, workspace, no_window, capsys):
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        root, ids = workspace
        cmd_review(build_parser().parse_args(["review", "--approve", ids[0], "--json"]))
        payload = json.loads(capsys.readouterr().out)
        assert "metrics" in payload
        assert payload["metrics"]["decisions"] == 1
        assert "proposals_per_minute" in payload["metrics"]
        assert "median_age_at_approval_seconds" in payload["metrics"]

    def test_a_failed_proposal_yields_a_nonzero_exit_and_a_report(self, workspace, capsys):
        """The no-touch window blocks 2 of 3; the failure must be visible."""
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        root, ids = workspace
        rc = cmd_review(build_parser().parse_args(["review", "--approve", ",".join(ids), "--json"]))
        payload = json.loads(capsys.readouterr().out)
        assert rc == 1
        assert len(payload["applied"]) == 1
        assert len(payload["failed"]) == 2


class TestKeyboardSession:
    def test_one_keystroke_per_proposal_produces_one_decision(self, workspace):
        from mind_mem.review_queue import load_queue
        from mind_mem.review_session import review_session

        root, ids = workspace
        out = io.StringIO()
        decisions = review_session(root, load_queue(root), keys=iter("aaac"), out=out)
        assert [d.proposal_id for d in decisions] == list(ids)
        assert all(d.action == "approve" for d in decisions)
        assert all(d.origin == "keypress" for d in decisions)

    def test_decisions_never_outnumber_operator_keystrokes(self, workspace):
        from mind_mem.review_queue import load_queue
        from mind_mem.review_session import review_session

        root, _ids = workspace
        keys = "asc"
        decisions = review_session(root, load_queue(root), keys=iter(keys), out=io.StringIO())
        assert len(decisions) <= len(keys)

    def test_skip_leaves_a_proposal_undecided(self, workspace):
        from mind_mem.review_queue import load_queue
        from mind_mem.review_session import review_session

        root, ids = workspace
        decisions = review_session(root, load_queue(root), keys=iter("asac"), out=io.StringIO())
        assert [d.proposal_id for d in decisions] == [ids[0], ids[2]]

    def test_quit_discards_every_decision(self, workspace):
        from mind_mem.review_queue import load_queue
        from mind_mem.review_session import review_session

        root, _ids = workspace
        assert review_session(root, load_queue(root), keys=iter("aaq"), out=io.StringIO()) == ()

    def test_exhausted_key_source_discards_rather_than_approves(self, workspace):
        """A closed pipe must never be read as consent."""
        from mind_mem.review_queue import load_queue
        from mind_mem.review_session import review_session

        root, _ids = workspace
        assert review_session(root, load_queue(root), keys=iter("aa"), out=io.StringIO()) == ()

    def test_reject_collects_a_reason_through_the_prompt(self, workspace):
        from mind_mem.review_queue import load_queue
        from mind_mem.review_session import review_session

        root, ids = workspace
        decisions = review_session(
            root,
            load_queue(root),
            keys=iter("rssc"),
            out=io.StringIO(),
            reason_prompt=lambda pid: "superseded upstream",
        )
        assert decisions[0].action == "reject"
        assert decisions[0].reason == "superseded upstream"

    def test_a_reject_with_no_reason_is_dropped_not_applied(self, workspace):
        from mind_mem.review_queue import load_queue
        from mind_mem.review_session import review_session

        root, _ids = workspace
        decisions = review_session(
            root,
            load_queue(root),
            keys=iter("rssc"),
            out=io.StringIO(),
            reason_prompt=lambda pid: "",
        )
        assert decisions == ()

    def test_the_session_writes_nothing(self, workspace):
        from mind_mem.review_queue import load_queue
        from mind_mem.review_session import review_session

        root, ids = workspace
        review_session(root, load_queue(root), keys=iter("aaac"), out=io.StringIO())
        assert all(proposal_status(root, pid) == "staged" for pid in ids)

    def test_the_session_renders_the_diff_for_each_proposal(self, workspace):
        from mind_mem.review_queue import load_queue
        from mind_mem.review_session import review_session

        root, _ids = workspace
        out = io.StringIO()
        review_session(root, load_queue(root), keys=iter("sssc"), out=out)
        rendered = out.getvalue()
        assert "+Tags: baseline,reviewed-1" in rendered
        assert "+Tags: baseline,reviewed-3" in rendered
