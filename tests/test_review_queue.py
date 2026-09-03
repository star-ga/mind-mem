# Copyright 2026 STARGA, Inc.
"""``review_queue`` — the read-only listing behind ``mm review``."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _review_fixtures import DECISION_FILE, build_workspace, mcp_budget, proposal_status  # noqa: E402,F401


@pytest.fixture
def workspace(tmp_path):
    root = str(tmp_path / "ws")
    os.makedirs(root)
    ids = build_workspace(root, 3)
    return root, ids


class TestLoadQueue:
    def test_lists_every_staged_proposal(self, workspace):
        from mind_mem.review_queue import load_queue

        root, ids = workspace
        items = load_queue(root)
        assert tuple(item.proposal_id for item in items) == ids

    def test_order_is_deterministic_and_id_lexicographic(self, workspace):
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        first = [item.proposal_id for item in load_queue(root)]
        second = [item.proposal_id for item in load_queue(root)]
        assert first == second == sorted(first)

    def test_carries_the_review_payload(self, workspace):
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        item = load_queue(root)[0]
        assert item.target_block == "D-20260801-001"
        assert item.risk == "low"
        assert item.evidence == ("observed drift in decision 1",)
        assert item.files_touched == (DECISION_FILE,)
        assert item.op_summary == ("update_field decisions/DECISIONS.md:D-20260801-001",)
        assert item.validation_errors == ()

    def test_applied_proposals_are_not_listed(self, workspace):
        from mind_mem.review_queue import load_queue

        root, ids = workspace
        path = os.path.join(root, "intelligence/proposed/EDITS_PROPOSED.md")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text.replace("Status: staged", "Status: applied", 1))
        listed = [item.proposal_id for item in load_queue(root)]
        assert ids[0] not in listed
        assert listed == list(ids[1:])

    def test_limit_truncates_without_reordering(self, workspace):
        from mind_mem.review_queue import load_queue

        root, ids = workspace
        assert [item.proposal_id for item in load_queue(root, limit=2)] == list(ids[:2])

    def test_rejects_a_non_positive_limit(self, workspace):
        from mind_mem.review_queue import ReviewQueueError, load_queue

        root, _ids = workspace
        with pytest.raises(ReviewQueueError):
            load_queue(root, limit=0)

    def test_rejects_a_missing_workspace(self, tmp_path):
        from mind_mem.review_queue import ReviewQueueError, load_queue

        with pytest.raises(ReviewQueueError):
            load_queue(str(tmp_path / "nope"))

    def test_surfaces_validation_errors_instead_of_raising(self, workspace):
        from mind_mem.review_queue import load_queue

        root, ids = workspace
        path = os.path.join(root, "intelligence/proposed/EDITS_PROPOSED.md")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text.replace("Risk: low", "Risk: catastrophic", 1))
        item = next(i for i in load_queue(root) if i.proposal_id == ids[0])
        assert item.validation_errors
        assert not item.applicable
        assert any("Risk" in err for err in item.validation_errors)

    def test_reads_the_queue_without_mutating_it(self, workspace):
        from mind_mem.review_queue import load_queue

        root, ids = workspace
        load_queue(root)
        assert proposal_status(root, ids[0]) == "staged"


class TestAge:
    def test_age_is_none_without_a_created_field(self, workspace):
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        assert load_queue(root)[0].created == ""
        assert load_queue(root)[0].age_seconds(now_iso="2026-08-29T12:00:00Z") is None

    def test_age_is_computed_from_created_against_an_injected_now(self, tmp_path):
        from mind_mem.review_queue import load_queue

        root = str(tmp_path / "ws")
        os.makedirs(root)
        build_workspace(root, 1, created=["2026-08-29T11:00:00Z"])
        item = load_queue(root)[0]
        assert item.age_seconds(now_iso="2026-08-29T12:00:00Z") == 3600.0

    def test_a_malformed_created_yields_no_age_rather_than_an_error(self, tmp_path):
        from mind_mem.review_queue import load_queue

        root = str(tmp_path / "ws")
        os.makedirs(root)
        build_workspace(root, 1, created=["not-a-timestamp"])
        assert load_queue(root)[0].age_seconds(now_iso="2026-08-29T12:00:00Z") is None


class TestQueueHealth:
    def test_reports_the_governance_gates_that_cap_throughput(self, workspace):
        from mind_mem.review_queue import queue_health

        root, _ids = workspace
        health = queue_health(root)
        assert health.governance_mode == "propose_apply"
        assert health.backlog_count == 3
        assert health.backlog_over_limit is False
        assert health.no_touch_ok is True

    def test_flags_detect_only_as_a_blocked_queue(self, workspace):
        """A ``detect_only`` workspace reports as blocked, not as healthy.

        The mode is flipped in ``mind-mem.json`` through ``mm config set``
        because that is the file ``apply_engine._get_mode`` reads and the
        spec binding attests. A hand edit would be drift the gate refuses;
        an edit to ``memory/intel-state.json`` would not reach the engine
        at all (see the companion test below).

        The pre-flip assertions are the positive control: they prove the
        fixture starts *unblocked*, so the blocker asserted after the flip
        was produced by the flip rather than inherited from a workspace
        that was already sitting on the shipped ``detect_only`` default.
        """
        from mind_mem.mm_cli import config_set
        from mind_mem.review_queue import queue_health

        root, _ids = workspace
        before = queue_health(root)
        assert before.governance_mode == "propose_apply"
        assert not [b for b in before.blockers if "detect_only" in b]

        config_set(os.path.join(root, "mind-mem.json"), "governance_mode", "detect_only")

        health = queue_health(root)
        assert health.governance_mode == "detect_only"
        assert health.blockers
        assert any("detect_only" in b for b in health.blockers)

    def test_the_unattested_state_file_does_not_change_the_reported_mode(self, workspace):
        """``memory/intel-state.json`` cannot flip what the queue reports.

        Regression guard for the two-files-one-word defect: the apply
        engine read ``intel-state.json`` while the governance gate read
        ``mind-mem.json``, so an edit to the unattested file changed what
        the engine would apply while the attested file — the one whose
        changes the spec binding records as DRIFT — said otherwise. Both
        now read the attested config, and ``queue_health`` reports what
        the engine will do, so this edit must be inert here.

        Paired with the test above, which flips the attested file and
        *does* see the blocker: that is the positive control proving this
        assertion can fail.
        """
        import json

        from mind_mem.review_queue import queue_health

        root, _ids = workspace
        state_path = os.path.join(root, "memory/intel-state.json")
        with open(state_path, encoding="utf-8") as handle:
            state = json.load(handle)
        assert state["governance_mode"] == "propose_apply"
        state["governance_mode"] = "detect_only"
        with open(state_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle)

        health = queue_health(root)
        assert health.governance_mode == "propose_apply"
        assert not [b for b in health.blockers if "detect_only" in b]
