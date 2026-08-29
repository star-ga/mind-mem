"""Pooling stratum artifacts into one delta -- and refusing to pool dishonestly.

A hundred tasks are run one stratum at a time, so the headline number is
assembled here rather than measured in one process.  That makes two
failure modes worth testing more than the happy path: pooling runs that
were not the same experiment, and counting one task's pair twice.  Both
would move the discordant count, which is the only thing the significance
claim rests on.
"""

from __future__ import annotations

import json
import os

import pytest

from mind_mem.bench.ab_cli import main
from mind_mem.bench.ab_report import (
    POOLED_DOC,
    Pair,
    ReportError,
    assert_comparable,
    build_report,
    collect_pairs,
    comparability_signature,
    group_summaries,
    load_artifact,
    pairs_of,
    pool_exclusions,
    pool_inert,
    pool_spend,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BUDGET = {"prompt_tokens": 8000, "memory_tokens": 1500, "output_tokens": 4000, "wall_seconds": 300, "steps": 40}
AGENT = {"name": "command", "argv": ["/bin/true"], "env_passthrough_keys": []}


def record(task_id: str, memory: bool, control: bool, tier: str = "behavioral", bucket: str = "small") -> dict:
    """One scored task row, in the shape ``run`` writes."""
    return {
        "task_id": task_id,
        "tier": tier,
        "size_bucket": bucket,
        "excluded": None,
        "outcome": {"memory_success": memory, "control_success": control},
    }


def artifact(rows: list[dict], *, excluded: list[dict] | None = None, inert: list[str] | None = None, **overrides) -> dict:
    """A minimal but structurally faithful run artifact."""
    payload = {
        "budget": dict(BUDGET),
        "agent": dict(AGENT),
        "task_set": {"sha256": "abc", "selection": "bucket:small"},
        "counts": {"scored": len(rows), "excluded": len(excluded or []), "agent_inert_task_ids": list(inert or [])},
        "spend": {
            "control": {"prompt_tokens": 100, "memory_tokens": 0, "agent_output_tokens_lower_bound": 10, "timeouts": 0},
            "memory": {"prompt_tokens": 400, "memory_tokens": 300, "agent_output_tokens_lower_bound": 20, "timeouts": 1},
        },
        "excluded": list(excluded or []),
        "results": [*rows, *(excluded or [])],
        "summary": {},
        "digest": "d" * 64,
    }
    payload.update(overrides)
    return payload


def write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


class TestPoolingIsPaired:
    def test_each_task_contributes_exactly_one_pair(self):
        pairs = pairs_of(artifact([record("a", True, False), record("b", False, False)]))
        assert [(p.task_id, p.memory, p.control) for p in pairs] == [("a", True, False), ("b", False, False)]

    def test_an_excluded_row_contributes_no_pair(self):
        excluded = [{"task_id": "x", "excluded": "setup_failed", "detail": {}}]
        assert pairs_of(artifact([record("a", True, True)], excluded=excluded)) == (Pair("a", True, True, "behavioral", "small"),)

    def test_the_pooled_summary_is_the_summary_of_the_concatenated_pairs(self, tmp_path):
        left = write(tmp_path, "l.json", artifact([record("a", True, False), record("b", True, False)]))
        right = write(tmp_path, "r.json", artifact([record("c", True, False), record("d", False, True)]))
        payload = build_report([left, right])
        assert payload["summary"]["n_tasks"] == 4
        assert payload["summary"]["memory_only"] == 3
        assert payload["summary"]["control_only"] == 1
        assert payload["summary"]["n_discordant"] == 4

    def test_four_discordant_pairs_are_still_reported_as_underpowered(self, tmp_path):
        left = write(tmp_path, "l.json", artifact([record("a", True, False)]))
        right = write(tmp_path, "r.json", artifact([record("b", True, False)]))
        payload = build_report([left, right])
        assert payload["summary"]["verdict"] == "underpowered"
        assert payload["summary"]["min_discordant_for_significance"] == 6


class TestPoolingRefusesWhatWouldMislead:
    def test_a_task_present_in_two_artifacts_is_refused(self, tmp_path):
        left = write(tmp_path, "l.json", artifact([record("a", True, False)]))
        right = write(tmp_path, "r.json", artifact([record("a", True, False)]))
        with pytest.raises(ReportError, match="counted twice"):
            build_report([left, right])

    def test_a_different_budget_is_refused(self, tmp_path):
        other = dict(BUDGET, wall_seconds=900)
        left = write(tmp_path, "l.json", artifact([record("a", True, False)]))
        right = write(tmp_path, "r.json", artifact([record("b", True, False)], budget=other))
        with pytest.raises(ReportError, match="not one experiment"):
            build_report([left, right])

    def test_a_different_agent_is_refused(self, tmp_path):
        left = write(tmp_path, "l.json", artifact([record("a", True, False)]))
        right = write(tmp_path, "r.json", artifact([record("b", True, False)], agent={"name": "none", "argv": []}))
        with pytest.raises(ReportError, match="not one experiment"):
            build_report([left, right])

    def test_a_different_task_set_is_refused(self, tmp_path):
        left = write(tmp_path, "l.json", artifact([record("a", True, False)]))
        right = write(tmp_path, "r.json", artifact([record("b", True, False)], task_set={"sha256": "zzz", "selection": "all"}))
        with pytest.raises(ReportError, match="not one experiment"):
            build_report([left, right])

    def test_the_comparability_signature_names_what_must_match(self):
        signature = comparability_signature(artifact([]))
        assert sorted(signature) == ["agent", "budget", "task_set_sha256"]

    def test_identical_runs_are_comparable(self):
        assert_comparable([artifact([]), artifact([])], ["a", "b"]) is None

    def test_a_file_that_is_not_a_run_artifact_is_refused(self, tmp_path):
        path = write(tmp_path, "nope.json", {"hello": "world"})
        with pytest.raises(ReportError, match="not an A/B run artifact"):
            load_artifact(path)

    def test_a_json_scalar_is_refused_before_key_checks(self, tmp_path):
        path = tmp_path / "scalar.json"
        path.write_text("42", encoding="utf-8")
        with pytest.raises(ReportError, match="not an object"):
            load_artifact(str(path))

    def test_pooling_nothing_is_refused_rather_than_reported_as_zero(self):
        with pytest.raises(ReportError, match="no artifacts"):
            build_report([])

    def test_artifacts_with_no_scored_task_are_refused(self, tmp_path):
        path = write(tmp_path, "empty.json", artifact([]))
        with pytest.raises(ReportError, match="no scored task"):
            build_report([path])

    def test_collect_pairs_names_both_artifacts_in_the_duplicate_message(self):
        with pytest.raises(ReportError, match="left.json and right.json"):
            collect_pairs([artifact([record("a", True, True)]), artifact([record("a", True, True)])], ["left.json", "right.json"])


class TestTheStrataAreShownBesideTheHeadline:
    def test_tiers_are_summarised_separately(self, tmp_path):
        rows = [record("a", True, False, tier="behavioral"), record("b", False, False, tier="api_construction")]
        payload = build_report([write(tmp_path, "a.json", artifact(rows))])
        assert payload["by_tier"]["behavioral"]["memory_only"] == 1
        assert payload["by_tier"]["api_construction"]["neither_passed"] == 1

    def test_size_buckets_are_summarised_separately(self, tmp_path):
        rows = [record("a", True, True, bucket="single_file"), record("b", False, False, bucket="large")]
        payload = build_report([write(tmp_path, "a.json", artifact(rows))])
        assert payload["by_size_bucket"]["single_file"]["both_passed"] == 1
        assert payload["by_size_bucket"]["large"]["neither_passed"] == 1

    def test_grouping_an_unknown_attribute_value_still_reports_it(self):
        pairs = [Pair("a", True, False, "unknown", "unknown")]
        assert group_summaries(pairs, "tier")["unknown"]["n_tasks"] == 1


class TestNothingIsQuietlyDropped:
    def test_exclusions_are_pooled_with_their_reasons_and_count(self):
        left = artifact([record("a", True, True)], excluded=[{"task_id": "x", "excluded": "setup_failed"}])
        right = artifact([record("b", True, True)], excluded=[{"task_id": "y", "excluded": "prompt_over_budget"}])
        pooled = pool_exclusions([left, right])
        assert pooled["total"] == 2
        assert pooled["by_reason"] == {"prompt_over_budget": 1, "setup_failed": 1}
        assert [row["task_id"] for row in pooled["tasks"]] == ["x", "y"]

    def test_an_inert_agent_stays_visible_after_pooling(self):
        assert pool_inert([artifact([], inert=["b"]), artifact([], inert=["a"])]) == ["a", "b"]

    def test_spend_is_summed_per_arm_so_an_unequal_arm_is_visible(self):
        totals = pool_spend([artifact([]), artifact([])])
        assert totals["control"]["prompt_tokens"] == 200
        assert totals["memory"]["prompt_tokens"] == 800
        assert totals["memory"]["timeouts"] == 2

    def test_the_report_states_its_own_design(self):
        assert set(POOLED_DOC) == {"question", "pooling", "refusals", "statistics"}


class TestTheReportIsDeterministicAndWired:
    def test_the_same_inputs_produce_the_same_digest(self, tmp_path):
        path = write(tmp_path, "a.json", artifact([record("a", True, False)]))
        assert build_report([path])["digest"] == build_report([path])["digest"]

    def test_a_changed_outcome_changes_the_digest(self, tmp_path):
        one = write(tmp_path, "a.json", artifact([record("a", True, False)]))
        two = write(tmp_path, "b.json", artifact([record("a", False, False)]))
        assert build_report([one])["digest"] != build_report([two])["digest"]

    def test_the_cli_prints_the_greppable_pooled_lines(self, tmp_path, capsys):
        path = write(tmp_path, "a.json", artifact([record("a", True, False), record("b", False, True)]))
        assert main(["report", "--artifact", path]) == 0
        out = capsys.readouterr().out
        for key in ("pooled_tasks", "pooled_delta", "pooled_discordant", "pooled_p_value", "pooled_verdict", "pooled_digest"):
            assert f"memory_ab_{key}: " in out

    def test_the_cli_writes_the_pooled_artifact_when_asked(self, tmp_path, capsys):
        path = write(tmp_path, "a.json", artifact([record("a", True, False)]))
        out_path = tmp_path / "pooled.json"
        assert main(["report", "--artifact", path, "--out", str(out_path), "--repo", str(tmp_path)]) == 0
        assert "memory_ab_pooled_artifact: " in capsys.readouterr().out
        assert json.loads(out_path.read_text(encoding="utf-8"))["n_pairs"] == 1

    def test_the_package_exports_the_pooling_api(self):
        import mind_mem.bench as bench

        assert {"build_report", "ReportError", "Pair", "pool_spend"} <= set(bench.__all__)


class TestThePositiveControlReplaysTheWholeFix:
    """A replay that skips deletions reports a grader failure that is its own.

    The first version wrote only added/modified files under ``src/``, so on
    a commit that fixes a defect by deleting a module the tests stayed red
    and the control announced "the grader cannot see a success". The grader
    was fine. These pin the three statuses.
    """

    def test_a_deletion_is_replayed_as_a_deletion(self, tmp_path):
        from mind_mem.bench.ab_agent import AgentRequest, make_reference_fix_agent

        tree = tmp_path / "tree"
        (tree / "src").mkdir(parents=True)
        (tree / "src" / "gone.py").write_text("stale\n", encoding="utf-8")
        fixer = make_reference_fix_agent(str(tmp_path), "HEAD", [("D", "src/gone.py")])
        result = fixer(AgentRequest("t", "selfcheck", "", str(tree), {}, 10, 0, 0))
        assert not (tree / "src" / "gone.py").exists()
        assert result.steps == 1
        assert "replayed 1 path" in result.tail

    def test_deleting_an_absent_file_is_not_an_error(self, tmp_path):
        from mind_mem.bench.ab_agent import AgentRequest, make_reference_fix_agent

        tree = tmp_path / "tree"
        tree.mkdir()
        fixer = make_reference_fix_agent(str(tmp_path), "HEAD", [("D", "src/never_there.py")])
        assert fixer(AgentRequest("t", "selfcheck", "", str(tree), {}, 10, 0, 0)).returncode == 0

    def test_the_whole_non_test_delta_is_collected_with_its_statuses(self):
        from mind_mem.bench.ab_cli import commit_fix_paths

        # A commit from this repository that deletes source files, adds a
        # test file and edits documentation: every status in one row.
        paths = commit_fix_paths(REPO_ROOT, "668df44fe17c")
        statuses = {status for status, _ in paths}
        collected = {path for _, path in paths}
        assert "D" in statuses and "M" in statuses
        assert "src/mind_mem/tier_recall.py" in collected
        assert "docs/configuration.md" in collected

    def test_no_test_side_path_is_ever_replayed(self):
        from mind_mem.bench.ab_cli import commit_fix_paths

        paths = commit_fix_paths(REPO_ROOT, "668df44fe17c")
        assert not [path for _, path in paths if path.startswith("tests/") or path == "conftest.py"]

    def test_a_fix_living_outside_src_is_still_replayed(self):
        from mind_mem.bench.ab_cli import commit_fix_paths

        collected = {path for _, path in commit_fix_paths(REPO_ROOT, "78c09fbe2c2f")}
        assert "benchmarks/feedback_success_bench.py" in collected
        assert "pyproject.toml" in collected
