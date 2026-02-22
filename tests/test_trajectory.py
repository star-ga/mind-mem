#!/usr/bin/env python3
"""Tests for trajectory.py — trajectory memory block operations."""

import os
import sys
import tempfile
import unittest
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from trajectory import (
    _TRAJ_ID_RE,
    compute_similarity,
    format_trajectory_md,
    generate_id,
    parse_trajectory_md,
    validate_block,
)


class TestTrajectoryId(unittest.TestCase):
    """Test trajectory ID generation and validation."""

    def test_id_pattern_valid(self):
        self.assertIsNotNone(_TRAJ_ID_RE.match("TRAJ-20260221-001"))
        self.assertIsNotNone(_TRAJ_ID_RE.match("TRAJ-20260221-999"))

    def test_id_pattern_invalid(self):
        self.assertIsNone(_TRAJ_ID_RE.match("D-20260221-001"))
        self.assertIsNone(_TRAJ_ID_RE.match("TRAJ-2026-001"))
        self.assertIsNone(_TRAJ_ID_RE.match("TRAJ-20260221"))

    def test_generate_id_format(self):
        tid = generate_id()
        self.assertTrue(tid.startswith("TRAJ-"))
        self.assertIsNotNone(_TRAJ_ID_RE.match(tid))

    def test_generate_id_increments(self):
        with tempfile.TemporaryDirectory() as td:
            traj_dir = os.path.join(td, "trajectories")
            os.makedirs(traj_dir)
            # Create first trajectory
            first = generate_id(td)
            self.assertTrue(first.endswith("-001"))
            # Write it
            with open(os.path.join(traj_dir, f"{first}.md"), "w") as f:
                f.write("test")
            # Next should be -002
            second = generate_id(td)
            self.assertTrue(second.endswith("-002"))


class TestValidateBlock(unittest.TestCase):
    """Test trajectory block validation."""

    def test_valid_block(self):
        block = {
            "_id": "TRAJ-20260221-001",
            "Task": "Deploy v1.0.6",
            "Date": "2026-02-21",
            "Outcome": "SUCCESS",
            "Reward": 1.0,
        }
        self.assertEqual(validate_block(block), [])

    def test_missing_required_fields(self):
        block = {"_id": "TRAJ-20260221-001"}
        errors = validate_block(block)
        self.assertEqual(len(errors), 3)

    def test_invalid_outcome(self):
        block = {
            "Task": "Test",
            "Date": "2026-02-21",
            "Outcome": "MAYBE",
        }
        errors = validate_block(block)
        self.assertTrue(any("Invalid outcome" in e for e in errors))

    def test_reward_out_of_range(self):
        block = {
            "Task": "Test",
            "Date": "2026-02-21",
            "Outcome": "SUCCESS",
            "Reward": 1.5,
        }
        errors = validate_block(block)
        self.assertTrue(any("out of range" in e for e in errors))

    def test_invalid_date(self):
        block = {
            "Task": "Test",
            "Date": "21-02-2026",
            "Outcome": "SUCCESS",
        }
        errors = validate_block(block)
        self.assertTrue(any("Invalid date" in e for e in errors))

    def test_all_outcomes_valid(self):
        for outcome in ("SUCCESS", "FAILURE", "PARTIAL", "ABORTED"):
            block = {"Task": "t", "Date": "2026-01-01", "Outcome": outcome}
            self.assertEqual(validate_block(block), [], f"{outcome} should be valid")


class TestParseTrajectoryMd(unittest.TestCase):
    """Test Markdown trajectory parsing."""

    def test_basic_parse(self):
        text = """\
[TRAJ-20260221-001]
Task: Deploy v1.0.6
Date: 2026-02-21
Outcome: SUCCESS
Reward: 1.0
"""
        block = parse_trajectory_md(text)
        self.assertEqual(block["_id"], "TRAJ-20260221-001")
        self.assertEqual(block["Task"], "Deploy v1.0.6")
        self.assertEqual(block["Outcome"], "SUCCESS")

    def test_parse_with_lists(self):
        text = """\
[TRAJ-20260221-002]
Task: Run benchmarks
Date: 2026-02-21
Outcome: PARTIAL
Lessons:
  - Check API quota first
  - Use Mistral as fallback
Steps:
  1. Setup workspace
  2. Run conv-0
  3. Compare results
"""
        block = parse_trajectory_md(text)
        self.assertEqual(len(block["Lessons"]), 2)
        self.assertEqual(len(block["Steps"]), 3)
        self.assertIn("Check API quota", block["Lessons"][0])

    def test_parse_empty(self):
        self.assertIsNone(parse_trajectory_md(""))

    def test_parse_non_trajectory(self):
        self.assertIsNone(parse_trajectory_md("[D-20260221-001]\nStatement: hello"))


class TestFormatTrajectoryMd(unittest.TestCase):
    """Test trajectory block formatting."""

    def test_roundtrip(self):
        block = {
            "_id": "TRAJ-20260221-001",
            "Task": "Deploy v1.0.6",
            "Date": "2026-02-21",
            "Outcome": "SUCCESS",
            "Reward": "1.0",
            "Lessons": ["Always run tests", "Check staging first"],
            "Steps": ["Pull latest", "Run pytest", "Build docker"],
        }
        md = format_trajectory_md(block)
        parsed = parse_trajectory_md(md)
        self.assertEqual(parsed["_id"], block["_id"])
        self.assertEqual(parsed["Task"], block["Task"])
        self.assertEqual(parsed["Outcome"], block["Outcome"])
        self.assertEqual(len(parsed["Lessons"]), 2)
        self.assertEqual(len(parsed["Steps"]), 3)


class TestComputeSimilarity(unittest.TestCase):
    """Test trajectory similarity computation."""

    def test_identical_trajectories(self):
        t = {"Task": "deploy production", "Tools": "git, docker", "Outcome": "SUCCESS"}
        self.assertGreater(compute_similarity(t, t), 0.8)

    def test_different_trajectories(self):
        a = {"Task": "deploy production", "Tools": "git, docker", "Outcome": "SUCCESS"}
        b = {"Task": "write unit tests", "Tools": "pytest, mock", "Outcome": "FAILURE"}
        self.assertLess(compute_similarity(a, b), 0.3)

    def test_similar_task_different_outcome(self):
        a = {"Task": "deploy production", "Tools": "git, docker", "Outcome": "SUCCESS"}
        b = {"Task": "deploy staging", "Tools": "git, docker", "Outcome": "FAILURE"}
        sim = compute_similarity(a, b)
        self.assertGreater(sim, 0.3)  # Tools match, partial task match
        self.assertLess(sim, 0.9)  # But outcomes differ

    def test_empty_blocks(self):
        self.assertIsInstance(compute_similarity({}, {}), float)

    def test_recency_decay_recent(self):
        """Trajectory from today should score higher than same trajectory from 60 days ago."""
        ref = date(2026, 2, 21)
        a = {"Task": "deploy production", "Tools": "git, docker", "Outcome": "SUCCESS"}
        b_today = {
            "Task": "deploy production",
            "Tools": "git, docker",
            "Outcome": "SUCCESS",
            "Date": ref.strftime("%Y-%m-%d"),
        }
        b_old = {
            "Task": "deploy production",
            "Tools": "git, docker",
            "Outcome": "SUCCESS",
            "Date": (ref - timedelta(days=60)).strftime("%Y-%m-%d"),
        }
        sim_today = compute_similarity(a, b_today, reference_date=ref)
        sim_old = compute_similarity(a, b_old, reference_date=ref)
        self.assertGreater(sim_today, sim_old)

    def test_recency_decay_halflife(self):
        """Trajectory from exactly 30 days ago should have ~50% of today's score."""
        ref = date(2026, 2, 21)
        a = {"Task": "deploy production", "Tools": "git, docker", "Outcome": "SUCCESS"}
        b_today = {
            "Task": "deploy production",
            "Tools": "git, docker",
            "Outcome": "SUCCESS",
            "Date": ref.strftime("%Y-%m-%d"),
        }
        b_halflife = {
            "Task": "deploy production",
            "Tools": "git, docker",
            "Outcome": "SUCCESS",
            "Date": (ref - timedelta(days=30)).strftime("%Y-%m-%d"),
        }
        sim_today = compute_similarity(a, b_today, reference_date=ref)
        sim_half = compute_similarity(a, b_halflife, reference_date=ref)
        ratio = sim_half / sim_today
        self.assertAlmostEqual(ratio, 0.5, places=2)

    def test_recency_decay_missing_date(self):
        """Trajectory without Date field should get full score (no penalty)."""
        ref = date(2026, 2, 21)
        a = {"Task": "deploy production", "Tools": "git, docker", "Outcome": "SUCCESS"}
        b_with_date = {
            "Task": "deploy production",
            "Tools": "git, docker",
            "Outcome": "SUCCESS",
            "Date": ref.strftime("%Y-%m-%d"),
        }
        b_no_date = {
            "Task": "deploy production",
            "Tools": "git, docker",
            "Outcome": "SUCCESS",
        }
        sim_dated = compute_similarity(a, b_with_date, reference_date=ref)
        sim_no_date = compute_similarity(a, b_no_date, reference_date=ref)
        self.assertAlmostEqual(sim_dated, sim_no_date, places=5)

    def test_recency_decay_custom_reference(self):
        """Custom reference_date should shift the decay calculation."""
        a = {"Task": "deploy production", "Tools": "git, docker", "Outcome": "SUCCESS"}
        b = {
            "Task": "deploy production",
            "Tools": "git, docker",
            "Outcome": "SUCCESS",
            "Date": "2026-01-01",
        }
        # With a reference date close to the trajectory date, decay should be small
        sim_close = compute_similarity(a, b, reference_date=date(2026, 1, 2))
        # With a reference date far from the trajectory date, decay should be large
        sim_far = compute_similarity(a, b, reference_date=date(2026, 7, 1))
        self.assertGreater(sim_close, sim_far)


if __name__ == "__main__":
    unittest.main()
