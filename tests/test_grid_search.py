#!/usr/bin/env python3
"""Tests for benchmarks/grid_search.py — grid generation and utility functions."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from grid_search import (
    PRIMARY_FIELDS,
    _weights_label,
    generate_full_grid,
    generate_grid,
)


class TestGenerateGrid(unittest.TestCase):
    """Test one-at-a-time grid generation."""

    def setUp(self):
        self.baseline = {
            "Statement": 3.0, "Title": 2.5, "Name": 2.0,
            "Summary": 1.5, "Context": 0.5,
            "Description": 1.2, "Tags": 0.8,
        }

    def test_grid_includes_baseline(self):
        combos = generate_grid(self.baseline)
        self.assertEqual(combos[0], self.baseline)

    def test_grid_count_one_at_a_time(self):
        """One-at-a-time: 5 fields * 2 non-baseline steps + 1 baseline = 11."""
        combos = generate_grid(self.baseline)
        self.assertEqual(len(combos), 11)

    def test_grid_varies_one_field(self):
        """Each non-baseline combo should differ from baseline in exactly one primary field."""
        combos = generate_grid(self.baseline)
        for combo in combos[1:]:  # skip baseline
            diffs = [
                f for f in PRIMARY_FIELDS
                if f in combo and f in self.baseline and combo[f] != self.baseline[f]
            ]
            self.assertEqual(len(diffs), 1, f"Expected 1 diff, got {diffs}")

    def test_grid_multiplier_values(self):
        """Check that varied values are 50% and 150% of baseline."""
        combos = generate_grid(self.baseline)
        statement_combos = [
            c for c in combos
            if c.get("Statement") != self.baseline["Statement"]
        ]
        statement_vals = sorted([c["Statement"] for c in statement_combos])
        self.assertEqual(statement_vals, [1.5, 4.5])

    def test_grid_custom_fields(self):
        combos = generate_grid(self.baseline, fields=["Statement", "Title"])
        # 2 fields * 2 steps + 1 baseline = 5
        self.assertEqual(len(combos), 5)

    def test_grid_custom_steps(self):
        combos = generate_grid(self.baseline, steps=[0.25, 0.5, 1.0, 1.5, 2.0])
        # 5 fields * 4 non-baseline steps + 1 baseline = 21
        self.assertEqual(len(combos), 21)

    def test_grid_missing_field_ignored(self):
        combos = generate_grid(self.baseline, fields=["Statement", "NonExistent"])
        # Only Statement is valid: 1 * 2 + 1 = 3
        self.assertEqual(len(combos), 3)

    def test_grid_preserves_non_primary_fields(self):
        """Non-primary fields should remain at baseline in all combos."""
        combos = generate_grid(self.baseline)
        for combo in combos:
            self.assertEqual(combo.get("Description"), 1.2)
            self.assertEqual(combo.get("Tags"), 0.8)


class TestGenerateFullGrid(unittest.TestCase):
    """Test full cartesian product grid generation."""

    def setUp(self):
        self.baseline = {
            "Statement": 3.0, "Title": 2.5, "Name": 2.0,
            "Summary": 1.5, "Context": 0.5,
        }

    def test_full_grid_count(self):
        """Full grid: 3^5 = 243 combinations."""
        combos = generate_full_grid(self.baseline)
        self.assertEqual(len(combos), 243)

    def test_full_grid_subset(self):
        """Full grid with 2 fields: 3^2 = 9."""
        combos = generate_full_grid(
            self.baseline, fields=["Statement", "Title"]
        )
        self.assertEqual(len(combos), 9)

    def test_full_grid_includes_baseline(self):
        """Baseline weights should appear in the full grid."""
        combos = generate_full_grid(self.baseline)
        baseline_found = any(
            all(c.get(f) == self.baseline[f] for f in PRIMARY_FIELDS)
            for c in combos
        )
        self.assertTrue(baseline_found)


class TestWeightsLabel(unittest.TestCase):
    """Test human-readable weight labeling."""

    def setUp(self):
        self.baseline = {
            "Statement": 3.0, "Title": 2.5, "Name": 2.0,
            "Summary": 1.5, "Context": 0.5,
        }

    def test_baseline_label(self):
        label = _weights_label(self.baseline, self.baseline)
        self.assertEqual(label, "baseline")

    def test_single_diff_label(self):
        variant = dict(self.baseline)
        variant["Statement"] = 4.5
        label = _weights_label(variant, self.baseline)
        self.assertEqual(label, "Statement=4.5")

    def test_multi_diff_label(self):
        variant = dict(self.baseline)
        variant["Statement"] = 1.5
        variant["Context"] = 0.75
        label = _weights_label(variant, self.baseline)
        self.assertIn("Statement=1.5", label)
        self.assertIn("Context=0.75", label)


if __name__ == "__main__":
    unittest.main()
