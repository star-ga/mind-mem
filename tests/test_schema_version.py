#!/usr/bin/env python3
"""Tests for schema_version.py — zero external deps (stdlib unittest)."""

import json
import os
import shutil
import tempfile
import unittest

from mind_mem.schema_version import (
    CURRENT_SCHEMA_VERSION,
    check_migration_needed,
    get_workspace_version,
    migrate_workspace,
)


class TestGetWorkspaceVersion(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_missing_config_defaults_to_1_0_0(self):
        version = get_workspace_version(self.td)
        self.assertEqual(version, "1.0.0")

    def test_reads_version_field(self):
        config_path = os.path.join(self.td, "mind-mem.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"version": "1.0.0"}, f)
        version = get_workspace_version(self.td)
        self.assertEqual(version, "1.0.0")

    def test_schema_version_takes_precedence(self):
        config_path = os.path.join(self.td, "mind-mem.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"version": "1.0.0", "schema_version": "2.0.0"}, f)
        version = get_workspace_version(self.td)
        self.assertEqual(version, "2.0.0")

    def test_corrupt_json_defaults_to_1_0_0(self):
        config_path = os.path.join(self.td, "mind-mem.json")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("not valid json {{{")
        version = get_workspace_version(self.td)
        self.assertEqual(version, "1.0.0")


class TestCheckMigrationNeeded(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_no_migration_needed_when_current(self):
        config_path = os.path.join(self.td, "mind-mem.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"schema_version": CURRENT_SCHEMA_VERSION}, f)
        steps = check_migration_needed(self.td)
        self.assertEqual(steps, [])

    def test_migration_needed_for_v1(self):
        config_path = os.path.join(self.td, "mind-mem.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"version": "1.0.0"}, f)
        steps = check_migration_needed(self.td)
        self.assertGreater(len(steps), 0)
        self.assertIn("1.0.0 -> 2.0.0", steps[0])

    def test_migration_needed_for_missing_config(self):
        steps = check_migration_needed(self.td)
        self.assertGreater(len(steps), 0)


class TestMigrateWorkspace(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _create_v1_workspace(self):
        """Create a minimal v1 workspace."""
        config_path = os.path.join(self.td, "mind-mem.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"version": "1.0.0", "workspace_path": "."}, f)
        os.makedirs(os.path.join(self.td, "decisions"), exist_ok=True)
        os.makedirs(os.path.join(self.td, "intelligence"), exist_ok=True)

    def test_migration_creates_proposed_dir(self):
        self._create_v1_workspace()
        result = migrate_workspace(self.td)
        self.assertTrue(result["migrated"])
        self.assertTrue(os.path.isdir(os.path.join(self.td, "intelligence", "proposed")))

    def test_migration_creates_shared_dir(self):
        self._create_v1_workspace()
        result = migrate_workspace(self.td)
        self.assertTrue(result["migrated"])
        self.assertTrue(os.path.isdir(os.path.join(self.td, "shared")))

    def test_migration_adds_schema_version(self):
        self._create_v1_workspace()
        migrate_workspace(self.td)
        config_path = os.path.join(self.td, "mind-mem.json")
        with open(config_path) as f:
            config = json.load(f)
        self.assertEqual(config["schema_version"], CURRENT_SCHEMA_VERSION)

    def test_migration_preserves_existing_config(self):
        self._create_v1_workspace()
        migrate_workspace(self.td)
        config_path = os.path.join(self.td, "mind-mem.json")
        with open(config_path) as f:
            config = json.load(f)
        self.assertEqual(config["version"], "1.0.0")
        self.assertEqual(config["workspace_path"], ".")

    def test_migration_returns_correct_versions(self):
        self._create_v1_workspace()
        result = migrate_workspace(self.td)
        self.assertEqual(result["from_version"], "1.0.0")
        self.assertEqual(result["to_version"], CURRENT_SCHEMA_VERSION)
        self.assertGreater(len(result["steps"]), 0)

    def test_idempotent_migration(self):
        """Running migrate twice should be safe and produce same result."""
        self._create_v1_workspace()
        result1 = migrate_workspace(self.td)
        self.assertTrue(result1["migrated"])

        result2 = migrate_workspace(self.td)
        self.assertFalse(result2["migrated"])
        self.assertEqual(result2["steps"], [])

        # Verify workspace is still correct
        self.assertTrue(os.path.isdir(os.path.join(self.td, "intelligence", "proposed")))
        self.assertTrue(os.path.isdir(os.path.join(self.td, "shared")))
        config_path = os.path.join(self.td, "mind-mem.json")
        with open(config_path) as f:
            config = json.load(f)
        self.assertEqual(config["schema_version"], CURRENT_SCHEMA_VERSION)

    def test_skip_if_already_current(self):
        config_path = os.path.join(self.td, "mind-mem.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"schema_version": CURRENT_SCHEMA_VERSION}, f)
        result = migrate_workspace(self.td)
        self.assertFalse(result["migrated"])
        self.assertEqual(result["from_version"], CURRENT_SCHEMA_VERSION)

    def test_migration_with_no_config_file(self):
        """Missing mind-mem.json should default to 1.0.0 and migrate."""
        result = migrate_workspace(self.td)
        self.assertTrue(result["migrated"])
        self.assertEqual(result["from_version"], "1.0.0")
        # After migration, mind-mem.json should exist with schema_version
        config_path = os.path.join(self.td, "mind-mem.json")
        self.assertTrue(os.path.isfile(config_path))
        with open(config_path) as f:
            config = json.load(f)
        self.assertEqual(config["schema_version"], CURRENT_SCHEMA_VERSION)

    def test_migration_with_preexisting_proposed_dir(self):
        """If intelligence/proposed/ already exists, migration should not fail."""
        self._create_v1_workspace()
        os.makedirs(os.path.join(self.td, "intelligence", "proposed"), exist_ok=True)
        result = migrate_workspace(self.td)
        self.assertTrue(result["migrated"])
        self.assertTrue(os.path.isdir(os.path.join(self.td, "intelligence", "proposed")))


class TestMigrationFailureIsReported(unittest.TestCase):
    """A step that could not be applied must not be reported as applied.

    The v2.0 -> v2.1 step used to swallow OSError/JSONDecodeError, so a
    read-only workspace, a full disk or one corrupt byte of JSON left the
    legacy keys in place while the CLI printed "Migrated 2.0.0 -> 2.1.0".
    """

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.config_path = os.path.join(self.td, "mind-mem.json")
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump({"schema_version": "2.0.0", "self_correcting_mode": True}, f)
        os.makedirs(os.path.join(self.td, "memory"), exist_ok=True)
        self.state_path = os.path.join(self.td, "memory", "intel-state.json")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _corrupt_intel_state(self):
        with open(self.state_path, "w", encoding="utf-8") as f:
            f.write("{not json")

    def test_failed_step_is_not_counted_as_migrated(self):
        self._corrupt_intel_state()
        result = migrate_workspace(self.td)
        self.assertTrue(result["failed"])
        self.assertEqual(result["steps"], [])
        self.assertFalse(result["migrated"])

    def test_failed_step_does_not_claim_the_target_version(self):
        self._corrupt_intel_state()
        result = migrate_workspace(self.td)
        self.assertEqual(result["to_version"], "2.0.0")
        self.assertNotEqual(result["to_version"], CURRENT_SCHEMA_VERSION)

    def test_failed_step_leaves_the_workspace_on_the_old_version(self):
        """So the next run retries instead of skipping a half-done step."""
        self._corrupt_intel_state()
        migrate_workspace(self.td)
        with open(self.config_path, encoding="utf-8") as f:
            config = json.load(f)
        self.assertEqual(config["schema_version"], "2.0.0")
        self.assertIn("self_correcting_mode", config)

    def test_successful_step_still_reports_success(self):
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump({"self_correcting_mode": False}, f)
        result = migrate_workspace(self.td)
        self.assertEqual(result["failed"], [])
        self.assertTrue(result["migrated"])
        self.assertEqual(result["to_version"], CURRENT_SCHEMA_VERSION)
        with open(self.config_path, encoding="utf-8") as f:
            config = json.load(f)
        self.assertEqual(config["schema_version"], CURRENT_SCHEMA_VERSION)
        self.assertIn("governance_mode", config)
        with open(self.state_path, encoding="utf-8") as f:
            state = json.load(f)
        self.assertIn("governance_mode", state)


if __name__ == "__main__":
    unittest.main()
