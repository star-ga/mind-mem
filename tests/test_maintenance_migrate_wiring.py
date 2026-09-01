"""``maintenance_migrate`` is actually reachable — from apply and from ``mm``.

``maintenance_migrate`` has, since v3.2.0, documented itself as
"invoked automatically by ``apply_engine.apply_proposal`` on first-run
detection of the old layout". That call site was never written. The
module was correct, tested and dead: a workspace created before the
v3.2.0 §2.2 snapshot-scope split kept its behavioural state flat in
``maintenance/``, where it is neither a ``SNAPSHOT_DIR`` (only
``maintenance/tracked`` is) nor a member of ``SNAPSHOT_EXCLUDE_DIRS``,
so it escaped both snapshot and rollback — precisely the atomicity hole
§2.2 was written to close, still open for every workspace that needed
the fix.

These tests pin the two call sites that close it, and pin that the gate
in front of them is real:

* ``apply_proposal`` splits a flat ``maintenance/`` on its first run,
  under the workspace lock and before the snapshot;
* a second apply moves zero files;
* with ``v4.maintenance_layout`` unset — the default — the flat layout
  is left byte-for-byte alone and no subdirectory is created;
* ``mm migrate --maintenance`` performs the same split, and refuses when
  the flag is off.

Each of the first three fails if the ``_migrate_maintenance_layout``
call in ``apply_proposal`` is deleted or its gate inverted.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from mind_mem.apply_engine import apply_proposal, compute_fingerprint
from mind_mem.init_workspace import init

# Legacy flat layout: one behavioural state file (must become tracked/,
# i.e. inside the snapshot) and one append-only report (must become
# append-only/, i.e. deliberately outside it).
_STATE_FILE = "dedup-state.json"
_REPORT_FILE = "compaction-2026-04-01.log"
_STATE_BODY = '{"hash": "pre-apply", "seen": ["block-a"]}'
_REPORT_BODY = "compaction pass 1\n"


def _enable_flag(ws: str) -> None:
    """Turn ``v4.maintenance_layout`` ON in the workspace's own config."""
    config_path = os.path.join(ws, "mind-mem.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    config.setdefault("v4", {})["maintenance_layout"] = {"enabled": True}
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def _write_flat_maintenance(ws: str) -> None:
    """Recreate the pre-v3.2.0 layout: state files loose in maintenance/."""
    base = os.path.join(ws, "maintenance")
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, _STATE_FILE), "w", encoding="utf-8") as handle:
        handle.write(_STATE_BODY)
    with open(os.path.join(base, _REPORT_FILE), "w", encoding="utf-8") as handle:
        handle.write(_REPORT_BODY)


def _stage_proposal(ws: str, proposal_id: str, decision_id: str) -> None:
    """Write a decision block plus a staged, valid proposal that edits it."""
    decisions = os.path.join(ws, "decisions", "DECISIONS.md")
    with open(decisions, "a", encoding="utf-8") as handle:
        handle.write(
            f"\n[{decision_id}]\ntype: decision\nStatus: active\n"
            f"Statement: Placeholder for {proposal_id}.\n"
            "Rationale: exercises the apply pipeline end to end.\n"
            "History:\n- created 2026-04-20 Status: active\n\n"
        )

    ops = [
        {
            "op": "set_status",
            "file": "decisions/DECISIONS.md",
            "target": decision_id,
            "status": "superseded",
        }
    ]
    fingerprint = compute_fingerprint({"ProposalId": proposal_id, "Type": "edit", "TargetBlock": decision_id, "Ops": ops})

    proposed_dir = os.path.join(ws, "intelligence", "proposed")
    os.makedirs(proposed_dir, exist_ok=True)
    with open(os.path.join(proposed_dir, "DECISIONS_PROPOSED.md"), "a", encoding="utf-8") as handle:
        handle.write(
            f"\n[{proposal_id}]\nProposalId: {proposal_id}\n"
            f"Type: edit\nTargetBlock: {decision_id}\n"
            f"Risk: low\nStatus: staged\nEvidence: wiring test\n"
            f"Rollback: revert the status change\nFingerprint: {fingerprint}\n"
            f"Ops:\n- op: set_status\n  file: decisions/DECISIONS.md\n"
            f"  target: {decision_id}\n  status: superseded\n"
        )


def _clear_no_touch_window(ws: str) -> None:
    """Drop ``last_apply_ts`` so a second apply is not cooldown-blocked.

    The no-touch window is a real governance rule about operator pacing,
    not about layout, and a test that applies twice back to back is not
    the thing it exists to stop. Clearing the recorded instant is how the
    rule sees a fresh workspace; the rule itself is left switched on.
    """
    state_path = os.path.join(ws, "memory", "intel-state.json")
    with open(state_path, encoding="utf-8") as handle:
        state = json.load(handle)
    state.pop("last_apply_ts", None)
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle)


@pytest.fixture
def ws(tmp_path: Path) -> str:
    """A real initialised workspace in ``propose`` mode with a flat maintenance/."""
    workspace = str(tmp_path / "ws")
    init(workspace)

    state_path = os.path.join(workspace, "memory", "intel-state.json")
    with open(state_path, encoding="utf-8") as handle:
        state = json.load(handle)
    state["governance_mode"] = "propose"
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle)

    _write_flat_maintenance(workspace)
    return workspace


def _maintenance_tree(ws: str) -> dict[str, str]:
    """Every file under ``maintenance/``, mapped relative-path → content."""
    base = os.path.join(ws, "maintenance")
    tree: dict[str, str] = {}
    for root, _dirs, files in os.walk(base):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, base).replace(os.sep, "/")
            with open(path, encoding="utf-8", errors="replace") as handle:
                tree[rel] = handle.read()
    return tree


# ``check_preconditions`` shells out to the validator, which reports
# issues on a freshly-scaffolded workspace for reasons unrelated to
# layout (empty template corpus). Every other apply-pipeline gate —
# mode, validation, dedup, backlog, cooldown, no-touch, contradiction,
# governance admission, ops, rollback — runs for real.
_PRECONDITIONS_PASS = patch(
    "mind_mem.apply_engine.check_preconditions",
    return_value=(True, ["validate: PASS (TOTAL 0 issues)"]),
)


class TestApplyMigratesFlatLayout:
    def test_apply_splits_flat_maintenance_into_namespaced_layout(self, ws: str) -> None:
        """A governed apply performs the v3.2.0 §2.2 split on first run.

        The state file must land under ``maintenance/tracked/`` — a
        SNAPSHOT_DIR — because that is the whole point: a dedup hash
        written mid-apply has to be captured and rolled back with the
        corpus. The report must land under ``maintenance/append-only/``,
        which snapshot and restore both skip, so a rollback does not
        discard observability written during the failed apply.
        """
        _enable_flag(ws)
        _stage_proposal(ws, "P-20260420-001", "D-20260420-001")

        with _PRECONDITIONS_PASS:
            ok, msg = apply_proposal(ws, "P-20260420-001", dry_run=False)
        assert ok, msg

        base = Path(ws) / "maintenance"
        assert (base / "tracked" / _STATE_FILE).read_text(encoding="utf-8") == _STATE_BODY
        assert (base / "append-only" / _REPORT_FILE).read_text(encoding="utf-8") == _REPORT_BODY
        assert not (base / _STATE_FILE).exists(), "state file left flat — the migration did not run"
        assert not (base / _REPORT_FILE).exists(), "report left flat — the migration did not run"

    def test_shipped_maintenance_scripts_are_not_relocated(self, ws: str) -> None:
        """init's own tooling stays flat — it is code, not corpus state.

        ``init_workspace.MAINTENANCE_SCRIPTS`` copies ``validate.sh`` and
        a set of ``*.py`` helpers into ``maintenance/``. Their paths are
        quoted to operators and pinned by other tests, and the classifier's
        unknown-default would otherwise sweep every one of them into
        ``tracked/``. The migration must leave them where they are.
        """
        _enable_flag(ws)
        _stage_proposal(ws, "P-20260420-002", "D-20260420-002")

        base = Path(ws) / "maintenance"
        scripts_before = sorted(p.name for p in base.iterdir() if p.is_file() and p.suffix in {".py", ".sh"})
        assert scripts_before, "fixture precondition: init ships maintenance scripts"

        with _PRECONDITIONS_PASS:
            ok, msg = apply_proposal(ws, "P-20260420-002", dry_run=False)
        assert ok, msg

        # The migration really ran (otherwise "nothing moved" proves nothing)…
        assert (base / "tracked" / _STATE_FILE).exists()
        # …and it moved none of the tooling.
        scripts_after = sorted(p.name for p in base.iterdir() if p.is_file() and p.suffix in {".py", ".sh"})
        assert scripts_after == scripts_before
        assert not (base / "tracked" / "validate.sh").exists()
        assert not list((base / "tracked").glob("*.py"))

    def test_second_apply_moves_zero_files(self, ws: str) -> None:
        """The migration is one-shot: run two, the second is a no-op.

        Compares the whole ``maintenance/`` tree — paths and contents —
        across the second apply. Anything the second run moved, renamed
        or duplicated (the collision path appends ``.1``) shows up here.
        """
        _enable_flag(ws)
        _stage_proposal(ws, "P-20260420-003", "D-20260420-003")
        _stage_proposal(ws, "P-20260420-004", "D-20260420-004")

        with _PRECONDITIONS_PASS:
            ok, msg = apply_proposal(ws, "P-20260420-003", dry_run=False)
            assert ok, msg
            after_first = _maintenance_tree(ws)

            _clear_no_touch_window(ws)
            ok, msg = apply_proposal(ws, "P-20260420-004", dry_run=False)
            assert ok, msg
            after_second = _maintenance_tree(ws)

        assert after_second == after_first
        assert f"tracked/{_STATE_FILE}" in after_first  # the first run really did migrate


class TestFlagOffIsUnchanged:
    def test_flag_off_leaves_the_flat_layout_untouched(self, ws: str) -> None:
        """Default config: apply behaves exactly as it did before the wiring.

        No subdirectory is created, nothing moves, and the flat files keep
        their bytes. This is the byte-identity half of the gate — if the
        call site ever stops consulting the flag, this is what fails.
        """
        _stage_proposal(ws, "P-20260420-005", "D-20260420-005")
        before = _maintenance_tree(ws)

        with _PRECONDITIONS_PASS:
            ok, msg = apply_proposal(ws, "P-20260420-005", dry_run=False)
        assert ok, msg

        assert _maintenance_tree(ws) == before
        base = Path(ws) / "maintenance"
        assert not (base / "tracked").exists()
        assert not (base / "append-only").exists()
        assert (base / _STATE_FILE).read_text(encoding="utf-8") == _STATE_BODY

    def test_flag_off_reads_nothing_from_the_maintenance_directory(self, ws: str) -> None:
        """Flag OFF short-circuits before any filesystem walk.

        ``migrate_if_enabled`` must not even ``listdir`` the directory —
        an "off" path that still walks the tree is one refactor away from
        being an "off" path that moves a file.
        """
        from mind_mem import maintenance_migrate

        with patch.object(maintenance_migrate, "migrate_maintenance") as never:
            assert maintenance_migrate.migrate_if_enabled(ws) is None
        never.assert_not_called()


class TestCliAlias:
    def _run(self, ws: str, *argv: str) -> subprocess.CompletedProcess[str]:
        env = {**os.environ, "MIND_MEM_WORKSPACE": ws, "PYTHONIOENCODING": "utf-8"}
        return subprocess.run(
            [sys.executable, "-m", "mind_mem.mm_cli", *argv],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env=env,
        )

    def test_mm_migrate_maintenance_performs_the_split(self, ws: str) -> None:
        _enable_flag(ws)
        result = self._run(ws, "migrate", "--maintenance")

        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["migrated"] is True
        assert payload["already_migrated"] is False
        assert payload["moved"] == {"tracked": 1, "append-only": 1}

        base = Path(ws) / "maintenance"
        assert (base / "tracked" / _STATE_FILE).read_text(encoding="utf-8") == _STATE_BODY
        assert (base / "append-only" / _REPORT_FILE).read_text(encoding="utf-8") == _REPORT_BODY

    def test_mm_migrate_maintenance_is_refused_when_the_flag_is_off(self, ws: str) -> None:
        before = _maintenance_tree(ws)
        result = self._run(ws, "migrate", "--maintenance")

        assert result.returncode == 1
        payload = json.loads(result.stdout)
        assert payload["migrated"] is False
        assert "maintenance_layout" in payload["error"]
        assert _maintenance_tree(ws) == before
        assert not (Path(ws) / "maintenance" / "tracked").exists()

    def test_bare_mm_migrate_names_the_missing_flag(self, ws: str) -> None:
        """``mm migrate`` with no target is a usage error, not a silent no-op."""
        result = self._run(ws, "migrate")
        assert result.returncode != 0
        assert "--maintenance" in result.stderr
