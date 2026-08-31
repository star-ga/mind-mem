"""Apply-engine gates that were reporting a verdict they had not earned.

Four defects, one theme — a check whose answer never reached the caller:

* ``check_deferred_cooldown`` compared an aware ``Created:`` (any value that
  spelled UTC with the documented ``Z``) against a naive *local* cutoff. The
  ``TypeError`` that raises was swallowed by the surrounding handler and the
  cooldown reported "no conflict" for every proposal that used the wire format
  the rest of the module documents.
* ``check_preconditions`` gated the intel scan on an ``intel_scan.py`` file
  sitting next to the module while the work it gates is
  ``python -m mind_mem.intel_scan`` — a different question, so the
  critical-findings scan could be skipped on a workspace where it would have
  run perfectly.
* ``rollback`` ran the (subprocess-heavy) post-rollback precondition check and
  then discarded its verdict, and skipped the whole governance-reconciliation
  block in silence when the snapshot carried no receipt.
* ``write_receipt`` dated receipts in local time while the same apply recorded
  a UTC ``last_apply_ts``.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone

import pytest

from mind_mem import apply_engine
from mind_mem.apply_engine import (
    check_deferred_cooldown,
    check_preconditions,
    create_snapshot,
    rollback,
    write_receipt,
)

# tzset() is POSIX-only; the Windows CI rows cannot rebind local time.
requires_tzset = pytest.mark.skipif(
    not hasattr(time, "tzset"),
    reason="time.tzset() is unavailable on this platform",
)

TARGET = "D-20260829-001"


@pytest.fixture
def restore_timezone():
    """Put the process timezone back however the test exits."""
    original = os.environ.get("TZ")
    yield
    if hasattr(time, "tzset"):
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        time.tzset()


def _workspace_with_rejected(ws: str, created: str) -> None:
    """A workspace holding one rejected proposal against ``TARGET``."""
    os.makedirs(os.path.join(ws, "intelligence", "proposed"), exist_ok=True)
    os.makedirs(os.path.join(ws, "memory"), exist_ok=True)
    with open(os.path.join(ws, "memory", "intel-state.json"), "w", encoding="utf-8") as fh:
        json.dump({}, fh)
    with open(os.path.join(ws, "intelligence", "proposed", "DECISIONS_PROPOSED.md"), "w", encoding="utf-8") as fh:
        fh.write(f"[P-20260829-001]\nProposalId: P-20260829-001\nType: edit\nTargetBlock: {TARGET}\nStatus: rejected\nCreated: {created}\n")


def _cooldown_for(created: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
        _workspace_with_rejected(ws, created)
        return check_deferred_cooldown(ws, {"TargetBlock": TARGET})


class TestDeferredCooldownReadsUtcTimestamps:
    """The cooldown must not be defeated by the documented ``Z`` designator."""

    def test_z_suffixed_created_is_still_inside_the_cooldown(self) -> None:
        """``Created: ...Z`` parses aware; the cutoff must be aware too.

        Before the fix this returned ``(True, "No cooldown conflict")``: the
        aware-vs-naive comparison raised TypeError into the ``except`` below
        it and the loop fell through.
        """
        fresh = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ok, reason = _cooldown_for(fresh)
        assert not ok, f"a rejected proposal stamped in UTC evaded the cooldown: {reason}"
        assert "cooldown" in reason

    def test_offsetless_created_behaves_the_same_as_the_z_form(self) -> None:
        """Both spellings of the same instant must reach the same verdict."""
        now = datetime.now(timezone.utc)
        with_z = _cooldown_for(now.strftime("%Y-%m-%dT%H:%M:%SZ"))
        without_z = _cooldown_for(now.strftime("%Y-%m-%dT%H:%M:%S"))
        assert with_z[0] == without_z[0] is False

    def test_explicit_offset_is_converted_not_rejected(self) -> None:
        # The offset is written as ``-11:00``, not via ``%z``. strftime emits
        # the colon-less ``-1100`` form, which datetime.fromisoformat rejects on
        # older interpreters and which strftime itself renders differently
        # across platforms -- so the test was asserting the host's strftime
        # rather than the product's timestamp handling, and failed on macOS and
        # Windows while passing on Linux. The property under test is that an
        # EXPLICIT offset is converted rather than rejected; spelling the offset
        # out tests exactly that, everywhere.
        moment = datetime.now(timezone.utc) - timedelta(hours=11)
        local_form = moment.astimezone(timezone(timedelta(hours=-11)))
        stamp = local_form.strftime("%Y-%m-%dT%H:%M:%S") + "-11:00"
        ok, _ = _cooldown_for(stamp)
        assert not ok

    def test_expired_cooldown_still_clears(self) -> None:
        """The fix must not turn the cooldown into an unconditional block."""
        old = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        ok, reason = _cooldown_for(old)
        assert ok, reason

    def test_unparseable_created_is_tolerated(self) -> None:
        ok, _ = _cooldown_for("not-a-timestamp")
        assert ok


class TestIntelScanGateProbesWhatItRuns:
    """The gate must ask the question the subprocess actually needs answered."""

    @staticmethod
    def _stub_subprocess(monkeypatch) -> list[list[str]]:
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            if "mind_mem.validate_py" in cmd:
                stdout = "TOTAL 0 issues\n"
            else:
                stdout = "TOTAL 0 critical\n"
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

        monkeypatch.setattr(apply_engine.subprocess, "run", fake_run)
        return calls

    def test_missing_source_file_does_not_skip_an_importable_module(self, monkeypatch, tmp_path) -> None:
        """No ``intel_scan.py`` on disk, module still importable → it must run.

        Before the fix the gate probed ``os.path.isfile(<pkg>/intel_scan.py)``
        and appended ``intel_scan: SKIP (script not found)``, returning ``True``
        with the critical-findings scan never run.
        """
        calls = self._stub_subprocess(monkeypatch)
        pkg_script = os.path.join(os.path.dirname(apply_engine.__file__), "intel_scan.py")
        real_isfile = os.path.isfile
        monkeypatch.setattr(
            apply_engine.os.path,
            "isfile",
            lambda path: False if os.path.abspath(str(path)) == pkg_script else real_isfile(path),
        )

        ok, report = check_preconditions(str(tmp_path))

        assert ok, report
        assert not any("SKIP" in line for line in report), report
        assert any("intel_scan: PASS" in line for line in report), report
        assert ["-m", "mind_mem.intel_scan"] == [c for c in calls[-1] if c in ("-m", "mind_mem.intel_scan")]

    def test_gate_skips_when_the_module_is_not_importable(self, monkeypatch, tmp_path) -> None:
        """The SKIP path is kept — it now keys on the real precondition."""
        self._stub_subprocess(monkeypatch)
        monkeypatch.setattr(apply_engine.importlib.util, "find_spec", lambda name: None)

        ok, report = check_preconditions(str(tmp_path))

        assert ok
        assert any("intel_scan: SKIP (module not importable)" in line for line in report), report


def _snapshot_workspace(ws: str, receipt_ts: str, with_receipt: bool = True) -> str:
    from mind_mem.init_workspace import init

    init(ws)
    snap_dir = create_snapshot(ws, receipt_ts, files_touched=["decisions/DECISIONS.md"])
    if with_receipt:
        with open(os.path.join(snap_dir, "APPLY_RECEIPT.md"), "w", encoding="utf-8") as fh:
            fh.write(f"[AR-{receipt_ts}]\nProposal: P-20260829-002\nResult: applied\n")
    return snap_dir


class TestRollbackReportsWhatItFound:
    """A rollback must not print a clean banner over a check it failed."""

    def test_failing_postcheck_is_announced(self, monkeypatch, capsys) -> None:
        """Before the fix the FAIL verdict was bound and never read again."""
        monkeypatch.setattr(
            apply_engine,
            "check_preconditions",
            lambda ws: (False, ["validate: FAIL (TOTAL 3 issues)"]),
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            _snapshot_workspace(ws, "20260829-120000")
            ok = rollback(ws, "20260829-120000")
        out = capsys.readouterr()
        assert ok is True, "the restore itself succeeded, so the default contract still reports True"
        assert "post-rollback validation FAILED" in out.out
        assert "POST-ROLLBACK VALIDATION FAILED" in out.out
        assert "post-rollback validation FAILED" in out.err

    def test_strict_makes_a_failing_postcheck_a_failing_rollback(self, monkeypatch) -> None:
        """``strict=True`` (what the CLI passes) turns the verdict into the result."""
        monkeypatch.setattr(
            apply_engine,
            "check_preconditions",
            lambda ws: (False, ["validate: FAIL (TOTAL 3 issues)"]),
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            _snapshot_workspace(ws, "20260829-120001")
            assert rollback(ws, "20260829-120001", strict=True) is False

    def test_strict_still_succeeds_on_a_clean_postcheck(self, monkeypatch) -> None:
        monkeypatch.setattr(
            apply_engine,
            "check_preconditions",
            lambda ws: (True, ["validate: PASS (TOTAL 0 issues)"]),
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            _snapshot_workspace(ws, "20260829-120002")
            assert rollback(ws, "20260829-120002", strict=True) is True

    def test_receiptless_snapshot_says_nothing_was_reconciled(self, monkeypatch, capsys) -> None:
        """A snapshot with no receipt is reachable (apply aborts before writing one).

        Before the fix the entire reconciliation block was skipped in silence
        and the run printed the same banner as a fully reconciled rollback.
        """
        monkeypatch.setattr(
            apply_engine,
            "check_preconditions",
            lambda ws: (True, ["validate: PASS (TOTAL 0 issues)"]),
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            _snapshot_workspace(ws, "20260829-120003", with_receipt=False)
            ok = rollback(ws, "20260829-120003")
        out = capsys.readouterr()
        assert ok is True
        assert "no APPLY_RECEIPT.md" in out.out
        assert "no proposal status was reconciled" in out.err


class TestReceiptDateIsUtc:
    """The receipt's ``Date:`` is a durable audit field, so it is UTC."""

    @staticmethod
    def _date_line(snap_dir: str) -> str:
        with open(os.path.join(snap_dir, "APPLY_RECEIPT.md"), encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("Date:"):
                    return line.split(":", 1)[1].strip()
        raise AssertionError("no Date: line in receipt")

    @requires_tzset
    def test_receipt_date_is_identical_across_timezones(self, restore_timezone, tmp_path) -> None:
        """Before the fix a host at UTC+14 dated the receipt a day ahead."""
        seen: set[str] = set()
        for index, tz_name in enumerate(("Pacific/Niue", "UTC", "Pacific/Kiritimati")):
            os.environ["TZ"] = tz_name
            time.tzset()
            snap_dir = tmp_path / f"snap{index}"
            snap_dir.mkdir()
            write_receipt(str(snap_dir), {"ProposalId": "P-1", "Ops": []}, "20260829-120004", [])
            seen.add(self._date_line(str(snap_dir)))
        assert len(seen) == 1, f"receipt Date: is timezone-dependent: {seen}"
        assert seen == {datetime.now(timezone.utc).strftime("%Y-%m-%d")}
