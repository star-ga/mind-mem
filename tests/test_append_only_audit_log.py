# Copyright 2026 STARGA, Inc.
"""``mind_mem.append_only`` and the protection it is allowed to claim (T-007).

The audit trails (``memory/deleted_blocks.jsonl``,
``memory/decrypted_files.jsonl``, ``.mind-mem-audit/chain.jsonl``) are
hash-chained, which makes tampering *detectable*. The OS-level
append-only attribute -- ``chattr +a`` on Linux, ``chflags uappnd`` on
macOS/BSD -- makes an in-place rewrite *impossible* without root. That
is the second, independent layer described in
``docs/append-only-audit-logs.md``.

Setting the attribute requires ``CAP_LINUX_IMMUTABLE`` (root) on Linux
and a filesystem that supports it; tmpfs, NFS and SMB do not. Windows
has no equivalent at all. So the interesting behaviour is not the happy
path -- it is what the helper says when the flag could NOT be set.

The defect this pins is the one fixed in
``tests/test_mm_cli_keyfile_perms.py``: a caller that asserts a
protection the filesystem refused. A helper that returned "hardened"
after a failed ``chattr`` would be worse than no helper at all, because
an operator would stop applying the runbook. So every assertion below is
of the form *claimed == actual*, and the actual is measured by this
test's own probe rather than by the module under test.

These tests must pass as a NON-ROOT user, which is exactly the case that
cannot set the flag -- the honest-degradation path is asserted directly,
never skipped.
"""

from __future__ import annotations

import errno
import os
import shutil
import subprocess  # nosec B404 -- test asserts the helper does not shell out
from pathlib import Path

import pytest
from _platform_compat import append_only_settable_unprivileged, assert_owner_only, is_root

from mind_mem.append_only import (
    AppendOnlyUnavailable,
    append_only_mechanism,
    ensure_append_only,
    is_append_only,
)

_ENV = "MIND_MEM_AUDIT_APPEND_ONLY"


def _in_place_write_refused(path: Path) -> bool | None:
    """Ground truth, measured independently of the module under test.

    An append-only file refuses ``open(O_WRONLY)`` without ``O_APPEND``
    with EPERM, on Linux and on BSD alike -- that is the property the
    helper exists to obtain, so it is the property this test measures.
    Opening for writing changes nothing on disk; nothing is written.

    Returns True (refused, i.e. append-only), False (accepted, i.e. not
    append-only) or None (could not tell -- e.g. no write permission at
    all, or a read-only mount).
    """
    try:
        fd = os.open(str(path), os.O_WRONLY)
    except PermissionError as exc:
        import errno

        return True if exc.errno == errno.EPERM else None
    except OSError:
        return None
    os.close(fd)
    return False


def _fresh_log(tmp_path: Path) -> Path:
    path = tmp_path / "audit.jsonl"
    path.write_text('{"seq": 1}\n', encoding="utf-8")
    return path


@pytest.mark.unit
def test_the_status_never_claims_a_flag_the_filesystem_did_not_apply(tmp_path: Path) -> None:
    """The single property: ``enforced`` matches the file on disk.

    Holds as root on ext4 (flag applied), as a normal user (refused),
    and on tmpfs (unsupported) -- the assertion is the equivalence, not
    the outcome.
    """
    log = _fresh_log(tmp_path)
    status = ensure_append_only(str(log))
    actual = _in_place_write_refused(log)
    assert status.enforced is (actual is True), (
        f"claimed enforced={status.enforced} for a file whose in-place-write probe says {actual}: {status.detail}"
    )
    assert status.path == str(log)
    assert status.detail, "a status with no explanation is not a report"


@pytest.mark.unit
@pytest.mark.skipif(is_root(), reason="root can actually set the flag; this pins the unprivileged refusal")
def test_a_refused_flag_is_reported_not_claimed(tmp_path: Path) -> None:
    """Non-root: the call must degrade, name the refusal, and not raise.

    Only meaningful where an unprivileged owner is actually REFUSED. On macOS
    the owner may set `chflags uappnd` without privilege, so the call succeeds
    and ``enforced=True`` is correct -- asserting False there tests the host,
    not the product. The general claim-matches-the-filesystem invariant is
    pinned unconditionally by ``test_the_claim_matches_the_filesystem`` above,
    which is the assertion that actually matters on every platform.
    """
    if append_only_settable_unprivileged(tmp_path):
        pytest.skip("this host lets an unprivileged owner set the flag (e.g. macOS chflags uappnd)")
    log = _fresh_log(tmp_path)
    status = ensure_append_only(str(log))

    assert status.enforced is False
    assert "NOT append-only" in status.detail
    assert "WARNING" in status.detail
    # And the file really is still rewritable in place -- the claim and
    # the filesystem agree.
    assert _in_place_write_refused(log) is False
    log.write_text("rewritten\n", encoding="utf-8")
    assert log.read_text(encoding="utf-8") == "rewritten\n"


@pytest.mark.unit
def test_no_mechanism_is_a_clean_no_op_not_a_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Windows shape: no ``chflags``, no ``chattr``, no exception.

    Reproduced by removing both capabilities rather than by running on
    Windows, because the helper probes for them and must therefore be
    testable through the probe.
    """
    log = _fresh_log(tmp_path)
    monkeypatch.delattr(os, "chflags", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    assert append_only_mechanism() == "none"
    status = ensure_append_only(str(log))
    assert status.enforced is False
    assert status.mechanism == "none"
    assert "NOT append-only" in status.detail
    assert "WARNING" in status.detail
    # A no-op, not a mangled file.
    assert log.read_text(encoding="utf-8") == '{"seq": 1}\n'


@pytest.mark.unit
def test_require_mode_fails_closed_when_the_flag_cannot_be_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator who demands the flag must not silently run without it."""
    log = _fresh_log(tmp_path)
    monkeypatch.delattr(os, "chflags", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setenv(_ENV, "require")

    with pytest.raises(AppendOnlyUnavailable) as excinfo:
        ensure_append_only(str(log))
    assert "NOT append-only" in str(excinfo.value)


@pytest.mark.unit
def test_off_mode_makes_no_filesystem_call_and_still_says_the_file_is_unprotected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = _fresh_log(tmp_path)

    def _forbidden(*_a: object, **_k: object) -> None:
        raise AssertionError("off must not touch the filesystem")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.setattr(os, "chflags", _forbidden, raising=False)
    monkeypatch.setenv(_ENV, "off")

    status = ensure_append_only(str(log))
    assert status.enforced is False
    assert status.mechanism == "disabled"
    assert "NOT append-only" in status.detail
    assert _ENV in status.detail


@pytest.mark.unit
def test_an_unknown_mode_is_refused_not_silently_treated_as_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo in the env var must not quietly disable the hardening."""
    log = _fresh_log(tmp_path)
    monkeypatch.setenv(_ENV, "requrie")
    with pytest.raises(ValueError) as excinfo:
        ensure_append_only(str(log))
    assert _ENV in str(excinfo.value)
    assert "require" in str(excinfo.value)


@pytest.mark.unit
def test_a_missing_file_is_reported_and_only_created_when_asked(tmp_path: Path) -> None:
    missing = tmp_path / "not-yet.jsonl"

    status = ensure_append_only(str(missing))
    assert status.enforced is False
    assert not missing.exists(), "the helper must not create the file unless asked"
    assert "does not exist" in status.detail
    assert "NOT append-only" in status.detail

    created = ensure_append_only(str(missing), create=True)
    assert missing.exists()
    assert missing.read_bytes() == b""
    assert_owner_only(missing)
    assert created.enforced is (_in_place_write_refused(missing) is True)


@pytest.mark.unit
def test_an_already_flagged_file_is_reported_enforced_without_a_second_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file the operator already hardened (the runbook path) reads as enforced.

    Simulated by a filesystem that refuses the in-place open, which is
    what an append-only file does -- no root needed to assert that the
    helper believes the filesystem over its own tool's exit code.
    """
    log = _fresh_log(tmp_path)
    real_open = os.open

    def refusing_open(file, flags, *args, **kwargs):  # type: ignore[no-untyped-def]
        if os.fspath(file) == str(log) and not flags & os.O_APPEND:
            raise PermissionError(1, "Operation not permitted")
        return real_open(file, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", refusing_open)

    def _forbidden(*_a: object, **_k: object) -> None:
        raise AssertionError("an already-append-only file must not be re-flagged")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.setattr(os, "chflags", _forbidden, raising=False)

    status = ensure_append_only(str(log))
    assert status.enforced is True
    assert "NOT append-only" not in status.detail
    assert "WARNING" not in status.detail


@pytest.mark.unit
def test_a_tool_that_exits_zero_without_setting_the_flag_is_not_believed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exit code 0 is not evidence; the read-back is.

    A wrapper on PATH named ``chattr`` that does nothing, an overlay that
    accepts the ioctl and drops it, a stubbed ``chflags`` -- all exit
    clean. The helper must still report the file as unprotected.
    """
    log = _fresh_log(tmp_path)
    monkeypatch.delattr(os, "chflags", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + str(name))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(args=["chattr"], returncode=0, stdout="", stderr=""),
    )

    status = ensure_append_only(str(log))
    assert status.enforced is False
    assert "NOT append-only" in status.detail
    assert _in_place_write_refused(log) is False


@pytest.mark.unit
def test_is_append_only_is_tri_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ "Cannot tell" must not collapse into either "yes" or "no"."""
    log = _fresh_log(tmp_path)
    assert is_append_only(str(log)) is False

    real_open = os.open

    def raising_open(errno_code: int):  # type: ignore[no-untyped-def]
        def _open(file, flags, *args, **kwargs):  # type: ignore[no-untyped-def]
            if os.fspath(file) == str(log):
                raise PermissionError(errno_code, "denied")
            return real_open(file, flags, *args, **kwargs)

        return _open

    import errno

    # EPERM on a plain write AND on an append is the IMMUTABLE shape
    # (chattr +i), not the append-only one. This arm used to assert True,
    # which blessed a file whose appends will fail -- corrected 2026-08-31
    # after an independent audit.
    monkeypatch.setattr(os, "open", raising_open(errno.EPERM))
    assert is_append_only(str(log)) is None, "EPERM on append too means write-blocked, not append-only"

    # The genuine chattr +a shape: plain write refused, append permitted.
    def eperm_unless_append(file, flags, *args, **kwargs):  # type: ignore[no-untyped-def]
        if os.fspath(file) == str(log) and not (flags & os.O_APPEND):
            raise PermissionError(errno.EPERM, "denied")
        return real_open(file, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", eperm_unless_append)
    assert is_append_only(str(log)) is True

    monkeypatch.setattr(os, "open", raising_open(errno.EACCES))
    assert is_append_only(str(log)) is None, "no write permission is not evidence of an append-only flag"


@pytest.mark.unit
def test_capability_is_detected_by_probing_not_by_the_platform_name() -> None:
    """``sys.platform == "linux"`` is a guess about the filesystem.

    A FAT volume on Linux, an NFS mount on macOS and a WSL path all
    break the platform-name shortcut, which is why
    ``tests/_platform_compat.py`` probes instead. Same rule here: the
    module must ask the host what it can do, not what it is called.
    """
    import inspect

    import mind_mem.append_only as mod

    source = inspect.getsource(mod)
    assert "sys.platform" not in source
    assert "platform.system" not in source
    assert "os.name" not in source


@pytest.mark.unit
def test_create_does_not_follow_a_dangling_symlink_out_of_the_workspace(tmp_path: Path) -> None:
    """``create=True`` must not write through a symlink it did not expect.

    An audit path that is a dangling symlink is a planted one: following
    it would create the "audit log" wherever the link points, and the
    caller would then harden -- and trust -- a file of someone else's
    choosing. Refusing is the honest outcome; the status says so.
    """
    target = tmp_path / "elsewhere.jsonl"
    link = tmp_path / "audit-link.jsonl"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):  # pragma: no cover - unprivileged Windows
        pytest.skip("this host cannot create symlinks")

    status = ensure_append_only(str(link), create=True)

    assert status.enforced is False
    assert not target.exists(), "the helper followed a dangling symlink and created the target"
    assert "NOT append-only" in status.detail


class TestTheProbeIsTwoSided:
    """EPERM on O_WRONLY alone must not be read as append-only.

    `chattr +i` (immutable) and several LSMs return the same EPERM that
    `chattr +a` does. Answering True on the write-refusal alone blesses a
    file where the O_APPEND writes this module promises will FAIL — the
    exact "claims a protection it does not have" failure the module exists
    to prevent. Raised by an independent audit, 2026-08-31.
    """

    def test_write_blocked_in_both_modes_is_not_append_only(self, tmp_path, monkeypatch):
        from mind_mem import append_only as ao

        target = tmp_path / "immutable.jsonl"
        target.write_text('{"seq": 1}\n', encoding="utf-8")
        real_open = os.open

        def refuse_everything(path, flags, *a, **kw):
            if str(path) == str(target):
                raise PermissionError(errno.EPERM, "Operation not permitted")
            return real_open(path, flags, *a, **kw)

        monkeypatch.setattr(os, "open", refuse_everything)
        # Immutable shape: EPERM on BOTH plain and append opens.
        assert ao.is_append_only(str(target)) is None, "a file refusing writes in BOTH modes is write-blocked, not append-only"

    def test_append_still_permitted_is_append_only(self, tmp_path, monkeypatch):
        from mind_mem import append_only as ao

        target = tmp_path / "appendable.jsonl"
        target.write_text('{"seq": 1}\n', encoding="utf-8")
        real_open = os.open

        def refuse_plain_allow_append(path, flags, *a, **kw):
            if str(path) == str(target) and not (flags & os.O_APPEND):
                raise PermissionError(errno.EPERM, "Operation not permitted")
            return real_open(path, flags, *a, **kw)

        monkeypatch.setattr(os, "open", refuse_plain_allow_append)
        # The genuine chattr +a shape: plain write refused, append allowed.
        assert ao.is_append_only(str(target)) is True
