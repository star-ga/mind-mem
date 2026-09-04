# Copyright 2026 STARGA, Inc.
"""OS-level append-only protection for the audit trails (T-007).

The hash-chained ledger makes tampering *detectable*. This module is the
second, independent layer described in ``docs/append-only-audit-logs.md``:
it asks the kernel to refuse in-place rewrites, via ``chattr +a``
(Linux) or ``chflags uappnd`` (macOS/BSD). After the flag is set,
``O_APPEND`` writes keep working while truncate, seek-and-write and
``unlink`` are refused.

**It raises the bar; it does not make tampering impossible, and this
module must never be described as if it did.** On Linux anyone holding
``CAP_LINUX_IMMUTABLE`` clears the flag with ``chattr -a`` and rewrites
freely. On the ``chflags`` path this sets ``UF_APPEND``, a *user* flag:
the file's owner sets it and can clear it **unprivileged** -- so the very
threat this layer addresses, a compromised agent running as the owner,
undoes it in one ``chflags`` call. Only ``SF_APPEND`` at securelevel >= 1
resists root on BSD, and this module does not use it.

The honest division of labour: **the hash chain is the detection layer
and remains the guarantee; this flag is a speed bump against accidental
and unprivileged rewrite.** Anything stronger stated about it is a claim
the mechanism does not support.

Three facts shape the whole design:

1. **Setting the flag usually fails.** It needs ``CAP_LINUX_IMMUTABLE``
   (root) on Linux, and a filesystem that implements it -- tmpfs, NFS
   and SMB do not. This is therefore a privileged *setup* helper, run by
   an installer or an operator, not something a write path calls on
   every append.
2. **A refusal must be reported, never papered over.** A helper that
   answered "hardened" after a failed ``chattr`` would be worse than no
   helper, because an operator would stop applying the runbook. Same
   defect class as the private key that printed "(private, 0600)" on a
   filesystem that had refused the chmod
   (``tests/test_mm_cli_keyfile_perms.py``). So the result is read back
   from the filesystem, and ``enforced`` is True only when the read-back
   confirms it.
3. **Capability is probed, not inferred from the platform name.** A FAT
   volume under Linux and an NFS mount under macOS both break the
   name-based shortcut, and Windows -- which has no equivalent at all --
   must be a clean no-op rather than a crash. So the module asks whether
   ``chflags(2)`` exists and whether ``chattr(1)`` is on PATH, exactly
   as ``tests/_platform_compat.py`` probes the filesystem instead of
   trusting the runner's name.

The read-back is behavioural: an append-only file refuses
``open(O_WRONLY)`` without ``O_APPEND`` with EPERM on both Linux and
BSD. That is the property callers actually want, it is one mechanism
covering both backends, and it believes the filesystem rather than a
tool's exit code -- an overlay that accepts the ioctl and drops it, or a
shim named ``chattr`` on PATH, both exit 0.

Usage::

    from mind_mem.append_only import ensure_append_only

    status = ensure_append_only(chain_path)
    if not status.enforced:
        log.warning("audit log not hardened", detail=status.detail)

``MIND_MEM_AUDIT_APPEND_ONLY`` selects the mode: ``try`` (default --
attempt, report honestly, never raise), ``require`` (fail closed:
raise :class:`AppendOnlyUnavailable` when the flag cannot be verified,
so a deployment that demands the protection cannot silently run
without it) or ``off`` (do not touch the file at all).
"""

from __future__ import annotations

import errno
import os
import shutil
import stat
import subprocess  # nosec B404 -- chattr(1) has no stdlib binding; argv list, no shell
from dataclasses import dataclass

from .observability import get_logger

__all__ = [
    "ENV_APPEND_ONLY_MODE",
    "AppendOnlyStatus",
    "AppendOnlyUnavailable",
    "append_only_mechanism",
    "ensure_append_only",
    "is_append_only",
]

_log = get_logger("append_only")

ENV_APPEND_ONLY_MODE = "MIND_MEM_AUDIT_APPEND_ONLY"

_MODES = ("off", "try", "require")
_DEFAULT_MODE = "try"

# chattr(1) is a local, non-interactive call; a hung binary must not wedge
# an installer.
_CHATTR_TIMEOUT_S = 10

_RUNBOOK = "docs/append-only-audit-logs.md"


class AppendOnlyUnavailable(RuntimeError):
    """Raised in ``require`` mode when the flag could not be verified."""


@dataclass(frozen=True)
class AppendOnlyStatus:
    """What was asked for, what was obtained, and the honest difference.

    ``enforced`` is True only when the file was read back and confirmed
    to refuse an in-place write. ``mechanism`` names how it was
    attempted (``chflags``, ``chattr``, ``preset`` when the operator had
    already applied it, ``disabled`` when the mode is ``off``, ``none``
    when nothing was attempted -- the host offers no mechanism, or the
    file was not there to flag). ``detail`` is the sentence a caller
    may show or log verbatim; when the file is not protected it says so
    in those words.
    """

    path: str
    enforced: bool
    mechanism: str
    detail: str


def append_only_mechanism() -> str:
    """Probe for an append-only mechanism: ``chflags``/``chattr``/``none``.

    ``os.chflags`` is bound only where the syscall exists (macOS, the
    BSDs), so its presence is the capability itself rather than a guess
    about the host. Linux has no stdlib binding for the equivalent
    ioctl, so ``chattr(1)`` is used when it is on PATH. Neither present
    (Windows, or a stripped container) answers ``"none"``, which callers
    treat as a no-op.
    """
    if getattr(os, "chflags", None) is not None:
        return "chflags"
    if shutil.which("chattr"):
        return "chattr"
    return "none"


def is_append_only(path: str) -> bool | None:
    """Is *path* append-only right now? True / False / None for "cannot tell".

    Measured behaviourally: an append-only file refuses
    ``open(O_WRONLY)`` -- no ``O_APPEND``, no ``O_TRUNC`` -- with EPERM,
    on Linux and BSD alike. Nothing is written; the descriptor is closed
    immediately.

    The tri-state matters. EACCES (no write permission at all), EROFS (a
    read-only mount) or EISDIR are *not* evidence of the flag, and
    folding them into False would claim the file is unprotected when we
    simply could not tell, while folding them into True would claim a
    protection that may not exist.
    """
    try:
        fd = os.open(path, os.O_WRONLY)
    except PermissionError as exc:
        if exc.errno != errno.EPERM:
            return None
        # TWO-SIDED. EPERM on a plain O_WRONLY is necessary but NOT
        # sufficient: `chattr +i` (immutable) and several LSMs produce the
        # same errno, and on an immutable file the O_APPEND writes this
        # module promises keep working do NOT. Answering True on the
        # write-refusal alone would bless a file the audit writer is about
        # to start failing on -- the exact "claims a protection it does not
        # have" failure this module exists to avoid.
        try:
            probe = os.open(path, os.O_WRONLY | os.O_APPEND)
        except OSError:
            # Write-blocked in BOTH modes: something is refusing writes, but
            # it is not append-only. Not a lie in either direction.
            return None
        os.close(probe)  # opened and closed; nothing was written
        return True
    except OSError:
        return None
    os.close(fd)
    return False


def _resolve_mode(mode: str | None) -> str:
    raw = mode if mode is not None else os.environ.get(ENV_APPEND_ONLY_MODE, _DEFAULT_MODE)
    value = raw.strip().lower() or _DEFAULT_MODE
    if value not in _MODES:
        # A typo must not quietly become "off" -- that is the silent
        # loss of a protection the operator asked for.
        raise ValueError(f"{ENV_APPEND_ONLY_MODE}={raw!r} is not one of {'/'.join(_MODES)}; refusing to guess which was meant")
    return value


def _unprotected(path: str, mechanism: str, why: str, mode: str) -> AppendOnlyStatus:
    """Build (and, in ``require`` mode, raise) the honest negative result."""
    detail = f"WARNING: NOT append-only - {why}"
    status = AppendOnlyStatus(path=path, enforced=False, mechanism=mechanism, detail=detail)
    if mode == "require":
        raise AppendOnlyUnavailable(f"{detail} (set {ENV_APPEND_ONLY_MODE}=try to downgrade this to a warning; see {_RUNBOOK})")
    _log.warning("append_only_not_enforced", path=path, mechanism=mechanism, detail=detail)
    return status


def _apply(path: str, mechanism: str) -> str | None:
    """Attempt the flag. Returns an error string, or None if the call succeeded.

    A returned None means only "the call did not complain" -- never that
    the flag is set. The caller verifies.
    """
    if mechanism == "chflags":
        chflags = getattr(os, "chflags", None)
        if chflags is None:  # pragma: no cover - probed immediately before
            return "chflags(2) disappeared between probe and call"
        try:
            current = getattr(os.stat(path), "st_flags", 0)
            chflags(path, current | stat.UF_APPEND)
        except OSError as exc:
            return f"chflags uappnd refused: {exc.strerror or exc}"
        return None

    binary = shutil.which("chattr") or "chattr"
    # abspath, because chattr(1) reads a leading dash as an option and
    # has no "--" end-of-options marker.
    try:
        proc = subprocess.run(  # nosec B603 - fixed argv, absolute path from shutil.which, shell=False
            [binary, "+a", os.path.abspath(path)],
            capture_output=True,
            text=True,
            timeout=_CHATTR_TIMEOUT_S,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"chattr +a could not run: {exc}"
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout).strip() or f"exit {proc.returncode}"
        return f"chattr +a refused: {message}"
    return None


def ensure_append_only(path: str, *, create: bool = False, mode: str | None = None) -> AppendOnlyStatus:
    """Make *path* OS-level append-only where the host supports it.

    Returns an :class:`AppendOnlyStatus` whose ``enforced`` field is True
    only when the filesystem confirmed the flag on read-back. Every
    failure -- unprivileged process, unsupported filesystem, a platform
    with no such flag -- is a reported result, not an exception, unless
    ``MIND_MEM_AUDIT_APPEND_ONLY=require`` asked for fail-closed.

    *create* opens a missing file 0600 with ``O_APPEND | O_NOFOLLOW``
    first; without it a missing file is reported as such (``chattr`` on a
    path that does not exist would otherwise be an obscure error, and
    creating files as a side effect of a hardening call should be the
    caller's decision).
    *mode* overrides the environment variable for a caller that knows
    what it needs.
    """
    resolved = _resolve_mode(mode)
    if resolved == "off":
        return AppendOnlyStatus(
            path=path,
            enforced=False,
            mechanism="disabled",
            detail=f"NOT append-only - hardening disabled by {ENV_APPEND_ONLY_MODE}=off",
        )

    if not os.path.exists(path):
        if not create:
            return _unprotected(
                path, "none", f"{path} does not exist yet; the file must exist before the flag can be set (see {_RUNBOOK})", resolved
            )
        # A dangling symlink at an audit path is a planted one: following it
        # would create -- then harden, then trust -- a file of someone else's
        # choosing. os.path.exists() is False for a dangling link, so this
        # branch is exactly where such a link lands.
        #
        # Checked explicitly rather than left to O_NOFOLLOW alone. That flag is
        # POSIX-only, so `getattr(os, "O_NOFOLLOW", 0)` degrades to 0 on
        # Windows and the open silently FOLLOWS the link and creates the
        # target -- the precise outcome the flag was there to prevent, on the
        # one platform that could not say so. os.path.islink is available
        # everywhere, so this needs no platform branch (and this module
        # forbids one: see test_no_platform_branching).
        if os.path.islink(path):
            return _unprotected(
                path,
                "none",
                f"{path} is a dangling symlink; refusing to create an audit log through it (see {_RUNBOOK})",
                resolved,
            )
        try:
            # O_NOFOLLOW still requested where it exists: it closes the race
            # between the islink check above and this open.
            fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
        except OSError as exc:
            return _unprotected(path, "none", f"{path} could not be created: {exc.strerror or exc}", resolved)
        os.close(fd)

    # An operator who already applied the runbook, or a second call on an
    # already-flagged file: believe the filesystem and change nothing.
    if is_append_only(path) is True:
        return AppendOnlyStatus(
            path=path,
            enforced=True,
            mechanism="preset",
            detail="append-only already enforced (verified: an in-place write is refused); left unchanged",
        )

    mechanism = append_only_mechanism()
    if mechanism == "none":
        return _unprotected(
            path,
            mechanism,
            f"this host offers no OS-level append-only flag (no chflags(2), no chattr(1)); see {_RUNBOOK} for the ACL/WORM alternatives",
            resolved,
        )

    error = _apply(path, mechanism)
    verified = is_append_only(path)
    if verified is True:
        detail = f"append-only enforced via {mechanism} (verified: an in-place write is refused)"
        _log.info("append_only_enforced", path=path, mechanism=mechanism)
        return AppendOnlyStatus(path=path, enforced=True, mechanism=mechanism, detail=detail)

    if error is not None:
        why = f"{error} - the flag needs elevated privileges (CAP_LINUX_IMMUTABLE / root) and a filesystem that supports it"
    elif verified is False:
        # Exit code 0 is not evidence: a shim on PATH, or a filesystem
        # that accepts the ioctl and drops it, both look like success.
        why = f"{mechanism} reported success but the file still accepts an in-place write, so the flag did not take"
    else:
        why = f"{mechanism} reported success but the result could not be read back, so no protection is claimed"
    return _unprotected(path, mechanism, why, resolved)
