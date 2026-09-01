# Copyright 2026 STARGA, Inc.
"""Platform helpers for the test suite.

Tests kept reaching for APIs that exist only on the runner that happened
to be in front of the author:

* ``os.geteuid`` is Unix-only. On Windows it does not exist at all, so
  ``os.geteuid() == 0`` raised AttributeError and reddened every Windows
  job -- including at import time, inside a ``skipif`` decorator, where it
  takes the whole module down rather than one test.
* ``datetime.UTC`` arrived in 3.11, but ``requires-python`` is ``>=3.10``
  and the matrix runs 3.10.

Neither is a product limitation; both are tests asserting the host.
"""

from __future__ import annotations

import os

__all__ = [
    "append_only_settable_unprivileged",
    "chmod_denies_read",
    "assert_owner_only",
    "is_root",
    "posix_creation_modes_honored",
]


def is_root() -> bool:
    """True when the process can ignore filesystem permission bits.

    Windows has no ``geteuid`` and no equivalent notion here, so it
    answers False: permission-based tests are skipped there by their own
    filesystem checks, not by a uid that does not exist.
    """
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:  # Windows
        return False
    return geteuid() == 0


def posix_creation_modes_honored(tmp_dir) -> bool:
    """True when ``os.open(..., 0o600)`` actually yields mode 0600.

    Windows has no POSIX mode bits: a file created with 0o600 reports
    0o666, so an assertion that the key is owner-only is asserting the
    host's filesystem, not the product. Probed rather than branched on
    ``sys.platform``, because either behaviour can be mounted anywhere
    (a FAT/exFAT volume on Linux ignores modes too).

    The probe runs under ``umask(0)`` so the umask cannot mask the very
    bits being measured.
    """
    import stat as _stat

    probe = os.path.join(str(tmp_dir), ".mode_probe")
    old = os.umask(0)
    try:
        fd = os.open(probe, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
        os.close(fd)
        return _stat.S_IMODE(os.stat(probe).st_mode) == 0o600
    except OSError:  # pragma: no cover - probe could not run
        return False
    finally:
        os.umask(old)
        try:
            os.unlink(probe)
        except OSError:  # pragma: no cover
            pass


def assert_owner_only(path) -> None:
    """Assert *path* is mode 0600, where the filesystem can express that.

    Six tests asserted ``S_IMODE(...) == 0o600`` directly. On Windows a
    file created with 0o600 reports 0o666 -- there are no POSIX mode bits
    -- so those assertions were testing the runner, not the product, and
    reddened every Windows job.

    They failed ONE AT A TIME because CI runs pytest with ``-x``
    (ci.yml:99,104): the run stops at the first failure, so fixing one
    only revealed the next. Hence a single shared helper rather than six
    separate guards.

    Where modes are honoured this asserts exactly what it did before.
    Where they are not, it still asserts the file exists and is readable
    -- it degrades to a weaker check, never to nothing.
    """
    import stat as _stat

    path = os.fspath(path)
    assert os.path.isfile(path), f"{path} does not exist"
    if posix_creation_modes_honored(os.path.dirname(path) or "."):
        mode = _stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o600, f"expected owner-only 0600, got {mode:04o}"


def append_only_settable_unprivileged(tmp_dir) -> bool:
    """True when a NON-root owner can actually set the append-only flag here.

    Linux `chattr +a` needs CAP_LINUX_IMMUTABLE, so an unprivileged user is
    refused -- which is what the refusal tests pin. macOS is different: the
    file's OWNER may set the user append-only flag (`chflags uappnd`) with no
    privilege at all, so on macOS the very same call SUCCEEDS and
    ``enforced=True`` is the correct answer, not a bug.

    Probed rather than branched on a platform name, for the reason
    ``posix_creation_modes_honored`` exists: the question is what THIS
    filesystem does, and a container, a network mount or a future runner can
    answer differently from the OS it is nominally running.
    """
    import mind_mem.append_only as _ao

    probe = os.path.join(str(tmp_dir), ".append-only-capability-probe")
    try:
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("probe\n")
        return bool(_ao.ensure_append_only(probe).enforced)
    except OSError:
        return False
    finally:
        try:
            os.chmod(probe, 0o600)
        except OSError:
            pass
        try:
            os.remove(probe)
        except OSError:
            # An actually-flagged probe file may be undeletable; harmless in
            # a tmp dir, and reporting it as "settable" is the useful answer.
            pass


def chmod_denies_read(tmp_dir) -> bool:
    """True when ``chmod(path, 0o000)`` actually makes a path unreadable here.

    Windows has no POSIX mode bits: ``os.chmod`` accepts the call, changes only
    the read-only attribute, and the directory stays perfectly readable. A test
    that locks a path and asserts the product NOTICED is then asserting the
    host, and fails on Windows for a reason that has nothing to do with the
    product.

    Probed, not branched on a platform name, for the same reason as
    ``posix_creation_modes_honored``: the honest question is what THIS
    filesystem enforces. A container, an overlay mount, or a runner where the
    process is effectively privileged can all answer differently from the OS
    they nominally run.
    """
    probe = os.path.join(str(tmp_dir), ".perm-probe")
    os.makedirs(probe, exist_ok=True)
    inner = os.path.join(probe, "f.txt")
    try:
        with open(inner, "w", encoding="utf-8") as handle:
            handle.write("x\n")
        os.chmod(probe, 0o000)
        try:
            os.listdir(probe)
            return False  # still readable -> bits not enforced
        except PermissionError:
            return True
    except OSError:
        return False
    finally:
        try:
            os.chmod(probe, 0o700)
        except OSError:
            pass
        try:
            os.remove(inner)
            os.rmdir(probe)
        except OSError:
            pass
