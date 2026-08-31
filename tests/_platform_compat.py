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

__all__ = ["assert_owner_only", "is_root", "posix_creation_modes_honored"]


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
