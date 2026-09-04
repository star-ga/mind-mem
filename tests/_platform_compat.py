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
* a subprocess environment scrubbed down to a hand-written POSIX dict
  (``PATH=/usr/bin:/bin``, ``HOME=...``) does not start an interpreter on
  Windows at all -- see :func:`minimal_child_env`.

None of these is a product limitation; all are tests asserting the host.
"""

from __future__ import annotations

import os
import sys

__all__ = [
    "append_only_settable_unprivileged",
    "atomic_cross_process_append",
    "child_import_path",
    "child_pythonpath",
    "chmod_denies_read",
    "chmod_denies_write",
    "assert_owner_only",
    "is_root",
    "minimal_child_env",
    "posix_creation_modes_honored",
]

# The variables Windows itself needs in a child environment BEFORE any test
# code runs. Every entry carries its reason; nothing is here "just in case".
#
#   SYSTEMROOT  BCryptGenRandom is reached through %SystemRoot%. Without it
#               ``_Py_HashRandomization_Init: failed to get random numbers to
#               initialize Python`` kills the interpreter before it executes
#               a single line -- the exact failure this list exists to fix.
#   PATH        Windows resolves ``python3XX.dll``, the CRT, and the DLLs the
#               stdlib extension modules link against through the process
#               search path. POSIX uses the ELF loader instead, which is why
#               a POSIX-only dict got away with a fabricated PATH.
#   PATHEXT     ``shutil.which()`` and ``subprocess`` find no executable at
#               all without it, so a child that shells out degrades silently
#               instead of failing loudly.
#   COMSPEC     the ``cmd.exe`` path any ``shell=True`` spawn beneath us needs.
#   TEMP, TMP   ``tempfile.gettempdir()`` reads these first on Windows; with
#               neither present it falls through to hardcoded candidates such
#               as ``C:\temp`` that need not exist on a runner.
_WINDOWS_CHILD_ENV = ("SYSTEMROOT", "PATH", "PATHEXT", "COMSPEC", "TEMP", "TMP")


def atomic_cross_process_append() -> bool:
    """True where two processes appending to one file cannot lose each other's writes.

    POSIX gives ``open(path, "a")`` the ``O_APPEND`` flag, and a single
    ``write()`` on an ``O_APPEND`` descriptor seeks to end and writes as one
    atomic operation, so unsynchronised appenders interleave whole records
    and never overwrite one another. The Windows CRT has no such guarantee:
    ``_O_APPEND`` is emulated as *seek to end, then write*, two operations
    with a window between them, so two processes can compute the same end
    offset and the second write lands on top of the first.

    Measured, not assumed. Three processes appending 20 chain records each
    with the store lock defeated, on the windows-latest runners:
    **46 of 60 rows** survived on the evidence chain and **54 of 60** on the
    audit ledger, with every worker exiting 0 and stderr empty. The same run
    on ubuntu-latest and macos-latest lands all 60 every time.

    This is a fact about the OS, not about mind-mem: under the lock the
    appends are serialised and every row lands on every platform, which is
    what the gates in ``test_chain_concurrency`` assert and what the Windows
    rows already prove. It matters only where a test deliberately removes
    the lock -- a mutation twin -- and must therefore not expect the
    filesystem to do the lock's job.

    Branching on the platform name rather than probing it, unlike the
    predicates below: a probe for this needs a genuine cross-process race,
    and a probe that can come out either way by luck is worse than none.
    Both sides of the branch assert; neither skips.
    """
    return sys.platform != "win32"


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


def chmod_denies_write(tmp_dir) -> bool:
    """True when clearing the write bit actually blocks creating a file here.

    Distinct from :func:`chmod_denies_read`: several tests set a directory to
    ``r-x------`` and assert the product NOTICED it could not write there.
    Windows ignores POSIX mode bits, so the create succeeds and the expected
    error never arrives -- "DID NOT RAISE", for a reason that is about the
    runner rather than the product.
    """
    probe = os.path.join(str(tmp_dir), ".write-probe")
    os.makedirs(probe, exist_ok=True)
    try:
        os.chmod(probe, 0o500)  # r-x------
        try:
            with open(os.path.join(probe, "canary"), "w", encoding="utf-8") as handle:
                handle.write("x")
            return False  # write went through -> bits not enforced
        except (PermissionError, OSError):
            return True
    except OSError:
        return False
    finally:
        try:
            os.chmod(probe, 0o700)
        except OSError:
            pass
        for name in ("canary",):
            try:
                os.remove(os.path.join(probe, name))
            except OSError:
                pass
        try:
            os.rmdir(probe)
        except OSError:
            pass


def child_import_path() -> str:
    """The ``sys.path`` entry a child interpreter needs to import ``mind_mem``.

    Derived from the package the PARENT actually imported, so a spawned
    interpreter measures the code under test rather than whichever copy it
    happens to find.

    This exists because a test that redirects ``HOME`` silently redirects
    Python's USER site-packages with it: ``~/.local/lib/pythonX.Y/
    site-packages`` is computed from the home directory, so a child launched
    under a sandboxed home cannot import a user-installed ``mind_mem`` and
    dies with ``ModuleNotFoundError``. CI never sees it -- there the package
    is installed system-wide, where the home directory is irrelevant -- so
    the divergence shows up only on a developer's box, which is the worst
    place for a suite to disagree with its own CI.
    """
    import mind_mem

    return os.path.dirname(os.path.dirname(os.path.abspath(mind_mem.__file__)))


def child_pythonpath(existing: str | None = None) -> str:
    """:func:`child_import_path` prepended to *existing* ``PYTHONPATH``.

    *existing* defaults to the parent's own ``PYTHONPATH``. Prepending rather
    than replacing keeps a caller-supplied path (an editable checkout, a
    coverage shim) reachable.
    """
    if existing is None:
        existing = os.environ.get("PYTHONPATH", "")
    head = child_import_path()
    parts = [head] + [p for p in existing.split(os.pathsep) if p and p != head]
    return os.pathsep.join(parts)


def minimal_child_env(home, **extra: str) -> dict[str, str]:
    """A SCRUBBED child environment that still starts an interpreter.

    For tests whose claim is about a fresh process -- "an interpreter that
    imports only X sees only Y" -- which cannot be shown in-process because
    the state under test is module-global. Those tests must NOT inherit
    ``os.environ``; that is the isolation they exist to create.

    Hand-writing the dict instead went wrong on two axes at once:

    * ``{"PATH": "/usr/bin:/bin", "HOME": ...}`` omits ``SystemRoot``, and
      Windows needs it to seed hash randomization. The child died in
      ``_Py_HashRandomization_Init`` before running any test code -- a fatal
      interpreter error reported as a plain assertion failure.
    * ``HOME`` is not the home variable on Windows (``expanduser`` reads
      ``USERPROFILE``), and it relocates user site-packages on POSIX, which
      is why the same test also failed locally with ``ModuleNotFoundError``.

    So: the OS minimum for the host we are on (see ``_WINDOWS_CHILD_ENV`` for
    the per-variable justification), both home variables pointed at *home*,
    an import path that survives the home redirect, and whatever the caller
    passes in ``extra``. Nothing else -- ``extra`` is where the test's own
    variables go, and no other parent variable comes through.
    """
    env: dict[str, str] = {}
    if sys.platform == "win32":
        for name in _WINDOWS_CHILD_ENV:
            value = os.environ.get(name)
            if value is not None:
                env[name] = value
    else:
        # Meaningful on POSIX, and deliberately not the parent's PATH: the
        # child resolves its interpreter by absolute path, so this only has
        # to cover a ``which`` lookup beneath us.
        env["PATH"] = "/usr/bin:/bin"
    home = os.fspath(home)
    # Both, always: ``posixpath.expanduser`` reads HOME and
    # ``ntpath.expanduser`` reads USERPROFILE, so setting one leaves the
    # other platform looking at the real user's home.
    env["HOME"] = home
    env["USERPROFILE"] = home
    env["PYTHONPATH"] = child_import_path()
    env.update(extra)
    return env
