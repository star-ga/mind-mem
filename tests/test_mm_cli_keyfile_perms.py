"""``mm sign-model --generate-key`` and the permissions it claims.

The secret key was written with ``Path.write_bytes`` — creating it under
the process umask, typically 0644 — and only then chmod-ed to 0600, with
the chmod wrapped in ``except OSError: pass``. The success line printed
"(private, 0600)" unconditionally, so on a filesystem that refuses chmod
(FAT/exFAT, some NFS and container mounts) the CLI asserted a permission
the file did not have, and even on a normal filesystem the key existed
world-readable for the window between the two calls.

The file is now created 0600 by the ``open`` itself, and the printed
description is read back from the file rather than assumed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import stat
from pathlib import Path

import pytest
from _platform_compat import assert_owner_only, posix_creation_modes_honored

from mind_mem.mm_cli import _cmd_sign_model, _write_private_key

_SECRET = b"\x11" * 32


@pytest.mark.unit
def test_the_cli_line_never_claims_a_mode_the_key_does_not_have(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: the printed permission must match the file on disk."""
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"\x00\x01\x02")
    prefix = tmp_path / "key"

    # A filesystem that refuses chmod — the failure the CLI used to swallow.
    monkeypatch.setattr(os, "chmod", lambda *a, **k: (_ for _ in ()).throw(PermissionError(1, "Operation not permitted")))
    old_umask = os.umask(0)
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = _cmd_sign_model(
                argparse.Namespace(
                    path=str(checkpoint),
                    key_file="",
                    generate_key=str(prefix),
                    no_sidecars=True,
                    json=False,
                )
            )
    finally:
        os.umask(old_umask)

    assert rc == 0
    sk_path = Path(str(prefix) + ".sk")
    mode = stat.S_IMODE(sk_path.stat().st_mode)
    line = err.getvalue()
    claimed = re.search(r"\(private, (\d{4})\)", line)
    if claimed:
        assert int(claimed.group(1), 8) == mode, f"CLI claimed {claimed.group(1)} for a file that is {mode:04o}"
    else:
        assert "WARNING" in line
    # The property this test is NAMED for -- the CLI never claims a mode the
    # key does not have -- is asserted above and holds on every platform.
    # This last line is different: it asserts POSIX creation-mode semantics.
    # chmod is patched to raise and umask is 0, so the creation mode is the
    # only protection left, and Windows has no POSIX mode bits -- os.open
    # with 0o600 reports 0o666 there. Probing the filesystem, not
    # sys.platform, since a mode-ignoring volume can be mounted anywhere.
    if posix_creation_modes_honored(tmp_path):
        assert mode == 0o600  # and the key is in fact owner-only
    else:
        # On such a filesystem the requirement is "report, don't claim" --
        # which is exactly what the claimed-vs-mode check above enforces.
        assert claimed is None or int(claimed.group(1), 8) == mode


@pytest.mark.unit
def test_the_key_is_owner_only_even_under_a_permissive_umask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A filesystem that refuses chmod: the creation mode is the only
    # protection the key gets.
    monkeypatch.setattr(os, "chmod", lambda *a, **k: (_ for _ in ()).throw(PermissionError(1, "Operation not permitted")))
    old_umask = os.umask(0)
    try:
        path = tmp_path / "k.sk"
        description = _write_private_key(path, _SECRET)
    finally:
        os.umask(old_umask)

    assert path.read_bytes() == _SECRET
    assert_owner_only(path)
    if posix_creation_modes_honored(str(path.parent)):
        assert description == "private, 0600"


@pytest.mark.unit
def test_a_mode_the_filesystem_refused_is_reported_not_claimed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "k.sk"
    real_open = os.open

    def open_ignoring_mode(file, flags, mode=0o777, *args, **kwargs):
        # A filesystem that ignores the creation mode (the Windows shape).
        return real_open(file, flags, 0o644, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_ignoring_mode)
    monkeypatch.setattr(os, "chmod", lambda *a, **k: (_ for _ in ()).throw(PermissionError(1, "nope")))
    old_umask = os.umask(0)
    try:
        description = _write_private_key(path, _SECRET)
    finally:
        os.umask(old_umask)

    assert not description.startswith("private, 0600")
    assert "0644" in description
    assert "NOT owner-only" in description
    assert "WARNING" in description


@pytest.mark.unit
def test_an_existing_permissive_key_file_is_replaced_not_reused(tmp_path: Path) -> None:
    path = tmp_path / "k.sk"
    path.write_bytes(b"stale")
    os.chmod(path, 0o666)
    description = _write_private_key(path, _SECRET)
    assert path.read_bytes() == _SECRET
    assert_owner_only(path)
    if posix_creation_modes_honored(str(path.parent)):
        assert description == "private, 0600"
