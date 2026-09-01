"""The tar-extraction guard in ``bench.repo_task_validation``.

``extract_tree`` uses ``filter="data"`` where the interpreter has it. On an
un-backported 3.10 -- still inside ``requires-python = ">=3.10"`` -- it falls
back, and the fallback used to call bare ``extractall``. These tests pin the
hand-rolled replacement so the fallback is not a hole that only opens on the
oldest supported interpreter.
"""

from __future__ import annotations

import io
import os
import tarfile

import pytest

from mind_mem.bench.repo_task_validation import _vetted_members, _within


def _tar_with(*members: tarfile.TarInfo) -> tarfile.TarFile:
    """An in-memory tar carrying exactly ``members`` (empty regular files)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as writer:
        for member in members:
            if member.isfile():
                writer.addfile(member, io.BytesIO(b""))
            else:
                writer.addfile(member)
    buf.seek(0)
    return tarfile.open(fileobj=buf, mode="r")


def _file(name: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.REGTYPE
    info.size = 0
    return info


def _symlink(name: str, target: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    return info


def test_within_accepts_the_destination_and_its_children(tmp_path) -> None:
    dest = str(tmp_path)
    assert _within(dest, dest)
    assert _within(dest, os.path.join(dest, "a", "b.txt"))


def test_within_rejects_a_sibling_that_shares_a_name_prefix(tmp_path) -> None:
    # `/tmp/x/dest-evil` must not pass a naive startswith check against
    # `/tmp/x/dest`; the separator is what makes the check sound.
    dest = str(tmp_path / "dest")
    assert not _within(dest, str(tmp_path / "dest-evil" / "f.txt"))


def test_ordinary_members_survive_the_filter(tmp_path) -> None:
    tar = _tar_with(_file("pkg/mod.py"), _file("README.md"))
    names = [m.name for m in _vetted_members(tar, str(tmp_path))]
    assert names == ["pkg/mod.py", "README.md"]


def test_traversing_member_is_refused(tmp_path) -> None:
    tar = _tar_with(_file("../escaped.txt"))
    with pytest.raises(RuntimeError, match="escapes destination"):
        list(_vetted_members(tar, str(tmp_path)))


def test_absolute_member_is_refused(tmp_path) -> None:
    tar = _tar_with(_file("/etc/passwd"))
    with pytest.raises(RuntimeError, match="escapes destination"):
        list(_vetted_members(tar, str(tmp_path)))


def test_symlink_pointing_outside_is_refused(tmp_path) -> None:
    tar = _tar_with(_symlink("link", "../../outside"))
    with pytest.raises(RuntimeError, match="points outside destination"):
        list(_vetted_members(tar, str(tmp_path)))


def test_symlink_staying_inside_is_kept(tmp_path) -> None:
    tar = _tar_with(_symlink("pkg/link", "mod.py"))
    assert [m.name for m in _vetted_members(tar, str(tmp_path))] == ["pkg/link"]


def test_device_nodes_are_dropped_rather_than_materialised(tmp_path) -> None:
    node = tarfile.TarInfo("dev/null")
    node.type = tarfile.CHRTYPE
    tar = _tar_with(_file("keep.py"), node)
    assert [m.name for m in _vetted_members(tar, str(tmp_path))] == ["keep.py"]
