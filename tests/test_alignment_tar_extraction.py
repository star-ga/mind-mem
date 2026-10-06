"""The docs-alignment authority unpacks a ``git archive`` tarball; prove it
cannot be made to write outside its extraction directory.

``scripts/alignment_authorities._extract_archive`` validates every member and
then extracts with the stdlib ``data`` filter where available, or writes
regular files one by one where not. Each hostile archive below is run through
BOTH paths, with a canary file planted outside the destination so a write
that escaped would be visible, not merely absent from the expected place.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from scripts import alignment_authorities as aa

PATHS = [pytest.param(True, id="data-filter"), pytest.param(False, id="member-by-member")]


def _tar(*members: tuple[tarfile.TarInfo, bytes | None]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for info, payload in members:
            if payload is not None:
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            else:
                tar.addfile(info)
    return buf.getvalue()


def _file(name: str, payload: bytes = b"pwned\n") -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.type = tarfile.REGTYPE
    return info, payload


def _link(name: str, target: str, kind: bytes = tarfile.SYMTYPE) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = target
    return info, None


@pytest.fixture
def layout(tmp_path: Path) -> tuple[Path, Path]:
    """``dest`` to extract into, and ``outside`` -- a sibling a slip would hit."""
    dest = tmp_path / "dest"
    outside = tmp_path / "outside"
    dest.mkdir()
    outside.mkdir()
    (outside / "victim.txt").write_text("original\n", encoding="utf-8")
    return dest, outside


def _assert_outside_untouched(outside: Path) -> None:
    assert sorted(p.name for p in outside.iterdir()) == ["victim.txt"]
    assert (outside / "victim.txt").read_text(encoding="utf-8") == "original\n"


@pytest.mark.parametrize("use_filter", PATHS)
def test_a_benign_archive_is_extracted(layout: tuple[Path, Path], use_filter: bool) -> None:
    """Positive control: the refusals below must not be a refusal of everything."""
    if use_filter and not hasattr(tarfile, "data_filter"):
        pytest.skip("this interpreter's tarfile has no extraction filters")
    dest, outside = layout
    aa._extract_archive(_tar(_file("src/mind_mem/tool.py", b"x = 1\n")), str(dest), use_data_filter=use_filter)
    assert (dest / "src/mind_mem/tool.py").read_bytes() == b"x = 1\n"
    _assert_outside_untouched(outside)


@pytest.mark.parametrize("use_filter", PATHS)
@pytest.mark.parametrize(
    ("archive", "reason"),
    [
        pytest.param(lambda outside: _tar(_file("../outside/victim.txt")), "escapes", id="dotdot-path"),
        pytest.param(lambda outside: _tar(_file("a/../../outside/victim.txt")), "escapes", id="nested-dotdot"),
        pytest.param(lambda outside: _tar(_file(str(outside / "absolute-slip.txt"))), "absolute", id="absolute-path"),
        pytest.param(
            lambda outside: _tar(_link("escape", "../outside"), _file("escape/victim.txt")),
            "link",
            id="symlink-escape",
        ),
        pytest.param(lambda outside: _tar(_link("hard", "../outside/victim.txt", tarfile.LNKTYPE)), "link", id="hardlink"),
        pytest.param(
            lambda outside: _device("dev"),
            "not a file or directory",
            id="device",
        ),
    ],
)
def test_a_hostile_archive_is_refused(layout: tuple[Path, Path], use_filter: bool, archive, reason: str) -> None:
    if use_filter and not hasattr(tarfile, "data_filter"):
        pytest.skip("this interpreter's tarfile has no extraction filters")
    dest, outside = layout
    with pytest.raises(aa.AuthorityError, match=reason):
        aa._extract_archive(archive(outside), str(dest), use_data_filter=use_filter)
    _assert_outside_untouched(outside)
    # Validation runs before any write, so a refused archive leaves nothing.
    assert list(dest.iterdir()) == []


def _device(name: str) -> bytes:
    info = tarfile.TarInfo(name)
    info.type = tarfile.CHRTYPE
    info.devmajor, info.devminor = 1, 3
    return _tar((info, None))
