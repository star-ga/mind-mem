# Copyright 2026 STARGA, Inc.
"""The suite must not abandon workspaces in the real ``/tmp``.

115 call sites in these tests do ``tempfile.mkdtemp(prefix="mm_...")`` with no
``dir=`` and no cleanup. Left alone that leaked **12,141 directories holding
469,413 inodes** — 57% of every inode on the tmpfs — and once the inode table
filled, ``shutil.copy2`` began failing inside ``init_workspace``. The suite
then reported dozens of errors that looked like broken product code and were
not, and the run died mid-traceback because pytest could no longer write its
own output.

``conftest._contain_bare_tempfiles`` fixes that centrally. This file exists so
that removing it fails loudly and immediately, rather than three weeks later
as an unexplained flood of unrelated errors.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def test_bare_mkdtemp_lands_under_the_pytest_tmp_tree(tmp_path_factory) -> None:
    """A no-``dir=`` mkdtemp must not land in the real /tmp.

    The invariant is containment under pytest's *basetemp*, not under this
    test's own ``tmp_path``. The containment fixture deliberately points at a
    sibling directory, because several tests scan their own ``tmp_path`` for
    stray files and would otherwise report the scratch dir as a leak from the
    code under test.
    """
    base = tmp_path_factory.getbasetemp()
    made = Path(tempfile.mkdtemp(prefix="mm_containment_probe_"))
    assert made.is_relative_to(base), (
        f"{made} escaped the pytest tmp tree ({base}) — the autouse "
        "containment fixture in conftest.py is gone, and this suite is "
        "leaking a workspace per call into /tmp again"
    )


def test_bare_named_temporary_file_is_contained_too(tmp_path_factory) -> None:
    """``tempfile.tempdir`` governs the whole module, not just mkdtemp."""
    base = tmp_path_factory.getbasetemp()
    with tempfile.NamedTemporaryFile(prefix="mm_containment_probe_", delete=False) as fh:
        made = Path(fh.name)
    assert made.is_relative_to(base), f"{made} escaped {base}"


def test_positive_control_the_probe_can_observe_an_escape(tmp_path) -> None:
    """An explicit ``dir=`` still goes where it is told.

    Without this the two assertions above would also pass against a
    ``tempfile`` that had been stubbed out entirely — the tests would be
    checking nothing. This proves the probe can still see a path that is
    NOT under ``tmp_path``.
    """
    outside = tmp_path
    made = Path(tempfile.mkdtemp(prefix="mm_containment_control_", dir=str(outside)))
    try:
        assert made.is_relative_to(tmp_path), (
            "an explicit dir= was overridden; the probe cannot distinguish contained from escaped, so the tests above prove nothing"
        )
    finally:
        made.rmdir()
