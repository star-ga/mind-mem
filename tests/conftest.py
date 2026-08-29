"""Shared test fixtures.

Currently one thing: a real governance admission for tests that exercise
``BlockStore.write_block`` directly.

Since the write invariant landed, ``write_block`` refuses any write with no
open admission. That is the point of it. Storage-layer unit tests still need
to call it directly, so they open a **real** admission through the gate --
a genuine chain entry, not a stubbed receipt.

There is deliberately no test-only constructor, no wildcard receipt and no
environment escape hatch in ``src/``. An invariant with a bypass reserved for
tests is not an invariant, and this project has already been bitten twice by
checks that reported success over work they never inspected.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import pytest


@pytest.fixture
def admitted(tmp_path) -> Iterator[None]:
    """Open a real governance admission for the duration of one test.

    Uses ``admit_proposal``, whose receipt covers any block id for its scope --
    the same ambient authority ``apply_engine`` runs under while applying an
    approved proposal. It is a real ``admit`` call writing a real chain entry;
    nothing here forges a receipt.
    """
    from mind_mem.governance_gate import get_gate

    workspace = str(tmp_path)
    gate = get_gate(workspace)
    if gate is None:  # pragma: no cover - defensive
        yield
        return
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            gate.admit_proposal(
                proposal_id="TEST-ADMISSION",
                content="[]",
                actor="pytest",
            )
        )
        yield


@pytest.fixture(autouse=True, scope="session")
def _child_processes_speak_utf8() -> Iterator[None]:
    """Make every subprocess this suite spawns write UTF-8.

    Many tests run ``mm_cli`` through ``subprocess.run(..., encoding="utf-8")``.
    On Windows a child Python writes stdout in the console codepage (cp1252),
    so any non-ASCII character the CLI prints comes back as bytes the parent
    cannot decode. The decode happens on subprocess's reader THREAD, where the
    UnicodeDecodeError does not propagate -- ``run`` returns with
    ``stdout=None`` and the test dies on ``"..." in None`` with a TypeError
    that says nothing about encoding. Measured 2026-08-29: an em dash in
    ``mm resume`` output (0x97 in cp1252) failed every Windows and macOS matrix
    row while passing on Linux, whose default is already UTF-8.

    Asking the child for UTF-8 is the other half of the parent already asking to
    decode UTF-8; without it the pair only agrees by accident of platform.
    Ten test modules spawn ``mm_cli`` and none of them set this, so it belongs
    here rather than in one helper.
    """
    import os

    previous = {k: os.environ.get(k) for k in ("PYTHONUTF8", "PYTHONIOENCODING")}
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
