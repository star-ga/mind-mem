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
