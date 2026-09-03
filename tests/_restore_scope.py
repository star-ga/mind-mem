# Copyright 2026 STARGA, Inc.
"""The RESTORE admission every ``BlockStore.restore`` requires from 5.0.2 on.

``restore`` is the third mutation on a store, beside ``write_block`` and
``delete_block``, and until 5.0.2 it was the only one held up by convention:
every sanctioned caller went through ``apply_engine.restore_snapshot``, and
nothing made a caller that did not fail. It is checked at the seam now, so a
test that exercises restore mechanics has to open the scope the sanctioned
caller opens.

One definition, imported by every such test, for the reason the production
seam has one: three hand-rolled copies of "what a restore receipt looks like"
would drift, and the copy that drifted first would be the one that quietly
stopped matching the gate. A real
:class:`~mind_mem.governance_gate.GovernanceGate` scope is opened here — never
a stand-in receipt — so a test using this helper fails if the gate stops
minting the shape ``require_restore_admission`` accepts.
"""

from __future__ import annotations

import contextlib
from typing import Iterator, Sequence

from mind_mem.admission import RESTORE_TIER
from mind_mem.apply_engine import RESTORE_VERB
from mind_mem.governance_gate import get_gate


@contextlib.contextmanager
def restoring(
    workspace: str,
    *,
    batch_id: str = "restore:unit-test",
    block_ids: Sequence[str] = ("D-001",),
    actor: str = "pytest",
) -> Iterator[None]:
    """Open the batch RESTORE scope ``apply_engine.restore_snapshot`` opens.

    Args:
        workspace: Workspace the gate is keyed on. A bare ``tmp_path`` is
            fine — the gate warns that the config is unbound and admits.
        batch_id: Identifies the restore in the chain record.
        block_ids: Ids the receipt covers. A restore reinstates a set and
            withdraws another, which is why the receipt is a batch rather
            than one block.
        actor: Recorded in the receipt.
    """
    with get_gate(workspace).admit_batch(
        action=RESTORE_VERB,
        batch_id=batch_id,
        block_ids=list(block_ids),
        content="[]",
        tier=RESTORE_TIER,
        actor=actor,
    ):
        yield
