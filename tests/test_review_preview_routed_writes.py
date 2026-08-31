# Copyright 2026 STARGA, Inc.
"""The preview must diff where the apply WRITES, not where the op says.

``_op_append_block`` hands its parsed block to ``store.write_block``, which
resolves the destination from the block-id prefix and ignores ``op["file"]``
entirely; ``validate_proposal`` never ties the two together. The preview built
its target list from ``FilesTouched`` plus each op's declared ``file``, so a
proposal declaring one file for a block routed to another wrote into a sandbox
path the preview neither seeded nor diffed — and reported
``available=False, reason="proposal would change nothing"`` for a proposal that
really does add the block.

A preview that disagrees with the apply is worse than no preview, and this
disagreed in the most expensive direction: it told the operator a real write
was a no-op.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _review_fixtures import DECISION_FILE, build_workspace, mcp_budget  # noqa: E402,F401

TASKS_FILE = "tasks/TASKS.md"
NEW_ID = "D-20260829-777"

NEW_BLOCK = (
    f"[{NEW_ID}]\n"
    "Type: decision\n"
    "Status: active\n"
    "Statement: Route previews through the file the store actually writes.\n"
    "Scope: global\n"
    "Rationale: the preview must not disagree with the apply.\n"
    "Supersedes: none\n"
    "Date: 2026-08-29\n"
    "Tags: preview\n"
    "Sources:\n- seed\n"
)


@pytest.fixture
def workspace(tmp_path: Any) -> str:
    root = str(tmp_path / "ws")
    os.makedirs(root)
    build_workspace(root, 2)
    tasks = os.path.join(root, TASKS_FILE)
    os.makedirs(os.path.dirname(tasks), exist_ok=True)
    if not os.path.isfile(tasks):
        with open(tasks, "w", encoding="utf-8") as handle:
            handle.write("# Tasks\n")
    return root


def _item(declared_file: str) -> Any:
    """An append_block proposal declaring *declared_file* for a ``D-`` block."""
    from mind_mem.review_queue import ReviewItem

    return ReviewItem(
        proposal_id="P-20260829-901",
        source_file="intelligence/proposed/EDITS_PROPOSED.md",
        proposal_type="add",
        target_block=NEW_ID,
        risk="low",
        status="staged",
        created="2026-08-29T00:00:00Z",
        rollback="delete the block",
        fingerprint="",
        files_touched=(declared_file,),
        ops=({"op": "append_block", "file": declared_file, "patch": NEW_BLOCK},),
    )


def test_declared_file_matching_the_routing_still_previews(workspace: str) -> None:
    """Control: the in-tree shape (declared file == routed file) is unaffected."""
    from mind_mem.review_preview import preview_diff

    result = preview_diff(workspace, _item(DECISION_FILE))
    assert result.available, result.reason
    assert NEW_ID in result.diff_text
    assert DECISION_FILE in result.diff_text


def test_a_routed_write_is_not_reported_as_changing_nothing(workspace: str) -> None:
    """The defect: op declares tasks/, ``write_block`` routes to decisions/."""
    from mind_mem.review_preview import preview_diff

    result = preview_diff(workspace, _item(TASKS_FILE))
    assert result.available, f"a real write previewed as unavailable: {result.reason}"
    assert result.reason == ""
    assert NEW_ID in result.diff_text
    assert DECISION_FILE in result.diff_text, "the diff must name the file the block lands in"
    assert DECISION_FILE in result.files


def test_the_preview_matches_what_an_apply_would_write(workspace: str) -> None:
    """The property the whole module exists for, on the routed path.

    Replays the same op through the production executor against a throwaway
    COPY of the workspace and compares the resulting file to the one the
    preview's diff described. A preview that agrees with a second preview
    proves nothing; agreeing with the executor is the claim.
    """
    import shutil

    from mind_mem.review_preview import preview_diff

    result = preview_diff(workspace, _item(TASKS_FILE))
    assert result.available

    applied = workspace + "-applied"
    shutil.copytree(workspace, applied)
    _execute_append(applied)

    with open(os.path.join(applied, DECISION_FILE), encoding="utf-8") as handle:
        after = handle.read()
    with open(os.path.join(workspace, DECISION_FILE), encoding="utf-8") as handle:
        before = handle.read()

    assert NEW_ID in after, "the executor really does write the block"
    assert NEW_ID not in before, "and the preview left the workspace alone"
    added = [line[1:] for line in result.diff_text.splitlines() if line.startswith("+") and not line.startswith("+++")]
    assert any(NEW_ID in line for line in added)


def _execute_append(root: str) -> None:
    from mind_mem.apply_engine import execute_op
    from mind_mem.block_store import MarkdownBlockStore
    from mind_mem.governance_gate import evict_gate, get_gate

    store = MarkdownBlockStore(root)
    gate = get_gate(root)
    try:
        with gate.admit_proposal(
            "P-20260829-901",
            "[]",
            actor="test",
            target_file="intelligence/proposed/EDITS_PROPOSED.md",
            metadata={"phase": "test"},
        ):
            ok, message = execute_op(root, {"op": "append_block", "file": TASKS_FILE, "patch": NEW_BLOCK}, store=store)
        assert ok, message
    finally:
        evict_gate(root)


def test_workspace_stays_byte_identical_across_the_routed_preview(workspace: str) -> None:
    """The routed path widens what is seeded; it must not widen what is written."""
    from mind_mem.review_preview import preview_diff

    before = _tree(workspace)
    result = preview_diff(workspace, _item(TASKS_FILE))
    assert result.sandbox_removed
    assert _tree(workspace) == before


def _tree(root: str) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for base, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(base, name)
            with open(path, "rb") as handle:
                out[os.path.relpath(path, root)] = handle.read()
    return out
