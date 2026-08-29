# Copyright 2026 STARGA, Inc.
"""``review_preview`` — the pre-apply diff ``mm review`` shows inline.

The diff must be produced by the production op executors against a
sandbox copy, never by a second implementation of op semantics: a diff
that disagrees with the apply is worse than no diff.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _review_fixtures import DECISION_FILE, build_workspace, mcp_budget  # noqa: E402,F401


@pytest.fixture
def workspace(tmp_path):
    root = str(tmp_path / "ws")
    os.makedirs(root)
    ids = build_workspace(root, 2)
    return root, ids


class TestPreviewDiff:
    def test_shows_the_field_the_proposal_would_change(self, workspace):
        from mind_mem.review_preview import preview_diff
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        result = preview_diff(root, load_queue(root)[0])
        assert result.available
        assert "-Tags: baseline" in result.diff_text
        assert "+Tags: baseline,reviewed-1" in result.diff_text
        assert DECISION_FILE in result.diff_text

    def test_leaves_the_real_workspace_byte_identical(self, workspace):
        from mind_mem.review_preview import preview_diff
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        target = os.path.join(root, DECISION_FILE)
        with open(target, "rb") as handle:
            before = handle.read()
        preview_diff(root, load_queue(root)[0])
        with open(target, "rb") as handle:
            assert handle.read() == before

    def test_writes_nothing_at_all_under_the_workspace(self, workspace):
        from mind_mem.review_preview import preview_diff
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        before = _tree(root)
        preview_diff(root, load_queue(root)[0])
        assert _tree(root) == before

    def test_removes_its_sandbox(self, workspace):
        from mind_mem.review_preview import preview_diff
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        result = preview_diff(root, load_queue(root)[0])
        assert result.available
        assert result.sandbox_removed

    def test_reports_unavailable_when_a_touched_file_is_missing(self, workspace):
        from mind_mem.review_preview import preview_diff
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        item = load_queue(root)[0]
        os.remove(os.path.join(root, DECISION_FILE))
        result = preview_diff(root, item)
        assert not result.available
        assert "not found" in result.reason.lower()

    def test_reports_unavailable_for_an_invalid_proposal(self, workspace):
        from mind_mem.review_preview import preview_diff
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        path = os.path.join(root, "intelligence/proposed/EDITS_PROPOSED.md")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text.replace("Risk: low", "Risk: catastrophic", 1))
        result = preview_diff(root, load_queue(root)[0])
        assert not result.available
        assert "valid" in result.reason.lower()

    def test_preview_adds_no_entry_to_the_workspace_governance_chain(self, workspace):
        """A preview is not an apply; the real chain must not record one."""
        from mind_mem.governance_gate import get_gate
        from mind_mem.review_preview import preview_diff
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        before = get_gate(root).chain.length
        assert preview_diff(root, load_queue(root)[0]).available
        assert get_gate(root).chain.length == before

    def test_diff_matches_what_the_apply_actually_writes(self, workspace):
        """The preview is only worth showing if it predicts the apply."""
        import contextlib
        import io

        from mind_mem.apply_engine import apply_proposal
        from mind_mem.review_preview import preview_diff
        from mind_mem.review_queue import load_queue

        root, ids = workspace
        predicted = preview_diff(root, load_queue(root)[0]).diff_text
        with contextlib.redirect_stdout(io.StringIO()):
            ok, message = apply_proposal(root, ids[0], dry_run=False)
        assert ok, message
        with open(os.path.join(root, DECISION_FILE), encoding="utf-8") as handle:
            applied_text = handle.read()
        for line in predicted.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                assert line[1:] in applied_text


def _tree(root: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(dirnames)
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            out[os.path.relpath(full, root)] = os.path.getsize(full)
    return out
