"""``_check_cross_refs`` must not report an integrity property it never tested.

Section 5 computed ``dangling = referenced - defined`` and, on an empty
``dangling``, logged the PASS "All cross-references resolve to defined IDs".
An empty ``referenced`` set makes that empty for the one reason that proves
nothing: no reference was found anywhere in the corpus. A workspace whose
governed corpus had been wiped by a bad write therefore scored the same
green PASS as a healthy one — which is the vacuity already fixed in
``_check_blocks``, ``_check_provenance`` and ``_check_signatures_v11``,
left standing in the one section that phrases its claim universally.

Both assertions below fail on the pre-fix tree, where the empty corpus
produced a PASS.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from mind_mem.validate_py import Validator


@pytest.fixture()
def ws(tmp_path: Path) -> str:
    from mind_mem.init_workspace import init

    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    init(workspace)
    return workspace


def _run_cross_refs(workspace: str) -> Validator:
    v = Validator(workspace)
    v._check_cross_refs()
    return v


class TestCrossRefVacuity:
    def test_empty_corpus_does_not_claim_all_references_resolve(self, ws: str) -> None:
        v = _run_cross_refs(ws)
        report = "\n".join(v.lines)

        assert "All cross-references resolve" not in report, report
        assert "nothing to resolve" in report
        assert v.warnings == 1
        assert v.issues == 0  # a reference-free workspace is not an error

    def test_a_resolving_reference_still_passes_and_is_counted(self, ws: str) -> None:
        """No-regression control, plus the count that makes the claim checkable."""
        decisions = Path(ws) / "decisions" / "DECISIONS.md"
        decisions.write_text(
            "\n[D-20260830-001]\nStatement: The gate refuses drifted writes.\nStatus: active\n\n---\n",
            encoding="utf-8",
        )
        tasks = Path(ws) / "tasks" / "TASKS.md"
        tasks.parent.mkdir(parents=True, exist_ok=True)
        tasks.write_text(
            "\n[T-20260830-001]\nStatement: Follow up on D-20260830-001.\nStatus: active\n\n---\n",
            encoding="utf-8",
        )

        v = _run_cross_refs(ws)
        report = "\n".join(v.lines)

        # The count is part of the claim now: "all of them" is only
        # checkable when the report says how many "them" were.
        match = re.search(r"All (\d+) cross-references resolve to defined IDs", report)
        assert match is not None, report
        assert int(match.group(1)) > 0
        assert v.issues == 0

    def test_a_dangling_reference_still_fails(self, ws: str) -> None:
        """No-regression control on the other branch."""
        tasks = Path(ws) / "tasks" / "TASKS.md"
        tasks.parent.mkdir(parents=True, exist_ok=True)
        tasks.write_text(
            "\n[T-20260830-001]\nStatement: Follow up on D-19990101-001.\nStatus: active\n\n---\n",
            encoding="utf-8",
        )

        v = _run_cross_refs(ws)

        assert v.issues == 1
        assert "MISSING: D-19990101-001" in "\n".join(v.lines)
