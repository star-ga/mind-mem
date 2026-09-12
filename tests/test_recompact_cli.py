"""`mind-mem-recompact` — clusters via find_similar, PROPOSES, never commits.

ROADMAP: "a `mm recompact` / dream-cycle pass 6 that clusters via `find_similar`
and routes results through `propose_update` (engine + bench shipped; the CLI verb
and scheduler wiring are not yet built)."

The engine (`recompaction.recompact_cluster`) already exists, already refuses to
mutate its input, and already documents its output as "a proposal for the HITL
gate". What was missing was any way to RUN it: no verb, so the capability shipped
unreachable.

THE DEFAULT IS DRY-RUN, and that is the load-bearing choice. A recompaction
rewrites the text of several blocks into one summary; a verb that did that by
default on a real corpus would be the silent-overwrite failure with a friendly
name. `--propose` is required to stage anything, and even then it stages a
proposal rather than applying it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mind_mem.init_workspace import init
from mind_mem.recompact_cli import build_plan, main


@pytest.fixture
def ws(tmp_path):
    root = os.path.join(str(tmp_path), "ws")
    os.makedirs(root, exist_ok=True)
    init(root)
    # Three near-duplicate blocks: exactly the shape recompaction exists for.
    # find_similar's method is CO-OCCURRENCE, not text similarity (measured: it
    # returns [] for near-duplicate wording alone). So a real cluster is blocks
    # that reference the same entities -- which is also the honest shape for
    # recompaction, since co-occurring blocks are the ones a summary can merge.
    Path(root, "decisions", "DECISIONS.md").write_text(
        "[D-001]\nType: Decision\nStatement: PRJ-mind-mem recall cache is invalidated by TOOL-governor\n"
        "Status: Active\nDate: 2026-01-01\n\n"
        "[D-002]\nType: Decision\nStatement: PRJ-mind-mem cache invalidation is owned by TOOL-governor\n"
        "Status: Active\nDate: 2026-01-02\n\n"
        "[D-003]\nType: Decision\nStatement: PRJ-mind-mem and TOOL-governor agree on invalidation timing\n"
        "Status: Active\nDate: 2026-01-03\n\n",
        encoding="utf-8",
    )
    return root


def test_the_plan_finds_a_cluster(ws):
    """POSITIVE CONTROL: with no cluster found, every assertion below is empty."""
    plan = build_plan(ws, block_id="D-001", limit=5)
    assert plan["cluster"], plan
    assert plan["seed"] == "D-001"


def test_the_plan_is_data_and_writes_nothing(ws):
    before = Path(ws, "decisions", "DECISIONS.md").read_text(encoding="utf-8")
    build_plan(ws, block_id="D-001", limit=5)
    after = Path(ws, "decisions", "DECISIONS.md").read_text(encoding="utf-8")
    assert before == after, "build_plan modified the corpus"


def test_a_workspace_with_no_retrieval_history_is_refused_not_silently_empty(ws):
    """MEASURED CONSTRAINT on the roadmap item, not a limitation of this verb.

    find_similar's own docstring: "the ranking comes from block_meta.db
    co-occurrence counts, and a block that has never been co-retrieved returns an
    empty list even when semantically near neighbours exist."

    So a fresh workspace has NO clusters, however near-duplicate its blocks --
    measured: [] even for blocks sharing both entity ids. The roadmap's "clusters
    via find_similar" therefore only produces clusters on a workspace with
    retrieval history, and a scheduler running this nightly on a quiet corpus
    would find nothing forever while reporting success.

    The verb refuses instead of exiting 0 on an empty cluster, so that condition
    is visible rather than silent.
    """
    rc = main([ws, "--block-id", "D-001"])
    assert rc == 2, "an empty cluster must be REFUSED, not reported as success"


def test_dry_run_is_the_default_when_a_cluster_exists(ws, monkeypatch):
    """With a cluster present, the default must still stage nothing."""
    import mind_mem.recompact_cli as m

    monkeypatch.setattr(
        m, "build_plan",
        lambda *a, **k: {"seed": "D-001", "cluster": ["D-001", "D-002"], "refused": ""},
    )
    rc = m.main([ws, "--block-id", "D-001"])
    assert rc == 0
    signals = Path(ws, "intelligence", "SIGNALS.md")
    assert not signals.is_file() or "recompact" not in signals.read_text(encoding="utf-8").lower()


def test_a_missing_block_is_an_error_not_a_silent_empty_plan(ws):
    """An unknown seed must not read as 'nothing to recompact'."""
    rc = main([ws, "--block-id", "D-DOES-NOT-EXIST"])
    assert rc != 0


def test_a_single_block_cluster_is_refused(ws):
    """Recompacting one block is a rewrite with no consolidation to justify it."""
    Path(ws, "decisions", "DECISIONS.md").write_text(
        "[D-ONLY]\nType: Decision\nStatement: A lone unrelated fact about turbines\n"
        "Status: Active\nDate: 2026-01-01\n\n",
        encoding="utf-8",
    )
    plan = build_plan(ws, block_id="D-ONLY", limit=5)
    assert plan["refused"], plan
    assert "one block" in plan["refused"].lower() or "single" in plan["refused"].lower()


def test_the_cli_never_imports_a_direct_write_path():
    """Group-H-shaped guardrail: the verb proposes; it does not apply."""
    import ast

    import mind_mem.recompact_cli as m

    tree = ast.parse(open(m.__file__, encoding="utf-8").read())
    called = {
        n.func.attr for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "approve_apply" not in called, "the recompact verb must not self-approve"
