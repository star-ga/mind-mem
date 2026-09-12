"""L1 — recall utility per context token, measured on our own corpus.

ROADMAP L1 asks for "a measured curve of utility against max_tokens, on our
traffic, with the honest possibility that the current default (2000) is already at
or past the knee. A result showing no headroom closes this as a negative finding
and is a valid outcome -- the point is to make the claim measurable, not to make
it flattering."

MEASURED 2026-09-11, 6 relevant blocks among 54 decoys, through the real
pack_recall_budget tool:

    max_tokens  included  relevant  spent  relevant/1k tokens
           100         3         3     66      45.45
           250         6         6    132      45.45
           500         6         6    132      45.45   <- no gain
          1000         6         6    132      45.45   <- no gain
          2000         6         6    132      45.45   <- no gain   (the DEFAULT)
          4000         6         6    132      45.45   <- no gain
          8000         6         6    132      45.45   <- no gain

THE FINDING: the knee is at ~250 tokens and the shipped default of 2000 is about
8x past it. Every relevant block is delivered by 250; the remaining 1,750 tokens
of default budget buy nothing on this shape of query. Utility per token is FLAT
above the knee rather than declining, because the packer stops when the ranked
list is exhausted -- it does not pad to the budget. So the default is wasteful of
*allowance*, not of actual context: a caller sizing a window around 2000 reserves
8x what the packer will spend.

That is a negative finding and L1 says so explicitly. What it closes: the claim is
now measurable and measured. What it does NOT license: lowering the default. This
is one query shape on a synthetic corpus; a query whose relevant set is genuinely
larger would need the headroom, and the honest next step is the same measurement
against real traffic, which needs the ground-truth set in
docs/design/eval-set-ground-truth.md.

These tests pin the curve so a future ranking change that quietly starts spending
the whole budget shows up as a red test with numbers.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from mind_mem.init_workspace import init
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.recall import pack_recall_budget

_QUERY = "compiler determinism byte-identical"
_N_RELEVANT = 6


@pytest.fixture
def corpus(tmp_path):
    ws = os.path.join(str(tmp_path), "ws")
    os.makedirs(ws, exist_ok=True)
    init(ws)
    rows = [
        f"[D-REL-{i:03d}]\nType: Decision\n"
        f"Statement: The compiler determinism gate requires byte-identical output "
        f"across substrates, case {i}\nStatus: Active\nDate: 2026-01-0{i + 1}\n\n"
        for i in range(_N_RELEVANT)
    ]
    rows += [
        f"[D-NOI-{i:03d}]\nType: Decision\n"
        f"Statement: Routine scheduling note {i} about meeting cadence and travel "
        f"policy\nStatus: Active\nDate: 2026-02-01\n\n"
        for i in range(54)
    ]
    Path(ws, "decisions", "DECISIONS.md").write_text("".join(rows), encoding="utf-8")
    return ws


def _measure(ws, max_tokens):
    with use_workspace(ws):
        out = json.loads(pack_recall_budget(_QUERY, max_tokens=max_tokens, limit=40))
    inc = out.get("included") or []
    relevant = sum(1 for r in inc if "REL" in str(r.get("_id") or ""))
    spent = sum(int(r.get("_token_cost") or 0) for r in inc)
    return {"included": len(inc), "relevant": relevant, "spent": spent}


def test_the_measurement_finds_the_relevant_set_at_all(corpus):
    """POSITIVE CONTROL. Every assertion below is meaningless if this fails."""
    got = _measure(corpus, 2000)
    assert got["relevant"] == _N_RELEVANT, got
    assert got["spent"] > 0, got


def test_the_default_budget_of_2000_is_past_the_knee(corpus):
    """L1's honest possibility, confirmed: the default buys nothing extra."""
    at_knee = _measure(corpus, 250)
    at_default = _measure(corpus, 2000)
    assert at_knee["relevant"] == at_default["relevant"] == _N_RELEVANT
    assert at_knee["spent"] == at_default["spent"], (
        f"the default spent more than the knee ({at_default['spent']} vs "
        f"{at_knee['spent']}) — re-measure; the curve has changed shape"
    )


def test_quadrupling_the_budget_above_the_knee_adds_nothing(corpus):
    for bigger in (4000, 8000):
        got = _measure(corpus, bigger)
        assert got["relevant"] == _N_RELEVANT, (bigger, got)
        assert got["spent"] == _measure(corpus, 250)["spent"], (bigger, got)


def test_below_the_knee_the_budget_DOES_bind(corpus):
    """POSITIVE CONTROL for the three tests above.

    Without this they would pass on a packer that ignored max_tokens entirely --
    "no gain from more budget" and "budget has no effect" look identical.
    """
    tight = _measure(corpus, 100)
    assert tight["included"] < _N_RELEVANT, (
        f"a 100-token budget included {tight['included']} blocks; the budget is "
        f"not binding, so this measurement says nothing about a knee"
    )
    assert tight["spent"] <= 100


def test_the_packer_does_not_pad_to_the_budget(corpus):
    """Why the curve is flat rather than declining above the knee.

    The packer stops when the ranked list is exhausted. If it ever started
    padding, utility per token would fall with budget and this test goes red.
    """
    got = _measure(corpus, 8000)
    assert got["spent"] < 8000 / 4, (
        f"spent {got['spent']} of an 8000 budget — the packer appears to be "
        f"padding, which changes L1's conclusion"
    )
