"""M2 — namespace retrieval reachability, asserted EMPIRICALLY.

ROADMAP M2: "Assert empirically, per namespace, what is reachable by search
versus only by direct get, with the retrieval scores printed ... an
index-configuration regression currently degrades recall silently instead of
failing loudly. NamespaceManager governs read/write ACLs but nothing asserts
retrieval REACHABILITY as a tested property."

That is today's recurring defect one layer out. `can_read` is an ACL predicate;
it says what a policy PERMITS. It does not say what `recall()` actually RETURNS.
Those two can drift -- an index that stops covering a namespace still passes
every ACL test, because the ACL is not what broke. The corpus is then quietly
half-invisible and nothing anywhere says so.

So this file asserts the end-to-end property through the real recall entry
point, and prints the scores it saw, so a degradation shows up as a red test
with numbers rather than as silence.

The mutation control at the bottom is the part that makes the rest mean
anything: it proves these assertions can fail.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mind_mem._recall_core import recall
from mind_mem.init_workspace import init
from mind_mem.namespaces import NamespaceManager

_ACL = {
    "default_policy": "read",
    "agents": {
        "alpha-1": {"namespaces": ["agents/alpha-1"], "write": ["agents/alpha-1"],
                     "read": ["shared", "agents/alpha-1"]},
        "beta-1": {"namespaces": ["agents/beta-1"], "write": ["agents/beta-1"],
                    "read": ["shared", "agents/beta-1"]},
        "*": {"namespaces": ["shared"], "write": [], "read": ["shared"]},
    },
}

#: A term unique to one namespace each, so a hit can only come from that corpus.
_MARKERS = {
    "shared": "zirconshared",
    "agents/alpha-1": "quartzitealpha",
    "agents/beta-1": "feldsparbeta",
}


def _write_block(path: Path, bid: str, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"[{bid}]\nType: Decision\nStatement: The {marker} appears once here\nStatus: Active\n\n",
        encoding="utf-8",
    )


@pytest.fixture
def ws(tmp_path):
    root = str(tmp_path / "ws")
    os.makedirs(root)
    init(root)
    Path(root, "mind-mem-acl.json").write_text(json.dumps(_ACL), encoding="utf-8")
    # shared lives in the workspace's own decisions/ dir; agent namespaces get
    # their own tree via NamespaceManager, which is the surface under test.
    _write_block(Path(root, "decisions", "ns_shared.md"), "D-SHARED-001", _MARKERS["shared"])
    for ns in ("agents/alpha-1", "agents/beta-1"):
        NamespaceManager(root, agent_id=None).init_namespace(ns)
        _write_block(Path(root, ns, "decisions", "DECISIONS.md"),
                     f"D-{ns.replace('/', '-').upper()}-001", _MARKERS[ns])
    return root


def _search(ws, term: str, agent_id):
    hits = recall(ws, term, limit=10, agent_id=agent_id) or []
    return [(h.get("_id", h.get("block_id", "?")), round(float(h.get("score", 0) or 0), 4)) for h in hits]


@pytest.mark.parametrize("ns", sorted(_MARKERS))
def test_a_namespaces_own_block_is_reachable_by_search(ws, ns, capsys):
    """Reachability, not permission. The scores are printed on purpose."""
    agent = None if ns == "shared" else ns.split("/", 1)[1]
    got = _search(ws, _MARKERS[ns], agent)
    print(f"  [{ns}] agent={agent!r} marker={_MARKERS[ns]!r} -> {got}")
    assert got, (
        f"namespace {ns!r} is INVISIBLE to search for an agent the ACL permits to read it. "
        f"can_read passing while recall returns nothing is exactly the silent "
        f"index-configuration regression M2 exists to make loud."
    )


def test_an_agent_does_not_see_another_agents_private_namespace(ws, capsys):
    """The negative half. Without it the test above passes on a broken ACL."""
    got = _search(ws, _MARKERS["agents/beta-1"], "alpha-1")
    print(f"  [alpha-1 searching beta's marker] -> {got}")
    assert got == [], f"alpha-1 reached beta-1's private namespace: {got}"


def test_the_reachability_assertion_can_actually_fail(ws, capsys):
    """MUTATION CONTROL. A test that cannot fail proves nothing.

    Remove the namespace's corpus file and the same assertion must go red. If
    this ever passes, the searches above are not measuring reachability.
    """
    target = Path(ws, "agents/alpha-1", "decisions", "DECISIONS.md")
    assert target.is_file(), "positive control: the corpus file must exist first"
    before = _search(ws, _MARKERS["agents/alpha-1"], "alpha-1")
    assert before, "positive control: it must be reachable BEFORE the mutation"

    target.unlink()
    after = _search(ws, _MARKERS["agents/alpha-1"], "alpha-1")
    print(f"  before={before}  after removing the corpus file={after}")
    assert after == [], (
        "the reachability search still returned hits after its only corpus file "
        "was removed, so it is not reading that corpus at all"
    )
