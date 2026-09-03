"""The recall cache belongs to a corpus state, not to a clock.

``recall_cache`` used to be a TTL cache invalidated by three explicit calls in
the governance MCP tools. That is invalidation "by whoever remembered": a
governed write that lands through the CLI, the HTTP transport, the apply engine
or federation replication calls none of them, and for the rest of the TTL window
the surface served the pre-write answer — while ``_apply_attestation`` stamped
the *post*-write ``index_anchor`` onto it. The run then attested a corpus state
it had not read, and replaying at that anchor did not reproduce the answer.

The fix is a key, not a call: the governed-ledger head is part of the cache key, so
an entry computed against a superseded corpus is unresolvable. These tests hold
that property, and each negative assertion is paired with a positive control
proving the cache was really caching and the method could see the staleness it
says is gone.
"""

from __future__ import annotations

import json
import os

import pytest

from mind_mem.recall_cache import cached_recall, make_cache_key, reset_singleton


@pytest.fixture(autouse=True)
def _fresh_singletons():
    from mind_mem import prefetch

    reset_singleton()
    prefetch.reset_cache()
    yield
    reset_singleton()
    prefetch.reset_cache()


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


class TestTheAnchorIsInTheKey:
    def test_two_anchors_are_two_keys(self) -> None:
        assert make_cache_key("q", index_anchor="a" * 64) != make_cache_key("q", index_anchor="b" * 64)

    def test_the_same_anchor_is_the_same_key(self) -> None:
        assert make_cache_key("q", index_anchor="a" * 64) == make_cache_key("q", index_anchor="a" * 64)

    def test_omitting_the_anchor_still_yields_a_key(self) -> None:
        """Callers outside the recall path keep working; they just get one bucket."""
        assert make_cache_key("q").startswith("mindmem:recall:default:")

    def test_the_anchor_is_not_confused_with_the_instant(self) -> None:
        anchored = make_cache_key("q", index_anchor="a" * 64, scoring_instant="")
        instanted = make_cache_key("q", index_anchor="", scoring_instant="a" * 64)

        assert anchored != instanted


class TestCachedRecallHonoursTheAnchor:
    def test_the_same_anchor_serves_the_cached_answer(self) -> None:
        """POSITIVE CONTROL: prove the cache caches before asserting it stops."""
        calls: list[str] = []

        def inner(query, limit, backend, active_only):
            calls.append(query)
            return f"answer-{len(calls)}"

        first = cached_recall(inner, "q", index_anchor="a" * 64)
        second = cached_recall(inner, "q", index_anchor="a" * 64)

        assert (first, second) == ("answer-1", "answer-1")
        assert len(calls) == 1

    def test_a_moved_anchor_does_not_serve_the_old_answer(self) -> None:
        calls: list[str] = []

        def inner(query, limit, backend, active_only):
            calls.append(query)
            return f"answer-{len(calls)}"

        first = cached_recall(inner, "q", index_anchor="a" * 64)
        second = cached_recall(inner, "q", index_anchor="b" * 64)

        assert first == "answer-1"
        assert second == "answer-2"
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# The wiring: a real recall over a real workspace and a real chain
# ---------------------------------------------------------------------------


def _write_corpus(root: str, statement: str) -> None:
    os.makedirs(os.path.join(root, "decisions"), exist_ok=True)
    for sub in ("tasks", "entities", "intelligence"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    with open(os.path.join(root, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as fh:
        fh.write(f"[D-20260101-001]\nStatement: {statement}\nStatus: active\nDate: 2026-01-01\n\n")


def _cached_workspace(root: str) -> None:
    os.makedirs(root, exist_ok=True)
    _write_corpus(root, "the retrieval rollout note landed on the original wording")
    with open(os.path.join(root, "mind-mem.json"), "w", encoding="utf-8") as fh:
        json.dump({"cache": {"enabled": True, "ttl_seconds": 3600}}, fh)


def _excerpts(envelope: str) -> list[str]:
    return [str(hit.get("excerpt", "")) for hit in json.loads(envelope)["results"]]


def _append_to_governed_ledger(workspace: str, block_id: str, action: str, content: str) -> None:
    """Move the workspace's ``index_anchor`` the way a governed write does.

    ``memory/hash_chain_v2.db`` is the ledger the governance gate appends to on
    every admitted write and every admitted delete, and it is what
    :func:`mind_mem.recall_attestation._resolve_index_anchor` reads. Moving the
    head through the same ledger is what makes a passing test mean the cache
    retires on the events the product retires it on.
    """
    from mind_mem.hash_chain_v2 import HashChainV2
    from mind_mem.recall_attestation import index_anchor_ledger_path

    HashChainV2(index_anchor_ledger_path(workspace)).append(block_id, action, content)


class TestAGovernedWriteRetiresTheCachedAnswer:
    def test_an_ungoverned_corpus_edit_is_still_served_from_the_cache(self, tmp_path) -> None:
        """POSITIVE CONTROL, and an honestly-stated bound.

        Rewriting ``DECISIONS.md`` by hand is not a governed write: it appends
        nothing to the governed ledger, so the head does not move and the cached
        entry legitimately still belongs to the corpus state the cache knows
        about. This test exists to prove two things the test below depends on —
        that the cache is genuinely holding an answer across calls, and that
        this method can *see* a stale answer when there is one. Without it,
        "the stale answer was not served" would pass just as well if the cache
        had never stored anything.
        """
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.recall import _recall_impl

        ws = str(tmp_path / "ungoverned")
        _cached_workspace(ws)

        with use_workspace(ws):
            before = _recall_impl("retrieval rollout note", scoring_instant="2026-01-02")
            _write_corpus(ws, "the retrieval rollout note landed on the superseding wording")
            after = _recall_impl("retrieval rollout note", scoring_instant="2026-01-02")

        assert any("original" in e for e in _excerpts(before)), "recall served nothing — the test would be vacuous"
        assert any("original" in e for e in _excerpts(after)), "the cache did not hold the answer across the two calls"

    def test_a_governed_write_through_a_non_invalidating_door_is_not_served_stale(self, tmp_path) -> None:
        """The defect this closes.

        Appending to the governed ledger is what every admitted write does, from
        every door — the CLI, the HTTP transport, the apply engine, federation
        replication — and none of those calls ``_invalidate_recall_cache``; only
        three MCP tools do. Before the anchor was in the key, the second recall
        below returned the pre-write excerpt for the rest of the TTL window
        while the attestation stamped the post-write anchor onto it.
        """
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.recall import _recall_impl

        ws = str(tmp_path / "governed")
        _cached_workspace(ws)

        with use_workspace(ws):
            before = _recall_impl("retrieval rollout note", scoring_instant="2026-01-02")
            assert any("original" in e for e in _excerpts(before)), "recall served nothing — the test would be vacuous"

            _write_corpus(ws, "the retrieval rollout note landed on the superseding wording")
            _append_to_governed_ledger(ws, "D-20260101-001", "update", "supersede the wording")
            after = _recall_impl("retrieval rollout note", scoring_instant="2026-01-02")

        assert any("superseding" in e for e in _excerpts(after))
        assert not any("original" in e for e in _excerpts(after))

    def test_the_attested_anchor_is_the_anchor_the_answer_was_keyed_under(self, tmp_path) -> None:
        """The two must never be two different opinions of "which corpus".

        The cache key and the attestation both take the head from
        :func:`mind_mem.prefetch.chain_head` / ``_resolve_index_anchor``. This
        pins that they agree on a run that actually served results.

        The ledger is seeded first, deliberately: on an empty ledger both sides
        answer :data:`GENESIS_ANCHOR` and the equality holds for the one reason
        that proves nothing. Asserting the anchor is a *real* value is what
        makes the agreement an agreement.
        """
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.recall import _recall_impl
        from mind_mem.prefetch import chain_head
        from mind_mem.recall_attestation import GENESIS_ANCHOR

        ws = str(tmp_path / "attested")
        _cached_workspace(ws)
        _append_to_governed_ledger(ws, "D-20260101-001", "create", "seed the ledger")

        with use_workspace(ws):
            envelope = json.loads(_recall_impl("retrieval rollout note", scoring_instant="2026-01-02"))
            head = chain_head(ws)

        assert envelope["count"] >= 1, "recall served nothing — the assertions below would be vacuous"
        assert head != GENESIS_ANCHOR, "the ledger was empty — the equality below would hold trivially"
        assert envelope["attestation"]["index_anchor"] == head
