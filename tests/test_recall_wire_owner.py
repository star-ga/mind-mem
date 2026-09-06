# Copyright 2026 STARGA, Inc.
"""One explicit wire owner for ``recall``, independent of registration order.

Two tools registered the same wire name: the v3.1.x tool in ``tools.recall`` and
the v3.2.0 consolidated dispatcher in ``tools.public``. FastMCP's registry is a
dict, so the LAST registration won and the intended dispatcher survived only
because ``server.py`` happened to register ``public`` after ``recall``. Reversing
those two lines silently swapped the wire surface, and the only signal was a
warning that also fires in the correct case.

The redundant registration is removed rather than policed: the legacy Python
function stays importable and is still the implementation the dispatcher calls,
but it no longer claims the wire name. That makes the outcome order-independent
instead of order-dependent-and-warned.

``explain`` is restored as part of the same change. It is a v3.1.x ``recall``
parameter that the dispatcher never accepted, so the wire lost it the moment the
dispatcher started winning. Modes whose implementations cannot produce it refuse
the flag rather than accepting and dropping it.
"""

import inspect
import json

import pytest
from fastmcp import FastMCP

from mind_mem.mcp.tools import public as public_mod
from mind_mem.mcp.tools import recall as recall_mod

WIRE_KEY = "tool:recall@"
EXPECTED_OWNER = "mind_mem.mcp.tools.public"


def _components(mcp):
    for name in dir(mcp):
        obj = getattr(mcp, name, None)
        if hasattr(obj, "_components"):
            return obj._components
    for attr in ("_provider", "provider", "_local_provider"):
        obj = getattr(mcp, attr, None)
        if obj is not None and hasattr(obj, "_components"):
            return obj._components
    raise AssertionError("could not reach the provider's component store")


def _build(order):
    mcp = FastMCP("wire-owner-probe")
    steps = {"legacy": recall_mod.register, "public": public_mod.register}
    for key in order:
        steps[key](mcp)
    return mcp


def _owner(mcp):
    comp = _components(mcp)[WIRE_KEY]
    fn = getattr(comp, "fn", None) or getattr(comp, "_fn", None)
    return fn.__module__


class TestTheWireOwnerIsOrderIndependent:
    @pytest.mark.parametrize("order", [("legacy", "public"), ("public", "legacy")])
    def test_public_owns_the_recall_wire_name_in_either_order(self, order):
        """Reversed order silently swapped the surface before this change."""
        assert _owner(_build(order)) == EXPECTED_OWNER, f"registration order {order} changed the wire owner of {WIRE_KEY}"

    def test_the_tool_inventory_is_identical_in_either_order(self):
        a = sorted(k for k in _components(_build(("legacy", "public"))) if k.startswith("tool:"))
        b = sorted(k for k in _components(_build(("public", "legacy"))) if k.startswith("tool:"))
        assert a == b, "registration order changed the exposed tool set"

    def test_no_duplicate_component_warning_is_emitted(self, caplog):
        """The warning must go because the duplicate stops happening.

        Not because it was suppressed: a global on_duplicate change would hide
        real collisions elsewhere, so this asserts the absence of THIS warning
        while leaving the provider's default policy alone.
        """
        with caplog.at_level("WARNING"):
            _build(("legacy", "public"))
        dupes = [r for r in caplog.records if "already exists" in r.getMessage()]
        assert not dupes, [r.getMessage() for r in dupes]


class TestNothingElseIsLost:
    def test_every_other_recall_module_tool_is_still_registered(self):
        names = {k for k in _components(_build(("legacy", "public"))) if k.startswith("tool:")}
        for expected in (
            "tool:pack_recall_budget@",
            "tool:recall_with_axis@",
            "tool:hybrid_search@",
            "tool:find_similar@",
            "tool:intent_classify@",
            "tool:retrieval_diagnostics@",
            "tool:prefetch@",
        ):
            assert expected in names, f"{expected} disappeared with the legacy recall registration"

    def test_the_legacy_python_function_remains_callable(self):
        """A wire registration was removed, not the implementation."""
        assert callable(recall_mod.recall)
        assert recall_mod.recall.__module__ == "mind_mem.mcp.tools.recall"
        assert "explain" in inspect.signature(recall_mod.recall).parameters


class TestExplainIsPassedThroughNotDropped:
    def test_explain_is_accepted_by_the_dispatcher(self):
        params = inspect.signature(public_mod.recall).parameters
        assert "explain" in params, "the dispatcher still does not accept explain"

    @pytest.mark.parametrize("mode", ["auto", "bm25", "hybrid"])
    def test_explain_reaches_the_implementation_for_every_explaining_mode(self, mode, monkeypatch):
        """Forwarding is asserted over the whole supported set, not one member."""
        seen = {}

        def _spy(query, **kw):
            seen.update(kw)
            return "{}"

        monkeypatch.setattr(recall_mod, "_recall_impl", _spy)
        public_mod.recall(query="q", mode=mode, explain=True)
        assert seen.get("explain") is True, f"mode={mode}: explain did not reach the implementation"

    def test_the_backend_alias_also_selects_an_explaining_mode(self, monkeypatch):
        """v3.1.x callers pass backend=, not mode=. The alias must carry explain."""
        seen = {}

        def _spy(query, **kw):
            seen.update(kw)
            return "{}"

        monkeypatch.setattr(recall_mod, "_recall_impl", _spy)
        public_mod.recall(query="q", backend="bm25", explain=True)
        assert seen.get("explain") is True, f"backend alias dropped explain: {seen}"
        assert seen.get("backend") == "bm25", f"backend alias did not resolve to the mode: {seen}"

    def test_explain_defaults_off_and_is_still_forwarded(self, monkeypatch):
        seen = {}

        def _spy(query, **kw):
            seen.update(kw)
            return "{}"

        monkeypatch.setattr(recall_mod, "_recall_impl", _spy)
        public_mod.recall(query="q", mode="auto")
        assert seen.get("explain") is False

    @pytest.mark.parametrize("mode", ["similar", "axis", "pack", "prefetch", "classify", "diagnostics"])
    def test_modes_that_cannot_explain_return_an_error_ENVELOPE(self, mode):
        """Parsed, not grepped.

        Asserting a substring appears somewhere in the output would also pass on
        a successful envelope that merely happened to contain the word, so the
        payload is parsed and the error field is asserted.
        """
        raw = public_mod.recall(query="q", mode=mode, explain=True, block_id="b1")
        payload = json.loads(raw)
        assert isinstance(payload, dict), f"mode={mode}: envelope is not an object"
        assert "error" in payload, f"mode={mode}: no error field; got keys {sorted(payload)}"
        assert "explain" in payload["error"], f"mode={mode}: error is not about explain: {payload['error']}"
        for supported in ("auto", "bm25", "hybrid"):
            assert supported in payload["error"], f"mode={mode}: the refusal does not name supported mode {supported!r}"

    def test_an_unsupported_mode_without_explain_is_not_refused(self, monkeypatch):
        """Positive control: the refusal is about the flag, not the mode."""
        monkeypatch.setattr(recall_mod.intent_classify, "__wrapped__", lambda query: '{"intent": "test-classify"}')
        payload = json.loads(public_mod.recall(query="q", mode="classify"))
        assert payload == {"intent": "test-classify"}

    def test_the_wire_schema_gains_explain_and_loses_nothing(self):
        comp = _components(_build(("legacy", "public")))[WIRE_KEY]
        props = set((getattr(comp, "parameters", None) or {}).get("properties", {}))
        for keep in (
            "query",
            "mode",
            "limit",
            "active_only",
            "backend",
            "block_id",
            "axes",
            "weights",
            "max_tokens",
            "signals",
            "scoring_instant",
        ):
            assert keep in props, f"{keep} disappeared from the recall wire schema"
        assert "explain" in props, "explain is still missing from the wire schema"


class TestTheExplainEnvelopeIsRealNotJustAccepted:
    """Signature acceptance proves nothing about the payload.

    An earlier version compared ``"_explain" in (hits[0] if hits else {})`` on
    both sides. With both sides empty that is ``False == False`` -- the test
    passed while ranking nothing and comparing nothing. Every assertion below
    now requires NONEMPTY ranked results first, and compares stable block ids
    and actual ``_explain`` content at a fixed ``scoring_instant`` on the same
    backend, so neither side can agree by being empty.
    """

    @pytest.fixture
    def workspace(self, tmp_path, monkeypatch):
        """Minimal recallable workspace, in the shape the repo's own tests use.

        A hand-rolled memory/*.md tree is NOT enough -- recall refuses a
        workspace with no ``decisions/`` directory, so an earlier version of
        this fixture produced an error envelope and both envelope tests SKIPPED.
        A skip is not coverage, which is the whole reason this is built the
        canonical way.
        """
        import os

        root = str(tmp_path)
        for sub in ("decisions", "tasks", "entities", "intelligence"):
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        with open(os.path.join(root, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as fh:
            for bid, statement, when in (
                ("D-20260827-001", "retrieval rollout notes shipped", "2026-08-27"),
                ("D-20260823-002", "retrieval rollout notes reviewed", "2026-08-23"),
                ("D-20260819-003", "compiler determinism and hash chains", "2026-08-19"),
            ):
                fh.write(f"[{bid}]\nStatement: {statement}\nStatus: active\nDate: {when}\n\n")
        for rel in ("entities/projects.md", "intelligence/SIGNALS.md"):
            with open(os.path.join(root, rel), "w", encoding="utf-8") as fh:
                fh.write(f"# {os.path.basename(rel)}\n")
        monkeypatch.setenv("MIND_MEM_WORKSPACE", root)
        monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
        return tmp_path

    #: Fixed so ranking cannot drift on a clock between the two calls.
    INSTANT = "2026-09-01"
    BACKEND = "bm25"
    QUERY = "retrieval rollout notes"

    def _results(self, raw):
        payload = json.loads(raw)
        assert isinstance(payload, dict), f"envelope is not an object: {raw[:160]}"
        assert "error" not in payload, f"envelope carried an error: {payload.get('error')}"
        results = payload.get("results")
        assert isinstance(results, list), f"no results list; keys={sorted(payload)}"
        assert results, "the fixture ranked nothing, so this comparison would prove nothing"
        return results

    def test_explain_true_adds_the_decomposition_and_false_does_not(self, workspace):
        on = self._results(public_mod.recall(query=self.QUERY, mode=self.BACKEND, limit=5, explain=True, scoring_instant=self.INSTANT))
        off = self._results(public_mod.recall(query=self.QUERY, mode=self.BACKEND, limit=5, scoring_instant=self.INSTANT))

        assert all("_explain" in h for h in on), "explain=True left hits without a decomposition"
        assert not any("_explain" in h for h in off), "explain=False leaked a decomposition"

        decomposition = on[0]["_explain"]
        assert isinstance(decomposition, dict) and decomposition, "_explain is empty"
        for field in ("bm25", "final", "rrf_rank"):
            assert field in decomposition, f"_explain lacks {field}: {sorted(decomposition)}"

    def test_the_dispatcher_and_the_legacy_function_produce_the_same_envelope(self, workspace):
        """Identical envelopes was the compatibility promise. Held to it exactly."""
        via_public = self._results(
            public_mod.recall(query=self.QUERY, mode=self.BACKEND, limit=5, explain=True, scoring_instant=self.INSTANT)
        )
        via_legacy = self._results(
            recall_mod.recall.__wrapped__(  # type: ignore[attr-defined]
                self.QUERY, limit=5, backend=self.BACKEND, explain=True, scoring_instant=self.INSTANT
            )
        )

        pub_ids = [h.get("_id") for h in via_public]
        leg_ids = [h.get("_id") for h in via_legacy]
        assert all(pub_ids), f"hits carry no stable id: {via_public[0].keys()}"
        assert pub_ids == leg_ids, f"ranking differs: public={pub_ids} legacy={leg_ids}"

        pub_exp = via_public[0]["_explain"]
        leg_exp = via_legacy[0]["_explain"]
        assert sorted(pub_exp) == sorted(leg_exp), f"_explain fields differ: public={sorted(pub_exp)} legacy={sorted(leg_exp)}"
        assert pub_exp == leg_exp, f"_explain values differ for {pub_ids[0]}: public={pub_exp} legacy={leg_exp}"
