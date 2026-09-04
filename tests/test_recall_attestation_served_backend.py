# Copyright 2026 STARGA, Inc.
"""The receipt must name the leg the run executed — no other.

Two defects, one bug class: the attested backend was *asserted* somewhere
upstream instead of *derived* from what actually ran.

(b) The MCP handler resolved its vector flags from the **requested** backend
    string (``"auto"`` / ``"hybrid"``) while ``envelope["backend"]`` — sitting
    in the very JSON it parses — already held the backend the run *used*. When
    the hybrid arm raised and the run fell back to FTS5, the envelope published
    ``warnings: "falling back to BM25"`` beside ``legs_ran`` naming a hybrid
    fusion that never executed. Two opinions of one run.

(c) The library entry hard-coded ``backend="bm25"`` into
    :func:`~mind_mem.recall.attest_and_record`, justified by a docstring
    claiming the engine "runs no dense leg on any path". ``_load_backend``
    returns a ``VectorBackend`` for ``recall.backend: "vector"`` and a
    ``PostgresRecallBackend`` for a Postgres block store, so on those paths the
    receipt under-claimed — the mirror image of (b).

Every test here is written so it *can* fail: each one asserts the scenario was
actually reached (the fallback happened, hits exist, the probe ran) before
asserting the claim, so a silently-not-executed path shows up as a failure
rather than as a pass over an empty set.

KNOWN RESIDUAL, deliberately not asserted here: ``derive_legs`` puts ``bm25``
in ``legs_ran`` unconditionally, which is wrong for a pure dense
``VectorBackend`` where no lexical leg runs. Correcting it needs per-hit
``_retrieval_source`` provenance that ``VectorBackend`` does not stamp today.
Scheduled for 5.1; the stand-in backend used below is shaped like the Postgres
one, where the fused claim IS correct, so no test here bakes in the wrong shape.
"""

from __future__ import annotations

import json
import os

import pytest

from mind_mem._recall_core import RecallBackend
from mind_mem.init_workspace import init
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.recall import _served_backend
from mind_mem.recall import attest_and_record, default_backend_for


def _write_workspace(root: str, recall_config: dict | None = None) -> str:
    """A real workspace with enough blocks that a recall cannot return nothing."""
    os.makedirs(root, exist_ok=True)
    init(root)
    path = os.path.join(root, "decisions", "DECISIONS.md")
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(12):
            fh.write(f"[SRVB-{i:03d}]\n")
            fh.write("Type: Decision\n")
            fh.write(f"Statement: attestation backend fallback provenance block {i}\n")
            fh.write(f"Date: 2026-0{(i % 9) + 1}-1{i % 10}\n\n")
    if recall_config is not None:
        cfg_path = os.path.join(root, "mind-mem.json")
        try:
            with open(cfg_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (OSError, json.JSONDecodeError):
            cfg = {}
        cfg["recall"] = {**cfg.get("recall", {}), **recall_config}
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)
    return root


# ---------------------------------------------------------------------------
# (b) — the served surface attests the leg it served
# ---------------------------------------------------------------------------


class TestServedBackendMapping:
    """The unit under the end-to-end test: envelope backend -> flag-resolution name."""

    @pytest.mark.parametrize(
        ("used", "expected"),
        [
            ("sqlite", "bm25"),  # FTS5 is a lexical engine
            ("scan", "bm25"),  # so is the full scan
            ("hybrid", "hybrid"),  # a real fusion keeps the flags it depends on
        ],
    )
    def test_the_used_backend_decides(self, used: str, expected: str) -> None:
        # The requested string is deliberately the one that would be WRONG, so
        # a regression that reads the request cannot pass this.
        assert _served_backend({"backend": used}, "auto") == expected

    def test_a_missing_backend_field_falls_back_to_the_request(self) -> None:
        """Degrade, never raise: nothing produces such an envelope today."""
        assert _served_backend({}, "auto") == "auto"
        assert _served_backend({"backend": ""}, "hybrid") == "hybrid"
        assert _served_backend({"backend": 7}, "bm25") == "bm25"


class TestFallbackServeAttestsOnlyTheLegItRan:
    """The demonstrated failure: hybrid raised, FTS served, receipt said hybrid."""

    @pytest.fixture
    def ws(self, tmp_path):
        # ``vector_enabled`` is what makes this test able to fail: without it
        # ``resolve_vector_flags`` answers (False, False) for every backend
        # string and the assertion below would hold for the wrong reason.
        return _write_workspace(str(tmp_path / "ws"), {"vector_enabled": True})

    def test_vector_flags_are_resolved_from_the_backend_that_served(self, ws, monkeypatch) -> None:
        from mind_mem.hybrid_recall import HybridBackend

        def _raise(self, *a, **kw):
            raise ValueError("vector index unreadable")

        monkeypatch.setattr(HybridBackend, "search", _raise)

        # Positive control on the fixture itself: the flags this workspace
        # resolves for a hybrid request must really be (True, True), otherwise
        # the legs assertion below could not distinguish fixed from broken.
        from mind_mem.recall import resolve_vector_flags

        assert resolve_vector_flags(ws, "auto") == (True, True), "fixture cannot exercise the defect"

        import mind_mem.mcp.tools.recall as recall_tool

        with use_workspace(ws):
            envelope = json.loads(recall_tool.recall("attestation backend fallback", limit=5, backend="hybrid"))

        # The scenario was reached, not merely assumed.
        assert envelope["backend"] in ("sqlite", "scan"), envelope["backend"]
        assert envelope.get("results"), "no hits: the legs assertion would be thin"
        assert any("falling back to BM25" in w for w in envelope.get("warnings", []))

        # ...and the receipt agrees with that warning instead of contradicting it.
        legs = envelope["attestation"]["legs_ran"]
        assert legs == ["bm25"], f"receipt named a leg the run never executed: {legs}"

    def test_from_config_raising_still_resolves_the_lexical_name(self, ws, monkeypatch) -> None:
        """The prescribed forcing mechanism, made non-vacuous by a spy.

        Patching ``from_config`` to raise also breaks ``resolve_vector_flags``
        (it constructs a backend too), so it would answer (False, False) and
        ``legs_ran == ["bm25"]`` would hold even with the defect present. The
        spy records the backend string the handler asked about, which is the
        value the fix actually changes.
        """
        import mind_mem.recall as recall_mod
        from mind_mem.hybrid_recall import HybridBackend

        def _raise(cls, *a, **kw):
            raise ValueError("hybrid backend unavailable")

        monkeypatch.setattr(HybridBackend, "from_config", classmethod(_raise))

        seen: list[str] = []
        real = recall_mod.resolve_vector_flags

        def _spy(workspace, backend, config=None):
            seen.append(backend)
            return real(workspace, backend, config)

        monkeypatch.setattr(recall_mod, "resolve_vector_flags", _spy)

        import mind_mem.mcp.tools.recall as recall_tool

        with use_workspace(ws):
            envelope = json.loads(recall_tool.recall("provenance block fallback", limit=5, backend="hybrid"))

        assert envelope["backend"] in ("sqlite", "scan")
        assert seen == ["bm25"], f"flags resolved for the REQUESTED backend, not the served one: {seen}"
        assert envelope["attestation"]["legs_ran"] == ["bm25"]


# ---------------------------------------------------------------------------
# (c) — the library entry stops under-claiming on the dense paths
# ---------------------------------------------------------------------------


class _StandInBackend(RecallBackend):
    """Shaped like ``PostgresRecallBackend``: a configured backend object.

    Used instead of a real ``VectorBackend`` so this test does not depend on
    the optional vector extra being installed, and so it does not assert the
    known-wrong pure-dense ``legs_ran`` shape documented in the module header.
    """

    def search(self, workspace, query, limit=10, active_only=False):  # pragma: no cover — never called
        return []

    def index(self, workspace):  # pragma: no cover — never called
        return None


class TestDefaultBackendIsResolvedNotAssumed:
    def test_the_builtin_scan_is_bm25(self, tmp_path) -> None:
        ws = _write_workspace(str(tmp_path / "scan"))
        assert default_backend_for(ws) == "bm25"

    def test_the_fts_index_is_bm25(self, tmp_path) -> None:
        ws = _write_workspace(str(tmp_path / "fts"), {"backend": "sqlite"})
        assert default_backend_for(ws) == "bm25"

    def test_a_configured_backend_object_is_auto(self, tmp_path, monkeypatch) -> None:
        import mind_mem.recall as recall_mod

        ws = _write_workspace(str(tmp_path / "obj"))
        monkeypatch.setattr(recall_mod, "_load_backend", lambda w: _StandInBackend())
        assert default_backend_for(ws) == "auto"

    def test_a_real_vector_config_resolves_to_auto(self, tmp_path) -> None:
        """The live rule, through the real ``_load_backend``, no stand-in."""
        from mind_mem._recall_core import _load_backend

        ws = _write_workspace(str(tmp_path / "vec"), {"backend": "vector", "vector_enabled": True})
        if not isinstance(_load_backend(ws), RecallBackend):
            pytest.skip("vector extra not installed: _load_backend cannot return a backend object here")
        assert default_backend_for(ws) == "auto"

    def test_a_broken_probe_degrades_to_the_claim_that_says_least(self, tmp_path, monkeypatch) -> None:
        import mind_mem.recall as recall_mod

        ws = _write_workspace(str(tmp_path / "broken"))

        def _boom(_w):
            raise RuntimeError("config unreadable")

        monkeypatch.setattr(recall_mod, "_load_backend", _boom)
        assert default_backend_for(ws) == "bm25"


class TestLibraryEntryAttestsTheDenseLeg:
    def test_a_configured_backend_reports_the_leg_it_requested(self, tmp_path, monkeypatch) -> None:
        import mind_mem.recall as recall_mod

        ws = _write_workspace(str(tmp_path / "ws"), {"vector_enabled": True})
        monkeypatch.setattr(recall_mod, "_load_backend", lambda w: _StandInBackend())

        # Positive control: the flags this workspace resolves for "auto" must
        # differ from the "bm25" shape, or the assertion cannot fail.
        assert recall_mod.resolve_vector_flags(ws, "auto") == (True, True)
        assert recall_mod.resolve_vector_flags(ws, "bm25") == (False, False)

        record = attest_and_record(ws, "attestation backend provenance", [{"_id": "SRVB-001", "score": 1.0}])
        assert record is not None, "no record: every assertion below would be vacuous"
        assert "vector" in record["legs_ran"], f"dense leg under-claimed: {record['legs_ran']}"

    def test_a_lexical_workspace_still_claims_only_bm25(self, tmp_path) -> None:
        """The behaviour the old hard-coded default got right is preserved.

        A workspace with vector recall enabled for some other surface must not
        get a vector leg marked DEGRADED on every scan-backed call.
        """
        import mind_mem.recall as recall_mod

        ws = _write_workspace(str(tmp_path / "ws"), {"vector_enabled": True})
        assert recall_mod.resolve_vector_flags(ws, "auto") == (True, True), "fixture cannot exercise the defect"

        record = attest_and_record(ws, "attestation backend provenance", [{"_id": "SRVB-002", "score": 1.0}])
        assert record is not None
        assert record["legs_ran"] == ["bm25"]
        assert record["legs_degraded"] == []

    def test_an_explicit_backend_argument_still_wins(self, tmp_path, monkeypatch) -> None:
        import mind_mem.recall as recall_mod

        ws = _write_workspace(str(tmp_path / "ws"), {"vector_enabled": True})
        monkeypatch.setattr(recall_mod, "_load_backend", lambda w: _StandInBackend())

        record = attest_and_record(ws, "explicit leg", [{"_id": "SRVB-003", "score": 1.0}], backend="bm25")
        assert record is not None
        assert record["legs_ran"] == ["bm25"]


# ---------------------------------------------------------------------------
# The swallowed diagnostic must leave a trace the caller can read
# ---------------------------------------------------------------------------


class TestExplainFailureIsVisible:
    @pytest.fixture
    def ws(self, tmp_path):
        return _write_workspace(str(tmp_path / "ws"))

    def test_a_failed_explain_annotation_names_itself_in_warnings(self, ws, monkeypatch) -> None:
        import mind_mem._recall_explain as explain_mod

        def _boom(*a, **kw):
            raise RuntimeError("ordering invariant misfired")

        monkeypatch.setattr(explain_mod, "attach_explain", _boom)

        import mind_mem.mcp.tools.recall as recall_tool

        with use_workspace(ws):
            envelope = json.loads(recall_tool.recall("attestation backend provenance", limit=5, explain=True))

        hits = envelope.get("results") or []
        assert hits, "no hits: _apply_explain returns early and never reaches the swallow"
        assert all("_explain" not in h for h in hits), "explain did not actually fail: the trace would be untested"
        warnings = envelope.get("warnings", [])
        assert any("Explain annotation unavailable" in w for w in warnings), warnings

    def test_a_successful_explain_adds_no_warning(self, ws) -> None:
        """The trace must be the failure signal, not noise on every request."""
        import mind_mem.mcp.tools.recall as recall_tool

        with use_workspace(ws):
            envelope = json.loads(recall_tool.recall("provenance attestation backend", limit=5, explain=True))

        assert envelope.get("results"), "no hits: the assertion below would be vacuous"
        assert not any("Explain annotation unavailable" in w for w in envelope.get("warnings", []))
