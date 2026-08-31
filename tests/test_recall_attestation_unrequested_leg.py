"""An unrequested vector leg must not be attested as degraded.

``derive_legs`` documents the rule in a comment: "A vector leg that was never
requested must not appear as degraded even if a stray marker leaked one in —
degradation is only meaningful relative to what the run asked for."

The code did not implement it. The discard was guarded by
``not (marker_names_vector or pg_fallback)``, and ``marker_names_vector`` is by
construction ``LEG_VECTOR in degraded`` at that point — nothing outside the
``vector_requested`` branch adds to ``degraded`` — so the guard was always
False and the discard was unreachable. A BM25-only run whose recorded marker
happened to name ``vector`` attested ``legs_degraded == ("vector",)``, exactly
what the comment said must not happen.

The one signal that legitimately survives an unrequested leg is the pgvector
``bm25_fallback`` provenance: that is per-hit evidence that a dense leg really
was attempted server-side, so it still degrades.
"""

from __future__ import annotations

from mind_mem.hybrid_recall import _as_results
from mind_mem.recall_attestation import derive_legs


class TestUnrequestedVectorLeg:
    def test_stray_marker_does_not_degrade_a_leg_the_run_never_asked_for(self) -> None:
        """Before the fix this returned ``legs_degraded == ("vector",)``."""
        results = _as_results([{"_id": "a"}], {"leg": "vector", "reason": "unavailable"})
        ran, degraded = derive_legs(results, vector_requested=False, vector_available=False)
        assert ran == ("bm25",)
        assert degraded == ()

    def test_a_stray_marker_naming_another_leg_is_still_kept(self) -> None:
        """Only the vector leg has a "was it requested" question to answer."""
        results = _as_results([{"_id": "a"}], {"leg": "graph", "reason": "expander_error"})
        _, degraded = derive_legs(results, vector_requested=False, vector_available=False)
        assert degraded == ("graph",)

    def test_pg_fallback_provenance_still_degrades_an_unrequested_leg(self) -> None:
        """Recorded per-hit evidence that a dense leg ran and fell back stands."""
        results = _as_results(
            [{"_id": "a", "_retrieval_source": "bm25_fallback"}],
            {"leg": "vector", "reason": "server_side_fallback"},
        )
        ran, degraded = derive_legs(results, vector_requested=False, vector_available=True)
        assert ran == ("bm25",)
        assert degraded == ("vector",)

    def test_requested_but_unserved_vector_is_still_degraded(self) -> None:
        """The fix must not silence the case the marker exists for."""
        results = _as_results([{"_id": "a"}], {"leg": "vector", "reason": "unavailable"})
        ran, degraded = derive_legs(results, vector_requested=True, vector_available=False)
        assert ran == ("bm25",)
        assert degraded == ("vector",)

    def test_requested_and_served_vector_reports_a_hybrid_fusion(self) -> None:
        results = _as_results([{"_id": "a"}], None)
        ran, degraded = derive_legs(results, vector_requested=True, vector_available=True)
        assert ran == ("bm25", "hybrid", "vector")
        assert degraded == ()
