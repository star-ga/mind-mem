"""Regression gate for the INDEPENDENCE of the validity gate's two opt-in
extensions — provenance class (a fifth criterion) and outcome attribution
(a factor on the mean).

Each shipped separately, each behind its own default-``False`` sub-flag, and
each is unaware of the other. This file locks in the properties that only
exist once both are present in the same function:

  * both off — the returned key set, key ORDER and every float are
    byte-identical to the original four-criteria gate (raw IEEE-754 bits
    compared, not ``==`` on rounded floats);
  * either one alone — enabling one leaves the other's keys absent and its
    contribution neutral;
  * both on — the composite is the five-way mean scaled by the outcome
    factor; neither extension perturbs the other's own outputs;
  * determinism — repeated evaluation is byte-identical in every
    combination. No clock, no randomness, no learned scoring.
"""

from __future__ import annotations

import itertools
import struct
from typing import Any

from mind_mem.outcome_attribution import OutcomeSignal
from mind_mem.validity_gate import validity_components

_ID = "MEM-20260401-001"

# Four base criteria, all neutral except status, which debits to 0.5. That
# leaves the composite sensitive to both extensions at once.
_BASE_KEYS = ("corroboration", "status", "contradiction", "staleness", "score")


def _hit(**extra: Any) -> dict[str, Any]:
    return {"_id": _ID, "status": "wip", "score": 0.9, **extra}


def _fingerprint(components: dict[str, Any]) -> bytes:
    """Byte-level fingerprint: key order, value types, raw float bits."""
    parts: list[bytes] = []
    for key, value in components.items():
        parts.append(key.encode())
        parts.append(type(value).__name__.encode())
        parts.append(struct.pack("<d", value) if isinstance(value, float) else repr(value).encode())
    return b"|".join(parts)


def _four_criteria_mean(components: dict[str, Any]) -> float:
    return 0.25 * (components["corroboration"] + components["status"] + components["contradiction"] + components["staleness"])


class TestBothOffIsTheOriginalGate:
    """The load-bearing property: two extensions off == zero extensions."""

    def test_key_set_and_order_are_the_original_four_criteria(self) -> None:
        components = validity_components(_hit(), set(), {})
        assert tuple(components) == _BASE_KEYS

    def test_composite_is_the_bit_exact_four_criteria_mean(self) -> None:
        components = validity_components(_hit(), set(), {})
        expected = round(_four_criteria_mean(components), 4)
        assert struct.pack("<d", components["score"]) == struct.pack("<d", expected)

    def test_explicit_off_matches_implicit_off_byte_for_byte(self) -> None:
        """Passing the defaults explicitly must not perturb a single bit."""
        implicit = validity_components(_hit(), set(), {})
        explicit = validity_components(_hit(), set(), {}, None, provenance_enabled=False, confirmed_ids=frozenset())
        assert _fingerprint(implicit) == _fingerprint(explicit)

    def test_no_extension_keys_leak_in(self) -> None:
        components = validity_components(_hit(), set(), {})
        assert "outcome" not in components
        assert "provenance" not in components
        assert "provenance_class" not in components


class TestEitherOneAlone:
    """Enabling one extension must not conjure the other's surface."""

    def test_provenance_only_adds_provenance_keys(self) -> None:
        components = validity_components(_hit(), set(), {}, provenance_enabled=True)
        assert "provenance" in components and "provenance_class" in components
        assert "outcome" not in components

    def test_outcome_only_adds_outcome_key(self) -> None:
        components = validity_components(_hit(), set(), {}, {})
        assert "outcome" in components
        assert "provenance" not in components and "provenance_class" not in components

    def test_outcome_with_no_evidence_for_this_block_is_neutral(self) -> None:
        """An empty/irrelevant signal map leaves the mean untouched."""
        base = validity_components(_hit(), set(), {})
        neutral = validity_components(_hit(), set(), {}, {"MEM-20260401-999": OutcomeSignal("x")})
        assert neutral["outcome"] == 1.0
        assert struct.pack("<d", neutral["score"]) == struct.pack("<d", base["score"])


class TestBothOnCompose:
    """Both on: five-way mean, scaled by the factor. Neither perturbs the other."""

    def test_composite_is_five_way_mean_times_outcome_factor(self) -> None:
        signals = {_ID: OutcomeSignal(_ID, failure=3)}
        both = validity_components(_hit(), set(), {}, signals, provenance_enabled=True, confirmed_ids=frozenset())
        five_way = 0.2 * (both["corroboration"] + both["status"] + both["contradiction"] + both["staleness"] + both["provenance"])
        assert both["score"] == round(five_way * both["outcome"], 4)

    def test_provenance_outputs_are_unchanged_by_the_outcome_extension(self) -> None:
        signals = {_ID: OutcomeSignal(_ID, failure=3)}
        prov_only = validity_components(_hit(), set(), {}, provenance_enabled=True)
        both = validity_components(_hit(), set(), {}, signals, provenance_enabled=True)
        assert both["provenance"] == prov_only["provenance"]
        assert both["provenance_class"] == prov_only["provenance_class"]

    def test_outcome_factor_is_unchanged_by_the_provenance_extension(self) -> None:
        signals = {_ID: OutcomeSignal(_ID, failure=3)}
        outcome_only = validity_components(_hit(), set(), {}, signals)
        both = validity_components(_hit(), set(), {}, signals, provenance_enabled=True)
        assert both["outcome"] == outcome_only["outcome"]

    def test_success_corroboration_still_lifts_c1_with_provenance_on(self) -> None:
        """The outcome extension's c1 lift survives the widened mean."""
        hit = _hit(fusion_sources=["bm25"])
        assert validity_components(hit, set(), {}, {}, provenance_enabled=True)["corroboration"] == 0.5
        lifted = validity_components(hit, set(), {}, {_ID: OutcomeSignal(_ID, success=5)}, provenance_enabled=True)
        assert lifted["corroboration"] == 1.0

    def test_key_order_is_stable_with_both_on(self) -> None:
        both = validity_components(_hit(), set(), {}, {_ID: OutcomeSignal(_ID, failure=3)}, provenance_enabled=True)
        assert tuple(both) == (
            "corroboration",
            "status",
            "contradiction",
            "staleness",
            "provenance",
            "provenance_class",
            "outcome",
            "score",
        )


class TestDeterminismAcrossEveryCombination:
    def test_every_flag_combination_is_byte_identical_across_runs(self) -> None:
        signal_maps: list[dict[str, OutcomeSignal] | None] = [
            None,
            {},
            {_ID: OutcomeSignal(_ID, failure=3)},
            {_ID: OutcomeSignal(_ID, success=5)},
            {_ID: OutcomeSignal(_ID, success=2, failure=4, neutral=1)},
        ]
        for signals, provenance in itertools.product(signal_maps, [False, True]):
            fingerprints = {
                _fingerprint(validity_components(_hit(ActorRole="importer"), set(), {}, signals, provenance_enabled=provenance))
                for _ in range(50)
            }
            assert len(fingerprints) == 1, f"non-deterministic for {signals=} {provenance=}"
