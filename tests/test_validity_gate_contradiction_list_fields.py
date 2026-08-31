"""Regression gate: c3 must debit blocks named in a *real* CONTRADICTIONS.md.

``intel_scan.write_contradictions`` writes the parties of a contradiction into
the ``Objects:`` and ``Evidence:`` **list** fields, and
``block_parser.parse_file`` renders a ``Key:``-then-``- item`` field as a Python
list.  ``validity_gate._load_contradicted_party_ids`` used to walk only ``str``
values, so it returned an empty set for every entry the scanner itself emits —
``validity["contradiction"]`` was a permanent 1.0 and the gate's contradiction
criterion could never fire in production.

The existing ``tests/test_validity_gate.py`` fixture hides this because it
embeds the party id in the scalar ``Statement:`` field, which the str-only walk
did read.  Every fixture here is the byte shape the scanner actually emits, and
``test_parties_live_only_in_list_fields`` pins that property so these tests
cannot start passing through a scalar field again.
"""

from __future__ import annotations

import os
from typing import Any

from mind_mem.block_parser import parse_file
from mind_mem.validity_gate import _load_contradicted_party_ids, apply_validity_gate, validity_components

# Byte-for-byte the block emitted by intel_scan.write_contradictions
# (intel_scan.py: "Objects:" / "Evidence:" list fields, Statement carrying only
# the conflict reason).
_EMITTED_BODY = """
[C-20260301-001]
Date: 2026-03-01
Severity: high
Type: decision_vs_decision
Statement: modality conflict: must vs must_not on axis=deploy
Objects:
- D-20260301-002
- D-20260301-003
Evidence:
- SIG-20260301-011: deploy/must
- SIG-20260301-012: deploy/must_not
ProposedFix: manual_review
Status: open
Resolution: none
Sources:
- decisions/DECISIONS.md
"""

# A block carrying its parties inside a nested ConstraintSignatures section:
# parse_file renders that as a list of dicts (with further nested dicts, ints
# and Nones), so it exercises the recursive walk beyond flat string lists.
_NESTED_BODY = """
[C-20260301-002]
Date: 2026-03-01
Severity: high
Type: decision_vs_decision
Statement: modality conflict on axis=deploy
ConstraintSignatures:
- id: CS-20260301-001
  source: D-20260305-007
  modality: must
Status: open
Resolution: none
"""

_CONTRADICTION_ID = "C-20260301-001"
_PARTY_A = "D-20260301-002"
_PARTY_B = "D-20260301-003"


def _workspace(tmp_path: Any, body: str) -> str:
    """Write ``body`` as ``intelligence/CONTRADICTIONS.md`` under a fresh root."""
    ws = tmp_path / "ws"
    intel = ws / "intelligence"
    intel.mkdir(parents=True)
    (intel / "CONTRADICTIONS.md").write_text(body, encoding="utf-8")
    return str(ws)


def test_parties_live_only_in_list_fields(tmp_path: Any) -> None:
    """Mechanism guard for every other test in this file.

    If a future fixture edit leaked a party id into a scalar field, the
    str-only walk would find it and the regression tests below would pass
    against the broken code.  Pin that they cannot.
    """
    path = os.path.join(_workspace(tmp_path, _EMITTED_BODY), "intelligence", "CONTRADICTIONS.md")
    (block,) = parse_file(path)

    assert isinstance(block["Objects"], list)
    assert block["Objects"] == [_PARTY_A, _PARTY_B]
    assert isinstance(block["Evidence"], list)

    scalars = [v for k, v in block.items() if not k.startswith("_") and isinstance(v, str)]
    for value in scalars:
        assert _PARTY_A not in value, f"fixture leaked a party id into a scalar field: {value!r}"
        assert _PARTY_B not in value, f"fixture leaked a party id into a scalar field: {value!r}"


def test_party_ids_read_from_objects_and_evidence_lists(tmp_path: Any) -> None:
    """The defect: list-valued fields were skipped, so this returned set()."""
    party_ids = _load_contradicted_party_ids(_workspace(tmp_path, _EMITTED_BODY))

    assert _PARTY_A in party_ids
    assert _PARTY_B in party_ids
    # Evidence: rows name the conflicting signals — also parties to the entry.
    assert "SIG-20260301-011" in party_ids
    assert "SIG-20260301-012" in party_ids


def test_contradiction_entry_is_not_a_party_to_itself(tmp_path: Any) -> None:
    """The leading-underscore key filter still holds: ``_id``/``_entities``
    are parser metadata, and a ``C-...`` entry id must never demote a hit."""
    party_ids = _load_contradicted_party_ids(_workspace(tmp_path, _EMITTED_BODY))

    assert _CONTRADICTION_ID not in party_ids


def test_party_ids_read_from_nested_dict_sections(tmp_path: Any) -> None:
    """Nested list-of-dict sections (ConstraintSignatures/Ops) are walked too."""
    party_ids = _load_contradicted_party_ids(_workspace(tmp_path, _NESTED_BODY))

    assert "D-20260305-007" in party_ids
    assert "C-20260301-002" not in party_ids


def test_absent_log_still_yields_empty_set(tmp_path: Any) -> None:
    """Neutral-by-absence is unchanged: no file -> no party ids -> c3 == 1.0."""
    ws = tmp_path / "empty"
    ws.mkdir()

    assert _load_contradicted_party_ids(str(ws)) == set()


def test_gate_debits_and_demotes_a_listed_party(tmp_path: Any) -> None:
    """End-to-end through the real Stage 2.65 entry point.

    Before the fix the contradicted hit scored ``contradiction == 1.0`` and was
    never demoted, so it kept its full score alongside the clean peer.
    """
    ws = _workspace(tmp_path, _EMITTED_BODY)
    hits: list[dict[str, Any]] = [
        {"_id": _PARTY_A, "score": 1.0, "status": "active"},
        {"_id": "D-20260301-900", "score": 1.0, "status": "active"},
    ]

    apply_validity_gate(hits, ws, {"validity_gate": {"enabled": True}})

    contradicted, clean = hits
    assert contradicted["validity"]["contradiction"] == 0.0
    assert contradicted["_validity_demoted"] is True
    assert contradicted["score"] < clean["score"]

    assert clean["validity"]["contradiction"] == 1.0
    assert "_validity_demoted" not in clean


def test_components_helper_agrees_with_loaded_party_ids(tmp_path: Any) -> None:
    """The pure math consumes the loaded set unchanged — no second extraction."""
    party_ids = _load_contradicted_party_ids(_workspace(tmp_path, _EMITTED_BODY))
    components = validity_components({"_id": _PARTY_B, "status": "active"}, party_ids, {})

    assert components["contradiction"] == 0.0
