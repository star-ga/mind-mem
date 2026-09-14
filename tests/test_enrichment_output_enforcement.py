# Copyright 2026 STARGA, Inc.
"""Model text must cross a real output gate before optional recall annotation."""

import copy
import hashlib
import json

import pytest

from mind_mem import llm_extractor as llm
from mind_mem.compliance.redaction import RedactionRefused


def fact(claim="Alice works on MIND.", **extra):
    return {"claim": claim, "confidence": 0.8, "category": "state", **extra}


@pytest.mark.parametrize(
    "row",
    [
        fact(category="not-declared"),
        fact(category=[]),
        fact(claim={}),
        fact(claim="x" * 2001),
        fact(claim="Alice\u202eworks"),
        fact(claim=""),
        fact(confidence=float("nan")),
        fact(confidence=float("inf")),
        fact(confidence=True),
        fact(confidence=99),
        fact(confidence=10**500),
        fact(confidence="0.8"),
        fact(confidence=-0.1),
    ],
)
def test_invalid_model_proposal_is_refused(row):
    assert llm.validate_extraction_output([row], "facts") == []


def test_cap_duplicates_and_literal_binding():
    rows = [fact(f"Claim {n}.") for n in range(40)]
    assert len(llm.validate_extraction_output(rows, "facts")) == 32
    source = "Alice works on MIND."
    result = llm.validate_extraction_output([fact(), fact(), fact("Alice owns MIND.")], "facts", source=source)
    assert len(result) == 1
    span = result[0]["source_span"]
    assert source[span["start"] : span["end"]] == result[0]["claim"]


def test_entities_do_not_smuggle_invented_context():
    rows = [
        {"name": "Alice", "type": "person", "context": "owns everything"},
        {"name": "Bob", "type": "person"},
        {"name": "Alice", "type": "god"},
    ]
    result = llm.validate_extraction_output(rows, "entities", source="Alice works on MIND.")
    assert len(result) == 1
    assert result[0]["context"] == ""


def model(monkeypatch, facts, entities=None):
    calls = []
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "_record_extraction_feedback", lambda *a, **k: None)

    def query(prompt, *args, **kwargs):
        calls.append(prompt)
        return json.dumps(facts if prompt.startswith("Extract factual") else entities or [])

    monkeypatch.setattr(llm, "_query_llm", query)
    return calls


def test_real_extractor_and_enrichment_filter_model_canary(tmp_path, monkeypatch):
    model(monkeypatch, [fact(), fact("MODEL-CANARY-UNBOUND"), fact(category="undeclared")])
    block = {"_id": "D-20260914-001", "file": "decisions/DECISIONS.md", "excerpt": "Alice works on MIND.", "score": 1.2}
    original = copy.deepcopy(block)
    result = llm.enrich_block(block, enabled=True, workspace=str(tmp_path))
    assert result is block
    assert [row["claim"] for row in result["llm_facts"]] == ["Alice works on MIND."]
    assert all(result[key] == value for key, value in original.items())
    info = result["llm_enrichment"]
    assert info["source_block_id"] == original["_id"]
    assert info["input_sha256"] == hashlib.sha256(original["excerpt"].encode()).hexdigest()
    assert info["semantic_verification"] == "not_established"
    assert info["evidence_status"] == "unproven_supplement"


def test_configured_batch_path_drops_unbound_claim(tmp_path, monkeypatch):
    (tmp_path / "mind-mem.json").write_text(json.dumps({"extraction": {"enabled": True, "enrich_on_recall": True}}), encoding="utf-8")
    model(monkeypatch, [fact("MODEL-CANARY-UNBOUND")])
    results = [{"_id": "D-20260914-001", "excerpt": "Alice works on MIND."}]
    assert llm.enrich_results(results, str(tmp_path)) is results
    assert "llm_facts" not in results[0]
    assert results[0]["llm_enrichment"]["status"] == "no_admissible_annotations"


def test_default_off_keeps_all_bytes_and_makes_no_model_call(tmp_path, monkeypatch):
    calls = model(monkeypatch, [fact()])
    results = [{"_id": "D-1", "excerpt": "Alice works on MIND.", "score": 1.0}]
    before = json.dumps(results)
    assert llm.enrich_results(results, str(tmp_path)) is results
    assert json.dumps(results) == before
    assert calls == []


@pytest.mark.parametrize("mode", ["reject", "redact"])
def test_configured_redaction_before_model_and_no_stale_annotations(tmp_path, monkeypatch, mode):
    (tmp_path / "mind-mem.json").write_text(
        json.dumps({"v4": {"redaction": {"enabled": True, "mode": mode, "detectors": ["email"]}}}), encoding="utf-8"
    )
    calls = model(monkeypatch, [fact("alice@example.com")])
    block = {"_id": "D-1", "excerpt": "Contact alice@example.com", "llm_facts": [fact("OLD-UNVALIDATED")]}
    if mode == "reject":
        with pytest.raises(RedactionRefused):
            llm.enrich_block(block, enabled=True, workspace=str(tmp_path))
        assert calls == []
    else:
        llm.enrich_block(block, enabled=True, workspace=str(tmp_path))
        assert calls and all("alice@example.com" not in prompt for prompt in calls)
    assert "llm_facts" not in block


def test_source_binding_uses_only_text_seen_by_model(tmp_path, monkeypatch):
    model(monkeypatch, [fact("HIDDEN-AFTER-WINDOW")])
    block = {"_id": "D-1", "excerpt": "x" * 2000 + "HIDDEN-AFTER-WINDOW"}
    llm.enrich_block(block, enabled=True, workspace=str(tmp_path))
    assert "llm_facts" not in block


def test_extract_facts_itself_rejects_undeclared_category(tmp_path, monkeypatch):
    model(monkeypatch, [fact("MODEL-CANARY-UNBOUND", category="not-a-declared-category", confidence=99)])
    assert llm.extract_facts("Alice works on MIND.", workspace=str(tmp_path)) == []
