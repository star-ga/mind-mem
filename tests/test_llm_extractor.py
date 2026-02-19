#!/usr/bin/env python3
"""Tests for the optional LLM entity/fact extractor module."""

import json
import os
import sys
import tempfile

# Add scripts/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from llm_extractor import (
    _parse_json_from_response,
    enrich_block,
    enrich_results,
    extract_entities,
    extract_facts,
    is_available,
    load_config,
)


class TestIsAvailable:
    """Test LLM backend availability detection."""

    def test_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)

    def test_returns_bool_for_ollama(self):
        result = is_available(backend="ollama")
        assert isinstance(result, bool)

    def test_returns_bool_for_llama_cpp(self):
        result = is_available(backend="llama-cpp")
        assert isinstance(result, bool)

    def test_unknown_backend_returns_false(self):
        result = is_available(backend="nonexistent-backend")
        assert result is False


class TestExtractEntities:
    """Test entity extraction (graceful fallback when LLM unavailable)."""

    def test_returns_list_when_unavailable(self):
        result = extract_entities("John met Mary in Paris on Monday.")
        assert isinstance(result, list)

    def test_empty_text_returns_empty(self):
        result = extract_entities("")
        assert result == []

    def test_whitespace_only_returns_empty(self):
        result = extract_entities("   ")
        assert result == []

    def test_custom_model_returns_list(self):
        result = extract_entities("Some text here.", model="llama3:8b")
        assert isinstance(result, list)


class TestExtractFacts:
    """Test fact extraction (graceful fallback when LLM unavailable)."""

    def test_returns_list_when_unavailable(self):
        result = extract_facts("Caroline is a counselor who loves painting.")
        assert isinstance(result, list)

    def test_empty_text_returns_empty(self):
        result = extract_facts("")
        assert result == []

    def test_whitespace_only_returns_empty(self):
        result = extract_facts("   \n  ")
        assert result == []


class TestEnrichBlock:
    """Test block enrichment with LLM metadata."""

    def test_returns_block_unchanged_when_disabled(self):
        block = {"_id": "TEST-001", "excerpt": "Some text", "score": 1.5}
        result = enrich_block(block, enabled=False)
        assert result is block
        assert "llm_entities" not in result
        assert "llm_facts" not in result

    def test_returns_block_unchanged_when_no_excerpt(self):
        block = {"_id": "TEST-002", "score": 1.0}
        result = enrich_block(block, enabled=True)
        assert result is block

    def test_returns_block_unchanged_when_llm_unavailable(self):
        block = {"_id": "TEST-003", "excerpt": "Test content here.", "score": 0.9}
        result = enrich_block(block, enabled=True)
        # Even when enabled, if no LLM backend is available, block is unchanged
        assert result is block


class TestEnrichResults:
    """Test batch enrichment of recall results."""

    def test_returns_results_when_disabled(self):
        results = [
            {"_id": "A", "excerpt": "foo", "score": 1.0},
            {"_id": "B", "excerpt": "bar", "score": 0.5},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            # No config file -> disabled by default
            enriched = enrich_results(results, workspace=tmpdir)
            assert enriched is results
            assert len(enriched) == 2

    def test_returns_results_when_config_disabled(self):
        results = [{"_id": "C", "excerpt": "baz", "score": 0.8}]
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"extraction": {"enabled": False, "model": "phi3:mini"}}
            with open(os.path.join(tmpdir, "mind-mem.json"), "w") as f:
                json.dump(config, f)
            enriched = enrich_results(results, workspace=tmpdir)
            assert enriched is results


class TestLoadConfig:
    """Test configuration loading from mind-mem.json."""

    def test_defaults_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(tmpdir)
            assert config["enabled"] is False
            assert config["model"] == "phi3:mini"
            assert config["backend"] == "auto"

    def test_reads_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = {
                "extraction": {
                    "enabled": True,
                    "model": "llama3:8b",
                    "backend": "ollama",
                }
            }
            with open(os.path.join(tmpdir, "mind-mem.json"), "w") as f:
                json.dump(cfg, f)
            config = load_config(tmpdir)
            assert config["enabled"] is True
            assert config["model"] == "llama3:8b"
            assert config["backend"] == "ollama"

    def test_partial_config_merges_with_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = {"extraction": {"enabled": True}}
            with open(os.path.join(tmpdir, "mind-mem.json"), "w") as f:
                json.dump(cfg, f)
            config = load_config(tmpdir)
            assert config["enabled"] is True
            assert config["model"] == "phi3:mini"  # default
            assert config["backend"] == "auto"  # default

    def test_invalid_json_returns_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "mind-mem.json"), "w") as f:
                f.write("{invalid json!!!")
            config = load_config(tmpdir)
            assert config["enabled"] is False
            assert config["model"] == "phi3:mini"

    def test_missing_extraction_section_returns_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = {"recall": {"backend": "scan"}}
            with open(os.path.join(tmpdir, "mind-mem.json"), "w") as f:
                json.dump(cfg, f)
            config = load_config(tmpdir)
            assert config["enabled"] is False


class TestParseJsonFromResponse:
    """Test JSON extraction from LLM output."""

    def test_parses_clean_json(self):
        text = '[{"name": "John", "type": "person"}]'
        result = _parse_json_from_response(text)
        assert len(result) == 1
        assert result[0]["name"] == "John"

    def test_parses_markdown_fenced_json(self):
        text = '```json\n[{"name": "Paris", "type": "place"}]\n```'
        result = _parse_json_from_response(text)
        assert len(result) == 1
        assert result[0]["name"] == "Paris"

    def test_returns_empty_on_invalid_json(self):
        text = "This is not JSON at all."
        result = _parse_json_from_response(text)
        assert result == []

    def test_filters_non_dict_items(self):
        text = '[{"name": "OK"}, "bad_string", 42, {"type": "place"}]'
        result = _parse_json_from_response(text)
        assert len(result) == 2
        assert result[0]["name"] == "OK"
        assert result[1]["type"] == "place"
