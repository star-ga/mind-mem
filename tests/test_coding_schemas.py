"""Tests for mind-mem coding-native memory schemas."""

import pytest

from mind_mem.coding_schemas import (
    CODING_BLOCK_TYPES,
    TYPE_ADR,
    TYPE_ALGO,
    TYPE_BUG,
    TYPE_CODE,
    TYPE_PERF,
    classify_coding_block,
    extract_code_metadata,
    format_adr_block,
    get_template,
)


class TestClassify:
    def test_adr_text(self):
        text = (
            "Architecture decision: we decided to adopt PostgreSQL over MongoDB."
            " Trade-off between consistency and flexibility."
        )
        assert classify_coding_block(text) == TYPE_ADR

    def test_code_text(self):
        text = "The function process_batch takes a list of items and returns the API endpoint response."
        assert classify_coding_block(text) == TYPE_CODE

    def test_perf_text(self):
        text = "Benchmark results: p99 latency is 45ms, throughput is 1200 rps."
        assert classify_coding_block(text) == TYPE_PERF

    def test_algo_text(self):
        text = "This sorting algorithm has O(n log n) time complexity and O(n) space complexity."
        assert classify_coding_block(text) == TYPE_ALGO

    def test_bug_text(self):
        text = "Bug: crash on startup. Reproduce by clicking login. Root cause: null pointer in auth module."
        assert classify_coding_block(text) == TYPE_BUG

    def test_no_match(self):
        text = "The weather today is sunny and warm."
        assert classify_coding_block(text) is None

    def test_ambiguous_defaults_to_highest(self):
        # Text with patterns from multiple categories
        text = "The benchmark showed a regression in the sorting algorithm."
        result = classify_coding_block(text)
        assert result in CODING_BLOCK_TYPES


class TestTemplates:
    def test_all_types_have_templates(self):
        for t in CODING_BLOCK_TYPES:
            template = get_template(t)
            assert isinstance(template, dict)
            assert "Type" in template

    def test_adr_template_fields(self):
        t = get_template(TYPE_ADR)
        assert t["Type"] == "ADR"
        assert "Context" in t
        assert "Decision" in t
        assert "Consequences" in t

    def test_code_template_fields(self):
        t = get_template(TYPE_CODE)
        assert "Name" in t
        assert "Signature" in t
        assert "Dependencies" in t

    def test_perf_template_fields(self):
        t = get_template(TYPE_PERF)
        assert "Baseline" in t
        assert "Current" in t
        assert "Unit" in t

    def test_algo_template_fields(self):
        t = get_template(TYPE_ALGO)
        assert "TimeComplexity" in t
        assert "SpaceComplexity" in t

    def test_bug_template_fields(self):
        t = get_template(TYPE_BUG)
        assert "ReproSteps" in t
        assert "RootCause" in t
        assert "Fix" in t

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            get_template("INVALID")


class TestExtractMetadata:
    def test_file_paths(self):
        text = "The bug is in src/auth/login.py and affects tests/test_auth.py"
        meta = extract_code_metadata(text)
        assert "files" in meta
        assert any("login.py" in f for f in meta["files"])

    def test_function_names(self):
        text = "def process_batch(items): pass\ndef validate_input(data): pass"
        meta = extract_code_metadata(text)
        assert "functions" in meta
        assert "process_batch" in meta["functions"]

    def test_class_names(self):
        text = "class UserService: ...\nstruct Config: ..."
        meta = extract_code_metadata(text)
        assert "classes" in meta
        assert "UserService" in meta["classes"]

    def test_complexity(self):
        text = "Time is O(n log n), space is O(1)"
        meta = extract_code_metadata(text)
        assert "complexity" in meta
        assert "O(n log n)" in meta["complexity"]

    def test_performance_numbers(self):
        text = "Latency dropped from 120ms to 45ms after optimization"
        meta = extract_code_metadata(text)
        assert "performance" in meta

    def test_empty_text(self):
        meta = extract_code_metadata("")
        assert meta == {}


class TestFormatADR:
    def test_basic_adr(self):
        result = format_adr_block(
            "Use PostgreSQL",
            "Need a relational database",
            "PostgreSQL for consistency guarantees",
            "Must maintain and operate PostgreSQL cluster",
        )
        assert "[ADR-" in result
        assert "Type: ADR" in result
        assert "Use PostgreSQL" in result

    def test_with_alternatives(self):
        result = format_adr_block(
            "Cache Strategy",
            "Need caching layer",
            "Use Redis",
            "Additional infrastructure",
            alternatives=["Memcached", "In-memory"],
        )
        assert "Memcached" in result
        assert "In-memory" in result

    def test_custom_block_id(self):
        result = format_adr_block(
            "Test",
            "context",
            "decision",
            "consequences",
            block_id="ADR-20260304-042",
        )
        assert "[ADR-20260304-042]" in result
