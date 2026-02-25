"""Tests for observation_compress module.

Covers:
  - Basic compression via mock llm_fn
  - Category-specific prompt selection (adversarial, temporal, multi-hop)
  - Empty / whitespace-only context passthrough
  - Default and unknown query_type fallback to default system prompt
  - Message format correctness
  - max_tokens forwarding
  - Template rendering (question + context appear in user message)
  - Module-level constants integrity
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

# ── import the module under test ──────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from observation_compress import (
    _CATEGORY_PROMPTS,
    COMPRESS_SYSTEM_PROMPT,
    COMPRESS_USER_TEMPLATE,
    compress_context,
)

# ── helpers ───────────────────────────────────────────────────────────


def _make_llm_fn(return_value: str = "compressed") -> MagicMock:
    """Return a MagicMock mimicking llm_fn(messages, model=..., max_tokens=...)."""
    fn = MagicMock(return_value=return_value)
    return fn


# ══════════════════════════════════════════════════════════════════════
# 1.  Basic compression call
# ══════════════════════════════════════════════════════════════════════


class TestBasicCompression:
    def test_returns_llm_output(self):
        """compress_context returns whatever llm_fn returns."""
        fn = _make_llm_fn("1. Observation one")
        result = compress_context("some context", "What happened?", fn)
        assert result == "1. Observation one"

    def test_llm_fn_called_once(self):
        fn = _make_llm_fn()
        compress_context("ctx", "q?", fn)
        fn.assert_called_once()


# ══════════════════════════════════════════════════════════════════════
# 2.  Category-specific prompts
# ══════════════════════════════════════════════════════════════════════


class TestCategoryPrompts:
    @pytest.mark.parametrize("category", ["adversarial", "temporal", "multi-hop"])
    def test_known_categories_use_specific_prompt(self, category):
        fn = _make_llm_fn()
        compress_context("ctx", "q?", fn, query_type=category)
        messages = fn.call_args[0][0]
        assert messages[0]["content"] == _CATEGORY_PROMPTS[category]

    def test_adversarial_prompt_contains_evidence_found(self):
        assert "EVIDENCE_FOUND" in _CATEGORY_PROMPTS["adversarial"]

    def test_temporal_prompt_mentions_chronological(self):
        assert "CHRONOLOGICAL" in _CATEGORY_PROMPTS["temporal"]

    def test_multi_hop_prompt_mentions_connections(self):
        assert "connections" in _CATEGORY_PROMPTS["multi-hop"]


# ══════════════════════════════════════════════════════════════════════
# 3.  Empty / whitespace context passthrough
# ══════════════════════════════════════════════════════════════════════


class TestEmptyContext:
    @pytest.mark.parametrize("ctx", ["", "   ", "\n", "\t\n  "])
    def test_empty_or_whitespace_returns_unchanged(self, ctx):
        fn = _make_llm_fn()
        result = compress_context(ctx, "q?", fn)
        assert result == ctx
        fn.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
# 4.  Default query_type → default system prompt
# ══════════════════════════════════════════════════════════════════════


class TestDefaultQueryType:
    def test_none_query_type_uses_default_prompt(self):
        fn = _make_llm_fn()
        compress_context("ctx", "q?", fn, query_type=None)
        messages = fn.call_args[0][0]
        assert messages[0]["content"] == COMPRESS_SYSTEM_PROMPT

    def test_omitted_query_type_uses_default_prompt(self):
        fn = _make_llm_fn()
        compress_context("ctx", "q?", fn)
        messages = fn.call_args[0][0]
        assert messages[0]["content"] == COMPRESS_SYSTEM_PROMPT


# ══════════════════════════════════════════════════════════════════════
# 5.  Unknown query_type → default system prompt
# ══════════════════════════════════════════════════════════════════════


class TestUnknownQueryType:
    @pytest.mark.parametrize("qtype", ["unknown", "factoid", "ADVERSARIAL", ""])
    def test_unknown_types_fall_back_to_default(self, qtype):
        fn = _make_llm_fn()
        compress_context("ctx", "q?", fn, query_type=qtype)
        messages = fn.call_args[0][0]
        assert messages[0]["content"] == COMPRESS_SYSTEM_PROMPT


# ══════════════════════════════════════════════════════════════════════
# 6.  Message format correctness
# ══════════════════════════════════════════════════════════════════════


class TestMessageFormat:
    def test_messages_are_list_of_two_dicts(self):
        fn = _make_llm_fn()
        compress_context("ctx", "q?", fn)
        messages = fn.call_args[0][0]
        assert isinstance(messages, list) and len(messages) == 2
        assert all(isinstance(m, dict) for m in messages)

    def test_first_message_is_system_role(self):
        fn = _make_llm_fn()
        compress_context("ctx", "q?", fn)
        assert fn.call_args[0][0][0]["role"] == "system"

    def test_second_message_is_user_role(self):
        fn = _make_llm_fn()
        compress_context("ctx", "q?", fn)
        assert fn.call_args[0][0][1]["role"] == "user"

    def test_user_message_contains_question_and_context(self):
        fn = _make_llm_fn()
        compress_context("my special context", "Who is Alice?", fn)
        user_content = fn.call_args[0][0][1]["content"]
        assert "Who is Alice?" in user_content
        assert "my special context" in user_content


# ══════════════════════════════════════════════════════════════════════
# 7.  max_tokens forwarded to llm_fn
# ══════════════════════════════════════════════════════════════════════


class TestMaxTokens:
    def test_default_max_tokens_is_400(self):
        fn = _make_llm_fn()
        compress_context("ctx", "q?", fn)
        assert fn.call_args[1]["max_tokens"] == 400

    def test_custom_max_tokens_forwarded(self):
        fn = _make_llm_fn()
        compress_context("ctx", "q?", fn, max_tokens=1024)
        assert fn.call_args[1]["max_tokens"] == 1024


# ══════════════════════════════════════════════════════════════════════
# 8.  Model parameter forwarding
# ══════════════════════════════════════════════════════════════════════


class TestModelParam:
    def test_default_model(self):
        fn = _make_llm_fn()
        compress_context("ctx", "q?", fn)
        assert fn.call_args[1]["model"] == "gpt-4o-mini"

    def test_custom_model_forwarded(self):
        fn = _make_llm_fn()
        compress_context("ctx", "q?", fn, model="claude-opus-4-6")
        assert fn.call_args[1]["model"] == "claude-opus-4-6"


# ══════════════════════════════════════════════════════════════════════
# 9.  Template rendering
# ══════════════════════════════════════════════════════════════════════


class TestTemplateRendering:
    def test_template_has_placeholders(self):
        assert "{question}" in COMPRESS_USER_TEMPLATE
        assert "{context}" in COMPRESS_USER_TEMPLATE

    def test_rendered_user_message_matches_template(self):
        fn = _make_llm_fn()
        compress_context("CTX_BLOCK", "Q_BLOCK", fn)
        user_content = fn.call_args[0][0][1]["content"]
        expected = COMPRESS_USER_TEMPLATE.format(question="Q_BLOCK", context="CTX_BLOCK")
        assert user_content == expected


# ══════════════════════════════════════════════════════════════════════
# 10. Constants integrity
# ══════════════════════════════════════════════════════════════════════


class TestConstants:
    def test_category_prompts_has_exactly_three_keys(self):
        assert set(_CATEGORY_PROMPTS.keys()) == {"adversarial", "temporal", "multi-hop"}

    def test_compress_system_prompt_is_nonempty_string(self):
        assert isinstance(COMPRESS_SYSTEM_PROMPT, str) and len(COMPRESS_SYSTEM_PROMPT) > 50

    def test_each_category_prompt_is_nonempty_string(self):
        for key, val in _CATEGORY_PROMPTS.items():
            assert isinstance(val, str) and len(val) > 50, f"{key} prompt too short"
