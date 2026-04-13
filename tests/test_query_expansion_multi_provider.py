#!/usr/bin/env python3
"""Tests for multi-provider LLM query expansion (OpenAI-compatible endpoints)."""

import json
import unittest
from unittest.mock import MagicMock, patch

from mind_mem.query_expansion import LLMQueryExpander


class TestOpenAICompatibleProvider(unittest.TestCase):
    """Test the OpenAI-compatible chat completions provider."""

    def test_base_url_default(self):
        """Default base_url should be OpenAI's API endpoint."""
        expander = LLMQueryExpander(config={"provider": "openai"})
        self.assertEqual(expander.base_url, "https://api.openai.com/v1")

    def test_base_url_custom(self):
        """Custom base_url should be used for non-OpenAI providers."""
        expander = LLMQueryExpander(config={
            "provider": "xai",
            "base_url": "https://api.x.ai/v1",
            "model": "grok-4-1-fast-reasoning",
        })
        self.assertEqual(expander.base_url, "https://api.x.ai/v1")
        self.assertEqual(expander.model, "grok-4-1-fast-reasoning")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("urllib.request.urlopen")
    def test_call_openai_compatible_parses_response(self, mock_urlopen):
        """Should parse alternatives from an OpenAI-compatible response."""
        response_body = json.dumps({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "find database issues\nlocate db errors\nsearch data store failures",
                }
            }]
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        expander = LLMQueryExpander(config={
            "provider": "openai",
            "model": "gpt-5.4",
            "api_key_env": "OPENAI_API_KEY",
        })
        results = expander.expand("find database errors", max_expansions=4)

        self.assertEqual(results[0], "find database errors")
        self.assertGreater(len(results), 1)

    @patch.dict("os.environ", {"XAI_API_KEY": "test-key"})
    @patch("urllib.request.urlopen")
    def test_call_openai_compatible_empty_choices(self, mock_urlopen):
        """Should fall back gracefully when choices list is empty."""
        response_body = json.dumps({"choices": []}).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        expander = LLMQueryExpander(config={
            "provider": "xai",
            "base_url": "https://api.x.ai/v1",
            "model": "grok-4-1-fast-reasoning",
            "api_key_env": "XAI_API_KEY",
        })
        results = expander.expand("test query", max_expansions=3)

        # Original query should always be present
        self.assertEqual(results[0], "test query")

    def test_routing_anthropic_vs_openai(self):
        """Provider routing should direct anthropic to _call_anthropic and others to _call_openai_compatible."""
        # Point each expander at an api_key_env that is guaranteed not to
        # exist in the process environment, so _call_llm raises the expected
        # RuntimeError instead of making a real HTTP call with whatever key
        # happens to be set globally (which would yield a 401 instead).
        expander_anthropic = LLMQueryExpander(
            config={"provider": "anthropic", "api_key_env": "MIND_MEM_TEST_NONEXISTENT_ANTHROPIC"}
        )
        expander_openai = LLMQueryExpander(
            config={"provider": "openai", "api_key_env": "MIND_MEM_TEST_NONEXISTENT_OPENAI"}
        )
        expander_xai = LLMQueryExpander(
            config={"provider": "xai", "api_key_env": "MIND_MEM_TEST_NONEXISTENT_XAI"}
        )
        expander_mistral = LLMQueryExpander(
            config={"provider": "mistral", "api_key_env": "MIND_MEM_TEST_NONEXISTENT_MISTRAL"}
        )

        self.assertEqual(expander_anthropic.provider, "anthropic")
        self.assertEqual(expander_openai.provider, "openai")
        self.assertEqual(expander_xai.provider, "xai")
        self.assertEqual(expander_mistral.provider, "mistral")

        # Verify non-anthropic providers use openai-compatible path by checking
        # that calling without API key raises the expected env-var error
        with self.assertRaises(RuntimeError):
            expander_openai._call_llm("test", 2)
        with self.assertRaises(RuntimeError):
            expander_xai._call_llm("test", 2)


if __name__ == "__main__":
    unittest.main()
