"""Tests for v3.4.0 retrieval features.

Covers union_recall, iterative_recall, chain_of_note, temporal_metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mind_mem.chain_of_note import _clean_bullets, chain_of_note_pack
from mind_mem.iterative_recall import _extract_followups, iterative_retrieve
from mind_mem.temporal_metadata import annotate_with_temporal_metadata
from mind_mem.union_recall import union_retrieve

# ---------------------------------------------------------------------------
# union_recall
# ---------------------------------------------------------------------------


class TestUnionRetrieve:
    def _corpus(self) -> dict[str, list[dict]]:
        return {
            "alice oauth": [
                {"_id": "A1", "excerpt": "Alice migrated to OIDC"},
                {"_id": "A2", "excerpt": "OIDC cutover in March"},
                {"_id": "B1", "excerpt": "Bob uses OAuth2"},
            ],
            "bob rate limiting": [
                {"_id": "B1", "excerpt": "Bob uses OAuth2"},
                {"_id": "B2", "excerpt": "Rate limit=100/min"},
            ],
        }

    def test_basic_union_preserves_all_unique(self):
        corpus = self._corpus()
        result = union_retrieve(
            ["alice oauth", "bob rate limiting"],
            lambda q: corpus.get(q, []),
            top_k_per_query=3,
            max_total=10,
        )
        ids = [b["_id"] for b in result]
        assert set(ids) == {"A1", "A2", "B1", "B2"}
        # First-seen order: alice's hits, then bob's
        assert ids[0] == "A1"
        assert ids[1] == "A2"

    def test_duplicate_block_kept_once_with_best_rank(self):
        corpus = self._corpus()
        result = union_retrieve(
            ["alice oauth", "bob rate limiting"],
            lambda q: corpus.get(q, []),
            top_k_per_query=3,
        )
        b1_hits = [b for b in result if b["_id"] == "B1"]
        assert len(b1_hits) == 1

    def test_empty_queries_skipped(self):
        corpus = self._corpus()
        result = union_retrieve(
            ["", "alice oauth", "   "],
            lambda q: corpus.get(q, []),
        )
        assert len(result) == 3  # only alice oauth hits matter

    def test_max_total_caps_output(self):
        large = {"q": [{"_id": f"X{i}", "excerpt": f"hit{i}"} for i in range(50)]}
        result = union_retrieve(["q"], lambda q: large[q], top_k_per_query=50, max_total=10)
        assert len(result) == 10

    def test_retriever_exception_is_fail_open(self):
        def bad(q):
            raise RuntimeError("boom")

        # Should not raise; returns empty list.
        result = union_retrieve(["q"], bad, top_k_per_query=5)
        assert result == []

    def test_first_seen_round_annotation(self):
        corpus = self._corpus()
        result = union_retrieve(
            ["alice oauth", "bob rate limiting"],
            lambda q: corpus.get(q, []),
            top_k_per_query=3,
        )
        by_id = {b["_id"]: b for b in result}
        assert by_id["A1"]["_union_first_seen"] == 0
        assert by_id["B2"]["_union_first_seen"] == 1


# ---------------------------------------------------------------------------
# iterative_recall
# ---------------------------------------------------------------------------


class TestExtractFollowups:
    def test_json_dict_shape(self):
        raw = '{"followups": ["query a", "query b"]}'
        assert _extract_followups(raw, max_followups=4) == ["query a", "query b"]

    def test_json_in_markdown_fence(self):
        raw = '```json\n{"followups": ["Alice OAuth migration"]}\n```'
        assert _extract_followups(raw, max_followups=4) == ["Alice OAuth migration"]

    def test_uppercase_language_tag_in_fence(self):
        raw = '```JSON\n{"followups": ["Bob rate limiting policy"]}\n```'
        assert _extract_followups(raw, max_followups=4) == ["Bob rate limiting policy"]

    def test_safe_regex_rejects_injection(self):
        raw = '{"followups": ["; DROP TABLE blocks; --", "Alice OAuth"]}'
        out = _extract_followups(raw, max_followups=4)
        assert "Alice OAuth" in out
        assert not any("DROP TABLE" in q for q in out)

    def test_done_token_returns_empty(self):
        assert _extract_followups("DONE", max_followups=4) == []
        assert _extract_followups("DONE — nothing else needed", max_followups=4) == []

    def test_line_fallback_when_json_fails(self):
        raw = "Here are follow-ups:\n- query one\n- query two"
        out = _extract_followups(raw, max_followups=4)
        assert out == ["query one", "query two"]

    def test_respects_max_followups(self):
        raw = '{"followups": ["query one", "query two", "query three", "query four", "query five"]}'
        assert len(_extract_followups(raw, max_followups=2)) == 2


class TestIterativeRetrieve:
    def test_seed_only_when_llm_says_done(self):
        corpus = {"q1": [{"_id": "A1", "excerpt": "evidence"}]}
        result = iterative_retrieve(
            "q1",
            retrieve_fn=lambda q: corpus.get(q, []),
            llm_fn=lambda _: "DONE",
            max_rounds=3,
        )
        assert [b["_id"] for b in result] == ["A1"]
        assert result[0]["_iter_round"] == 0

    def test_followups_add_new_evidence(self):
        corpus = {
            "q1": [{"_id": "A1", "excerpt": "first hit"}],
            "followup_q": [{"_id": "B1", "excerpt": "bridge evidence"}],
        }
        responses = iter(['{"followups": ["followup_q"]}', "DONE"])

        result = iterative_retrieve(
            "q1",
            retrieve_fn=lambda q: corpus.get(q, []),
            llm_fn=lambda _: next(responses),
            max_rounds=3,
        )
        ids = [b["_id"] for b in result]
        assert "A1" in ids and "B1" in ids
        by_id = {b["_id"]: b for b in result}
        assert by_id["A1"]["_iter_round"] == 0
        assert by_id["B1"]["_iter_round"] == 1

    def test_max_rounds_enforced(self):
        calls = {"n": 0}

        def llm(_):
            calls["n"] += 1
            return f'{{"followups": ["q_round_{calls["n"]}"]}}'

        corpus = {"original": [{"_id": "A0", "excerpt": "seed"}]}
        # Every round generates new followups; max_rounds=2 means only
        # one LLM call (round 0 is seed, round 1 calls LLM once).
        iterative_retrieve(
            "original",
            retrieve_fn=lambda q: corpus.get(q, []),
            llm_fn=llm,
            max_rounds=2,
        )
        # max_rounds=2 → rounds 0 and 1 → exactly 1 LLM call
        assert calls["n"] == 1

    def test_rejects_max_rounds_zero(self):
        with pytest.raises(ValueError):
            iterative_retrieve("q", lambda _: [], lambda _: "", max_rounds=0)

    def test_empty_question_returns_empty(self):
        assert iterative_retrieve("", lambda _: [], lambda _: "") == []
        assert iterative_retrieve("   ", lambda _: [], lambda _: "") == []


# ---------------------------------------------------------------------------
# chain_of_note
# ---------------------------------------------------------------------------


class TestCleanBullets:
    def test_strips_bullet_markers(self):
        raw = "- first bullet\n* second bullet\n• third bullet"
        assert _clean_bullets(raw) == ["first bullet", "second bullet", "third bullet"]

    def test_no_direct_evidence_returns_empty(self):
        raw = "(no direct evidence)"
        assert _clean_bullets(raw) == []

    def test_drops_preamble_lines(self):
        raw = "Here are the key facts:\n- Fact one\n- Fact two"
        assert _clean_bullets(raw) == ["Fact one", "Fact two"]

    def test_strips_markdown_fence(self):
        raw = "```\n- fact a\n- fact b\n```"
        assert _clean_bullets(raw) == ["fact a", "fact b"]


class TestChainOfNotePack:
    def test_empty_blocks_returns_empty(self):
        assert chain_of_note_pack("q", [], lambda p: "") == ""

    def test_condenser_returns_clean_bullets(self):
        blocks = [
            {"_id": "A1", "excerpt": "Alice used OAuth"},
            {"_id": "A2", "excerpt": "Bob used OIDC"},
        ]
        response = "- Alice used OAuth [1]\n- Bob used OIDC [2]"
        out = chain_of_note_pack("q", blocks, lambda _: response)
        assert "Alice used OAuth [1]" in out
        assert "Bob used OIDC [2]" in out

    def test_empty_response_falls_back_to_raw(self):
        blocks = [{"_id": "A1", "excerpt": "Raw fact"}]
        out = chain_of_note_pack("q", blocks, lambda _: "", fallback_on_empty=True)
        assert "Raw fact" in out

    def test_empty_response_without_fallback_returns_empty(self):
        blocks = [{"_id": "A1", "excerpt": "Raw fact"}]
        out = chain_of_note_pack("q", blocks, lambda _: "", fallback_on_empty=False)
        assert out == ""

    def test_llm_exception_falls_back(self):
        blocks = [{"_id": "A1", "excerpt": "Raw"}]

        def bad(_):
            raise RuntimeError("boom")

        out = chain_of_note_pack("q", blocks, bad, fallback_on_empty=True)
        assert "Raw" in out


# ---------------------------------------------------------------------------
# temporal_metadata
# ---------------------------------------------------------------------------


class TestTemporalMetadata:
    def _now(self) -> datetime:
        return datetime(2026, 4, 21, tzinfo=timezone.utc)

    def test_prefixes_days_ago(self):
        blocks = [{"excerpt": "fact", "created_at": "2026-04-11"}]
        out = annotate_with_temporal_metadata(blocks, now=self._now())
        assert out[0]["excerpt"].startswith("[Block date: 2026-04-11]")
        assert "fact" in out[0]["excerpt"]

    def test_original_block_not_mutated(self):
        blocks = [{"excerpt": "fact", "date": "2026-01-01"}]
        _ = annotate_with_temporal_metadata(blocks, now=self._now())
        assert blocks[0]["excerpt"] == "fact"

    def test_unparseable_date_leaves_excerpt_alone(self):
        blocks = [{"excerpt": "fact", "created_at": "not-a-date"}]
        out = annotate_with_temporal_metadata(blocks, now=self._now())
        assert out[0]["excerpt"] == "fact"

    def test_unix_timestamp_accepted(self):
        ts = int(datetime(2026, 4, 1, tzinfo=timezone.utc).timestamp())
        blocks = [{"excerpt": "fact", "timestamp": ts}]
        out = annotate_with_temporal_metadata(blocks, now=self._now())
        assert "Block date: 2026-04-01" in out[0]["excerpt"]

    def test_locomo_style_date_parses(self):
        blocks = [{"Statement": "text", "Date": "1:56 pm on 7 May, 2023"}]
        out = annotate_with_temporal_metadata(blocks, now=self._now())
        assert out[0]["Statement"].startswith("[Block date: 2023-05-07]")

    def test_capitalised_date_key_picked_up(self):
        blocks = [{"excerpt": "text", "Date": "2023-05-07"}]
        out = annotate_with_temporal_metadata(blocks, now=self._now())
        assert out[0]["excerpt"].startswith("[Block date: 2023-05-07]")

    def test_nested_metadata_date(self):
        blocks = [{"excerpt": "fact", "metadata": {"created_at": "2026-04-21"}}]
        out = annotate_with_temporal_metadata(blocks, now=self._now())
        assert out[0]["excerpt"].startswith("[Block date: 2026-04-21]")

    def test_future_date_rejected(self):
        blocks = [{"excerpt": "fact", "created_at": "2099-01-01"}]
        out = annotate_with_temporal_metadata(blocks, now=self._now())
        assert out[0]["excerpt"] == "fact"  # delta < 0 → skip

    def test_iso_with_z_suffix(self):
        blocks = [{"excerpt": "fact", "created_at": "2026-04-20T12:00:00Z"}]
        out = annotate_with_temporal_metadata(blocks, now=self._now())
        assert "Block date: 2026-04-20" in out[0]["excerpt"]

    def test_records_delta_days_field(self):
        blocks = [{"excerpt": "fact", "created_at": "2026-04-15"}]
        out = annotate_with_temporal_metadata(blocks, now=self._now())
        assert out[0]["_temporal_delta_days"] == 6
