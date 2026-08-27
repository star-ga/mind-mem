# Copyright 2026 STARGA, Inc.
"""Tests for the conversational chat layer (`chat_with_memory`).

Every test here uses a **deterministic stub generator injected through
the ``generator=`` seam** — no accelerator, no network, no service
process. The stubs are plain functions over ``ChatRequest``; the real
service adapter is never constructed.

Covers the acceptance gate:

* every claim sentence carries >= 1 ``[[block_id]]`` that resolves;
* a fabricated / unresolvable id fails validation (both raise + reject);
* empty recall returns the literal ``"no record found"`` and never
  reaches a generator;
* the whole surface is reproducible byte-for-byte across runs.
"""

from __future__ import annotations

import json
import os

import pytest

from mind_mem.chat_citations import (
    NO_RECORD,
    CitationError,
    extract_citations,
    split_claim_sentences,
    strip_citations,
    validate_answer,
)
from mind_mem.chat_generators import (
    ChatRequest,
    EvidenceItem,
    extractive_generator,
    resolve_generator,
)
from mind_mem.chat_memory import (
    MAX_QUESTION_CHARS,
    ChatAnswer,
    chat_with_memory,
    make_workspace_resolver,
)

DECISIONS_MD = """\
[D-20260301-001]
Statement: Deploys run on Friday afternoons behind a feature flag
Status: active
Date: 2026-03-01

---

[D-20260301-002]
Statement: Postgres is the system of record for block storage
Status: active
Date: 2026-03-01
"""

TASKS_MD = """\
[T-20260301-001]
Title: Automate the Friday deploy checklist
Status: active
Date: 2026-03-01
"""

FABRICATED_ID = "D-19990101-999"


@pytest.fixture()
def workspace(tmp_path) -> str:
    """A minimal Markdown-backed workspace with three real blocks."""
    ws = tmp_path / "ws"
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (ws / sub).mkdir(parents=True)
    (ws / "decisions" / "DECISIONS.md").write_text(DECISIONS_MD, encoding="utf-8")
    (ws / "tasks" / "TASKS.md").write_text(TASKS_MD, encoding="utf-8")
    (ws / "mind-mem.json").write_text('{"recall": {"backend": "bm25"}}', encoding="utf-8")
    return str(ws)


# ---------------------------------------------------------------------------
# Deterministic stub generators (dependency-injected — never a service)
# ---------------------------------------------------------------------------


def stub_cited(request: ChatRequest) -> str:
    """One cited sentence per evidence item. Deterministic."""
    return " ".join(f"Evidence {i} says something [[{item.block_id}]]." for i, item in enumerate(request.evidence, start=1))


def stub_fabricating(request: ChatRequest) -> str:
    """Cites a block id that does not exist in the workspace."""
    return f"The deploy window moved to Tuesday [[{FABRICATED_ID}]]."


def stub_uncited(request: ChatRequest) -> str:
    """Makes a claim with no citation at all."""
    return "The deploy window moved to Tuesday."


def stub_partially_cited(request: ChatRequest) -> str:
    """First sentence cited, second one not."""
    first = request.evidence[0].block_id
    return f"Deploys are on Friday [[{first}]]. They are also sometimes on Monday."


def stub_explodes(request: ChatRequest) -> str:  # pragma: no cover — must never run
    raise AssertionError("generator must not be invoked when recall is empty")


def stub_non_string(request: ChatRequest):
    return 42


def empty_recall(workspace: str, question: str, limit: int) -> list:
    return []


# ---------------------------------------------------------------------------
# Gate 1 — every claim sentence carries a resolving citation
# ---------------------------------------------------------------------------


class TestGroundedAnswers:
    def test_every_claim_sentence_carries_a_resolving_citation(self, workspace):
        result = chat_with_memory(workspace, "when do deploys run?", generator=stub_cited)

        assert isinstance(result, ChatAnswer)
        assert result.grounded is True
        assert result.rejected is False
        assert result.citations

        resolver = make_workspace_resolver(workspace)
        for sentence in split_claim_sentences(result.answer):
            cited = extract_citations(sentence)
            assert cited, f"uncited claim sentence: {sentence!r}"
            for block_id in cited:
                assert resolver(block_id), f"citation does not resolve: {block_id}"

    def test_citations_resolve_through_the_block_store(self, workspace):
        from mind_mem.storage import get_block_store

        result = chat_with_memory(workspace, "what is the system of record?", generator=stub_cited)
        store = get_block_store(workspace)
        for block_id in result.citations:
            assert store.get_by_id(block_id) is not None

    def test_default_generator_is_grounded_and_offline(self, workspace):
        """The in-box default needs no injection, no network, no accelerator."""
        result = chat_with_memory(workspace, "deploys friday", limit=3)

        assert result.grounded is True
        assert result.citations
        resolver = make_workspace_resolver(workspace)
        for sentence in split_claim_sentences(result.answer):
            assert extract_citations(sentence)
        assert all(resolver(bid) for bid in result.citations)

    def test_answer_is_reproducible_byte_for_byte(self, workspace):
        first = chat_with_memory(workspace, "deploys friday", generator=stub_cited)
        second = chat_with_memory(workspace, "deploys friday", generator=stub_cited)
        assert first.answer == second.answer
        assert first.citations == second.citations

    def test_evidence_ids_are_exposed_for_audit(self, workspace):
        result = chat_with_memory(workspace, "deploys friday", generator=stub_cited)
        evidence_ids = [item.block_id for item in result.evidence]
        assert evidence_ids
        assert set(result.citations) <= set(evidence_ids)

    def test_to_dict_is_json_serialisable(self, workspace):
        result = chat_with_memory(workspace, "deploys friday", generator=stub_cited)
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["grounded"] is True
        assert payload["report"]["ok"] is True


# ---------------------------------------------------------------------------
# Gate 2 — fabricated / unresolvable ids fail validation
# ---------------------------------------------------------------------------


class TestFabricatedCitations:
    def test_fabricated_id_raises_by_default(self, workspace):
        with pytest.raises(CitationError) as excinfo:
            chat_with_memory(workspace, "when do deploys run?", generator=stub_fabricating)
        assert FABRICATED_ID in excinfo.value.report.unresolved
        assert excinfo.value.report.ok is False

    def test_fabricated_id_rejected_in_reject_mode(self, workspace):
        result = chat_with_memory(
            workspace,
            "when do deploys run?",
            generator=stub_fabricating,
            on_invalid="reject",
        )
        assert result.rejected is True
        assert result.grounded is False
        assert result.answer == NO_RECORD
        assert FABRICATED_ID in result.report.unresolved

    def test_uncited_claim_is_rejected(self, workspace):
        with pytest.raises(CitationError) as excinfo:
            chat_with_memory(workspace, "when do deploys run?", generator=stub_uncited)
        assert excinfo.value.report.uncited_sentences

    def test_partially_cited_answer_is_rejected(self, workspace):
        result = chat_with_memory(
            workspace,
            "when do deploys run?",
            generator=stub_partially_cited,
            on_invalid="reject",
        )
        assert result.rejected is True
        assert any("Monday" in s for s in result.report.uncited_sentences)

    def test_real_id_outside_evidence_is_recorded_and_optionally_fatal(self, workspace):
        """A resolvable id the generator was never shown is flagged."""
        outsider = "T-20260301-001"

        def cite_outsider(request: ChatRequest) -> str:
            return f"Some unrelated claim [[{outsider}]]."

        def only_first_decision(ws, question, limit):
            return [{"_id": "D-20260301-001", "excerpt": "Deploys run on Friday", "score": 1.0}]

        lenient = chat_with_memory(
            workspace,
            "deploys",
            generator=cite_outsider,
            recall_fn=only_first_decision,
        )
        assert lenient.grounded is True
        assert outsider in lenient.report.out_of_evidence

        strict = chat_with_memory(
            workspace,
            "deploys",
            generator=cite_outsider,
            recall_fn=only_first_decision,
            require_in_evidence=True,
            on_invalid="reject",
        )
        assert strict.rejected is True

    def test_workspace_resolver_fails_closed(self, workspace):
        """An id we cannot prove exists is treated as fabricated."""
        resolver = make_workspace_resolver(workspace)
        assert resolver(FABRICATED_ID) is False
        assert resolver("") is False
        assert resolver(None) is False
        assert resolver("D-20260301-001") is True


# ---------------------------------------------------------------------------
# Gate 3 — empty recall returns the literal marker, never a fabrication
# ---------------------------------------------------------------------------


class TestEmptyRecall:
    def test_empty_recall_returns_literal_no_record(self, workspace):
        result = chat_with_memory(workspace, "anything", generator=stub_explodes, recall_fn=empty_recall)
        assert result.answer == "no record found"
        assert result.answer == NO_RECORD
        assert result.no_record is True
        assert result.citations == ()
        assert result.grounded is True

    def test_generator_is_never_invoked_on_empty_recall(self, workspace):
        calls: list[ChatRequest] = []

        def recording(request: ChatRequest) -> str:  # pragma: no cover — asserted empty
            calls.append(request)
            return "fabricated"

        chat_with_memory(workspace, "anything", generator=recording, recall_fn=empty_recall)
        assert calls == []

    def test_hits_without_ids_count_as_empty_recall(self, workspace):
        def idless(ws, question, limit):
            return [{"excerpt": "no id here"}, {"_id": "   "}]

        result = chat_with_memory(workspace, "anything", generator=stub_explodes, recall_fn=idless)
        assert result.answer == NO_RECORD

    def test_real_recall_on_an_absent_topic_yields_no_record(self, workspace):
        result = chat_with_memory(workspace, "zzqqxx nonexistent submarine telemetry", limit=5)
        assert result.answer == NO_RECORD
        assert result.no_record is True

    def test_extractive_generator_returns_marker_without_evidence(self):
        request = ChatRequest(question="q", prompt="p", evidence=())
        assert extractive_generator(request) == NO_RECORD


# ---------------------------------------------------------------------------
# Boundary validation
# ---------------------------------------------------------------------------


class TestBoundaryValidation:
    def test_missing_workspace_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="workspace not found"):
            chat_with_memory(str(tmp_path / "nope"), "q", generator=stub_cited)

    @pytest.mark.parametrize("bad", ["", "   ", None, 7])
    def test_bad_workspace_rejected(self, bad):
        with pytest.raises(ValueError, match="workspace"):
            chat_with_memory(bad, "q", generator=stub_cited)

    @pytest.mark.parametrize("bad", ["", "   ", None, 7])
    def test_bad_question_rejected(self, workspace, bad):
        with pytest.raises(ValueError, match="question"):
            chat_with_memory(workspace, bad, generator=stub_cited)

    def test_oversized_question_rejected(self, workspace):
        with pytest.raises(ValueError, match="characters"):
            chat_with_memory(workspace, "x" * (MAX_QUESTION_CHARS + 1), generator=stub_cited)

    @pytest.mark.parametrize("bad", [0, -1, 51, True, "8"])
    def test_bad_limit_rejected(self, workspace, bad):
        with pytest.raises(ValueError, match="limit"):
            chat_with_memory(workspace, "q", limit=bad, generator=stub_cited)

    def test_bad_on_invalid_mode_rejected(self, workspace):
        with pytest.raises(ValueError, match="on_invalid"):
            chat_with_memory(workspace, "q", generator=stub_cited, on_invalid="explode")

    def test_non_string_generator_output_rejected(self, workspace):
        with pytest.raises(TypeError, match="must return str"):
            chat_with_memory(workspace, "deploys", generator=stub_non_string)


# ---------------------------------------------------------------------------
# Citation primitives
# ---------------------------------------------------------------------------


class TestCitationPrimitives:
    def test_extract_citations_is_ordered_and_deduped(self):
        text = "a [[B-2]] b [[B-1]] c [[B-2]]"
        assert extract_citations(text) == ("B-2", "B-1")

    def test_extract_citations_tolerates_non_strings(self):
        assert extract_citations(None) == ()
        assert extract_citations("") == ()

    def test_strip_citations(self):
        assert strip_citations("Deploys are Friday [[D-1]].") == "Deploys are Friday ."

    def test_bullets_are_separate_claims(self):
        answer = "- Deploys are Friday [[D-1]]\n- Postgres is the store [[D-2]]"
        claims = split_claim_sentences(answer)
        assert len(claims) == 2
        assert all(extract_citations(c) for c in claims)

    def test_fenced_code_is_not_a_claim(self):
        answer = "```\ncode block\n```\nDeploys are Friday [[D-1]]."
        claims = split_claim_sentences(answer)
        assert len(claims) == 1
        assert extract_citations(claims[0]) == ("D-1",)

    def test_a_heading_that_asserts_something_is_still_a_claim(self):
        """Fail-closed: a heading is the easiest place to smuggle a claim."""
        report = validate_answer("# Deploys moved to Tuesday", resolver=lambda _: True)
        assert report.ok is False
        assert report.uncited_sentences

    def test_pure_decoration_is_not_a_claim(self):
        assert split_claim_sentences("---\n***\n   \n") == ()

    def test_no_record_answer_has_no_claims(self):
        assert split_claim_sentences(NO_RECORD) == ()
        assert split_claim_sentences("No record found.") == ()

    def test_validate_answer_accepts_the_no_record_marker(self):
        report = validate_answer(NO_RECORD, resolver=lambda _: False)
        assert report.ok is True
        assert report.citations == ()

    def test_validate_answer_rejects_empty_output(self):
        report = validate_answer("   ", resolver=lambda _: True)
        assert report.ok is False

    def test_chunked_citation_resolves_via_parent_id(self):
        known = {"D-20260301-001"}
        report = validate_answer(
            "Deploys are Friday [[D-20260301-001.2]].",
            resolver=lambda bid: bid in known,
        )
        assert report.ok is True

    def test_report_summary_names_the_violation(self):
        report = validate_answer("Claim [[X-1]].", resolver=lambda _: False)
        assert "X-1" in report.summary()
        assert report.to_dict()["ok"] is False

    def test_resolver_must_be_callable(self):
        with pytest.raises(TypeError):
            validate_answer("x", resolver="not-callable")


# ---------------------------------------------------------------------------
# Chain-of-note condensation is opt-in and off by default
# ---------------------------------------------------------------------------


class TestCondenserIsOptIn:
    def _record_prompt(self, sink):
        def generator(request: ChatRequest) -> str:
            sink.append(request.prompt)
            return f"Recorded [[{request.evidence[0].block_id}]]."

        return generator

    def test_default_path_does_not_condense(self, workspace):
        prompts: list[str] = []
        chat_with_memory(workspace, "deploys friday", generator=self._record_prompt(prompts))
        assert len(prompts) == 1
        # The un-condensed prompt renders the raw evidence lines verbatim.
        assert "[[D-20260301-001]]" in prompts[0]
        assert "[Block date: 2026-03-01]" in prompts[0]

    def test_flag_off_prompt_is_byte_identical_to_explicit_none(self, workspace):
        with_default: list[str] = []
        with_none: list[str] = []
        chat_with_memory(workspace, "deploys friday", generator=self._record_prompt(with_default))
        chat_with_memory(workspace, "deploys friday", generator=self._record_prompt(with_none), condenser=None)
        assert with_default[0] == with_none[0]

    def test_condenser_notes_are_anchored_to_block_ids(self, workspace):
        prompts: list[str] = []

        def condenser(prompt: str) -> str:
            return "Deploys run on Friday [1]\nPostgres stores the blocks [2]"

        chat_with_memory(
            workspace,
            "deploys friday",
            generator=self._record_prompt(prompts),
            condenser=condenser,
        )
        assert "[1]" not in prompts[0]
        assert "[[D-20260301-001]]" in prompts[0]

    def test_out_of_range_note_markers_are_dropped_not_misattributed(self, workspace):
        prompts: list[str] = []

        def condenser(prompt: str) -> str:
            return "Real note [1]\nBogus note [99]"

        chat_with_memory(
            workspace,
            "deploys friday",
            generator=self._record_prompt(prompts),
            condenser=condenser,
        )
        assert "[99]" not in prompts[0]

    def test_useless_condenser_output_falls_back_with_a_warning(self, workspace):
        prompts: list[str] = []

        def condenser(prompt: str) -> str:
            return "(no direct evidence)"

        result = chat_with_memory(
            workspace,
            "deploys friday",
            generator=self._record_prompt(prompts),
            condenser=condenser,
        )
        assert any("chain-of-note" in w for w in result.warnings)
        assert "[[D-20260301-001]]" in prompts[0]


# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------


class TestGeneratorRegistry:
    def test_extractive_is_the_named_default(self):
        assert resolve_generator("extractive") is extractive_generator

    def test_service_generator_is_constructible_without_network(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:65535")
        generator = resolve_generator("service")
        assert callable(generator)

    @pytest.mark.parametrize("bad", ["", "   ", "gpt", None])
    def test_unknown_generator_rejected(self, bad):
        with pytest.raises(ValueError):
            resolve_generator(bad)

    def test_extractive_folds_multi_sentence_excerpts_into_one_claim(self):
        request = ChatRequest(
            question="q",
            prompt="p",
            evidence=(EvidenceItem(block_id="D-1", excerpt="First part. Second part."),),
        )
        answer = extractive_generator(request)
        claims = split_claim_sentences(answer)
        assert len(claims) == 1
        assert extract_citations(claims[0]) == ("D-1",)

    def test_extractive_truncates_long_excerpts(self):
        request = ChatRequest(
            question="q",
            prompt="p",
            evidence=(EvidenceItem(block_id="D-1", excerpt="x" * 900),),
        )
        answer = extractive_generator(request, max_chars=50)
        assert len(answer) < 200
        assert "[[D-1]]" in answer


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestCli:
    def test_grounded_answer_exits_zero(self, workspace, capsys):
        from mind_mem.chat_cli import main

        code = main(["-w", workspace, "-n", "3", "deploys friday"])
        out = capsys.readouterr().out
        assert code == 0
        assert "[[D-20260301-001]]" in out
        assert "Sources:" in out

    def test_json_output_is_parseable(self, workspace, capsys):
        from mind_mem.chat_cli import main

        code = main(["-w", workspace, "--json", "deploys friday"])
        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        assert payload["citations"]

    def test_no_record_exit_code(self, workspace, capsys):
        from mind_mem.chat_cli import main

        code = main(["-w", workspace, "zzqqxx nonexistent submarine telemetry"])
        assert code == 2
        assert capsys.readouterr().out.strip().startswith(NO_RECORD)

    def test_unknown_generator_exit_code(self, workspace, capsys):
        from mind_mem.chat_cli import main

        code = main(["-w", workspace, "-g", "nope", "deploys"])
        assert code == 1
        assert "unknown generator" in capsys.readouterr().err

    def test_missing_workspace_exit_code(self, tmp_path, capsys):
        from mind_mem.chat_cli import main

        code = main(["-w", str(tmp_path / "absent"), "deploys"])
        assert code == 1

    def test_console_script_is_declared(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pyproject = open(os.path.join(root, "pyproject.toml"), encoding="utf-8").read()
        assert 'mind-mem-chat = "mind_mem.chat_cli:main"' in pyproject


# ---------------------------------------------------------------------------
# MCP surface
# ---------------------------------------------------------------------------


class TestMcpTool:
    def test_tool_is_user_scoped_in_the_acl(self):
        pytest.importorskip("fastmcp")
        from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS

        assert "chat_with_memory" in USER_TOOLS
        assert "chat_with_memory" not in ADMIN_TOOLS

    def test_tool_returns_cited_json(self, workspace, monkeypatch):
        pytest.importorskip("fastmcp")
        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        from mind_mem.mcp.tools import chat as chat_tool

        payload = json.loads(chat_tool.chat_with_memory("deploys friday", limit=3))
        assert payload["grounded"] is True
        assert payload["citations"]

    def test_tool_rejects_empty_question(self, workspace, monkeypatch):
        pytest.importorskip("fastmcp")
        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        from mind_mem.mcp.tools import chat as chat_tool

        payload = json.loads(chat_tool.chat_with_memory("   "))
        assert "error" in payload

    def test_tool_rejects_unknown_generator(self, workspace, monkeypatch):
        pytest.importorskip("fastmcp")
        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        from mind_mem.mcp.tools import chat as chat_tool

        payload = json.loads(chat_tool.chat_with_memory("deploys", generator="nope"))
        assert "unknown generator" in payload["error"]

    def test_tool_reports_no_record(self, workspace, monkeypatch):
        pytest.importorskip("fastmcp")
        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        from mind_mem.mcp.tools import chat as chat_tool

        payload = json.loads(chat_tool.chat_with_memory("zzqqxx nonexistent submarine telemetry"))
        assert payload["answer"] == NO_RECORD

    def test_tool_is_registered_on_the_server(self):
        pytest.importorskip("fastmcp")
        from mind_mem.mcp.tools import chat as chat_tool

        registered: list[str] = []

        class FakeMcp:
            def tool(self, fn):
                registered.append(fn.__name__)
                return fn

        chat_tool.register(FakeMcp())
        assert registered == ["chat_with_memory"]
