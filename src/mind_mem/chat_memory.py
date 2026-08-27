"""Conversational chat layer — grounded answers with ``[[block_id]]`` citations.

**New public surface** (roadmap Group B). ``chat_with_memory`` is the
question-answering front door to a mind-mem workspace: it recalls
evidence, asks a pluggable generator for an answer, and then *refuses
to return an ungrounded one*.

The guarantee
-------------
Every answer this module returns satisfies all three of:

1. **Every claim sentence carries at least one citation.** A sentence
   with no ``[[block_id]]`` is an uncited claim and fails validation.
2. **Every cited id resolves in the workspace.** Ids are resolved
   through the configured block store; a fabricated id can never
   survive. Unresolvable ids either raise :class:`~mind_mem.chat_citations.CitationError`
   (``on_invalid="raise"``, the default) or are rejected into the
   no-record answer (``on_invalid="reject"``).
3. **Empty recall returns the literal string** ``"no record found"``.
   No generator is invoked at all on that path, so there is nothing to
   fabricate from.

Composition, not new machinery
------------------------------
The layer wires together existing subsystems and adds only the
grounding contract:

* :mod:`mind_mem.recall` / :mod:`mind_mem.hybrid_recall` — evidence
  retrieval (injectable as ``recall_fn``).
* :mod:`mind_mem.answer_quality` — question-category classification and
  the per-category prompt template.
* :mod:`mind_mem.chain_of_note` — **opt-in** evidence condensation.
  Pass a ``condenser``; the ``[N]`` markers it emits are re-anchored to
  ``[[block_id]]`` before the answerer sees them. Default is ``None``
  (off), which leaves the prompt byte-identical to the un-condensed
  path.
* :mod:`mind_mem.chat_generators` — the answerer seam. The in-box
  :func:`~mind_mem.chat_generators.extractive_generator` is
  deterministic and offline; the service adapter is opt-in.
* :mod:`mind_mem.chat_citations` — extraction + validation.

Usage
-----
::

    from mind_mem.chat_memory import chat_with_memory

    result = chat_with_memory("/path/to/workspace", "when do we deploy?")
    print(result.answer)        # "... [[D-20260301-001]]."
    print(result.citations)     # ("D-20260301-001",)

Surfaces: this function (Python API), the ``chat_with_memory`` MCP tool
(:mod:`mind_mem.mcp.tools.chat`), and the ``mind-mem-chat`` console
script (:mod:`mind_mem.chat_cli`).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .answer_quality import classify_question_category, prompt_for_category
from .chat_citations import (
    NO_RECORD,
    CitationError,
    CitationReport,
    enforce,
    extract_citations,
    validate_answer,
)
from .chat_generators import ChatRequest, EvidenceItem, Generator, extractive_generator
from .observability import get_logger

_log = get_logger("chat_memory")

__all__ = [
    "MAX_QUESTION_CHARS",
    "NO_RECORD",
    "ChatAnswer",
    "CitationError",
    "chat_with_memory",
    "make_workspace_resolver",
]


#: Boundary cap on question length, matching the MCP recall surface.
MAX_QUESTION_CHARS = 8192

#: Boundary cap on how many blocks may be requested per turn.
MAX_LIMIT = 50

_ON_INVALID_MODES = ("raise", "reject")

_WHITESPACE = re.compile(r"\s+")

#: ``[3]`` — chain-of-note's positional marker, re-anchored to a block id.
_NOTE_MARKER = re.compile(r"\[(\d{1,3})\]")

_GROUNDING_RULES = (
    "Grounding rules (mandatory):\n"
    "* Cite the source of every sentence with its block id in double "
    "square brackets, e.g. [[D-20260301-001]].\n"
    "* Use ONLY the block ids listed in the evidence. Never invent an id.\n"
    '* If the evidence does not answer the question, reply exactly: "no record found".\n\n'
)


@dataclass(frozen=True)
class ChatAnswer:
    """Immutable result of one :func:`chat_with_memory` turn."""

    question: str
    answer: str
    citations: tuple[str, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    category: str = "single-hop"
    report: CitationReport | None = None
    grounded: bool = False
    no_record: bool = False
    rejected: bool = False
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": list(self.citations),
            "evidence": [item.to_dict() for item in self.evidence],
            "category": self.category,
            "report": self.report.to_dict() if self.report is not None else None,
            "grounded": self.grounded,
            "no_record": self.no_record,
            "rejected": self.rejected,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Boundary validation
# ---------------------------------------------------------------------------


def _validate_inputs(workspace: str, question: str, limit: int, on_invalid: str) -> None:
    """Fail fast and loudly on malformed caller input."""
    if not isinstance(workspace, str) or not workspace.strip():
        raise ValueError("workspace must be a non-empty string")
    if not os.path.isdir(workspace):
        raise ValueError(f"workspace not found: {workspace!r}")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"question must be ≤{MAX_QUESTION_CHARS} characters, got {len(question)}")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > MAX_LIMIT:
        raise ValueError(f"limit must be an int in 1..{MAX_LIMIT}, got {limit!r}")
    if on_invalid not in _ON_INVALID_MODES:
        raise ValueError(f"on_invalid must be one of {_ON_INVALID_MODES}, got {on_invalid!r}")


# ---------------------------------------------------------------------------
# Evidence + resolution
# ---------------------------------------------------------------------------


def _default_recall(workspace: str, question: str, limit: int) -> list[dict[str, Any]]:
    """Recall through the workspace's configured backend."""
    from .recall import recall as recall_engine

    return list(recall_engine(workspace, question, limit=limit))


def _to_evidence(hits: Sequence[Any]) -> tuple[EvidenceItem, ...]:
    """Normalise recall hits into immutable evidence items.

    Hits without a usable ``_id`` are dropped: an evidence item with no
    id could never be cited, so keeping it would only invite an
    ungrounded sentence.
    """
    items: list[EvidenceItem] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        block_id = hit.get("_id")
        if not isinstance(block_id, str) or not block_id.strip():
            continue
        excerpt = hit.get("excerpt") or hit.get("Statement") or hit.get("Title") or ""
        try:
            score = float(hit.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        items.append(
            EvidenceItem(
                block_id=block_id.strip(),
                excerpt=_WHITESPACE.sub(" ", str(excerpt)).strip(),
                score=score,
                source=str(hit.get("file", "") or ""),
                date=str(hit.get("Date", "") or ""),
            )
        )
    return tuple(items)


def make_workspace_resolver(workspace: str) -> Callable[[str], bool]:
    """Build a ``block_id -> bool`` predicate backed by the block store.

    Results are memoised per resolver instance so validating an answer
    with repeated citations does not re-scan the corpus. A store failure
    resolves to ``False`` (fail-closed): an id we cannot prove exists is
    treated as fabricated.
    """
    from .storage import get_block_store

    cache: dict[str, bool] = {}
    store_box: list[Any] = []

    def _store() -> Any:
        if not store_box:
            store_box.append(get_block_store(workspace))
        return store_box[0]

    def _resolve(block_id: str) -> bool:
        if not isinstance(block_id, str) or not block_id.strip():
            return False
        key = block_id.strip()
        if key in cache:
            return cache[key]
        try:
            found = _store().get_by_id(key) is not None
        except Exception as exc:  # pragma: no cover — fail-closed on store errors
            _log.warning("chat_resolver_failed", block_id=key, error=str(exc))
            found = False
        cache[key] = found
        return found

    return _resolve


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _render_facts(evidence: Sequence[EvidenceItem], max_chars: int) -> str:
    """Render evidence as citation-anchored lines, capped at *max_chars*."""
    lines: list[str] = []
    total = 0
    for item in evidence:
        prefix = f"[[{item.block_id}]]"
        if item.date:
            prefix = f"{prefix} [Block date: {item.date}]"
        line = f"{prefix} {item.excerpt}".strip()
        if total + len(line) > max_chars and lines:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def _anchor_note_markers(notes: str, evidence: Sequence[EvidenceItem]) -> str:
    """Rewrite chain-of-note ``[N]`` markers as ``[[block_id]]``.

    ``chain_of_note_pack`` cites 1-based positions into the block list it
    was given. Out-of-range markers are dropped rather than mapped to a
    neighbouring block — a wrong citation is worse than none, because the
    validator would happily resolve it.
    """

    def _swap(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if 1 <= index <= len(evidence):
            return f"[[{evidence[index - 1].block_id}]]"
        return ""

    return _NOTE_MARKER.sub(_swap, notes)


def _build_prompt(question: str, category: str, facts: str) -> str:
    """Compose the category template with the grounding rules prepended."""
    body = prompt_for_category(category, question, facts=facts)
    return _GROUNDING_RULES + body


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def chat_with_memory(
    workspace: str,
    question: str,
    *,
    generator: Generator | None = None,
    limit: int = 8,
    recall_fn: Callable[[str, str, int], Sequence[Any]] | None = None,
    resolver: Callable[[str], bool] | None = None,
    category: str | None = None,
    condenser: Callable[[str], str] | None = None,
    on_invalid: str = "raise",
    require_in_evidence: bool = False,
    max_evidence_chars: int = 4000,
) -> ChatAnswer:
    """Answer *question* from *workspace* with verified citations.

    Args:
        workspace: Path to a mind-mem workspace root. Must exist.
        question: Natural-language question, 1..``MAX_QUESTION_CHARS`` chars.
        generator: ``ChatRequest -> str`` answerer. Defaults to the
            deterministic offline
            :func:`~mind_mem.chat_generators.extractive_generator`.
            Inject a stub here in tests; inject a service adapter in
            production.
        limit: Max blocks to recall (1..``MAX_LIMIT``).
        recall_fn: ``(workspace, question, limit) -> hits``. Defaults to
            :func:`mind_mem.recall.recall`, which routes to the
            workspace's configured backend.
        resolver: ``block_id -> bool``. Defaults to
            :func:`make_workspace_resolver` over the configured block
            store.
        category: Force a question category instead of classifying.
        condenser: **Opt-in** chain-of-note condenser (``prompt -> text``).
            ``None`` (default) skips condensation entirely, leaving the
            prompt identical to the un-condensed path.
        on_invalid: ``"raise"`` (default) raises
            :class:`~mind_mem.chat_citations.CitationError` on an
            ungrounded answer; ``"reject"`` returns a ``rejected``
            :class:`ChatAnswer` carrying the no-record string.
        require_in_evidence: Also fail when a citation resolves in the
            workspace but was not among the recalled evidence. Default
            ``False``.
        max_evidence_chars: Cap on the rendered evidence block.

    Returns:
        A :class:`ChatAnswer`. ``answer`` is either a grounded response
        or the literal ``"no record found"``.

    Raises:
        ValueError: Malformed input at the boundary.
        CitationError: The answer failed the grounding contract and
            ``on_invalid="raise"``.
    """
    _validate_inputs(workspace, question, limit, on_invalid)
    asked = question.strip()

    hits = (recall_fn or _default_recall)(workspace, asked, limit)
    evidence = _to_evidence(hits or ())

    if not evidence:
        _log.info("chat_no_record", question_chars=len(asked))
        return ChatAnswer(
            question=asked,
            answer=NO_RECORD,
            category=category or classify_question_category(asked),
            report=CitationReport(ok=True),
            grounded=True,
            no_record=True,
        )

    resolved_category = category or classify_question_category(asked)
    facts = _render_facts(evidence, max_evidence_chars)
    warnings: list[str] = []

    if condenser is not None:
        from .chain_of_note import chain_of_note_pack

        notes = chain_of_note_pack(
            asked,
            [{"excerpt": item.excerpt} for item in evidence],
            condenser,
            max_blocks=len(evidence),
            max_chars=max_evidence_chars,
            # Its own fallback returns the raw, index-anchored evidence
            # render; ours is already block-id-anchored and warns, so
            # take the empty signal and handle it here.
            fallback_on_empty=False,
        )
        anchored = _anchor_note_markers(notes, evidence).strip()
        if anchored and extract_citations(anchored):
            facts = anchored
        else:
            warnings.append("chain-of-note produced no anchored notes; used raw evidence")

    prompt = _build_prompt(asked, resolved_category, facts)
    request = ChatRequest(question=asked, prompt=prompt, evidence=evidence, category=resolved_category)

    answer = (generator or extractive_generator)(request)
    if not isinstance(answer, str):
        raise TypeError(f"generator must return str, got {type(answer).__name__}")
    answer = answer.strip()

    active_resolver = resolver or make_workspace_resolver(workspace)
    report = validate_answer(
        answer,
        resolver=active_resolver,
        evidence_ids=request.evidence_ids(),
        require_in_evidence=require_in_evidence,
    )

    if not report.ok:
        _log.warning("chat_answer_rejected", reason=report.summary())
        enforce(report, on_invalid=on_invalid)
        return ChatAnswer(
            question=asked,
            answer=NO_RECORD,
            evidence=evidence,
            category=resolved_category,
            report=report,
            grounded=False,
            no_record=True,
            rejected=True,
            warnings=tuple(warnings) + (report.summary(),),
        )

    is_no_record = not report.citations
    _log.info("chat_answered", citations=len(report.citations), evidence=len(evidence))
    return ChatAnswer(
        question=asked,
        answer=answer,
        citations=report.citations,
        evidence=evidence,
        category=resolved_category,
        report=report,
        grounded=True,
        no_record=is_no_record,
        warnings=tuple(warnings),
    )
