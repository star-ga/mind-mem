"""Citation extraction + validation for the conversational chat layer.

Every answer produced by :func:`mind_mem.chat_memory.chat_with_memory`
must be *grounded*: each claim sentence carries at least one
``[[block_id]]`` citation, and every cited id must resolve to a real
block in the workspace. This module owns that contract and nothing
else — extraction, sentence segmentation, and the verdict object.

Contract
--------
1. A **citation** is the literal token ``[[<block_id>]]``. Ids are
   matched non-greedily and may not themselves contain brackets.
2. A **claim sentence** is any sentence-sized fragment of the answer
   that carries at least one alphanumeric character and is not the
   literal no-record marker. Bullet/heading decoration is stripped
   before that test but does NOT exempt the line: a heading with text
   in it is a claim and must be cited. Only fenced code blocks, blank
   lines and pure-decoration fragments are exempt. Fail-closed by
   design — a heading is the easiest place to smuggle an unsupported
   assertion past a laxer splitter.
3. An answer **passes** when (a) every claim sentence carries >= 1
   citation and (b) every citation resolves through the caller's
   ``resolver`` predicate.
4. A failure is reported as a :class:`CitationReport` with
   ``ok=False``; callers choose whether to raise
   :class:`CitationError` or reject the answer.

The resolver is injected rather than imported so this module stays
free of storage concerns and is trivially testable against a set
literal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "CITATION_PATTERN",
    "NO_RECORD",
    "CitationError",
    "CitationReport",
    "enforce",
    "extract_citations",
    "sequence_ids",
    "split_claim_sentences",
    "strip_citations",
    "validate_answer",
]


#: The literal answer returned when recall produced nothing. Callers
#: compare against this exact string — never a fabricated answer.
NO_RECORD = "no record found"

#: ``[[block_id]]`` — ids may not contain square brackets.
CITATION_PATTERN = re.compile(r"\[\[([^\[\]]{1,128}?)\]\]")

# Sentence boundary: terminator followed by whitespace. Citations sit
# *inside* the sentence they support, so a boundary is only taken after
# the closing bracket has been consumed.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# Leading list/heading decoration stripped before the claim test.
_DECORATION = re.compile(r"^[\s>*\-•#0-9.)]+")

_ALNUM = re.compile(r"[0-9A-Za-zÀ-ɏ]")

#: A chunked recall hit carries a ``.N`` suffix (``D-20260101-001.2``).
#: Resolution falls back to the parent id when the suffixed form is
#: unknown to the store.
_CHUNK_SUFFIX = re.compile(r"\.\d+$")


class CitationError(ValueError):
    """Raised when an answer fails the grounding contract.

    Carries the full :class:`CitationReport` so callers can log the
    exact violations rather than re-deriving them from the message.
    """

    def __init__(self, report: "CitationReport") -> None:
        self.report = report
        super().__init__(report.summary())


@dataclass(frozen=True)
class CitationReport:
    """Immutable verdict for one generated answer."""

    ok: bool
    citations: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    uncited_sentences: tuple[str, ...] = ()
    out_of_evidence: tuple[str, ...] = ()

    def summary(self) -> str:
        """One-line human-readable reason, stable across runs."""
        if self.ok:
            return f"grounded: {len(self.citations)} citation(s)"
        parts: list[str] = []
        if self.unresolved:
            parts.append("unresolvable block id(s): " + ", ".join(self.unresolved))
        if self.uncited_sentences:
            parts.append(f"{len(self.uncited_sentences)} uncited claim sentence(s)")
        if self.out_of_evidence:
            parts.append("citation(s) outside the recalled evidence: " + ", ".join(self.out_of_evidence))
        return "; ".join(parts) or "answer failed the grounding contract"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "citations": list(self.citations),
            "unresolved": list(self.unresolved),
            "uncited_sentences": list(self.uncited_sentences),
            "out_of_evidence": list(self.out_of_evidence),
            "summary": self.summary(),
        }


def extract_citations(text: str) -> tuple[str, ...]:
    """Return the ``[[block_id]]`` ids in *text*, ordered, de-duplicated.

    Non-string input yields an empty tuple rather than raising — this is
    called on generator output, which is untrusted at the boundary.
    """
    if not isinstance(text, str) or not text:
        return ()
    seen: dict[str, None] = {}
    for match in CITATION_PATTERN.finditer(text):
        block_id = match.group(1).strip()
        if block_id:
            seen.setdefault(block_id, None)
    return tuple(seen)


def strip_citations(text: str) -> str:
    """Remove citation tokens, collapsing the whitespace they leave."""
    if not isinstance(text, str) or not text:
        return ""
    return re.sub(r"\s+", " ", CITATION_PATTERN.sub(" ", text)).strip()


def split_claim_sentences(answer: str) -> tuple[str, ...]:
    """Split *answer* into claim-sized fragments.

    Lines are split first (so bullet lists produce one claim per
    bullet), then each line is segmented on sentence terminators.
    Fenced code blocks, fragments with no alphanumeric content and the
    no-record marker are dropped. Leading bullet / heading decoration is
    stripped only to run the alphanumeric test — a decorated line that
    still says something is a claim and must be cited.
    """
    if not isinstance(answer, str):
        return ()
    text = answer.strip()
    if not text or _is_no_record(text):
        return ()

    claims: list[str] = []
    in_fence = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line:
            continue
        for fragment in _SENTENCE_BOUNDARY.split(line):
            candidate = fragment.strip()
            if not candidate:
                continue
            body = _DECORATION.sub("", candidate).strip()
            if not body or not _ALNUM.search(strip_citations(body)):
                continue
            if _is_no_record(body):
                continue
            claims.append(candidate)
    return tuple(claims)


def _is_no_record(text: str) -> bool:
    """True when *text* is the literal no-record marker (punctuation-tolerant)."""
    normalised = re.sub(r"\s+", " ", text.strip().strip(".!? ").lower())
    return normalised == NO_RECORD


def _resolve_with_parent(block_id: str, resolver: Callable[[str], bool]) -> bool:
    """Resolve *block_id*, falling back to its un-chunked parent id."""
    if resolver(block_id):
        return True
    parent = _CHUNK_SUFFIX.sub("", block_id)
    return parent != block_id and resolver(parent)


def validate_answer(
    answer: str,
    *,
    resolver: Callable[[str], bool],
    evidence_ids: Iterable[str] = (),
    require_in_evidence: bool = False,
) -> CitationReport:
    """Check *answer* against the grounding contract.

    Args:
        answer: Generated answer text.
        resolver: ``block_id -> bool`` predicate. Must return True only
            for ids that exist in the workspace being answered from.
        evidence_ids: The ids that were actually handed to the
            generator. Citations outside this set are recorded as
            ``out_of_evidence``.
        require_in_evidence: Promote ``out_of_evidence`` from a recorded
            observation to a hard failure. Default ``False`` keeps the
            contract exactly as the acceptance gate states it —
            resolution in the workspace is the hard requirement.

    Returns:
        A :class:`CitationReport`. Never raises on generator output;
        malformed / empty answers simply fail with ``ok=False``.
    """
    if not callable(resolver):
        raise TypeError("resolver must be callable")

    if not isinstance(answer, str) or not answer.strip():
        return CitationReport(ok=False, uncited_sentences=("<empty answer>",))

    if _is_no_record(answer):
        # The abstention path is grounded by construction: it asserts
        # nothing, so it needs no citation.
        return CitationReport(ok=True)

    citations = extract_citations(answer)
    known = {str(bid) for bid in evidence_ids}

    unresolved = tuple(bid for bid in citations if not _resolve_with_parent(bid, resolver))
    out_of_evidence = tuple(bid for bid in citations if known and bid not in known)

    uncited = tuple(sentence for sentence in split_claim_sentences(answer) if not extract_citations(sentence))

    ok = not unresolved and not uncited
    if require_in_evidence and out_of_evidence:
        ok = False

    return CitationReport(
        ok=ok,
        citations=citations,
        unresolved=unresolved,
        uncited_sentences=uncited,
        out_of_evidence=out_of_evidence,
    )


def enforce(report: CitationReport, *, on_invalid: str) -> None:
    """Raise :class:`CitationError` when *report* failed and mode is ``raise``."""
    if report.ok or on_invalid != "raise":
        return
    raise CitationError(report)


def sequence_ids(evidence: Sequence[Any]) -> tuple[str, ...]:
    """Best-effort id extraction from a sequence of evidence-like objects."""
    ids: list[str] = []
    for item in evidence:
        block_id = getattr(item, "block_id", None)
        if block_id is None and isinstance(item, dict):
            block_id = item.get("_id")
        if isinstance(block_id, str) and block_id:
            ids.append(block_id)
    return tuple(ids)
