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
2. **Every cited id resolves in the workspace and recalled evidence by default.**
   Ids are resolved through the configured block store; a fabricated or
   out-of-evidence id cannot survive the default strict mode. Unresolvable
   ids either raise :class:`~mind_mem.chat_citations.CitationError`
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

import json
import os
import re
import sqlite3
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
from .semantic_capability import (
    SEMANTIC_VERIFICATION_NOT_ESTABLISHED,
    semantic_entailment_verification_available,
)

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
    semantic_required: bool = False
    # The ranked recall evidence receipt, when the default serving entry
    # produced one.  This is deliberately separate from the answer: the
    # generator's prose and the canonicalized evidence projection are not
    # sealed by a ranked-recall attestation.
    attestation: dict[str, Any] | None = None
    attestation_scope: str | None = None
    # Optional graph-grounded projection.  The graph edge ids and their
    # provenance are carried here; answer citations remain block ids because
    # this is the Group-B chat contract.  Structural citation membership is
    # not semantic entailment.
    graph_evidence: dict[str, Any] | None = None

    @property
    def semantic_verification(self) -> str:
        """The runtime-derived semantic status; generators cannot set it."""
        return SEMANTIC_VERIFICATION_NOT_ESTABLISHED

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
            # This is derived by the serving layer, never copied from a
            # generator or caller-supplied payload.
            "semantic_verification": SEMANTIC_VERIFICATION_NOT_ESTABLISHED,
            "semantic_required": self.semantic_required,
            "attestation": self.attestation,
            "attestation_scope": self.attestation_scope,
        }
        # Keep the historical default JSON shape byte/field compatible;
        # graph metadata exists only when the caller opts into graph_seed.
        if self.graph_evidence is not None:
            payload["graph_evidence"] = self.graph_evidence
        return payload


# ---------------------------------------------------------------------------
# Boundary validation
# ---------------------------------------------------------------------------


def _validate_inputs(
    workspace: str,
    question: str,
    limit: int,
    on_invalid: str,
    semantic_required: bool,
) -> None:
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
    if not isinstance(semantic_required, bool):
        raise ValueError(f"semantic_required must be a bool, got {semantic_required!r}")


# ---------------------------------------------------------------------------
# Evidence + resolution
# ---------------------------------------------------------------------------


def _default_recall(
    workspace: str,
    question: str,
    limit: int,
    agent_id: str | None = None,
) -> Sequence[dict[str, Any]]:
    """Recall through the workspace's configured backend."""
    from .recall import recall as recall_engine

    # Keep the ServedResults carrier intact.  Converting it to ``list`` here
    # used to discard the ranked recall attestation before chat could expose
    # it.  The caller still treats this as a read-only Sequence.
    return recall_engine(workspace, question, limit=limit, agent_id=agent_id)


def _unproven_attestation(reason: str) -> dict[str, Any]:
    """Return the dependency-free marker used when ranked proof is absent."""
    return {
        "served_seq": None,
        "served_row_hash": None,
        "served_proof": "unproven",
        "ledger_error": reason,
    }


def _ranked_attestation(
    hits: Sequence[Any],
) -> tuple[dict[str, Any], str]:
    """Accept only a coherent default serving carrier; never trust extensions.

    Chat canonicalizes evidence fields before generation, so the receipt's
    scope is the ranked recall list.  The internal checks establish structural
    consistency for the trusted default serving boundary; they do not
    authenticate an arbitrary caller-generated payload.  Custom recall
    functions therefore remain explicitly unproven.
    """
    candidate = getattr(hits, "attestation", None)
    if not isinstance(candidate, dict):
        return _unproven_attestation("default recall returned no ranked attestation"), "unproven"

    if candidate.get("served_proof") == "unproven":
        return dict(candidate), "unproven"
    if candidate.get("served_proof") != "recorded":
        return _unproven_attestation("ranked attestation has an unknown proof status"), "unproven"

    # RecallAttestation.from_dict intentionally owns only the hash-bound
    # fields.  These serving fields are the ledger join, so validate them at
    # this response boundary instead of accepting a structurally valid but
    # unjoinable ``recorded`` claim.
    served_seq = candidate.get("served_seq")
    served_row_hash = candidate.get("served_row_hash")
    if not isinstance(served_seq, int) or isinstance(served_seq, bool) or served_seq < 0:
        return _unproven_attestation("recorded ranked attestation has no valid served sequence"), "unproven"
    if not isinstance(served_row_hash, str) or re.fullmatch(r"[0-9a-f]{64}", served_row_hash) is None:
        return _unproven_attestation("recorded ranked attestation has no valid served row hash"), "unproven"

    ids: list[str] = []
    for hit in hits:
        if not isinstance(hit, dict) or not isinstance(hit.get("_id"), str) or not hit["_id"]:
            return _unproven_attestation("ranked attestation cannot bind malformed evidence ids"), "unproven"
        ids.append(hit["_id"])

    try:
        from .recall_attestation import RecallAttestation
        from .recall_digests import served_set_digest

        parsed = RecallAttestation.from_dict(candidate)
        if not parsed.is_internally_consistent():
            raise ValueError("ranked attestation hash is inconsistent")
        if parsed.result_count != len(ids) or parsed.results_digest != served_set_digest(ids):
            raise ValueError("ranked attestation does not bind the returned evidence ids")
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        return _unproven_attestation(f"ranked attestation refused: {exc}"), "unproven"
    return dict(candidate), "ranked_recall_evidence"


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


def _servable_ids_for_agent(workspace: str, agent_id: str | None) -> set[str] | None:
    blocks = _admitted_blocks_for_agent(workspace, agent_id)
    return None if blocks is None else set(blocks)


def _admitted_blocks_for_agent(workspace: str, agent_id: str | None) -> dict[str, dict[str, Any]] | None:
    """Resolve the same live namespace partition used by MCP retrieval."""
    if not agent_id:
        return None
    try:
        from .namespace_retrieval import admitted_namespace_blocks

        return admitted_namespace_blocks(workspace, agent_id) or {}
    except Exception as exc:  # pragma: no cover - fail closed for bound callers
        _log.warning("chat_namespace_resolution_failed", error=str(exc))
        return {}


def _admitted_graph_blocks(workspace: str, agent_id: str | None) -> dict[str, dict[str, Any]]:
    """Return the canonical admitted source projection for graph reads.

    Bound callers use the namespace resolver, which carries principal ACL and
    source identity. Operator/unbound callers still go through the configured
    backend and the ordinary admission gate; the graph must never use the
    raw markdown loader as a second, source-less authority.
    """
    bound = _admitted_blocks_for_agent(workspace, agent_id)
    if bound is not None:
        return bound
    from .admissibility import admit_expansion_corpus
    from .storage import _load_workspace_config, iter_blocks

    config = _load_workspace_config(workspace, quiet=True)
    admitted = admit_expansion_corpus(
        iter_blocks(workspace, config=config, active_only=False),
        workspace=workspace,
    )
    result: dict[str, dict[str, Any]] = {}
    for block in admitted:
        block_id = block.get("_id")
        source = block.get("_source_file") or block.get("_source") or block.get("file")
        if isinstance(block_id, str) and block_id and isinstance(source, str) and source:
            result[block_id] = dict(block)
    return result


def make_workspace_resolver(
    workspace: str,
    agent_id: str | None = None,
    *,
    servable_ids: set[str] | None = None,
) -> Callable[[str], bool]:
    """Build a ``block_id -> bool`` predicate backed by the block store.

    Results are memoised per resolver instance so validating an answer
    with repeated citations does not re-scan the corpus. A store failure
    resolves to ``False`` (fail-closed): an id we cannot prove exists is
    treated as fabricated.
    """
    from .admission import admit_read_one
    from .storage import get_block_store

    cache: dict[str, bool] = {}
    store_box: list[Any] = []
    allowed_ids = _servable_ids_for_agent(workspace, agent_id) if servable_ids is None else servable_ids

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
        if allowed_ids is not None and key not in allowed_ids:
            cache[key] = False
            return False
        if allowed_ids is not None:
            # The live namespace resolver already proved this ID's source,
            # admission state, and ACL. The generic block store may omit
            # shared/agent Markdown roots, so asking it again would turn a
            # valid namespace-bound citation into a false rejection.
            cache[key] = True
            return True
        try:
            block = _store().get_by_id(key)
            found = bool(admit_read_one(block, workspace=workspace, surface="chat"))
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


def _graph_evidence_payload(context: Any, *, error: str | None = None) -> dict[str, Any]:
    """Serialize the optional graph projection without changing chat text.

    Edge ids are evidence metadata, not ``chat_citations`` ids.  The latter
    must remain source block ids so the existing workspace resolver can prove
    each answer citation against the admitted corpus.
    """
    if error is not None:
        return {
            "status": "unproven",
            "error": error,
            "citation_scope": "source_block_id",
            "semantic_verification": SEMANTIC_VERIFICATION_NOT_ESTABLISHED,
        }
    return {
        "status": "served" if context.triples else "unproven",
        "citation_scope": "source_block_id",
        "semantic_verification": SEMANTIC_VERIFICATION_NOT_ESTABLISHED,
        "context": context.as_dict(),
    }


def _graph_context_for_chat(
    workspace: str,
    seed: str,
    *,
    agent_id: str | None,
) -> tuple[Any | None, dict[str, Any]]:
    """Read a graph context and bind provenance to the live chat corpus."""
    if not isinstance(seed, str) or not seed.strip():
        raise ValueError("graph_seed must be a non-empty string")
    if len(seed) > 512:
        raise ValueError("graph_seed must be ≤512 characters")
    from .edge_grounded_answer import build_context
    from .knowledge_graph import KnowledgeGraph, default_db_path

    db_path = default_db_path(workspace)
    if not os.path.isfile(db_path):
        return None, _graph_evidence_payload(None, error="knowledge graph is unavailable")
    try:
        admitted_blocks = _admitted_graph_blocks(workspace, agent_id)
        admitted_ids = set(admitted_blocks)
    except (OSError, ValueError, TypeError) as exc:
        return None, _graph_evidence_payload(None, error=f"graph provenance unavailable: {exc}")
    try:
        graph = KnowledgeGraph.open_read_only(db_path)
        try:
            context = build_context(
                graph,
                seed.strip(),
                known_block_ids=admitted_ids,
                admitted_source_ids=admitted_ids,
            )
        finally:
            graph.close()
    except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
        return None, _graph_evidence_payload(None, error=f"knowledge graph unavailable: {exc}")
    if not context.triples:
        # An empty projected graph is a refusal, not a citable diagnostic
        # context. In particular, a private, quarantined, or deleted source
        # must not be distinguishable through graph gaps or counts.
        return None, _graph_evidence_payload(None, error="graph provenance unavailable; answer withheld")
    # Provenance is an admission boundary, not merely a diagnostic gap. The
    # graph reader computes corroboration over every matching claim, including
    # rows that are not themselves the traversed source. Require every
    # traversed and corroborating document to be admitted before exposing any
    # part of the context. Refusing the whole context prevents a private edge
    # from influencing visible confidence, ranking, or traversal.
    provenance_ids = _graph_provenance_ids(context)
    support_ids = {
        str(hit.get("_id"))
        for hit in _graph_supporting_hits(workspace, context, agent_id=agent_id)
        if isinstance(hit, dict) and isinstance(hit.get("_id"), str)
    }
    if provenance_ids - support_ids:
        return None, _graph_evidence_payload(None, error="graph provenance unavailable; answer withheld")
    return context, _graph_evidence_payload(context)


def _graph_provenance_ids(context: Any) -> set[str]:
    """Return every source document that can affect a graph context."""
    ids: set[str] = set()
    for triple in getattr(context, "triples", ()):
        source_block_id = getattr(triple, "source_block_id", None)
        if isinstance(source_block_id, str) and source_block_id.strip():
            ids.add(source_block_id)
        for block_id in getattr(triple, "corroborating_blocks", ()):
            if isinstance(block_id, str) and block_id.strip():
                ids.add(block_id)
    return ids


def _graph_support_fingerprint(hits: Sequence[dict[str, Any]]) -> str:
    """Stable snapshot of canonical graph source documents for revalidation."""
    return json.dumps(list(hits), sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))


def _graph_supporting_hits(
    workspace: str,
    context: Any,
    *,
    agent_id: str | None,
) -> list[dict[str, Any]]:
    """Load only canonical admitted documents named by served graph edges."""
    source_ids = _graph_provenance_ids(context)
    if not source_ids:
        return []
    allowed = _admitted_graph_blocks(workspace, agent_id)
    candidates: dict[str, dict[str, Any] | None] = {}
    candidates.update({key: value for key, value in allowed.items() if key in source_ids})
    hits: list[dict[str, Any]] = []
    for block_id in sorted(source_ids):
        if block_id not in candidates or candidates[block_id] is None:
            continue
        block_value = candidates[block_id]
        assert block_value is not None
        block: dict[str, Any] = block_value
        hit = dict(block)
        hit["_id"] = block_id
        hit["excerpt"] = str(block.get("excerpt") or block.get("content") or block.get("Statement") or block.get("Title") or "")
        hits.append(hit)
    return hits


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
    require_in_evidence: bool = True,
    max_evidence_chars: int = 4000,
    agent_id: str | None = None,
    semantic_required: bool = False,
    graph_seed: str | None = None,
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
        require_in_evidence: Require every citation to be among recalled
            evidence. The default is ``True``. Pass ``False`` only for the
            explicit legacy advisory mode; such an answer is never marked
            grounded when it cites outside the evidence.
        max_evidence_chars: Cap on the rendered evidence block.
        semantic_required: Require a reviewed semantic entailment verifier.
            The current runtime has no such verifier, so ``True`` returns an
            explicit abstention before recall or generation.
        graph_seed: Opt-in seed for a read-only edge-grounded projection.
            When supplied, chat may cite only the canonical source documents
            behind served edges. Missing graph/provenance is an explicit
            abstention; the default block-recall path is unchanged.

    Returns:
        A :class:`ChatAnswer`. ``answer`` is either a grounded response
        or the literal ``"no record found"``.

    Raises:
        ValueError: Malformed input at the boundary.
        CitationError: The answer failed the grounding contract and
            ``on_invalid="raise"``.
    """
    _validate_inputs(workspace, question, limit, on_invalid, semantic_required)
    asked = question.strip()

    if semantic_required and not semantic_entailment_verification_available():
        _log.info("chat_semantic_verification_unavailable")
        return ChatAnswer(
            question=asked,
            answer=NO_RECORD,
            category=category or classify_question_category(asked),
            report=None,
            grounded=False,
            no_record=True,
            rejected=True,
            warnings=("semantic verification unavailable; answer withheld",),
            semantic_required=True,
            attestation_scope="none",
        )

    graph_context = None
    graph_payload: dict[str, Any] | None = None
    graph_source_ids: set[str] = set()
    graph_support_snapshot = ""

    hits: Sequence[Any]
    if recall_fn is None:
        hits = _default_recall(workspace, asked, limit, agent_id=agent_id)
        serving_attestation, attestation_scope = _ranked_attestation(hits)
    else:
        # Injected recall functions are a deliberate low-level test/extension
        # seam with the historical three-argument contract. Public MCP calls
        # use the default path above, where the verified principal is bound.
        hits = recall_fn(workspace, asked, limit)
        # An extension function is not an attestation authority.  Even if it
        # returns an object with an ``attestation`` attribute, accepting that
        # caller-supplied value would turn chat's response field into a proof
        # of data the serving entry never recorded.
        serving_attestation = _unproven_attestation("custom recall function has no trusted serving receipt")
        attestation_scope = "unproven"

    allowed_blocks = _admitted_blocks_for_agent(workspace, agent_id)
    if allowed_blocks is not None:
        # An injected recall function is an extension seam, not an ACL
        # authority. Filter its returned evidence before any generator sees
        # excerpts, so a custom function cannot smuggle private or relabeled
        # content into the prompt even when it ignores ``agent_id``. Rebuild
        # content fields from the canonical admitted block; an extension must
        # provide the matching source coordinate to be usable on this path.
        canonical_hits: list[dict[str, Any]] = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            block_id = hit.get("_id")
            canonical = allowed_blocks.get(str(block_id))
            source = hit.get("_source_file") or hit.get("file")
            canonical_source = canonical.get("_source_file") if canonical else None
            if canonical is None or not isinstance(source, str) or source != canonical_source:
                continue
            if any(hit[field] != canonical_source for field in ("file", "_source_file") if field in hit):
                continue
            canonical_hit = {
                "_id": str(block_id),
                "_source_file": canonical_source,
                "file": canonical_source,
                "score": hit.get("score", 0.0),
            }
            canonical_hit["excerpt"] = str(canonical.get("excerpt") or canonical.get("Statement") or canonical.get("Title") or "")
            for field in ("Statement", "Title", "Date", "Status"):
                if field in canonical:
                    canonical_hit[field] = canonical[field]
            canonical_hits.append(canonical_hit)
        hits = canonical_hits

    # Recall is an injectable seam and may change admission state. Construct
    # the graph projection only after it returns, so traversal, corroboration,
    # and source support all use the same current admitted snapshot.
    if graph_seed is not None:
        graph_context, graph_payload = _graph_context_for_chat(workspace, graph_seed, agent_id=agent_id)
        if graph_context is None or not graph_context.triples:
            detail = "graph evidence unavailable; answer withheld"
            if graph_payload.get("error"):
                detail = str(graph_payload["error"])
            return ChatAnswer(
                question=asked,
                answer=NO_RECORD,
                category=category or classify_question_category(asked),
                grounded=False,
                no_record=True,
                rejected=True,
                warnings=(detail,),
                semantic_required=semantic_required,
                graph_evidence=graph_payload,
                attestation_scope="none",
            )
        graph_source_ids = {triple.source_block_id for triple in graph_context.triples}

    if graph_context is not None:
        # Replace every caller-supplied hit for a graph provenance id with
        # the canonical admitted document. An extension may not relabel a
        # valid id while changing the excerpt that the generator sees.
        support_hits = _graph_supporting_hits(workspace, graph_context, agent_id=agent_id)
        graph_support_snapshot = _graph_support_fingerprint(support_hits)
        support_by_id = {str(hit["_id"]): hit for hit in support_hits}
        base_hits = tuple(hit for hit in hits if not (isinstance(hit, dict) and str(hit.get("_id") or "") in graph_source_ids))
        hits = base_hits + tuple(support_by_id.values())
        evidence_ids = {str(hit.get("_id")) for hit in hits if isinstance(hit, dict) and isinstance(hit.get("_id"), str)}
        if not graph_source_ids.issubset(evidence_ids):
            # A graph edge without its current supporting document cannot be
            # promoted into a Group-B answer, even if the edge row exists.
            return ChatAnswer(
                question=asked,
                answer=NO_RECORD,
                category=category or classify_question_category(asked),
                grounded=False,
                no_record=True,
                rejected=True,
                warnings=("graph provenance document unavailable; answer withheld",),
                semantic_required=semantic_required,
                graph_evidence=_graph_evidence_payload(None, error="graph provenance unavailable; answer withheld"),
                attestation_scope="none",
            )

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
            semantic_required=semantic_required,
            attestation=serving_attestation,
            attestation_scope=attestation_scope,
            graph_evidence=graph_payload,
        )

    resolved_category = category or classify_question_category(asked)
    facts = _render_facts(evidence, max_evidence_chars)
    warnings: list[str] = []
    graph_facts = ""

    if graph_context is not None:
        graph_lines = [
            "Graph evidence is structural only; cite the supporting document id, not the edge id.",
            "Every graph claim must be supported by one of these source documents:",
        ]
        for triple in graph_context.ranked_triples:
            graph_lines.append(
                f"- {triple.subject} {triple.predicate} {triple.object} "
                f"(edge {triple.edge_id}; supporting document [[{triple.source_block_id}]])"
            )
        if graph_context.gaps:
            graph_lines.append("Graph gaps (the graph does not establish):")
            graph_lines.extend(f"- {gap.kind}: {gap.detail}" for gap in graph_context.gaps)
        # Keep this separate while the optional chain-of-note condenser works
        # on document evidence below; graph evidence must never disappear as
        # a side effect of an unrelated prompt projection.
        graph_facts = "\n".join(graph_lines)

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

    if graph_context is not None:
        latest_context, _latest_payload = _graph_context_for_chat(workspace, graph_seed or "", agent_id=agent_id)
        latest_support = _graph_supporting_hits(workspace, latest_context, agent_id=agent_id) if latest_context is not None else []
        if (
            latest_context is None
            or latest_context != graph_context
            or _graph_support_fingerprint(latest_support) != graph_support_snapshot
        ):
            return ChatAnswer(
                question=asked,
                answer=NO_RECORD,
                evidence=(),
                category=resolved_category,
                grounded=False,
                no_record=True,
                rejected=True,
                warnings=("graph evidence changed during condensation; answer withheld",),
                semantic_required=semantic_required,
                attestation=serving_attestation,
                attestation_scope=attestation_scope,
                graph_evidence=_graph_evidence_payload(
                    None,
                    error="graph or provenance changed during condensation",
                ),
            )

    if graph_facts:
        facts = facts + "\n\n" + graph_facts

    prompt = _build_prompt(asked, resolved_category, facts)
    request = ChatRequest(question=asked, prompt=prompt, evidence=evidence, category=resolved_category)

    answer = (generator or extractive_generator)(request)
    if not isinstance(answer, str):
        raise TypeError(f"generator must return str, got {type(answer).__name__}")
    answer = answer.strip()

    if graph_context is not None:
        latest_context, latest_payload = _graph_context_for_chat(workspace, graph_seed or "", agent_id=agent_id)
        latest_support = _graph_supporting_hits(workspace, latest_context, agent_id=agent_id) if latest_context is not None else []
        if (
            latest_context is None
            or latest_context != graph_context
            or _graph_support_fingerprint(latest_support) != graph_support_snapshot
        ):
            # The pre-generation evidence may now be revoked or otherwise
            # unavailable. Do not return the old context/evidence alongside a
            # refusal: that would disclose a snapshot the second admission
            # check has just invalidated.
            changed_payload = _graph_evidence_payload(
                None,
                error="graph or provenance changed during generation",
            )
            return ChatAnswer(
                question=asked,
                answer=NO_RECORD,
                evidence=(),
                category=resolved_category,
                grounded=False,
                no_record=True,
                rejected=True,
                warnings=("graph evidence changed during generation; answer withheld",),
                semantic_required=semantic_required,
                attestation=serving_attestation,
                attestation_scope=attestation_scope,
                graph_evidence=changed_payload,
            )

    active_resolver = resolver or make_workspace_resolver(
        workspace,
        agent_id=agent_id,
        servable_ids=None if allowed_blocks is None else set(allowed_blocks),
    )
    report = validate_answer(
        answer,
        resolver=active_resolver,
        # In graph mode the only admissible citation set is the provenance
        # document set behind the served edges. This prevents a generator
        # from answering a graph question with an unrelated recalled block.
        evidence_ids=graph_source_ids or request.evidence_ids(),
        require_in_evidence=True if graph_context is not None else require_in_evidence,
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
            semantic_required=semantic_required,
            attestation=serving_attestation,
            attestation_scope=attestation_scope,
            graph_evidence=graph_payload,
        )

    is_no_record = not report.citations
    advisory_out_of_scope = bool(report.out_of_evidence) and not require_in_evidence
    if advisory_out_of_scope:
        warnings.append("legacy advisory mode: citation outside recalled evidence; answer is ungrounded")
    _log.info("chat_answered", citations=len(report.citations), evidence=len(evidence))
    return ChatAnswer(
        question=asked,
        answer=answer,
        citations=report.citations,
        evidence=evidence,
        category=resolved_category,
        report=report,
        grounded=not advisory_out_of_scope,
        no_record=is_no_record,
        warnings=tuple(warnings),
        semantic_required=semantic_required,
        attestation=serving_attestation,
        attestation_scope=attestation_scope,
        graph_evidence=graph_payload,
    )
