# Copyright 2026 STARGA, Inc.
"""The evidence panel ``mm review`` renders next to each proposal.

An operator approving a proposal has to answer one question: *does this
proposal actually reflect the block it claims to change?* Answering it
today means leaving the queue, running ``mm inspect``, then
``verify_chain``, then ``stale_blocks`` — three tools and a context
switch per proposal, which is why nobody does it.

This module gathers the four things that answer it — the target block's
own text, its provenance edges, governance chain validity, and whether
the target is flagged stale — into one read-only panel.

Every panel degrades to a stated note rather than an exception. An
evidence panel that can crash a queue listing is a panel that gets
deleted, and then approvals happen blind again.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from .review_queue import ReviewItem

__all__ = ["EvidencePanel", "MAX_EXCERPT_CHARS", "gather"]

#: Cap on the quoted target block. Long enough to check the claim,
#: short enough that thirty of them still fit in a review session.
MAX_EXCERPT_CHARS = 600

#: Depth for the rendered provenance chain, matching ``mm inspect``.
CHAIN_DEPTH = 3

#: Edge types that mean "these two blocks disagree".
CONFLICT_EDGES: frozenset[str] = frozenset({"contradicts", "supersedes"})


@dataclass(frozen=True)
class EvidencePanel:
    """Everything needed to check a proposal against its target."""

    proposal_id: str
    target_block: str
    target_excerpt: str = ""
    dependencies: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    causal_chains: tuple[tuple[str, ...], ...] = ()
    chain_valid: bool | None = None
    chain_summary: str = ""
    stale: bool = False
    stale_reason: str = ""
    notes: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "target_block": self.target_block,
            "target_excerpt": self.target_excerpt,
            "dependencies": list(self.dependencies),
            "conflicts": list(self.conflicts),
            "causal_chains": [list(chain) for chain in self.causal_chains],
            "chain_valid": self.chain_valid,
            "chain_summary": self.chain_summary,
            "stale": self.stale,
            "stale_reason": self.stale_reason,
            "notes": list(self.notes),
        }


def gather(workspace: str, item: ReviewItem) -> EvidencePanel:
    """Assemble the evidence panel for *item*. Never raises."""
    root = os.path.realpath(workspace)
    notes: list[str] = []

    excerpt = _excerpt(root, item.target_block, notes)
    deps, conflicts, chains = _provenance(root, item.target_block, notes)
    chain_valid, chain_summary = _chain(root)
    stale, stale_reason = _staleness(root, item.target_block, notes)

    return EvidencePanel(
        proposal_id=item.proposal_id,
        target_block=item.target_block,
        target_excerpt=excerpt,
        dependencies=deps,
        conflicts=conflicts,
        causal_chains=chains,
        chain_valid=chain_valid,
        chain_summary=chain_summary,
        stale=stale,
        stale_reason=stale_reason,
        notes=tuple(notes),
    )


def _excerpt(root: str, block_id: str, notes: list[str]) -> str:
    """The target block's own text, bounded, whitespace preserved."""
    if not block_id:
        return ""
    try:
        from .block_store import MarkdownBlockStore

        block = MarkdownBlockStore(root).get_by_id(block_id)
    except Exception as exc:  # noqa: BLE001 — a missing target is a note, not a crash
        notes.append(f"target block unreadable: {type(exc).__name__}: {exc}")
        return ""
    if not block:
        notes.append(f"target block {block_id} not found in the corpus")
        return ""
    text = "\n".join(f"{key}: {value}" for key, value in _display_fields(block))
    return text[:MAX_EXCERPT_CHARS]


def _display_fields(block: Any) -> list[tuple[str, Any]]:
    """Public fields of a parsed block, parser internals dropped."""
    if not isinstance(block, dict):
        return [("block", str(block))]
    return [(key, value) for key, value in block.items() if not key.startswith("_")]


def _provenance(root: str, block_id: str, notes: list[str]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Dependency edges, conflict edges and causal chains for *block_id*."""
    if not block_id:
        return (), (), ()
    try:
        from .causal_graph import CausalGraph

        graph = CausalGraph(root)
        outgoing = [edge.to_dict() for edge in graph.dependencies(block_id)]
        incoming = [edge.to_dict() for edge in graph.dependents(block_id)]
        chains = graph.causal_chain(block_id, max_depth=CHAIN_DEPTH)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"provenance unavailable: {type(exc).__name__}: {exc}")
        return (), (), ()

    deps = tuple(f"{edge['target_id']} [{edge['edge_type']}]" for edge in outgoing)
    conflicts = tuple(
        f"{edge.get('source_id', '?')} -> {edge.get('target_id', '?')} [{edge['edge_type']}]"
        for edge in outgoing + incoming
        if edge.get("edge_type") in CONFLICT_EDGES
    )
    return deps, conflicts, tuple(tuple(chain) for chain in chains)


def _chain(root: str) -> tuple[bool | None, str]:
    """Governance hash-chain and evidence-chain validity for the workspace."""
    try:
        from .mcp.infra.workspace import use_workspace
        from .mcp.tools import audit

        with use_workspace(root):
            payload = json.loads(audit.verify_chain())
    except Exception as exc:  # noqa: BLE001
        return None, f"chain verification unavailable: {type(exc).__name__}: {exc}"
    if "error" in payload:
        return None, str(payload["error"])
    hash_chain = payload.get("hash_chain", {})
    evidence = payload.get("evidence_chain", {})
    summary = (
        f"hash_chain valid={hash_chain.get('valid')} length={hash_chain.get('length')} "
        f"broken_at={hash_chain.get('broken_at')}; evidence valid={evidence.get('valid')}"
    )
    return bool(payload.get("valid")), summary


def _staleness(root: str, block_id: str, notes: list[str]) -> tuple[bool, str]:
    """Whether the proposal's target is flagged stale, and why."""
    if not block_id:
        return False, ""
    try:
        from .causal_graph import CausalGraph

        flags = CausalGraph(root).get_stale_blocks()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"staleness unavailable: {type(exc).__name__}: {exc}")
        return False, ""
    for flag in flags:
        if flag.get("block_id") == block_id:
            return True, str(flag.get("reason", "") or "flagged stale, no reason recorded")
    return False, ""
