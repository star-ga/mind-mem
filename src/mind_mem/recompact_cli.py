"""Safe CLI orchestration for proposal-only memory recompaction.

The fixed-point implementation lives in :mod:`mind_mem.recompaction`.  This
module supplies the production boundary: similarity is discovered through the
existing ``find_similar`` tool, bodies are reloaded through the admitted block
store, and a result is either printed as a dry-run proposal or staged through
``propose_update``.  Nothing is auto-applied and the default compressor is the
local Echo control.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

from .recompaction import RecompactionConfig, recompact_cluster

_MAX_COMPRESSED_CHARS = 10_000
_BLOCK_ID = re.compile(r"^[A-Z]+-[a-zA-Z0-9_.-]+$")


class RecompactError(ValueError):
    """A caller or corpus condition that prevents a safe proposal."""


def _bounded_compressor(compressor: Callable[[str, list[dict[str, Any]]], str]) -> Callable[[str, list[dict[str, Any]]], str]:
    """Reject malformed or unbounded plugin output before fixed-point use."""

    def checked(text: str, blocks: list[dict[str, Any]]) -> str:
        value = compressor(text, blocks)
        if not isinstance(value, str):
            raise RecompactError("compressor must return a string")
        if len(value) > _MAX_COMPRESSED_CHARS:
            raise RecompactError(f"compressor output exceeds {_MAX_COMPRESSED_CHARS} characters")
        if any(ord(char) < 32 and char not in "\n\t\r" for char in value):
            raise RecompactError("compressor output contains control characters")
        return value

    return checked


def _similar_ids(raw: str | dict[str, Any]) -> list[str]:
    try:
        payload: Any = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError) as exc:
        raise RecompactError(f"find_similar returned malformed JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RecompactError("find_similar returned a non-object result")
    if payload.get("error"):
        raise RecompactError(f"find_similar failed: {payload['error']}")
    similar = payload.get("similar")
    if not isinstance(similar, list):
        raise RecompactError("find_similar returned no similar block list")
    ids: list[str] = []
    for item in similar:
        block_id = item if isinstance(item, str) else item.get("block_id") if isinstance(item, dict) else None
        if isinstance(block_id, str) and block_id not in ids:
            ids.append(block_id)
    return ids


def _source_record(block: dict[str, Any]) -> dict[str, Any]:
    """Return non-content source coordinates for a proposal receipt."""

    return {
        "block_id": str(block.get("_id", "")),
        "source_file": str(block.get("_source_file", "")),
        "line": block.get("_line", block.get("line", 0)),
        "status": str(block.get("Status", block.get("status", ""))),
    }


def resolve_similarity_cluster(
    workspace: str,
    block_id: str,
    *,
    limit: int = 5,
    finder: Callable[[str, int], str | dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Resolve a target and its admitted co-occurrence neighbours.

    ``finder`` is an injection seam for deterministic tests. The default is
    the real MCP ``find_similar`` implementation, scoped with its existing
    workspace context. Every returned ID must still be present in the active
    corpus, so an old index cannot cause withheld content to enter a proposal.
    """

    if not isinstance(block_id, str) or not _BLOCK_ID.fullmatch(block_id):
        raise RecompactError(f"invalid block_id: {block_id!r}")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50:
        raise RecompactError("limit must be an integer in [1, 50]")
    from .storage import iter_active_blocks

    blocks = {str(block.get("_id", "")): block for block in iter_active_blocks(workspace)}
    target = blocks.get(block_id)
    if target is None:
        raise RecompactError(f"active block not found: {block_id}")

    if finder is None:
        from .mcp.infra.workspace import use_workspace
        from .mcp.tools.recall import find_similar

        scope = use_workspace(workspace)
        finder = find_similar
    else:
        scope = nullcontext()
    with scope:
        neighbour_ids = _similar_ids(finder(block_id, limit))

    missing = [candidate for candidate in neighbour_ids if candidate not in blocks]
    if missing:
        raise RecompactError(f"similarity index returned blocks outside the active corpus: {', '.join(missing[:5])}")
    cluster = [target] + [blocks[candidate] for candidate in neighbour_ids if candidate != block_id]
    if len(cluster) < 2:
        raise RecompactError("at least one active similar block is required; no proposal was created")
    return cluster


def make_recompact_proposal(
    workspace: str,
    block_id: str,
    *,
    compressor: Callable[[str, list[dict[str, Any]]], str],
    config: RecompactionConfig | None = None,
    limit: int = 5,
    dream: bool = False,
    finder: Callable[[str, int], str | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a bounded, provenance-bearing proposal without writing it."""

    cluster = resolve_similarity_cluster(workspace, block_id, limit=limit, finder=finder)
    result = recompact_cluster(cluster, compressor=_bounded_compressor(compressor), config=config)
    return {
        "status": "proposal" if result.changed else "no_change",
        "mode": "dream" if dream else "recompact",
        "requires_approval": True,
        "semantic_verification": "not_established",
        "source_ids": list(result.source_ids),
        "sources": [_source_record(block) for block in cluster],
        "input_digest": result.input_digest,
        "output_digest": result.output_digest,
        "converged": result.converged,
        "iterations": result.iterations,
        "changed": result.changed,
        "text": result.text,
        "trajectory_digests": [hashlib.sha256(item.encode("utf-8")).hexdigest() for item in result.trajectory],
        "write": "not_written",
    }


def stage_recompact_proposal(workspace: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Stage a result through the existing governed proposal service."""

    if payload.get("status") != "proposal":
        return {**payload, "write": "skipped_no_change"}
    source_ids = [str(item) for item in payload["source_ids"]]
    rationale = (
        f"{payload['mode']} fixed-point proposal from {len(source_ids)} active blocks; "
        f"input_digest={payload['input_digest']}; source_ids={','.join(source_ids)}. "
        "Semantic verification is not established; operator review is required."
    )
    tags = ",".join(["recompaction", *[f"source-{item}" for item in source_ids[:14]]])
    from .mcp.infra.workspace import use_workspace
    from .mcp.tools.governance import propose_update

    with use_workspace(workspace):
        raw = propose_update(
            "task",
            str(payload["text"]),
            rationale=rationale,
            tags=tags,
            confidence="medium",
            actor_id="mm-recompact",
            actor_role="maintenance",
            tool_id="mm recompact",
            purpose="stage a fixed-point memory recompaction for operator review",
            content_source="agent",
        )
    try:
        response = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RecompactError(f"proposal service returned malformed JSON: {exc}") from exc
    if not isinstance(response, dict):
        raise RecompactError("proposal service returned a non-object result")
    if response.get("status") != "proposed":
        return {**payload, "write": "refused", "proposal_response": response}
    return {**payload, "write": "staged", "proposal_response": response}


def compressor_for(name: str, model: str | None = None) -> Callable[[str, list[dict[str, Any]]], str]:
    """Select the local deterministic control or an explicit local model."""

    if name == "echo":
        from .compressors import EchoCompressor

        return EchoCompressor()
    if name == "ollama":
        if not model:
            raise RecompactError("--model is required with --compressor ollama")
        from .compressors import OllamaCompressor

        return OllamaCompressor(model=model)
    raise RecompactError(f"unsupported compressor: {name}")


__all__ = [
    "RecompactError",
    "compressor_for",
    "make_recompact_proposal",
    "resolve_similarity_cluster",
    "stage_recompact_proposal",
]
