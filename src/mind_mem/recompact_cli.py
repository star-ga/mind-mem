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
import unicodedata
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .recompaction import RecompactionConfig, cluster_digest, recompact_cluster

_MAX_COMPRESSED_CHARS = 10_000
_MAX_PROPOSAL_CHARS = 500  # capture.append_signals' governed statement limit
_MAX_SOURCE_BYTES = 8 * 1024 * 1024
_BLOCK_ID = re.compile(r"^[A-Z]+-[a-zA-Z0-9_.-]+$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


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
        if any(unicodedata.category(char).startswith("C") and char not in "\n\t\r" for char in value):
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
        if not isinstance(block_id, str) or not _BLOCK_ID.fullmatch(block_id):
            raise RecompactError("find_similar returned a malformed block id")
        if block_id in ids:
            raise RecompactError(f"find_similar returned duplicate block id: {block_id}")
        ids.append(block_id)
    return ids


def _source_record(workspace: str, block: dict[str, Any]) -> dict[str, Any]:
    """Return non-content source coordinates for a proposal receipt."""

    block_id = block.get("_id")
    source_file = block.get("_source_file")
    if not isinstance(block_id, str) or not _BLOCK_ID.fullmatch(block_id):
        raise RecompactError("active corpus contains a malformed block id")
    if not isinstance(source_file, str) or not source_file or Path(source_file).is_absolute():
        raise RecompactError(f"block {block_id} has no workspace-relative source file")
    root = Path(workspace).resolve()
    path = (root / source_file).resolve()
    if path != root and root not in path.parents:
        raise RecompactError(f"block {block_id} source escapes the workspace")
    try:
        stat = path.stat()
    except OSError as exc:
        raise RecompactError(f"block {block_id} source cannot be read: {exc}") from exc
    if not path.is_file():
        raise RecompactError(f"block {block_id} source is not a regular file")
    if stat.st_size > _MAX_SOURCE_BYTES:
        raise RecompactError(f"block {block_id} source exceeds {_MAX_SOURCE_BYTES} bytes")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            remaining = stat.st_size
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise RecompactError(f"block {block_id} source changed while being read")
                digest.update(chunk)
                remaining -= len(chunk)
            if source.read(1):
                raise RecompactError(f"block {block_id} source changed while being read")
    except OSError as exc:
        raise RecompactError(f"block {block_id} source cannot be read: {exc}") from exc
    return {
        "block_id": block_id,
        "source_file": source_file,
        "line": block.get("_line", block.get("line", 0)),
        "status": str(block.get("Status", block.get("status", ""))),
        "source_sha256": digest.hexdigest(),
        "source_size": stat.st_size,
    }


def _active_cluster(workspace: str, source_ids: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reload source blocks and source bytes for a proposal identity check."""

    from .storage import iter_active_blocks

    active: dict[str, dict[str, Any]] = {}
    for block in iter_active_blocks(workspace):
        block_id = block.get("_id")
        if not isinstance(block_id, str) or not _BLOCK_ID.fullmatch(block_id):
            raise RecompactError("active corpus contains a malformed block id")
        if block_id in active:
            raise RecompactError(f"active corpus contains duplicate block id: {block_id}")
        active[block_id] = block
    try:
        cluster = [active[item] for item in source_ids]
    except KeyError as exc:
        raise RecompactError(f"source block disappeared from the active corpus: {exc.args[0]}") from None
    return cluster, [_source_record(workspace, block) for block in cluster]


def _validate_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise RecompactError(f"proposal {field} must be a lowercase SHA-256 hex digest")
    return value


def _verify_payload_current(workspace: str, payload: dict[str, Any]) -> None:
    """Reject forged/stale payloads immediately before a governed stage."""

    source_ids = payload.get("source_ids")
    if not isinstance(source_ids, list) or not source_ids or any(not isinstance(item, str) for item in source_ids):
        raise RecompactError("proposal source_ids must be a non-empty list of strings")
    if len(set(source_ids)) != len(source_ids):
        raise RecompactError("proposal source_ids must not contain duplicates")
    if any(not _BLOCK_ID.fullmatch(item) for item in source_ids):
        raise RecompactError("proposal source_ids contains a malformed block id")
    input_digest = _validate_digest(payload.get("input_digest"), "input_digest")
    output_digest = _validate_digest(payload.get("output_digest"), "output_digest")
    text = payload.get("text")
    if not isinstance(text, str) or len(text) > _MAX_COMPRESSED_CHARS:
        raise RecompactError("proposal text is missing or exceeds the compressor bound")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != output_digest:
        raise RecompactError("proposal output_digest does not match proposal text")
    if any(unicodedata.category(char).startswith("C") and char not in "\n\t\r" for char in text):
        raise RecompactError("proposal text contains control characters")
    cluster, sources = _active_cluster(workspace, source_ids)
    if cluster_digest(cluster) != input_digest:
        raise RecompactError("active source blocks changed since the proposal was computed")
    if payload.get("sources") != sources:
        raise RecompactError("proposal source identity is stale or forged")


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

    blocks: dict[str, dict[str, Any]] = {}
    for block in iter_active_blocks(workspace):
        candidate_id = block.get("_id")
        if not isinstance(candidate_id, str) or not _BLOCK_ID.fullmatch(candidate_id):
            raise RecompactError("active corpus contains a malformed block id")
        if candidate_id in blocks:
            raise RecompactError(f"active corpus contains duplicate block id: {candidate_id}")
        blocks[candidate_id] = block
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
        try:
            raw_result = finder(block_id, limit)
        except Exception as exc:  # noqa: BLE001 - finder failures are caller-visible refusals
            raise RecompactError(f"find_similar failed: {type(exc).__name__}: {exc}") from exc
        neighbour_ids = _similar_ids(raw_result)

    if len(neighbour_ids) > limit:
        raise RecompactError(f"find_similar returned {len(neighbour_ids)} blocks for limit={limit}")

    missing = [candidate for candidate in neighbour_ids if candidate not in blocks]
    if missing:
        raise RecompactError(f"similarity index returned blocks outside the active corpus: {', '.join(missing[:5])}")
    if block_id in neighbour_ids:
        raise RecompactError("find_similar returned the target block as a neighbour")
    cluster = [target] + [blocks[candidate] for candidate in neighbour_ids]
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
    source_ids = [str(block["_id"]) for block in cluster]
    source_records = [_source_record(workspace, block) for block in cluster]
    input_digest = cluster_digest(cluster)
    result = recompact_cluster([dict(block) for block in cluster], compressor=_bounded_compressor(compressor), config=config)
    # A model/plugin call is allowed to take time. The active files and rows
    # must still be the exact bytes from which the proposal was derived when
    # it returns; otherwise this is stale evidence, not a proposal.
    current_cluster, current_sources = _active_cluster(workspace, source_ids)
    if cluster_digest(current_cluster) != input_digest or current_sources != source_records:
        raise RecompactError("active source changed while recompaction was running")
    if result.input_digest != input_digest:
        raise RecompactError("recompaction input digest disagrees with the bound source snapshot")
    return {
        "status": "proposal" if result.changed else "no_change",
        "mode": "dream" if dream else "recompact",
        "requires_approval": True,
        "semantic_verification": "not_established",
        "source_ids": list(result.source_ids),
        "sources": source_records,
        "input_digest": result.input_digest,
        "output_digest": result.output_digest,
        "converged": result.converged,
        "iterations": result.iterations,
        "changed": result.changed,
        "text": result.text,
        "trajectory_digests": [hashlib.sha256(item.encode("utf-8")).hexdigest() for item in result.trajectory],
        "write": "not_written",
    }


def stage_recompact_proposal(
    workspace: str,
    payload: dict[str, Any],
    *,
    provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Stage a result through the existing governed proposal service."""

    if not isinstance(payload, dict):
        raise RecompactError("proposal payload must be an object")
    if payload.get("status") != "proposal":
        return {**payload, "write": "skipped_no_change"}
    text = payload.get("text")
    if not isinstance(text, str) or len(text) > _MAX_PROPOSAL_CHARS:
        return {
            **payload,
            "write": "refused",
            "error": f"proposal text exceeds governed statement limit of {_MAX_PROPOSAL_CHARS} characters",
        }
    _verify_payload_current(workspace, payload)
    source_ids = [str(item) for item in payload["source_ids"]]
    rationale = (
        f"{payload['mode']} fixed-point proposal from {len(source_ids)} active blocks; "
        f"input_digest={payload['input_digest']}; source_ids={','.join(source_ids)}. "
        "Semantic verification is not established; operator review is required."
    )
    tags = ",".join(["recompaction", *[f"source-{item}" for item in source_ids[:14]]])
    from .mcp.infra.workspace import use_workspace
    from .mcp.tools.governance import propose_update

    fields = {} if provenance is None else dict(provenance)
    allowed = {"actor_id", "actor_role", "session_id", "tool_id", "purpose"}
    unknown = sorted(set(fields) - allowed)
    if unknown:
        raise RecompactError(f"unsupported provenance fields: {', '.join(unknown)}")
    if any(not isinstance(value, str) for value in fields.values()):
        raise RecompactError("provenance values must be strings supplied by the caller")
    with use_workspace(workspace):
        raw = propose_update("task", text, rationale=rationale, tags=tags, confidence="medium", **fields)
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
