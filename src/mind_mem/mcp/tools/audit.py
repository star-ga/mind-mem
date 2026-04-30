"""Audit MCP tools — Merkle proofs, hash chain + evidence chain verification.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, audit domain). Four tools:

* ``verify_merkle`` — prove a block's Merkle inclusion against the
  live tree built from the FTS index.
* ``verify_chain`` — walk both the SHA3-512 governance hash chain
  and the evidence chain and report any integrity breaks.
* ``list_evidence`` — enumerate governance evidence objects with
  optional ``block_id`` / ``action`` filters.
* ``mind_mem_verify`` — expose the standalone ``mind-mem-verify``
  CLI over MCP with path-escape guards on the ``snapshot`` arg.
"""

from __future__ import annotations

import json
import os

from ._helpers import get_logger, metrics

from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace

_log = get_logger("mcp_server")


@mcp_tool_observe
def verify_merkle(block_id: str, content_hash: str) -> str:
    """Verify a block's Merkle inclusion against the live tree.

    Builds the Merkle tree from the current block index and returns a
    JSON envelope with the proof and an ``ok`` flag indicating whether
    the caller-supplied content hash reproduces the stored root.

    Args:
        block_id: Identifier of the block to prove.
        content_hash: Claimed SHA-256 (or SHA3-512) of the block's
            canonical content. The exact digest algorithm is irrelevant
            to the tree — the caller must match whatever went in.

    Returns:
        JSON with ``ok`` (bool), ``root`` (hex), ``proof`` (list of
        sibling/direction pairs), and ``error`` when verification fails.
    """
    from mind_mem.merkle_tree import MerkleTree

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(block_id, str) or not block_id.strip():
        return json.dumps({"ok": False, "error": "block_id must be a non-empty string"})
    if not isinstance(content_hash, str) or not content_hash.strip():
        return json.dumps({"ok": False, "error": "content_hash must be a non-empty string"})

    try:
        from mind_mem.sqlite_index import merkle_leaves

        leaves = merkle_leaves(ws)
    except (ImportError, AttributeError):
        leaves = []

    if not leaves:
        return json.dumps(
            {
                "ok": False,
                "error": "no block index available — run 'mind-mem-scan' first",
            }
        )

    tree = MerkleTree()
    tree.build(leaves)
    try:
        proof = tree.get_proof(block_id)
    except KeyError:
        return json.dumps(
            {
                "ok": False,
                "error": f"block_id not in tree: {block_id!r}",
                "root": tree.root_hash,
            }
        )

    ok = tree.verify_proof(block_id, content_hash, proof, tree.root_hash)
    return json.dumps(
        {
            "ok": bool(ok),
            "root": tree.root_hash,
            "proof": proof,
            "proof_format_version": 1,
            "block_id": block_id,
            "_schema_version": "1.0",
        },
        indent=2,
    )


@mcp_tool_observe
def mind_mem_verify(snapshot: str = "") -> str:
    """Run the standalone `mind-mem-verify` CLI against the current workspace.

    Exposes the external verifier via MCP so agents can run it without
    shelling out. ``snapshot`` is optional; when set it points to a
    snapshot directory **relative to the workspace** whose manifest
    will be checked against the live chain + Merkle tree. Absolute
    paths or `..` traversal are rejected so an MCP caller cannot ask
    the verifier to read outside the workspace.
    """
    from mind_mem.verify_cli import verify_workspace

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    snap = snapshot.strip() or None
    if snap is not None:
        if len(snap) > 512:
            return json.dumps({"error": "snapshot path too long"})
        if os.path.isabs(snap) or snap.startswith(("/", "\\")):
            return json.dumps({"error": "snapshot must be a workspace-relative path"})
        resolved = os.path.realpath(os.path.join(ws, snap))
        if not resolved.startswith(os.path.realpath(ws) + os.sep):
            return json.dumps({"error": f"snapshot path escapes workspace: {snap!r}"})
    report = verify_workspace(ws, snapshot=snap)
    envelope = report.as_dict()
    envelope["_schema_version"] = "1.0"
    return json.dumps(envelope, indent=2)


@mcp_tool_observe
def verify_chain() -> str:
    """Verify the integrity of the SHA3-512 governance hash chain.

    Walks every entry in the chain and checks that each entry_hash matches
    its recomputed value and that chain linkage is unbroken.

    Returns:
        JSON with valid (bool), length (int), and broken_at (int, -1 if valid).
    """
    ws = _workspace()
    try:
        from mind_mem.governance_gate import get_gate

        gate = get_gate(ws)
        chain = gate.chain
        hc_valid, broken_at = chain.verify_chain()
        length = chain.length

        evidence = gate.evidence
        ev_valid, broken_ids = evidence.verify_chain()
    except Exception as exc:
        _log.warning("verify_chain_failed", error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Chain verification failed: {exc}",
            },
            indent=2,
        )

    overall_valid = hc_valid and ev_valid
    metrics.inc("mcp_verify_chain")
    _log.info(
        "mcp_verify_chain",
        valid=overall_valid,
        hash_chain_valid=hc_valid,
        length=length,
        broken_at=broken_at,
        evidence_valid=ev_valid,
        evidence_broken_ids=broken_ids,
    )
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "valid": overall_valid,
            "hash_chain": {
                "valid": hc_valid,
                "length": length,
                "broken_at": broken_at,
            },
            "evidence_chain": {
                "valid": ev_valid,
                "broken_ids": broken_ids,
            },
        },
        indent=2,
    )


@mcp_tool_observe
def list_evidence(
    block_id: str = "",
    action: str = "",
    limit: int = 20,
) -> str:
    """List governance evidence objects, optionally filtered by block_id or action.

    Args:
        block_id: Filter to evidence records for this block ID (optional).
        action: Filter by evidence action type — PROPOSE, APPLY, ROLLBACK,
                CONTRADICT, DRIFT, RESOLVE, VERIFY (optional).
        limit: Maximum number of records to return (default 20).

    Returns:
        JSON array of evidence objects as dicts.
    """
    ws = _workspace()
    try:
        from mind_mem.evidence_objects import EvidenceAction
        from mind_mem.governance_gate import get_gate

        gate = get_gate(ws)
        evidence = gate.evidence

        if block_id:
            records = evidence.get_evidence_for_block(block_id)
        elif action:
            try:
                ev_action = EvidenceAction(action.upper())
            except ValueError:
                return json.dumps(
                    {
                        "_schema_version": MCP_SCHEMA_VERSION,
                        "error": (
                            f"Unknown action: {action!r}. Valid values: PROPOSE, APPLY, ROLLBACK, CONTRADICT, DRIFT, RESOLVE, VERIFY"
                        ),
                    },
                    indent=2,
                )
            records = evidence.get_evidence_by_action(ev_action)
        else:
            records = evidence.get_latest(limit)

        records = records[-limit:] if len(records) > limit else records

    except Exception as exc:
        _log.warning("list_evidence_failed", error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Evidence listing failed: {exc}",
            },
            indent=2,
        )

    metrics.inc("mcp_list_evidence")
    _log.info("mcp_list_evidence", block_id=block_id, action=action, count=len(records))
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "count": len(records),
            "evidence": [r.to_dict() for r in records],
        },
        indent=2,
        default=str,
    )


def register(mcp) -> None:
    """Wire the audit tools onto *mcp*."""
    mcp.tool(verify_merkle)
    mcp.tool(mind_mem_verify)
    mcp.tool(verify_chain)
    mcp.tool(list_evidence)
