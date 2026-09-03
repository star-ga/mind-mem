"""Audit MCP tools — Merkle proofs, hash chain + evidence chain verification.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, audit domain). Four tools:

* ``verify_merkle`` — prove a block's Merkle inclusion against the
  live tree built from the FTS index.
* ``verify_chain`` — verify every hash-linked ledger the product
  writes, by delegating to the single aggregate verifier
  (:func:`mind_mem.verify_cli.verify_workspace`). It walks nothing
  itself; see its docstring for why that is the fix.
* ``list_evidence`` — enumerate governance evidence objects with
  optional ``block_id`` / ``action`` filters.
* ``mind_mem_verify`` — expose the standalone ``mind-mem-verify``
  CLI over MCP with path-escape guards on the ``snapshot`` arg.
"""

from __future__ import annotations

import json
import os

from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import get_logger, metrics

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


#: Non-ledger checks :func:`~mind_mem.verify_cli.verify_workspace` also
#: produces, republished beside the ledgers so the envelope carries every
#: row the verifier ran rather than the subset this module remembers.
_EXTRA_CHECKS: tuple[str, ...] = ("spec_binding", "snapshot_anchor", "merkle_root", "chain_head")


@mcp_tool_observe
def verify_chain() -> str:
    """Verify every hash-linked ledger in the workspace, as one verdict.

    Delegates to :func:`mind_mem.verify_cli.verify_workspace` — the single
    aggregate verifier — and republishes its rows. It keeps no walk of its
    own, and that is the fix rather than a style choice: until 5.0.2 this
    tool walked the hash chain and the evidence chain only, so tampering
    with the field-audit sidecar or the served-recall ledger left it
    reporting ``valid: true`` while ``mind-mem-verify`` on the same
    workspace reported the workspace broken. A two-of-four verdict
    published under the name ``valid`` is worse than no verdict.

    Two consequences worth naming. ``valid`` is now ``report.ok``, so it
    goes false when **any** ledger fails, and a ledger added to
    :data:`~mind_mem.verify_cli.LEDGER_CHECKS` is covered here the day it
    is added rather than the day someone remembers this file. And the
    tool no longer constructs the governance gate, so verifying a
    workspace creates nothing in it.

    TWO VERDICTS, because the verifier answers two questions and this
    tool is named after one of them. ``valid`` is the **declared ledger
    hierarchy** — every name in
    :data:`~mind_mem.verify_cli.LEDGER_CHECKS`, and fail-closed on a row
    the verifier did not produce. ``workspace_valid`` is ``report.ok``,
    the verifier's whole answer, which also folds in the two non-ledger
    checks (``spec_binding``, ``snapshot_anchor``). They differ exactly
    when the ledgers are intact and the governance config has drifted
    from its binding — a real finding, and not a chain break, so it is
    reported under its own name rather than collapsed into this tool's.
    Neither is hidden and neither is inferred: ``exit_code`` is the
    verifier's, ``checks`` carries every row, and
    ``mind_mem_verify`` publishes the whole report. What is gone is the
    *subset* — a verdict computed here from ledgers this module happened
    to remember.

    Returns:
        JSON with ``valid`` (bool — every declared ledger),
        ``workspace_valid`` (bool — the verifier's whole verdict),
        ``exit_code``, ``ledgers`` (the declared hierarchy), ``checks``
        (every row by name), ``missing``, ``messages``, and one object
        per check carrying its ``valid`` flag plus that check's
        structured facts (``hash_chain`` keeps ``length`` /
        ``broken_at``, ``evidence_chain`` keeps ``broken_ids``).
    """
    ws = _workspace()
    try:
        from mind_mem.verify_cli import LEDGER_CHECKS, verify_workspace

        report = verify_workspace(ws)
    except Exception as exc:
        _log.warning("verify_chain_failed", error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Chain verification failed: {exc}",
            },
            indent=2,
        )

    # ``get`` defaults to False, not True: a ledger the verifier did not
    # produce a row for must not be counted as verified. The row is
    # unconditional today (``tests/test_ledger_hierarchy.py``), and this
    # is what keeps the failure closed if that ever stops being true.
    ledger_valid = all(report.checks.get(name, False) for name in LEDGER_CHECKS)
    envelope: dict = {
        "_schema_version": MCP_SCHEMA_VERSION,
        "valid": ledger_valid,
        "workspace_valid": report.ok,
        "exit_code": report.exit_code,
        "ledgers": list(LEDGER_CHECKS),
        "checks": dict(report.checks),
        "missing": list(report.missing),
        "messages": list(report.messages),
    }
    for name in LEDGER_CHECKS + _EXTRA_CHECKS:
        if name in report.checks:
            envelope[name] = {"valid": report.checks[name], **report.details.get(name, {})}

    metrics.inc("mcp_verify_chain")
    _log.info(
        "mcp_verify_chain",
        valid=ledger_valid,
        workspace_valid=report.ok,
        exit_code=report.exit_code,
        checks=dict(report.checks),
        missing=list(report.missing),
    )
    return json.dumps(envelope, indent=2)


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


def _anchor_history_path(ws: str) -> str:
    """Where the local anchor trail lives for workspace *ws*."""
    return os.path.join(ws, "maintenance", "ledger_anchors.jsonl")


@mcp_tool_observe
def anchor_root(chain: str = "local", tx_hash: str = "", block_height: int = 0) -> str:
    """Record the CURRENT Merkle root in the local external-anchor trail.

    This is the step that turns "tamper-evident locally" into "anchored
    externally": ``verify_merkle`` and ``verify_chain`` prove the store is
    internally consistent, but a holder of the store could in principle rebuild
    a consistent history. An anchor pins a root to a point in time outside the
    store, so a later rewrite is detectable even by someone who trusts nothing
    in the workspace.

    The root is computed here from the live block index rather than accepted
    from the caller -- anchoring a root the caller supplied would anchor the
    caller's claim, not the store's state.

    Posting to a real chain needs web3 keys and network access, which this
    library deliberately does not carry. When no external poster is wired the
    entry records ``status="pending"`` with no tx hash and still gives a
    complete local trail; an integrator wraps their poster around this call and
    passes ``tx_hash`` once it clears.

    Args:
        chain: Ledger identifier the root was (or will be) posted to.
        tx_hash: Transaction hash from an external poster. Empty means the
            anchor is pending.
        block_height: External ledger height, when known.

    Returns:
        JSON with ``ok``, the anchored ``root``, and the recorded ``entry``.
    """
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(chain, str) or not chain.strip():
        return json.dumps({"ok": False, "error": "chain must be a non-empty string"})
    if not isinstance(block_height, int) or isinstance(block_height, bool) or block_height < 0:
        return json.dumps({"ok": False, "error": "block_height must be a non-negative integer"})
    if not isinstance(tx_hash, str):
        return json.dumps({"ok": False, "error": "tx_hash must be a string"})

    try:
        from mind_mem.merkle_tree import MerkleTree
        from mind_mem.sqlite_index import merkle_leaves

        leaves = merkle_leaves(ws)
    except (ImportError, AttributeError):
        leaves = []

    if not leaves:
        return json.dumps({"ok": False, "error": "no block index available -- run 'mind-mem-scan' first"})

    tree = MerkleTree()
    tree.build(leaves)

    try:
        from mind_mem.ledger_anchor import AnchorHistory
        from mind_mem.ledger_anchor import anchor_root as _record

        history = AnchorHistory(_anchor_history_path(ws))
        entry = _record(
            history,
            tree.root_hash,
            block_height=block_height,
            chain=chain.strip(),
            tx_hash=tx_hash.strip() or None,
        )
    except Exception as exc:
        _log.warning("anchor_root_failed", error=str(exc))
        return json.dumps({"ok": False, "error": f"anchor failed: {exc}"})

    metrics.inc("mcp_anchor_root")
    _log.info("mcp_anchor_root", root=tree.root_hash, chain=chain, status=entry.status)
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "ok": True,
            "root": tree.root_hash,
            "leaves": len(leaves),
            "entry": entry.as_dict(),
        },
        indent=2,
    )


@mcp_tool_observe
def anchor_history(limit: int = 20) -> str:
    """List recorded external anchors, newest last, with integrity problems.

    ``problems`` is not decoration: the trail is append-only JSONL, so a
    truncated or hand-edited line is exactly the tampering an anchor exists to
    reveal. Damaged lines are REPORTED rather than silently skipped -- a trail
    that quietly drops what it cannot parse is worse than no trail.

    Args:
        limit: Maximum entries to return (most recent).

    Returns:
        JSON with ``entries``, ``count``, ``latest`` and ``problems``.
    """
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        return json.dumps({"ok": False, "error": "limit must be a positive integer"})

    try:
        from mind_mem.ledger_anchor import AnchorHistory

        history = AnchorHistory(_anchor_history_path(ws))
        entries = history.all()
        problems = history.problems()
        latest = history.latest()
    except Exception as exc:
        _log.warning("anchor_history_failed", error=str(exc))
        return json.dumps({"ok": False, "error": f"anchor history unreadable: {exc}"})

    metrics.inc("mcp_anchor_history")
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "ok": True,
            "count": len(entries),
            "entries": [e.as_dict() for e in entries[-limit:]],
            "latest": latest.as_dict() if latest else None,
            "problems": problems,
        },
        indent=2,
    )


def register(mcp) -> None:
    """Wire the audit tools onto *mcp*."""
    mcp.tool(verify_merkle)
    mcp.tool(mind_mem_verify)
    mcp.tool(verify_chain)
    mcp.tool(list_evidence)
    mcp.tool(anchor_root)
    mcp.tool(anchor_history)
