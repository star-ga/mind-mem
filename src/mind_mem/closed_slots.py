"""Closed, versioned fact slots on the governed Markdown store.

A closed slot is an authored configuration member.  This module only stages
proposals; the existing ``approve_apply`` gate is the only path that can make
a slot block part of the source of truth.  Unslotted blocks keep their normal
free-form/detective semantics.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final, Mapping

from .apply_engine import compute_fingerprint, validate_proposal
from .block_parser import parse_blocks, parse_file
from .block_store import _render_block
from .mind_filelock import FileLock
from .storage import get_block_store

SLOT_CONFIG_KEY: Final = "closed_slots"
PROPOSAL_FILE: Final = "intelligence/proposed/EDITS_PROPOSED.md"
DECISION_FILE: Final = "decisions/DECISIONS.md"
_NAME_RE: Final = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_BLOCK_ID_RE: Final = re.compile(r"^D-(\d{8})-(\d{3})$")
_PROPOSAL_ID_RE: Final = re.compile(r"^P-\d{8}-\d{3}$")


class ClosedSlotError(ValueError):
    """Base class for closed-slot declaration and staging failures."""


class SlotConfigError(ClosedSlotError):
    """The authored closed-slot declaration is malformed."""


class UnknownSlotError(ClosedSlotError):
    """The namespace or member is outside the authored closed set."""


class PendingSlotProposalError(ClosedSlotError):
    """A different update for the same slot is already awaiting review."""


class SlotInvariantError(ClosedSlotError):
    """The store contains more than one active occupant for one slot."""


@dataclass(frozen=True)
class SlotDeclaration:
    """One authored version of a closed namespace."""

    namespace: str
    version: int
    slots: tuple[str, ...]

    def contains(self, slot: str) -> bool:
        return slot in self.slots


def _name(value: object, field: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise SlotConfigError(f"{field} must match {_NAME_RE.pattern!r}")
    return value


def load_slot_declarations(workspace: str) -> dict[str, SlotDeclaration]:
    """Read and validate the authored ``closed_slots`` config section.

    The section is opt-in.  If present, malformed declarations fail closed;
    they are never silently treated as an open namespace.
    """

    path = os.path.join(os.path.abspath(workspace), "mind-mem.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SlotConfigError(f"cannot read workspace config: {exc}") from exc
    if not isinstance(config, dict):
        raise SlotConfigError("mind-mem.json must contain an object")
    section = config.get(SLOT_CONFIG_KEY)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise SlotConfigError("closed_slots must be an object")
    schema_version = section.get("version")
    if type(schema_version) is not int or schema_version < 1:
        raise SlotConfigError("closed_slots.version must be a positive integer")
    namespaces = section.get("namespaces")
    if not isinstance(namespaces, dict) or not namespaces:
        raise SlotConfigError("closed_slots.namespaces must be a non-empty object")

    declarations: dict[str, SlotDeclaration] = {}
    for raw_namespace, raw_decl in namespaces.items():
        namespace = _name(raw_namespace, "namespace")
        if namespace in declarations:
            raise SlotConfigError(f"duplicate namespace: {namespace}")
        if not isinstance(raw_decl, dict):
            raise SlotConfigError(f"closed_slots.namespaces.{namespace} must be an object")
        version = raw_decl.get("version")
        if type(version) is not int or version < 1:
            raise SlotConfigError(f"namespace {namespace!r} version must be a positive integer")
        slots = raw_decl.get("slots")
        if not isinstance(slots, list) or not slots:
            raise SlotConfigError(f"namespace {namespace!r} slots must be a non-empty list")
        cleaned = tuple(_name(raw_slot, f"slot in {namespace}") for raw_slot in slots)
        if len(set(cleaned)) != len(cleaned):
            raise SlotConfigError(f"namespace {namespace!r} contains duplicate slot members")
        declarations[namespace] = SlotDeclaration(namespace, version, cleaned)
    return declarations


def require_slot(workspace: str, namespace: str, slot: str) -> SlotDeclaration:
    """Validate a caller's closed-set member and return its declaration."""

    namespace = _name(namespace, "namespace")
    slot = _name(slot, "slot")
    declaration = load_slot_declarations(workspace).get(namespace)
    if declaration is None:
        raise UnknownSlotError(f"namespace {namespace!r} is not a declared closed slot set")
    if not declaration.contains(slot):
        raise UnknownSlotError(f"slot {slot!r} is not a member of namespace {namespace!r}; allowed={list(declaration.slots)!r}")
    return declaration


def _slot_fields(block: Mapping[str, Any]) -> tuple[str, str] | None:
    namespace = block.get("SlotNamespace")
    slot = block.get("SlotName")
    if isinstance(namespace, str) and isinstance(slot, str) and namespace and slot:
        return namespace, slot
    return None


def find_active_slot_occupant(workspace: str, namespace: str, slot: str) -> dict[str, Any] | None:
    """Find the sole active block occupying ``(namespace, slot)``."""

    blocks = get_block_store(workspace).get_all(active_only=True)
    matches = [b for b in blocks if _slot_fields(b) == (namespace, slot)]
    if len(matches) > 1:
        ids = [str(b.get("_id", "")) for b in matches]
        raise SlotInvariantError(f"slot {namespace}/{slot} has multiple active occupants: {ids}")
    return matches[0] if matches else None


def _proposal_blocks(workspace: str) -> list[dict[str, Any]]:
    path = os.path.join(workspace, PROPOSAL_FILE)
    return parse_file(path) if os.path.isfile(path) else []


def _pending_slot_proposal(workspace: str, namespace: str, slot: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return ``(proposal, parsed new block)`` for a staged slot update."""

    for proposal in _proposal_blocks(workspace):
        if proposal.get("Status") != "staged":
            continue
        if proposal.get("SlotNamespace") != namespace or proposal.get("SlotName") != slot:
            continue
        ops = proposal.get("Ops")
        if not isinstance(ops, list):
            raise SlotInvariantError(f"staged slot proposal {proposal.get('ProposalId')} has no operations")
        for op in ops:
            if isinstance(op, dict) and op.get("op") in {"append_block", "supersede_decision"}:
                patch = op.get("patch") or op.get("new_block")
                if isinstance(patch, str):
                    blocks = parse_blocks(patch)
                    if len(blocks) == 1:
                        return proposal, blocks[0]
        raise SlotInvariantError(f"staged slot proposal {proposal.get('ProposalId')} has no block payload")
    return None


def _next_block_id(workspace: str, date_compact: str, staged_text: str = "") -> str:
    """Allocate a collision-free D id under the proposal-file lock."""

    highest = 0
    store = get_block_store(workspace)
    for block in store.get_all(active_only=False):
        match = _BLOCK_ID_RE.fullmatch(str(block.get("_id", "")))
        if match and match.group(1) == date_compact:
            highest = max(highest, int(match.group(2)))
    for raw in re.findall(rf"D-{re.escape(date_compact)}-(\d{{3}})", staged_text):
        highest = max(highest, int(raw))
    if highest >= 999:
        raise ClosedSlotError(f"decision id space exhausted for {date_compact}")
    return f"D-{date_compact}-{highest + 1:03d}"


def _next_proposal_id(existing: str, date_compact: str) -> str:
    used = [int(raw) for raw in re.findall(rf"P-{re.escape(date_compact)}-(\d{{3}})", existing)]
    nxt = max(used, default=0) + 1
    if nxt > 999:
        raise ClosedSlotError(f"proposal id space exhausted for {date_compact}")
    return f"P-{date_compact}-{nxt:03d}"


def _one_line(text: str) -> str:
    return " ".join(str(text).replace("[", "(").replace("]", ")").split())[:280]


def _render_proposal(proposal: Mapping[str, Any]) -> str:
    ops_lines: list[str] = []
    for op in proposal["Ops"]:
        ops_lines.extend([f"- op: {op['op']}", f"  file: {op['file']}"])
        if "target" in op:
            ops_lines.append(f"  target: {op['target']}")
        ops_lines.append("  patch: |")
        ops_lines.extend(f"    {line}" for line in str(op["patch"]).splitlines())
    evidence = "\n".join(f"- {line}" for line in proposal["Evidence"])
    touched = "\n".join(f"- {line}" for line in proposal["FilesTouched"])
    sources = "\n".join(f"- {line}" for line in proposal["Sources"])
    return (
        f"\n[{proposal['ProposalId']}]\n"
        f"ProposalId: {proposal['ProposalId']}\nType: {proposal['Type']}\n"
        f"TargetBlock: {proposal['TargetBlock']}\nRisk: {proposal['Risk']}\n"
        f"Evidence:\n{evidence}\nRollback: {proposal['Rollback']}\nOps:\n"
        + "\n".join(ops_lines)
        + f"\nFingerprint: {proposal['Fingerprint']}\nStatus: {proposal['Status']}\n"
        f"SlotNamespace: {proposal['SlotNamespace']}\nSlotName: {proposal['SlotName']}\n"
        f"SlotSetVersion: {proposal['SlotSetVersion']}\n"
        f"FilesTouched:\n{touched}\nSources:\n{sources}\n"
    )


def stage_slot_update(
    workspace: str,
    namespace: str,
    slot: str,
    value: str,
    *,
    rationale: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Stage a new or superseding slot proposal; never changes source truth."""

    if not isinstance(workspace, str) or not os.path.isdir(workspace):
        raise ClosedSlotError("workspace must be an existing directory")
    if not isinstance(value, str) or not value.strip():
        raise ClosedSlotError("value must be a non-empty string")
    if len(value) > 500:
        raise ClosedSlotError("value exceeds the 500-character Statement limit")
    if not isinstance(rationale, str) or len(rationale.strip()) < 8:
        raise ClosedSlotError("rationale must be at least 8 non-whitespace characters")
    declaration = require_slot(workspace, namespace, slot)
    stamp = now or datetime.now(timezone.utc)
    date_iso = stamp.astimezone(timezone.utc).strftime("%Y-%m-%d")
    date_compact = date_iso.replace("-", "")
    proposal_path = os.path.join(workspace, PROPOSAL_FILE)
    if not os.path.isfile(proposal_path):
        raise ClosedSlotError(f"missing proposal file: {PROPOSAL_FILE}")

    with FileLock(proposal_path):
        with open(proposal_path, encoding="utf-8") as handle:
            existing_text = handle.read()
        pending = _pending_slot_proposal(workspace, namespace, slot)
        if pending is not None:
            pending_proposal, pending_block = pending
            if pending_block.get("Statement") == value:
                return {
                    "status": "already_staged",
                    "proposal_id": pending_proposal.get("ProposalId"),
                    "namespace": namespace,
                    "slot": slot,
                }
            raise PendingSlotProposalError(f"slot {namespace}/{slot} already has staged proposal {pending_proposal.get('ProposalId')}")

        occupant = find_active_slot_occupant(workspace, namespace, slot)
        if occupant is not None and occupant.get("Statement") == value:
            return {
                "status": "reasserted",
                "namespace": namespace,
                "slot": slot,
                "occupant_id": occupant.get("_id"),
                "value_digest": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                "source_changed": False,
            }

        new_id = _next_block_id(workspace, date_compact, existing_text)
        block: dict[str, Any] = {
            "_id": new_id,
            "Statement": value,
            "Date": date_iso,
            "Status": "active",
            "Type": "decision",
            "Scope": "global",
            "Rationale": _one_line(rationale),
            "Supersedes": "none",
            # The legacy validator consumes Tags as a comma-delimited scalar;
            # retain that wire shape while keeping slot metadata additive.
            "Tags": f"closed-slot, {namespace}, {slot}",
            "Sources": [f"closed-slot:{namespace}/{slot}"],
            "SlotNamespace": namespace,
            "SlotName": slot,
            "SlotSetVersion": str(declaration.version),
            "SlotValueDigest": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        }
        # Proposal block serialization is line based; omit the renderer's
        # terminal newline so parsing the staged text preserves the exact
        # fingerprinted operation payload.
        patch = _render_block(block).rstrip("\n")
        if occupant is None:
            operation = {"op": "append_block", "file": DECISION_FILE, "target": new_id, "patch": patch}
            target = new_id
            action = "created"
        else:
            operation = {"op": "supersede_decision", "file": DECISION_FILE, "target": occupant["_id"], "patch": patch}
            target = str(occupant["_id"])
            action = "supersession"
        proposal_id = _next_proposal_id(existing_text, date_compact)
        proposal: dict[str, Any] = {
            "ProposalId": proposal_id,
            "Type": "edit",
            "TargetBlock": target,
            "Risk": "medium",
            "Evidence": [_one_line(f"M4 closed slot {namespace}/{slot}: {action} governed fact")],
            "Rollback": "restore_snapshot",
            "Ops": [operation],
            "Status": "staged",
            "FilesTouched": [DECISION_FILE],
            "Sources": [f"closed-slot:{namespace}/{slot}"],
            "SlotNamespace": namespace,
            "SlotName": slot,
            "SlotSetVersion": str(declaration.version),
        }
        proposal["Fingerprint"] = compute_fingerprint(proposal)
        errors = validate_proposal(proposal)
        if errors:
            raise ClosedSlotError(f"generated slot proposal failed validation: {errors}")
        if f"Fingerprint: {proposal['Fingerprint']}" in existing_text:
            raise PendingSlotProposalError(f"an identical proposal is already staged: {proposal_id}")
        with open(proposal_path, "a", encoding="utf-8") as handle:
            handle.write(_render_proposal(proposal))
    return {
        "status": "staged",
        "proposal_id": proposal_id,
        "namespace": namespace,
        "slot": slot,
        "operation": action,
        "target_block": target,
        "new_block": new_id,
        "slot_set_version": declaration.version,
        "value_digest": block["SlotValueDigest"],
        "next_step": f"Review, then approve_apply('{proposal_id}', dry_run=False).",
    }


__all__ = [
    "ClosedSlotError",
    "PendingSlotProposalError",
    "SlotConfigError",
    "SlotDeclaration",
    "SlotInvariantError",
    "UnknownSlotError",
    "find_active_slot_occupant",
    "load_slot_declarations",
    "require_slot",
    "stage_slot_update",
]
