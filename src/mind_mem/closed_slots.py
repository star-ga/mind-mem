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


def _screen_slot_write(
    workspace: str,
    namespace: str,
    slot: str,
    value: str,
    rationale: str,
    provenance: Mapping[str, str] | None,
    *,
    record_redaction: bool = True,
) -> tuple[str, str, dict[str, str]]:
    """Run the same pre-write controls as ``propose_update``.

    Slot staging is a governed write proposal, so it must not be a second
    door around compliance, sanitisation, or the quality gate.  This helper
    is deliberately shared by the staging path and keeps the default-off
    path inert apart from the existing configuration probe.
    """
    from .apply_engine import _sanitize_reason_for_markdown
    from .block_provenance import MAX_PROVENANCE_VALUE_LEN, PROVENANCE_FIELDS, clean_provenance_value
    from .codepoint_sanitize import sanitize_text_for_ingest
    from .compliance.prewrite import PreWritePolicy, screen
    from .compliance.provenance_policy import ProvenanceConfigError, ProvenanceRequired
    from .compliance.redaction import MODE_OFF, RedactionConfigError

    canonical_to_param = {field: param for param, field in PROVENANCE_FIELDS.items()}
    supplied: dict[str, str] = {}
    for field, raw in (provenance or {}).items():
        param = canonical_to_param.get(field)
        if param is None:
            raise SlotConfigError(f"unknown provenance field: {field}")
        if raw in (None, ""):
            continue
        if not isinstance(raw, str):
            raise SlotConfigError(f"provenance field {param!r} must be a string")
        if len(raw) > MAX_PROVENANCE_VALUE_LEN:
            raise SlotConfigError(f"{param} exceeds {MAX_PROVENANCE_VALUE_LEN} chars (provenance values are metadata, not content)")
        try:
            cleaned = clean_provenance_value(param, sanitize_text_for_ingest(raw, workspace, source=f"slot.{field}"))
        except (TypeError, ValueError) as exc:
            raise SlotConfigError(f"provenance_invalid: {exc}") from exc
        if cleaned:
            supplied[field] = cleaned

    cleaned_value = _sanitize_reason_for_markdown(sanitize_text_for_ingest(value.strip(), workspace, source="slot.statement"))
    cleaned_rationale = _sanitize_reason_for_markdown(sanitize_text_for_ingest(rationale.strip(), workspace, source="slot.rationale"))
    try:
        policy = PreWritePolicy.resolve(workspace)
        results: dict[str, Any] = {
            "statement": screen(
                cleaned_value, policy=policy, provenance=supplied, target=f"decision.slot.{namespace}.{slot}", record=False
            ),
            "rationale": screen(
                cleaned_rationale, policy=policy, provenance=supplied, target=f"decision.slot.{namespace}.{slot}.rationale", record=False
            ),
        }
        if policy.redaction_mode != MODE_OFF:
            for field, value in supplied.items():
                results[f"provenance:{field}"] = screen(
                    value,
                    policy=policy,
                    provenance=supplied,
                    target=f"decision.slot.{namespace}.{slot}.{field}",
                    record=False,
                )
    except (ProvenanceConfigError, RedactionConfigError) as exc:
        raise SlotConfigError(f"compliance_config_invalid: {exc}") from exc
    except ProvenanceRequired:
        # Preserve the ordinary proposal door's typed refusal so the MCP
        # facade can return its stable ``provenance_required`` envelope.
        raise

    # A redaction policy may rewrite the statement, but the slot digest must
    # describe exactly what is stored.  Carry the screened text forward and
    # retain the same field names as the ordinary proposal door.
    result_value = results["statement"].text
    result_rationale = results["rationale"].text
    for field, result in results.items():
        if field.startswith("provenance:") and field.rsplit(":", 1)[-1] != "Purpose" and result.changed:
            raise ClosedSlotError(f"redaction_identity_refused: {field.rsplit(':', 1)[-1]}")

    if policy.redaction_mode != MODE_OFF and record_redaction:
        from .compliance.audit import record_redaction as _record_redaction

        for field, result in results.items():
            _record_redaction(
                workspace, result.redaction, target=f"decision.slot.{namespace}.{slot}.{field}", agent=supplied.get("ActorId", "")
            )

    if policy.redaction_mode != MODE_OFF:
        for field, result in results.items():
            if field.startswith("provenance:") and field.rsplit(":", 1)[-1] == "Purpose":
                supplied["Purpose"] = result.text

    from .mcp.infra.config import _get_quality_gate_mode
    from .quality_gate import validate_block

    qg_mode = _get_quality_gate_mode(workspace)
    if qg_mode != "off":
        from .mcp.tools.governance import _recent_statements

        quality_verdict = validate_block(result_value, strict=qg_mode == "strict", recent=_recent_statements(workspace))
        if not quality_verdict.accept:
            raise ClosedSlotError(f"quality_gate_rejection: {', '.join(quality_verdict.reasons)}")

    if _v4_enabled_for_slots(workspace):
        from .v4.block_metadata import validate_block as _validate_metadata_block
        from .v4.feature_flags import FeatureDisabledError

        v4_fields: dict[str, Any] = {
            "statement": result_value,
            "confidence": "medium",
            "tags": ["closed-slot", namespace, slot],
            **{canonical_to_param[field]: value for field, value in supplied.items()},
        }
        try:
            metadata_verdict = _validate_metadata_block("decision", v4_fields, workspace=workspace)
        except FeatureDisabledError:
            # The flag can be edited between the quiet probe and validation;
            # the second read's disabled answer is the current policy.
            metadata_verdict = None
        if metadata_verdict is None:
            return result_value, result_rationale, supplied
        if not metadata_verdict.ok:
            raise ClosedSlotError(f"schema_validation_rejection: {metadata_verdict.reason}")
    return result_value, result_rationale, supplied


def _v4_enabled_for_slots(workspace: str) -> bool:
    """Return whether the ordinary v4 metadata gate is active for this workspace."""

    from .v4.block_metadata import FLAG
    from .v4.feature_flags import is_enabled_for_workspace

    # An explicit workspace is the policy boundary for direct callers.  Do
    # not fall back to the process ambient workspace here: a caller may be
    # staging one workspace while another request owns the process context.
    return is_enabled_for_workspace(workspace, FLAG)


def _revalidate_slot_policy(
    workspace: str,
    block: Mapping[str, Any],
    *,
    namespace: str,
    slot: str,
) -> None:
    """Re-run current write policy before a staged slot reaches source truth."""

    from .compliance.provenance_policy import ProvenanceRequired
    from .compliance.redaction import RedactionRefused

    provenance = {field: block[field] for field in ("ActorId", "ActorRole", "SessionId", "ToolId", "Purpose") if block.get(field)}
    try:
        value, rationale, current_provenance = _screen_slot_write(
            workspace,
            namespace,
            slot,
            str(block.get("Statement", "")),
            str(block.get("Rationale", "")),
            provenance,
            record_redaction=False,
        )
    except (ProvenanceRequired, RedactionRefused, ClosedSlotError) as exc:
        raise SlotConfigError(f"closed-slot {namespace}/{slot} current policy requires restaging: {exc}") from exc

    stored_provenance = {field: str(block[field]) for field in provenance if block.get(field)}
    if value != str(block.get("Statement", "")) or rationale != str(block.get("Rationale", "")):
        raise SlotConfigError(f"closed-slot {namespace}/{slot} current redaction differs; restage the proposal")
    if current_provenance != stored_provenance:
        raise SlotConfigError(f"closed-slot {namespace}/{slot} current provenance normalization differs; restage the proposal")


def _validate_slot_payload(
    workspace: str,
    blocks: list[dict[str, Any]],
    *,
    active_blocks: list[dict[str, Any]],
    target: dict[str, Any] | None = None,
) -> tuple[str, str] | None:
    """Validate a slot payload at the source-of-truth write boundary."""
    slot_blocks = [block for block in blocks if _slot_fields(block) is not None]
    if not slot_blocks:
        return None
    if len(blocks) != 1:
        raise SlotInvariantError("a closed-slot operation must contain exactly one block")
    block = slot_blocks[0]
    identity = _slot_fields(block)
    assert identity is not None
    namespace, slot = identity
    declaration = require_slot(workspace, namespace, slot)
    if str(block.get("SlotSetVersion", "")) != str(declaration.version):
        raise SlotConfigError(
            f"closed-slot {namespace}/{slot} declaration version changed: "
            f"proposal={block.get('SlotSetVersion')!r}, current={declaration.version}"
        )
    expected_digest = hashlib.sha256(str(block.get("Statement", "")).encode("utf-8")).hexdigest()
    if block.get("SlotValueDigest") != expected_digest:
        raise SlotInvariantError(f"closed-slot {namespace}/{slot} value digest does not match Statement")
    _revalidate_slot_policy(workspace, block, namespace=namespace, slot=slot)
    if target is None:
        occupants = [item for item in active_blocks if _slot_fields(item) == identity]
        if len(occupants) > 1:
            raise SlotInvariantError(f"slot {namespace}/{slot} has multiple active occupants")
        if occupants:
            raise SlotInvariantError(f"closed-slot {namespace}/{slot} already has an active occupant")
    else:
        old_identity = _slot_fields(target)
        if old_identity != identity:
            raise SlotInvariantError("closed-slot identity does not match target")
        if target.get("Status") != "active":
            raise SlotInvariantError("closed-slot target is no longer active")
        occupants = [item for item in active_blocks if item.get("_id") != target.get("_id") and _slot_fields(item) == identity]
        if occupants:
            raise SlotInvariantError(f"closed-slot {namespace}/{slot} already has another active occupant")
    return identity


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
        + "".join(
            f"{field}: {proposal[field]}\n"
            for field in ("ActorId", "ActorRole", "SessionId", "ToolId", "Purpose")
            if proposal.get(field)
        )
        + f"FilesTouched:\n{touched}\nSources:\n{sources}\n"
    )


def stage_slot_update(
    workspace: str,
    namespace: str,
    slot: str,
    value: str,
    *,
    rationale: str,
    actor_id: str = "",
    actor_role: str = "",
    session_id: str = "",
    tool_id: str = "",
    purpose: str = "",
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
    value, rationale, provenance = _screen_slot_write(
        workspace,
        namespace,
        slot,
        value,
        rationale,
        {
            "ActorId": actor_id,
            "ActorRole": actor_role,
            "SessionId": session_id,
            "ToolId": tool_id,
            "Purpose": purpose,
        },
    )
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
        block.update(provenance)
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
        proposal.update(provenance)
        proposal["Fingerprint"] = compute_fingerprint(proposal)
        errors = validate_proposal(proposal)
        if errors:
            raise ClosedSlotError(f"generated slot proposal failed validation: {errors}")
        # Keep slot staging subject to the same bounded proposal admission as
        # ordinary ``propose_update`` calls.  In particular, a closed-set
        # helper must not become an unbounded backlog or duplicate-fingerprint
        # side door around governance.
        from .apply_engine import check_backlog_limit, check_fingerprint_dedup

        backlog_count, over_limit = check_backlog_limit(workspace)
        if over_limit:
            raise ClosedSlotError(f"proposal backlog limit exceeded ({backlog_count} staged)")
        duplicate, duplicate_id = check_fingerprint_dedup(workspace, proposal)
        if duplicate:
            raise PendingSlotProposalError(f"proposal fingerprint already staged as {duplicate_id}")
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
    "_validate_slot_payload",
]
