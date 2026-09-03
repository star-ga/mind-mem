# Copyright 2026 STARGA, Inc.
"""``provenance: off | recommended | required`` — the half that was missing.

The five provenance fields (``ActorId``, ``ActorRole``, ``SessionId``,
``ToolId``, ``Purpose``) have shipped for releases and flow through the
propose door. What never existed is the *policy* that decides whether a
block is allowed to arrive without them, which is the part an operator
buys: optional metadata that nobody has to supply is metadata that, at
audit time, half the corpus does not have.

Three states and no fourth:

``off``
    The flag is not on for this workspace. Nothing is checked, nothing is
    read past the one probe, and this build is indistinguishable from the
    one that never had the policy.
``recommended``
    Missing fields are reported as warnings. The write proceeds. This is
    the migration state — a corpus written before the policy existed does
    not become un-writable the day an operator turns it on.
``required``
    A write missing any required field is refused, by
    :func:`require_provenance`, before it reaches a store.

*Which* fields are required is configurable and defaults to all five,
because "required" without a named field list is an opinion rather than
a policy. A configured field outside the known five is a refusal, not a
quietly-ignored key: a policy naming a field nobody checks is worse than
no policy, since it reads as protection.

Reads config once per call and nothing else — no clock, no store, no
model. Evaluation is a pure function of the block's fields and the
resolved policy, which is what lets ``mm compliance provenance`` report
the same answer for the same corpus on every run.

Copyright STARGA, Inc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..v4.feature_flags import flag_config_for_workspace, is_enabled_for_workspace

__all__ = [
    "POLICY_OFF",
    "POLICY_RECOMMENDED",
    "POLICY_REQUIRED",
    "PROVENANCE_FIELDS",
    "PROVENANCE_FLAG",
    "ProvenanceConfigError",
    "ProvenanceDecision",
    "ProvenanceRequired",
    "evaluate_provenance",
    "require_provenance",
    "resolve_policy",
    "resolve_required_fields",
]

#: The declared flag this module consumes. Wiring it is what moves
#: ``provenance`` out of ``UNIMPLEMENTED`` in the flag registry.
PROVENANCE_FLAG = "provenance"

POLICY_OFF = "off"
POLICY_RECOMMENDED = "recommended"
POLICY_REQUIRED = "required"

#: Settable values. ``off`` is the absence of the flag, so it is not one
#: of them — there is exactly one way to be off.
_SETTABLE_POLICIES = (POLICY_RECOMMENDED, POLICY_REQUIRED)

#: The five Group E fields, in the canonical block-field spelling used by
#: :mod:`mind_mem.block_provenance`. Order is the declaration order, which
#: is the order every report prints them in.
PROVENANCE_FIELDS: tuple[str, ...] = ("ActorId", "ActorRole", "SessionId", "ToolId", "Purpose")


class ProvenanceConfigError(ValueError):
    """The workspace declares a policy or field name that does not exist."""


class ProvenanceRequired(RuntimeError):
    """A ``required`` policy refused a write that omitted provenance."""

    def __init__(self, decision: "ProvenanceDecision") -> None:
        super().__init__(f"provenance policy 'required': missing {', '.join(decision.missing)}; write refused")
        self.decision = decision


@dataclass(frozen=True)
class ProvenanceDecision:
    """The verdict for one block or one pending write."""

    policy: str
    required: tuple[str, ...]
    present: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def admitted(self) -> bool:
        """False only under ``required`` with something missing."""
        return not (self.policy == POLICY_REQUIRED and self.missing)

    @property
    def warnings(self) -> tuple[str, ...]:
        """Advisory messages. Empty unless the policy is ``recommended``."""
        if self.policy != POLICY_RECOMMENDED or not self.missing:
            return ()
        return (f"provenance policy 'recommended': missing {', '.join(self.missing)}",)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "required": list(self.required),
            "present": list(self.present),
            "missing": list(self.missing),
            "admitted": self.admitted,
            "warnings": list(self.warnings),
        }


def resolve_policy(workspace: str) -> str:
    """This workspace's policy, or :data:`POLICY_OFF`.

    One probe. Resolve it once per command and pass the answer down: the
    probe reads a file, and re-reading it per block would make the OFF
    path cost a syscall per item that the unwired build never paid.
    """
    if not is_enabled_for_workspace(workspace, PROVENANCE_FLAG):
        return POLICY_OFF
    raw = flag_config_for_workspace(workspace, PROVENANCE_FLAG).get("policy", POLICY_RECOMMENDED)
    if raw not in _SETTABLE_POLICIES:
        raise ProvenanceConfigError(f"v4.provenance.policy is {raw!r}; expected one of {list(_SETTABLE_POLICIES)}")
    return str(raw)


def resolve_required_fields(workspace: str) -> tuple[str, ...]:
    """The fields this workspace requires, defaulting to all five.

    A name outside :data:`PROVENANCE_FIELDS` raises. Accepting it would
    put a field in the policy that no check can ever satisfy or fail,
    which is a policy that looks enforced and is not.
    """
    raw = flag_config_for_workspace(workspace, PROVENANCE_FLAG).get("fields")
    if raw is None:
        return PROVENANCE_FIELDS
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise ProvenanceConfigError("v4.provenance.fields must be a list of field names")
    unknown = sorted({x for x in raw if x not in PROVENANCE_FIELDS})
    if unknown:
        raise ProvenanceConfigError(f"v4.provenance.fields names unknown field(s) {unknown}; known: {list(PROVENANCE_FIELDS)}")
    chosen = {str(x) for x in raw}
    return tuple(f for f in PROVENANCE_FIELDS if f in chosen)


def _has_value(fields: Mapping[str, Any], name: str) -> bool:
    value = fields.get(name)
    return isinstance(value, str) and value.strip() != ""


def evaluate_provenance(
    fields: Mapping[str, Any],
    *,
    policy: str,
    required: Sequence[str] | None = None,
) -> ProvenanceDecision:
    """Judge one block's provenance against an already-resolved policy.

    Pure: no config read, no IO. The policy and the field list come from
    :func:`resolve_policy` / :func:`resolve_required_fields`, resolved
    once by the caller.
    """
    if policy not in (POLICY_OFF, *_SETTABLE_POLICIES):
        raise ProvenanceConfigError(f"unknown provenance policy {policy!r}")
    wanted: Iterable[str] = PROVENANCE_FIELDS if required is None else required
    ordered = tuple(f for f in PROVENANCE_FIELDS if f in set(wanted))
    present = tuple(f for f in ordered if _has_value(fields, f))
    missing = tuple(f for f in ordered if f not in present)
    return ProvenanceDecision(policy=policy, required=ordered, present=present, missing=missing)


def require_provenance(
    fields: Mapping[str, Any],
    *,
    policy: str,
    required: Sequence[str] | None = None,
) -> ProvenanceDecision:
    """Evaluate, and raise :class:`ProvenanceRequired` if the policy refuses."""
    decision = evaluate_provenance(fields, policy=policy, required=required)
    if not decision.admitted:
        raise ProvenanceRequired(decision)
    return decision
