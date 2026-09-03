# Copyright 2026 STARGA, Inc.
"""One pre-write door: provenance policy, then the detector chain, then the ledger.

Two compliance controls run before a write, and the order between them is
load-bearing. Provenance goes first: if the policy refuses a write for
missing attribution, nothing should have scanned the text, hashed it, or
touched the ledger on its behalf. Redaction goes second, and its audit
entry is written last, so an entry exists only for a pass that actually
happened.

:class:`PreWritePolicy` exists so a batch caller resolves the workspace's
configuration **once** and then screens N documents against the resolved
answer. The alternative — each document re-probing the config — turns an
OFF feature into a syscall per item, which is a cost the build without
the feature never paid. :meth:`PreWritePolicy.inert` is the caller's
early-out for exactly that case.

This module is the seam. Its in-tree callers today are the ``mm
compliance`` verbs and the ``redacted`` export policy; the propose door
is the next caller, and it is deliberately a separate change so that the
door's own tests move with it.

Copyright STARGA, Inc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .audit import record_redaction
from .detectors import Detector
from .provenance_policy import (
    POLICY_OFF,
    ProvenanceDecision,
    evaluate_provenance,
    require_provenance,
    resolve_policy,
    resolve_required_fields,
)
from .redaction import MODE_OFF, RedactionResult, redact, redaction_chain_for_workspace, resolve_mode

__all__ = ["PreWritePolicy", "Screening", "screen", "screen_for_workspace"]


@dataclass(frozen=True)
class PreWritePolicy:
    """Everything the door needs, resolved once from config.

    Immutable on purpose: a screening batch must not change policy
    half-way through because someone edited ``mind-mem.json``, or the
    bundle it produced would be two policies stapled together.
    """

    workspace: str
    redaction_mode: str
    detectors: tuple[Detector, ...]
    provenance_policy: str
    required_fields: tuple[str, ...]

    @classmethod
    def resolve(cls, workspace: str) -> "PreWritePolicy":
        redaction_mode = resolve_mode(workspace)
        detectors = redaction_chain_for_workspace(workspace) if redaction_mode != MODE_OFF else ()
        provenance_policy = resolve_policy(workspace)
        required = resolve_required_fields(workspace) if provenance_policy != POLICY_OFF else ()
        return cls(
            workspace=workspace,
            redaction_mode=redaction_mode,
            detectors=detectors,
            provenance_policy=provenance_policy,
            required_fields=required,
        )

    @property
    def inert(self) -> bool:
        """True when both controls are off, so the caller can skip the door."""
        return self.redaction_mode == MODE_OFF and self.provenance_policy == POLICY_OFF

    def to_dict(self) -> dict[str, object]:
        return {
            "redaction_mode": self.redaction_mode,
            "detectors": [d.name for d in self.detectors],
            "provenance_policy": self.provenance_policy,
            "required_fields": list(self.required_fields),
        }


@dataclass(frozen=True)
class Screening:
    """What the door decided, and the text a caller may now write."""

    text: str
    redaction: RedactionResult
    provenance: ProvenanceDecision
    audit_seq: Optional[int] = None

    @property
    def changed(self) -> bool:
        return self.redaction.changed

    def to_dict(self) -> dict[str, object]:
        return {
            "redaction": self.redaction.to_dict(),
            "provenance": self.provenance.to_dict(),
            "audit_seq": self.audit_seq,
            "changed": self.changed,
        }


def screen(
    text: str,
    *,
    policy: PreWritePolicy,
    provenance: Mapping[str, Any] | None = None,
    target: str = "",
    agent: str = "",
    record: bool = True,
    detectors: Sequence[Detector] | None = None,
) -> Screening:
    """Run the pre-write controls against *text* under a resolved *policy*.

    Raises :class:`~mind_mem.compliance.provenance_policy.ProvenanceRequired`
    when attribution is missing under a ``required`` policy, and
    :class:`~mind_mem.compliance.redaction.RedactionRefused` when the text
    carries something a ``reject`` workspace will not store. Both are
    refusals of the *write*, which is the point: a compliance control that
    logs and proceeds is a log line, not a control.

    *record* exists for the read-only surfaces (``mm compliance scan``,
    the export pass) that must not append to the ledger. It never widens
    what a write may do — a caller that rewrites text and passes
    ``record=False`` has produced an unaudited redaction, which is why the
    write verbs do not expose it.
    """
    fields = provenance or {}
    if policy.provenance_policy == POLICY_OFF:
        decision = evaluate_provenance(fields, policy=POLICY_OFF, required=())
    else:
        decision = require_provenance(fields, policy=policy.provenance_policy, required=policy.required_fields)

    chain = policy.detectors if detectors is None else tuple(detectors)
    result = redact(text, mode=policy.redaction_mode, detectors=chain)

    seq: Optional[int] = None
    if record and result.mode != MODE_OFF:
        entry = record_redaction(policy.workspace, result, target=target or "(unnamed)", agent=agent)
        seq = entry.seq if entry is not None else None
    return Screening(text=result.text, redaction=result, provenance=decision, audit_seq=seq)


def screen_for_workspace(
    text: str,
    *,
    workspace: str,
    provenance: Mapping[str, Any] | None = None,
    target: str = "",
    agent: str = "",
    record: bool = True,
) -> Screening:
    """Resolve this workspace's policy and screen one document with it.

    The single-document convenience. A caller with many documents should
    resolve :class:`PreWritePolicy` once and call :func:`screen`.
    """
    return screen(text, policy=PreWritePolicy.resolve(workspace), provenance=provenance, target=target, agent=agent, record=record)
