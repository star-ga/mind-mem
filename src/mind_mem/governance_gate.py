# Copyright 2026 STARGA, Inc.
"""GovernanceGate — single choke-point for all block writes.

Every block write must pass through GovernanceGate.admit().  The gate
verifies spec-hash consistency, creates an evidence object, and appends
an entry to the SHA3-512 hash chain.  If the spec-hash has drifted the
gate raises GovernanceBypassError and the write is blocked.

Usage:
    from .governance_gate import GovernanceGate, GovernanceBypassError

    gate = GovernanceGate(workspace="/path/to/ws")
    gate.admit("WRITE", "D-20260401-001", "content of block")
"""

from __future__ import annotations

import os
import threading
from typing import Optional

from .evidence_objects import EvidenceAction, EvidenceChain
from .hash_chain_v2 import HashChainV2
from .observability import get_logger
from .spec_binding import SpecBindingManager


# Resolve the current_agent_id contextvar lazily so the API layer is not a
# hard dependency of the governance layer (the REST API is optional).
def _current_agent() -> str:
    """Return the authenticated agent ID from the REST context, or 'system'."""
    try:
        from mind_mem.api.rest import current_agent_id  # noqa: PLC0415

        return current_agent_id.get()
    except Exception:
        return "system"

_log = get_logger("governance_gate")


class GovernanceBypassError(Exception):
    """Raised when a write is blocked because the spec-hash has drifted."""


# ---------------------------------------------------------------------------
# Module-level lazy singletons keyed by workspace path
# ---------------------------------------------------------------------------

_gate_lock = threading.Lock()
_gates: dict[str, "GovernanceGate"] = {}


def get_gate(workspace: str) -> "GovernanceGate":
    """Return (creating if needed) the GovernanceGate singleton for *workspace*."""
    ws = os.path.realpath(workspace)
    with _gate_lock:
        if ws not in _gates:
            _gates[ws] = GovernanceGate(ws)
        return _gates[ws]


# ---------------------------------------------------------------------------
# GovernanceGate
# ---------------------------------------------------------------------------


class GovernanceGate:
    """Single choke-point for all governance-audited block writes.

    Initialised once per workspace.  Holds references to the shared
    HashChainV2 and EvidenceChain so every write contributes to the
    same persistent audit trail.

    Args:
        workspace: Absolute path to the mind-mem workspace.
        config_path: Optional path to mind-mem.json.  Defaults to
            ``<workspace>/mind-mem.json``.
    """

    def __init__(self, workspace: str, config_path: Optional[str] = None) -> None:
        self._ws = os.path.realpath(workspace)
        self._config_path = config_path or os.path.join(self._ws, "mind-mem.json")

        memory_dir = os.path.join(self._ws, "memory")
        os.makedirs(memory_dir, exist_ok=True)

        self._chain = HashChainV2(os.path.join(memory_dir, "hash_chain_v2.db"))
        self._evidence = EvidenceChain(store_path=os.path.join(memory_dir, "evidence_chain.jsonl"))
        self._spec_mgr = SpecBindingManager(self._config_path)
        # Serialize admit() so evidence-then-chain writes cannot interleave
        # across threads: two interleaved admits could write evidence A,
        # evidence B, chain B, chain A — the evidence and chain orderings
        # would diverge, breaking audit-trail-to-chain-head correlation.
        self._admit_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def chain(self) -> HashChainV2:
        """The underlying HashChainV2 instance."""
        return self._chain

    @property
    def evidence(self) -> EvidenceChain:
        """The underlying EvidenceChain instance."""
        return self._evidence

    def admit(
        self,
        action: str,
        block_id: str,
        content: str,
        actor: str = "",
        target_file: str = "",
        metadata: Optional[dict] = None,
    ) -> bool:
        """Admit a write through the governance gate.

        Steps:
        1. Verify spec-hash is current.  Raise GovernanceBypassError if drifted.
        2. Create an evidence object for the action.
        3. Append a hash-chain entry.

        Args:
            action:  Verb describing the write (e.g. "WRITE", "APPLY", "DELETE").
            block_id: Logical block identifier.
            content: Raw content being written (hashed, not stored inline).
            actor:   Identity of the caller (default "system").
            target_file: Relative path of the target file (optional).
            metadata: Extra contextual data (optional).

        Returns:
            True when the write is admitted.

        Raises:
            GovernanceBypassError: When the spec-hash has drifted and the
                write is blocked.
        """
        # Resolve actor: explicit argument wins; fall back to contextvar then "system"
        effective_actor = actor if actor else _current_agent()

        with self._admit_lock:
            # Step 1 — spec-hash check (only when a binding exists)
            spec_hash = self._current_spec_hash()
            if spec_hash is None:
                _log.debug("governance_gate.no_binding", block_id=block_id, action=action)
            else:
                valid, reason = self._spec_mgr.verify()
                if not valid:
                    _log.warning(
                        "governance_gate.spec_drifted",
                        block_id=block_id,
                        action=action,
                        reason=reason,
                    )
                    raise GovernanceBypassError(f"GovernanceGate blocked write to '{block_id}': spec-hash drifted. {reason}")

            # Step 2 — create evidence object
            ev_action = _map_action(action)
            meta = dict(metadata or {})
            if spec_hash:
                meta["spec_hash"] = spec_hash
            # Always surface the resolved agent ID in metadata so the audit
            # record carries attribution regardless of which field consumers read.
            meta.setdefault("agent_id", effective_actor)
            self._evidence.create(
                action=ev_action,
                actor=effective_actor,
                target_block_id=block_id,
                target_file=target_file,
                payload=content,
                metadata=meta,
            )

            # Step 3 — append to hash chain. If this fails after evidence
            # was persisted, log the inconsistency loudly so operators can
            # reconcile the two stores. A true two-phase commit is not
            # possible across JSONL + SQLite; best-effort atomicity via the
            # lock + ordered write is the strongest guarantee here.
            try:
                self._chain.append(block_id, action, content)
            except Exception:
                _log.error(
                    "governance_gate.chain_append_failed_after_evidence",
                    block_id=block_id,
                    action=action,
                    actor=effective_actor,
                )
                raise

            _log.debug(
                "governance_gate.admitted",
                block_id=block_id,
                action=action,
                actor=effective_actor,
            )
            return True

    def current_spec_hash(self) -> Optional[str]:
        """Return the current spec_hash from the binding, or None."""
        return self._current_spec_hash()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_spec_hash(self) -> Optional[str]:
        binding = self._spec_mgr.get_binding()
        return binding.spec_hash if binding is not None else None


# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------

_ACTION_MAP: dict[str, EvidenceAction] = {
    "WRITE": EvidenceAction.APPLY,
    "APPLY": EvidenceAction.APPLY,
    "CREATE": EvidenceAction.APPLY,
    "DELETE": EvidenceAction.ROLLBACK,
    "ROLLBACK": EvidenceAction.ROLLBACK,
    "PROPOSE": EvidenceAction.PROPOSE,
    "VERIFY": EvidenceAction.VERIFY,
    "RESOLVE": EvidenceAction.RESOLVE,
    "DRIFT": EvidenceAction.DRIFT,
    "CONTRADICT": EvidenceAction.CONTRADICT,
}


def _map_action(action: str) -> EvidenceAction:
    """Map a free-form action string to an EvidenceAction enum value."""
    upper = action.upper()
    return _ACTION_MAP.get(upper, EvidenceAction.APPLY)
