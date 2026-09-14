# Copyright 2026 STARGA, Inc.
"""Explicit content lifetimes, independent of recall/access frequency.

This is a read-only policy: it never deletes a fact, renews a date, grants
admission, or promotes trust. ``ContentCategory`` and ``ContentValidFrom``
are ordinary governed block fields. Renewing them requires the same reviewed
mutation as renewing the fact. Storage kind and lineage edge kind are separate
taxonomies and deliberately cannot select a content lifetime.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

DURABLE_CATEGORIES = frozenset({"decision", "architecture", "credential"})
EXPIRING_CATEGORIES = frozenset({"infra", "status"})
CATEGORIES = DURABLE_CATEGORIES | EXPIRING_CATEGORIES
_ID_DATE = re.compile(r"^[A-Z]+-(\d{4})(\d{2})(\d{2})-\d{3}(?:$|[-.])")


@dataclass(frozen=True)
class ContentLifetime:
    category: str
    state: str
    valid_from: str | None = None
    age_days: int | None = None
    ttl_days: int | None = None

    @property
    def needs_review(self) -> bool:
        return self.state in {"stale", "invalid_category", "invalid_date", "future_date"}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContentLifecyclePolicy:
    """Opt-in, whole-day UTC lifetimes; there are no guessed numeric defaults."""

    ttl_days: Mapping[str, int]

    @classmethod
    def from_recall_config(cls, cfg: Mapping[str, Any]) -> ContentLifecyclePolicy | None:
        gate = cfg.get("validity_gate", {})
        if not isinstance(gate, Mapping):
            return None
        raw = gate.get("content_categories", {})
        if not isinstance(raw, Mapping):
            raise ValueError("validity_gate.content_categories must be an object")
        enabled = raw.get("enabled", False)
        if type(enabled) is not bool:
            raise ValueError("content_categories.enabled must be a boolean")
        if not enabled:
            return None
        if gate.get("enabled") is not True:
            raise ValueError("content_categories requires validity_gate.enabled=true")
        lifetimes = raw.get("ttl_days")
        if not isinstance(lifetimes, Mapping) or set(lifetimes) != EXPIRING_CATEGORIES:
            raise ValueError("content_categories.ttl_days must declare exactly infra and status")
        if any(type(value) is not int or not 1 <= value <= 36500 for value in lifetimes.values()):
            raise ValueError("content category TTLs must be integer days in [1, 36500]")
        return cls(dict(lifetimes))

    def evaluate(self, block: Mapping[str, Any], *, as_of: date) -> ContentLifetime | None:
        """Evaluate semantic time, never access time, mtime or confirmations.

        Missing category keeps historical behavior. An explicitly invalid
        category/date needs review rather than becoming fresh by omission.
        A durable category bypasses time decay, but does not override status,
        contradiction, provenance or any other admission/validity decision.
        """
        raw = block.get("ContentCategory")
        if raw is None:
            return None
        if not isinstance(raw, str) or raw not in CATEGORIES:
            return ContentLifetime("unknown", "invalid_category")
        if raw in DURABLE_CATEGORIES:
            return ContentLifetime(raw, "durable")
        ttl = self.ttl_days[raw]
        # An explicit malformed renewal MUST NOT fall back to an older date.
        stamp = block.get("ContentValidFrom", block.get("Date"))
        if stamp is None:
            match = _ID_DATE.match(str(block.get("_id", "")))
            stamp = "-".join(match.groups()) if match else None
        try:
            if not isinstance(stamp, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", stamp):
                raise ValueError("content date must be YYYY-MM-DD")
            start = date.fromisoformat(stamp)
        except ValueError:
            return ContentLifetime(raw, "invalid_date", ttl_days=ttl)
        age = (as_of - start).days
        state = "future_date" if age < 0 else "stale" if age >= ttl else "current"
        return ContentLifetime(raw, state, start.isoformat(), age, ttl)


def workspace_policy(workspace: str) -> ContentLifecyclePolicy | None:
    """Consume the same request-bound configuration as retrieval."""
    from .request_context import context_config_for
    from .storage import _load_workspace_config

    config = context_config_for(workspace)
    if config is None:
        config = _load_workspace_config(workspace)
    recall = config.get("recall", {})
    return ContentLifecyclePolicy.from_recall_config(recall) if isinstance(recall, Mapping) else None


def live_content_blocks(workspace: str, *, active_only: bool = False) -> dict[str, dict[str, Any]]:
    """Canonical fields, so cached/indexed metadata cannot renew a fact."""
    from .request_context import context_config_for
    from .storage import iter_blocks

    bound = context_config_for(workspace)
    config = dict(bound) if bound is not None else None
    return {str(block["_id"]): block for block in iter_blocks(workspace, config=config, active_only=active_only) if block.get("_id")}


def filter_revoked_credentials(items: list[dict], workspace: str) -> list[dict]:
    """A revoked credential is withheld even when historical statuses are served.

    Read the current governed category/status rather than trusting stale index
    fields. Other historical facts remain queryable under the existing policy.
    This cannot turn any withheld item into admitted content.
    """
    if not items or workspace_policy(workspace) is None:
        return items
    blocks = live_content_blocks(workspace)
    revoked = {
        block_id
        for block_id, block in blocks.items()
        if block.get("ContentCategory") == "credential" and str(block.get("Status", "")).strip().lower() == "revoked"
    }
    return [item for item in items if item.get("_id") not in revoked]
