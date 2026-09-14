# Copyright 2026 STARGA, Inc.
"""Explicit content lifetimes, independent of recall/access frequency.

This is a read-only policy: it never deletes a fact, renews a date, grants
admission, or promotes trust. ``ContentCategory`` and ``ContentValidFrom``
are ordinary governed block fields. Renewing them requires the same reviewed
mutation as renewing the fact. Storage kind and lineage edge kind are separate
taxonomies and deliberately cannot select a content lifetime.
"""

from __future__ import annotations

import os
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


ContentIdentity = tuple[str, str, str]


def content_identity(block: Mapping[str, Any]) -> ContentIdentity | None:
    """Return the source-bound identity; IDs alone are not globally unique."""
    block_id = block.get("_id") or block.get("id")
    source = block.get("_source_file") or block.get("_source") or block.get("file")
    if not isinstance(block_id, str) or not block_id or not isinstance(source, str) or not source:
        return None
    normalized = source.replace("\\", "/")
    parts = normalized.split("/")
    if not normalized or os.path.isabs(normalized) or any(part in {"", ".", ".."} for part in parts):
        return None
    normalized = "/".join(parts)
    from .namespace_retrieval import namespace_for_path

    return (namespace_for_path(normalized), normalized, block_id)


def _safe_source_path(workspace: str, source: str) -> str | None:
    """Resolve one corpus source without following a symlinked path."""
    if not isinstance(source, str) or not source or "\x00" in source:
        return None
    normalized = source.replace("\\", "/")
    parts = normalized.split("/")
    if os.path.isabs(normalized) or not normalized or any(part in {"", ".", ".."} for part in parts):
        return None
    root = os.path.realpath(workspace)
    candidate = os.path.abspath(os.path.join(root, *parts))
    if not candidate.startswith(root + os.sep) or os.path.realpath(candidate) != candidate:
        return None
    return candidate if os.path.isfile(candidate) else None


def _identity_blocks(blocks: list[dict[str, Any]], workspace: str) -> dict[ContentIdentity, dict[str, Any]]:
    """Re-read the exact source files named by already ACL-selected rows."""
    from .block_parser import parse_file

    sources: dict[str, str] = {}
    for block in blocks:
        source = block.get("_source_file") or block.get("_source") or block.get("file")
        if isinstance(source, str):
            path = _safe_source_path(workspace, source)
            if path is not None:
                sources.setdefault(path, source.replace("\\", "/"))
    out: dict[ContentIdentity, dict[str, Any]] = {}
    for path, source in sources.items():
        try:
            parsed = parse_file(path)
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        for row in parsed:
            row["_source_file"] = source
            identity = content_identity(row)
            if identity is not None and identity not in out:
                out[identity] = row
    return out


def live_content_blocks(
    workspace: str,
    *,
    active_only: bool = False,
    blocks: list[dict[str, Any]] | None = None,
) -> dict[ContentIdentity, dict[str, Any]]:
    """Read canonical fields bound to ``(namespace, source, id)``.

    ``blocks`` is used by a serving path that already performed ACL selection;
    it re-reads only those named sources and never discovers another agent's
    namespace.  The no-argument maintenance path retains the backend's normal
    workspace enumeration.  A plain ID map would let duplicate IDs in two
    namespace sources inherit the wrong date or revocation state.
    """
    if blocks is not None:
        return _identity_blocks(blocks, workspace)
    from .request_context import context_config_for
    from .storage import iter_blocks

    bound = context_config_for(workspace)
    config = dict(bound) if bound is not None else None
    rows = iter_blocks(workspace, config=config, active_only=active_only)
    out: dict[ContentIdentity, dict[str, Any]] = {}
    for block in rows:
        identity = content_identity(block)
        if identity is not None and identity not in out:
            out[identity] = block
    return out


def content_block_for(records: Mapping[ContentIdentity, dict[str, Any]], block: Mapping[str, Any] | str) -> dict[str, Any]:
    """Look up a source-bound row, with a unique-ID fallback for old callers."""
    if isinstance(block, str):
        block_id = block
        identity = None
    else:
        block_id = str(block.get("_id") or block.get("id") or "")
        identity = content_identity(block)
    if identity is not None:
        return records.get(identity, {})
    if isinstance(block, Mapping) and any(block.get(key) not in (None, "") for key in ("_source_file", "_source", "file")):
        # A caller supplied a source claim, but it is malformed or absent
        # from the live corpus.  Falling back by ID would let a forged path
        # borrow another namespace's lifecycle state.
        return {}
    matches = [row for key, row in records.items() if key[2] == block_id]
    return matches[0] if len(matches) == 1 else {}


def filter_revoked_credentials(items: list[dict], workspace: str) -> list[dict]:
    """A revoked credential is withheld even when historical statuses are served.

    Read the current governed category/status rather than trusting stale index
    fields. Other historical facts remain queryable under the existing policy.
    This cannot turn any withheld item into admitted content.
    """
    if not items or workspace_policy(workspace) is None:
        return items
    blocks = live_content_blocks(workspace)
    blocks.update(live_content_blocks(workspace, blocks=items))
    kept: list[dict] = []
    for item in items:
        current = content_block_for(blocks, item)
        source_claimed = any(item.get(key) not in (None, "") for key in ("_source_file", "_source", "file"))
        if source_claimed and not current:
            # Cached/indexed rows are only trustworthy when the claimed source
            # can be re-read and bound to a live parsed row.  Refuse unresolved
            # source claims even when stale metadata omits ContentCategory.
            continue
        # A credential without a source identity cannot be checked against
        # the governing bytes.  Withhold it rather than allowing an indexed
        # row with forged/missing metadata to bypass revocation.
        if item.get("ContentCategory") == "credential" and content_identity(item) is None:
            continue
        if current.get("ContentCategory") == "credential" and str(current.get("Status", "")).strip().lower() == "revoked":
            continue
        if content_identity(item) is None and not current:
            # An ID-only row is ambiguous when two sources share an ID.  Any
            # revoked credential among those candidates makes the row unsafe
            # to serve; never resolve the ambiguity in the permissive way.
            candidates = [row for key, row in blocks.items() if key[2] == str(item.get("_id") or "")]
            if any(
                row.get("ContentCategory") == "credential"
                and str(row.get("Status", "")).strip().lower() == "revoked"
                for row in candidates
            ):
                continue
        kept.append(item)
    return kept
