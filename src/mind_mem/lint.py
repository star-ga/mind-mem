# Copyright 2026 STARGA, Inc.
"""Deterministic corpus lint — stable findings for the governed repair path.

``lint(workspace)`` walks the Markdown corpus and returns an immutable
tuple of :class:`Finding` records. Every finding carries a **stable,
content-addressed id** (``LF-xxxxxxxx``) so a caller can hand exactly
one finding back to :func:`mind_mem.lint_autofix.lint_autofix` and get
a repair *proposal* — never a direct write.

Three rules ship here, one per finding class:

``stale_date``
    The block's ``Date:`` field disagrees with the date anchor encoded
    in its canonical id (``D-YYYYMMDD-NNN``). The id is the immutable
    anchor, so the repair rewrites ``Date`` to match it.

``missing_metadata``
    A schema-required field (per ``validate_py``'s required-field list)
    is present but empty. Repair is offered only for fields with a
    deterministic, content-free default — the lint never invents prose.

``duplicate_block``
    Two or more ``active`` decision blocks assert the same normalised
    statement. The lowest id wins; each later twin gets a repair that
    marks it ``superseded``.

Findings are **advisory data**. This module reads the corpus and writes
nothing at all; the whole write path lives behind governance in
``lint_autofix``.

The surface is gated by the ``v4.lint`` feature flag (default OFF), so
an existing workspace sees identical behaviour until the flag is set.

Zero external deps — stdlib plus ``block_parser``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Final, Mapping, Sequence

from .block_parser import parse_file
from .v4.feature_flags import FeatureDisabledError, is_enabled

__all__ = [
    "Finding",
    "FeatureDisabledError",
    "LintError",
    "RULES",
    "RULE_DUPLICATE_BLOCK",
    "RULE_MISSING_METADATA",
    "RULE_STALE_DATE",
    "find_finding",
    "lint",
]

RULE_STALE_DATE: Final = "stale_date"
RULE_MISSING_METADATA: Final = "missing_metadata"
RULE_DUPLICATE_BLOCK: Final = "duplicate_block"

RULES: Final[tuple[str, ...]] = (
    RULE_STALE_DATE,
    RULE_MISSING_METADATA,
    RULE_DUPLICATE_BLOCK,
)

#: Corpus files this lint understands, with their id prefix. Mirrors the
#: block families ``validate_py`` validates.
_LINTED_FILES: Final[tuple[tuple[str, str], ...]] = (
    ("decisions/DECISIONS.md", "D"),
    ("tasks/TASKS.md", "T"),
)

#: Schema-required fields per id prefix — copied from ``validate_py``'s
#: ``_check_decisions`` / ``_check_tasks`` required lists.
_REQUIRED_FIELDS: Final[Mapping[str, tuple[str, ...]]] = {
    "D": ("Date", "Status", "Scope", "Statement", "Rationale", "Supersedes", "Tags", "Sources"),
    "T": ("Date", "Title", "Status", "Priority", "Project", "Owner", "Sources"),
}

#: The only defaults the lint is willing to propose. Both are closed-set
#: schema sentinels, not content — inventing a ``Statement`` or a
#: ``Rationale`` would be fabricating memory, so those stay unfixable.
_FIELD_DEFAULTS: Final[Mapping[str, str]] = {
    "Supersedes": "none",
    "Scope": "global",
}

_ID_ANCHOR_RE: Final = re.compile(r"^[A-Z]+-(\d{4})(\d{2})(\d{2})-\d{3}$")
_WS_RE: Final = re.compile(r"\s+")
_PUNCT_RE: Final = re.compile(r"[^\w\s]+")
_FINDING_ID_RE: Final = re.compile(r"^LF-[0-9a-f]{8}$")


class LintError(RuntimeError):
    """Raised when the lint cannot run against the given workspace."""


@dataclass(frozen=True)
class Finding:
    """One deterministic corpus defect.

    ``repair`` is the apply-engine op that fixes it, held as an
    immutable tuple of ``(key, value)`` pairs; it is empty when the
    defect has no content-free repair (``autofixable`` is then False).
    """

    finding_id: str
    rule: str
    block_id: str
    file: str
    detail: str
    autofixable: bool
    repair: tuple[tuple[str, str], ...] = ()

    def repair_op(self) -> dict[str, str]:
        """Return the repair as a fresh apply-engine op dict."""
        return dict(self.repair)

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly projection (used by callers and tests)."""
        return {
            "finding_id": self.finding_id,
            "rule": self.rule,
            "block_id": self.block_id,
            "file": self.file,
            "detail": self.detail,
            "autofixable": self.autofixable,
            "repair": self.repair_op(),
        }


def _finding_id(rule: str, rel_path: str, block_id: str, key: str) -> str:
    """Content-addressed, stable across runs and machines."""
    canon = json.dumps([rule, rel_path, block_id, key], sort_keys=True)
    return "LF-" + hashlib.sha256(canon.encode("utf-8")).hexdigest()[:8]


def is_finding_id(candidate: str) -> bool:
    """True when *candidate* has the ``LF-xxxxxxxx`` shape."""
    return bool(_FINDING_ID_RE.match(candidate))


def _anchor_date(block_id: str) -> str | None:
    """``D-20260101-001`` -> ``2026-01-01``; None when not a real date."""
    m = _ID_ANCHOR_RE.match(block_id)
    if not m:
        return None
    year, month, day = (int(part) for part in m.groups())
    try:
        return _dt.date(year, month, day).isoformat()
    except ValueError:
        return None


def _normalise(text: str) -> str:
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", text.lower())).strip()


def _is_empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return not value
    return False


def _scan_stale_date(blocks: Sequence[dict], rel_path: str) -> list[Finding]:
    out: list[Finding] = []
    for block in blocks:
        block_id = str(block.get("_id", ""))
        anchor = _anchor_date(block_id)
        if anchor is None:
            continue
        current = block.get("Date")
        if not isinstance(current, str) or not current.strip():
            continue  # absent/empty Date is a missing_metadata defect
        # deferred: age-based staleness (a Date far in the past) is not
        # reported here because there is no content-free repair for it
        # - upgrade path: report it with ``autofixable=False`` once the
        # age threshold is a workspace config knob.
        if current.strip() == anchor:
            continue
        out.append(
            Finding(
                finding_id=_finding_id(RULE_STALE_DATE, rel_path, block_id, "Date"),
                rule=RULE_STALE_DATE,
                block_id=block_id,
                file=rel_path,
                detail=f"Date '{current.strip()}' drifted from id anchor '{anchor}'",
                autofixable=True,
                repair=(
                    ("op", "update_field"),
                    ("file", rel_path),
                    ("target", block_id),
                    ("field", "Date"),
                    ("value", anchor),
                ),
            )
        )
    return out


def _scan_missing_metadata(blocks: Sequence[dict], rel_path: str, prefix: str) -> list[Finding]:
    out: list[Finding] = []
    required = _REQUIRED_FIELDS.get(prefix, ())
    for block in blocks:
        block_id = str(block.get("_id", ""))
        for field_name in required:
            present = field_name in block
            if present and not _is_empty(block[field_name]):
                continue
            default = _FIELD_DEFAULTS.get(field_name)
            # An absent field line cannot be repaired by ``update_field``
            # (the apply engine only rewrites a field it can find), so
            # only a present-but-empty field is auto-fixable.
            # deferred: a wholly absent field cannot be repaired - upgrade
            # path: add an ``insert_field`` op to apply_engine, then drop
            # the ``present`` conjunct below.
            fixable = bool(default) and present
            repair: tuple[tuple[str, str], ...] = ()
            if fixable:
                repair = (
                    ("op", "update_field"),
                    ("file", rel_path),
                    ("target", block_id),
                    ("field", field_name),
                    ("value", str(default)),
                )
            state = "empty" if present else "absent"
            out.append(
                Finding(
                    finding_id=_finding_id(RULE_MISSING_METADATA, rel_path, block_id, field_name),
                    rule=RULE_MISSING_METADATA,
                    block_id=block_id,
                    file=rel_path,
                    detail=f"required field '{field_name}' is {state}",
                    autofixable=fixable,
                    repair=repair,
                )
            )
    return out


def _scan_duplicate_block(blocks: Sequence[dict], rel_path: str, prefix: str) -> list[Finding]:
    """Duplicate active decisions — the lowest id wins, twins supersede."""
    if prefix != "D":
        # Only decision blocks have a ``superseded`` status; task twins
        # are left to a human.
        return []
    groups: dict[str, list[str]] = {}
    for block in blocks:
        if str(block.get("Status", "")).strip() != "active":
            continue
        statement = block.get("Statement")
        if not isinstance(statement, str):
            continue
        key = _normalise(statement)
        if not key:
            continue
        groups.setdefault(key, []).append(str(block.get("_id", "")))

    out: list[Finding] = []
    for key, ids in groups.items():
        if len(ids) < 2:
            continue
        winner, *twins = sorted(ids)
        for twin in twins:
            out.append(
                Finding(
                    finding_id=_finding_id(RULE_DUPLICATE_BLOCK, rel_path, twin, winner),
                    rule=RULE_DUPLICATE_BLOCK,
                    block_id=twin,
                    file=rel_path,
                    detail=f"duplicate statement of {winner}",
                    autofixable=True,
                    repair=(
                        ("op", "set_status"),
                        ("file", rel_path),
                        ("target", twin),
                        ("status", "superseded"),
                    ),
                )
            )
    return out


def _flag_enabled(workspace: str) -> bool:
    """``v4.lint`` state for *workspace*, falling back to ambient config.

    ``feature_flags.is_enabled`` resolves config from the environment,
    which is right for process-wide surfaces but wrong for an API that
    takes an explicit workspace — so the workspace's own
    ``mind-mem.json`` is consulted first.
    """
    config_path = os.path.join(workspace, "mind-mem.json")
    try:
        with open(config_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return is_enabled("lint")
    block = data.get("v4")
    if isinstance(block, dict):
        sub = block.get("lint")
        if isinstance(sub, dict):
            return sub.get("enabled") is True
    return is_enabled("lint")


def require_lint_enabled(workspace: str) -> None:
    """Raise :class:`FeatureDisabledError` when ``v4.lint`` is OFF."""
    if not _flag_enabled(workspace):
        raise FeatureDisabledError(
            'mind-mem v4 surface \'lint\' is disabled. Enable via mind-mem.json: "v4": { "lint": { "enabled": true } }'
        )


def lint(workspace: str, *, rules: Sequence[str] | None = None) -> tuple[Finding, ...]:
    """Return every finding in *workspace*, deterministically ordered.

    Args:
        workspace: Workspace root. Must exist.
        rules: Optional subset of :data:`RULES`. ``None`` runs them all.

    Raises:
        LintError: the workspace is missing, or an unknown rule is named.
        FeatureDisabledError: the ``v4.lint`` flag is OFF.
    """
    if not workspace or not isinstance(workspace, str):
        raise LintError("workspace must be a non-empty path string")
    ws = os.path.abspath(workspace)
    if not os.path.isdir(ws):
        raise LintError(f"workspace does not exist: {ws}")
    require_lint_enabled(ws)

    selected = tuple(rules) if rules is not None else RULES
    unknown = [r for r in selected if r not in RULES]
    if unknown:
        raise LintError(f"unknown lint rule(s): {sorted(unknown)} (known: {list(RULES)})")

    findings: list[Finding] = []
    for rel_path, prefix in _LINTED_FILES:
        path = os.path.join(ws, rel_path)
        if not os.path.isfile(path):
            continue
        try:
            blocks = parse_file(path)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise LintError(f"failed to parse {rel_path}: {exc}") from exc
        if RULE_STALE_DATE in selected:
            findings.extend(_scan_stale_date(blocks, rel_path))
        if RULE_MISSING_METADATA in selected:
            findings.extend(_scan_missing_metadata(blocks, rel_path, prefix))
        if RULE_DUPLICATE_BLOCK in selected:
            findings.extend(_scan_duplicate_block(blocks, rel_path, prefix))

    findings.sort(key=lambda f: (f.file, f.rule, f.block_id, f.finding_id))
    return tuple(findings)


def find_finding(workspace: str, finding_id: str) -> Finding | None:
    """Return the finding with *finding_id*, or ``None`` when absent."""
    for finding in lint(workspace):
        if finding.finding_id == finding_id:
            return finding
    return None
