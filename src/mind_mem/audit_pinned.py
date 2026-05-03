"""Pinned-model audit pipeline — release-CI gate for ``mind-mem.json``.

Companion to :mod:`mind_mem.model_audit`, :mod:`mind_mem.model_signing`,
and :mod:`mind_mem.model_gate`. The pinned-model pipeline reads an
``audit_pinned_models`` list from ``mind-mem.json`` and runs the full
seven-check audit (and optional Ed25519 verification) against every
entry. Any HIGH finding or verify failure produces a non-zero exit so a
CI workflow can fail the build before a release ships with an
unaudited / drifted / failed checkpoint.

Pin schema (extension to ``mind-mem.json``)::

    {
      "audit_pinned_models": [
        "/abs/or/relative/path",
        {
          "path": "/another/path",
          "verify": true,
          "allow_publishers": ["my-internal-org"]
        }
      ]
    }

Both shapes are accepted: a bare string is treated as a path with
default options (no verify, no extra publishers); an object form
declares per-entry overrides. The list may be empty — a config file
with ``"audit_pinned_models": []`` is a no-op pass and a config file
without the key at all is the same.

Exit-code contract (used by ``mm audit-pinned``):

* ``0`` — every pinned model passed audit (and verify, when requested);
  also returned when the config file is missing or the list is empty.
* ``1`` — at least one model produced a HIGH finding or failed verify.
* ``2`` — config could not be parsed, or a pinned path is missing while
  ``--fail-on-missing`` was set.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Re-import upstream modules lazily inside the audit function so that
# importing this module is cheap and doesn't drag in the full audit
# pipeline at CLI parse time.


@dataclass(frozen=True)
class PinnedModel:
    """One entry from ``audit_pinned_models``.

    ``path`` is the (possibly-relative) path to the checkpoint. The
    ``verify`` flag opts the entry into Ed25519 verification on top of
    audit. ``allow_publishers`` extends the canonical provenance
    allowlist with operator-specific HF org slugs.
    """

    path: str
    verify: bool = False
    allow_publishers: tuple[str, ...] = ()


@dataclass
class PinnedAuditFinding:
    """Outcome of auditing one pinned model."""

    path: str
    exists: bool = False
    audit_passed: bool = False
    audit_summary: dict[str, Any] = field(default_factory=dict)
    verify_attempted: bool = False
    verify_passed: bool = False
    verify_error_kind: str | None = None
    verify_error_detail: str | None = None
    error: str | None = None  # populated when path missing / unreadable


@dataclass
class PinnedAuditReport:
    """Aggregate report covering every pinned model."""

    findings: list[PinnedAuditFinding] = field(default_factory=list)
    config_present: bool = False

    @property
    def passed(self) -> bool:
        """True iff every finding is clean. Empty report passes."""
        return all(self._finding_passed(f) for f in self.findings)

    @staticmethod
    def _finding_passed(f: PinnedAuditFinding) -> bool:
        if f.error and f.exists is False:
            # A missing path on its own does not fail by default —
            # ``--fail-on-missing`` is the operator's choice. The CLI
            # consults that flag separately. Here we only check
            # actual audit / verify outcomes.
            return True
        if not f.audit_passed:
            return False
        if f.verify_attempted and not f.verify_passed:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_present": self.config_present,
            "passed": self.passed,
            "findings": [
                {
                    "path": f.path,
                    "exists": f.exists,
                    "audit_passed": f.audit_passed,
                    "audit_summary": f.audit_summary,
                    "verify_attempted": f.verify_attempted,
                    "verify_passed": f.verify_passed,
                    "verify_error_kind": f.verify_error_kind,
                    "verify_error_detail": f.verify_error_detail,
                    "error": f.error,
                }
                for f in self.findings
            ],
        }


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class PinnedConfigError(ValueError):
    """Raised on schema violations in the ``audit_pinned_models`` list."""


def load_pinned_models(config_path: str | os.PathLike[str]) -> list[PinnedModel]:
    """Read and validate the ``audit_pinned_models`` list from a
    ``mind-mem.json``.

    Returns ``[]`` if the file doesn't exist OR doesn't contain the
    ``audit_pinned_models`` key (so a CI hook can no-op cleanly when
    the operator hasn't opted into pinning yet). Raises
    :class:`PinnedConfigError` when the file is unreadable or the list
    contains entries the schema can't make sense of — that's a real
    misconfiguration the operator wants to see.
    """
    p = Path(config_path).expanduser()
    if not p.is_file():
        return []
    try:
        cfg = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PinnedConfigError(f"could not read {p}: {exc}") from exc

    if not isinstance(cfg, dict):
        raise PinnedConfigError(f"{p}: top-level value must be a JSON object")

    raw = cfg.get("audit_pinned_models")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise PinnedConfigError(f"{p}: 'audit_pinned_models' must be a JSON array")

    out: list[PinnedModel] = []
    for i, item in enumerate(raw):
        out.append(_normalise_entry(item, i, source=str(p)))
    return out


def _normalise_entry(item: Any, idx: int, *, source: str) -> PinnedModel:
    """Accept either a bare path string or an object with optional
    ``verify`` / ``allow_publishers`` fields."""
    if isinstance(item, str):
        return PinnedModel(path=item)
    if not isinstance(item, dict):
        raise PinnedConfigError(f"{source}: audit_pinned_models[{idx}] must be a string or object, got {type(item).__name__}")
    path = item.get("path")
    if not isinstance(path, str) or not path:
        raise PinnedConfigError(f"{source}: audit_pinned_models[{idx}] missing required 'path' field")
    verify = bool(item.get("verify", False))
    allow = item.get("allow_publishers", [])
    if not isinstance(allow, list) or not all(isinstance(s, str) for s in allow):
        raise PinnedConfigError(f"{source}: audit_pinned_models[{idx}].allow_publishers must be a list of strings")
    return PinnedModel(
        path=path,
        verify=verify,
        allow_publishers=tuple(allow),
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def audit_pinned(
    config_path: str | os.PathLike[str] = "mind-mem.json",
    *,
    workspace: str | os.PathLike[str] = ".",
) -> PinnedAuditReport:
    """Audit every entry in ``audit_pinned_models``. Relative paths are
    resolved against ``workspace`` so the operator can pin paths
    relative to the repo root regardless of the CI runner's cwd.

    Empty list (or missing config) → empty report, ``passed=True``.
    """
    from mind_mem.model_audit import audit_model
    from mind_mem.model_signing import verify_model

    pinned = load_pinned_models(config_path)
    report = PinnedAuditReport(config_present=Path(config_path).is_file())
    workspace_path = Path(workspace).expanduser()

    for entry in pinned:
        finding = PinnedAuditFinding(path=entry.path)
        target = Path(entry.path).expanduser()
        if not target.is_absolute():
            target = (workspace_path / target).resolve()
        finding.exists = target.is_dir()

        if not finding.exists:
            finding.error = f"pinned path is not an existing directory: {target}"
            report.findings.append(finding)
            continue

        try:
            audit_report = audit_model(
                target,
                allow_extra_publishers=entry.allow_publishers or None,
            )
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            finding.error = f"audit_model failed: {exc}"
            report.findings.append(finding)
            continue

        finding.audit_passed = audit_report.passed
        finding.audit_summary = _summarise_audit(audit_report)

        if entry.verify:
            finding.verify_attempted = True
            try:
                verify_result = verify_model(target)
            except (FileNotFoundError, NotADirectoryError) as exc:
                finding.verify_error_kind = "missing_file"
                finding.verify_error_detail = str(exc)
            else:
                finding.verify_passed = verify_result.passed
                if not verify_result.passed:
                    finding.verify_error_kind = verify_result.error_kind
                    finding.verify_error_detail = verify_result.error_detail

        report.findings.append(finding)

    return report


def _summarise_audit(report: Any) -> dict[str, Any]:
    """Reduce a full audit report to the fields the pinned-audit cares
    about — counts of HIGH / MEDIUM / LOW findings + the names of the
    failed checks (if any)."""
    levels: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    failed_checks: list[str] = []
    for check in getattr(report, "checks", []):
        if not getattr(check, "passed", True):
            failed_checks.append(getattr(check, "name", "<unnamed>"))
            for finding in getattr(check, "findings", []):
                level = getattr(finding, "level", "LOW")
                levels[level] = levels.get(level, 0) + 1
    return {
        "checks_failed": failed_checks,
        "findings_high": levels.get("HIGH", 0),
        "findings_medium": levels.get("MEDIUM", 0),
        "findings_low": levels.get("LOW", 0),
        "file_count": len(getattr(report, "manifest", {})),
    }


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------


def format_pinned_report_text(report: PinnedAuditReport) -> str:
    """Operator-facing text summary. CI logs render this directly so the
    failure cause is obvious without parsing JSON."""
    lines: list[str] = []
    if not report.findings:
        if report.config_present:
            lines.append("audit-pinned: no models pinned (audit_pinned_models is empty or absent)")
        else:
            lines.append("audit-pinned: no mind-mem.json found — nothing to do")
        return "\n".join(lines)

    lines.append(f"audit-pinned: {len(report.findings)} model(s) checked")
    for f in report.findings:
        if f.error and not f.exists:
            lines.append(f"  SKIP {f.path}  ({f.error})")
            continue
        if f.error:
            lines.append(f"  FAIL {f.path}  ({f.error})")
            continue
        marker = "PASS" if f.audit_passed else "FAIL"
        summary = f.audit_summary
        lines.append(
            f"  {marker} {f.path}  "
            f"(HIGH={summary.get('findings_high', 0)} "
            f"MEDIUM={summary.get('findings_medium', 0)} "
            f"LOW={summary.get('findings_low', 0)})"
        )
        if not f.audit_passed and summary.get("checks_failed"):
            for c in summary["checks_failed"]:
                lines.append(f"      failed check: {c}")
        if f.verify_attempted:
            v_marker = "PASS" if f.verify_passed else "FAIL"
            lines.append(f"      verify {v_marker}" + (f"  [{f.verify_error_kind}] {f.verify_error_detail}" if not f.verify_passed else ""))

    overall = "PASS" if report.passed else "FAIL"
    lines.append(f"overall: {overall}")
    return "\n".join(lines)


__all__ = [
    "PinnedAuditFinding",
    "PinnedAuditReport",
    "PinnedConfigError",
    "PinnedModel",
    "audit_pinned",
    "format_pinned_report_text",
    "load_pinned_models",
]
