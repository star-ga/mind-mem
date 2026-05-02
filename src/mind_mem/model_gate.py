"""Load-gate registry for ``mm audit-model`` checkpoints.

Companion to :mod:`mind_mem.model_audit`, :mod:`mind_mem.model_signing`,
and :mod:`mind_mem.model_provenance`. A "load gate" is the policy
boundary every local model must clear before mind-mem's extractor /
embedding backends will consume it.

Threat model: the operator points mind-mem at a local checkpoint
directory (HF layout). Without a gate, the extractor backend
(``backends.transformers``) calls ``AutoModel.from_pretrained`` on a
path that may have been swapped under the operator since the last
audit. The gate refuses to load a path that:

  1. has never been audited, OR
  2. has drifted from its last-known manifest (file edits since the
     audit), OR
  3. last audited and the audit failed.

An explicit ``trust_without_audit=True`` flag is provided so an
operator can force-load a checkpoint they know is safe (e.g. for
recovery / development) — this still records the override in the
gate ledger so the WARNING is auditable.

Registry format (``~/.mind-mem/model_gate.json``)::

    {
      "/abs/path/to/checkpoint": {
        "audited_at": "2026-05-02T18:00:00Z",
        "manifest_sha256": "<64-hex>",
        "audit_passed": true,
        "audit_report_summary": {"checks_failed": [], "file_count": 12},
        "trust_without_audit": false
      },
      ...
    }

Atomic writes (write-temp + rename) so a crash mid-update never
leaves a half-written ledger. The registry path is overridable via
``MIND_MEM_GATE_REGISTRY`` for tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reasons returned in ``GateDecision.reason``. Strings (not enums) so
# they round-trip through JSON cleanly.
REASON_TRUSTED_FRESH = "trusted_fresh"
REASON_AUDITED_NOW = "audited_now"
REASON_DRIFT_RE_AUDITED = "drift_re_audited"
REASON_AUDIT_FAILED = "audit_failed"
REASON_AUDIT_FAILED_OVERRIDE = "audit_failed_override"
REASON_NEVER_AUDITED_OVERRIDE = "never_audited_override"
REASON_PATH_NOT_FOUND = "path_not_found"

DEFAULT_REGISTRY_FILENAME = "model_gate.json"


@dataclass
class GateDecision:
    """Outcome of a ``gate_check`` call.

    ``passed`` is True when the load is allowed; the ``reason`` field
    distinguishes between "audited and clean" (best case),
    "audited-now-and-clean", "drift forced re-audit", and the explicit
    override paths (``trust_without_audit=True``).
    """

    passed: bool
    reason: str
    path: str
    manifest_sha256: str
    audit_passed: bool | None = None
    audit_summary: dict[str, Any] = field(default_factory=dict)


def _registry_path() -> Path:
    """Return the gate-registry path, honouring ``MIND_MEM_GATE_REGISTRY``."""
    override = os.environ.get("MIND_MEM_GATE_REGISTRY", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    home = Path(os.path.expanduser("~/.mind-mem"))
    return home / DEFAULT_REGISTRY_FILENAME


def _load_registry() -> dict[str, dict[str, Any]]:
    """Read the registry from disk; return empty dict on missing/corrupt."""
    p = _registry_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    # Drop entries with the wrong shape so a hand-edited file can't
    # poison subsequent reads.
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _save_registry(reg: dict[str, dict[str, Any]]) -> None:
    """Atomically write the registry — write-temp + os.replace."""
    p = _registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".gate.", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(reg, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, p)
    except Exception:
        # Clean up the temp file on any error so we don't leak it.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _now_iso() -> str:
    """Timezone-aware UTC ISO timestamp; the ``Z`` suffix matches the
    rest of the mind-mem evidence chain."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _compute_manifest_sha256(root: Path) -> str:
    """SHA-256 of the deterministic manifest_text — uniquely identifies a
    checkpoint's file set so any drift shows up as a different digest.
    Uses :func:`mind_mem.model_signing.compute_manifest_text` so the
    digest is consistent with the signing path.
    """
    from mind_mem.model_signing import compute_manifest_text

    text, _ = compute_manifest_text(root)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _summarise_audit(report: Any) -> dict[str, Any]:
    """Compact audit report summary stored in the registry.

    We don't keep the full report (it can include large evidence
    lists). The summary is enough to render an explanation in
    ``mm gate list`` without re-running the audit.
    """
    failed = [c.name for c in report.checks if not c.passed]
    return {
        "checks_failed": failed,
        "file_count": report.file_count,
        "total_bytes": report.total_bytes,
    }


def gate_check(
    path: str | Path,
    *,
    trust_without_audit: bool = False,
    allow_extra_publishers: tuple[str, ...] | None = None,
) -> GateDecision:
    """Decide whether mind-mem should consume a checkpoint at ``path``.

    Behaviour:
      * If the path doesn't exist → ``passed=False``,
        ``reason=path_not_found``.
      * If the path is in the registry, the recorded ``manifest_sha256``
        matches the current one, and ``audit_passed`` is True →
        ``passed=True``, ``reason=trusted_fresh``.
      * If the path is in the registry but the recorded sha256
        differs → re-run the audit, persist the new entry, return
        ``reason=drift_re_audited``.
      * If the path isn't in the registry → run the audit, persist
        the result, return ``reason=audited_now``.
      * If the audit fails and ``trust_without_audit=False`` →
        ``passed=False``, ``reason=audit_failed``.
      * If the audit fails and ``trust_without_audit=True`` →
        ``passed=True``, ``reason=audit_failed_override`` and the
        registry records ``trust_without_audit=True`` (auditable).
      * If the path is never-audited and ``trust_without_audit=True``
        is passed → registry records the override, returns
        ``reason=never_audited_override``.
    """
    from mind_mem.model_audit import audit_model

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        return GateDecision(
            passed=False,
            reason=REASON_PATH_NOT_FOUND,
            path=str(root),
            manifest_sha256="",
        )

    current_sha = _compute_manifest_sha256(root)
    reg = _load_registry()
    entry = reg.get(str(root))

    if entry and entry.get("manifest_sha256") == current_sha and entry.get("audit_passed"):
        # Fast path — checkpoint is unchanged since the last clean
        # audit. No need to re-hash the world.
        return GateDecision(
            passed=True,
            reason=REASON_TRUSTED_FRESH,
            path=str(root),
            manifest_sha256=current_sha,
            audit_passed=True,
            audit_summary=entry.get("audit_report_summary", {}),
        )

    # Either no entry OR the manifest drifted OR the prior audit
    # failed — re-audit and update the registry.
    drift = entry is not None and entry.get("manifest_sha256") != current_sha

    if not trust_without_audit:
        report = audit_model(root, allow_extra_publishers=allow_extra_publishers)
        summary = _summarise_audit(report)
        passed = report.passed
        new_entry: dict[str, Any] = {
            "audited_at": _now_iso(),
            "manifest_sha256": current_sha,
            "audit_passed": passed,
            "audit_report_summary": summary,
            "trust_without_audit": False,
        }
        reg[str(root)] = new_entry
        _save_registry(reg)
        if passed:
            reason = REASON_DRIFT_RE_AUDITED if drift else REASON_AUDITED_NOW
        else:
            reason = REASON_AUDIT_FAILED
        return GateDecision(
            passed=passed,
            reason=reason,
            path=str(root),
            manifest_sha256=current_sha,
            audit_passed=passed,
            audit_summary=summary,
        )

    # Override path — the operator forced load without an audit.
    # Either there's no entry yet (never_audited_override) or the
    # last entry failed (audit_failed_override). Distinguish so the
    # ledger reflects intent.
    if entry is None or entry.get("audit_passed") is None:
        reason = REASON_NEVER_AUDITED_OVERRIDE
        audit_passed: bool | None = None
        summary = {}
    else:
        reason = REASON_AUDIT_FAILED_OVERRIDE
        audit_passed = bool(entry.get("audit_passed"))
        summary = entry.get("audit_report_summary", {})

    reg[str(root)] = {
        "audited_at": _now_iso(),
        "manifest_sha256": current_sha,
        "audit_passed": audit_passed,
        "audit_report_summary": summary,
        "trust_without_audit": True,
    }
    _save_registry(reg)
    return GateDecision(
        passed=True,
        reason=reason,
        path=str(root),
        manifest_sha256=current_sha,
        audit_passed=audit_passed,
        audit_summary=summary,
    )


def gate_list() -> list[dict[str, Any]]:
    """Return the registry as a JSON-serialisable list (path included)."""
    reg = _load_registry()
    return [{"path": k, **v} for k, v in sorted(reg.items())]


def gate_remove(path: str | Path) -> bool:
    """Remove ``path`` from the registry. Return True iff something was removed."""
    root = str(Path(path).expanduser().resolve())
    reg = _load_registry()
    if root not in reg:
        return False
    del reg[root]
    _save_registry(reg)
    return True
