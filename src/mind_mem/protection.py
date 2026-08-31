"""Runtime protection layer for mind-mem (v3.3.0+).

Defence-in-depth for the shipped wheel. Not cryptographic protection
against a determined attacker with local code access — Python's model
makes that impossible without native compilation — but raises the cost
of tamper-with-silent-drift attacks.

Layers (each fails open by default; ``MIND_MEM_INTEGRITY=strict`` turns
them into hard faults):

1. **Integrity manifest** — SHA-256 over critical source files baked
   into ``_integrity_manifest.json`` at wheel-build time, verified at
   first import.
2. **License/author stamp** — ``__author__`` and ``__license__``
   constants that downstream consumers can pin with ``assert``.
3. **Tamper telemetry** — when strict mode is off, mismatches emit a
   structured log event (``protection.integrity_mismatch``) so
   governance dashboards can alert.
4. **Import-path guard** — a world-writable package directory on POSIX
   is reported as a warning and, under ``MIND_MEM_INTEGRITY=strict``,
   refuses the import (prevents trivial file-swap).
5. **Frozen constants** — critical thresholds (``AUTH_HEADER``,
   ``AUDIT_TAG``) exposed as immutable module attributes so patches
   visible in source are detectable.

The manifest is *optional* — in development or editable installs there
is no manifest and the hash layer has nothing to check. The remaining
layers still report: an absent manifest is silent, but a manifest that
is present and unreadable is a warning like any other, because deleting
or truncating one must never be a way to turn a strict-mode check into a
pass. Wheels built via the release workflow
(`scripts/build_integrity_manifest.py`) bake the manifest in.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

__author__: Final[str] = "STARGA Inc <noreply@star.ga>"
__license__: Final[str] = "Apache-2.0"
__protection_version__: Final[str] = "1.0"

AUTH_HEADER: Final[str] = "X-MindMem-Token"
AUDIT_TAG: Final[str] = "TAG_v1"

_log = logging.getLogger("mind_mem.protection")

_MANIFEST_FILENAME = "_integrity_manifest.json"
_STRICT_ENV = "MIND_MEM_INTEGRITY"

_CRITICAL_MODULES: Final[tuple[str, ...]] = (
    "recall.py",
    "recall_vector.py",
    "apply_engine.py",
    "audit_chain.py",
    "encryption.py",
    "feature_gate.py",
    "answer_quality.py",
    "truth_score.py",
    "graph_recall.py",
    "entity_prefetch.py",
    "evidence_bundle.py",
    "rerank_ensemble.py",
    "consensus_vote.py",
    "tenant_audit.py",
    "tenant_kms.py",
    "governance_raft.py",
    "query_planner.py",
    "session_boost.py",
    "trust_scores.py",
    "provenance_class.py",
    "validity_gate.py",
    "storage/sharded_pg.py",
)


@dataclass(frozen=True)
class IntegrityReport:
    ok: bool
    mode: str
    manifest_present: bool
    checked: int
    mismatched: tuple[str, ...] = field(default_factory=tuple)
    missing: tuple[str, ...] = field(default_factory=tuple)
    extra: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _strict() -> bool:
    return os.environ.get(_STRICT_ENV, "").lower() in {"1", "strict", "true", "yes"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def _manifest_path() -> Path:
    return _package_root() / _MANIFEST_FILENAME


def _load_manifest() -> tuple[dict[str, str] | None, str]:
    """Load the baked integrity manifest.

    Returns ``(files, error)``. ``(None, "")`` means no manifest is baked
    in — the ordinary editable install or source checkout. A non-empty
    error means a manifest **is** present and could not be read; that is
    an attack shape, not a dev convenience (deleting or truncating the
    manifest must not be a way to turn a strict-mode check into a pass),
    so it is reported rather than folded into the absent case.
    """
    path = _manifest_path()
    if not path.is_file():
        return None, ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"integrity manifest present but unreadable: {type(exc).__name__}"
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, dict):
        return None, "integrity manifest present but malformed: no 'files' mapping"
    return {k: v for k, v in files.items() if isinstance(v, str)}, ""


def _world_writable(path: Path) -> bool:
    if os.name != "posix":
        return False
    try:
        mode = path.stat().st_mode
    except OSError:
        return False
    return bool(mode & stat.S_IWOTH)


def verify_integrity() -> IntegrityReport:
    """Return a report on package integrity; ``ok`` is overall pass/fail."""
    strict = _strict()
    mode = "strict" if strict else "fail-open"
    warnings: list[str] = []

    root = _package_root()
    if _world_writable(root):
        warnings.append(
            f"package directory is world-writable: {root}",
        )

    manifest, manifest_error = _load_manifest()
    if manifest_error:
        warnings.append(manifest_error)

    mismatched: list[str] = []
    missing: list[str] = []
    extra: list[str] = []
    checked = 0

    # A missing manifest is normal (dev/editable install) and leaves the
    # hash layer with nothing to check — but the other layers still have a
    # verdict to deliver. Returning early here would skip both the
    # warnings rule below and the strict-mode raise, which is how a
    # world-writable package directory and an unreadable manifest both
    # reported ``ok=True`` under MIND_MEM_INTEGRITY=strict.
    if manifest is not None:
        for rel, expected in manifest.items():
            path = root / rel
            if not path.is_file():
                missing.append(rel)
                continue
            actual = _sha256(path)
            checked += 1
            if actual != expected:
                mismatched.append(rel)

        manifest_keys = set(manifest.keys())
        discovered = {str(p.relative_to(root)).replace(os.sep, "/") for p in root.rglob("*.py") if p.is_file()}
        # Only report "extra" critical files — unexpected additions to the
        # tracked set. Ignore freshly-added modules outside the manifest.
        extra = sorted(rel for rel in _CRITICAL_MODULES if rel in discovered and rel not in manifest_keys)

    ok = not mismatched and not missing and not warnings

    report = IntegrityReport(
        ok=ok,
        mode=mode,
        # The file is on disk even when it could not be parsed; saying
        # "no manifest" there would describe a corrupt manifest as an
        # editable install.
        manifest_present=manifest is not None or bool(manifest_error),
        checked=checked,
        mismatched=tuple(mismatched),
        missing=tuple(missing),
        extra=tuple(extra),
        warnings=tuple(warnings),
    )

    if not ok:
        _log.warning(
            "protection.integrity_mismatch mode=%s mismatched=%d missing=%d warnings=%d",
            mode,
            len(mismatched),
            len(missing),
            len(warnings),
        )
        if strict:
            raise RuntimeError(
                f"mind-mem integrity check failed (strict mode): mismatched={mismatched} missing={missing} warnings={warnings}",
            )

    return report


__all__ = [
    "AUDIT_TAG",
    "AUTH_HEADER",
    "IntegrityReport",
    "__author__",
    "__license__",
    "__protection_version__",
    "verify_integrity",
]
