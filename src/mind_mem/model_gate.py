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

The override is deliberately *one-shot*: an entry written by the
override path never carries ``audit_passed: true``, and an entry
carrying ``trust_without_audit: true`` never satisfies the
``trusted_fresh`` fast path. Otherwise one unblock would record
"this manifest passed an audit" for bytes no audit ever saw, and
every later default check would sail through on it — closing
threat 2 above forever. Re-running the audit is the only way back
to a trusted entry.

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
REASON_DRIFT_OVERRIDE = "drift_override"
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
    #: Human-readable elaboration on ``reason``. Additive and optional -- the
    #: reason CODE stays the branchable value; this only sharpens the message
    #: (e.g. "path_not_found" for a path that exists but is a file).
    detail: str = ""


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
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    # Drop entries with the wrong shape so a hand-edited file can't
    # poison subsequent reads.
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write *payload* as JSON — write-temp + os.replace.

    Extracted from ``_save_registry`` so the promotion ledger added in
    5.0.1 gets the same crash-safety the gate registry has always had,
    rather than a second, weaker writer. Serialisation is unchanged
    (``indent=2, sort_keys=True`` + trailing newline), so the registry
    file is byte-for-byte what it was before the extraction.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".gate.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        # Clean up the temp file on any error so we don't leak it.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _save_registry(reg: dict[str, dict[str, Any]]) -> None:
    """Atomically write the registry — write-temp + os.replace."""
    _atomic_write_json(_registry_path(), reg)


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
      * If the path is in the registry, the entry was written by an
        audit (not by an override), the recorded ``manifest_sha256``
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
      * If the checkpoint drifted and ``trust_without_audit=True`` is
        passed → ``passed=True``, ``reason=drift_override``; the new
        manifest is recorded with ``audit_passed=None``, because no
        audit has ever seen these bytes.
      * If the path is never-audited and ``trust_without_audit=True``
        is passed → registry records the override, returns
        ``reason=never_audited_override``.

    Invariant: no ``trust_without_audit=True`` call ever writes
    ``audit_passed=True``, and no entry carrying
    ``trust_without_audit=True`` is ever served from the
    ``trusted_fresh`` fast path. An override buys exactly one load;
    the next default check re-audits.
    """
    from mind_mem.model_audit import audit_model

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        # A checkpoint is a DIRECTORY. Reporting "path_not_found" for a path
        # that plainly exists sent a reader hunting for a missing file; say
        # which of the two it actually is. The reason CODE is unchanged so
        # nothing branching on it breaks -- only the detail is sharper.
        return GateDecision(
            passed=False,
            reason=REASON_PATH_NOT_FOUND,
            path=str(root),
            manifest_sha256="",
            detail=("exists but is not a directory" if root.exists() else "no such path"),
        )

    current_sha = _compute_manifest_sha256(root)
    reg = _load_registry()
    entry = reg.get(str(root))

    # ``trust_without_audit`` is in this condition on purpose: an entry
    # written by the override path has no audit behind it, so serving
    # it here would turn a one-shot operator unblock into a permanent
    # pass. The write path below never records ``audit_passed=True``
    # beside an override any more, but a ledger written by an older
    # build (or hand-edited) can still hold that pair — this refuses
    # it and re-audits.
    entry_is_audit_clean = bool(
        entry and not entry.get("trust_without_audit") and entry.get("manifest_sha256") == current_sha and entry.get("audit_passed")
    )

    if entry_is_audit_clean and entry is not None:  # second clause narrows for the type checker
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
    # Nothing here may record ``audit_passed=True``: these exact bytes
    # have not been audited, and a True beside this manifest is what
    # the fast path above trusts. Only a recorded *failure* that
    # belongs to these bytes is carried forward — that is real
    # evidence, and keeping it makes repeated overrides idempotent.
    audit_passed: bool | None
    prior_summary = entry.get("audit_report_summary") if entry else None
    if drift:
        # The stored verdict describes different bytes; it says
        # nothing about the checkpoint being loaded now.
        reason = REASON_DRIFT_OVERRIDE
        audit_passed = None
        summary = {}
    elif entry is not None and entry.get("audit_passed") is False:
        reason = REASON_AUDIT_FAILED_OVERRIDE
        audit_passed = False
        summary = prior_summary if isinstance(prior_summary, dict) else {}
    else:
        # No entry, an entry with no verdict, or an entry claiming a
        # pass that no audit of these bytes can back (a ledger written
        # by an older build) — all three mean "never audited".
        reason = REASON_NEVER_AUDITED_OVERRIDE
        audit_passed = None
        summary = {}

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


# ---------------------------------------------------------------------------
# Weight-promotion ledger — the registry half of ``online_trainer``
# ---------------------------------------------------------------------------
#
# 5.0.1. ``online_trainer.WeightRegistry`` shipped a SECOND registry: active /
# candidate / rollback slots plus a revert log, all in a process-local dict
# behind an ``RLock``, with no test and no persistence. A promotion decision
# that does not survive the process is not an audit trail — a swap that
# regressed recall would be un-reconstructible the moment the daemon
# restarted, which is exactly when an operator goes looking.
#
# Rather than wire that second registry, its semantics are merged HERE, into
# the module that already owns "which model bytes may mind-mem consume" and
# already has an atomic, path-overridable, tested on-disk ledger.
# ``online_trainer.WeightRegistry`` survives as a thin facade over these
# functions so the historical API keeps resolving, but there is exactly ONE
# implementation of the promotion rule and exactly one place it is written
# down.
#
# The merge also buys a coupling neither half had alone: promotion runs the
# LOAD GATE on the candidate's own bytes (``verify_load_gate``, default on).
# The old registry would happily promote a path that had never been audited,
# which made the gate above bypassable by a single write to the weight
# registry. Fail-closed: no audit, no promotion.

#: Ledger filename, written beside the gate registry.
DEFAULT_PROMOTION_FILENAME = "model_promotions.json"

#: The MRR a candidate must beat the incumbent by before it may be promoted.
#: Preserved from ``online_trainer.WeightRegistry.promote``'s default.
MIN_IMPROVEMENT_DEFAULT = 0.01

#: Cap on the retained event log. Preserved from the in-memory deque's
#: ``maxlen``; the oldest events are dropped first.
PROMOTION_EVENT_CAP = 10_000

# Reasons returned in ``PromotionDecision.reason``.
REASON_PROMOTED = "promoted"
REASON_NO_CANDIDATE = "no_candidate"
REASON_INSUFFICIENT_IMPROVEMENT = "insufficient_improvement"
REASON_LOAD_GATE_REFUSED = "load_gate_refused"
REASON_REVERTED = "reverted"
REASON_NO_ROLLBACK = "no_rollback"

# Event kinds appended to the persisted log. A REFUSAL is logged as loudly as
# a success: "the candidate was refused" is the interesting half of the
# record, and the in-memory version logged only reverts.
EVENT_PROMOTED = "promote"
EVENT_PROMOTE_REFUSED = "promote_refused"
EVENT_REVERTED = "revert"
EVENT_REVERT_REFUSED = "revert_refused"


@dataclass(frozen=True)
class WeightRef:
    """A version-stamped reference to one set of model weights.

    Lives here rather than in ``online_trainer`` because the ledger that
    persists it lives here; ``online_trainer`` re-exports the name so the
    historical import path keeps working.
    """

    model_id: str
    version: str
    path: str
    base_mrr: float
    promoted_at: str = ""
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "path": self.path,
            "base_mrr": self.base_mrr,
            "promoted_at": self.promoted_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "WeightRef | None":
        """Rebuild from a ledger row; ``None`` when the row is unusable.

        Never raises: a hand-edited or half-written ledger row degrades to
        "no such ref", which every caller already handles, rather than
        taking down the daemon pass that read it.
        """
        if not isinstance(data, dict):
            return None
        try:
            return cls(
                model_id=str(data["model_id"]),
                version=str(data.get("version", "")),
                path=str(data.get("path", "")),
                base_mrr=float(data.get("base_mrr", 0.0)),
                promoted_at=str(data.get("promoted_at", "")),
                metadata=dict(data.get("metadata") or {}),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of a :func:`promote_weights` / :func:`revert_weights` call."""

    ok: bool
    reason: str
    model_id: str
    min_improvement: float = MIN_IMPROVEMENT_DEFAULT
    candidate_mrr: float | None = None
    baseline_mrr: float | None = None
    improvement: float | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "model_id": self.model_id,
            "min_improvement": self.min_improvement,
            "candidate_mrr": self.candidate_mrr,
            "baseline_mrr": self.baseline_mrr,
            "improvement": self.improvement,
            "detail": self.detail,
        }


def evaluate_promotion(
    *,
    candidate_mrr: float,
    baseline_mrr: float | None,
    min_improvement: float = MIN_IMPROVEMENT_DEFAULT,
) -> tuple[bool, str]:
    """The promotion rule, and the only copy of it.

    Pure — no clock, no IO, no ledger. A candidate must beat the incumbent
    baseline by at least *min_improvement*; a regression and a
    too-small-to-be-signal improvement are refused identically, because a
    swap that cannot be told apart from noise is not an improvement.

    ``baseline_mrr is None`` means there is no incumbent (first promotion
    for this model), which nothing can regress against.
    """
    if baseline_mrr is None:
        return True, REASON_PROMOTED
    if candidate_mrr < baseline_mrr + min_improvement:
        return False, REASON_INSUFFICIENT_IMPROVEMENT
    return True, REASON_PROMOTED


def _promotion_ledger_path(ledger_path: str | Path | None = None) -> Path:
    """Resolve the promotion ledger.

    Explicit argument wins; then ``MIND_MEM_PROMOTION_LEDGER``; otherwise it
    sits beside whatever ``_registry_path`` resolved to, so a test that
    isolates the gate registry isolates the ledger with it and no test can
    write the operator's real one by forgetting a second env var.
    """
    if ledger_path is not None:
        return Path(ledger_path).expanduser().resolve()
    override = os.environ.get("MIND_MEM_PROMOTION_LEDGER", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _registry_path().parent / DEFAULT_PROMOTION_FILENAME


def _empty_ledger() -> dict[str, Any]:
    return {"models": {}, "events": []}


def load_promotion_ledger(ledger_path: str | Path | None = None) -> dict[str, Any]:
    """Read the ledger; an absent or corrupt file reads as empty.

    Same degradation contract as :func:`_load_registry` — a ledger nobody
    can parse must not be able to stop a promotion decision from being
    made and recorded afresh.
    """
    p = _promotion_ledger_path(ledger_path)
    if not p.is_file():
        return _empty_ledger()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return _empty_ledger()
    if not isinstance(data, dict):
        return _empty_ledger()
    models = data.get("models")
    events = data.get("events")
    return {
        "models": {k: v for k, v in models.items() if isinstance(v, dict)} if isinstance(models, dict) else {},
        "events": [e for e in events if isinstance(e, dict)] if isinstance(events, list) else [],
    }


def _save_promotion_ledger(ledger: dict[str, Any], ledger_path: str | Path | None = None) -> None:
    ledger["events"] = list(ledger.get("events", []))[-PROMOTION_EVENT_CAP:]
    _atomic_write_json(_promotion_ledger_path(ledger_path), ledger)


def _slot(ledger: dict[str, Any], model_id: str) -> dict[str, Any]:
    models = ledger.setdefault("models", {})
    slot = models.get(model_id)
    if not isinstance(slot, dict):
        slot = {"active": None, "candidate": None, "rollback": None}
        models[model_id] = slot
    return slot


def _append_event(ledger: dict[str, Any], event: dict[str, Any]) -> None:
    ledger.setdefault("events", []).append(event)


def register_candidate(ref: WeightRef, *, ledger_path: str | Path | None = None) -> None:
    """Record *ref* as the candidate for its model. Persisted immediately."""
    ledger = load_promotion_ledger(ledger_path)
    _slot(ledger, ref.model_id)["candidate"] = ref.as_dict()
    _save_promotion_ledger(ledger, ledger_path)


def set_active_weights(ref: WeightRef, *, ledger_path: str | Path | None = None) -> None:
    """Record *ref* as the active weights for its model, bypassing the rule.

    This is how a baseline is seeded (an operator declaring what is already
    running), not how a candidate wins — that is :func:`promote_weights`,
    and it is the only path that consults :func:`evaluate_promotion` or the
    load gate.
    """
    ledger = load_promotion_ledger(ledger_path)
    _slot(ledger, ref.model_id)["active"] = ref.as_dict()
    _save_promotion_ledger(ledger, ledger_path)


def active_weights(model_id: str, *, ledger_path: str | Path | None = None) -> WeightRef | None:
    return WeightRef.from_dict(_slot(load_promotion_ledger(ledger_path), model_id).get("active"))


def candidate_weights(model_id: str, *, ledger_path: str | Path | None = None) -> WeightRef | None:
    return WeightRef.from_dict(_slot(load_promotion_ledger(ledger_path), model_id).get("candidate"))


def rollback_weights(model_id: str, *, ledger_path: str | Path | None = None) -> WeightRef | None:
    return WeightRef.from_dict(_slot(load_promotion_ledger(ledger_path), model_id).get("rollback"))


def promote_weights(
    model_id: str,
    *,
    candidate_mrr: float,
    min_improvement: float = MIN_IMPROVEMENT_DEFAULT,
    verify_load_gate: bool = True,
    now: str | None = None,
    ledger_path: str | Path | None = None,
) -> PromotionDecision:
    """Promote the registered candidate to active, or refuse and say why.

    Order of checks is deliberate: the MRR rule is pure and free, the load
    gate hashes a checkpoint directory. A candidate that cannot clear the
    cheap bar never pays for the expensive one.

    Every outcome — promotion, MRR refusal, gate refusal — is appended to
    the persisted event log before returning. A refusal that is not written
    down is a refusal nobody can review.

    ``now`` is injected. The ledger is an operator surface, not the scored
    recall path, but a caller that wants a reproducible ledger (a test, a
    replay) must be able to get one without patching the clock.
    """
    stamp = now or _now_iso()
    ledger = load_promotion_ledger(ledger_path)
    slot = _slot(ledger, model_id)

    cand = WeightRef.from_dict(slot.get("candidate"))
    prev = WeightRef.from_dict(slot.get("active"))
    baseline = prev.base_mrr if prev is not None else None
    improvement = None if baseline is None else candidate_mrr - baseline

    def _refuse(reason: str, detail: str = "") -> PromotionDecision:
        decision = PromotionDecision(
            ok=False,
            reason=reason,
            model_id=model_id,
            min_improvement=min_improvement,
            candidate_mrr=candidate_mrr,
            baseline_mrr=baseline,
            improvement=improvement,
            detail=detail,
        )
        _append_event(ledger, {"event": EVENT_PROMOTE_REFUSED, "at": stamp, **decision.as_dict()})
        _save_promotion_ledger(ledger, ledger_path)
        return decision

    if cand is None:
        return _refuse(REASON_NO_CANDIDATE, "no candidate weights registered")

    ok, reason = evaluate_promotion(
        candidate_mrr=candidate_mrr,
        baseline_mrr=baseline,
        min_improvement=min_improvement,
    )
    if not ok:
        return _refuse(
            reason,
            f"candidate={candidate_mrr:.4f} baseline={baseline:.4f} needs >= {(baseline or 0.0) + min_improvement:.4f}",
        )

    if verify_load_gate:
        gate = gate_check(cand.path)
        if not gate.passed:
            return _refuse(REASON_LOAD_GATE_REFUSED, f"load gate refused {cand.path}: {gate.reason}")

    promoted = WeightRef(
        model_id=cand.model_id,
        version=cand.version,
        path=cand.path,
        base_mrr=candidate_mrr,
        promoted_at=stamp,
        metadata=dict(cand.metadata),
    )
    if prev is not None:
        slot["rollback"] = prev.as_dict()
    slot["active"] = promoted.as_dict()
    slot["candidate"] = None

    decision = PromotionDecision(
        ok=True,
        reason=REASON_PROMOTED,
        model_id=model_id,
        min_improvement=min_improvement,
        candidate_mrr=candidate_mrr,
        baseline_mrr=baseline,
        improvement=improvement,
        detail=f"{cand.version} -> active",
    )
    _append_event(ledger, {"event": EVENT_PROMOTED, "at": stamp, **decision.as_dict()})
    _save_promotion_ledger(ledger, ledger_path)
    return decision


def revert_weights(
    model_id: str,
    *,
    reason: str,
    now: str | None = None,
    ledger_path: str | Path | None = None,
) -> PromotionDecision:
    """Restore the rollback slot to active. The revert log is PERSISTED.

    The in-memory predecessor kept reverts in a ``deque`` that died with the
    process; an auto-revert fired by an MRR regression at 03:00 left no
    trace by morning. Here both the state change and the event survive.
    """
    stamp = now or _now_iso()
    ledger = load_promotion_ledger(ledger_path)
    slot = _slot(ledger, model_id)
    rb = WeightRef.from_dict(slot.get("rollback"))

    if rb is None:
        decision = PromotionDecision(ok=False, reason=REASON_NO_ROLLBACK, model_id=model_id, detail=reason)
        _append_event(ledger, {"event": EVENT_REVERT_REFUSED, "at": stamp, **decision.as_dict()})
        _save_promotion_ledger(ledger, ledger_path)
        return decision

    slot["active"] = rb.as_dict()
    slot["rollback"] = None
    decision = PromotionDecision(
        ok=True,
        reason=REASON_REVERTED,
        model_id=model_id,
        baseline_mrr=rb.base_mrr,
        detail=reason,
    )
    _append_event(ledger, {"event": EVENT_REVERTED, "at": stamp, **decision.as_dict()})
    _save_promotion_ledger(ledger, ledger_path)
    return decision


def promotion_events(
    *,
    limit: int = 50,
    model_id: str | None = None,
    ledger_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """The most recent persisted promotion / revert events, oldest first."""
    events = load_promotion_ledger(ledger_path).get("events", [])
    if model_id is not None:
        events = [e for e in events if e.get("model_id") == model_id]
    return events[-max(0, int(limit)) :] if limit else []


def promotion_stats(ledger_path: str | Path | None = None) -> dict[str, Any]:
    """Ledger snapshot for MCP observability.

    Deliberately counts-and-ids only: model ids, versions, baselines and
    per-kind event counts. No checkpoint contents, no metadata blobs.
    """
    ledger = load_promotion_ledger(ledger_path)
    models = ledger.get("models", {})
    events = ledger.get("events", [])
    counts: dict[str, int] = {}
    for event in events:
        kind = str(event.get("event", "?"))
        counts[kind] = counts.get(kind, 0) + 1
    return {
        "models": sorted(models.keys()),
        "active": {
            m: ref.as_dict()
            for m, slot in sorted(models.items())
            if isinstance(slot, dict) and (ref := WeightRef.from_dict(slot.get("active"))) is not None
        },
        "candidate_pending": sorted(m for m, slot in models.items() if isinstance(slot, dict) and slot.get("candidate")),
        "rollback_available": sorted(m for m, slot in models.items() if isinstance(slot, dict) and slot.get("rollback")),
        "events": len(events),
        "events_by_kind": dict(sorted(counts.items())),
    }
