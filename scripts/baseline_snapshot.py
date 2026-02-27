"""Baseline snapshot for intent drift detection.

Captures versioned snapshots of intent classification distribution before
learned routing evolves.  Drift detection compares current state against a
frozen baseline using chi-squared goodness-of-fit.

Addresses: star-ga/mind-mem#431
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "freeze_baseline",
    "detect_drift",
    "compare_baselines",
    "list_baselines",
    "_cli",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASELINES_DIR = "intelligence/baselines"
_BASELINE_PREFIX = "intent-baseline-v"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _baselines_path(workspace: str) -> Path:
    return Path(os.path.abspath(workspace)) / _BASELINES_DIR


def _list_versions(workspace: str) -> list[int]:
    """Return sorted list of existing baseline version numbers."""
    d = _baselines_path(workspace)
    if not d.is_dir():
        return []
    versions: list[int] = []
    for f in d.iterdir():
        name = f.stem  # e.g. intent-baseline-v3
        if name.startswith(_BASELINE_PREFIX):
            try:
                versions.append(int(name[len(_BASELINE_PREFIX) :]))
            except ValueError:
                continue
    versions.sort()
    return versions


def _next_version(workspace: str) -> int:
    versions = _list_versions(workspace)
    return (versions[-1] + 1) if versions else 1


def _load_baseline(workspace: str, version: int) -> dict:
    version = int(version)
    p = _baselines_path(workspace) / f"{_BASELINE_PREFIX}{version}.json"
    if not p.exists():
        raise FileNotFoundError(f"Baseline v{version} not found at {p}")
    with open(p, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
        return data


def _config_fingerprint(workspace: str) -> dict[str, Any]:
    """Capture recall config hash so drift from config changes is identifiable."""
    config_path = Path(workspace) / "mind-mem.json"
    cfg: dict = {}
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as fh:
                raw = json.load(fh)
            recall = raw.get("recall", {})
            cfg = {
                "bm25_weight": recall.get("bm25_weight", 1.0),
                "vector_weight": recall.get("vector_weight", 1.0),
                "rrf_k": recall.get("rrf_k", 60),
                "vector_model": recall.get("vector_model", "all-MiniLM-L6-v2"),
                "vector_enabled": recall.get("vector_enabled", False),
                "backend": recall.get("backend", "scan"),
            }
        except (json.JSONDecodeError, OSError):
            cfg = {"error": "could not read mind-mem.json"}
    else:
        # Use defaults from hybrid_recall
        cfg = {
            "bm25_weight": 1.0,
            "vector_weight": 1.0,
            "rrf_k": 60,
            "vector_model": "all-MiniLM-L6-v2",
            "vector_enabled": False,
            "backend": "scan",
        }

    # Stable hash of config for quick diff
    cfg_str = json.dumps(cfg, sort_keys=True)
    cfg["_hash"] = hashlib.sha256(cfg_str.encode()).hexdigest()[:16]
    return cfg


# ---------------------------------------------------------------------------
# Chi-squared goodness-of-fit
# ---------------------------------------------------------------------------


def _chi_squared(observed: dict[str, int], expected: dict[str, int]) -> dict[str, Any]:
    """Chi-squared test comparing observed vs expected intent distributions.

    Returns chi2 statistic, degrees of freedom, p-value, and per-category
    contributions.  Uses scipy if available, otherwise a manual approximation
    of the survival function.
    """
    all_intents = sorted(set(observed) | set(expected))
    if not all_intents:
        return {"chi2": 0.0, "df": 0, "p_value": 1.0, "per_intent": {}}

    # Scale expected to match observed total (compare shape, not magnitude)
    obs_total = sum(observed.get(k, 0) for k in all_intents)
    exp_total = sum(expected.get(k, 0) for k in all_intents)
    if exp_total == 0 or obs_total == 0:
        return {"chi2": 0.0, "df": 0, "p_value": 1.0, "per_intent": {}}

    scale = obs_total / exp_total

    chi2 = 0.0
    per_intent: dict[str, dict] = {}
    for intent in all_intents:
        o = observed.get(intent, 0)
        e = expected.get(intent, 0) * scale
        if e > 0:
            contrib = (o - e) ** 2 / e
        else:
            contrib = float(o) if o > 0 else 0.0
        chi2 += contrib
        per_intent[intent] = {
            "observed": o,
            "expected": round(e, 2),
            "contribution": round(contrib, 4),
        }

    df = max(len(all_intents) - 1, 1)

    # p-value via scipy if available, else rough upper-bound approximation
    p_value: float
    try:
        from scipy.stats import chi2 as chi2_dist

        p_value = float(chi2_dist.sf(chi2, df))
    except ImportError:
        # Approximation: for large chi2, p ≈ 0; for chi2 ≈ df, p ≈ 0.5
        # This is intentionally conservative (overestimates p).
        if chi2 <= 0:
            p_value = 1.0
        elif chi2 < df:
            p_value = 1.0 - (chi2 / (2 * df))
        else:
            # Rough: p decreases roughly exponentially past df
            p_value = max(0.001, 2.71828 ** (-(chi2 - df) / (2 * df)))

    return {
        "chi2": round(chi2, 4),
        "df": df,
        "p_value": round(p_value, 6),
        "per_intent": per_intent,
    }


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def freeze_baseline(
    workspace: str,
    tag: int | None = None,
    last_n: int = 50,
    max_age_days: int = 7,
) -> dict[str, Any]:
    """Capture a versioned baseline snapshot of the current intent distribution.

    Args:
        workspace: mind-mem workspace root path.
        tag: Explicit version tag. Auto-increments if None.
        last_n: Queries to analyze (passed to retrieval_diagnostics).
        max_age_days: Age window for queries.

    Returns:
        The baseline artifact dict (also written to disk).
    """
    if tag is not None and tag < 1:
        raise ValueError("tag must be a positive integer")

    # Get current diagnostics
    from mind_mem.retrieval_graph import retrieval_diagnostics

    diag = retrieval_diagnostics(workspace, last_n=last_n, max_age_days=max_age_days)

    version = tag if tag is not None else _next_version(workspace)

    # Build low-confidence summary
    lc_queries = diag.get("low_confidence_queries", [])
    total_queries = diag.get("queries_analyzed", 0)
    lc_fraction = len(lc_queries) / total_queries if total_queries > 0 else 0.0

    artifact = {
        "version_tag": f"v{version}",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "queries_analyzed": total_queries,
        "intent_distribution": diag.get("intent_distribution", {}),
        "intent_confidence_histogram": {
            intent: data.get("avg_confidence")
            for intent, data in diag.get("intent_quality", {}).items()
            if data.get("avg_confidence") is not None
        },
        "intent_quality": {
            intent: {
                "queries": data.get("queries", 0),
                "avg_top_score": data.get("avg_top_score"),
                "p50_top_score": data.get("p50_top_score"),
            }
            for intent, data in diag.get("intent_quality", {}).items()
        },
        "low_confidence_fraction": round(lc_fraction, 4),
        "low_confidence_queries": lc_queries[:10],
        "score_distribution": diag.get("score_distribution", {}),
        "config_fingerprint": _config_fingerprint(workspace),
    }

    # Write to disk
    out_dir = _baselines_path(workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_BASELINE_PREFIX}{version}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)

    return artifact


def detect_drift(
    workspace: str,
    baseline_tag: int | None = None,
    last_n: int = 50,
    max_age_days: int = 7,
    significance: float = 0.05,
) -> dict[str, Any]:
    """Compare current state against a frozen baseline.

    Args:
        workspace: mind-mem workspace root path.
        baseline_tag: Version to compare against. Uses latest if None.
        last_n: Queries to analyze for current state.
        max_age_days: Age window for queries.
        significance: p-value threshold for declaring significant drift.

    Returns:
        Drift report dict.
    """
    # Load baseline
    versions = _list_versions(workspace)
    if not versions:
        return {"error": "No baselines found. Run freeze first."}

    version = int(baseline_tag) if baseline_tag is not None else versions[-1]
    baseline = _load_baseline(workspace, version)

    # Get current diagnostics
    from mind_mem.retrieval_graph import retrieval_diagnostics

    current_diag = retrieval_diagnostics(workspace, last_n=last_n, max_age_days=max_age_days)

    current_dist = current_diag.get("intent_distribution", {})
    baseline_dist = baseline.get("intent_distribution", {})

    # Chi-squared test
    chi_result = _chi_squared(current_dist, baseline_dist)

    # Low-confidence analysis
    current_lc = current_diag.get("low_confidence_queries", [])
    current_total = current_diag.get("queries_analyzed", 0)
    current_lc_frac = len(current_lc) / current_total if current_total > 0 else 0.0
    baseline_lc_frac = baseline.get("low_confidence_fraction", 0.0)

    lc_direction = "stable"
    if current_lc_frac < baseline_lc_frac - 0.05:
        lc_direction = "shrinking"
    elif current_lc_frac > baseline_lc_frac + 0.05:
        lc_direction = "growing"

    # False certainty detection: LC shrinking but quality flat/declining
    false_certainty = False
    current_quality = current_diag.get("intent_quality", {})
    baseline_quality = baseline.get("intent_quality", {})
    if lc_direction == "shrinking":
        # Check if quality is flat or declining in previously low-confidence intents
        quality_deltas: list[float] = []
        for intent, bq in baseline_quality.items():
            cq = current_quality.get(intent, {})
            b_score = bq.get("avg_top_score")
            c_score = cq.get("avg_top_score")
            if b_score is not None and c_score is not None:
                quality_deltas.append(c_score - b_score)
        if quality_deltas:
            avg_delta = sum(quality_deltas) / len(quality_deltas)
            if avg_delta <= 0:
                false_certainty = True

    # Config change detection
    config_changed = False
    current_config = _config_fingerprint(workspace)
    baseline_config = baseline.get("config_fingerprint", {})
    if current_config.get("_hash") != baseline_config.get("_hash"):
        config_changed = True

    significant = chi_result["p_value"] < significance

    return {
        "baseline_version": baseline.get("version_tag"),
        "baseline_frozen_at": baseline.get("frozen_at"),
        "current_queries_analyzed": current_total,
        "chi_squared_test": chi_result,
        "significant_drift": significant,
        "significance_threshold": significance,
        "low_confidence": {
            "baseline_fraction": baseline_lc_frac,
            "current_fraction": round(current_lc_frac, 4),
            "direction": lc_direction,
            "false_certainty_warning": false_certainty,
        },
        "config_changed": config_changed,
        "config_diff": ({"baseline": baseline_config, "current": current_config} if config_changed else None),
    }


def compare_baselines(workspace: str, v1: int, v2: int) -> dict[str, Any]:
    """Compare two frozen baselines directly.

    Args:
        workspace: mind-mem workspace root path.
        v1: First baseline version.
        v2: Second baseline version.

    Returns:
        Comparison report dict.
    """
    v1, v2 = int(v1), int(v2)
    b1 = _load_baseline(workspace, v1)
    b2 = _load_baseline(workspace, v2)

    chi_result = _chi_squared(
        b2.get("intent_distribution", {}),
        b1.get("intent_distribution", {}),
    )

    # Config change detection
    config_changed = b1.get("config_fingerprint", {}).get("_hash") != b2.get("config_fingerprint", {}).get("_hash")

    return {
        "v1": b1.get("version_tag"),
        "v1_frozen_at": b1.get("frozen_at"),
        "v2": b2.get("version_tag"),
        "v2_frozen_at": b2.get("frozen_at"),
        "chi_squared_test": chi_result,
        "low_confidence_v1": b1.get("low_confidence_fraction"),
        "low_confidence_v2": b2.get("low_confidence_fraction"),
        "config_changed": config_changed,
    }


def list_baselines(workspace: str) -> list[dict[str, Any]]:
    """List all frozen baselines with summary metadata.

    Args:
        workspace: mind-mem workspace root path.

    Returns:
        List of baseline summary dicts sorted by version.
    """
    versions = _list_versions(workspace)
    result: list[dict] = []
    for v in versions:
        try:
            b = _load_baseline(workspace, v)
            result.append(
                {
                    "version_tag": b.get("version_tag"),
                    "frozen_at": b.get("frozen_at"),
                    "queries_analyzed": b.get("queries_analyzed"),
                    "intent_count": len(b.get("intent_distribution", {})),
                    "low_confidence_fraction": b.get("low_confidence_fraction"),
                    "config_hash": b.get("config_fingerprint", {}).get("_hash"),
                }
            )
        except (FileNotFoundError, json.JSONDecodeError):
            result.append({"version_tag": f"v{v}", "error": "corrupt or missing"})
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    """mind-mem-baseline CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="mind-mem-baseline",
        description="Manage intent classification baseline snapshots.",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        default=".",
        help="mind-mem workspace root (default: current directory)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # freeze
    freeze_p = sub.add_parser("freeze", help="Capture a new baseline snapshot")
    freeze_p.add_argument("--tag", type=int, help="Explicit version tag")
    freeze_p.add_argument("--last-n", type=int, default=50)
    freeze_p.add_argument("--max-age-days", type=int, default=7)

    # drift
    drift_p = sub.add_parser("drift", help="Detect drift against a baseline")
    drift_p.add_argument("--baseline", type=int, help="Baseline version (default: latest)")
    drift_p.add_argument("--last-n", type=int, default=50)
    drift_p.add_argument("--max-age-days", type=int, default=7)
    drift_p.add_argument("--significance", type=float, default=0.05)

    # compare
    compare_p = sub.add_parser("compare", help="Compare two baselines")
    compare_p.add_argument("v1", type=int, help="First baseline version")
    compare_p.add_argument("v2", type=int, help="Second baseline version")

    # list
    sub.add_parser("list", help="List all baselines")

    args = parser.parse_args()

    if args.command == "freeze":
        result = freeze_baseline(
            args.workspace,
            tag=args.tag,
            last_n=args.last_n,
            max_age_days=args.max_age_days,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "drift":
        result = detect_drift(
            args.workspace,
            baseline_tag=args.baseline,
            last_n=args.last_n,
            max_age_days=args.max_age_days,
            significance=args.significance,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "compare":
        result = compare_baselines(args.workspace, args.v1, args.v2)
        print(json.dumps(result, indent=2))

    elif args.command == "list":
        baselines = list_baselines(args.workspace)
        print(json.dumps(baselines, indent=2))


if __name__ == "__main__":
    _cli()
