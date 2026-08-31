"""Pool several A/B run artifacts into one delta, with its uncertainty.

A hundred-odd tasks times two arms does not fit in one process on a shared
box, so the suite is run one **stated stratum at a time**
(``--select bucket:<name>``) and the strata are pooled here.

Pooling is legitimate because the design is paired: each task contributes
exactly one ``(memory_success, control_success)`` pair, which the
per-stratum artifact already recorded, so the pooled summary is the one a
single serial run would have produced.  Nothing is re-graded and nothing is
recomputed from raw output -- this reads verdicts that pytest already
decided.

Two refusals keep the pooling honest rather than convenient:

* artifacts produced under **different budgets or different agents** are
  not comparable and are refused, because pooling them would silently mix
  two experiments into one headline -- and so is an artifact that does not
  **state** its budget, agent or task-set digest, because a field left
  unstated would otherwise read as agreement with every other run;
* a task appearing in **more than one artifact** is refused, because
  counting a pair twice inflates the discordant count that the whole
  significance claim rests on.

The breakdowns by tier and by size bucket are reported next to the pooled
number, not instead of it: a stratum where both arms always fail carries no
information, and a reader can only see that if the strata are shown.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .ab_stats import summarise

#: Fields that must agree across artifacts before they may be pooled.
COMPARABILITY_KEYS: tuple[str, ...] = ("budget", "agent", "task_set_sha256")

#: Keys every A/B run artifact carries; their absence means "not one of ours".
#: ``agent`` and ``task_set`` are here because the comparability check reads
#: them: an artifact that does not state them cannot be shown to be the same
#: experiment, and defaulting absence to ``{}`` made two unrelated runs match.
#: ``excluded`` is here because :func:`pool_exclusions` reads it directly, and
#: a bare ``KeyError`` is not a refusal a caller can act on.
REQUIRED_ARTIFACT_KEYS: tuple[str, ...] = (
    "budget",
    "summary",
    "results",
    "counts",
    "spend",
    "agent",
    "task_set",
    "excluded",
)


class ReportError(ValueError):
    """The artifacts handed in cannot honestly be pooled."""


@dataclass(frozen=True)
class Pair:
    """One task's paired outcome, carried with the strata it belongs to."""

    task_id: str
    memory: bool
    control: bool
    tier: str
    size_bucket: str


def load_artifact(path: str) -> dict[str, Any]:
    """Read one run artifact, refusing anything that is not one."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ReportError(f"{path} is not an A/B run artifact (not an object)")
    missing = [key for key in REQUIRED_ARTIFACT_KEYS if key not in payload]
    if missing:
        raise ReportError(f"{path} is not an A/B run artifact (missing {', '.join(missing)})")
    return payload


def pairs_of(artifact: Mapping[str, Any]) -> tuple[Pair, ...]:
    """Lift the scored records into paired outcomes; excluded rows are skipped."""
    rows: list[Pair] = []
    for record in artifact["results"]:
        if record.get("excluded"):
            continue
        outcome = record["outcome"]
        rows.append(
            Pair(
                task_id=str(record["task_id"]),
                memory=bool(outcome["memory_success"]),
                control=bool(outcome["control_success"]),
                tier=str(record.get("tier", "unknown")),
                size_bucket=str(record.get("size_bucket", "unknown")),
            )
        )
    return tuple(rows)


def comparability_signature(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """What must match for two runs to be halves of one experiment.

    A field the artifact does not state comes back empty; emptiness is not
    a match, so callers go through :func:`unstated_comparability_fields`
    (as :func:`assert_comparable` does) before comparing two signatures.
    """
    task_set = artifact.get("task_set")
    return {
        "budget": artifact.get("budget", {}),
        "agent": artifact.get("agent", {}),
        "task_set_sha256": task_set.get("sha256", "") if isinstance(task_set, Mapping) else "",
    }


def unstated_comparability_fields(artifact: Mapping[str, Any]) -> tuple[str, ...]:
    """Which comparability fields the artifact does not actually state.

    An absent or empty field is not agreement: two runs of different agents
    over different task sets, neither carrying the metadata, produced one
    identical signature and pooled into a single headline delta.
    """
    signature = comparability_signature(artifact)
    return tuple(name for name in COMPARABILITY_KEYS if not signature[name])


def assert_comparable(artifacts: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> None:
    """Refuse to pool runs that were not the same experiment."""
    seen: dict[str, str] = {}
    for artifact, label in zip(artifacts, labels):
        unstated = unstated_comparability_fields(artifact)
        if unstated:
            raise ReportError(
                f"{label} does not state {', '.join(unstated)}; an unstated field is not agreement, "
                "so this artifact cannot be shown to be the same experiment as the others"
            )
        key = json.dumps(comparability_signature(artifact), sort_keys=True, ensure_ascii=False)
        seen.setdefault(key, label)
    if len(seen) > 1:
        raise ReportError(
            f"these artifacts are not one experiment: {sorted(seen.values())} differ in "
            f"{', '.join(COMPARABILITY_KEYS)}; pooling them would mix two runs into one headline"
        )


def collect_pairs(artifacts: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> tuple[Pair, ...]:
    """Concatenate the strata, refusing a task that appears twice."""
    pairs: list[Pair] = []
    origin: dict[str, str] = {}
    for artifact, label in zip(artifacts, labels):
        for pair in pairs_of(artifact):
            if pair.task_id in origin:
                raise ReportError(f"task {pair.task_id} appears in both {origin[pair.task_id]} and {label}; it would be counted twice")
            origin[pair.task_id] = label
            pairs.append(pair)
    return tuple(pairs)


def group_summaries(pairs: Sequence[Pair], attribute: str) -> dict[str, Any]:
    """Summarise each stratum of ``attribute`` separately."""
    names = sorted({str(getattr(pair, attribute)) for pair in pairs})
    return {name: summarise([(p.memory, p.control) for p in pairs if str(getattr(p, attribute)) == name]).as_dict() for name in names}


def pool_spend(artifacts: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Add up what each arm spent across every pooled run."""
    totals: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        for arm, row in artifact["spend"].items():
            bucket = totals.setdefault(arm, {})
            for key, value in row.items():
                bucket[key] = bucket.get(key, 0) + int(value)
    return {arm: dict(sorted(row.items())) for arm, row in sorted(totals.items())}


def pool_exclusions(artifacts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every task that could not be attempted, with its reason and count."""
    reasons: dict[str, int] = {}
    rows: list[dict[str, str]] = []
    for artifact in artifacts:
        for record in artifact["excluded"]:
            reason = str(record.get("excluded", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
            rows.append({"task_id": str(record.get("task_id", "")), "reason": reason})
    return {"total": len(rows), "by_reason": dict(sorted(reasons.items())), "tasks": sorted(rows, key=lambda r: r["task_id"])}


def pool_inert(artifacts: Sequence[Mapping[str, Any]]) -> list[str]:
    """Tasks where neither arm touched a file -- an agent that never ran."""
    inert: list[str] = []
    for artifact in artifacts:
        inert.extend(str(task_id) for task_id in artifact["counts"].get("agent_inert_task_ids", []))
    return sorted(inert)


def build_report(paths: Sequence[str]) -> dict[str, Any]:
    """Pool the named artifacts into one report with its own digest."""
    if not paths:
        raise ReportError("no artifacts to pool")
    artifacts = [load_artifact(path) for path in paths]
    assert_comparable(artifacts, paths)
    pairs = collect_pairs(artifacts, paths)
    if not pairs:
        raise ReportError("the pooled artifacts contain no scored task")
    payload: dict[str, Any] = {
        "what_this_is": POOLED_DOC,
        "inputs": [
            {"path": path, "digest": artifact.get("digest", ""), "selection": artifact.get("task_set", {}).get("selection", "")}
            for path, artifact in zip(paths, artifacts)
        ],
        "budget": artifacts[0]["budget"],
        "agent": artifacts[0].get("agent", {}),
        "n_pairs": len(pairs),
        "summary": summarise([(p.memory, p.control) for p in pairs]).as_dict(),
        "by_tier": group_summaries(pairs, "tier"),
        "by_size_bucket": group_summaries(pairs, "size_bucket"),
        "spend": pool_spend(artifacts),
        "excluded": pool_exclusions(artifacts),
        "agent_inert_task_ids": pool_inert(artifacts),
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


#: Written into every pooled report so the file states its own design.
POOLED_DOC: dict[str, str] = {
    "question": "Does the same agent, at the same budget, complete more real repository tasks with mind-mem than without it?",
    "pooling": (
        "Paired outcomes concatenated across stated strata, each run as its own process. One pair per task, taken from "
        "the per-stratum artifact; nothing is re-graded and no task may appear twice."
    ),
    "refusals": (
        "Artifacts differing in budget, agent or task-set digest are refused, as is one that does not state them, "
        "as is a task present in two artifacts."
    ),
    "statistics": "Paired McNemar exact, two-sided. Below 6 discordant pairs no split can reach p<=0.05; such a delta is noise.",
}
