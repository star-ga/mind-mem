"""An unstated comparability field is not agreement.

``comparability_signature`` defaulted a missing ``agent`` to ``{}`` and a
missing ``task_set.sha256`` to ``""``, while the artifact validator did
not require either. Two runs of different agents over different task
sets, neither carrying the metadata, therefore produced one identical
signature: ``assert_comparable`` saw a single signature, ``collect_pairs``
saw no duplicate task ids (different task sets), and the two experiments
merged into one headline delta and one p-value — under a report whose own
text says such artifacts are refused.

``run_suite`` is public API and takes ``meta`` as an arbitrary mapping;
only the shipped CLI happens to populate those keys, so this is reachable
by any programmatic caller.

Also pinned: ``pool_exclusions`` reads ``excluded``, which was not a
required key, so a hand-built artifact raised a bare ``KeyError`` instead
of the module's own refusal type.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.bench.ab_report import (
    ReportError,
    assert_comparable,
    build_report,
    load_artifact,
    unstated_comparability_fields,
)

BUDGET = {"prompt_tokens": 8000, "memory_tokens": 1500, "output_tokens": 4000, "wall_seconds": 300, "steps": 40}
AGENT = {"name": "command", "argv": ["/bin/true"], "env_passthrough_keys": []}


def record(task_id: str, memory: bool, control: bool) -> dict:
    return {
        "task_id": task_id,
        "tier": "behavioral",
        "size_bucket": "small",
        "excluded": None,
        "outcome": {"memory_success": memory, "control_success": control},
    }


def artifact(rows: list[dict], **overrides) -> dict:
    payload = {
        "budget": dict(BUDGET),
        "agent": dict(AGENT),
        "task_set": {"sha256": "abc", "selection": "bucket:small"},
        "counts": {"scored": len(rows), "excluded": 0, "agent_inert_task_ids": []},
        "spend": {"control": {"prompt_tokens": 100}, "memory": {"prompt_tokens": 400}},
        "excluded": [],
        "results": list(rows),
        "summary": {},
        "digest": "d" * 64,
    }
    payload.update(overrides)
    return payload


def write(tmp_path: Path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


@pytest.mark.unit
def test_two_runs_that_state_no_agent_are_not_pooled_as_one(tmp_path: Path) -> None:
    left = artifact([record("a", True, False)])
    right = artifact([record("b", True, False)])
    for payload in (left, right):
        del payload["agent"]
        del payload["task_set"]
    paths = [write(tmp_path, "l.json", left), write(tmp_path, "r.json", right)]
    with pytest.raises(ReportError):
        build_report(paths)


@pytest.mark.unit
def test_an_empty_agent_or_digest_is_named_as_unstated() -> None:
    assert unstated_comparability_fields(artifact([])) == ()
    assert unstated_comparability_fields(artifact([], agent={})) == ("agent",)
    assert unstated_comparability_fields(artifact([], task_set={"selection": "all"})) == ("task_set_sha256",)
    assert unstated_comparability_fields(artifact([], budget={}, agent={})) == ("budget", "agent")


@pytest.mark.unit
def test_assert_comparable_names_the_artifact_that_stated_nothing() -> None:
    with pytest.raises(ReportError, match="does not state"):
        assert_comparable([artifact([], agent={})], ["left.json"])


@pytest.mark.unit
def test_a_stated_and_matching_pair_still_pools(tmp_path: Path) -> None:
    paths = [
        write(tmp_path, "l.json", artifact([record("a", True, False)])),
        write(tmp_path, "r.json", artifact([record("b", False, True)])),
    ]
    payload = build_report(paths)
    assert payload["n_pairs"] == 2


@pytest.mark.unit
def test_an_artifact_without_the_exclusion_list_is_refused_as_such(tmp_path: Path) -> None:
    payload = artifact([record("a", True, False)])
    del payload["excluded"]
    with pytest.raises(ReportError, match="not an A/B run artifact"):
        load_artifact(write(tmp_path, "a.json", payload))
