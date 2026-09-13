"""Every serving door binds ONE snapshot, or says it could not.

WHY THIS FILE EXISTS. My previous candidate fixed the snapshot on the ranked MCP recall path and
I described it as "one policy snapshot per request". An independent probe showed that was an
overclaim: three other doors still re-resolved policy after retrieval, because they all route
through ``recall.attest_and_record``, which resolved vector flags, config hash and index anchor
itself. So one fix covered the surface the existing tests exercised and left the shared helper
underneath untouched.

The reviewed consequences, worst first:

* the PREFETCH door wrote a RECORDED **v2** row whose context digest bound a hash from after
  retrieval to a generation from before it — a digest asserting a context that existed at neither
  moment, and v2 is precisely the shape that makes the assertion;
* the AXIS door recorded the later hash too (retrieval under ``4b586a38…``, row carrying
  ``734aaf80…``). It passes ``generation="__not_bound__"`` so the row is v1, which HIDES the
  context digest without making the reread coherent;
* on a derivation failure both doors answered ``"attestation": null`` with no reason at all, while
  ranked MCP emitted an explicit unproven marker. ``attest_and_record`` returned ``None``.

These controls are the follow-up the review asked for: repeat the boundary mutation on the public
doors, assert no v2 row mixes a generation with a hash, and assert explicit unproven output when
derivation cannot bind.

WHAT "CORRECT" MEANS HERE, and it is deliberately a disjunction rather than one shape. A door
whose policy changed mid-request may either bind the RETRIEVAL snapshot coherently, or refuse with
an explicit unproven marker and no row. What it may never do is record a row carrying coordinates
from two moments. Asserting only the first would reject a legitimate fail-closed implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mind_mem import prefetch
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools import public
from mind_mem.served_ledger import ServedRunV2, read_served_runs

# A and B must differ in BOTH coordinates a door can reread: the pipeline hash (via extraction)
# AND the vector flags (via recall.vector_enabled). An earlier version changed only extraction, so
# flipping it moved the hash but left the flags identical — and mutation testing showed the tests
# then could not see a flags reread at all.
_CONFIG_A = {
    "cache": {"enabled": False},
    "extraction": {"backend": "ollama", "model": "a-model"},
    "recall": {"vector_enabled": False},
}
_CONFIG_B = {
    "cache": {"enabled": False},
    "extraction": {"backend": "ollama", "model": "b-model"},
    "recall": {"vector_enabled": True},
}


def _seed_workspace(root: Path) -> str:
    decisions = root / "decisions"
    decisions.mkdir(parents=True)
    for subdir in ("tasks", "entities", "intelligence"):
        (root / subdir).mkdir()
    with (decisions / "DECISIONS.md").open("w", encoding="utf-8", newline="\n") as handle:
        for n in range(4):
            handle.write(f"[D-DOOR-{n:03d}]\nStatement: deterministic compiler retrieval context {n}\nStatus: active\nDate: 2026-01-01\n\n")
    (root / "mind-mem.json").write_text(json.dumps(_CONFIG_A, ensure_ascii=False), encoding="utf-8", newline="\n")
    return str(root)


@pytest.fixture(autouse=True)
def _reset_anticipation_cache() -> Any:
    prefetch.reset_cache()
    yield
    prefetch.reset_cache()


def _flip_config_once(workspace: str, flipped: dict[str, bool]) -> None:
    """Rewrite the config to B, once, to simulate a governed write mid-request."""
    if flipped.get("done"):
        return
    flipped["done"] = True
    Path(workspace, "mind-mem.json").write_text(json.dumps(_CONFIG_B, ensure_ascii=False), encoding="utf-8", newline="\n")


def _expected_hash_a(workspace: str) -> str:
    """The pipeline hash of config A, read while A is still on disk.

    The test must know this INDEPENDENTLY. An earlier version recomputed the context digest from
    the attestation's OWN reported config_hash and compared it to the row — which is self-consistent
    by construction: a row bound to hash B with a generation from A matches its own mixed
    coordinates perfectly. Mutation testing showed that check could not see the defect at all.
    """
    from mind_mem.pipeline_hash import current_pipeline_hash

    return current_pipeline_hash(workspace)


def _assert_coherent_or_refused(workspace: str, payload: dict[str, Any], door: str, expected_hash: str) -> None:
    """Bind the RETRIEVAL snapshot, or refuse explicitly. Never a mixed row.

    Deliberately a disjunction: a door whose policy changed mid-request may bind the retrieval
    snapshot coherently, or fail closed with an explicit marker and no row. Asserting only the
    first would reject a legitimate fail-closed implementation. What it may never do is record a
    row carrying coordinates from two moments.
    """
    assert payload.get("results"), f"{door}: the door must still answer; proof is auxiliary"
    attestation = payload.get("attestation")
    assert isinstance(attestation, dict), (
        f"{door}: a derivation failure must leave an explicit marker, not null — "
        f"a door answering with attestation=None says nothing about why: {payload!r}"
    )
    rows = read_served_runs(workspace)

    if attestation.get("served_proof") == "unproven":
        assert attestation.get("served_seq") is None, attestation
        assert attestation.get("ledger_error"), f"{door}: an unproven result must carry a REASON an operator can act on"
        assert rows == (), f"{door}: a refusal must write no row at all"
        return

    assert attestation.get("served_proof") == "recorded", attestation
    assert len(rows) == 1, rows
    row = rows[-1]

    # THE DISCRIMINATING ASSERTION. The recorded hash must be the one retrieval ran under, not
    # whatever the file said afterwards. This is what a self-consistency check cannot see.
    assert attestation.get("config_hash") == expected_hash, (
        f"{door}: the row was bound to a config hash from AFTER retrieval "
        f"(got {attestation.get('config_hash')!r}, retrieval ran under {expected_hash!r}), so its "
        f"context describes a moment the run never occupied"
    )
    assert attestation["index_anchor"] == row.index_anchor, f"{door}: the attestation and its row disagree about the corpus head"
    # THE FLAGS must come from the same snapshot as the hash. Config A disables the vector leg, B
    # enables it; a door that rereads reports the vector leg as having run on a BM25-only run.
    legs = attestation.get("legs_ran") or []
    assert not any("vector" in str(leg).lower() for leg in legs), (
        f"{door}: legs_ran={legs!r} claims a vector leg, but retrieval ran under config A which "
        f"disables it — the flags were resolved from a later config than the hash"
    )
    if isinstance(row, ServedRunV2):
        from mind_mem.served_ledger import context_digest

        assert row.context_digest == context_digest(
            workspace=workspace,
            config_hash=expected_hash,
            generation=_generation_of_config_a(),
            index_anchor=row.index_anchor,
        ), f"{door}: the v2 context digest does not match the RETRIEVAL snapshot's (workspace, hash, generation, anchor)"


def _generation_of_config_a() -> str:
    from mind_mem.mcp.infra.constants import MCP_SCHEMA_VERSION
    from mind_mem.prefetch import anticipation_generation_identity

    return anticipation_generation_identity(_CONFIG_A, str(MCP_SCHEMA_VERSION)) or ""


def test_the_prefetch_door_never_records_a_mixed_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The door that wrote a recorded v2 mixed row. Bind A coherently, or refuse."""
    workspace = _seed_workspace(tmp_path / "prefetch-boundary")
    expected = _expected_hash_a(workspace)
    flipped: dict[str, bool] = {}

    import mind_mem.recall as recall_mod

    real = recall_mod.attest_and_record

    def flip_then_attest(*args: Any, **kwargs: Any) -> Any:
        _flip_config_once(workspace, flipped)
        return real(*args, **kwargs)

    monkeypatch.setattr(recall_mod, "attest_and_record", flip_then_attest)

    with use_workspace(workspace):
        payload = json.loads(public.recall("deterministic compiler", mode="prefetch", signals="deterministic compiler", limit=5))

    assert flipped.get("done"), "the boundary mutation never ran — this test would prove nothing"
    _assert_coherent_or_refused(workspace, payload, "prefetch", expected)


def test_the_axis_door_never_records_a_mixed_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The v1 row hid the digest but not the reread. Bind A coherently, or refuse."""
    workspace = _seed_workspace(tmp_path / "axis-boundary")
    expected = _expected_hash_a(workspace)
    flipped: dict[str, bool] = {}

    import mind_mem.recall as recall_mod

    real = recall_mod.attest_and_record

    def flip_then_attest(*args: Any, **kwargs: Any) -> Any:
        _flip_config_once(workspace, flipped)
        return real(*args, **kwargs)

    monkeypatch.setattr(recall_mod, "attest_and_record", flip_then_attest)

    with use_workspace(workspace):
        payload = json.loads(public.recall("deterministic compiler", mode="axis", limit=5))

    assert flipped.get("done"), "the boundary mutation never ran — this test would prove nothing"
    _assert_coherent_or_refused(workspace, payload, "axis", expected)


@pytest.mark.parametrize("mode,extra", [("prefetch", {"signals": "deterministic compiler"}), ("axis", {})])
def test_a_derivation_failure_is_explicit_on_every_door(
    mode: str, extra: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``attest_and_record`` returned None, so these doors lost the reason entirely.

    Ranked MCP already emitted a marker; that asymmetry meant two of three doors could fail to
    prove a serve and say nothing about it, which is the exact class RA.1 exists to remove.
    """
    workspace = _seed_workspace(tmp_path / f"fail-{mode}")

    import mind_mem.recall as recall_mod

    monkeypatch.setattr(
        recall_mod,
        "resolve_vector_flags",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("DOOR_PROBE_FAIL")),
    )

    with use_workspace(workspace):
        payload = json.loads(public.recall("deterministic compiler", mode=mode, limit=5, **extra))

    assert payload.get("results"), f"{mode}: the door must still answer"
    attestation = payload.get("attestation")
    assert isinstance(attestation, dict), f"{mode}: a probe failure must produce an explicit marker, not attestation=None"
    assert attestation.get("served_proof") == "unproven", attestation
    assert "DOOR_PROBE_FAIL" in (attestation.get("ledger_error") or ""), (
        f"{mode}: the reason must name the actual failure, not a downstream symptom"
    )
    assert read_served_runs(workspace) == (), f"{mode}: no row may be written"


def test_attest_and_record_hands_its_snapshot_to_the_flags_resolver(tmp_path: Path) -> None:
    """A CALL-LEVEL assertion, and labelled as one because an outcome test is not achievable here.

    The flags half of the snapshot fix removes a live config read inside ``attest_and_record``. I
    tried to gate it on an observable outcome and could not: with config A disabling the vector leg
    and B enabling it, the attestation is byte-identical either way — ``legs_ran=['bm25']``,
    ``legs_degraded=[]`` — because a temporary workspace has no vector backend, so
    ``vector_available`` is False regardless and ``legs_ran`` reports what actually ran rather than
    what the config permitted. Reproducing the difference needs a live vector backend, which these
    tests deliberately do not stand up.

    So this asserts the mechanism instead of its consequence: the resolver must RECEIVE the
    caller's mapping. That does discriminate the regression (dropping the argument makes it None),
    and it is stated as a weaker claim rather than dressed up as an end-to-end proof. If a vector
    backend ever becomes available in test, the outcome assertion is the one to add.
    """
    import mind_mem.recall as recall_mod

    seen: list[Any] = []
    real = recall_mod.resolve_vector_flags

    def recording(workspace: str, backend: str, config: Any = None) -> tuple[bool, bool]:
        seen.append(config)
        return real(workspace, backend, config)

    original = recall_mod.resolve_vector_flags
    recall_mod.resolve_vector_flags = recording
    try:
        workspace = _seed_workspace(tmp_path / "flags-call")
        snapshot = {"cache": {"enabled": False}, "recall": {"vector_enabled": False}}
        recall_mod.attest_and_record(
            workspace,
            "deterministic compiler",
            [],
            generation="__not_bound__",
            config=snapshot,
            config_hash="a" * 64,
            index_anchor="b" * 64,
        )
    finally:
        recall_mod.resolve_vector_flags = original

    assert seen, "resolve_vector_flags was never called — this test would prove nothing"
    assert seen[0] is snapshot, (
        f"attest_and_record must hand the caller's captured mapping to the flags resolver; it "
        f"passed {seen[0]!r}, so the resolver would load live config instead"
    )
