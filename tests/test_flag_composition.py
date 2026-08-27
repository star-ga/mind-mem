# Copyright 2026 STARGA, Inc.
"""Cross-feature composition gate — every default-OFF recall flag ON together.

Each 4.10.0 surface shipped with its own proof that it is inert while OFF and
correct while ON *alone*. The untested surface is the product: what the
pipeline does when the validity gate (with **both** sub-flags), guardrail
surfacing, outcome attribution and persisted world-staleness are all live over
the same corpus, in the same call.

They meet in exactly two places, and this file pins both:

* :func:`mind_mem.validity_gate.validity_components` — provenance class is a
  fifth *criterion*, outcome attribution a *factor*, and world-staleness
  reaches it indirectly, as rows :func:`persist_world_staleness` writes into
  the ``block_staleness`` table the gate already reads (Stage 2.65).
* :func:`mind_mem.guardrail_surface.apply_guardrail_surfacing` — runs *last*,
  after every filter and after the gate, so a demoted guardrail must still
  surface.

Locked in, over the full 32-combination flag matrix:

  (a) recall returns a sane, deterministic result in every combination —
      three consecutive runs are byte-identical on the whole hit list;
  (b) no combination raises;
  (c) guardrail displacement stays bounded by ``max_surfaced`` no matter
      which other features are on;
  (d) validity demotion and guardrail forced-surfacing do not fight — a
      guardrail demoted to below the knee still comes back at position 0.

Two composition behaviours found here are characterised, not asserted away —
see :class:`TestDocumentedCompositionQuirks`.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
from typing import Any

import pytest

from mind_mem._recall_core import recall
from mind_mem.guardrails import MAX_SURFACED_HARD_CAP
from mind_mem.init_workspace import init
from mind_mem.outcome_attribution import report_outcome
from mind_mem.world_staleness import persist_world_staleness, world_staleness_report

QUERY = "widget rollout ledger migration"

ID_CLEAN = "D-20260801-001"
ID_WORLD_STALE = "D-20260801-002"
ID_OUTCOME_FAIL = "D-20260801-003"
ID_CONTRADICTED = "D-20260801-004"
ID_EXTERNAL = "D-20260801-005"
ID_GR_DEMOTABLE = "GR-20260801-900"
ID_GR_HIGH = "GR-20260801-901"
ID_GR_MEDIUM = "GR-20260801-902"

#: A dead path anchor: nothing under the configured root resolves it.
DEAD_ANCHOR = "path:src/deleted_module.py"

DECISIONS = f"""
[{ID_CLEAN}]
Type: Decision
Date: 2026-08-01
Status: active
Statement: Widget rollout ledger migration proceeds on the clean control path.
Rationale: composition fixture — no debit on any criterion
Tags: composition-fixture

[{ID_WORLD_STALE}]
Type: Decision
Date: 2026-08-01
Status: active
Statement: Widget rollout ledger migration notes live beside the code.
Rationale: composition fixture — cites a path the world deleted
Tags: composition-fixture
Anchors:
- {DEAD_ANCHOR}

[{ID_OUTCOME_FAIL}]
Type: Decision
Date: 2026-08-01
Status: active
Statement: Widget rollout ledger migration retry policy was applied in production.
Rationale: composition fixture — repeatedly implicated in failed outcomes
Tags: composition-fixture

[{ID_CONTRADICTED}]
Type: Decision
Date: 2026-08-01
Status: deprecated
Statement: Widget rollout ledger migration old plan superseded and contradicted.
Rationale: composition fixture — dead status plus an open contradiction
Tags: composition-fixture

[{ID_EXTERNAL}]
Type: Decision
Date: 2026-08-01
Status: active
Statement: Widget rollout ledger migration external ingest note.
Rationale: composition fixture — external-ingest provenance class
ActorRole: importer
ToolId: imported:chroma
Source: imported:chroma
Tags: composition-fixture

[{ID_GR_DEMOTABLE}]
Type: Guardrail
Date: 2026-08-01
Status: active
Statement: Widget rollout ledger migration must never run git reset --hard without git status.
Severity: critical
TriggerTools: Bash
TriggerCommands: git reset --hard
ActorRole: importer
ToolId: imported:chroma
Source: imported:chroma
Anchors:
- {DEAD_ANCHOR}
Tags: composition-fixture

[{ID_GR_HIGH}]
Type: Guardrail
Date: 2026-08-01
Status: active
Statement: Widget rollout ledger migration writes need a reviewed rollback script.
Severity: high
TriggerTools: Bash
Tags: composition-fixture

[{ID_GR_MEDIUM}]
Type: Guardrail
Date: 2026-08-01
Status: active
Statement: Widget rollout ledger migration secrets never enter source control.
Severity: medium
TriggerTools: Bash
Tags: composition-fixture
"""

CONTRADICTIONS = f"""
[C-20260801-001]
Date: 2026-08-01
Severity: high
Type: decision_vs_decision
Statement: Block {ID_CONTRADICTED} conflicts with the active rollout decision
Status: open
Resolution: none
"""

GUARDRAIL_CTX: dict[str, Any] = {"tool": "Bash", "command": "git reset --hard HEAD~1"}

#: Guardrail source + bound. The guardrails live in the ranked corpus so the
#: ranker retrieves them too — that is what makes "promoted in place" and
#: "re-injected after being cut" both reachable.
GUARDRAIL_CFG: dict[str, Any] = {
    "enabled": True,
    "sources": ["decisions/DECISIONS.md"],
    "max_surfaced": 3,
}


# ---------------------------------------------------------------------------
# Fixture corpus
# ---------------------------------------------------------------------------


def _build_master(root: str, *, persist_world: bool) -> str:
    """Seed one workspace: corpus, outcome evidence, optional world rows.

    Evidence is seeded unconditionally — the flags decide whether the
    pipeline *reads* it, so a flag-off combination is proven inert against a
    corpus that has something to find.
    """
    ws = os.path.join(root, "ws")
    project = os.path.join(root, "project")
    os.makedirs(os.path.join(project, "src"), exist_ok=True)
    with open(os.path.join(project, "src", "live.py"), "w", encoding="utf-8") as fh:
        fh.write("def live():\n    return 1\n")

    os.makedirs(ws)
    init(ws)
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as fh:
        fh.write(DECISIONS)
    os.makedirs(os.path.join(ws, "intelligence"), exist_ok=True)
    with open(os.path.join(ws, "intelligence", "CONTRADICTIONS.md"), "w", encoding="utf-8") as fh:
        fh.write(CONTRADICTIONS)

    for task in ("t1", "t2", "t3", "t4"):
        report_outcome(ws, [ID_OUTCOME_FAIL], "failure", task_id=task, actor_id="composition-fixture")
    for task in ("s1", "s2", "s3"):
        report_outcome(ws, [ID_CLEAN], "success", task_id=task, actor_id="composition-fixture")

    if persist_world:
        _write_config(ws, v4={"world_staleness": {"enabled": True, "roots": [project], "inline": False}})
        report = world_staleness_report(ws)
        assert set(report.stale_blocks) == {ID_WORLD_STALE, ID_GR_DEMOTABLE}, report.stale_blocks
        assert persist_world_staleness(ws, report)
    return ws


def _write_config(ws: str, *, recall_cfg: dict[str, Any] | None = None, v4: dict[str, Any] | None = None) -> None:
    cfg_path = os.path.join(ws, "mind-mem.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if recall_cfg is not None:
        cfg.setdefault("recall", {}).update(recall_cfg)
    if v4 is not None:
        cfg.setdefault("v4", {}).update(v4)
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)


@pytest.fixture(scope="module")
def masters(tmp_path_factory) -> dict[bool, str]:
    """Two seeded workspaces: with and without persisted world-staleness rows."""
    return {
        persist: _build_master(str(tmp_path_factory.mktemp(f"master_world_{int(persist)}")), persist_world=persist)
        for persist in (False, True)
    }


def _workspace(masters: dict[bool, str], tmp_path, name: str, *, world: bool, recall_cfg: dict[str, Any]) -> str:
    """A private copy of a master, configured for one flag combination.

    A fresh path per combination sidesteps ``_recall_core``'s mtime-keyed
    config cache entirely — no two combinations ever share a cache entry.
    """
    ws = str(tmp_path / name)
    shutil.copytree(masters[world], ws)
    _write_config(ws, recall_cfg=recall_cfg)
    return ws


# ---------------------------------------------------------------------------
# Flag matrix
# ---------------------------------------------------------------------------


def _recall_cfg(*, gate: bool, provenance: bool, outcome: bool, guardrails: bool) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if gate:
        vg: dict[str, Any] = {"enabled": True}
        if provenance:
            vg["provenance_class"] = {"enabled": True}
        if outcome:
            vg["outcome_attribution"] = {"enabled": True}
        cfg["validity_gate"] = vg
    if guardrails:
        cfg["guardrails"] = dict(GUARDRAIL_CFG)
    return cfg


#: (gate, provenance, outcome, world, guardrails) — 32 combinations.
COMBOS = list(itertools.product([False, True], repeat=5))


def _combo_id(combo: tuple[bool, ...]) -> str:
    names = ("gate", "prov", "outcome", "world", "rails")
    on = [n for n, flag in zip(names, combo) if flag]
    return "+".join(on) if on else "all_off"


def _fingerprint(hits: list[dict[str, Any]]) -> str:
    """Whole-response fingerprint — every key of every hit, order preserved."""
    return json.dumps(hits, sort_keys=True, default=repr)


def _ids(hits: list[dict[str, Any]]) -> list[str]:
    return [h["_id"] for h in hits]


@pytest.mark.parametrize("combo", COMBOS, ids=_combo_id)
def test_every_flag_combination_is_sane_and_deterministic(masters, tmp_path, combo) -> None:
    """(a) + (b): no combination raises, and each is byte-identical over 3 runs.

    Determinism here is the property the release actually claims — *same
    stored state in, same response out* — so each of the three runs gets its
    own pristine copy of the identical seeded workspace. Repeating the query
    against one workspace instead measures something else: Stage 2.8's
    co-retrieval graph learns from every logged retrieval, and that loop
    predates this release (see
    :func:`test_repeated_recall_drift_is_the_pre_existing_co_retrieval_loop`).
    """
    gate, provenance, outcome, world, guardrails = combo
    cfg = _recall_cfg(gate=gate, provenance=provenance, outcome=outcome, guardrails=guardrails)
    ctx = GUARDRAIL_CTX if guardrails else None

    runs = [
        recall(
            _workspace(masters, tmp_path, f"ws_{_combo_id(combo)}_{i}", world=world, recall_cfg=cfg),
            QUERY,
            limit=10,
            guardrail_context=ctx,
        )
        for i in range(3)
    ]
    prints = {_fingerprint(r) for r in runs}
    assert len(prints) == 1, f"non-deterministic recall for {_combo_id(combo)}"

    hits = runs[0]
    assert hits, f"empty recall for {_combo_id(combo)}"
    assert len(_ids(hits)) == len(set(_ids(hits))), f"duplicate ids for {_combo_id(combo)}"
    for hit in hits:
        assert isinstance(hit["score"], float) and hit["score"] >= 0.0

    # Ranked tail stays sorted; only force-surfaced guardrails sit out of order.
    ranked = [h["score"] for h in hits if not h.get("guardrail")]
    assert ranked == sorted(ranked, reverse=True), f"ranked tail out of order for {_combo_id(combo)}"

    # The gate annotates iff it is on — no leakage from the other features.
    annotated = [h for h in hits if "validity" in h]
    if gate:
        assert annotated, "gate on but nothing annotated"
    else:
        assert not annotated, f"validity annotation leaked with the gate off ({_combo_id(combo)})"


def test_the_matrix_is_not_vacuous_the_flags_really_move_the_response(masters, tmp_path) -> None:
    """Guard against a green matrix that proves nothing.

    If every combination returned the same thing, "deterministic in all 32"
    would be trivially true. Each flag must be observable in the response.
    """

    def _run(*, world: bool, ctx: dict[str, Any] | None, **flags: bool) -> list[dict[str, Any]]:
        cfg = _recall_cfg(
            gate=flags.get("gate", False),
            provenance=flags.get("provenance", False),
            outcome=flags.get("outcome", False),
            guardrails=ctx is not None,
        )
        name = f"ws_vac_{int(world)}_{int(ctx is not None)}_{''.join(str(int(v)) for v in flags.values())}"
        return recall(_workspace(masters, tmp_path, name, world=world, recall_cfg=cfg), QUERY, limit=10, guardrail_context=ctx)

    all_off = _run(world=True, ctx=None)
    gate_only = _run(world=True, ctx=None, gate=True)
    assert _fingerprint(all_off) != _fingerprint(gate_only)
    assert all("validity" not in h for h in all_off)

    # outcome attribution: only this flag can debit the failure-implicated block
    without = next(h for h in gate_only if h["_id"] == ID_OUTCOME_FAIL)
    with_outcome = next(h for h in _run(world=True, ctx=None, gate=True, outcome=True) if h["_id"] == ID_OUTCOME_FAIL)
    assert "outcome" not in without["validity"]
    assert with_outcome["validity"]["outcome"] == pytest.approx(0.5833)
    assert with_outcome["_validity_demoted"] is True

    # world-staleness rows: only their presence can zero the staleness criterion
    no_world = next(h for h in _run(world=False, ctx=None, gate=True) if h["_id"] == ID_WORLD_STALE)
    assert no_world["validity"]["staleness"] == 1.0
    assert "_validity_demoted" not in no_world

    # guardrails: only the context can put a constraint at position 0
    assert not all_off[0].get("guardrail")
    assert _run(world=True, ctx=GUARDRAIL_CTX)[0]["guardrail"] is True


def test_the_external_ingest_fixture_matches_what_the_importer_really_stamps() -> None:
    """The importers are the other half of this composition.

    ``mm import`` writes blocks that the validity gate's provenance
    criterion demotes to ``external-ingest``. This pins the fixture to the
    real stamp so the composition cannot silently stop being exercised if
    the importer's provenance shape changes.
    """
    from mind_mem.importers.engine import build_import_block
    from mind_mem.importers.records import ImportRecord
    from mind_mem.provenance_class import EXTERNAL_INGEST, classify_provenance

    real = build_import_block(ImportRecord(system="chroma", external_id="x-1", text="imported memory"))
    assert classify_provenance(real) == EXTERNAL_INGEST

    fixture_fields = {"ActorRole": "importer", "ToolId": "imported:chroma"}
    assert {k: real[k] for k in fixture_fields} == fixture_fields
    assert classify_provenance(fixture_fields) == EXTERNAL_INGEST


def test_seeded_evidence_is_inert_until_its_flag_is_on(masters, tmp_path) -> None:
    """The corpus carries outcome rows and world-staleness rows in both
    masters; with the gate off the response must not know it.

    This is the composition form of every feature's own flag-off proof: the
    evidence exists, several features are wired to the same tables, and the
    default path is still byte-identical.
    """
    with_rows = recall(_workspace(masters, tmp_path, "ws_inert_on", world=True, recall_cfg={}), QUERY, limit=10)
    without_rows = recall(_workspace(masters, tmp_path, "ws_inert_off", world=False, recall_cfg={}), QUERY, limit=10)
    assert _fingerprint(with_rows) == _fingerprint(without_rows)


def test_repeated_recall_drift_is_the_pre_existing_co_retrieval_loop(masters, tmp_path) -> None:
    """Repeating one query against one workspace is *not* byte-stable — and
    none of the 4.10.0 flags is why.

    Stage 2.8 propagates scores over the co-retrieval graph that
    ``log_retrieval`` writes on every call, so the weakest tail hit gains a
    ``_co_retrieval_boost`` on run 2 and again on run 3. Two things are
    pinned here: (1) the drift reproduces with **every** new flag off, given
    only a wide enough page (``knee_cutoff: false``), so it is inherited, not
    composed; (2) the drift is confined to boosted hits — every unboosted
    hit's id and score is stable across repeats.
    """
    ws = _workspace(masters, tmp_path, "ws_drift_flags_off", world=True, recall_cfg={"knee_cutoff": False})

    runs = [recall(ws, QUERY, limit=10) for _ in range(3)]
    assert len({_fingerprint(r) for r in runs}) > 1, "expected the inherited co-retrieval drift"
    assert any(h.get("_co_retrieval_boost") for h in runs[-1]), "drift without a co-retrieval boost"

    def _stable(hits: list[dict[str, Any]]) -> list[tuple[str, float]]:
        return [(h["_id"], h["score"]) for h in hits if not h.get("_co_retrieval_boost")]

    assert _stable(runs[0])[: len(_stable(runs[2]))] == _stable(runs[2])[: len(_stable(runs[0]))]


@pytest.mark.parametrize("combo", [c for c in COMBOS if c[4]], ids=_combo_id)
def test_guardrail_displacement_stays_bounded_whatever_else_is_on(masters, tmp_path, combo) -> None:
    """(c): the other features never let guardrails displace more than the bound."""
    gate, provenance, outcome, world, _ = combo
    cfg = _recall_cfg(gate=gate, provenance=provenance, outcome=outcome, guardrails=True)
    bound = 1
    cfg["guardrails"]["max_surfaced"] = bound
    ws = _workspace(masters, tmp_path, f"ws_disp_{_combo_id(combo)}", world=world, recall_cfg=cfg)

    limit = 3
    without = recall(ws, QUERY, limit=limit)
    with_rails = recall(ws, QUERY, limit=limit, guardrail_context=GUARDRAIL_CTX)

    surfaced = [h for h in with_rails if h.get("guardrail")]
    assert 0 < len(surfaced) <= bound <= MAX_SURFACED_HARD_CAP
    assert surfaced == with_rails[: len(surfaced)], "guardrails must lead the response"

    # The response never grows, and at most `bound` ranked hits are pushed out.
    assert len(with_rails) <= max(len(without), len(surfaced))
    displaced = [bid for bid in _ids(without) if bid not in set(_ids(with_rails))]
    assert len(displaced) <= bound, f"{_combo_id(combo)} displaced {displaced}"


@pytest.mark.parametrize("combo", [c for c in COMBOS if c[4]], ids=_combo_id)
def test_max_surfaced_bounds_a_multi_guardrail_fire(masters, tmp_path, combo) -> None:
    """(c): three guardrails fire; the bound still holds with everything else on."""
    gate, provenance, outcome, world, _ = combo
    cfg = _recall_cfg(gate=gate, provenance=provenance, outcome=outcome, guardrails=True)
    cfg["guardrails"]["max_surfaced"] = 2
    ws = _workspace(masters, tmp_path, f"ws_multi_{_combo_id(combo)}", world=world, recall_cfg=cfg)

    hits = recall(ws, QUERY, limit=5, guardrail_context=GUARDRAIL_CTX)
    surfaced = [h["_id"] for h in hits if h.get("guardrail")]
    assert len(surfaced) == 2, surfaced
    # Deterministic (severity, id) order: critical before high.
    assert surfaced == [ID_GR_DEMOTABLE, ID_GR_HIGH]


class TestDemotionVersusForcedSurfacing:
    """(d): the two mechanisms must not fight."""

    ALL_ON = dict(gate=True, provenance=True, outcome=True, guardrails=True)

    def test_the_gate_really_does_demote_the_guardrail(self, masters, tmp_path) -> None:
        """Precondition: world-staleness + external provenance push V under the bar."""
        ws = _workspace(masters, tmp_path, "ws_precondition", world=True, recall_cfg=_recall_cfg(**self.ALL_ON))
        ranked = recall(ws, QUERY, limit=10)
        rail = next(h for h in ranked if h["_id"] == ID_GR_DEMOTABLE)
        assert rail["validity"]["staleness"] == 0.0, "world-staleness row not reaching the gate"
        assert rail["validity"]["provenance_class"] == "external-ingest"
        assert rail["validity"]["score"] == pytest.approx(0.65)
        assert rail["_validity_demoted"] is True

    def test_a_demoted_guardrail_still_surfaces_first(self, masters, tmp_path) -> None:
        ws = _workspace(masters, tmp_path, "ws_demoted_first", world=True, recall_cfg=_recall_cfg(**self.ALL_ON))
        hits = recall(ws, QUERY, limit=10, guardrail_context=GUARDRAIL_CTX)
        assert hits[0]["_id"] == ID_GR_DEMOTABLE
        assert hits[0]["guardrail"] is True
        assert hits[0]["surfaced_by"] == "guardrail_trigger"

    def test_a_guardrail_demoted_below_the_cut_is_re_injected(self, masters, tmp_path) -> None:
        """The hard case: demotion drops it out of the page entirely."""
        ws = _workspace(masters, tmp_path, "ws_demoted_cut", world=True, recall_cfg=_recall_cfg(**self.ALL_ON))
        limit = 3
        ranked = recall(ws, QUERY, limit=limit)
        assert ID_GR_DEMOTABLE not in _ids(ranked), "precondition: demotion must cut it from the page"

        hits = recall(ws, QUERY, limit=limit, guardrail_context=GUARDRAIL_CTX)
        assert hits[0]["_id"] == ID_GR_DEMOTABLE, "a demoted guardrail was lost"
        assert hits[0]["guardrail"] is True

    def test_a_post_filter_that_empties_the_page_still_surfaces_it(self, masters, tmp_path) -> None:
        """A filter aimed at evidence must not silence a constraint."""
        ws = _workspace(masters, tmp_path, "ws_filtered", world=True, recall_cfg=_recall_cfg(**self.ALL_ON))
        filtered = recall(ws, QUERY, limit=10, min_maturity=1.0)
        hits = recall(ws, QUERY, limit=10, min_maturity=1.0, guardrail_context=GUARDRAIL_CTX)
        assert len(filtered) < 3, "precondition: the filter must bite"
        assert hits[0]["_id"] == ID_GR_DEMOTABLE


class TestBackendAsymmetry:
    """Where the two mechanisms stop composing: a non-default recall backend.

    ``apply_validity_gate`` is Stage 2.65 of the BM25 scan pipeline, but the
    ``sqlite`` and vector backends return from ``recall()`` long before it
    (``_recall_core.recall`` early-returns through ``_apply_post_filters``
    around line 710). Guardrail surfacing lives *inside*
    ``_apply_post_filters``, so it survives that early return and the gate
    does not: on those backends guardrails fire and nothing is ever scored
    for validity — no annotation, no demotion, and no provenance / outcome /
    world-staleness evidence read at all.

    ``recall.backend`` is ``"sqlite"`` for a Postgres block store
    (``init_workspace._BACKEND_RECALL``), so this is a real deployment, not a
    hypothetical one. Pinned as a tripwire, not as an endorsement: if the
    gate is ever hoisted into the shared funnel, this test is the one that
    should fail and be rewritten.
    """

    def _sqlite_ws(self, masters, tmp_path) -> str:
        from mind_mem.sqlite_index import build_index

        cfg = _recall_cfg(gate=True, provenance=True, outcome=True, guardrails=True)
        cfg["backend"] = "sqlite"
        ws = _workspace(masters, tmp_path, "ws_sqlite", world=True, recall_cfg=cfg)
        build_index(ws)
        return ws

    def test_guardrails_still_surface_on_the_sqlite_backend(self, masters, tmp_path) -> None:
        hits = recall(self._sqlite_ws(masters, tmp_path), QUERY, limit=10, guardrail_context=GUARDRAIL_CTX)
        assert hits[0]["guardrail"] is True
        assert hits[0]["_id"] == ID_GR_DEMOTABLE

    def test_the_validity_gate_does_not_run_on_the_sqlite_backend(self, masters, tmp_path) -> None:
        hits = recall(self._sqlite_ws(masters, tmp_path), QUERY, limit=10, guardrail_context=GUARDRAIL_CTX)
        assert hits, "precondition: the sqlite path returns something"
        assert all("validity" not in h for h in hits), "gate reached the sqlite path — update this test"
        assert all("_validity_demoted" not in h for h in hits)


class TestDocumentedCompositionQuirks:
    """Behaviours that only exist in combination. Characterised deliberately.

    Neither is a crash and neither is silent — both are visible in the
    ``validity`` annotation — but both change a *decision* that the same
    corpus gets with one fewer flag on, which is exactly what a composition
    gate is for.
    """

    def test_provenance_class_can_rescue_a_block_the_four_criteria_gate_demotes(self, masters, tmp_path) -> None:
        """A neutral fifth criterion is not neutral to the *threshold*.

        ``D-…-002`` is world-stale and carries no provenance fields, so its
        class is ``unknown`` (weight 1.0). Four criteria: 0.25*(1+1+1+0) =
        0.75, below the 0.8 bar — demoted. Five criteria: 0.2*(1+1+1+0+1) =
        0.80, exactly at the bar (``V < threshold`` fires) — not demoted.
        Turning the provenance sub-flag ON therefore *un-demotes* a
        world-stale block. Widening the mean dilutes every existing debit;
        that is inherent to a mean, not a bug, but it is a real cross-flag
        change of outcome and it is pinned here.
        """
        off = _workspace(
            masters,
            tmp_path,
            "ws_prov_off",
            world=True,
            recall_cfg=_recall_cfg(gate=True, provenance=False, outcome=False, guardrails=False),
        )
        on = _workspace(
            masters,
            tmp_path,
            "ws_prov_on",
            world=True,
            recall_cfg=_recall_cfg(gate=True, provenance=True, outcome=False, guardrails=False),
        )
        stale_off = next(h for h in recall(off, QUERY, limit=10) if h["_id"] == ID_WORLD_STALE)
        stale_on = next(h for h in recall(on, QUERY, limit=10) if h["_id"] == ID_WORLD_STALE)

        assert stale_off["validity"]["score"] == pytest.approx(0.75)
        assert stale_off["_validity_demoted"] is True
        assert stale_on["validity"]["score"] == pytest.approx(0.80)
        assert "_validity_demoted" not in stale_on

    def test_a_re_injected_guardrail_loses_its_demotion_annotation(self, masters, tmp_path) -> None:
        """The demotion markers are path-dependent, by construction.

        ``guardrail_to_hit`` copies the ranked hit when the ranker still had
        one (markers preserved) and otherwise renders a fresh hit from the
        block (score 0.0, no ``validity`` key). So a consumer sees "this
        guardrail's cited file is gone" only when the guardrail survived the
        page — never when demotion cut it. The constraint still surfaces
        either way, which is the contract; the diagnostic does not travel
        with it.
        """
        ws = _workspace(masters, tmp_path, "ws_marker_loss", world=True, recall_cfg=_recall_cfg(**TestDemotionVersusForcedSurfacing.ALL_ON))

        survived = recall(ws, QUERY, limit=10, guardrail_context=GUARDRAIL_CTX)[0]
        assert survived["_id"] == ID_GR_DEMOTABLE
        assert survived["_validity_demoted"] is True
        assert survived["validity"]["score"] == pytest.approx(0.65)

        re_injected = recall(ws, QUERY, limit=3, guardrail_context=GUARDRAIL_CTX)[0]
        assert re_injected["_id"] == ID_GR_DEMOTABLE
        assert "validity" not in re_injected
        assert "_validity_demoted" not in re_injected
        assert re_injected["score"] == 0.0
