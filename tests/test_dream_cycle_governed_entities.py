#!/usr/bin/env python3
"""The dream cycle's auto-created entities are governed blocks (AUD-06).

``dream_cycle._create_entity_file`` used to write
``entities/<PREFIX>-<slug>.md`` with a bare ``open(..., "w")``. Measured on
a fresh ``init()`` workspace before the fix::

    created: entities/PRJ-widgetco.md
    parse_file blocks: 0
    store.get_by_id('PRJ-widgetco'): None
    in get_all ids: False
    in iter_active_blocks: False
    evidence rows before/after: 0 0   hash_chain before/after: 0 0
    file in list_blocks: True

Three separate defects in one write: it was **ungoverned** (both ledgers
+0), it was **unresolvable** (every reader answered "no such block" while
``list_blocks`` counted the file, so it inflated ``GET /status``
``memory_count``), and the ``RepairAction`` it produced named an id that
did not exist. It was also invisible to the write-path scanner, because
the basename it minted is in no registry — the hole that
:func:`_write_path_scan.corpus_dir_hit` now closes.

Reachable from ``POST /consolidate {"auto_repair": true}`` and from the
daemon, so this is a live door, not dead code.
"""

from __future__ import annotations

import json
import os

import pytest
from _ledger_rows import count_chain_authorisations, count_evidence_authorisations

from mind_mem.dream_cycle import (
    _ENTITY_PREFIX_MAP,
    ENTITY_STATUS,
    DreamCycleReport,
    EntityProposal,
    _create_entity_block,
    pass_auto_repair,
)
from mind_mem.enums import INITIAL_STATUS, IngestTier, is_servable
from mind_mem.init_workspace import init


def _proposal(slug: str = "widgetco", entity_type: str = "project") -> EntityProposal:
    return EntityProposal(
        entity_type=entity_type,
        slug=slug,
        source_pattern="github_url",
        excerpt="saw https://github.com/acme/" + slug + " in the log",
        source_file="memory/2026-09-02.md",
    )


def _evidence_admissions(ws: str) -> int:
    """Governed scopes that have authorised something, per the evidence chain.

    Not raw rows: a scope leaves an authorisation *and* a close record
    saying whether the write landed, and every count here means the
    first. ``tests/_ledger_rows`` holds that convention once.
    """
    return count_evidence_authorisations(ws)


def _chain_admissions(ws: str) -> int:
    """The same count from the hash chain."""
    return count_chain_authorisations(ws)


@pytest.fixture()
def ws(tmp_path) -> str:
    workspace = str(tmp_path / "wsp")
    init(workspace)
    return workspace


# ---------------------------------------------------------------------------
# The block is resolvable — the half the old code got wrong most quietly
# ---------------------------------------------------------------------------


def test_created_entity_is_returned_by_get_by_id(ws: str) -> None:
    """The id the RepairAction names is an id ``get_by_id`` resolves."""
    from mind_mem.storage import get_block_store

    block_id = _create_entity_block(ws, _proposal())
    assert block_id == "PRJ-widgetco"

    store = get_block_store(ws)
    block = store.get_by_id(block_id)
    assert block is not None, "the auto-created entity is unresolvable — this is the pre-fix behaviour"
    assert block["_id"] == block_id
    assert block_id in {b.get("_id") for b in store.get_all(active_only=False)}


def test_created_entity_lands_pending_and_recall_withholds_it(ws: str) -> None:
    """Withheld until released, and withheld for the RIGHT reason.

    The status is asserted against ``INITIAL_STATUS`` rather than the
    literal ``"pending"``: the tier table is the one place that decides
    this, and a hand-spelled copy here would keep passing after a retier
    that made the block servable.
    """
    from mind_mem.recall import recall
    from mind_mem.storage import get_block_store, iter_active_blocks

    block_id = _create_entity_block(ws, _proposal())
    block = get_block_store(ws).get_by_id(block_id)
    assert block is not None

    expected = INITIAL_STATUS[IngestTier.AUTO_CAPTURE]
    assert expected is not None
    assert block["Status"] == expected.value
    assert not is_servable(expected), "AUTO_CAPTURE stopped minting a withheld status; auto-discovered entities would now be published"

    assert block_id not in {b.get("_id") for b in iter_active_blocks(ws)}
    assert block_id not in {h.get("_id") for h in recall(ws, "widgetco", limit=25)}


def test_released_entity_becomes_recallable(ws: str) -> None:
    """The round trip: withheld is a STATE, not a dead end.

    A quarantine you cannot leave is a deletion with extra steps, so the
    exit has to be tested and not just the entrance.
    """
    from mind_mem.enums import Status
    from mind_mem.governance_gate import get_gate
    from mind_mem.storage import get_block_store, iter_active_blocks

    block_id = _create_entity_block(ws, _proposal())
    store = get_block_store(ws)
    block = dict(store.get_by_id(block_id) or {})
    assert block, "nothing to release"

    block["Status"] = Status.ACTIVE.value
    with get_gate(ws).admit_proposal(
        "P-test-release-entity",
        block_id,
        actor="test",
        target_file=os.path.join("entities", "projects.md"),
    ):
        store.write_block(block)

    store.invalidate_cache()
    assert block_id in {b.get("_id") for b in iter_active_blocks(ws)}


# ---------------------------------------------------------------------------
# The write is governed — both ledgers, measured
# ---------------------------------------------------------------------------


def test_created_entity_appends_to_both_ledgers(ws: str) -> None:
    """Evidence chain and hash chain each gain a row.

    Both numbers are read BEFORE as well as after. A bare ``after >= 1``
    would also pass on a workspace that arrived with rows already in it,
    which is how a governance assertion ends up proving the fixture rather
    than the code.
    """
    before_evidence, before_chain = _evidence_admissions(ws), _chain_admissions(ws)
    _create_entity_block(ws, _proposal())
    assert _evidence_admissions(ws) == before_evidence + 1
    assert _chain_admissions(ws) == before_chain + 1


def test_no_ungoverned_entity_file_is_left_behind(ws: str) -> None:
    """The block lands in the routed corpus file, not a minted one.

    ``entities/`` is a corpus directory, so ``_discover_files`` picks up
    every ``.md`` in it. A stray ``PRJ-widgetco.md`` would be inside the
    store's read set and outside every registry — exactly the shape the
    widened scan D exists to prevent.
    """
    _create_entity_block(ws, _proposal())
    names = sorted(p for p in os.listdir(os.path.join(ws, "entities")) if p.endswith(".md"))
    assert "PRJ-widgetco.md" not in names
    with open(os.path.join(ws, "entities", "projects.md"), "r", encoding="utf-8") as handle:
        assert "[PRJ-widgetco]" in handle.read()


def test_write_is_refused_without_an_admission(ws: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION CONTROL: remove the gate scope and the write must fail.

    Proves the admission is load-bearing rather than decorative. Without
    this, every assertion above would pass just as well if the
    ``admit_block`` context manager were replaced by ``nullcontext()``.
    """
    import contextlib

    import mind_mem.governance_gate as gg

    class _Ungated:
        def admit_block(self, *args, **kwargs):
            return contextlib.nullcontext()

    monkeypatch.setattr(gg, "get_gate", lambda _ws: _Ungated())

    from mind_mem.admission import UngatedWriteError

    with pytest.raises(UngatedWriteError):
        _create_entity_block(ws, _proposal(slug="ungated"))


# ---------------------------------------------------------------------------
# Dedup and the repair-pass contract
# ---------------------------------------------------------------------------


def test_second_call_is_a_no_op(ws: str) -> None:
    """Dedup now asks the STORE, not the filesystem.

    The old check was ``os.path.exists(filepath)``, which is blind on a
    non-Markdown backend (the on-disk corpus files stay empty templates
    there), so the same entity would be recreated on every run.
    """
    assert _create_entity_block(ws, _proposal()) == "PRJ-widgetco"
    assert _create_entity_block(ws, _proposal()) is None


def test_auto_repair_reports_an_id_that_resolves(ws: str) -> None:
    """End to end through ``pass_auto_repair``: the reported target exists.

    This is the defect a reader would actually hit. ``RepairAction(target=...)``
    was already the entity id before the fix; what changed is that the id
    now names something.
    """
    from mind_mem.storage import get_block_store

    report = DreamCycleReport(
        timestamp="2026-09-02T00:00:00",
        workspace=ws,
        entity_proposals=(_proposal(), _proposal("acmecorp")),
    )
    actions = pass_auto_repair(ws, report)

    created = [a for a in actions if a.action_type == "entity_created"]
    assert {a.target for a in created} == {"PRJ-widgetco", "PRJ-acmecorp"}

    store = get_block_store(ws)
    for action in created:
        assert store.get_by_id(action.target) is not None, f"auto-repair reported {action.target} but no such block exists"
        assert ENTITY_STATUS.value in action.detail


def test_consolidate_route_reaches_the_governed_writer(ws: str) -> None:
    """The HTTP door is the reason this is a live path, so exercise it.

    ``POST /consolidate {"auto_repair": true}`` -> ``run_dream_cycle`` ->
    ``run_auto_repair`` -> the governed writer. Asserting the route returns
    200 proves reachability; asserting the ledger moved proves the write
    that route performs is the governed one.
    """
    from mind_mem.http_transport import _handle_consolidate

    os.makedirs(os.path.join(ws, "memory"), exist_ok=True)
    log = os.path.join(ws, "memory", "2026-09-02.md")
    with open(log, "w", encoding="utf-8") as handle:
        handle.write("Reviewed https://github.com/acme/widgetco today.\n")

    before = _chain_admissions(ws)
    # ``actor`` is keyword-only with no default: the route cannot be reached
    # without the identity the credential resolved to.
    status, body = _handle_consolidate(ws, {"auto_repair": True}, actor="test-operator")
    assert status == 200, body
    assert body["auto_repair"] is True
    # Either nothing was discovered (chain unmoved) or every discovery went
    # through the gate. What must never happen is a discovery with no row.
    from mind_mem.storage import get_block_store

    store = get_block_store(ws)
    entity_prefixes = tuple(f"{p}-" for p in _ENTITY_PREFIX_MAP.values())
    minted = [b for b in store.get_all(active_only=False) if str(b.get("_id", "")).startswith(entity_prefixes)]
    assert _chain_admissions(ws) - before == len(minted)


# ---------------------------------------------------------------------------
# Fix-by-construction: an entity type with no routable prefix is impossible
# ---------------------------------------------------------------------------


def test_every_entity_prefix_routes_to_a_corpus_file() -> None:
    """No ``UNK`` fallback survives, and none can come back.

    The old code read the prefix map with ``.get(entity_type, "UNK")`` and
    wrote ``entities/UNK-<slug>.md``. Through ``write_block`` that id has no
    canonical file and raises — so the guarantee has to be that the case
    cannot arise, not that it is handled. ``_assert_entity_prefixes_route``
    enforces it at import; this pins both directions of the same fact.
    """
    from mind_mem.corpus_registry import BLOCK_PREFIX_MAP
    from mind_mem.dream_cycle import _ENTITY_GROUPS

    assert {etype for etype, _p, _i, _m in _ENTITY_GROUPS} <= set(_ENTITY_PREFIX_MAP)
    unroutable = sorted(p for p in _ENTITY_PREFIX_MAP.values() if p not in BLOCK_PREFIX_MAP)
    assert not unroutable, f"entity prefixes {unroutable} cannot be written by write_block"


def test_import_time_guard_rejects_an_unroutable_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION CONTROL for the guard itself.

    A guard nobody can watch fail is a comment. Point the map at a prefix
    the registry does not route and the check must raise.
    """
    import mind_mem.dream_cycle as dc

    monkeypatch.setitem(dc._ENTITY_PREFIX_MAP, "project", "NOPE")
    with pytest.raises(RuntimeError, match="NOPE"):
        dc._assert_entity_prefixes_route()


def test_status_is_read_from_the_tier_table_not_a_literal() -> None:
    """``ENTITY_STATUS`` is derived, so a retier moves the block with it."""
    assert ENTITY_STATUS is INITIAL_STATUS[IngestTier.AUTO_CAPTURE]


def test_created_block_json_roundtrips(ws: str) -> None:
    """The stored block survives a JSON round trip (no stray objects)."""
    from mind_mem.storage import get_block_store

    block_id = _create_entity_block(ws, _proposal())
    block = get_block_store(ws).get_by_id(block_id)
    assert json.loads(json.dumps(block)) == block
