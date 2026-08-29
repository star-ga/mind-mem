"""RA.1 — the served-set ledger: proof of what was served, joinable to outcome.

Everyone measures retrieval. The join that is not available anywhere else is
*which exact answer* was served on a given run, so a later outcome can be
credited to that run rather than correlated against a query string. That join
only holds if the record of the answer is append-only, content-derived, and
structurally unable to feed back into the ranking it describes.

Four properties carry that, and each has a test here:

T13  the row carries exactly nine fields and no tenth. A per-item score or a
     ``degraded`` marker in a ledger row is the beginning of a feedback loop.
T14  ``run_id`` names THE ANSWER, stably across days. Two runs of the same
     answer on different scoring instants share a ``run_id``; ``seq`` is what
     distinguishes the rows.
T16  a flipped byte is caught, and the verdict names the row it was in.
T17  the write path reads no clock. Not "uses the injected one where
     convenient" — the accessor is monkeypatched to *raise*.

T12, the import rail, lives in ``test_recall_attestation_v2.py`` next to the
closure walker that already enforces it for the attestation.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from mind_mem.recall_digests import query_hash, served_set_digest
from mind_mem.served_ledger import (
    GENESIS_ROW_HASH,
    LEDGER_RELPATH,
    RUN_TAG,
    ServedRun,
    append_served_run,
    ledger_enabled,
    ledger_path,
    read_served_runs,
    row_hash,
    run_id,
    verify_served_chain,
)

PIPELINE = "b" * 64
ANCHOR = "c" * 64

#: The row, exactly. RA.1 names these nine and no others: an attestation, a
#: ``degraded`` marker, a leg list or any per-item score in a ledger row is a
#: stored judgement about an answer, and a stored judgement is the thing a
#: later ranking learns to read back.
ROW_FIELDS = frozenset(
    {
        "seq",
        "prev_row_hash",
        "run_id",
        "query_hash",
        "served_digest",
        "ids",
        "pipeline_hash",
        "index_anchor",
        "scoring_instant",
    }
)


#: One fully-populated row, for the checks that need a value rather than a
#: workspace. Built from the schema's own field names so it cannot drift.
_ROW = ServedRun(
    seq=0,
    prev_row_hash=GENESIS_ROW_HASH,
    run_id=run_id(query_hash=query_hash("q"), served_digest=served_set_digest(["D-1"]), pipeline_hash=PIPELINE),
    query_hash=query_hash("q"),
    served_digest=served_set_digest(["D-1"]),
    ids=("D-1",),
    pipeline_hash=PIPELINE,
    index_anchor=ANCHOR,
    scoring_instant="2026-08-29",
)


def _ws(tmp_path: Path, *, enabled: bool) -> str:
    ws = tmp_path / ("on" if enabled else "off")
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(json.dumps({"served_ledger": {"enabled": enabled}}), encoding="utf-8")
    return str(ws)


def _append(ws: str, ids: list[str], instant: str = "2026-08-29", question: str = "q") -> ServedRun | None:
    return append_served_run(
        ws,
        query_hash=query_hash(question),
        served_digest=served_set_digest(ids),
        ids=ids,
        pipeline_hash=PIPELINE,
        index_anchor=ANCHOR,
        scoring_instant=instant,
    )


# ---------------------------------------------------------------------------
# T13 — the row is exactly nine fields
# ---------------------------------------------------------------------------


def test_t13_the_row_schema_is_exactly_the_nine_named_fields() -> None:
    """An equality, not a superset check: adding ``degraded`` fails the build."""
    assert dataclasses.is_dataclass(ServedRun)
    assert {f.name for f in dataclasses.fields(ServedRun)} == ROW_FIELDS


def test_t13_the_row_is_frozen() -> None:
    """Append-only evidence that can be mutated in place is not evidence."""
    row = ServedRun(
        seq=0,
        prev_row_hash=GENESIS_ROW_HASH,
        run_id="d" * 64,
        query_hash=query_hash("q"),
        served_digest=served_set_digest(["D-1"]),
        ids=("D-1",),
        pipeline_hash=PIPELINE,
        index_anchor=ANCHOR,
        scoring_instant="2026-08-29",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.seq = 1  # type: ignore[misc]


def test_t13_the_persisted_row_carries_no_field_the_schema_does_not(tmp_path: Path) -> None:
    """The on-disk shape *is* the schema — a tenth field cannot arrive by JSON."""
    ws = _ws(tmp_path, enabled=True)
    _append(ws, ["D-1", "D-2"])
    line = Path(ledger_path(ws)).read_text(encoding="utf-8").splitlines()[0]
    assert set(json.loads(line)) == ROW_FIELDS


# ---------------------------------------------------------------------------
# T14 — off writes nothing; on chains; run_id is stable across instants
# ---------------------------------------------------------------------------


def test_t14_ledger_off_creates_no_file(tmp_path: Path) -> None:
    ws = _ws(tmp_path, enabled=False)
    assert ledger_enabled(ws) is False
    assert _append(ws, ["D-1", "D-2"]) is None
    assert not Path(ledger_path(ws)).exists()
    assert not (Path(ws) / LEDGER_RELPATH).parent.exists()


def test_t14_ledger_absent_from_config_is_off(tmp_path: Path) -> None:
    """Default OFF means *absent* is off, not just ``false``."""
    ws = tmp_path / "bare"
    ws.mkdir()
    (ws / "mind-mem.json").write_text(json.dumps({"recall": {}}), encoding="utf-8")
    assert ledger_enabled(str(ws)) is False
    assert _append(str(ws), ["D-1"]) is None


def test_t14_ledger_on_writes_a_chain_that_verifies(tmp_path: Path) -> None:
    ws = _ws(tmp_path, enabled=True)
    _append(ws, ["D-1", "D-2"], question="first")
    _append(ws, ["D-3"], question="second")
    _append(ws, ["D-1", "D-2"], question="first")

    rows = read_served_runs(ws)
    assert [r.seq for r in rows] == [0, 1, 2]
    assert rows[0].prev_row_hash == GENESIS_ROW_HASH
    assert rows[1].prev_row_hash == row_hash(rows[0])
    assert rows[2].prev_row_hash == row_hash(rows[1])

    verdict = verify_served_chain(ws)
    assert verdict.ok is True, verdict.reason
    assert verdict.rows_checked == 3
    assert verdict.bad_seq is None


def test_t14_run_id_is_identical_across_two_scoring_instants(tmp_path: Path) -> None:
    """``run_id`` identifies THE ANSWER, stably across days.

    Excluding the scoring instant is the amendment that makes "has this exact
    answer been served before?" answerable at all. A repeated ``run_id`` is a
    legitimate row; ``seq`` is the ledger's unique key.
    """
    ws = _ws(tmp_path, enabled=True)
    monday = _append(ws, ["D-1", "D-2"], instant="2026-08-24", question="same")
    friday = _append(ws, ["D-1", "D-2"], instant="2026-08-28", question="same")
    assert monday is not None and friday is not None

    assert monday.run_id == friday.run_id
    assert monday.scoring_instant != friday.scoring_instant
    assert monday.seq != friday.seq
    assert len({r.seq for r in read_served_runs(ws)}) == 2


def test_t14_run_id_is_keyed_on_rank_order_not_a_set(tmp_path: Path) -> None:
    """A reordered answer is a different answer.

    A ledger that stores rank order but keys on the unordered set has two
    distinct answers under one key, which an append-only ledger cannot be
    consistent with.
    """
    ws = _ws(tmp_path, enabled=True)
    forward = _append(ws, ["D-1", "D-2"], question="q")
    reverse = _append(ws, ["D-2", "D-1"], question="q")
    assert forward is not None and reverse is not None
    assert forward.run_id != reverse.run_id


def test_t14_run_id_matches_the_decided_preimage() -> None:
    """``SHA256("MM_RUN_v1\\0" || query_hash || served_digest || pipeline_hash)``.

    Spelled out here against hashlib rather than against the module, so a
    change to the encoding has to change this literal too.
    """
    import hashlib

    q = query_hash("q")
    s = served_set_digest(["D-1", "D-2"])
    expected = hashlib.sha256(RUN_TAG.encode("ascii") + b"\x00" + (q + s + PIPELINE).encode("ascii")).hexdigest()
    assert run_id(query_hash=q, served_digest=s, pipeline_hash=PIPELINE) == expected


def test_t14_run_id_excludes_the_scoring_instant_structurally() -> None:
    """Not "we chose not to pass it" — there is no parameter to pass it to."""
    import inspect

    params = set(inspect.signature(run_id).parameters)
    assert params == {"query_hash", "served_digest", "pipeline_hash"}


# ---------------------------------------------------------------------------
# T16 — tamper detection names the row
# ---------------------------------------------------------------------------


def _tamper(ws: str, seq: int, field: str, value: Any) -> None:
    path = Path(ledger_path(ws))
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[seq])
    row[field] = value
    lines[seq] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query_hash", "0" * 64),
        ("pipeline_hash", "9" * 64),
        ("index_anchor", "1" * 64),
        ("scoring_instant", "1999-01-01"),
        ("run_id", "f" * 64),
        ("prev_row_hash", "e" * 64),
    ],
)
def test_t16_a_flipped_field_breaks_the_chain_and_names_the_row(tmp_path: Path, field: str, value: Any) -> None:
    ws = _ws(tmp_path, enabled=True)
    for n in range(3):
        _append(ws, [f"D-{n}"], question=f"q{n}")
    assert verify_served_chain(ws).ok is True

    _tamper(ws, 1, field, value)
    verdict = verify_served_chain(ws)
    assert verdict.ok is False
    assert verdict.bad_seq == 1, f"tamper in row 1 reported at {verdict.bad_seq}"
    assert field in verdict.reason or "chain" in verdict.reason


def test_t16_editing_the_served_ids_alone_is_caught(tmp_path: Path) -> None:
    """The ids are what the ledger exists to prove. Swapping one without
    touching ``served_digest`` must not pass — otherwise the digest is
    decoration rather than a commitment."""
    ws = _ws(tmp_path, enabled=True)
    _append(ws, ["D-1", "D-2"], question="q")
    _tamper(ws, 0, "ids", ["D-1", "D-EVIL"])
    verdict = verify_served_chain(ws)
    assert verdict.ok is False
    assert verdict.bad_seq == 0
    assert "served_digest" in verdict.reason


def test_t16_a_deleted_row_breaks_the_chain(tmp_path: Path) -> None:
    """Append-only: removing history is a tamper, not a shrink."""
    ws = _ws(tmp_path, enabled=True)
    for n in range(3):
        _append(ws, [f"D-{n}"], question=f"q{n}")
    path = Path(ledger_path(ws))
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")

    verdict = verify_served_chain(ws)
    assert verdict.ok is False
    assert verdict.bad_seq == 1


# ---------------------------------------------------------------------------
# T17 — no clock on the write path
# ---------------------------------------------------------------------------


def test_t17_a_ledger_write_succeeds_with_the_clock_accessor_raising(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``_read_utc_today`` is THE named clock read in this package.

    Making it raise is the only honest way to assert the write path never
    reaches for "now": a passing test that merely injects an instant proves
    the parameter is threaded, not that nothing else reads a clock.
    """
    import mind_mem.scoring_instant as si

    ws = _ws(tmp_path, enabled=True)

    def _boom() -> Any:
        raise AssertionError("the ledger write path read a clock")

    monkeypatch.setattr(si, "_read_utc_today", _boom)

    row = _append(ws, ["D-1", "D-2"], instant="2026-08-29")
    assert row is not None
    assert row.scoring_instant == "2026-08-29"
    assert verify_served_chain(ws).ok is True


def test_t17_verification_reads_no_clock_either(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mind_mem.scoring_instant as si

    ws = _ws(tmp_path, enabled=True)
    _append(ws, ["D-1"])
    monkeypatch.setattr(si, "_read_utc_today", lambda: (_ for _ in ()).throw(AssertionError("clock")))
    assert verify_served_chain(ws).ok is True


# ---------------------------------------------------------------------------
# T16 (cont.) — the last row, and telling the two edits apart
# ---------------------------------------------------------------------------


def test_t16_the_last_row_is_sealed_by_the_head_sidecar(tmp_path: Path) -> None:
    """No successor binds the final row's ``index_anchor`` / ``scoring_instant``.

    Without the head file those two would be editable until the next append,
    which is a window an append-only ledger must not have. The sidecar is not
    a tenth row field — the schema stays nine.
    """
    ws = _ws(tmp_path, enabled=True)
    _append(ws, ["D-1"], question="a")
    _append(ws, ["D-2"], question="b")
    assert verify_served_chain(ws).ok is True

    _tamper(ws, 1, "scoring_instant", "1999-01-01")
    verdict = verify_served_chain(ws)
    assert verdict.ok is False
    assert verdict.bad_seq == 1


def test_t16_a_rewritten_prev_link_is_blamed_on_its_own_row(tmp_path: Path) -> None:
    """The two single-row edits produce different break patterns.

    Editing row j's sealed content breaks one link, at j+1. Editing row j's own
    ``prev_row_hash`` breaks two, at j and j+1. Reading the pattern is what
    lets the verdict accuse j in both cases instead of accusing j+1 of its
    predecessor's edit.
    """
    ws = _ws(tmp_path, enabled=True)
    for n in range(4):
        _append(ws, [f"D-{n}"], question=f"q{n}")

    _tamper(ws, 1, "prev_row_hash", "a" * 64)
    assert verify_served_chain(ws).bad_seq == 1

    ws2 = _ws(tmp_path / "second", enabled=True)
    for n in range(4):
        _append(ws2, [f"D-{n}"], question=f"q{n}")
    _tamper(ws2, 1, "index_anchor", "d" * 64)
    assert verify_served_chain(ws2).bad_seq == 1


def _perturb(value: Any) -> Any:
    """A different value of the same shape — whatever shape the field has.

    Type-driven rather than field-driven, so a field added to the schema
    tomorrow is perturbed by the same code that perturbs the nine today. A
    perturbation that returned the value unchanged would make the assertion
    below vacuous, so each branch is required to change it.
    """
    if isinstance(value, int):
        return value + 1
    if isinstance(value, list):
        return [*value, "D-EVIL"]
    if isinstance(value, str):
        if value and set(value) <= set("0123456789abcdef"):
            return ("1" if value[0] != "1" else "2") + value[1:]
        return value + "-TAMPERED"
    raise AssertionError(f"no perturbation for {type(value).__name__} — extend this before adding the field")


@pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(ServedRun)])
def test_t16_every_field_in_the_schema_is_sealed(tmp_path: Path, field: str) -> None:
    """EVERY declared field, derived from the schema — not a list of nine.

    The gap this closes: ``row_hash`` used to name its eight covered fields by
    hand, and nothing asserted that enumeration equalled the schema. A tenth
    field could be declared, serialised, and read back while contributing to
    no hash — an unsealed field sitting inside a row whose whole purpose is to
    be tamper-evident. Parametrising over ``dataclasses.fields`` means a field
    added without sealing it fails HERE, in the same commit that adds it,
    rather than being discovered by whoever later trusted the row.
    """
    ws = _ws(tmp_path, enabled=True)
    _append(ws, ["D-1", "D-2"], question="first")
    _append(ws, ["D-3"], question="second")
    assert verify_served_chain(ws).ok is True

    line = json.loads(Path(ledger_path(ws)).read_text(encoding="utf-8").splitlines()[0])
    before = line[field]
    after = _perturb(before)
    assert after != before, f"the perturbation of {field} changed nothing — the assertion would be vacuous"

    _tamper(ws, 0, field, after)
    verdict = verify_served_chain(ws)
    assert verdict.ok is False, f"{field} was edited on disk and the chain still verified"
    assert verdict.bad_seq == 0, f"{field} tamper in row 0 reported at row {verdict.bad_seq}"


def test_t16_the_only_unsealed_field_is_the_one_the_digest_covers() -> None:
    """``ids`` is the single exclusion, and it is not an exception to the rule.

    It is sealed transitively: ``served_digest`` commits to it, the digest is
    re-derived from the ids on every verification, and the digest itself is in
    the hash. Adding a second name to this set removes a field from the chain,
    which is why the set is asserted by equality rather than by membership.
    """
    from mind_mem.served_ledger import _HASH_EXCLUDED, _hashed_values

    assert _HASH_EXCLUDED == {"ids"}
    sealed = {f.name for f in dataclasses.fields(ServedRun)} - _HASH_EXCLUDED
    assert len(_hashed_values(_ROW)) == len(sealed)
    assert sealed | {"ids"} == ROW_FIELDS


def test_t16_a_declared_field_survives_the_round_trip() -> None:
    """Write it, read it, get it back — for every field, from the schema.

    ``from_row`` used to name its arguments by hand: a declared field the
    author forgot to thread was dropped on read, so the value on disk and the
    value every check saw were different values.
    """
    restored = ServedRun.from_row(_ROW.to_row())
    assert restored == _ROW
    assert set(_ROW.to_row()) == {f.name for f in dataclasses.fields(ServedRun)}


def test_t16_an_unknown_row_key_is_refused_not_ignored(tmp_path: Path) -> None:
    """Smuggling a tenth field past the schema must fail the read, not be dropped."""
    ws = _ws(tmp_path, enabled=True)
    _append(ws, ["D-1"])
    _tamper(ws, 0, "degraded", {"leg": "vector", "reason": "down"})
    verdict = verify_served_chain(ws)
    assert verdict.ok is False
    assert "schema" in verdict.reason


# ---------------------------------------------------------------------------
# The live surface — a ledger nothing writes to records nothing
# ---------------------------------------------------------------------------


def _live_workspace(tmp_path: Path, name: str, *, ledger: bool) -> str:
    ws = tmp_path / name
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "decisions" / "DECISIONS.md").write_text(
        "[D-20260829-001]\nStatement: the latency decision\nDate: 2026-08-29\nStatus: active\n\n---\n\n",
        encoding="utf-8",
    )
    (ws / "mind-mem.json").write_text(
        json.dumps(
            {
                "recall": {"vector_enabled": False, "provider": "local"},
                "cache": {"enabled": False},
                "served_ledger": {"enabled": ledger},
            }
        ),
        encoding="utf-8",
    )
    return str(ws)


def _mcp_recall(monkeypatch: pytest.MonkeyPatch, ws: str) -> Any:
    """The real MCP recall handler, pointed at *ws*."""
    import mind_mem.mcp.tools.recall as mcp_recall

    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)
    return mcp_recall


def test_the_mcp_recall_handler_writes_the_row_when_the_ledger_is_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Traced from a real entry point, not from the module's own API.

    A ledger with green unit tests and no production writer records nothing at
    all. What this pins is the wiring: ``_recall_impl`` is the choke point both
    ``recall`` and ``hybrid_search`` delegate to, and the row it writes must
    carry the SAME digests the run's own attestation published — otherwise the
    two records cannot be joined, which is the only reason the ledger exists.
    """
    ws = _live_workspace(tmp_path, "wired", ledger=True)
    mcp_recall = _mcp_recall(monkeypatch, ws)

    envelope = json.loads(mcp_recall._recall_impl("latency decision", limit=5, scoring_instant="2026-08-29"))
    attestation = envelope["attestation"]

    rows = read_served_runs(ws)
    assert len(rows) == 1, "the handler did not write a row"
    row = rows[0]
    assert row.served_digest == attestation["results_digest"]
    assert row.query_hash == attestation["query_hash"]
    assert row.pipeline_hash == attestation["config_hash"]
    assert row.index_anchor == attestation["index_anchor"]
    assert row.scoring_instant == attestation["scoring_instant"]
    assert row.ids == tuple(hit.get("_id", "") for hit in envelope["results"])
    assert verify_served_chain(ws).ok is True


def test_the_mcp_recall_handler_writes_nothing_when_the_ledger_is_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default OFF has to hold on the live path, not only in the module."""
    ws = _live_workspace(tmp_path, "quiet", ledger=False)
    mcp_recall = _mcp_recall(monkeypatch, ws)

    envelope = json.loads(mcp_recall._recall_impl("latency decision", limit=5, scoring_instant="2026-08-29"))
    assert envelope["attestation"]["results_digest"]
    assert read_served_runs(ws) == ()
    assert not Path(ledger_path(ws)).exists()


def test_two_identical_live_runs_share_a_run_id_and_differ_in_seq(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The join key on the live path: same answer, same id, two occurrences."""
    ws = _live_workspace(tmp_path, "repeat", ledger=True)
    mcp_recall = _mcp_recall(monkeypatch, ws)

    for instant in ("2026-08-24", "2026-08-29"):
        mcp_recall._recall_impl("latency decision", limit=5, scoring_instant=instant)

    rows = read_served_runs(ws)
    assert len(rows) == 2
    assert rows[0].run_id == rows[1].run_id
    assert rows[0].scoring_instant != rows[1].scoring_instant
    assert rows[0].seq != rows[1].seq
    assert verify_served_chain(ws).ok is True


# ---------------------------------------------------------------------------
# The ruling on `credited`, enforced structurally rather than by comment
# ---------------------------------------------------------------------------


def test_the_ledger_cannot_write_the_tier_ladder() -> None:
    """``credited`` may not promote a tier, and not by accident either.

    ``ROADMAP.md:94`` would have ``credited`` write
    ``block_tier_meta.confirmations`` and "buy trust tiers" — an unreviewed
    state transition driven by an agent-reported outcome, which is the
    do-not-build item wearing a smaller hat. The ruling routes it through a
    proposal, or through a ``plan_consolidation`` output that ``approve_apply``
    executes. A comment saying so is worth less than an import that cannot
    exist, so this asserts the second one.
    """
    import ast
    import pathlib

    import mind_mem

    source = (pathlib.Path(mind_mem.__file__).parent / "served_ledger.py").read_text(encoding="utf-8")
    imported = {node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom) and node.module} | {
        alias.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Import) for alias in node.names
    }
    assert "recall_digests" in imported, "guard is vacuous unless the walker sees the real imports"
    assert not {"memory_tiers", ".memory_tiers", "mind_mem.memory_tiers"} & imported

    # Prose may quote the ruling; CODE may not reach the ladder. Docstrings are
    # excluded deliberately — the module states the ruling in its own words, and
    # a check that could not tell an explanation from a call would force the
    # explanation out of the file.
    tree = ast.parse(source)
    docstrings = {
        ast.get_docstring(node, clean=False)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    code_tokens: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            code_tokens.add(node.id)
        elif isinstance(node, ast.Attribute):
            code_tokens.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value not in docstrings:
            code_tokens.add(node.value)
    assert "served_set_digest" in code_tokens, "guard is vacuous unless the walker sees real code"
    for forbidden in ("block_tier_meta", "block_tiers", "confirmations", "MemoryTier", "TierManager"):
        assert forbidden not in code_tokens, f"the ledger reaches the tier ladder ({forbidden})"


# ---------------------------------------------------------------------------
# Fail loud, not quiet — a ledger that silently skips rows records nothing
# ---------------------------------------------------------------------------


def test_a_row_whose_digest_disagrees_with_its_ids_is_refused(tmp_path: Path) -> None:
    """The cross-check is at the write, not only at verification.

    The digest is supplied rather than recomputed so the row commits to the
    value the run's attestation already published. That only holds if a
    disagreement is refused here — otherwise the ledger would happily store a
    row whose own verification it knows will fail.
    """
    ws = _ws(tmp_path, enabled=True)
    with pytest.raises(ValueError, match="does not match ids"):
        append_served_run(
            ws,
            query_hash=query_hash("q"),
            served_digest=served_set_digest(["D-1"]),
            ids=["D-1", "D-2"],
            pipeline_hash=PIPELINE,
            index_anchor=ANCHOR,
            scoring_instant="2026-08-29",
        )
    assert read_served_runs(ws) == ()


@pytest.mark.parametrize("field", ["query_hash", "pipeline_hash", "index_anchor"])
def test_a_short_digest_is_refused_because_run_id_has_no_separators(tmp_path: Path, field: str) -> None:
    """``run_id`` concatenates three digests with nothing between them.

    That is unambiguous only while each is fixed-width, so the width is a
    contract rather than an assumption — a 16-character "hash" would let two
    different runs collide, which is the one thing the id may not do.
    """
    ws = _ws(tmp_path, enabled=True)
    kwargs: dict[str, Any] = {
        "query_hash": query_hash("q"),
        "served_digest": served_set_digest(["D-1"]),
        "ids": ["D-1"],
        "pipeline_hash": PIPELINE,
        "index_anchor": ANCHOR,
        "scoring_instant": "2026-08-29",
    }
    kwargs[field] = "75545dbd7eea83d0"
    with pytest.raises(ValueError, match="64-character lowercase hex"):
        append_served_run(ws, **kwargs)
    assert read_served_runs(ws) == ()
