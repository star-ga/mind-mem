"""RA.1's open half — the RIGHT-HAND SIDE of the served/outcome join.

ROADMAP (RA.1 — served-set ledger): "**Still open:** `report_outcome(run_id=…)` — the
right-hand side of the join; `run_id` appears nowhere in `outcome_attribution.py`."

The served ledger records WHAT was served, stably keyed by a content-derived `run_id`.
Nothing recorded whether that answer HELPED, so the ledger could say "these ids were
served under this run" and no query could ever ask "did serving them work". This closes
that.

Three design decisions, each the alternative to something worse:

**A separate append-only sidecar, not a column on the calibration table.** RA.1's
structural rail is that nothing on the scoring path may import the ledger, in any
spelling. A join living in the calibration DB would put the ledger's key inside the
store the scoring path reads, and the rail would then be a convention instead of a
property. The sidecar is append-only and hash-chained like the ledger it joins.

**An unknown `run_id` is REFUSED, not recorded.** An orphan join row makes the join
silently drop — a caller believes the outcome was attributed and every later query
under-counts, with no signal anywhere. The failure has to land on the caller who can
still fix it.

**Absence stays legal and stays VISIBLE.** `run_id` is optional, because most reporters
never saw a recall run. But an unattributed outcome must not be countable as an
attributed one, so the result says which it is rather than leaving the caller to infer
it from a missing key.

No clock, no randomness: `seq` orders the file and the row hash is derived.
"""

from __future__ import annotations

import pytest

from mind_mem.served_outcome_join import (
    JoinRefused,
    append_outcome_join,
    join_rows,
    outcomes_for_run,
)

RUN_A = "a" * 64
RUN_B = "b" * 64


def _ws(tmp_path, known=(RUN_A,)):
    """A workspace whose ledger is stubbed to know exactly `known`."""
    import mind_mem.served_outcome_join as soj

    soj._known_run_ids.cache_clear()
    return str(tmp_path), known


def test_a_join_row_is_appended_for_a_known_run(tmp_path, monkeypatch):
    import mind_mem.served_outcome_join as soj

    monkeypatch.setattr(soj, "_known_run_ids", lambda ws: frozenset({RUN_A}))
    row = append_outcome_join(str(tmp_path), run_id=RUN_A, outcome_id="OUT-1")
    assert row is not None
    assert row.run_id == RUN_A and row.outcome_id == "OUT-1"
    assert row.seq == 1


def test_an_unknown_run_id_is_REFUSED_not_recorded(tmp_path, monkeypatch):
    """THE LOAD-BEARING REFUSAL. An orphan row makes the join silently drop, so a
    caller believes the outcome was attributed while every later query under-counts."""
    import mind_mem.served_outcome_join as soj

    monkeypatch.setattr(soj, "_known_run_ids", lambda ws: frozenset({RUN_A}))
    with pytest.raises(JoinRefused, match="not in the served ledger"):
        append_outcome_join(str(tmp_path), run_id=RUN_B, outcome_id="OUT-1")
    assert join_rows(str(tmp_path)) == (), "a refused join still wrote a row"


def test_the_refusal_names_the_run_so_it_can_be_chased(tmp_path, monkeypatch):
    import mind_mem.served_outcome_join as soj

    monkeypatch.setattr(soj, "_known_run_ids", lambda ws: frozenset({RUN_A}))
    with pytest.raises(JoinRefused) as exc:
        append_outcome_join(str(tmp_path), run_id=RUN_B, outcome_id="OUT-1")
    assert RUN_B[:12] in str(exc.value)


def test_rows_are_append_only_and_hash_chained(tmp_path, monkeypatch):
    """Each row commits to the previous one, so deleting or editing a middle row is
    detectable — the same property the served ledger has, for the same reason: the
    join is evidence about what a run achieved."""
    import mind_mem.served_outcome_join as soj

    monkeypatch.setattr(soj, "_known_run_ids", lambda ws: frozenset({RUN_A}))
    first = append_outcome_join(str(tmp_path), run_id=RUN_A, outcome_id="OUT-1")
    second = append_outcome_join(str(tmp_path), run_id=RUN_A, outcome_id="OUT-2")
    assert second.seq == 2
    assert second.prev_row_hash == soj.row_hash(first)
    assert first.prev_row_hash != second.prev_row_hash


def test_the_same_pair_twice_is_idempotent_not_a_second_row(tmp_path, monkeypatch):
    """One outcome for one run is one fact. A second row would double-count it in
    every derived view, and a view that double-counts is worse than no view."""
    import mind_mem.served_outcome_join as soj

    monkeypatch.setattr(soj, "_known_run_ids", lambda ws: frozenset({RUN_A}))
    append_outcome_join(str(tmp_path), run_id=RUN_A, outcome_id="OUT-1")
    again = append_outcome_join(str(tmp_path), run_id=RUN_A, outcome_id="OUT-1")
    assert again is None, "the duplicate wrote a second row"
    assert len(join_rows(str(tmp_path))) == 1


def test_outcomes_for_run_returns_only_that_runs_outcomes(tmp_path, monkeypatch):
    import mind_mem.served_outcome_join as soj

    monkeypatch.setattr(soj, "_known_run_ids", lambda ws: frozenset({RUN_A, RUN_B}))
    append_outcome_join(str(tmp_path), run_id=RUN_A, outcome_id="OUT-1")
    append_outcome_join(str(tmp_path), run_id=RUN_B, outcome_id="OUT-2")
    append_outcome_join(str(tmp_path), run_id=RUN_A, outcome_id="OUT-3")
    assert outcomes_for_run(str(tmp_path), RUN_A) == ("OUT-1", "OUT-3")
    assert outcomes_for_run(str(tmp_path), RUN_B) == ("OUT-2",)


def test_an_unknown_run_has_no_outcomes_and_does_not_raise(tmp_path):
    """A read must not refuse: asking about a run with no outcomes is a normal
    question with the answer "none"."""
    assert outcomes_for_run(str(tmp_path), RUN_B) == ()


def test_a_missing_sidecar_reads_as_empty_not_as_an_error(tmp_path):
    assert join_rows(str(tmp_path)) == ()


def test_the_module_reads_no_clock_and_no_randomness():
    """`seq` orders the file and the row hash is derived. A clock here would make two
    identical joins produce different rows, and the chain unverifiable on replay."""
    import ast
    import inspect

    import mind_mem.served_outcome_join as soj

    tree = ast.parse(inspect.getsource(soj))
    banned = {"time", "random", "uuid", "datetime"}
    found = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not (found & banned), sorted(found & banned)


# ---------------------------------------------------------------------------
# Wiring: report_outcome is the RA.1 ask. A join module nothing calls is the
# same gap with more code.
# ---------------------------------------------------------------------------


def test_report_outcome_accepts_run_id_and_records_the_join(tmp_path, monkeypatch):
    """RA.1's literal ask: `report_outcome(run_id=…)`."""
    import mind_mem.served_outcome_join as soj
    from mind_mem.outcome_attribution import report_outcome

    ws = str(tmp_path)
    monkeypatch.setattr(soj, "_known_run_ids", lambda w: frozenset({RUN_A}))
    got = report_outcome(ws, ["DEC-1"], "success", run_id=RUN_A)

    assert got["run_attributed"] is True, got
    assert got["outcome_id"] in outcomes_for_run(ws, RUN_A)


def test_an_outcome_with_no_run_id_is_reported_as_NOT_attributed(tmp_path):
    """POSITIVE CONTROL, and the honest half. Most reporters never saw a recall run, so
    this must stay legal — but it must SAY it is unattributed rather than leave the
    caller to infer it from a missing key, or an unattributed outcome gets counted as
    an attributed one."""
    from mind_mem.outcome_attribution import report_outcome

    got = report_outcome(str(tmp_path), ["DEC-1"], "success")
    assert got["run_attributed"] is False, got
    assert join_rows(str(tmp_path)) == ()


def test_report_outcome_RAISES_on_a_run_id_the_ledger_does_not_know(tmp_path, monkeypatch):
    """The refusal must reach the CALLER through report_outcome, not be swallowed into
    a false `run_attributed: false`. Those two look identical to the caller and only
    one of them is fixable by them."""
    import mind_mem.served_outcome_join as soj
    from mind_mem.outcome_attribution import report_outcome

    monkeypatch.setattr(soj, "_known_run_ids", lambda w: frozenset({RUN_A}))
    with pytest.raises(JoinRefused):
        report_outcome(str(tmp_path), ["DEC-1"], "success", run_id=RUN_B)


def test_the_outcome_is_still_recorded_when_the_join_is_refused(tmp_path, monkeypatch):
    """The join runs AFTER the outcome is durably recorded. A refused join must not
    lose the outcome itself — the report was valid, only its attribution was not."""
    import mind_mem.served_outcome_join as soj
    from mind_mem.outcome_attribution import load_outcome_signals, report_outcome

    monkeypatch.setattr(soj, "_known_run_ids", lambda w: frozenset({RUN_A}))
    with pytest.raises(JoinRefused):
        report_outcome(str(tmp_path), ["DEC-1"], "success", run_id=RUN_B)
    signals = load_outcome_signals(str(tmp_path), ["DEC-1"])
    assert signals, "the outcome was lost along with its refused join"


def test_the_scoring_path_does_not_import_the_join(tmp_path):
    """RA.1's structural rail, extended to this module. The join carries the ledger's
    key, so if the scoring path could import it the rail would be a convention again.
    Walked over the AST, not grepped for a substring, so prose mentioning the module
    name cannot satisfy it."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "src" / "mind_mem"
    scoring = ["_recall_core.py", "hybrid_recall.py", "recall_vector.py", "recall.py"]
    for name in scoring:
        path = root / name
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mod = ""
            if isinstance(node, ast.Import):
                mod = node.names[0].name
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
            assert "served_outcome_join" not in mod, (
                f"{name} imports the served/outcome join; the scoring path must not "
                f"reach the ledger's key in any spelling"
            )
