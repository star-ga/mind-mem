# Copyright 2026 STARGA, Inc.
"""RA.5 — the dashboard: one lifecycle ladder, rendered, and the verdicts behind it.

RA.0 collapsed three block-lifecycle tier ladders into one so this could
exist: *"a dashboard cannot render one axis while three exist"*
(``tests/test_tier_axis_collapse.py``). :class:`~mind_mem.memory_tiers.MemoryTier`
is the survivor and the only axis rendered here.

Four panels, and each one is a claim the other three cannot make:

**Lifecycle tiers** — how many blocks sit on each rung of the one ladder,
crossed with the retention class RA.4 computes, so the two axes are visible
together. They answer different questions and are routinely confused: a tier
says *how settled* a block is, a retention class says *how much scrutiny its
death needs*. A ``WORKING`` block can be ``PROTECTED`` (a release decision
written this morning) and a ``VERIFIED`` one can be ``GOVERNED``. Rendering
them on one grid is the point of the panel.

**Served-set ledger** — RA.1's chain, and its verdict. Until this module,
:func:`~mind_mem.served_ledger.verify_served_chain` was reachable from the
test suite and a documentation paragraph and from nothing an operator could
run: a tamper-evidence check nobody can invoke is a claim, not a control.
This panel calls it, and ``mm dashboard`` exits non-zero when it fails.

**Precision, waste and serve counts** — RA.2's derived views, reused whole
from :mod:`mind_mem.accountability_views` rather than recomputed here. One
report, one set of numbers; a dashboard that re-derived them would be a second
place for them to disagree.

**Replay** — a pointer, not a number: :mod:`mind_mem.replay_check` answers it
per attestation, because "did THIS run serve what it said" is a question about
one envelope and not about a workspace.

FOUR REFUSALS, inherited deliberately from :mod:`mind_mem.accountability_views`
because a dashboard is where they are easiest to lose:

* **It stores nothing and creates nothing.** Every handle comes from that
  module's ``mode=ro`` opener — reused rather than re-spelled, so there is one
  read-only rail and not two. A dashboard over a workspace with no tier store
  leaves it with no tier store.
* **It never names withheld content.** The corpus is read through
  :func:`~mind_mem.admissibility.admit_corpus`, the shared gate, and the tier
  panel publishes **counts only** — no id list anywhere. A tier row naming a
  block the gate now withholds is reported as a number under its own key.
* **An absent source is unavailable with a reason, never a zero.** No tier
  store means ``available=False``; it does not mean "zero blocks are VERIFIED".
* **Nothing here reaches a ranking.** This module imports the ledger, which
  the scoring path may not; the direction is one-way and structurally enforced
  (``tests/test_recall_attestation_v2.py`` T12).

NO CLOCK. Every input is a stored field or a pure function of one, so two
hosts rendering the same workspace print the same dashboard. ``updated_at``
sits in the tier store and is deliberately not read: the moment a panel says
"promoted in the last 7 days" it is a different report on every host.

    python -m mind_mem.accountability_dashboard --workspace .   # or: mm dashboard
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

# The read-only rail is imported, never re-spelled: one opener that refuses to
# create a database, used by every accountability surface. A second copy is a
# second thing to get wrong, and this module is the one most likely to be
# extended by someone in a hurry.
from .accountability_views import _read_only_connect, _rows, accountability_report
from .admissibility import admit_corpus
from .memory_tiers import MemoryTier
from .observability import get_logger
from .retention_class import RETENTION_CLASSES, retention_class
from .served_ledger import ledger_enabled, ledger_path, read_served_runs, verify_served_chain

_log = get_logger("accountability_dashboard")

#: Where ``compaction.run_promotion_cycle`` keeps the one ladder's store.
#: Spelled here because that path is built inline at its writer and is not a
#: constant anyone can import; ``tests/test_accountability_dashboard.py`` pins
#: the two spellings together, so moving the store fails the build rather than
#: silently emptying this panel.
TIER_STORE_RELPATH: Final = os.path.join("intelligence", "tiers.db")

#: Report tag. Additive-only, like ``MM_ACCOUNTABILITY_v1``: a panel may gain a
#: key and the dashboard may gain a panel, but a key that exists keeps its
#: meaning. Nothing hashes this, so there is no layout to forge.
DASHBOARD_TAG: Final = "MM_DASHBOARD_v1"

#: Width of the label column in the rendered text, so the panels line up.
_LABEL_WIDTH: Final = 22

__all__ = [
    "DASHBOARD_TAG",
    "TIER_NAMES",
    "TIER_STORE_RELPATH",
    "LedgerPanel",
    "TierCensus",
    "dashboard",
    "ledger_panel",
    "main",
    "render",
    "tier_census",
]


# ---------------------------------------------------------------------------
# Panel 1 — the one lifecycle ladder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierCensus:
    """Counts over the one lifecycle ladder. Derived, never stored.

    ``untracked_admitted`` is its own number and not folded into ``WORKING``.
    :meth:`~mind_mem.memory_tiers.TierManager.get_tier` answers ``WORKING`` for
    a block with no row, which is the right default for a *lookup* and the
    wrong one for a *census*: "we have never assigned this block a tier" and
    "we assigned it the first tier" are different facts, and a dashboard that
    merged them would report a promotion cycle had run when it had not.

    ``tracked_not_admitted`` counts tier rows naming a block the admission gate
    does not currently admit — a block quarantined after it was tiered, or one
    removed from the corpus. It is a count and never a list: publishing those
    ids would make this panel a read surface around the gate.
    """

    available: bool
    reason: str
    store: str
    tracked: int
    by_tier: Mapping[str, int]
    unknown_tier_values: int
    demotion_reasons: Mapping[str, int]
    corpus_admitted: int
    tracked_admitted: int
    untracked_admitted: int
    tracked_not_admitted: int
    by_tier_retention: Mapping[str, Mapping[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "store": self.store,
            "tracked": self.tracked,
            "by_tier": dict(self.by_tier),
            "unknown_tier_values": self.unknown_tier_values,
            "demotion_reasons": dict(self.demotion_reasons),
            "corpus_admitted": self.corpus_admitted,
            "tracked_admitted": self.tracked_admitted,
            "untracked_admitted": self.untracked_admitted,
            "tracked_not_admitted": self.tracked_not_admitted,
            "by_tier_retention": {tier: dict(row) for tier, row in self.by_tier_retention.items()},
        }


#: Tier names in ladder order — WORKING first, VERIFIED last. Derived from the
#: enum rather than listed, so a fifth rung is rendered the day it is added.
TIER_NAMES: Final[tuple[str, ...]] = tuple(tier.name for tier in sorted(MemoryTier, key=lambda t: t.value))


def _admitted_blocks(workspace: str) -> list[dict]:
    """The admitted corpus, through the shared gate and never around it."""
    from .storage import get_block_store

    try:
        blocks = get_block_store(workspace).get_all()
    except Exception as exc:  # pragma: no cover — a missing corpus is an empty one
        _log.debug("dashboard_corpus_read_failed", error=str(exc))
        blocks = []
    return admit_corpus(blocks)


def _tier_name(value: Any) -> str:
    """The ladder name for a stored tier value, or ``""`` when it is not one."""
    try:
        return MemoryTier(int(value)).name
    except (TypeError, ValueError):
        return ""


def _unavailable_census(store: str, reason: str) -> TierCensus:
    """No tier store: unavailable with a reason, never a corpus of zeroes."""
    return TierCensus(
        available=False,
        reason=reason,
        store=store,
        tracked=0,
        by_tier={},
        unknown_tier_values=0,
        demotion_reasons={},
        corpus_admitted=0,
        tracked_admitted=0,
        untracked_admitted=0,
        tracked_not_admitted=0,
        by_tier_retention={},
    )


def tier_census(workspace: str) -> TierCensus:
    """The one lifecycle ladder, counted and crossed with retention class.

    Read-only: opens the tier store ``mode=ro`` and returns
    ``available=False`` when there is none, rather than creating one. Reads no
    clock, so the census is reproducible on any host on any day.

    Args:
        workspace: Workspace root.

    Returns:
        A :class:`TierCensus`. Counts only — this panel publishes no block ids.
    """
    store = os.path.join(workspace, TIER_STORE_RELPATH)
    conn = _read_only_connect(store)
    if conn is None:
        return _unavailable_census(
            store,
            "no tier store: a promotion cycle has never run in this workspace "
            "(`mm compact` runs one). Not a claim that no block is tiered.",
        )
    try:
        table = _rows(conn, "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'block_tiers'")
        rows = _rows(conn, "SELECT id, tier, demotion_reason FROM block_tiers")
    finally:
        conn.close()
    if not table:
        # Distinguished from "empty" deliberately. ``_rows`` answers ``[]`` for a
        # missing table as well as an empty one, and reporting both as "nothing
        # has been registered" would state a fact about the ladder that was
        # really a fact about the schema.
        return _unavailable_census(
            store,
            "the tier store exists but has no block_tiers table: it was created by something other "
            "than the ladder, or its schema has not been initialised",
        )
    if not rows:
        return _unavailable_census(
            store,
            "the tier store exists but holds no tier rows: nothing has been registered on the ladder yet",
        )

    by_tier = dict.fromkeys(TIER_NAMES, 0)
    demotions: dict[str, int] = {}
    tier_of: dict[str, str] = {}
    unknown = 0
    for row in rows:
        name = _tier_name(row["tier"])
        if not name:
            unknown += 1
            continue
        block_id = str(row["id"] or "")
        if not block_id:
            unknown += 1
            continue
        by_tier[name] += 1
        tier_of[block_id] = name
        reason = str(row["demotion_reason"] or "")
        if reason:
            demotions[reason] = demotions.get(reason, 0) + 1

    admitted = _admitted_blocks(workspace)
    grid: dict[str, dict[str, int]] = {name: dict.fromkeys(RETENTION_CLASSES, 0) for name in TIER_NAMES}
    tracked_admitted = 0
    for block in admitted:
        name = tier_of.get(str(block.get("_id", "")), "")
        if not name:
            continue
        tracked_admitted += 1
        grid[name][retention_class(block)] += 1

    return TierCensus(
        available=True,
        reason="",
        store=store,
        tracked=len(tier_of),
        by_tier=by_tier,
        unknown_tier_values=unknown,
        demotion_reasons=dict(sorted(demotions.items())),
        corpus_admitted=len(admitted),
        tracked_admitted=tracked_admitted,
        untracked_admitted=len(admitted) - tracked_admitted,
        tracked_not_admitted=len(tier_of) - tracked_admitted,
        by_tier_retention=grid,
    )


# ---------------------------------------------------------------------------
# Panel 2 — the ledger's own verdict, finally reachable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerPanel:
    """RA.1's chain verdict, plus the two facts needed to read it.

    ``enabled`` and ``present`` are reported separately because they answer
    different questions and a dashboard that showed only one would mislead:
    a workspace can hold a sealed ledger from before the flag was switched
    off (rows, not enabled), and an enabled workspace that has served nothing
    yet has no file at all.
    """

    enabled: bool
    present: bool
    rows: int
    ok: bool
    bad_seq: int | None
    reason: str
    head: str
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "present": self.present,
            "rows": self.rows,
            "ok": self.ok,
            "bad_seq": self.bad_seq,
            "reason": self.reason,
            "head": self.head,
            "path": self.path,
        }


def ledger_panel(workspace: str) -> LedgerPanel:
    """Run the served-set ledger's chain check and report its verdict.

    This is the wiring, and the wiring is the point:
    :func:`~mind_mem.served_ledger.verify_served_chain` has always existed and
    has never been callable from an operator surface. ``ok`` is ``True`` for a
    workspace that has no ledger — there is nothing to have tampered with —
    and the ``reason`` says so rather than leaving a bare green.

    Args:
        workspace: Workspace root.

    Returns:
        A :class:`LedgerPanel`. Read-only; creates nothing.
    """
    path = ledger_path(workspace)
    present = os.path.isfile(path)
    verdict = verify_served_chain(workspace)
    try:
        rows = len(read_served_runs(workspace))
    except (ValueError, json.JSONDecodeError):
        # An unreadable row is already the chain verdict's finding; the count
        # is simply unknown, and reporting 0 would read as "empty and fine".
        rows = -1
    reason = verdict.reason
    if verdict.ok and not present:
        # Default ON since 5.0.2 (``served_ledger.ledger_enabled`` is the
        # authority): an absent file no longer means the feature is off, so
        # the old wording printed a stale default into an operator-facing
        # verdict. Same three-way silence as ``replay_check._silence_reason``.
        reason = (
            "no ledger in this workspace: the served-set ledger records by default since 5.0.2, "
            "so an absent file means nothing has been served here yet, or the workspace opted out "
            'with {"served_ledger": {"enabled": false}}, or there is no readable mind-mem.json to '
            "serve from. `enabled` on this panel says which. Nothing to verify either way"
        )
    return LedgerPanel(
        enabled=ledger_enabled(workspace),
        present=present,
        rows=rows,
        ok=verdict.ok,
        bad_seq=verdict.bad_seq,
        reason=reason,
        head=verdict.head,
        path=path,
    )


# ---------------------------------------------------------------------------
# The dashboard + its renderer
# ---------------------------------------------------------------------------


def dashboard(workspace: str) -> dict[str, Any]:
    """Every panel over *workspace*, recomputed. Writes nothing, ever.

    ``accountability`` is :func:`~mind_mem.accountability_views.accountability_report`
    verbatim — reused, not re-derived, so the dashboard and ``mm accountability``
    can never print two different precisions for one workspace.
    """
    return {
        "schema": DASHBOARD_TAG,
        "workspace": os.path.abspath(workspace),
        "tiers": tier_census(workspace).to_dict(),
        "ledger": ledger_panel(workspace).to_dict(),
        "accountability": accountability_report(workspace),
        "replay": {
            "how": "mind_mem.replay_check.replay_check(workspace, envelope['attestation'])",
            "why_not_a_number": (
                "replay is a question about one recall envelope, not about a workspace: the "
                "attestation lives on the response and is never persisted"
            ),
        },
    }


def _line(label: str, value: Any) -> str:
    return f"  {label:<{_LABEL_WIDTH}} {value}"


def _req(panel: Mapping[str, Any], key: str, *, where: str) -> Any:
    """Read *key* off *panel*, or fail loudly.

    ``panel.get(key, 0)`` is how a dashboard prints a number nobody measured:
    a view renames a field, the renderer keeps asking for the old name, and the
    panel silently reports zero forever. That is not hypothetical — the first
    cut of this renderer asked ``precision_by_intent`` for ``served`` and
    ``credited``, which that view has never published, and printed ``0`` for
    both while dropping the per-intent rows entirely. Reading a panel strictly
    turns that class of drift into a failure in the commit that causes it.
    """
    if key not in panel:
        raise KeyError(f"{where} panel has no {key!r}: the dashboard and the view it renders have drifted apart")
    return panel[key]


def _tier_rows(census: Mapping[str, Any]) -> list[str]:
    """The tier x retention grid, in ladder order then class order."""
    grid = _req(census, "by_tier_retention", where="tiers")
    by_tier = _req(census, "by_tier", where="tiers")
    header = f"  {'tier':<{_LABEL_WIDTH}} {'total':>7}" + "".join(f" {name:>10}" for name in RETENTION_CLASSES)
    out = [header]
    for name in TIER_NAMES:
        cells = grid.get(name) or {}
        out.append(
            f"  {name:<{_LABEL_WIDTH}} {by_tier.get(name, 0):>7}" + "".join(f" {cells.get(klass, 0):>10}" for klass in RETENTION_CLASSES)
        )
    return out


def _tier_panel(tiers: Mapping[str, Any]) -> list[str]:
    """Panel 1 — the one lifecycle ladder, crossed with retention class."""
    out = ["LIFECYCLE TIERS  (memory_tiers.MemoryTier — the one ladder)"]
    if not _req(tiers, "available", where="tiers"):
        return out + [_line("unavailable:", _req(tiers, "reason", where="tiers"))]
    out += _tier_rows(tiers)
    out.append(_line("tracked blocks", _req(tiers, "tracked", where="tiers")))
    out.append(_line("admitted corpus", _req(tiers, "corpus_admitted", where="tiers")))
    out.append(_line("untracked admitted", _req(tiers, "untracked_admitted", where="tiers")))
    out.append(_line("tiered, not admitted", _req(tiers, "tracked_not_admitted", where="tiers")))
    if _req(tiers, "unknown_tier_values", where="tiers"):
        out.append(_line("unknown tier values", tiers["unknown_tier_values"]))
    for reason, count in sorted(_req(tiers, "demotion_reasons", where="tiers").items()):
        out.append(_line(f"demoted: {reason}", count))
    return out


def _ledger_lines(ledger: Mapping[str, Any]) -> list[str]:
    """Panel 2 — RA.1's chain verdict, the one thing on this page that can fail."""
    out = [
        "SERVED-SET LEDGER  (RA.1 — append-only, chain-verified)",
        _line("chain", "OK" if _req(ledger, "ok", where="ledger") else "FAILED"),
        _line("enabled", _req(ledger, "enabled", where="ledger")),
        _line("rows", _req(ledger, "rows", where="ledger")),
    ]
    if _req(ledger, "reason", where="ledger"):
        out.append(_line("note", ledger["reason"]))
    if _req(ledger, "bad_seq", where="ledger") is not None:
        out.append(_line("first bad row", ledger["bad_seq"]))
    return out


def _block_precision_lines(view: Mapping[str, Any]) -> list[str]:
    """Panel 3a — block-level precision: credited *anywhere*, per intent.

    Reported apart from the run-level join below because they are different
    measurements over different denominators, and a dashboard that printed one
    number called "precision" would be printing whichever it happened to reach.
    """
    out = ["PRECISION — block level  (of the blocks served under an intent, how many are credited anywhere)"]
    if not _req(view, "available", where="precision_by_intent"):
        return out + [_line("unavailable:", _req(view, "reason", where="precision_by_intent"))]
    out.append(_line("serve observations", _req(view, "observations", where="precision_by_intent")))
    out.append(_line("credit rows", _req(view, "credit_rows", where="precision_by_intent")))
    out.append(
        _line(
            "credits, unserved block",
            _req(view, "credit_rows_on_unserved_blocks", where="precision_by_intent"),
        )
    )
    for row in _req(view, "rows", where="precision_by_intent"):
        served = _req(row, "served_blocks", where="precision_by_intent.rows")
        credited = _req(row, "credited_blocks", where="precision_by_intent.rows")
        precision = _req(row, "precision", where="precision_by_intent.rows")
        out.append(_line(f"  {_req(row, 'intent', where='precision_by_intent.rows')}", f"{credited}/{served} = {precision}"))
    out.append(_line("window", _req(view, "window", where="precision_by_intent")))
    return out


def _run_precision_lines(view: Mapping[str, Any]) -> list[str]:
    """Panel 3b — the run-level join: these runs served these blocks."""
    out = ["PRECISION — run level  (joined on served_ledger.run_id)"]
    if not _req(view, "available", where="run_precision"):
        return out + [_line("unavailable:", _req(view, "reason", where="run_precision"))]
    out.append(_line("runs", _req(view, "runs", where="run_precision")))
    out.append(_line("joined runs", _req(view, "joined_runs", where="run_precision")))
    out.append(_line("served", _req(view, "served", where="run_precision")))
    out.append(_line("credited", _req(view, "credited", where="run_precision")))
    out.append(_line("precision", _req(view, "precision", where="run_precision")))
    out.append(
        _line(
            "credits, unjoinable",
            _req(view, "credit_rows_with_unjoinable_query_id", where="run_precision"),
        )
    )
    for row in _req(view, "by_intent", where="run_precision"):
        served = _req(row, "served", where="run_precision.by_intent")
        credited = _req(row, "credited", where="run_precision.by_intent")
        precision = _req(row, "precision", where="run_precision.by_intent")
        out.append(_line(f"  {_req(row, 'intent', where='run_precision.by_intent')}", f"{credited}/{served} = {precision}"))
    return out


def _waste_lines(view: Mapping[str, Any]) -> list[str]:
    """Panel 4 — unserved content, with PROTECTED blocks named as not-waste."""
    out = [
        "WASTE  (admitted blocks with no serve evidence — a question, not a verdict)",
        _line("admitted", _req(view, "corpus_admitted", where="waste")),
        _line("withheld", _req(view, "corpus_withheld", where="waste")),
        _line("served at least once", _req(view, "served_at_least_once", where="waste")),
        _line("unserved", _req(view, "unserved", where="waste")),
    ]
    by_class = _req(view, "unserved_by_retention_class", where="waste")
    for klass in RETENTION_CLASSES:
        out.append(_line(f"  unserved {klass}", by_class.get(klass, 0)))
    out.append(_line("unserved ratio", _req(view, "unserved_ratio", where="waste")))
    out.append(_line("window", _req(view, "window", where="waste")))
    return out


def _serve_count_lines(view: Mapping[str, Any]) -> list[str]:
    """Panel 5 — how often, split into what survives the prune and what does not."""
    out = ["SERVE COUNTS  (durable = ledger, windowed = 30-day log)"]
    if not _req(view, "available", where="serve_counts"):
        return out + [_line("unavailable:", _req(view, "reason", where="serve_counts"))]
    out.append(_line("blocks seen", _req(view, "blocks", where="serve_counts")))
    out.append(_line("blocks, durable", _req(view, "durable_blocks", where="serve_counts")))
    out.append(_line("durable serves", _req(view, "durable_serves", where="serve_counts")))
    out.append(_line("windowed serves", _req(view, "windowed_serves", where="serve_counts")))
    return out


def render(report: Mapping[str, Any]) -> str:
    """Render a :func:`dashboard` payload as deterministic fixed-width text.

    Pure: takes the report it is handed and reads nothing else — no clock, no
    filesystem, no config — so the same payload renders identically anywhere,
    and a caller can render a dashboard captured on another host.

    Strict: every value comes through :func:`_req`, so a panel that no longer
    publishes a field this renderer names raises instead of printing a zero.

    Raises:
        KeyError: the report and the views it renders have drifted apart.
    """
    acct = _req(report, "accountability", where="report")
    sections = [
        [f"mind-mem dashboard  ({_req(report, 'schema', where='report')})", f"workspace: {_req(report, 'workspace', where='report')}"],
        _tier_panel(_req(report, "tiers", where="report")),
        _ledger_lines(_req(report, "ledger", where="report")),
        _block_precision_lines(_req(acct, "precision_by_intent", where="accountability")),
        _run_precision_lines(_req(acct, "run_precision", where="accountability")),
        _waste_lines(_req(acct, "waste", where="accountability")),
        _serve_count_lines(_req(acct, "serve_counts", where="accountability")),
        ["REPLAY", _line("per-envelope", _req(_req(report, "replay", where="report"), "how", where="replay"))],
    ]
    return "\n\n".join("\n".join(block) for block in sections) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m mind_mem.accountability_dashboard --workspace <path>``.

    Exit status carries the one verdict on the page: ``1`` when the served-set
    ledger fails its chain check, ``0`` otherwise. A dashboard that always
    exits 0 cannot be a cron job, and a tamper-evidence check nobody can fail
    on is decoration.
    """
    parser = argparse.ArgumentParser(
        prog="python -m mind_mem.accountability_dashboard",
        description="RA.5 — the lifecycle-tier dashboard. Reads only; stores nothing.",
    )
    parser.add_argument("--workspace", default=".", help="Workspace root (default: the current directory).")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON instead of text.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent (0 for one line).")
    args = parser.parse_args(argv)
    report = dashboard(args.workspace)
    if args.json:
        print(json.dumps(report, indent=args.indent or None, sort_keys=True))
    else:
        print(render(report))
    return 0 if (report.get("ledger") or {}).get("ok") else 1


if __name__ == "__main__":  # pragma: no cover — exercised as a subprocess in tests
    sys.exit(main())
