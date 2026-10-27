# Copyright 2026 STARGA, Inc.
"""Every read leg that can hand block content to a caller asks the shared filter.

Wiring-discipline rule 1: *any leg that reads blocks calls the admission
filter (``admit_corpus``) — do not hand-roll a status check.* The legs
covered here each decided servability with a private status whitelist, or
with none at all:

* guardrails / task frames / dead ends — a per-module ``_LIVE_STATUSES``
  set was the only gate, so a release decision or a revoked-credential
  ruling never reached them;
* the dream cycle's Markdown passes — raw daily-log text, including the text
  of a quarantined block inside it, was mined and surfaced verbatim;
* ``project_profile``'s unbound index path and ``build_core``'s leaf set —
  status-less SQL over the recall index.

Each negative assertion is paired with a positive control (rule 6), and the
three lifecycle legs are proven wired by mutating the shared filter and
watching the loader follow it (a gate that cannot fail is not a gate).
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pytest

import mind_mem.admissibility as admissibility

CANARY = "zebracanaryquarantine"


def _write(root, rel: str, body: str) -> str:
    path = os.path.join(str(root), rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


def _withhold(target_id: str):
    """A mutated shared filter that withholds exactly *target_id*."""
    real = admissibility.admit_corpus

    def fake(blocks, **kwargs):
        rows = real(blocks, **kwargs)
        return [row for row in rows if row.get("_id") != target_id]

    return fake


# ---------------------------------------------------------------------------
# Lifecycle legs: guardrails, task frames, dead ends
# ---------------------------------------------------------------------------

GUARDRAILS = """[GR-20261008-001]
Type: Guardrail
Statement: Never run `git reset --hard` without checking `git status` first.
Severity: critical
TriggerTools: Bash
TriggerCommands: git reset --hard
Status: active

[GR-20261008-002]
Type: Guardrail
Statement: Never force-push to main.
Severity: high
TriggerTools: Bash
TriggerCommands: git push --force
Status: active
"""

FRAMES = """[TF-20261008-001]
Goal: Ship the admission routing fix
Status: active

[TF-20261008-002]
Goal: Second frame
Status: active
"""

DEAD_ENDS = """[DE-20261008-001]
Approach: Rebuild the index from scratch every query
WhyFailed: Linear in corpus bytes on every recall
Outcome: refuted
TriggerCommands: reindex --full
Status: active

[DE-20261008-002]
Approach: Cache keyed on mtime alone
WhyFailed: Same-tick edits are invisible
Outcome: refuted
TriggerCommands: cache --mtime
Status: active
"""


def _guardrail_ids(ws: str) -> set[str]:
    from mind_mem.guardrails import load_guardrails

    return {g.block_id for g in load_guardrails(ws)}


def _frame_ids(ws: str) -> set[str]:
    from mind_mem.task_frames import load_task_frames

    return {f.block_id for f in load_task_frames(ws)}


def _dead_end_ids(ws: str) -> set[str]:
    from mind_mem.dead_ends import load_dead_ends

    return {d.block_id for d in load_dead_ends(ws)}


LIFECYCLE_LEGS = [
    pytest.param("guardrails/GUARDRAILS.md", GUARDRAILS, "GR-20261008-002", "GR-20261008-001", _guardrail_ids, id="guardrails"),
    pytest.param("frames/FRAMES.md", FRAMES, "TF-20261008-002", "TF-20261008-001", _frame_ids, id="task_frames"),
    pytest.param("frames/DEAD-ENDS.md", DEAD_ENDS, "DE-20261008-002", "DE-20261008-001", _dead_end_ids, id="dead_ends"),
]


@pytest.mark.parametrize(("rel", "body", "target", "control", "load"), LIFECYCLE_LEGS)
def test_lifecycle_leg_follows_the_shared_filter(tmp_path, monkeypatch, rel, body, target, control, load) -> None:
    """Mutate ``admit_corpus`` to withhold one block: the loader must follow it."""
    ws = str(tmp_path)
    _write(ws, rel, body)

    # Positive control: with the real filter both blocks are live.
    assert {target, control} <= load(ws)

    monkeypatch.setattr(admissibility, "admit_corpus", _withhold(target))
    loaded = load(ws)
    assert target not in loaded, f"{rel}: loader ignored the shared admission filter"
    assert control in loaded, "the mutation must withhold only the target"


@pytest.mark.parametrize(("rel", "body", "target", "control", "load"), LIFECYCLE_LEGS)
def test_lifecycle_leg_withholds_unadmitted_status(tmp_path, rel, body, target, control, load) -> None:
    """A status that never passed the gate is withheld; its sibling is served."""
    ws = str(tmp_path)
    marker = f"[{target}]"
    head, tail = body.split(marker)
    _write(ws, rel, head + marker + tail.replace("Status: active", "Status: quarantined", 1))

    loaded = load(ws)
    assert target not in loaded
    assert control in loaded


@pytest.mark.parametrize(
    "module_name",
    ["mind_mem.guardrails", "mind_mem.task_frames", "mind_mem.dead_ends"],
)
def test_lifecycle_refinement_never_names_an_unrecognised_status(module_name) -> None:
    """What remains per-module is a lifecycle refinement, inside admission.

    A status in the module's live set that admission does not recognise
    would be a second, private admission rule — exactly what this file
    exists to remove.
    """
    import importlib

    module = importlib.import_module(module_name)
    live = module._LIVE_STATUSES
    assert live, "positive control: the set is non-empty"
    for status in live:
        assert admissibility.is_admissible_status(status), (module_name, status)


# ---------------------------------------------------------------------------
# Dream cycle: raw Markdown passes
# ---------------------------------------------------------------------------


def _daily_log(ws: str, body: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    for d in ("memory", "entities", "decisions", "tasks", "intelligence"):
        os.makedirs(os.path.join(ws, d), exist_ok=True)
    _write(ws, f"memory/{today}.md", body)
    return today


DAILY_LOG = f"""Morning notes: see https://github.com/star-ga/proseproject for details.

[IMP-20261008-001]
Type: ImportedMemory
Status: quarantined
Statement: Pulled from https://github.com/evil/{CANARY} — unreviewed.

[D-20261008-001]
Type: Decision
Status: active
Statement: Adopt https://github.com/star-ga/admittedproject for routing.
"""


def test_entity_discovery_skips_withheld_block_text(tmp_path) -> None:
    from mind_mem.dream_cycle import pass_entity_discovery

    ws = str(tmp_path)
    _daily_log(ws, DAILY_LOG)

    proposals = pass_entity_discovery(ws)
    slugs = {p.slug for p in proposals}
    # Positive controls: prose outside any block and an admitted block's text.
    assert "proseproject" in slugs, slugs
    assert "admittedproject" in slugs, slugs
    assert CANARY not in slugs
    assert all(CANARY not in p.excerpt for p in proposals)


def test_consolidation_skips_withheld_block_text(tmp_path) -> None:
    from mind_mem.dream_cycle import pass_consolidation

    ws = str(tmp_path)
    for d in ("memory",):
        os.makedirs(os.path.join(ws, d), exist_ok=True)
    withheld_line = f"Statement: the {CANARY} payload must be promoted everywhere"
    admitted_line = "Statement: the admitted routing fact is repeated across days"
    for offset in range(3):
        day = datetime.fromordinal(datetime.now().toordinal() - offset).strftime("%Y-%m-%d")
        _write(
            ws,
            f"memory/{day}.md",
            f"[IMP-2026100{offset}-001]\nStatus: quarantined\n{withheld_line}\n\n"
            f"[D-2026100{offset}-001]\nStatus: active\n{admitted_line}\n",
        )

    facts = [c.fact_text for c in pass_consolidation(ws, min_occurrences=3)]
    assert any("admitted routing fact" in f for f in facts), facts  # positive control
    assert all(CANARY not in f for f in facts), facts


def test_citation_repair_skips_withheld_block_text(tmp_path) -> None:
    from mind_mem.dream_cycle import pass_citation_repair

    ws = str(tmp_path)
    for d in ("memory", "entities", "decisions", "tasks", "intelligence"):
        os.makedirs(os.path.join(ws, d), exist_ok=True)
    _write(
        ws,
        "decisions/DECISIONS.md",
        "[D-20261008-001]\nStatus: quarantined\nStatement: cites D-20990101-001 " + CANARY + "\n\n"
        "[D-20261008-002]\nStatus: active\nStatement: cites D-20990101-002 too\n",
    )

    broken = pass_citation_repair(ws)
    cited = {b.cited_id for b in broken}
    assert "D-20990101-002" in cited, cited  # positive control
    assert "D-20990101-001" not in cited, cited
    assert all(CANARY not in b.context for b in broken)
    # Line numbers stay true to the file: masking blanks, it never deletes.
    assert next(b for b in broken if b.cited_id == "D-20990101-002").line_number == 7


# ---------------------------------------------------------------------------
# Index readers: project_profile (unbound) and build_core's leaf set
# ---------------------------------------------------------------------------


@pytest.fixture()
def indexed_ws(tmp_path):
    from mind_mem.sqlite_index import _connect, _init_schema

    ws = tmp_path / "ws"
    (ws / "decisions").mkdir(parents=True)
    conn = _connect(str(ws))
    try:
        _init_schema(conn)
        rows = (
            ("D-20261008-001", "active", "Admitted routing decision alphaword"),
            ("D-20261008-002", "active", "Second admitted decision"),
            ("D-20261008-003", "quarantined", f"Withheld {CANARY} statement"),
        )
        for bid, status, statement in rows:
            conn.execute(
                "INSERT INTO blocks (id, type, file, line, status, date, json_blob) "
                "VALUES (?, 'decision', 'decisions/DECISIONS.md', 1, ?, '2026-10-08', ?)",
                (bid, status, json.dumps({"_id": bid, "statement": statement})),
            )
            conn.execute(
                "INSERT INTO index_meta (block_id, content_hash, file_path) VALUES (?, ?, 'decisions/DECISIONS.md')",
                (bid, "h-" + bid),
            )
        conn.commit()
    finally:
        conn.close()
    return ws


def test_project_profile_unbound_path_admits_index_rows(indexed_ws) -> None:
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools.consolidation import project_profile

    with use_workspace(str(indexed_ws)):
        raw = project_profile()
    payload = json.loads(raw)
    assert payload["total_blocks"] == 2, payload  # positive control: admitted rows counted
    assert "alphaword" in raw  # positive control: admitted text is mined
    assert CANARY not in raw


def test_core_leaves_are_admission_filtered(indexed_ws) -> None:
    from mind_mem.sqlite_index import merkle_leaves

    ws = str(indexed_ws)
    every = {bid for bid, _h in merkle_leaves(ws)}
    # The integrity anchor still covers every indexed block ...
    assert "D-20261008-003" in every
    admitted = {bid for bid, _h in merkle_leaves(ws, admitted_only=True)}
    # ... while what build_core ships is only what recall may serve.
    assert admitted == {"D-20261008-001", "D-20261008-002"}


def test_build_core_ships_only_admitted_blocks(indexed_ws, monkeypatch) -> None:
    import tarfile

    # build_core writes a bundle file, so it is admin-scope (no user-scope tool
    # writes); this test is about WHAT it ships, so it runs with the admin grant.
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")

    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools.core import build_core

    with use_workspace(str(indexed_ws)):
        payload = json.loads(build_core("ns", "1.0.0"))
    assert "error" not in payload, payload
    with tarfile.open(payload["path"], "r:gz") as tf:
        blob = b"".join(tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()).decode("utf-8", "replace")
    assert "D-20261008-001" in blob  # positive control
    assert "D-20261008-003" not in blob
