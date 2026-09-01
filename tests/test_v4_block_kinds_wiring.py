# Copyright 2026 STARGA, Inc.
"""``v4.block_kinds`` is WIRED — 5.1.0 restoration slice.

The 5.0.0 sweep deleted this module because nothing imported it. It is back,
and this file is the evidence that it is *reached and doing work*, not merely
importable.

The consumer is the ``mm kinds`` CLI namespace (``mm_cli._cmd_kinds_backfill``
/ ``_cmd_kinds_list``) over :mod:`mind_mem.v4.kind_backfill`. That closes the
module's own deferred note: ``blocks.kind`` shipped with two readers
(``get_block_kind``, ``list_blocks_by_kind``) and no in-package writer, so it
answered ``unspecified`` for every id on every workspace forever.

Working definition, asserted below: **after ``mm kinds backfill``,
``list_blocks_by_kind(ws, ENTITY)`` returns the entity blocks and nothing
else — and the quarantined block in the same corpus is not among them.**
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from mind_mem import mm_cli
from mind_mem.v4 import block_kinds, kind_backfill

#: Text that exists only inside the quarantined block. If it ever reaches the
#: v4 side store, the admission filter did not run.
CANARY = "ZZ-QUARANTINED-CANARY-ZZ"


def _build_workspace(root: Path) -> None:
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "decisions" / "DECISIONS.md").write_text(
        "[D-20260101-001]\n"
        "Statement: Use PostgreSQL for the user database\n"
        "Status: active\n"
        "Date: 2026-01-01\n"
        "\n---\n\n"
        "[D-20260102-009]\n"
        f"Statement: {CANARY} untrusted inbox text\n"
        "Status: quarantined\n"
        "Date: 2026-01-02\n",
        encoding="utf-8",
    )
    (root / "tasks" / "TASKS.md").write_text(
        "[T-20260101-001]\nDescription: Set up PostgreSQL in staging\nStatus: open\n",
        encoding="utf-8",
    )
    (root / "entities" / "projects.md").write_text(
        "[PRJ-mind-mem]\nName: mind-mem\nStatus: active\nFile: src/mind_mem/recall.py\n",
        encoding="utf-8",
    )


def _write_config(root: Path, *, block_kinds_on: bool) -> Path:
    cfg = root / "mind-mem.json"
    body: dict = {"version": "5.1.0", "recall": {"backend": "scan"}}
    if block_kinds_on:
        body["v4"] = {"block_kinds": {"enabled": True}}
    cfg.write_text(json.dumps(body), encoding="utf-8")
    return cfg


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    _build_workspace(root)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(root))
    monkeypatch.setenv("MIND_MEM_SCOPE", "user")
    return root


@pytest.fixture
def armed(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, block_kinds_on=True)))
    return workspace


@pytest.fixture
def disarmed(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, block_kinds_on=False)))
    return workspace


def _rows(ws: Path) -> list[tuple[str, str]]:
    with closing(sqlite3.connect(ws / "index.db")) as conn, conn:
        return sorted(conn.execute("SELECT id, kind FROM blocks").fetchall())


# ---------------------------------------------------------------------------
# The working definition
# ---------------------------------------------------------------------------


class TestTheBackfillWritesKindsTheReadersCanSee:
    def test_cli_backfill_makes_list_blocks_by_kind_answer(self, armed: Path) -> None:
        """The whole point: a reader that could only ever return ``[]``."""
        assert block_kinds.list_blocks_by_kind(armed, block_kinds.BlockKind.ENTITY) == []

        assert mm_cli.main(["kinds", "backfill"]) == 0

        assert block_kinds.list_blocks_by_kind(armed, block_kinds.BlockKind.ENTITY) == ["PRJ-mind-mem"]
        assert block_kinds.get_block_kind(armed, "PRJ-mind-mem") is block_kinds.BlockKind.ENTITY
        assert block_kinds.get_block_kind(armed, "D-20260101-001") is block_kinds.BlockKind.SYNTHESIS
        assert block_kinds.get_block_kind(armed, "T-20260101-001") is block_kinds.BlockKind.STRUCTURED

    def test_multi_label_tags_round_trip_through_the_same_pass(self, armed: Path) -> None:
        """A project that names a source file is BOTH entity and code."""
        assert mm_cli.main(["kinds", "backfill"]) == 0
        tags = block_kinds.get_block_kind_tags(armed, "PRJ-mind-mem")
        assert tags == {block_kinds.BlockKind.ENTITY, block_kinds.BlockKind.CODE}

    def test_mm_kinds_list_prints_what_the_backfill_wrote(self, armed: Path, capsys: pytest.CaptureFixture) -> None:
        assert mm_cli.main(["kinds", "backfill"]) == 0
        capsys.readouterr()
        assert mm_cli.main(["kinds", "list", "--kind", "entity"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["block_ids"] == ["PRJ-mind-mem"]
        assert payload["tags"]["PRJ-mind-mem"] == ["code", "entity"]

    def test_the_pass_is_replayable(self, armed: Path) -> None:
        """Two runs over an unchanged corpus write identical rows.

        The backfill has no clock and no unseeded RNG on purpose; if that ever
        stops being true, ``mm kinds list`` becomes a function of when it last
        ran rather than of the corpus.
        """
        assert mm_cli.main(["kinds", "backfill"]) == 0
        first = _rows(armed)
        assert mm_cli.main(["kinds", "backfill"]) == 0
        assert _rows(armed) == first


# ---------------------------------------------------------------------------
# Admission — with the positive control the rule demands
# ---------------------------------------------------------------------------


class TestQuarantinedContentNeverReachesTheKindIndex:
    def test_the_canary_block_really_is_in_the_corpus(self, armed: Path) -> None:
        """Positive control. Without this the next test proves nothing.

        A ``not in`` assertion passes trivially when the thing never existed,
        so first prove the quarantined block IS enumerated by the very call
        the backfill makes, and that its text IS on disk.
        """
        from mind_mem.storage import iter_blocks

        raw = iter_blocks(str(armed), active_only=False)
        assert "D-20260102-009" in {b.get("_id") for b in raw}
        assert CANARY in (armed / "decisions" / "DECISIONS.md").read_text(encoding="utf-8")

    def test_the_quarantined_block_is_withheld_from_the_index(self, armed: Path) -> None:
        assert mm_cli.main(["kinds", "backfill"]) == 0
        ids = {bid for bid, _kind in _rows(armed)}
        assert "D-20260102-009" not in ids
        assert ids == {"D-20260101-001", "PRJ-mind-mem", "T-20260101-001"}

        with closing(sqlite3.connect(armed / "index.db")) as conn, conn:
            hits = conn.execute("SELECT id FROM blocks WHERE content LIKE ?", (f"%{CANARY}%",)).fetchall()
        assert hits == [], "quarantined text reached the v4 side store"

    def test_a_block_quarantined_since_the_last_run_is_pruned(self, armed: Path) -> None:
        """The index must be able to LOSE a block, not only gain one.

        Filtering at write time is not enough on its own: the row a previous
        run wrote survives a later quarantine, so ``list_blocks_by_kind`` goes
        on naming it and ``blocks.content`` goes on holding withheld text.
        Convergence is what makes re-running the backfill a repair.
        """
        assert mm_cli.main(["kinds", "backfill"]) == 0
        assert "PRJ-mind-mem" in {bid for bid, _k in _rows(armed)}

        projects = armed / "entities" / "projects.md"
        projects.write_text(
            projects.read_text(encoding="utf-8").replace("Status: active", "Status: quarantined"),
            encoding="utf-8",
        )

        result = kind_backfill.backfill(armed)
        assert result.rows_pruned >= 1
        assert "PRJ-mind-mem" not in {bid for bid, _k in _rows(armed)}
        with closing(sqlite3.connect(armed / "index.db")) as conn, conn:
            assert conn.execute("SELECT 1 FROM block_kind_tags WHERE block_id = 'PRJ-mind-mem'").fetchone() is None

    def test_a_stale_row_is_still_withheld_from_mm_kinds_list_before_a_reprune(self, armed: Path, capsys: pytest.CaptureFixture) -> None:
        """Between two backfills the index is a cache, and caches go stale
        fail-open. The reader re-checks the live corpus, and says how many it
        withheld, so drift is visible instead of silent."""
        assert mm_cli.main(["kinds", "backfill"]) == 0
        projects = armed / "entities" / "projects.md"
        projects.write_text(
            projects.read_text(encoding="utf-8").replace("Status: active", "Status: quarantined"),
            encoding="utf-8",
        )
        capsys.readouterr()
        assert mm_cli.main(["kinds", "list", "--kind", "entity"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["block_ids"] == []
        assert payload["withheld"] == 1

    def test_removing_the_admission_call_would_be_caught(self, armed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mutation control: neuter ``admit_corpus`` and the canary lands.

        A filter that is never exercised is indistinguishable from a filter
        that was deleted. This makes the difference observable: with the gate
        turned into a pass-through the assertions above go red, which is what
        makes them evidence rather than decoration.
        """
        import mind_mem.admissibility as adm

        monkeypatch.setattr(adm, "admit_corpus", lambda blocks, **kw: [dict(b) for b in blocks])
        assert mm_cli.main(["kinds", "backfill"]) == 0
        ids = {bid for bid, _kind in _rows(armed)}
        assert "D-20260102-009" in ids, "the mutation did not change the outcome — the gate is not load-bearing here"


# ---------------------------------------------------------------------------
# The wiring is what produces the rows
# ---------------------------------------------------------------------------


class TestTheCallSiteIsLoadBearing:
    def test_the_cli_reaches_the_module_writer(self, armed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``mm kinds backfill`` calls ``block_kinds.set_block_kind`` itself.

        Remove the call site from ``kind_backfill.backfill`` and this records
        nothing — the module is back to having a reader and no writer.
        """
        seen: list[tuple[str, str]] = []
        real = block_kinds.set_block_kind

        def _spy(ws, block_id, kind, *, content=None):
            seen.append((block_id, kind.value if hasattr(kind, "value") else str(kind)))
            return real(ws, block_id, kind, content=content)

        monkeypatch.setattr(block_kinds, "set_block_kind", _spy)
        assert mm_cli.main(["kinds", "backfill"]) == 0
        assert sorted(seen) == [
            ("D-20260101-001", "synthesis"),
            ("PRJ-mind-mem", "entity"),
            ("T-20260101-001", "structured"),
        ]

    def test_classify_block_is_pure_and_total(self, armed: Path) -> None:
        """Every classified block gets at least one kind, and no I/O happens."""
        assert block_kinds.classify_block({"_id": "X-1"}) == {block_kinds.BlockKind.UNSPECIFIED}
        assert block_kinds.classify_block({"_id": "PRJ-x"}) == {block_kinds.BlockKind.ENTITY}
        assert block_kinds.classify_block({"_id": "D-1", "_source_label": "decisions"}) == {block_kinds.BlockKind.SYNTHESIS}
        assert not (armed / "index.db").exists()

    def test_set_block_kind_never_blanks_stored_content(self, armed: Path) -> None:
        """A kind-only update must not empty the text the summariser reads."""
        block_kinds.set_block_kind(armed, "B-1", block_kinds.BlockKind.SOURCE, content="hello world")
        block_kinds.set_block_kind(armed, "B-1", block_kinds.BlockKind.CONCEPT)
        with closing(sqlite3.connect(armed / "index.db")) as conn, conn:
            row = conn.execute("SELECT content, kind FROM blocks WHERE id = 'B-1'").fetchone()
        assert row == ("hello world", "concept")


# ---------------------------------------------------------------------------
# Flag OFF is the 5.0.0 baseline
# ---------------------------------------------------------------------------


class TestFlagOffIsUnchanged:
    def test_backfill_refuses_and_writes_nothing(self, disarmed: Path, capsys: pytest.CaptureFixture) -> None:
        assert mm_cli.main(["kinds", "backfill"]) == 64
        assert "block_kinds" in capsys.readouterr().err
        assert not (disarmed / "index.db").exists()

    def test_list_refuses_and_writes_nothing(self, disarmed: Path) -> None:
        assert mm_cli.main(["kinds", "list", "--kind", "entity"]) == 64
        assert not (disarmed / "index.db").exists()

    def test_an_unknown_kind_is_rejected_before_any_database_work(self, armed: Path) -> None:
        assert mm_cli.main(["kinds", "list", "--kind", "not-a-kind"]) == 64
        assert not (armed / "index.db").exists()

    def test_flag_is_registered(self) -> None:
        from mind_mem.v4 import feature_flags

        assert "block_kinds" in feature_flags.ALL_V4_FLAGS


def test_the_pass_writes_only_the_v4_side_store(armed: Path) -> None:
    """No corpus file is touched: this is an index build, not an ingest."""
    before = {p: p.read_bytes() for p in sorted(armed.rglob("*.md"))}
    assert mm_cli.main(["kinds", "backfill"]) == 0
    after = {p: p.read_bytes() for p in sorted(armed.rglob("*.md"))}
    assert before == after
    assert not (armed / "proposals").exists()
    assert (armed / "index.db").is_file()


def test_backfill_result_counts_are_real(armed: Path) -> None:
    result = kind_backfill.backfill(armed)
    assert result.blocks_scanned == 4
    assert result.blocks_admitted == 3
    assert result.kinds_written == 3
    assert result.kind_counts == {"synthesis": 1, "entity": 1, "structured": 1}
