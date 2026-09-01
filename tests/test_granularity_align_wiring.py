# Copyright 2026 STARGA, Inc.
"""Acceptance gate for the ``granularity_align`` wiring (5.1.0 restoration, slice 2).

``granularity_align`` shipped in v4.0.x, kept its own tests, and was called by
nothing — which is why 5.0.0 deleted it. The module was restored and is now
reachable through ``plan_consolidation``: with ``v4.granularity_align`` on, the
consolidation plan carries a ``granularity_align`` section listing pairs of
blocks that make the same claim at different levels of abstraction, plus the
merged block each pair would collapse to.

Four contracts, one class each:

1. the candidates are really there, and are really the module's output —
   thresholds, caps and text all flow through to the answer;
2. the corpus is UNCHANGED by the call — the section is a proposal, and only
   ``approve_apply`` may write;
3. with the flag OFF the tool's JSON is byte-identical to the pre-wiring
   implementation, proved against that implementation loaded from git;
4. the flag PROBE is unobservable when the answer is "off" — the slice-1
   lesson: a probe that logs on a malformed config makes flag-off detectable.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.sqlite_index import _connect, _init_schema


@contextlib.contextmanager
def _mind_mem_stderr():
    """Collect everything mind-mem's own loggers write, as an operator sees it.

    Neither ``caplog`` nor ``capsys``/``capfd`` is reliable here:
    :class:`mind_mem.observability.StructuredLogger` sets ``propagate = False``
    (so nothing reaches the root logger caplog attaches to) and hands its
    handler the ``sys.stderr`` object that existed when the logger was first
    built — which, under pytest, is not the stream the capture fixtures later
    install. So redirect the handlers themselves, and ``sys.stderr`` too, so a
    logger created *inside* the block is captured as well.
    """
    buffer = io.StringIO()
    restore: list[tuple[logging.StreamHandler, object]] = []
    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        if not name.startswith("mind-mem") or not isinstance(logger, logging.Logger):
            continue
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                restore.append((handler, handler.stream))
                handler.setStream(buffer)
    saved_stderr = sys.stderr
    sys.stderr = buffer
    try:
        yield buffer
    finally:
        sys.stderr = saved_stderr
        for handler, stream in restore:
            handler.setStream(stream)


def _events(buffer: io.StringIO) -> list[tuple[str, str]]:
    """``(level, event)`` for every line mind-mem logged.

    Timestamps differ between two runs of the same code, so comparing raw
    stderr would fail for a reason that has nothing to do with the wiring.
    The event names are what an operator would notice appearing.
    """
    events: list[tuple[str, str]] = []
    for line in buffer.getvalue().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            events.append(("raw", line))
            continue
        events.append((str(entry.get("level", "")), str(entry.get("event", ""))))
    return events


# Two pairs, one unrelated-to-each-other. Similarities are stable properties of
# the text (term-frequency cosine, no clock, no randomness):
#   (001, 002) ~= 0.80   — same claim, different abstraction level
#   (003, 004) ~= 1.00   — near-identical restatement
#   everything else = 0.0
BLOCK_A = "DEC-20200101-001"
BLOCK_B = "DEC-20200101-002"
BLOCK_C = "DEC-20200101-003"
BLOCK_D = "DEC-20200101-004"

CORPUS: tuple[tuple[str, str], ...] = (
    (BLOCK_A, "use Q16.16 fixed-point arithmetic for deterministic scoring across substrates"),
    (BLOCK_B, "all scoring must use fixed-point Q16.16 arithmetic for deterministic scoring across every substrate"),
    (BLOCK_C, "the release workflow publishes wheels to PyPI on tag push"),
    (BLOCK_D, "release workflow publishes the wheels on a tag push to PyPI"),
)


def _write_config(ws: Path, settings: object) -> None:
    """Write ``mind-mem.json`` into the workspace with the given flag value."""
    ws.joinpath("mind-mem.json").write_text(
        json.dumps({"version": "5.1.0", "v4": {"granularity_align": settings}}),
        encoding="utf-8",
    )


@pytest.fixture()
def indexed_workspace(tmp_path, monkeypatch):
    """A workspace whose index was written by ``sqlite_index`` itself.

    ``MIND_MEM_CONFIG`` is pinned at a path that does not exist so the ambient
    fallback is a known quantity: these tests are about the workspace's own
    config, and an operator config elsewhere on the machine must not decide
    whether they pass.
    """
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "absent-mind-mem.json"))
    ws = tmp_path / "ws"
    (ws / "decisions").mkdir(parents=True)

    conn = _connect(str(ws))  # the writer decides where the DB lives
    try:
        _init_schema(conn)
        for bid, statement in CORPUS:
            conn.execute(
                "INSERT INTO blocks (id, type, file, line, status, date, tags, parent_id, json_blob) "
                "VALUES (?, 'decision', 'decisions/DECISIONS.md', 1, 'active', '2020-01-01', 'determinism', '', ?)",
                (bid, json.dumps({"Statement": statement, "_id": bid, "Status": "active"})),
            )
            conn.execute(
                "INSERT INTO block_meta (id, importance, access_count, last_accessed) VALUES (?, 0.1, 0, '2020-01-02T00:00:00Z')",
                (bid,),
            )
        conn.commit()
    finally:
        conn.close()
    return ws


def _plan(ws: Path, **kwargs) -> dict:
    from mind_mem.mcp.tools.consolidation import plan_consolidation as tool

    with use_workspace(str(ws)):
        return json.loads(tool(**kwargs))


def _tree_digest(root: Path) -> dict[str, str]:
    """SHA-256 of every regular file under *root*, keyed by relative path.

    SQLite's ``-wal`` / ``-shm`` sidecars are skipped: a read-only connection
    to a WAL database *creates* them, so their presence is a property of
    having opened the index at all, not of having written to it. The write
    that would matter shows up two ways this file checks separately — the
    ``recall.db`` bytes themselves, and an empty write-ahead log.
    """
    digest: dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.endswith(("-wal", "-shm")):
                continue
            path = Path(dirpath) / name
            digest[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


# ---------------------------------------------------------------------------
# 1. The candidates are the module's real output
# ---------------------------------------------------------------------------


class TestMergeCandidatesReachThePlan:
    def test_flag_registered(self) -> None:
        """Guard: an unregistered flag reads as OFF forever, silently."""
        from mind_mem.v4 import feature_flags

        assert "granularity_align" in feature_flags.ALL_V4_FLAGS

    def test_candidates_appear_in_the_consolidation_plan(self, indexed_workspace) -> None:
        _write_config(indexed_workspace, {"enabled": True})
        section = _plan(indexed_workspace)["granularity_align"]

        assert section["scanned_blocks"] == 4
        pairs = {(c["id_a"], c["id_b"]) for c in section["candidates"]}
        assert pairs == {(BLOCK_C, BLOCK_D), (BLOCK_A, BLOCK_B)}

        # Descending similarity is the module's own ordering contract.
        sims = [c["similarity"] for c in section["candidates"]]
        assert sims == sorted(sims, reverse=True)
        assert sims[0] > 0.99  # near-identical restatement
        assert 0.75 <= sims[1] < 0.95  # same claim, different abstraction

    def test_each_candidate_carries_the_merged_block(self, indexed_workspace) -> None:
        """``merge_blocks`` runs too — the section is actionable, not a hint."""
        _write_config(indexed_workspace, {"enabled": True})
        section = _plan(indexed_workspace)["granularity_align"]

        by_pair = {(c["id_a"], c["id_b"]): c for c in section["candidates"]}
        merged = by_pair[(BLOCK_A, BLOCK_B)]["merged"]

        assert merged["merged_from"] == [BLOCK_B, BLOCK_A]  # keep_longer picked B
        assert merged["block_id"] == BLOCK_B
        # The merged statement is real block text pulled from the index, not a
        # placeholder: it is the longer of the two source statements.
        assert merged["statement"] == dict(CORPUS)[BLOCK_B]

    def test_threshold_from_config_drives_the_candidate_set(self, indexed_workspace) -> None:
        """The configured threshold reaches ``find_merge_candidates`` itself."""
        _write_config(indexed_workspace, {"enabled": True, "min_similarity": 0.95})
        section = _plan(indexed_workspace)["granularity_align"]

        assert section["min_similarity"] == 0.95
        # Only the near-identical pair survives a 0.95 threshold.
        assert [(c["id_a"], c["id_b"]) for c in section["candidates"]] == [(BLOCK_C, BLOCK_D)]

    def test_an_out_of_range_threshold_falls_back_to_the_default(self, indexed_workspace) -> None:
        _write_config(indexed_workspace, {"enabled": True, "min_similarity": "loose"})
        section = _plan(indexed_workspace)["granularity_align"]

        from mind_mem.granularity_align import DEFAULT_MIN_SIMILARITY

        assert section["min_similarity"] == DEFAULT_MIN_SIMILARITY
        assert len(section["candidates"]) == 2

    def test_candidate_cap_is_honoured(self, indexed_workspace) -> None:
        _write_config(indexed_workspace, {"enabled": True, "max_candidates": 1})
        section = _plan(indexed_workspace)["granularity_align"]

        assert section["max_candidates"] == 1
        assert [(c["id_a"], c["id_b"]) for c in section["candidates"]] == [(BLOCK_C, BLOCK_D)]

    def test_block_scan_is_bounded_and_says_so(self, indexed_workspace) -> None:
        """An O(n^2) comparison over an unbounded corpus is a DoS, not a feature."""
        _write_config(indexed_workspace, {"enabled": True, "max_blocks": 2})
        section = _plan(indexed_workspace)["granularity_align"]

        assert section["scanned_blocks"] == 2
        assert section["truncated"] is True
        # ``ORDER BY id LIMIT`` — the deterministic prefix, so the 001/002 pair.
        assert [(c["id_a"], c["id_b"]) for c in section["candidates"]] == [(BLOCK_A, BLOCK_B)]

    def test_section_also_reaches_the_maturity_gate_branch(self, indexed_workspace) -> None:
        """Both return paths of the tool are wired, not just the default one."""
        _write_config(indexed_workspace, {"enabled": True})
        payload = _plan(indexed_workspace, maturity_gate=True, min_maturity=0.5)

        assert "maturity_gate" in payload
        assert len(payload["granularity_align"]["candidates"]) == 2

    def test_output_is_deterministic(self, indexed_workspace) -> None:
        """Same corpus, same config, same bytes — no clock, no randomness."""
        _write_config(indexed_workspace, {"enabled": True})
        from mind_mem.mcp.tools.consolidation import plan_consolidation as tool

        with use_workspace(str(indexed_workspace)):
            first = tool()
            second = tool()
        assert first == second

    def test_fact_card_subblocks_are_not_merge_subjects(self, indexed_workspace) -> None:
        """Sub-blocks are extracted fact cards; no file holds them to edit."""
        conn = _connect(str(indexed_workspace))
        try:
            conn.execute(
                "INSERT INTO blocks (id, type, file, line, status, date, tags, parent_id, json_blob) "
                "VALUES ('FACT-1', 'fact', 'decisions/DECISIONS.md', 2, 'active', '2020-01-01', '', ?, ?)",
                (BLOCK_A, json.dumps({"Statement": dict(CORPUS)[BLOCK_A]})),
            )
            conn.commit()
        finally:
            conn.close()

        _write_config(indexed_workspace, {"enabled": True})
        section = _plan(indexed_workspace)["granularity_align"]

        assert section["scanned_blocks"] == 4
        ids = {c["id_a"] for c in section["candidates"]} | {c["id_b"] for c in section["candidates"]}
        assert "FACT-1" not in ids


# ---------------------------------------------------------------------------
# 2. Proposal-only: nothing is written
# ---------------------------------------------------------------------------


class TestTheCorpusIsUntouched:
    def test_workspace_bytes_are_identical_before_and_after(self, indexed_workspace) -> None:
        _write_config(indexed_workspace, {"enabled": True})
        before = _tree_digest(indexed_workspace)

        section = _plan(indexed_workspace)["granularity_align"]
        assert len(section["candidates"]) == 2  # the call really did the work

        assert _tree_digest(indexed_workspace) == before
        # And nothing is sitting in the write-ahead log either.
        wal = indexed_workspace / ".mind-mem-index" / "recall.db-wal"
        assert not wal.exists() or wal.stat().st_size == 0

    def test_the_source_blocks_survive_verbatim(self, indexed_workspace) -> None:
        """A merge that had actually been applied would have rewritten these."""
        _write_config(indexed_workspace, {"enabled": True})
        _plan(indexed_workspace)

        conn = _connect(str(indexed_workspace))
        try:
            rows = dict(conn.execute("SELECT id, json_blob FROM blocks").fetchall())
        finally:
            conn.close()

        assert set(rows) == {bid for bid, _ in CORPUS}
        for bid, statement in CORPUS:
            assert json.loads(rows[bid])["Statement"] == statement

    def test_no_proposal_queue_is_created(self, indexed_workspace) -> None:
        """The section is data. Routing it is a separate, human-gated step."""
        _write_config(indexed_workspace, {"enabled": True})
        _plan(indexed_workspace)

        assert not (indexed_workspace / "SIGNALS.md").exists()
        assert not (indexed_workspace / "decisions" / "DECISIONS.md").exists()

    def test_the_section_states_its_own_contract(self, indexed_workspace) -> None:
        _write_config(indexed_workspace, {"enabled": True})
        section = _plan(indexed_workspace)["granularity_align"]

        assert section["applied"] is False
        assert section["route"] == "propose_update -> approve_apply"

    def test_the_tool_module_cannot_reach_a_writer(self) -> None:
        """Structural, not by comment: no write-capable import exists here.

        ``granularity_align`` is proposal-only *by contract*
        (``granularity_align.py`` docstring), and a contract stated only in
        prose is one refactor away from being false.
        """
        import ast

        import mind_mem

        source = (Path(mind_mem.__file__).parent / "mcp" / "tools" / "consolidation.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module} | {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        }
        assert "mind_mem.granularity_align" in imported, "guard is vacuous unless the walker sees the real imports"
        forbidden = {
            "mind_mem.apply_engine",
            "mind_mem.block_store",
            "mind_mem.governance_gate",
            "mind_mem.self_editing",
        }
        assert not forbidden & imported

    def test_the_index_is_opened_read_only(self, indexed_workspace) -> None:
        """Belt and braces: the connection string itself forbids a write."""
        import sqlite3

        opened: list[str] = []
        real_connect = sqlite3.connect

        def _spy(target, *args, **kwargs):
            opened.append(str(target))
            return real_connect(target, *args, **kwargs)

        _write_config(indexed_workspace, {"enabled": True})
        monkey = pytest.MonkeyPatch()
        try:
            monkey.setattr(sqlite3, "connect", _spy)
            _plan(indexed_workspace)
        finally:
            monkey.undo()

        index_opens = [target for target in opened if "recall.db" in target]
        assert index_opens, "no index connection was opened — the guard would pass vacuously"
        assert all("mode=ro" in target for target in index_opens), index_opens


# ---------------------------------------------------------------------------
# 3. Flag OFF is byte-identical to the pre-wiring tool
# ---------------------------------------------------------------------------


def _pre_wiring_tool(tmp_path: Path):
    """Load ``HEAD``'s consolidation.py verbatim and hand back its tool.

    The module is loaded under a ``mind_mem.mcp.tools.*`` name so its relative
    imports resolve against the real package — this is the pre-change code
    running against the current tree, not a re-implementation of it.
    """
    repo = Path(__file__).resolve().parent.parent
    try:
        blob = subprocess.run(
            ["git", "-C", str(repo), "show", "HEAD:src/mind_mem/mcp/tools/consolidation.py"],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        pytest.skip("git reference revision unavailable")

    ref_path = tmp_path / "pre_wiring_consolidation.py"
    ref_path.write_bytes(blob)
    name = "mind_mem.mcp.tools._pre_granularity_consolidation"
    spec = importlib.util.spec_from_file_location(name, ref_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # pragma: no cover
        sys.modules.pop(name, None)
        raise
    return module


class TestFlagOffIsByteIdentical:
    @pytest.mark.parametrize(
        "settings",
        [
            pytest.param(None, id="no-flag-key"),
            pytest.param({"enabled": False}, id="explicitly-off"),
            pytest.param({"enabled": "true"}, id="string-true-is-not-true"),
            pytest.param(True, id="bare-true-cannot-enable"),
            pytest.param({"min_similarity": 0.1}, id="tunable-without-enable"),
        ],
    )
    def test_matches_the_pre_wiring_implementation(self, indexed_workspace, tmp_path, settings) -> None:
        if settings is not None:
            _write_config(indexed_workspace, settings)

        pre = _pre_wiring_tool(tmp_path)
        try:
            with use_workspace(str(indexed_workspace)):
                expected = pre.plan_consolidation()
                expected_gated = pre.plan_consolidation(maturity_gate=True, min_maturity=0.5)
        finally:
            sys.modules.pop("mind_mem.mcp.tools._pre_granularity_consolidation", None)

        from mind_mem.mcp.tools.consolidation import plan_consolidation as tool

        with use_workspace(str(indexed_workspace)):
            assert tool() == expected
            assert tool(maturity_gate=True, min_maturity=0.5) == expected_gated
        assert "granularity_align" not in expected

    def test_the_guard_would_notice_the_section(self, indexed_workspace, tmp_path) -> None:
        """Positive control: the same comparison FAILS with the flag on.

        Without this, a comparison that could never differ would report
        "byte-identical" for a wiring that does nothing at all.
        """
        _write_config(indexed_workspace, {"enabled": True})

        pre = _pre_wiring_tool(tmp_path)
        try:
            with use_workspace(str(indexed_workspace)):
                expected = pre.plan_consolidation()
        finally:
            sys.modules.pop("mind_mem.mcp.tools._pre_granularity_consolidation", None)

        from mind_mem.mcp.tools.consolidation import plan_consolidation as tool

        with use_workspace(str(indexed_workspace)):
            assert tool() != expected


# ---------------------------------------------------------------------------
# 4. The probe is unobservable when the answer is "off"  (slice-1 lesson)
# ---------------------------------------------------------------------------


class TestTheFlagProbeIsSilent:
    """caplog cannot see these records — ``StructuredLogger`` sets
    ``propagate = False`` and writes to its own stderr handler — so these
    tests read the file descriptor, which is what an operator would see.
    """

    def test_a_malformed_config_says_nothing_the_pre_wiring_build_did_not(self, indexed_workspace, tmp_path, monkeypatch) -> None:
        """A probe that logs on a bad config makes flag-off detectable.

        Slice 1 shipped exactly this bug: the probe called
        ``feature_flags.is_enabled``, which warns ``v4_config_unreadable``, so
        the wired build emitted a line the unwired build did not. The
        comparison is against that unwired build, run on the same broken
        config, rather than against a list of strings this test invented.
        """
        from mind_mem.v4 import feature_flags

        indexed_workspace.joinpath("mind-mem.json").write_text('{"v4": {,}', encoding="utf-8")
        ambient = tmp_path / "ambient-mind-mem.json"
        ambient.write_text('{"v4": {,}', encoding="utf-8")
        monkeypatch.setenv("MIND_MEM_CONFIG", str(ambient))

        pre = _pre_wiring_tool(tmp_path)
        try:
            monkeypatch.setattr(feature_flags, "_last_config_warning", None)
            with _mind_mem_stderr() as before, use_workspace(str(indexed_workspace)):
                expected = pre.plan_consolidation()
        finally:
            sys.modules.pop("mind_mem.mcp.tools._pre_granularity_consolidation", None)

        from mind_mem.mcp.tools.consolidation import plan_consolidation as tool

        monkeypatch.setattr(feature_flags, "_last_config_warning", None)
        with _mind_mem_stderr() as after, use_workspace(str(indexed_workspace)):
            got = tool()

        assert got == expected
        assert "granularity_align" not in json.loads(got)
        assert _events(after) == _events(before)
        assert "v4_config_unreadable" not in after.getvalue()
        # The dedup state is untouched, so a later LOUD read still warns.
        assert feature_flags._last_config_warning is None

    def test_the_quiet_read_leaves_the_warning_state_alone(self, tmp_path, monkeypatch) -> None:
        """Directly on the helper, so the property is pinned where it lives."""
        from mind_mem.v4 import feature_flags

        config = tmp_path / "mind-mem.json"
        config.write_text("{oops", encoding="utf-8")
        monkeypatch.setenv("MIND_MEM_CONFIG", str(config))
        monkeypatch.setattr(feature_flags, "_last_config_warning", None)

        with _mind_mem_stderr() as quiet:
            assert feature_flags.flag_config("granularity_align", quiet=True) == {}
        assert quiet.getvalue() == ""
        assert feature_flags._last_config_warning is None

        # The loud path still behaves exactly as before — the quiet read did
        # not swallow the warning it owes its caller.
        with _mind_mem_stderr() as loud:
            assert feature_flags.flag_config("granularity_align") == {}
        assert "v4_config_unreadable" in loud.getvalue()
        assert feature_flags._last_config_warning is not None

    def test_ambient_config_can_enable_it(self, indexed_workspace, tmp_path, monkeypatch) -> None:
        """``MIND_MEM_CONFIG`` still works — the quiet read is the same lookup."""
        ambient = tmp_path / "ambient-mind-mem.json"
        ambient.write_text(json.dumps({"v4": {"granularity_align": {"enabled": True}}}), encoding="utf-8")
        monkeypatch.setenv("MIND_MEM_CONFIG", str(ambient))

        section = _plan(indexed_workspace)["granularity_align"]
        assert len(section["candidates"]) == 2

    def test_workspace_config_outranks_ambient(self, indexed_workspace, tmp_path, monkeypatch) -> None:
        ambient = tmp_path / "ambient-mind-mem.json"
        ambient.write_text(json.dumps({"v4": {"granularity_align": {"enabled": True}}}), encoding="utf-8")
        monkeypatch.setenv("MIND_MEM_CONFIG", str(ambient))
        _write_config(indexed_workspace, {"enabled": False})

        assert "granularity_align" not in _plan(indexed_workspace)
