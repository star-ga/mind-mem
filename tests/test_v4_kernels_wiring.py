# Copyright 2026 STARGA, Inc.
"""``v4.kernels`` is WIRED — 5.0.1 restoration slice.

The module holds the four named retrieval strategies
(``surprise_weighted`` / ``lineage_first`` / ``contradicts_first`` /
``graph_walk``) that :mod:`mind_mem.v4.cognitive_kernel` declares but does not
implement. Its consumer is ``mm recall --kernel <name>``
(``mm_cli._cmd_kernel_recall``).

Two things make that a real wiring rather than an import:

* **The import IS the registration.** ``cognitive_kernel`` does not import
  this module, so every name except ``default`` raises ``KeyError`` until
  something imports ``mind_mem.v4.kernels``. The CLI does, and a subprocess
  below proves what happens when nothing does.
* **A strategy actually runs.** ``graph_walk`` walks the ``co_retrieval``
  graph and returns ids the governed recall path never produced — which is
  exactly why the CLI filters its output through the admission gate.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest
from _platform_compat import minimal_child_env

from mind_mem import mm_cli
from mind_mem.v4 import cognitive_kernel

CANARY_ID = "D-20260102-009"


def _build_workspace(root: Path) -> None:
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "decisions" / "DECISIONS.md").write_text(
        "[D-20260101-001]\nStatement: Use PostgreSQL for the user database\nStatus: active\n"
        "\n---\n\n"
        f"[{CANARY_ID}]\nStatement: untrusted inbox text about PostgreSQL\nStatus: quarantined\n",
        encoding="utf-8",
    )
    (root / "entities" / "projects.md").write_text(
        "[PRJ-mind-mem]\nName: mind-mem\nStatus: active\n",
        encoding="utf-8",
    )


def _seed_co_retrieval(root: Path) -> None:
    """An edge from an ADMITTED block to the QUARANTINED one.

    This is the positive control. ``graph_walk`` reaches ids straight out of
    this graph, so without the admission filter in the CLI leg the withheld
    block becomes visible through a kernel that recall would never have
    surfaced it through.
    """
    db = root / ".mind-mem-index" / "recall.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS co_retrieval ("
            "mem1_id TEXT NOT NULL, mem2_id TEXT NOT NULL, weight REAL DEFAULT 0.0, "
            "hit_count INTEGER DEFAULT 0, updated_at TEXT, PRIMARY KEY (mem1_id, mem2_id))"
        )
        conn.execute(
            "INSERT OR REPLACE INTO co_retrieval (mem1_id, mem2_id, weight, hit_count) VALUES (?, ?, 1.0, 1)",
            ("D-20260101-001", CANARY_ID),
        )
        conn.commit()


def _write_config(root: Path, *, kernels: bool) -> Path:
    cfg = root / "mind-mem.json"
    body: dict = {"version": "5.0.1", "recall": {"backend": "scan"}, "cache": {"enabled": False}}
    if kernels:
        body["v4"] = {"cognitive_kernel": {"enabled": True}}
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
    monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, kernels=True)))
    return workspace


@pytest.fixture
def disarmed(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, kernels=False)))
    return workspace


def _run(capsys: pytest.CaptureFixture, *argv: str) -> dict:
    assert mm_cli.main(list(argv)) == 0
    return json.loads(capsys.readouterr().out)


# ---------------------------------------------------------------------------
# The working definition
# ---------------------------------------------------------------------------


class TestAllFiveKernelsAreRoutable:
    def test_the_cli_registers_every_named_strategy(self, armed: Path, capsys: pytest.CaptureFixture) -> None:
        payload = _run(capsys, "recall", "PostgreSQL", "--kernel", "default")
        assert payload["registered_kernels"] == [
            "contradicts_first",
            "default",
            "graph_walk",
            "lineage_first",
            "surprise_weighted",
        ]

    @pytest.mark.parametrize(
        "kernel",
        ["default", "surprise_weighted", "lineage_first", "contradicts_first", "graph_walk"],
    )
    def test_every_kernel_answers(self, armed: Path, capsys: pytest.CaptureFixture, kernel: str) -> None:
        payload = _run(capsys, "recall", "PostgreSQL", "--kernel", kernel)
        assert payload["kernel"] == kernel

    def test_graph_walk_degrades_when_there_is_no_lineage_table(self, tmp_path: Path) -> None:
        """The module's own documented degradation.

        Checked against the probe rather than through the CLI on purpose: the
        default kernel runs FIRST inside every strategy, and the v3 recall it
        delegates to creates ``.mind-mem-index/recall.db`` (with the
        ``co_retrieval`` table) on its way past. So by the time a CLI-driven
        ``graph_walk`` looks, the table exists — the degraded branch is real
        but is not reachable that way, and asserting it there would have been
        asserting something the pipeline had already made false.
        """
        import mind_mem.v4.kernels as kernels

        assert kernels._open_lineage_graph(str(tmp_path / "never-used")) is None

    def test_graph_walk_walks_the_graph_when_there_is_one(self, armed: Path, capsys: pytest.CaptureFixture) -> None:
        _seed_co_retrieval(armed)
        payload = _run(capsys, "recall", "PostgreSQL", "--kernel", "graph_walk")
        assert payload["metadata"].get("degraded") is not True
        assert payload["metadata"]["visited"] >= 2

    def test_an_unknown_kernel_name_is_refused_loudly(self, armed: Path, capsys: pytest.CaptureFixture) -> None:
        assert mm_cli.main(["recall", "PostgreSQL", "--kernel", "nope"]) == 64
        assert "valid kernels" in capsys.readouterr().err

    def test_a_valid_but_unimplemented_kernel_names_the_registered_ones(self, armed: Path, capsys: pytest.CaptureFixture) -> None:
        """``recent_first`` is a routing name with no built-in strategy."""
        assert mm_cli.main(["recall", "PostgreSQL", "--kernel", "recent_first"]) == 64
        assert "no kernel registered" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The import IS the registration
# ---------------------------------------------------------------------------


class TestImportingIsWhatRegisters:
    def test_without_the_import_nothing_but_default_resolves(self, armed: Path) -> None:
        """A fresh interpreter that imports only ``cognitive_kernel``.

        In-process this cannot be shown — ``mind_mem.v4.kernels`` is already
        in ``sys.modules`` and the registry is module-global — so the claim is
        checked where it is actually true: a new process.
        """
        script = (
            "from mind_mem.v4.cognitive_kernel import available_kernels, mind_recall\n"
            "print(sorted(k.value for k in available_kernels()))\n"
            "try:\n"
            "    mind_recall('.', 'q', kernel='graph_walk')\n"
            "except KeyError as e:\n"
            "    print('KEYERROR')\n"
        )
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            # Scrubbed, NOT inherited -- the whole claim is about a fresh
            # process. ``minimal_child_env`` adds only what the host needs
            # to start an interpreter at all (on Windows that is
            # ``SystemRoot``, without which the child dies in
            # ``_Py_HashRandomization_Init``) plus an import path that
            # survives pointing the home directory at a tmp dir.
            env=minimal_child_env(
                armed,
                MIND_MEM_WORKSPACE=str(armed),
                MIND_MEM_CONFIG=str(armed / "mind-mem.json"),
            ),
            timeout=120,
            encoding="utf-8",
            errors="replace",
        )
        assert out.returncode == 0, out.stderr
        assert "['default']" in out.stdout, out.stdout
        assert "KEYERROR" in out.stdout, out.stdout

    def test_the_cli_leg_carries_the_import(self) -> None:
        """Static check: the import is in the function, not merely in the file.

        A module-level import elsewhere in ``mm_cli`` would make this pass by
        accident and break the moment that unrelated import moved.
        """
        import ast
        import inspect
        import textwrap

        src = textwrap.dedent(inspect.getsource(mm_cli._cmd_kernel_recall))
        imported = {alias.name for node in ast.walk(ast.parse(src)) if isinstance(node, ast.Import) for alias in node.names}
        assert "mind_mem.v4.kernels" in imported


# ---------------------------------------------------------------------------
# Admission — with the positive control
# ---------------------------------------------------------------------------


class TestKernelHitsAreAdmissionFiltered:
    def test_the_graph_really_reaches_the_quarantined_block(self, armed: Path) -> None:
        """Positive control: the strategy itself DOES return the canary."""
        import mind_mem.v4.kernels as kernels

        _seed_co_retrieval(armed)
        result = kernels.graph_walk_kernel(str(armed), "PostgreSQL")
        assert CANARY_ID in {h.block_id for h in result.hits}

    def test_the_cli_withholds_it(self, armed: Path, capsys: pytest.CaptureFixture) -> None:
        _seed_co_retrieval(armed)
        payload = _run(capsys, "recall", "PostgreSQL", "--kernel", "graph_walk")
        assert CANARY_ID not in {h["block_id"] for h in payload["hits"]}
        assert payload["withheld"] >= 1

    def test_neutering_the_gate_changes_the_outcome(
        self, armed: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation control."""
        import mind_mem.admissibility as adm

        _seed_co_retrieval(armed)
        monkeypatch.setattr(adm, "admissible", lambda blocks, **kw: frozenset(str(b.get("_id", "")) for b in blocks))
        payload = _run(capsys, "recall", "PostgreSQL", "--kernel", "graph_walk")
        assert CANARY_ID in {h["block_id"] for h in payload["hits"]}, "the mutation did not change the outcome"


# ---------------------------------------------------------------------------
# Flag OFF / no --kernel
# ---------------------------------------------------------------------------


class TestFlagOffIsUnchanged:
    def test_plain_recall_is_byte_identical(self, workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, kernels=False)))
        assert mm_cli.main(["recall", "PostgreSQL"]) == 0
        off = capsys.readouterr().out

        monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, kernels=True)))
        assert mm_cli.main(["recall", "PostgreSQL"]) == 0
        assert capsys.readouterr().out == off

    def test_no_kernel_argument_never_touches_the_v4_surface(self, armed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Not even a flag read: the branch is on the CLI argument."""

        def _explode(*a, **kw):
            raise AssertionError("the v4 kernel path ran without --kernel")

        monkeypatch.setattr(cognitive_kernel, "mind_recall", _explode)
        monkeypatch.setattr(mm_cli, "_cmd_kernel_recall", _explode)
        assert mm_cli.main(["recall", "PostgreSQL"]) == 0

    def test_kernel_with_the_flag_off_refuses(self, disarmed: Path, capsys: pytest.CaptureFixture) -> None:
        assert mm_cli.main(["recall", "PostgreSQL", "--kernel", "graph_walk"]) == 64
        assert "cognitive_kernel" in capsys.readouterr().err

    def test_flag_is_registered(self) -> None:
        from mind_mem.v4 import feature_flags

        assert "cognitive_kernel" in feature_flags.ALL_V4_FLAGS
