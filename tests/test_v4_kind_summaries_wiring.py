# Copyright 2026 STARGA, Inc.
"""``v4.kind_summaries`` is WIRED — 5.1.0 restoration slice.

Two halves, both reached from a real entry point:

* **refresh** — ``mm kinds backfill`` (``mm_cli._cmd_kinds_backfill`` ->
  :func:`mind_mem.v4.kind_backfill.backfill`) rebuilds one summary per kind,
  in the architect's dependency order: it runs *after* ``block_kinds`` has
  written the ``blocks`` rows, because ``refresh_summary`` reads them.
* **read** — the ``category_summary`` MCP tool surfaces the stored summaries
  beside the category distiller's output. ``category_summary`` answers "what
  does this workspace know about X" along the category axis; this is the same
  question along the kind axis.

Working definition, asserted below: **after a backfill, ``category_summary``
returns one summary per kind with a ``block_count`` that matches the number
of blocks of that kind.**
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem import mm_cli
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.benchmark import category_summary
from mind_mem.v4 import kind_summaries


def _build_workspace(root: Path) -> None:
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "decisions" / "DECISIONS.md").write_text(
        "[D-20260101-001]\nStatement: Use PostgreSQL for the user database\nStatus: active\n"
        "\n---\n\n"
        "[D-20260103-001]\nStatement: Use Redis for caching hot recall results\nStatus: active\n",
        encoding="utf-8",
    )
    (root / "entities" / "projects.md").write_text(
        "[PRJ-mind-mem]\nName: mind-mem\nStatus: active\n",
        encoding="utf-8",
    )


def _write_config(root: Path, *, kinds: bool, summaries: bool) -> Path:
    cfg = root / "mind-mem.json"
    v4: dict = {}
    if kinds:
        v4["block_kinds"] = {"enabled": True}
    if summaries:
        v4["kind_summaries"] = {"enabled": True}
    body: dict = {"version": "5.1.0", "recall": {"backend": "scan"}}
    if v4:
        body["v4"] = v4
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
    monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, kinds=True, summaries=True)))
    return workspace


@pytest.fixture
def kinds_only(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """block_kinds ON, kind_summaries OFF — the step-2-skipped baseline."""
    monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, kinds=True, summaries=False)))
    return workspace


# ---------------------------------------------------------------------------
# The working definition
# ---------------------------------------------------------------------------


class TestRefreshAndReadRoundTrip:
    def test_backfill_refreshes_one_summary_per_kind(self, armed: Path) -> None:
        assert kind_summaries.list_summaries(armed) == []

        assert mm_cli.main(["kinds", "backfill"]) == 0

        stored = {s.kind: s for s in kind_summaries.list_summaries(armed)}
        assert set(stored) == {"synthesis", "entity"}
        assert stored["synthesis"].block_count == 2
        assert stored["entity"].block_count == 1
        assert "PostgreSQL" in stored["synthesis"].summary
        assert "mind-mem" in stored["entity"].summary

    def test_category_summary_surfaces_them(self, armed: Path) -> None:
        assert mm_cli.main(["kinds", "backfill"]) == 0
        with use_workspace(str(armed)):
            payload = json.loads(category_summary("database"))
        sections = {s["kind"]: s for s in payload["kind_summaries"]}
        assert set(sections) == {"synthesis", "entity"}
        assert sections["synthesis"]["block_count"] == 2

    def test_the_read_side_is_read_only(self, armed: Path) -> None:
        """``category_summary`` must not create or mutate the store."""
        with use_workspace(str(armed)):
            category_summary("database")
        assert not (armed / "index.db").exists()

        assert mm_cli.main(["kinds", "backfill"]) == 0
        before = (armed / "index.db").read_bytes()
        with use_workspace(str(armed)):
            category_summary("database")
        assert (armed / "index.db").read_bytes() == before


# ---------------------------------------------------------------------------
# Dependency order — step 2 depends on step 1 having run
# ---------------------------------------------------------------------------


class TestItRunsAfterBlockKinds:
    def test_summaries_are_built_from_the_rows_block_kinds_wrote(self, armed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Order matters: reverse it and every summary is empty.

        ``refresh_summary`` reads ``blocks(content, kind)``. Run it before the
        classifier has written those rows and it returns ``None`` for every
        kind — which is exactly what this records if the ordering regresses.
        """
        import mind_mem.v4.block_kinds as bk

        order: list[str] = []
        real_set = bk.set_block_kind
        real_refresh = kind_summaries.refresh_summary

        def _spy_set(*a, **kw):
            order.append("set_block_kind")
            return real_set(*a, **kw)

        def _spy_refresh(*a, **kw):
            order.append("refresh_summary")
            return real_refresh(*a, **kw)

        monkeypatch.setattr(bk, "set_block_kind", _spy_set)
        monkeypatch.setattr(kind_summaries, "refresh_summary", _spy_refresh)
        assert mm_cli.main(["kinds", "backfill"]) == 0

        assert "set_block_kind" in order and "refresh_summary" in order
        last_write = len(order) - 1 - order[::-1].index("set_block_kind")
        assert order.index("refresh_summary") > last_write

    def test_the_refresh_call_site_is_load_bearing(self, armed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(kind_summaries, "refresh_summary", lambda ws, kind: calls.append(kind))
        assert mm_cli.main(["kinds", "backfill"]) == 0
        assert sorted(calls) == ["entity", "synthesis"]


# ---------------------------------------------------------------------------
# Flag OFF
# ---------------------------------------------------------------------------


class TestFlagOffIsUnchanged:
    def test_backfill_skips_the_step_entirely(self, kinds_only: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _explode(*a, **kw):
            raise AssertionError("kind_summaries ran with the flag OFF")

        monkeypatch.setattr(kind_summaries, "refresh_summary", _explode)
        assert mm_cli.main(["kinds", "backfill"]) == 0

    def test_category_summary_payload_is_byte_identical(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A present-but-disabled section changes not one byte of the output."""
        monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, kinds=False, summaries=False)))
        with use_workspace(str(workspace)):
            without = category_summary("database")

        monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, kinds=True, summaries=False)))
        with use_workspace(str(workspace)):
            with_disabled = category_summary("database")

        assert with_disabled == without
        assert "kind_summaries" not in json.loads(without)

    def test_flag_off_never_calls_the_module(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The probe is a config read and nothing else."""
        monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, kinds=False, summaries=False)))

        def _explode(*a, **kw):
            raise AssertionError("list_summaries ran with the flag OFF")

        monkeypatch.setattr(kind_summaries, "list_summaries", _explode)
        with use_workspace(str(workspace)):
            assert "kind_summaries" not in json.loads(category_summary("database"))

    def test_flag_is_registered(self) -> None:
        from mind_mem.v4 import feature_flags

        assert "kind_summaries" in feature_flags.ALL_V4_FLAGS
