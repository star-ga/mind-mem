# Copyright 2026 STARGA, Inc.
"""``v4.hnsw_kind_index`` is WIRED — 5.0.1 restoration slice.

Three call sites, all reachable:

* **write** — step 4 of ``mm kinds backfill`` registers each admitted block's
  embedding under its primary kind, so a kind partition exists to scan.
* **read** — ``find_similar(block_id, limit, kind=...)`` (MCP tool) answers
  from that partition instead of the co-occurrence table.
* **status** — ``index_stats`` reports ``backend_status``, which is the only
  way an operator can learn which backend actually serves a kind query.

Working definition, asserted below: **after a backfill, ``find_similar`` with
a ``kind`` returns the other block of that kind, and the reported ``method``
says brute-force — because that is what runs.** The module ships no ANN
backend (its own docstring says so); a payload claiming "HNSW" would promise
a complexity guarantee nothing here provides.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from mind_mem import mm_cli
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.memory_ops import index_stats
from mind_mem.mcp.tools.recall import find_similar
from mind_mem.v4 import hnsw_kind_index

CANARY_ID = "PRJ-quarantined-canary"


def _build_workspace(root: Path) -> None:
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "decisions" / "DECISIONS.md").write_text(
        "[D-20260101-001]\nStatement: Use PostgreSQL for the user database\nStatus: active\n",
        encoding="utf-8",
    )
    (root / "entities" / "projects.md").write_text(
        "[PRJ-mind-mem]\nName: mind-mem persistent memory\nStatus: active\n"
        "\n---\n\n"
        "[PRJ-mind-nerve]\nName: mind-nerve skill router\nStatus: active\n"
        "\n---\n\n"
        f"[{CANARY_ID}]\nName: untrusted peer project\nStatus: quarantined\n",
        encoding="utf-8",
    )


def _write_config(root: Path, *, hnsw: bool) -> Path:
    cfg = root / "mind-mem.json"
    v4: dict = {"block_kinds": {"enabled": True}, "embedding_pipeline": {"enabled": True}}
    if hnsw:
        v4["hnsw_kind_index"] = {"enabled": True}
    cfg.write_text(json.dumps({"version": "5.0.1", "recall": {"backend": "scan"}, "v4": v4}), encoding="utf-8")
    return cfg


@pytest.fixture(autouse=True)
def _no_model_download(monkeypatch: pytest.MonkeyPatch):
    """Keep the stdlib hashed-trigram embedder: no sentence-transformers."""
    import mind_mem.recall_vector as rv

    def _boom(config):
        raise ImportError("sentence-transformers not installed")

    monkeypatch.setattr(rv, "VectorBackend", _boom)


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
    monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, hnsw=True)))
    return workspace


@pytest.fixture
def disarmed(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, hnsw=False)))
    return workspace


def _registered(ws: Path) -> dict[str, str]:
    with closing(sqlite3.connect(ws / "index.db")) as conn, conn:
        return dict(conn.execute("SELECT block_id, kind FROM block_kind_embeddings"))


# ---------------------------------------------------------------------------
# The working definition
# ---------------------------------------------------------------------------


class TestKindFilteredNeighbours:
    def test_backfill_registers_one_row_per_admitted_block(self, armed: Path) -> None:
        assert mm_cli.main(["kinds", "backfill"]) == 0
        assert _registered(armed) == {
            "D-20260101-001": "synthesis",
            "PRJ-mind-mem": "entity",
            "PRJ-mind-nerve": "entity",
        }

    def test_find_similar_answers_from_the_kind_partition(self, armed: Path) -> None:
        assert mm_cli.main(["kinds", "backfill"]) == 0
        with use_workspace(str(armed)):
            payload = json.loads(find_similar("PRJ-mind-mem", 5, "entity"))
        assert payload["kind"] == "entity"
        assert [h["block_id"] for h in payload["similar"]] == ["PRJ-mind-nerve"]
        assert payload["similar"][0]["distance"] >= 0.0

    def test_the_source_block_is_not_its_own_neighbour(self, armed: Path) -> None:
        assert mm_cli.main(["kinds", "backfill"]) == 0
        with use_workspace(str(armed)):
            payload = json.loads(find_similar("PRJ-mind-mem", 5, "entity"))
        assert "PRJ-mind-mem" not in [h["block_id"] for h in payload["similar"]]

    def test_the_partition_really_filters_by_kind(self, armed: Path) -> None:
        """A synthesis block is not returned as an entity neighbour."""
        assert mm_cli.main(["kinds", "backfill"]) == 0
        with use_workspace(str(armed)):
            payload = json.loads(find_similar("PRJ-mind-mem", 5, "entity"))
        assert "D-20260101-001" not in [h["block_id"] for h in payload["similar"]]

    def test_the_method_is_labelled_honestly(self, armed: Path) -> None:
        """No ANN backend exists yet — the payload must not imply one."""
        assert mm_cli.main(["kinds", "backfill"]) == 0
        with use_workspace(str(armed)):
            payload = json.loads(find_similar("PRJ-mind-mem", 5, "entity"))
        assert payload["method"] == "kind-partition-brute-force-cosine"
        assert "hnsw" not in payload["method"].lower()

        with use_workspace(str(armed)):
            stats = json.loads(index_stats())
        assert stats["v4_indexes"]["hnsw_kind_index"]["backend"] == "brute_force"


# ---------------------------------------------------------------------------
# The call sites are load-bearing
# ---------------------------------------------------------------------------


class TestTheCallSitesAreLoadBearing:
    def test_the_register_call_site_is_reached(self, armed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[tuple[str, str]] = []
        real = hnsw_kind_index.register_block_embedding

        def _spy(ws, bid, kind, emb):
            seen.append((bid, kind))
            return real(ws, bid, kind, emb)

        monkeypatch.setattr(hnsw_kind_index, "register_block_embedding", _spy)
        assert mm_cli.main(["kinds", "backfill"]) == 0
        assert sorted(seen) == [
            ("D-20260101-001", "synthesis"),
            ("PRJ-mind-mem", "entity"),
            ("PRJ-mind-nerve", "entity"),
        ]

    def test_the_knn_call_site_is_reached(self, armed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        assert mm_cli.main(["kinds", "backfill"]) == 0
        seen: list[str] = []
        real = hnsw_kind_index.knn_by_kind

        def _spy(ws, kind, query, *, k=10):
            seen.append(kind)
            return real(ws, kind, query, k=k)

        monkeypatch.setattr(hnsw_kind_index, "knn_by_kind", _spy)
        with use_workspace(str(armed)):
            find_similar("PRJ-mind-mem", 5, "entity")
        assert seen == ["entity"]

    def test_get_block_embedding_round_trips_the_registered_vector(self, armed: Path) -> None:
        """The reader the register/query pair shipped without.

        Re-embedding the source block instead would give a vector from
        whatever embedder is loaded *now*, which need not even share the
        partition's dimension — a silently empty answer.
        """
        assert mm_cli.main(["kinds", "backfill"]) == 0
        vec = hnsw_kind_index.get_block_embedding(armed, "PRJ-mind-mem")
        assert len(vec) == 128
        hits = hnsw_kind_index.knn_by_kind(armed, "entity", vec, k=2)
        assert hits[0][0] == "PRJ-mind-mem"
        assert hits[0][1] == pytest.approx(0.0, abs=1e-5)

    def test_a_truncated_payload_is_refused_not_unpacked(self, armed: Path) -> None:
        assert mm_cli.main(["kinds", "backfill"]) == 0
        with closing(sqlite3.connect(armed / "index.db")) as conn, conn:
            conn.execute("UPDATE block_kind_embeddings SET payload = ? WHERE block_id = ?", (b"\x00\x01", "PRJ-mind-mem"))
        assert hnsw_kind_index.get_block_embedding(armed, "PRJ-mind-mem") == []


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


class TestWithheldBlocksAreNotNeighbours:
    def test_the_canary_is_in_the_corpus(self, armed: Path) -> None:
        """Positive control."""
        from mind_mem.storage import iter_blocks

        assert CANARY_ID in {b.get("_id") for b in iter_blocks(str(armed), active_only=False)}

    def test_it_is_never_registered(self, armed: Path) -> None:
        assert mm_cli.main(["kinds", "backfill"]) == 0
        assert CANARY_ID not in _registered(armed)

    def test_a_stale_embedding_row_is_pruned_on_the_next_backfill(self, armed: Path) -> None:
        """Re-running the backfill reclaims the vector, not just hides it."""
        assert mm_cli.main(["kinds", "backfill"]) == 0
        assert "PRJ-mind-nerve" in _registered(armed)

        projects = armed / "entities" / "projects.md"
        projects.write_text(
            projects.read_text(encoding="utf-8").replace(
                "[PRJ-mind-nerve]\nName: mind-nerve skill router\nStatus: active",
                "[PRJ-mind-nerve]\nName: mind-nerve skill router\nStatus: quarantined",
            ),
            encoding="utf-8",
        )
        assert mm_cli.main(["kinds", "backfill"]) == 0
        assert "PRJ-mind-nerve" not in _registered(armed)
        assert hnsw_kind_index.get_block_embedding(armed, "PRJ-mind-nerve") == []

    def test_a_block_quarantined_after_the_backfill_stops_being_served(self, armed: Path) -> None:
        """The partition is a cache, and a cache goes stale fail-open.

        Registering only admitted blocks is not enough on its own: the row
        survives a later quarantine. The read leg re-checks against the LIVE
        corpus, so the neighbour disappears without a re-backfill.
        """
        assert mm_cli.main(["kinds", "backfill"]) == 0
        with use_workspace(str(armed)):
            before = json.loads(find_similar("PRJ-mind-mem", 5, "entity"))
        assert [h["block_id"] for h in before["similar"]] == ["PRJ-mind-nerve"]

        projects = armed / "entities" / "projects.md"
        projects.write_text(
            projects.read_text(encoding="utf-8").replace(
                "[PRJ-mind-nerve]\nName: mind-nerve skill router\nStatus: active",
                "[PRJ-mind-nerve]\nName: mind-nerve skill router\nStatus: quarantined",
            ),
            encoding="utf-8",
        )
        assert CANARY_ID in projects.read_text(encoding="utf-8")

        assert hnsw_kind_index.get_block_embedding(armed, "PRJ-mind-nerve"), "the stale row must still exist"
        with use_workspace(str(armed)):
            after = json.loads(find_similar("PRJ-mind-mem", 5, "entity"))
        assert after["similar"] == []


# ---------------------------------------------------------------------------
# Flag OFF
# ---------------------------------------------------------------------------


class TestFlagOffIsUnchanged:
    def test_find_similar_is_byte_identical(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, hnsw=False)))
        with use_workspace(str(workspace)):
            without_kind = find_similar("PRJ-mind-mem", 5)
            with_kind = find_similar("PRJ-mind-mem", 5, "entity")
        assert with_kind == without_kind
        assert json.loads(with_kind)["method"] == "co-occurrence"

    def test_flag_off_never_calls_the_module(self, disarmed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _explode(*a, **kw):
            raise AssertionError("hnsw_kind_index ran with the flag OFF")

        monkeypatch.setattr(hnsw_kind_index, "knn_by_kind", _explode)
        monkeypatch.setattr(hnsw_kind_index, "get_block_embedding", _explode)
        monkeypatch.setattr(hnsw_kind_index, "register_block_embedding", _explode)
        monkeypatch.setattr(hnsw_kind_index, "backend_status", _explode)
        assert mm_cli.main(["kinds", "backfill"]) == 0
        with use_workspace(str(disarmed)):
            find_similar("PRJ-mind-mem", 5, "entity")
            assert "v4_indexes" not in json.loads(index_stats())

    def test_a_kind_query_on_a_never_backfilled_workspace_falls_through(self, armed: Path) -> None:
        """Passing ``kind`` must not turn into an error for the caller."""
        with use_workspace(str(armed)):
            payload = json.loads(find_similar("PRJ-mind-mem", 5, "entity"))
        assert payload["method"] == "co-occurrence"

    def test_index_stats_does_not_create_the_store(self, armed: Path) -> None:
        """A diagnostic must not be the thing that builds a store.

        ``backend_status`` opens index.db to probe for sqlite-vec, and opening
        a SQLite path creates it. Reported as unbuilt instead.
        """
        with use_workspace(str(armed)):
            stats = json.loads(index_stats())
        assert stats["v4_indexes"]["hnsw_kind_index"]["status"] == "not_built"
        assert not (armed / "index.db").exists()

    def test_flag_is_registered(self) -> None:
        from mind_mem.v4 import feature_flags

        assert "hnsw_kind_index" in feature_flags.ALL_V4_FLAGS
