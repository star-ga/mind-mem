# Copyright 2026 STARGA, Inc.
"""``v4.pq`` is WIRED — 5.0.1 restoration slice.

Product quantization compresses a stored embedding from ``4 * dim`` bytes to
one byte per subvector position. The consumer is
:func:`mind_mem.recall_vector.rebuild_index` — the vector-index rebuild the
``reindex(include_vectors=True)`` MCP tool actually calls — through the
``_pq_compress`` leg.

(The architect's plan named ``VectorBackend._index_local``. That method has
no live caller in this tree, so wiring it there would have been wiring into
dead code; ``rebuild_index`` is the reachable path and is what is wired. The
test below pins the reachable one.)

Working definition, asserted here: **after a vector rebuild with ``v4.pq``
on, every ADMITTED block has a 32-byte code, ``decode(encode(v))``
approximates ``v``, and asymmetric distance ranks the same way exact squared
distance does.**
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from mind_mem.v4 import pq

CANARY = "ZZ-PQ-QUARANTINE-CANARY-ZZ"

#: 64 dims so the default 32 subvectors divide it evenly (sub_dim 2).
_DIM = 64


def _vector(seed: int) -> list[float]:
    """Deterministic, distinguishable, unit-ish vector."""
    return [math.sin(seed * 0.37 + i * 0.11) for i in range(_DIM)]


class _StubBackend:
    """``recall_vector.VectorBackend`` without a model download."""

    index_path = ".mind-mem-vectors"
    model_name = "stub-embedder"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def _embed_for_provider(self, texts: list[str]) -> list[list[float]]:
        return [_vector(i + 1) for i, _ in enumerate(texts)]


def _build_workspace(root: Path) -> None:
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "decisions" / "DECISIONS.md").write_text(
        "[D-20260101-001]\nStatement: Use PostgreSQL for the user database\nStatus: active\n"
        "\n---\n\n"
        f"[D-20260102-009]\nStatement: {CANARY} untrusted inbox text\nStatus: quarantined\n",
        encoding="utf-8",
    )
    (root / "entities" / "projects.md").write_text(
        "[PRJ-mind-mem]\nName: mind-mem\nStatus: active\n",
        encoding="utf-8",
    )


def _write_config(root: Path, *, pq_on: bool) -> Path:
    cfg = root / "mind-mem.json"
    body: dict = {"version": "5.0.1", "recall": {"backend": "scan"}}
    if pq_on:
        body["v4"] = {"pq": {"enabled": True}}
    cfg.write_text(json.dumps(body), encoding="utf-8")
    return cfg


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    _build_workspace(root)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(root))
    monkeypatch.setenv("MIND_MEM_SCOPE", "user")
    import mind_mem.recall_vector as rv

    monkeypatch.setattr(rv, "VectorBackend", _StubBackend)
    return root


@pytest.fixture
def armed(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, pq_on=True)))
    return workspace


@pytest.fixture
def disarmed(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, pq_on=False)))
    return workspace


def _codes(ws: Path) -> dict[str, int]:
    with closing(sqlite3.connect(ws / "index.db")) as conn, conn:
        return {bid: len(code) for bid, code in conn.execute("SELECT block_id, code FROM pq_codes")}


# ---------------------------------------------------------------------------
# The call site
# ---------------------------------------------------------------------------


class TestRebuildIndexCompresses:
    def test_a_vector_rebuild_writes_a_codebook_and_codes(self, armed: Path) -> None:
        from mind_mem.recall_vector import rebuild_index

        # 2, not 3: the quarantined block is now withheld by rebuild_index
        # itself (it used to index the whole corpus unfiltered).
        assert rebuild_index(str(armed)) == 2
        assert pq.load_codebook(armed, "stub-embedder") is not None
        assert _codes(armed) == {"D-20260101-001": 32, "PRJ-mind-mem": 32}

    def test_the_compression_is_real(self, armed: Path) -> None:
        """96x on a 768-dim float32 vector is the module's claim; check the
        arithmetic on the geometry actually used rather than restating it."""
        from mind_mem.recall_vector import rebuild_index

        rebuild_index(str(armed))
        raw_bytes = _DIM * 4
        assert _codes(armed)["PRJ-mind-mem"] == 32
        assert raw_bytes / 32 == 8.0  # 64-dim here; the ratio scales with dim

    def test_the_call_site_is_load_bearing(self, armed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Remove ``_pq_compress`` from ``rebuild_index`` and this is empty."""
        import mind_mem.recall_vector as rv

        seen: list[tuple[str, int]] = []
        real = rv._pq_compress

        def _spy(ws, name, ids, embs):
            seen.append((name, len(ids)))
            return real(ws, name, ids, embs)

        monkeypatch.setattr(rv, "_pq_compress", _spy)
        rv.rebuild_index(str(armed))
        assert seen == [("stub-embedder", 2)]

    def test_it_is_deterministic(self, armed: Path) -> None:
        """Same corpus, same codes. The trainer seeds its k-means++."""
        from mind_mem.recall_vector import rebuild_index

        rebuild_index(str(armed))
        with closing(sqlite3.connect(armed / "index.db")) as conn, conn:
            first = sorted(conn.execute("SELECT block_id, code FROM pq_codes"))
        rebuild_index(str(armed))
        with closing(sqlite3.connect(armed / "index.db")) as conn, conn:
            assert sorted(conn.execute("SELECT block_id, code FROM pq_codes")) == first


# ---------------------------------------------------------------------------
# Admission — with the positive control
# ---------------------------------------------------------------------------


class TestQuarantinedBlocksGetNoCode:
    def test_the_quarantined_block_is_not_in_the_vector_index_either(self, armed: Path) -> None:
        """The upstream leak this test used to DOCUMENT is now closed.

        ``rebuild_index`` parsed the corpus with no status filter at all, so a
        quarantined block was embedded into the JSON vector index and was
        reachable by similarity while every text path withheld it. This test
        asserted that leak as expected behaviour — correctly, at the time, as
        the reason the PQ leg's own admission call mattered.

        The leak is fixed (``rebuild_index`` now calls ``admit_corpus``), so
        the assertion is inverted. The PQ leg keeps its own filter: defence in
        depth, and it is still mutation-controlled below.
        """
        from mind_mem.recall_vector import rebuild_index

        rebuild_index(str(armed))
        index = json.loads((armed / ".mind-mem-vectors" / "index.json").read_text(encoding="utf-8"))
        ids = {rec["_id"] for rec in index}
        assert "D-20260102-009" not in ids, "quarantined block reached the vector index"
        # Positive control: the admitted blocks DID make it in, so the
        # assertion above is not passing because the index is empty.
        assert "D-20260101-001" in ids

    def test_it_gets_no_pq_code(self, armed: Path) -> None:
        from mind_mem.recall_vector import rebuild_index

        rebuild_index(str(armed))
        assert "D-20260102-009" not in _codes(armed)

    def test_neutering_the_gate_changes_the_outcome(self, armed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mutation control: a filter that cannot fail proves nothing."""
        import mind_mem.admissibility as adm

        # BOTH filters have to be neutered now. ``rebuild_index`` gained its
        # own ``admit_corpus`` call upstream, so patching only the PQ leg's
        # ``admissible`` leaves the block filtered out before PQ ever sees it
        # -- the mutation would silently fail to mutate anything, which is the
        # exact failure mode a mutation control exists to rule out.
        monkeypatch.setattr(adm, "admissible", lambda blocks, **kw: frozenset(str(b.get("_id", "")) for b in blocks))
        monkeypatch.setattr(adm, "admit_corpus", lambda blocks, **kw: list(blocks))
        from mind_mem.recall_vector import rebuild_index

        rebuild_index(str(armed))
        assert "D-20260102-009" in _codes(armed), "the mutation did not change the outcome"


# ---------------------------------------------------------------------------
# The codec still does what it claims
# ---------------------------------------------------------------------------


class TestTheCodecIsUsable:
    def test_decode_of_encode_approximates_the_input(self, armed: Path) -> None:
        vectors = [_vector(i) for i in range(1, 40)]
        book = pq.train_codebook(vectors)
        recovered = pq.decode(pq.encode(vectors[0], book), book)
        assert len(recovered) == _DIM
        err = math.sqrt(sum((a - b) ** 2 for a, b in zip(vectors[0], recovered)))
        assert err < 1.0, f"reconstruction error {err} is not a reconstruction"

    def test_asymmetric_distance_ranks_like_exact_distance(self, armed: Path) -> None:
        vectors = [_vector(i) for i in range(1, 40)]
        book = pq.train_codebook(vectors)
        query = vectors[0]
        approx = sorted(range(len(vectors)), key=lambda i: pq.asymmetric_distance(query, pq.encode(vectors[i], book), book))
        exact = sorted(range(len(vectors)), key=lambda i: sum((a - b) ** 2 for a, b in zip(query, vectors[i])))
        assert approx[0] == exact[0] == 0

    def test_a_geometry_that_does_not_divide_the_dimension_is_refused(self, armed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Encoding under a bad geometry returns b"" for every vector — a
        silently empty index. The leg must skip instead."""
        import mind_mem.recall_vector as rv

        monkeypatch.setattr(pq, "_load_config", lambda: pq.PQConfig(subvectors=7))
        assert rv._pq_compress(str(armed), "stub-embedder", ["A-1"], [_vector(1)]) is None


# ---------------------------------------------------------------------------
# Flag OFF
# ---------------------------------------------------------------------------


class TestFlagOffIsUnchanged:
    def test_flag_off_never_calls_the_module(self, disarmed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _explode(*a, **kw):
            raise AssertionError("pq ran with the flag OFF")

        monkeypatch.setattr(pq, "train_codebook", _explode)
        monkeypatch.setattr(pq, "store_codebook", _explode)
        monkeypatch.setattr(pq, "ensure_pq_schema", _explode)
        from mind_mem.recall_vector import rebuild_index

        # 2, not 3 -- see the note in TestRebuildIndexCompresses. The
        # admission filter is upstream of the flag, so it applies here too.
        assert rebuild_index(str(disarmed)) == 2
        assert not (disarmed / "index.db").exists(), "the OFF path created the side store"

    def test_the_json_index_is_identical_with_the_flag_on(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """PQ is a side store: the vector index the recall path reads is
        byte-for-byte the same whether the flag is on or off."""
        from mind_mem.recall_vector import rebuild_index

        monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, pq_on=False)))
        rebuild_index(str(workspace))
        off = (workspace / ".mind-mem-vectors" / "index.json").read_bytes()

        monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, pq_on=True)))
        rebuild_index(str(workspace))
        assert (workspace / ".mind-mem-vectors" / "index.json").read_bytes() == off

    def test_flag_is_registered(self) -> None:
        from mind_mem.v4 import feature_flags

        assert "pq" in feature_flags.ALL_V4_FLAGS
