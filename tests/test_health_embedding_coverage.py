# Copyright 2026 STARGA, Inc.
"""``memory_health`` can finally answer the question it exists to ask.

Two defects made the embedding-coverage probe structurally incapable of
reporting coverage:

1. It called ``recall_vector._index_path(ws)`` — a function that has never
   existed in this repo. Every call raised ``AttributeError``, the probe fell
   into its except branch, and the dashboard reported coverage as ``"unknown"``
   on every workspace, forever. A ``# type: ignore[attr-defined]`` on that line
   had silenced the checker that would have caught it.
2. Inside that unreachable branch it ``struct.unpack``-ed a binary ``<I``
   header out of ``index.json`` — a JSON file no writer has ever given a
   binary header.

The fix routes the probe through ``recall_vector.load_local_index``, the same
reader the dense search path uses, so probe and product cannot disagree about
the format.

Working definition, asserted below: **the probe reports four DISTINCT states —
no index, canonical index with a real coverage number, legacy list shape, and
unreadable/invalid — and ``"unknown"`` survives only where the reason is
stated.**

Every positive case proves the index EXISTS on disk in the shape under test
and that the probe read THAT file, before asserting the number. A test that
only asserts "a number came back" passes trivially against a default.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mind_mem.recall_vector as recall_vector
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.memory_ops import memory_health

#: Total blocks written into every fixture workspace. Kept different from
#: every embedded count below so a coverage number can never be right by
#: coincidence (100.0 from N == M, or 0.0 from a default).
TOTAL_BLOCKS = 4

#: A non-default index directory. The probe must read the CONFIGURED location:
#: pointing it at a directory the default would miss is what proves the number
#: came off the file this test wrote.
INDEX_DIR = "custom-vectors"


def _build_workspace(root: Path) -> None:
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "decisions" / "DECISIONS.md").write_text(
        "".join(f"[D-2026010{i}-001]\nStatement: Decision number {i}\nStatus: active\n\n" for i in range(1, TOTAL_BLOCKS + 1)),
        encoding="utf-8",
    )


def _write_config(root: Path) -> Path:
    cfg = root / "mind-mem.json"
    cfg.write_text(
        json.dumps({"version": "5.0.2", "recall": {"backend": "scan", "index_path": INDEX_DIR}}),
        encoding="utf-8",
    )
    return cfg


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    _build_workspace(root)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(root))
    monkeypatch.setenv("MIND_MEM_SCOPE", "user")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(root)))
    return root


def _index_file(ws: Path) -> Path:
    return ws / INDEX_DIR / "index.json"


def _write_raw_index(ws: Path, body: str) -> Path:
    path = _index_file(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _write_canonical_index(ws: Path, embedded: int) -> Path:
    """Write a canonical index through the PRODUCT's own writer.

    Not a hand-rolled ``json.dump``: hand-rolling a second writer is the exact
    defect the shape-contract work removed, and a test that hand-rolls one can
    pass against a shape the product never produces.
    """
    recall_vector.save_local_index(
        str(ws),
        INDEX_DIR,
        recall_vector.canonical_local_index(
            "all-MiniLM-L6-v2",
            3,
            [{"id": f"D-2026010{i}-001", "excerpt": "x", "status": "active"} for i in range(1, embedded + 1)],
            [[0.1, 0.2, 0.3] for _ in range(embedded)],
        ),
    )
    return _index_file(ws)


def _health(ws: Path) -> dict:
    with use_workspace(str(ws)):
        return json.loads(memory_health())


# ---------------------------------------------------------------------------
# The phantom accessor
# ---------------------------------------------------------------------------


def test_the_function_the_probe_used_to_call_still_does_not_exist() -> None:
    """Pins the root cause so nobody reintroduces the call.

    ``_index_path`` is not a renamed private helper that came back under
    another name — it never existed. The real accessors are the module-level
    ``local_index_file`` and ``load_local_index``.
    """
    assert not hasattr(recall_vector, "_index_path")
    assert callable(recall_vector.load_local_index)
    assert callable(recall_vector.local_index_file)


# ---------------------------------------------------------------------------
# The four states
# ---------------------------------------------------------------------------


class TestTheProbeReportsDistinctStates:
    def test_no_index_reads_absent_not_unknown(self, workspace: Path) -> None:
        assert not _index_file(workspace).exists(), "fixture must start with no index"

        payload = _health(workspace)

        assert payload["embedding_index_state"] == "missing"
        assert payload["embedded_blocks"] == 0
        assert payload["embedding_coverage_pct"] == 0.0
        assert "embedding_index_error" not in payload

    def test_canonical_index_yields_the_real_coverage_number(self, workspace: Path) -> None:
        """The point of the probe. Prove the file, then prove the number.

        3 of 4 blocks embedded is 75.0%: a value no default and no
        all-or-nothing bug can produce.
        """
        path = _write_canonical_index(workspace, embedded=3)

        # Positive control: the index EXISTS, in the canonical shape, with the
        # count under test -- read back off disk, not assumed.
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert path.is_file()
        assert on_disk["schema"] == recall_vector.LOCAL_INDEX_SCHEMA
        assert len(on_disk["blocks"]) == 3
        assert recall_vector.is_canonical_local_index(on_disk)

        payload = _health(workspace)

        assert payload["total_blocks"] == TOTAL_BLOCKS, "the denominator is the one this test computed against"
        assert payload["embedding_index_state"] == "ok"
        assert payload["embedded_blocks"] == 3
        assert payload["embedding_coverage_pct"] == 75.0
        assert payload["embedded_blocks"] != "unknown"
        assert any("Embedding coverage is 75.0%" in r for r in payload["recommendations"])

    def test_the_number_tracks_the_file_and_is_not_a_constant(self, workspace: Path) -> None:
        """Two different indexes, two different numbers, same workspace.

        A probe hard-coding a plausible answer passes the case above; it
        cannot pass this one.
        """
        _write_canonical_index(workspace, embedded=1)
        assert len(json.loads(_index_file(workspace).read_text(encoding="utf-8"))["blocks"]) == 1
        first = _health(workspace)

        _write_canonical_index(workspace, embedded=4)
        assert len(json.loads(_index_file(workspace).read_text(encoding="utf-8"))["blocks"]) == 4
        second = _health(workspace)

        assert (first["embedded_blocks"], first["embedding_coverage_pct"]) == (1, 25.0)
        assert (second["embedded_blocks"], second["embedding_coverage_pct"]) == (4, 100.0)

    def test_full_coverage_raises_no_reindex_recommendation(self, workspace: Path) -> None:
        _write_canonical_index(workspace, embedded=TOTAL_BLOCKS)

        payload = _health(workspace)

        assert payload["embedding_coverage_pct"] == 100.0
        assert not any("Embedding coverage" in r for r in payload["recommendations"])
        assert not any("No vector index found" in r for r in payload["recommendations"])

    def test_legacy_list_shape_is_its_own_answer(self, workspace: Path) -> None:
        """Countable, but not the same health answer as a canonical index.

        Legacy records carry no status/date/file, so the coverage number is
        real while the metadata is incomplete -- the operator needs to be told
        to reindex, which "ok" would never say.
        """
        path = _write_raw_index(
            workspace,
            json.dumps([{"_id": f"D-2026010{i}-001", "embedding": [0.1, 0.2], "text": "x"} for i in range(1, 3)]),
        )

        # Positive control: the file exists and is the LEGACY list shape.
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(on_disk, list) and len(on_disk) == 2
        assert not recall_vector.is_canonical_local_index(on_disk)

        payload = _health(workspace)

        assert payload["embedding_index_state"] == "legacy_list_shape"
        assert payload["embedded_blocks"] == 2
        assert payload["embedding_coverage_pct"] == 50.0
        assert any("legacy list shape" in r for r in payload["recommendations"])

    def test_unreadable_index_is_an_error_with_a_reason(self, workspace: Path) -> None:
        path = _write_raw_index(workspace, "{ this is not json")
        assert path.is_file(), "the corrupt file must exist, or this proves nothing"

        payload = _health(workspace)

        assert payload["embedding_index_state"] == "unreadable"
        assert payload["embedded_blocks"] == "unknown"
        assert payload["embedding_coverage_pct"] == "unknown"
        assert "could not be read" in payload["embedding_index_error"]
        assert any("unreadable" in r for r in payload["recommendations"])

    def test_invalid_shape_is_distinguished_from_unreadable(self, workspace: Path) -> None:
        """Valid JSON the reader refuses is not the same failure as bad JSON."""
        path = _write_raw_index(workspace, json.dumps({"schema": "something/else@9", "vectors": {}}))
        assert json.loads(path.read_text(encoding="utf-8"))["schema"] == "something/else@9"

        payload = _health(workspace)

        assert payload["embedding_index_state"] == "invalid_shape"
        assert payload["embedded_blocks"] == "unknown"
        assert "shape no reader accepts" in payload["embedding_index_error"]


# ---------------------------------------------------------------------------
# The probe reads the file the product reads
# ---------------------------------------------------------------------------


class TestProbeAndProductShareOneReader:
    def test_the_configured_index_path_is_honoured(self, workspace: Path) -> None:
        """An index at the DEFAULT location must not be found under this config.

        This is what proves the number in the canonical case came off the file
        under test rather than off a path the probe guessed.
        """
        default_index = workspace / ".mind-mem-vectors" / "index.json"
        default_index.parent.mkdir(parents=True, exist_ok=True)
        recall_vector.save_local_index(
            str(workspace),
            ".mind-mem-vectors",
            recall_vector.canonical_local_index("m", 3, [{"id": "x"}] * 4, [[0.0]] * 4),
        )
        assert default_index.is_file()

        payload = _health(workspace)

        assert payload["embedding_index_state"] == "missing", "the probe read the default path, not the configured one"
        assert payload["embedded_blocks"] == 0

    def test_the_probe_goes_through_load_local_index(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Break the shared reader and the probe must change its answer.

        Pins the wiring itself: if a future edit re-hand-rolls the read, this
        stays green while the format contract quietly forks again.
        """
        _write_canonical_index(workspace, embedded=3)
        calls: list[tuple[str, str]] = []
        real = recall_vector.load_local_index

        def _spy(ws: str, index_path: str):
            calls.append((ws, index_path))
            return real(ws, index_path)

        monkeypatch.setattr(recall_vector, "load_local_index", _spy)

        payload = _health(workspace)

        assert calls == [(str(workspace), INDEX_DIR)]
        assert payload["embedded_blocks"] == 3
