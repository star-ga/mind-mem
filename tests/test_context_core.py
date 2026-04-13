# Copyright 2026 STARGA, Inc.
"""Tests for .mmcore context cores (v2.3.0)."""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
from pathlib import Path

import pytest

from mind_mem.context_core import (
    CORE_FORMAT_VERSION,
    CoreLoadError,
    CoreManifest,
    CoreRegistry,
    LoadedCore,
    build_core,
    load_core,
)


@pytest.fixture()
def tmp_core():
    with tempfile.TemporaryDirectory() as td:
        yield str(Path(td) / "bundle.mmcore")


# ---------------------------------------------------------------------------
# Namespace validation
# ---------------------------------------------------------------------------


class TestNamespaceValidation:
    def test_empty_rejected(self, tmp_core: str) -> None:
        with pytest.raises(ValueError, match="namespace"):
            build_core(tmp_core, namespace="", version="1")

    def test_path_separators_rejected(self, tmp_core: str) -> None:
        with pytest.raises(ValueError, match="path separators"):
            build_core(tmp_core, namespace="foo/bar", version="1")

    def test_too_long_rejected(self, tmp_core: str) -> None:
        with pytest.raises(ValueError, match="≤128 chars"):
            build_core(tmp_core, namespace="x" * 129, version="1")


# ---------------------------------------------------------------------------
# Build round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_empty_core_builds_and_loads(self, tmp_core: str) -> None:
        manifest = build_core(tmp_core, namespace="test", version="1.0")
        assert manifest.block_count == 0
        assert manifest.edge_count == 0
        loaded = load_core(tmp_core)
        assert loaded.manifest.namespace == "test"
        assert loaded.blocks == []
        assert loaded.edges == []

    def test_core_with_blocks_and_edges(self, tmp_core: str) -> None:
        blocks = [{"_id": "B-1", "type": "decision"}, {"_id": "B-2", "type": "task"}]
        edges = [{"subject": "a", "predicate": "depends_on", "object": "b"}]
        manifest = build_core(
            tmp_core,
            namespace="project",
            version="2.1",
            blocks=blocks,
            edges=edges,
            retrieval_policies={"rrf_k": 60, "bm25_weight": 1.0},
            ontology={"entities": ["PERSON", "PROJECT"]},
        )
        assert manifest.block_count == 2
        assert manifest.edge_count == 1
        assert manifest.has_retrieval_policies is True
        assert manifest.has_ontology is True

        loaded = load_core(tmp_core)
        assert loaded.blocks == blocks
        assert loaded.edges == edges
        assert loaded.retrieval_policies == {"rrf_k": 60, "bm25_weight": 1.0}
        assert loaded.ontology == {"entities": ["PERSON", "PROJECT"]}

    def test_format_version_recorded(self, tmp_core: str) -> None:
        m = build_core(tmp_core, namespace="x", version="1")
        assert m.format_version == CORE_FORMAT_VERSION


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_two_builds_produce_identical_tarballs(self, tmp_path: Path) -> None:
        # Callers that want reproducible builds pin built_at explicitly.
        a = tmp_path / "a.mmcore"
        b = tmp_path / "b.mmcore"
        blocks = [{"_id": "B-1"}, {"_id": "B-2"}]
        kwargs = dict(namespace="x", version="1", blocks=blocks, built_at="2026-04-13T00:00:00Z")
        build_core(str(a), **kwargs)
        build_core(str(b), **kwargs)
        assert a.read_bytes() == b.read_bytes()

    def test_dict_key_order_does_not_change_hash(self, tmp_path: Path) -> None:
        # Hash covers payload entries only — built_at lives in the
        # manifest and is deliberately excluded from the content hash
        # so rebuilds at different times still round-trip verification.
        a = tmp_path / "a.mmcore"
        b = tmp_path / "b.mmcore"
        build_core(
            str(a),
            namespace="x", version="1",
            blocks=[{"a": 1, "b": 2}],
        )
        build_core(
            str(b),
            namespace="x", version="1",
            blocks=[{"b": 2, "a": 1}],  # same semantics, different insertion
        )
        assert load_core(str(a)).manifest.content_hash == load_core(str(b)).manifest.content_hash


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


class TestIntegrity:
    def test_tampered_payload_fails_verify(self, tmp_core: str) -> None:
        build_core(
            tmp_core, namespace="x", version="1",
            blocks=[{"_id": "B-1"}, {"_id": "B-2"}],
        )
        # Repackage the tar with a modified blocks.jsonl but the
        # manifest still claiming the original content_hash.
        with tarfile.open(tmp_core, "r:gz") as tf:
            entries = {m.name: tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}
        tampered = b'{"_id":"MALICIOUS"}\n' + entries["blocks.jsonl"]
        entries["blocks.jsonl"] = tampered
        with tarfile.open(tmp_core, "w:gz") as tf:
            for name in sorted(entries):
                data = entries[name]
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                info.mtime = 0
                tf.addfile(info, io.BytesIO(data))
        with pytest.raises(CoreLoadError, match="content hash mismatch"):
            load_core(tmp_core, verify=True)

    def test_verify_false_skips_integrity_check(self, tmp_core: str) -> None:
        build_core(
            tmp_core, namespace="x", version="1",
            blocks=[{"_id": "B-1"}, {"_id": "B-2"}],
        )
        # Same tamper as above.
        with tarfile.open(tmp_core, "r:gz") as tf:
            entries = {m.name: tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}
        entries["blocks.jsonl"] = b'{"_id":"X"}\n' + entries["blocks.jsonl"]
        with tarfile.open(tmp_core, "w:gz") as tf:
            for name in sorted(entries):
                data = entries[name]
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                info.mtime = 0
                tf.addfile(info, io.BytesIO(data))
        # With verify=False we still parse it, just trust the caller.
        loaded = load_core(tmp_core, verify=False)
        assert any(b["_id"] == "X" for b in loaded.blocks)

    def test_missing_manifest_raises(self, tmp_core: str) -> None:
        # Build a tar with no manifest.json entry.
        with tarfile.open(tmp_core, "w:gz") as tf:
            data = b'{"_id":"x"}\n'
            info = tarfile.TarInfo(name="blocks.jsonl")
            info.size = len(data)
            info.mtime = 0
            tf.addfile(info, io.BytesIO(data))
        with pytest.raises(CoreLoadError, match="missing manifest"):
            load_core(tmp_core)

    def test_unknown_entry_rejected(self, tmp_path: Path) -> None:
        """Audit regression: archives with non-known entries are refused."""
        bundle = tmp_path / "odd.mmcore"
        # Build a valid core, then repack with an extra unknown file.
        build_core(str(bundle), namespace="x", version="1", blocks=[{"_id": "A"}])
        with tarfile.open(bundle, "r:gz") as tf:
            entries = {m.name: tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}
        entries["EXTRA_STUFF.bin"] = b"sneaky"
        with tarfile.open(bundle, "w:gz") as tf:
            for name in sorted(entries):
                data = entries[name]
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                info.mtime = 0
                tf.addfile(info, io.BytesIO(data))
        with pytest.raises(CoreLoadError, match="unknown entry"):
            load_core(str(bundle))

    def test_oversized_entry_rejected(self, tmp_path: Path) -> None:
        """Audit regression: entries larger than max_entry_bytes are refused."""
        bundle = tmp_path / "huge.mmcore"
        build_core(str(bundle), namespace="x", version="1", blocks=[{"_id": "A"}])
        with pytest.raises(CoreLoadError, match="exceeds"):
            load_core(str(bundle), max_entry_bytes=4)

    def test_block_count_mismatch_rejected(self, tmp_path: Path) -> None:
        """Audit regression: manifest count must match decoded entries."""
        bundle = tmp_path / "lied.mmcore"
        build_core(str(bundle), namespace="x", version="1", blocks=[{"_id": "A"}])
        # Rewrite manifest.block_count without changing blocks.jsonl.
        with tarfile.open(bundle, "r:gz") as tf:
            entries = {m.name: tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}
        manifest_obj = json.loads(entries["manifest.json"])
        manifest_obj["block_count"] = 999
        entries["manifest.json"] = json.dumps(manifest_obj, sort_keys=True, separators=(",", ":")).encode()
        with tarfile.open(bundle, "w:gz") as tf:
            for name in sorted(entries):
                data = entries[name]
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                info.mtime = 0
                tf.addfile(info, io.BytesIO(data))
        with pytest.raises(CoreLoadError, match="block count mismatch"):
            load_core(str(bundle))

    def test_unsupported_format_version_rejected(self, tmp_path: Path) -> None:
        # Hand-craft a core with a bogus format_version.
        bundle = tmp_path / "odd.mmcore"
        manifest = {
            "namespace": "x",
            "version": "1",
            "format_version": "99.0",
            "built_at": "2026-04-13T00:00:00Z",
            "block_count": 0,
            "edge_count": 0,
            "has_retrieval_policies": False,
            "has_ontology": False,
            "content_hash": "0" * 64,
            "metadata": {},
        }
        with tarfile.open(bundle, "w:gz") as tf:
            body = json.dumps(manifest).encode("utf-8")
            info = tarfile.TarInfo(name="manifest.json")
            info.size = len(body)
            info.mtime = 0
            tf.addfile(info, io.BytesIO(body))
        with pytest.raises(CoreLoadError, match="unsupported core format"):
            load_core(str(bundle))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_load_unload_round_trip(self, tmp_core: str) -> None:
        build_core(tmp_core, namespace="foo", version="1", blocks=[{"_id": "A"}])
        reg = CoreRegistry()
        loaded = reg.load(tmp_core)
        assert loaded.manifest.namespace == "foo"
        assert "foo" in reg.namespaces()
        assert reg.unload("foo") is True
        assert reg.namespaces() == []

    def test_unload_missing_returns_false(self) -> None:
        assert CoreRegistry().unload("nope") is False

    def test_max_cores_enforced(self, tmp_path: Path) -> None:
        reg = CoreRegistry(max_cores=2)
        for i in range(2):
            bundle = tmp_path / f"c{i}.mmcore"
            build_core(str(bundle), namespace=f"ns-{i}", version="1")
            reg.load(str(bundle))
        overflow = tmp_path / "c3.mmcore"
        build_core(str(overflow), namespace="ns-3", version="1")
        with pytest.raises(RuntimeError, match="registry full"):
            reg.load(str(overflow))

    def test_reload_same_namespace_replaces(self, tmp_path: Path) -> None:
        reg = CoreRegistry(max_cores=1)
        bundle = tmp_path / "c.mmcore"
        build_core(str(bundle), namespace="same", version="1")
        first = reg.load(str(bundle))
        # Reloading the same namespace does not overflow the registry.
        second = reg.load(str(bundle))
        assert first.manifest.namespace == second.manifest.namespace

    def test_max_cores_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            CoreRegistry(max_cores=0)

    def test_stats_reports_loaded_cores(self, tmp_path: Path) -> None:
        reg = CoreRegistry()
        for name in ("alpha", "beta"):
            bundle = tmp_path / f"{name}.mmcore"
            build_core(str(bundle), namespace=name, version="1", blocks=[{"_id": f"B-{name}"}])
            reg.load(str(bundle))
        stats = reg.stats()
        names = [s["namespace"] for s in stats]
        assert names == ["alpha", "beta"]
