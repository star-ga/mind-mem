# Copyright 2026 STARGA, Inc.
"""Snapshot content must not reach the corpus through a symlink.

The legacy restore path has three sinks that trust the snapshot directory's
shape: the ``intelligence`` root is joined raw, ``copytree`` dereferences
symlinked entries, and ``MANIFEST.json`` is opened by a raw join. A snapshot
directory is not attacker-authored under normal operation -- but restore also
runs on operator-supplied and interrupted snapshots, and a corpus poisoned this
way is subsequently SERVED by recall, which is what makes it worth confining.

Note on the fix chosen for copytree: ``symlinks=True`` is NOT sufficient. It
stops the copy dereferencing, and then plants an escaping symlink in the corpus
that the parser follows on read -- the same primitive, moved from copy time to
parse time. The entries are dropped instead.
"""

import os

import pytest

from mind_mem import block_store


@pytest.fixture
def secret(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    f = outside / "stolen.md"
    f.write_text("# RESTORE-CANARY\n", encoding="utf-8")
    return outside, f


class TestTheManifestIsReadFromInsideTheSnapshot:
    def test_a_symlinked_manifest_is_refused(self, tmp_path, secret):
        outside, _ = secret
        foreign = outside / "MANIFEST.json"
        foreign.write_text('{"files": ["pwned.md"], "version": 1}', encoding="utf-8")

        snap = tmp_path / "snap"
        snap.mkdir()
        try:
            os.symlink(str(foreign), str(snap / "MANIFEST.json"))
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")

        with pytest.raises(ValueError):
            block_store._read_manifest(str(snap))

    def test_a_real_manifest_still_reads(self, tmp_path):
        """Positive control: the refusal above is about the symlink."""
        snap = tmp_path / "snap"
        snap.mkdir()
        (snap / "MANIFEST.json").write_text('{"files": ["a.md"], "version": 1}', encoding="utf-8")
        got = block_store._read_manifest(str(snap))
        assert got is not None and got["files"] == ["a.md"]

    def test_an_absent_manifest_is_still_none(self, tmp_path):
        """Legacy snapshots have no manifest; that must stay distinguishable."""
        snap = tmp_path / "snap"
        snap.mkdir()
        assert block_store._read_manifest(str(snap)) is None

    @pytest.mark.parametrize(
        "payload",
        [
            "null",
            '{"files": null}',
            '{"files": [7]}',
            '{"files": [], "cleanup_inventory": []}',
            '{"files": [], "version": "two"}',
        ],
    )
    def test_malformed_manifest_shapes_are_refused(self, tmp_path, payload):
        snap = tmp_path / "snap"
        snap.mkdir()
        (snap / "MANIFEST.json").write_text(payload, encoding="utf-8")

        with pytest.raises(ValueError, match="snapshot manifest"):
            block_store._read_manifest(str(snap))


class TestCopytreeDoesNotCarrySymlinkedContent:
    def test_symlinked_entries_are_dropped_not_followed(self, tmp_path, secret):
        outside, stolen = secret
        src = tmp_path / "src"
        src.mkdir()
        (src / "real.md").write_text("# real\n", encoding="utf-8")
        try:
            os.symlink(str(stolen), str(src / "linked.md"))
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")

        dst = tmp_path / "dst"
        import shutil

        shutil.copytree(str(src), str(dst), ignore=block_store._drop_symlinks)

        assert (dst / "real.md").exists(), "positive control: real content must still copy"
        assert not (dst / "linked.md").exists(), "a symlinked entry was carried into the corpus"
        assert "RESTORE-CANARY" not in "".join(p.read_text(encoding="utf-8") for p in dst.rglob("*.md"))


class TestTheIntelligenceRootIsConfined:
    def test_a_symlinked_intelligence_root_is_not_followed(self, tmp_path, secret):
        outside, _ = secret
        snap = tmp_path / "snap"
        snap.mkdir()
        try:
            os.symlink(str(outside), str(snap / "intelligence"))
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")

        resolved = block_store._safe_intel_root(str(snap))
        assert resolved == "", "a symlinked intelligence root must not resolve"

    def test_a_real_intelligence_root_resolves(self, tmp_path):
        """Positive control."""
        snap = tmp_path / "snap"
        (snap / "intelligence").mkdir(parents=True)
        assert block_store._safe_intel_root(str(snap)).endswith("intelligence")
