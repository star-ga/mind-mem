# Copyright 2026 STARGA, Inc.
"""A proposal's FilesTouched must not name a path outside the workspace.

These are exploit tests, not style tests. Each one FAILS on the code as it was:
the sink is reached with proposal-supplied content, and the assertion is about
bytes actually read or written, not about a function returning cleanly.

The taint is real. ``validate_proposal`` rejects traversal only on ``op["file"]``
and only asserts that the ops' files are a SUBSET of FilesTouched -- so an EXTRA
FilesTouched entry naming anything at all is accepted, and it survives to the
diff and snapshot legs unnormalised unless it happens to be absolute.
"""

import os

import pytest

from mind_mem import apply_engine


class TestTheDiffLegDoesNotReadOutsideTheWorkspace:
    def test_a_traversing_files_touched_entry_reads_no_foreign_bytes(self, tmp_path):
        # ws and snap sit at DIFFERENT depths deliberately. At equal depths the
        # same relative path resolves to the same file on both sides, the diff
        # comes out empty, and the test passes while the read still happened --
        # a green result proving nothing.
        ws = tmp_path / "ws"
        snap = tmp_path / "a" / "b" / "snap"
        (ws / "memory").mkdir(parents=True)
        snap.mkdir(parents=True)

        secret = tmp_path / "outside" / "secret.txt"
        secret.parent.mkdir()
        secret.write_text("SUPERSECRET-CANARY\n", encoding="utf-8")

        # Positive control FIRST: a legitimate entry really does produce a diff,
        # so a later empty result means confinement and not a broken fixture.
        (snap / "note.md").write_text("before\n", encoding="utf-8")
        (ws / "note.md").write_text("after\n", encoding="utf-8")
        legit = apply_engine.generate_diff_text(str(ws), str(snap), ["note.md"])
        assert "before" in legit and "after" in legit, "fixture is broken; the control must diff"

        # The exploit: an extra FilesTouched entry pointing outside the workspace.
        rel = os.path.relpath(str(secret), str(snap))
        out = apply_engine.generate_diff_text(str(ws), str(snap), [rel])
        assert "SUPERSECRET-CANARY" not in out, "a proposal-supplied FilesTouched entry read a file outside the workspace"

    def test_a_symlinked_entry_reads_no_foreign_bytes(self, tmp_path):
        ws = tmp_path / "ws"
        snap = tmp_path / "snap"
        (ws / "memory").mkdir(parents=True)
        snap.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("SYMLINK-CANARY\n", encoding="utf-8")
        try:
            os.symlink(str(secret), str(snap / "innocent.md"))
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")

        out = apply_engine.generate_diff_text(str(ws), str(snap), ["innocent.md"])
        assert "SYMLINK-CANARY" not in out, "a symlinked snapshot entry was dereferenced and read"


class TestTheSnapshotLegDoesNotWriteOutsideTheSnapshot:
    def test_an_entry_that_resolves_inside_still_writes_inside(self, tmp_path):
        """The source check passes; the destination join is what escapes.

        A rel_path like ``../ws/memory/x.md`` REALPATHS back inside the
        workspace, so the ``startswith(ws_real)`` guard admits it -- and then the
        destination is built by joining that same raw string onto snap_dir,
        which lands outside. Source-validated is not destination-validated.
        """
        from mind_mem.block_store import MarkdownBlockStore

        ws = tmp_path / "ws"
        (ws / "memory").mkdir(parents=True)
        (ws / "memory" / "real.md").write_text("# real\n", encoding="utf-8")

        snap_dir = tmp_path / "snapdir"
        snap_dir.mkdir()
        victim = tmp_path / "snapdir_sibling"
        victim.mkdir()

        store = MarkdownBlockStore(str(ws))
        # Resolves to <ws>/memory/real.md (inside), but joins to
        # <snap_dir>/../snapdir_sibling/pwned.md (outside).
        rel = os.path.join("..", ws.name, "memory", "real.md")
        escaped_dst = snap_dir.parent / ws.name / "memory" / "real.md"

        # Positive control: a normal entry really is copied.
        store.snapshot(str(snap_dir), files_touched=["memory/real.md"])
        assert (snap_dir / "memory" / "real.md").exists(), "control failed; fixture is broken"

        before = escaped_dst.read_text(encoding="utf-8") if escaped_dst.exists() else None
        try:
            store.snapshot(str(snap_dir), files_touched=[rel])
        except ValueError:
            pass  # refusing loudly is an acceptable outcome
        after = escaped_dst.read_text(encoding="utf-8") if escaped_dst.exists() else None
        assert after == before, "the snapshot leg wrote through a destination outside snap_dir"

    def test_a_full_snapshot_does_not_follow_a_preexisting_destination_symlink(self, tmp_path):
        from mind_mem.block_store import MarkdownBlockStore

        ws = tmp_path / "ws"
        (ws / "decisions").mkdir(parents=True)
        (ws / "decisions" / "DECISIONS.md").write_text("# snapshot canary\n", encoding="utf-8")
        (ws / "AGENTS.md").write_text("positive control\n", encoding="utf-8")

        snap_dir = tmp_path / "snap"
        snap_dir.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        try:
            os.symlink(str(outside), str(snap_dir / "decisions"), target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")

        manifest = MarkdownBlockStore(str(ws)).snapshot(str(snap_dir))

        assert not (outside / "DECISIONS.md").exists(), "full snapshot followed a destination symlink"
        assert (snap_dir / "AGENTS.md").read_text(encoding="utf-8") == "positive control\n"
        assert "decisions/DECISIONS.md" not in manifest["files"]
