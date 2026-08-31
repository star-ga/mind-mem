"""Text-range ops (``insert_after_block`` / ``replace_range``) must be atomic.

Both ops used to build their new file content in memory and then commit it
with a raw ``open(filepath, "w")`` + ``writelines``. That call truncates the
destination the instant it opens it, so *any* failure between the truncate and
the final flush — a bad byte in the patch, a full disk, a killed process —
leaves the corpus file short or empty with no way back. Every other op in
``apply_engine`` commits through the ``FileLock`` + ``_atomic_write``
(temp file + ``os.replace``) path, where the destination is not touched until
a complete new file exists on the same filesystem.

The failure injected here is a real one rather than a mock: a lone surrogate
(``chr(0xDC80)``) in the patch text is unencodable by every strict codec, so the
encode raises partway through writing the new content. Under the raw-``open``
implementation the destination is already truncated at that point; under the
atomic implementation the destination has not been opened at all.

The traversal tests pin the ``_safe_resolve`` choke point in ``execute_op``,
which is the only entry point that reaches these two ops.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mind_mem.apply_engine import _op_insert_after_block, _op_replace_range, execute_op

#: Unencodable by any strict codec — the fault we inject mid-write.
# Built at runtime, NOT written as a source literal: a lone surrogate in
# module source makes pytest's assertion-rewriting compile() raise
# UnicodeEncodeError, which kills COLLECTION of the entire file before a
# single test runs. chr() keeps the injected failure exactly as real.
LONE_SURROGATE = chr(0xDC80)

TARGET = "D-20260213-001"

INSERT_FIXTURE = f"[{TARGET}]\nStatus: active\nStatement: original\n\n[D-20260213-002]\nStatus: active\n"

RANGE_FIXTURE = f"[{TARGET}]\nStatus: active\n<!-- START -->\noriginal body\n<!-- END -->\nTags: pinned\n"


def _write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return text


def _stray_temp_files(directory: Path, keep: str) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name != keep)


class TestTextRangeOpsAreAtomic:
    """A failure partway through the write must leave the original intact."""

    def test_insert_after_block_does_not_truncate_on_write_failure(self, tmp_path: Path) -> None:
        target_file = tmp_path / "DECISIONS.md"
        original = _write(target_file, INSERT_FIXTURE)

        with pytest.raises(UnicodeEncodeError):
            _op_insert_after_block(
                str(target_file),
                {"target": TARGET, "patch": f"[D-20260213-009]\nStatus: active\nStatement: {LONE_SURROGATE}"},
            )

        assert target_file.read_text(encoding="utf-8") == original, (
            "insert_after_block truncated the corpus file when the write failed partway"
        )

    def test_replace_range_does_not_truncate_on_write_failure(self, tmp_path: Path) -> None:
        target_file = tmp_path / "DECISIONS.md"
        original = _write(target_file, RANGE_FIXTURE)

        with pytest.raises(UnicodeEncodeError):
            _op_replace_range(
                str(target_file),
                {
                    "target": TARGET,
                    "range": {"start": "<!-- START -->", "end": "<!-- END -->"},
                    "patch": f"<!-- START -->\nnew body {LONE_SURROGATE}",
                },
            )

        assert target_file.read_text(encoding="utf-8") == original, "replace_range truncated the corpus file when the write failed partway"

    def test_insert_after_block_leaves_no_temp_file_behind(self, tmp_path: Path) -> None:
        target_file = tmp_path / "DECISIONS.md"
        _write(target_file, INSERT_FIXTURE)

        with pytest.raises(UnicodeEncodeError):
            _op_insert_after_block(
                str(target_file),
                {"target": TARGET, "patch": f"[D-20260213-009]\nStatus: active\nStatement: {LONE_SURROGATE}"},
            )

        assert _stray_temp_files(tmp_path, "DECISIONS.md") == [], "aborted insert_after_block left a temp file behind"

    def test_replace_range_leaves_no_temp_file_behind(self, tmp_path: Path) -> None:
        target_file = tmp_path / "DECISIONS.md"
        _write(target_file, RANGE_FIXTURE)

        with pytest.raises(UnicodeEncodeError):
            _op_replace_range(
                str(target_file),
                {
                    "target": TARGET,
                    "range": {"start": "<!-- START -->", "end": "<!-- END -->"},
                    "patch": f"<!-- START -->\nnew body {LONE_SURROGATE}",
                },
            )

        assert _stray_temp_files(tmp_path, "DECISIONS.md") == [], "aborted replace_range left a temp file behind"

    def test_successful_ops_still_commit_their_content(self, tmp_path: Path) -> None:
        """The atomic path must not become a no-op — the happy path still writes."""
        insert_file = tmp_path / "DECISIONS.md"
        _write(insert_file, INSERT_FIXTURE)
        ok, msg = _op_insert_after_block(str(insert_file), {"target": TARGET, "patch": "[D-20260213-009]\nStatus: active"})
        assert ok, msg
        assert "D-20260213-009" in insert_file.read_text(encoding="utf-8")

        range_file = tmp_path / "TASKS.md"
        _write(range_file, RANGE_FIXTURE)
        ok, msg = _op_replace_range(
            str(range_file),
            {
                "target": TARGET,
                "range": {"start": "<!-- START -->", "end": "<!-- END -->"},
                "patch": "<!-- START -->\nreplacement body",
            },
        )
        assert ok, msg
        body = range_file.read_text(encoding="utf-8")
        assert "replacement body" in body
        assert "original body" not in body
        assert "<!-- END -->" in body


class TestTextRangeOpsRejectTraversal:
    """``_safe_resolve`` must gate these two ops exactly as it gates the others."""

    @staticmethod
    def _outside(tmp_path: Path) -> tuple[Path, Path, str]:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        victim = tmp_path / "outside.md"
        return workspace, victim, _write(victim, RANGE_FIXTURE)

    @pytest.mark.parametrize(
        "op",
        [
            {"op": "insert_after_block", "target": TARGET, "patch": "[D-20260213-009]\nStatus: active"},
            {
                "op": "replace_range",
                "target": TARGET,
                "range": {"start": "<!-- START -->", "end": "<!-- END -->"},
                "patch": "<!-- START -->\npwned",
            },
        ],
        ids=["insert_after_block", "replace_range"],
    )
    def test_dotdot_path_is_refused(self, tmp_path: Path, op: dict) -> None:
        workspace, victim, original = self._outside(tmp_path)

        ok, msg = execute_op(str(workspace), {**op, "file": "../outside.md"})

        assert ok is False
        assert "SECURITY" in msg
        assert victim.read_text(encoding="utf-8") == original

    @pytest.mark.parametrize(
        "op",
        [
            {"op": "insert_after_block", "target": TARGET, "patch": "[D-20260213-009]\nStatus: active"},
            {
                "op": "replace_range",
                "target": TARGET,
                "range": {"start": "<!-- START -->", "end": "<!-- END -->"},
                "patch": "<!-- START -->\npwned",
            },
        ],
        ids=["insert_after_block", "replace_range"],
    )
    def test_symlink_escape_is_refused(self, tmp_path: Path, op: dict) -> None:
        workspace, victim, original = self._outside(tmp_path)
        link = workspace / "DECISIONS.md"
        try:
            os.symlink(victim, link)
        except (OSError, NotImplementedError) as exc:  # pragma: no cover — unprivileged Windows
            pytest.skip(f"symlink creation unavailable: {exc}")

        ok, msg = execute_op(str(workspace), {**op, "file": "DECISIONS.md"})

        assert ok is False
        assert "SECURITY" in msg
        assert victim.read_text(encoding="utf-8") == original
