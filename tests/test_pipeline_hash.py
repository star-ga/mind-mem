"""Tests for v3.9 hash-of-code pipeline invalidation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mind_mem.pipeline_hash import (
    PipelineHashInputs,
    compute_pipeline_hash,
    current_pipeline_hash,
    pipeline_dirty_blocks,
    reextract_dirty_blocks,
    stamp_transform_hash,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(workspace: Path, config: dict) -> None:
    (workspace / "mind-mem.json").write_text(json.dumps(config))


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "memory").mkdir()
    (ws / "decisions").mkdir()
    return ws


# ---------------------------------------------------------------------------
# compute_pipeline_hash — pure function tests
# ---------------------------------------------------------------------------


class TestComputePipelineHash:
    def test_deterministic(self) -> None:
        i = PipelineHashInputs(
            package_version="3.9.0",
            backend="ollama",
            model="mind-mem:4b",
            extractor_source_sha256="a" * 64,
            prompt_template_sha256="b" * 64,
        )
        assert compute_pipeline_hash(i) == compute_pipeline_hash(i)

    def test_changes_on_version_bump(self) -> None:
        a = PipelineHashInputs("3.9.0", "ollama", "x", "h1", "h2")
        b = PipelineHashInputs("3.9.1", "ollama", "x", "h1", "h2")
        assert compute_pipeline_hash(a) != compute_pipeline_hash(b)

    def test_changes_on_backend_swap(self) -> None:
        a = PipelineHashInputs("3.9.0", "ollama", "x", "h1", "h2")
        b = PipelineHashInputs("3.9.0", "openai-compatible", "x", "h1", "h2")
        assert compute_pipeline_hash(a) != compute_pipeline_hash(b)

    def test_changes_on_model_swap(self) -> None:
        a = PipelineHashInputs("3.9.0", "ollama", "old-model", "h", "")
        b = PipelineHashInputs("3.9.0", "ollama", "new-model", "h", "")
        assert compute_pipeline_hash(a) != compute_pipeline_hash(b)

    def test_changes_on_extractor_source_change(self) -> None:
        a = PipelineHashInputs("3.9.0", "ollama", "x", "h1", "")
        b = PipelineHashInputs("3.9.0", "ollama", "x", "h2", "")
        assert compute_pipeline_hash(a) != compute_pipeline_hash(b)

    def test_changes_on_prompt_template_change(self) -> None:
        a = PipelineHashInputs("3.9.0", "ollama", "x", "h", "old-template")
        b = PipelineHashInputs("3.9.0", "ollama", "x", "h", "new-template")
        assert compute_pipeline_hash(a) != compute_pipeline_hash(b)

    def test_nul_separation_resists_concat_collision(self) -> None:
        # If we joined with empty string, ('a','b','c','d','e') and
        # ('ab','c','d','e') would collide. NUL-separator prevents that.
        a = PipelineHashInputs("3.9.0a", "b", "c", "d", "e")
        b = PipelineHashInputs("3.9.0", "ab", "c", "d", "e")
        assert compute_pipeline_hash(a) != compute_pipeline_hash(b)

    def test_output_shape_is_hex_sha256(self) -> None:
        digest = compute_pipeline_hash(PipelineHashInputs("v", "b", "m", "x", "y"))
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


# ---------------------------------------------------------------------------
# current_pipeline_hash — exercises config loading + file hashing
# ---------------------------------------------------------------------------


class TestCurrentPipelineHash:
    def test_no_config_uses_defaults(self, workspace: Path) -> None:
        # No mind-mem.json — should still produce a stable hash.
        h = current_pipeline_hash(str(workspace))
        assert isinstance(h, str)
        assert len(h) == 64

    def test_returns_inputs_when_requested(self, workspace: Path) -> None:
        result = current_pipeline_hash(str(workspace), return_inputs=True)
        assert isinstance(result, tuple)
        digest, inputs = result
        assert isinstance(digest, str)
        assert inputs.backend == "ollama"  # default
        assert inputs.model == "default"

    def test_unknown_backend_distinct_hash(self, workspace: Path, tmp_path: Path) -> None:
        _write_config(workspace, {"extraction": {"backend": "made-up-backend", "model": "x"}})
        h_unknown = current_pipeline_hash(str(workspace))
        # Default backend should differ
        _write_config(workspace, {"extraction": {"backend": "ollama", "model": "x"}})
        h_default = current_pipeline_hash(str(workspace))
        assert h_unknown != h_default

    def test_prompt_template_path_changes_hash(self, workspace: Path) -> None:
        tpl1 = workspace / "tpl1.txt"
        tpl1.write_text("one")
        tpl2 = workspace / "tpl2.txt"
        tpl2.write_text("two")

        _write_config(workspace, {"extraction": {"backend": "ollama", "model": "x", "prompt_template": str(tpl1)}})
        h1 = current_pipeline_hash(str(workspace))
        _write_config(workspace, {"extraction": {"backend": "ollama", "model": "x", "prompt_template": str(tpl2)}})
        h2 = current_pipeline_hash(str(workspace))
        assert h1 != h2

    def test_changing_prompt_template_contents_changes_hash(self, workspace: Path) -> None:
        tpl = workspace / "tpl.txt"
        tpl.write_text("original")
        _write_config(workspace, {"extraction": {"backend": "ollama", "model": "x", "prompt_template": str(tpl)}})
        h1 = current_pipeline_hash(str(workspace))
        # Mutate the template; hash must change.
        tpl.write_text("rewritten")
        h2 = current_pipeline_hash(str(workspace))
        assert h1 != h2

    def test_malformed_config_falls_back_to_defaults(self, workspace: Path) -> None:
        (workspace / "mind-mem.json").write_text("{ not json }")
        h = current_pipeline_hash(str(workspace))
        assert isinstance(h, str) and len(h) == 64


# ---------------------------------------------------------------------------
# pipeline_dirty_blocks — workspace integration
# ---------------------------------------------------------------------------


class TestPipelineDirtyBlocks:
    def test_empty_workspace_returns_empty(self, workspace: Path) -> None:
        _write_config(
            workspace,
            {"version": "3.9.0", "block_store": {"backend": "markdown"}},
        )
        assert pipeline_dirty_blocks(str(workspace)) == []

    def test_blocks_without_transform_hash_are_dirty(self, workspace: Path) -> None:
        _write_config(
            workspace,
            {"version": "3.9.0", "block_store": {"backend": "markdown"}},
        )
        (workspace / "decisions" / "DECISIONS.md").write_text("[D-20260503-001]\nStatus: active\nStatement: pre-v3.9 block\n")
        dirty = pipeline_dirty_blocks(str(workspace))
        assert "D-20260503-001" in dirty

    def test_blocks_with_matching_hash_are_clean(self, workspace: Path) -> None:
        _write_config(
            workspace,
            {"version": "3.9.0", "block_store": {"backend": "markdown"}},
        )
        current = current_pipeline_hash(str(workspace))
        assert isinstance(current, str)
        (workspace / "decisions" / "DECISIONS.md").write_text(
            f"[D-20260503-002]\nStatus: active\nStatement: post-v3.9 block\nTransformHash: {current}\n"
        )
        dirty = pipeline_dirty_blocks(str(workspace))
        assert "D-20260503-002" not in dirty

    def test_blocks_with_old_hash_are_dirty(self, workspace: Path) -> None:
        _write_config(
            workspace,
            {"version": "3.9.0", "block_store": {"backend": "markdown"}},
        )
        old_hash = hashlib.sha256(b"old pipeline").hexdigest()
        (workspace / "decisions" / "DECISIONS.md").write_text(
            f"[D-20260503-003]\nStatus: active\nStatement: stale block\nTransformHash: {old_hash}\n"
        )
        dirty = pipeline_dirty_blocks(str(workspace))
        assert "D-20260503-003" in dirty


# ---------------------------------------------------------------------------
# v3.10: stamp_transform_hash + reextract_dirty_blocks
# ---------------------------------------------------------------------------


class TestStampTransformHash:
    def test_returns_copy_with_hash(self, workspace: Path) -> None:
        _write_config(workspace, {"version": "3.10.0", "block_store": {"backend": "markdown"}})
        block = {"_id": "D-20260503-100", "Statement": "test"}
        stamped = stamp_transform_hash(str(workspace), block)
        assert "TransformHash" in stamped
        assert isinstance(stamped["TransformHash"], str)
        assert len(stamped["TransformHash"]) == 64  # hex sha256

    def test_does_not_mutate_input(self, workspace: Path) -> None:
        _write_config(workspace, {"version": "3.10.0", "block_store": {"backend": "markdown"}})
        block = {"_id": "D-20260503-101", "Statement": "test"}
        original = dict(block)
        stamp_transform_hash(str(workspace), block)
        assert block == original

    def test_overwrites_existing_hash(self, workspace: Path) -> None:
        _write_config(workspace, {"version": "3.10.0", "block_store": {"backend": "markdown"}})
        block = {"_id": "D-20260503-102", "TransformHash": "stale"}
        stamped = stamp_transform_hash(str(workspace), block)
        assert stamped["TransformHash"] != "stale"

    def test_drops_lowercase_form(self, workspace: Path) -> None:
        _write_config(workspace, {"version": "3.10.0", "block_store": {"backend": "markdown"}})
        block = {"_id": "D-20260503-103", "transform_hash": "old-lower"}
        stamped = stamp_transform_hash(str(workspace), block)
        assert "transform_hash" not in stamped
        assert "TransformHash" in stamped

    def test_rejects_non_dict(self, workspace: Path) -> None:
        _write_config(workspace, {"version": "3.10.0", "block_store": {"backend": "markdown"}})
        with pytest.raises(TypeError, match="dict"):
            stamp_transform_hash(str(workspace), "not a block")  # type: ignore[arg-type]

    def test_hash_matches_current_pipeline(self, workspace: Path) -> None:
        _write_config(workspace, {"version": "3.10.0", "block_store": {"backend": "markdown"}})
        block = {"_id": "D-20260503-104"}
        stamped = stamp_transform_hash(str(workspace), block)
        assert stamped["TransformHash"] == current_pipeline_hash(str(workspace))


class TestReextractDirtyBlocks:
    def test_dry_run_lists_without_writing(self, workspace: Path) -> None:
        _write_config(workspace, {"version": "3.10.0", "block_store": {"backend": "markdown"}})
        (workspace / "decisions" / "DECISIONS.md").write_text("[D-20260503-200]\nStatus: active\nStatement: stale\n")
        result = reextract_dirty_blocks(str(workspace), dry_run=True)
        assert result["dry_run"] is True
        assert result["processed"] == 0
        assert "D-20260503-200" in result["ids"]

    def test_processes_dirty_blocks(self, workspace: Path) -> None:
        _write_config(workspace, {"version": "3.10.0", "block_store": {"backend": "markdown"}})
        (workspace / "decisions" / "DECISIONS.md").write_text("[D-20260503-201]\nStatus: active\nStatement: stale\n")
        result = reextract_dirty_blocks(str(workspace))
        assert result["dry_run"] is False
        assert result["processed"] == 1
        # After processing, the block should now carry the current hash.
        text = (workspace / "decisions" / "DECISIONS.md").read_text()
        assert "TransformHash:" in text

    def test_after_reextract_no_dirty(self, workspace: Path) -> None:
        _write_config(workspace, {"version": "3.10.0", "block_store": {"backend": "markdown"}})
        (workspace / "decisions" / "DECISIONS.md").write_text("[D-20260503-202]\nStatus: active\nStatement: stale\n")
        reextract_dirty_blocks(str(workspace))
        # Re-check: should now be clean.
        dirty = pipeline_dirty_blocks(str(workspace))
        assert "D-20260503-202" not in dirty

    def test_limit_caps_work(self, workspace: Path) -> None:
        _write_config(workspace, {"version": "3.10.0", "block_store": {"backend": "markdown"}})
        (workspace / "decisions" / "DECISIONS.md").write_text(
            "[D-20260503-300]\nStatus: active\nStatement: a\n\n"
            "[D-20260503-301]\nStatus: active\nStatement: b\n\n"
            "[D-20260503-302]\nStatus: active\nStatement: c\n"
        )
        result = reextract_dirty_blocks(str(workspace), limit=2, dry_run=True)
        assert len(result["ids"]) == 2

    def test_invalid_limit_rejected(self, workspace: Path) -> None:
        _write_config(workspace, {"version": "3.10.0", "block_store": {"backend": "markdown"}})
        with pytest.raises(ValueError, match="limit"):
            reextract_dirty_blocks(str(workspace), limit=0)

    def test_empty_workspace_clean(self, workspace: Path) -> None:
        _write_config(workspace, {"version": "3.10.0", "block_store": {"backend": "markdown"}})
        result = reextract_dirty_blocks(str(workspace))
        assert result["processed"] == 0
        assert result["skipped"] == 0
