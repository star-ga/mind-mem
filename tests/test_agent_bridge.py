# Copyright 2026 STARGA, Inc.
"""Tests for the universal agent bridge + vault sync (v2.7.0)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mind_mem.agent_bridge import (
    KNOWN_AGENTS,
    AgentFormatter,
    UnknownAgentError,
    VaultBlock,
    VaultBridge,
)

# ---------------------------------------------------------------------------
# AgentFormatter
# ---------------------------------------------------------------------------


@pytest.fixture()
def fmt() -> AgentFormatter:
    return AgentFormatter(max_blocks=5)


@pytest.fixture()
def sample_blocks() -> list[dict]:
    return [
        {"_id": "D-001", "type": "decision", "file": "decisions/auth.md", "excerpt": "Use OAuth2"},
        {"_id": "T-002", "type": "task", "file": "tasks/jwt.md", "excerpt": "Rotate JWT keys"},
    ]


class TestAgentFormatterDispatch:
    @pytest.mark.parametrize("agent", KNOWN_AGENTS)
    def test_known_agents_render_without_error(self, fmt: AgentFormatter, sample_blocks: list[dict], agent: str) -> None:
        out = fmt.inject(agent, "JWT auth", sample_blocks)
        assert isinstance(out, str)
        assert out.strip()

    def test_unknown_agent_raises(self, fmt: AgentFormatter) -> None:
        with pytest.raises(UnknownAgentError):
            fmt.inject("not-an-agent", "q", [])

    def test_max_blocks_caps_output(self, fmt: AgentFormatter) -> None:
        many = [{"_id": f"B-{i}", "type": "note", "excerpt": f"text {i}"} for i in range(20)]
        out = fmt.inject("generic", "q", many)
        # Only the first max_blocks=5 should appear.
        assert "[B-0]" in out
        assert "[B-4]" in out
        assert "[B-5]" not in out


class TestAgentFormatterContent:
    def test_claude_renders_markdown_headings(self, fmt: AgentFormatter, sample_blocks: list[dict]) -> None:
        out = fmt.inject("claude-code", "q", sample_blocks)
        assert "# mind-mem context" in out
        assert "## decision — D-001" in out

    def test_codex_uses_bullet_list(self, fmt: AgentFormatter, sample_blocks: list[dict]) -> None:
        out = fmt.inject("codex", "auth", sample_blocks)
        assert "Context for: auth" in out
        assert "**decision** [D-001]: Use OAuth2" in out

    def test_gemini_uses_system_tag(self, fmt: AgentFormatter, sample_blocks: list[dict]) -> None:
        out = fmt.inject("gemini", "auth", sample_blocks)
        assert "system:" in out
        assert "[D-001]" in out

    def test_cursor_uses_workspace_memory_header(self, fmt: AgentFormatter, sample_blocks: list[dict]) -> None:
        out = fmt.inject("cursor", "auth", sample_blocks)
        assert "Workspace memory" in out

    def test_aider_uses_yaml_repo_map(self, fmt: AgentFormatter, sample_blocks: list[dict]) -> None:
        out = fmt.inject("aider", "auth", sample_blocks)
        assert out.startswith("repo_map:")
        assert "id: D-001" in out

    def test_block_without_text_renders_no_excerpt_marker(self, fmt: AgentFormatter) -> None:
        out = fmt.inject("claude-code", "q", [{"_id": "X", "type": "note"}])
        assert "_(no excerpt)_" in out

    def test_handles_alternate_id_fields(self, fmt: AgentFormatter) -> None:
        # `id` and `block_id` are both accepted in addition to `_id`.
        out = fmt.inject("generic", "q", [{"id": "Z-1", "excerpt": "x"}])
        assert "[Z-1]" in out
        out = fmt.inject("generic", "q", [{"block_id": "Z-2", "excerpt": "x"}])
        assert "[Z-2]" in out


# ---------------------------------------------------------------------------
# VaultBridge.scan
# ---------------------------------------------------------------------------


@pytest.fixture()
def vault():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        root = Path(td)
        (root / "decisions").mkdir()
        (root / "tasks").mkdir()
        (root / ".obsidian").mkdir()
        (root / "decisions" / "auth.md").write_text(
            "---\nid: D-AUTH\ntype: decision\ntitle: Auth flow\n---\n\nUse OAuth2 with PKCE.\n",
            encoding="utf-8",
        )
        (root / "tasks" / "jwt.md").write_text(
            "---\nid: T-JWT\ntype: task\n---\n\nRotate signing keys.\n",
            encoding="utf-8",
        )
        # Excluded directory entry that should be skipped.
        (root / ".obsidian" / "core.md").write_text("# noise", encoding="utf-8")
        # File without frontmatter — frontmatter dict empty, body kept.
        (root / "README.md").write_text("plain text\nno frontmatter\n", encoding="utf-8")
        yield root


class TestVaultScan:
    def test_scan_returns_blocks_for_each_md(self, vault: Path) -> None:
        bridge = VaultBridge(vault_root=str(vault))
        blocks = bridge.scan()
        ids = {b.block_id for b in blocks}
        assert "D-AUTH" in ids
        assert "T-JWT" in ids

    def test_excluded_directory_skipped(self, vault: Path) -> None:
        bridge = VaultBridge(vault_root=str(vault))
        for b in bridge.scan():
            assert ".obsidian" not in b.relative_path

    def test_files_without_frontmatter_keep_body(self, vault: Path) -> None:
        bridge = VaultBridge(vault_root=str(vault))
        blocks = bridge.scan()
        readme = next(b for b in blocks if b.relative_path == "README.md")
        assert readme.frontmatter == {}
        assert "plain text" in readme.body

    def test_sync_dirs_filter(self, vault: Path) -> None:
        bridge = VaultBridge(vault_root=str(vault))
        blocks = bridge.scan(sync_dirs=["decisions"])
        assert all(b.relative_path.startswith("decisions") for b in blocks)

    def test_sync_dir_escape_rejected(self, vault: Path) -> None:
        bridge = VaultBridge(vault_root=str(vault))
        with pytest.raises(ValueError, match="escapes vault root"):
            bridge.scan(sync_dirs=["../"])

    def test_missing_vault_root_raises(self) -> None:
        bridge = VaultBridge(vault_root="/no/such/path/4f5b9d")
        with pytest.raises(FileNotFoundError):
            bridge.scan()


# ---------------------------------------------------------------------------
# VaultBridge.write
# ---------------------------------------------------------------------------


class TestVaultWrite:
    def test_write_creates_file_with_frontmatter(self, vault: Path) -> None:
        bridge = VaultBridge(vault_root=str(vault))
        block = VaultBlock(
            relative_path="entities/Alice.md",
            block_id="E-ALICE",
            block_type="entity",
            title="Alice",
            body="Engineer at STARGA.",
            frontmatter={},
        )
        path = bridge.write(block)
        text = Path(path).read_text(encoding="utf-8")
        assert text.startswith("---")
        assert "id: E-ALICE" in text
        assert "Engineer at STARGA." in text

    def test_write_refuses_overwrite_by_default(self, vault: Path) -> None:
        bridge = VaultBridge(vault_root=str(vault))
        block = VaultBlock(
            relative_path="decisions/auth.md",  # already exists
            block_id="D-AUTH",
            block_type="decision",
            title="Auth flow",
            body="X",
        )
        with pytest.raises(FileExistsError):
            bridge.write(block)

    def test_write_overwrite_flag_replaces(self, vault: Path) -> None:
        bridge = VaultBridge(vault_root=str(vault))
        block = VaultBlock(
            relative_path="decisions/auth.md",
            block_id="D-AUTH",
            block_type="decision",
            title="Auth flow",
            body="REPLACED",
        )
        bridge.write(block, overwrite=True)
        text = (vault / "decisions" / "auth.md").read_text(encoding="utf-8")
        assert "REPLACED" in text

    def test_write_path_escape_rejected(self, vault: Path) -> None:
        bridge = VaultBridge(vault_root=str(vault))
        with pytest.raises(ValueError, match="escapes vault root"):
            bridge.write(
                VaultBlock(
                    relative_path="../escape.md",
                    block_id="X",
                    block_type="x",
                    title="x",
                    body="x",
                )
            )

    def test_write_empty_relative_path_rejected(self, vault: Path) -> None:
        bridge = VaultBridge(vault_root=str(vault))
        with pytest.raises(ValueError, match="non-empty relative path"):
            bridge.write(
                VaultBlock(
                    relative_path="",
                    block_id="X",
                    block_type="x",
                    title="x",
                    body="x",
                )
            )

    def test_round_trip_preserves_id_and_body(self, vault: Path) -> None:
        bridge = VaultBridge(vault_root=str(vault))
        block = VaultBlock(
            relative_path="entities/Round.md",
            block_id="E-ROUND",
            block_type="entity",
            title="Round Trip",
            body="Body line one.\nBody line two.",
        )
        bridge.write(block)
        scanned = bridge.scan(sync_dirs=["entities"])
        match = next(b for b in scanned if b.relative_path == "entities/Round.md")
        assert match.block_id == "E-ROUND"
        assert "Body line one" in match.body
