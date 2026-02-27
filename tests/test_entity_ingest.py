#!/usr/bin/env python3
"""Tests for the entity_ingest module — extraction, filtering, signal generation."""

from mind_mem.entity_ingest import (
    entities_to_signals,
    extract_entities,
    filter_new_entities,
    load_existing_entities,
)

# ---------------------------------------------------------------------------
# load_existing_entities
# ---------------------------------------------------------------------------


class TestLoadExistingEntities:
    """Test loading entity IDs from entities/*.md files."""

    def test_empty_workspace(self, tmp_path):
        """Workspace with no entities/ dir returns empty sets."""
        result = load_existing_entities(str(tmp_path))
        assert result == {"projects": set(), "tools": set(), "people": set()}

    def test_empty_entities_dir(self, tmp_path):
        """entities/ dir exists but is empty."""
        (tmp_path / "entities").mkdir()
        result = load_existing_entities(str(tmp_path))
        assert result == {"projects": set(), "tools": set(), "people": set()}

    def test_project_entity(self, tmp_path):
        """Extracts PRJ- slugs from entity files."""
        edir = tmp_path / "entities"
        edir.mkdir()
        (edir / "projects.md").write_text("[PRJ-mind-mem] mind-mem project\n")
        result = load_existing_entities(str(tmp_path))
        assert "mind-mem" in result["projects"]

    def test_tool_entity(self, tmp_path):
        """Extracts TOOL- slugs from entity files."""
        edir = tmp_path / "entities"
        edir.mkdir()
        (edir / "tools.md").write_text("[TOOL-docker] container tool\n")
        result = load_existing_entities(str(tmp_path))
        assert "docker" in result["tools"]

    def test_person_entity(self, tmp_path):
        """Extracts PER- slugs from entity files."""
        edir = tmp_path / "entities"
        edir.mkdir()
        (edir / "people.md").write_text("[PER-alice] Alice Smith\n")
        result = load_existing_entities(str(tmp_path))
        assert "alice" in result["people"]

    def test_multiple_types_in_one_file(self, tmp_path):
        """Multiple entity types in a single file are all extracted."""
        edir = tmp_path / "entities"
        edir.mkdir()
        content = "[PRJ-foo] Foo project\n[TOOL-bar] Bar tool\n[PER-baz] Baz person\n"
        (edir / "mixed.md").write_text(content)
        result = load_existing_entities(str(tmp_path))
        assert "foo" in result["projects"]
        assert "bar" in result["tools"]
        assert "baz" in result["people"]

    def test_name_field_extraction(self, tmp_path):
        """Name: fields are added to the appropriate set based on filename."""
        edir = tmp_path / "entities"
        edir.mkdir()
        (edir / "project-index.md").write_text("Name: My Cool Project\n")
        result = load_existing_entities(str(tmp_path))
        assert "my-cool-project" in result["projects"]

    def test_ignores_non_md_files(self, tmp_path):
        """Non-.md files in entities/ are skipped."""
        edir = tmp_path / "entities"
        edir.mkdir()
        (edir / "data.json").write_text('{"PRJ-hidden": true}')
        result = load_existing_entities(str(tmp_path))
        assert result == {"projects": set(), "tools": set(), "people": set()}

    def test_malformed_entity_file(self, tmp_path):
        """File with no valid entity markers returns empty sets."""
        edir = tmp_path / "entities"
        edir.mkdir()
        (edir / "broken.md").write_text(
            "This file has no [brackets or valid entity refs.\nRandom text, PRJ without brackets, [BADPREFIX-foo]\n"
        )
        result = load_existing_entities(str(tmp_path))
        assert result == {"projects": set(), "tools": set(), "people": set()}


# ---------------------------------------------------------------------------
# extract_entities
# ---------------------------------------------------------------------------


class TestExtractEntities:
    """Test regex-based entity extraction."""

    def test_github_url(self):
        """Extracts repo slug from GitHub URLs."""
        entities = extract_entities("See https://github.com/star-ga/mind-mem for details.")
        slugs = [e["slug"] for e in entities]
        assert "mind-mem" in slugs
        match = next(e for e in entities if e["slug"] == "mind-mem")
        assert match["entity_type"] == "project"
        assert match["source_pattern"] == "github_repo"

    def test_github_url_strips_trailing_dot(self):
        """Trailing dots from URLs are stripped from repo names."""
        entities = extract_entities("Repo: https://github.com/user/repo.name.")
        slugs = [e["slug"] for e in entities]
        assert "repo.name" in slugs

    def test_local_path(self):
        """Extracts project name from local /home/... paths."""
        entities = extract_entities("Located at /home/user/myproject/ for dev.")
        slugs = [e["slug"] for e in entities]
        assert "myproject" in slugs
        match = next(e for e in entities if e["slug"] == "myproject")
        assert match["source_pattern"] == "local_project"

    def test_local_path_ignores_system_dirs(self):
        """System directories from IGNORE_DIRS are filtered out."""
        for dirname in ["bin", "lib", "node_modules", ".git", ".cache"]:
            entities = extract_entities(f"/home/user/{dirname}/something")
            slugs = [e["slug"] for e in entities if e["source_pattern"] == "local_project"]
            assert dirname not in slugs, f"{dirname} should be ignored"

    def test_local_path_ignores_short_names(self):
        """Path segments < 3 chars are filtered out."""
        entities = extract_entities("/home/user/ab/something")
        slugs = [e["slug"] for e in entities if e["source_pattern"] == "local_project"]
        assert "ab" not in slugs

    def test_prj_reference(self):
        """Explicit PRJ- references are detected."""
        entities = extract_entities("Related to PRJ-cognet for AI work.")
        slugs = [e["slug"] for e in entities]
        assert "cognet" in slugs

    def test_mcp_server_reference(self):
        """MCP server references are extracted as tools."""
        entities = extract_entities("Start mcp-server-memory to enable recall.")
        tools = [e for e in entities if e["entity_type"] == "tool"]
        slugs = [e["slug"] for e in tools]
        assert "memory" in slugs
        assert tools[0]["source_pattern"] == "mcp_server"

    def test_mcp_underscore_variant(self):
        """mcp_server_ prefix variant is also detected."""
        entities = extract_entities("Uses mcp_server_search for queries.")
        tools = [e for e in entities if e["entity_type"] == "tool"]
        slugs = [e["slug"] for e in tools]
        assert "search" in slugs

    def test_cli_tools(self):
        """Known CLI tool names are detected."""
        entities = extract_entities("Run docker build then npm install.")
        slugs = [e["slug"] for e in entities if e["entity_type"] == "tool"]
        assert "docker" in slugs
        assert "npm" in slugs

    def test_tool_ignore_list(self):
        """Tools in TOOL_IGNORE are filtered out."""
        # "make" is in TOOL_IGNORE
        entities = extract_entities("Run make to build the project.")
        slugs = [e["slug"] for e in entities if e["entity_type"] == "tool"]
        assert "make" not in slugs

    def test_tool_ref(self):
        """Explicit TOOL- references are detected."""
        entities = extract_entities("See TOOL-codex-cli for info.")
        slugs = [e["slug"] for e in entities]
        assert "codex-cli" in slugs

    def test_at_mention(self):
        """@handle mentions are detected as people."""
        entities = extract_entities("Thanks @alice-dev for the review.")
        people = [e for e in entities if e["entity_type"] == "person"]
        slugs = [e["slug"] for e in people]
        assert "alice-dev" in slugs

    def test_per_reference(self):
        """Explicit PER- references are detected."""
        entities = extract_entities("Contact PER-bob for details.")
        people = [e for e in entities if e["entity_type"] == "person"]
        slugs = [e["slug"] for e in people]
        assert "bob" in slugs

    def test_at_mention_filters_common_words(self):
        """Common non-person @mentions are filtered."""
        entities = extract_entities("Notify @everyone and @bot about the change.")
        slugs = [e["slug"] for e in entities if e["entity_type"] == "person"]
        assert "everyone" not in slugs
        assert "bot" not in slugs

    def test_dedup_across_patterns(self):
        """Same slug appearing multiple times is only returned once."""
        text = "Check https://github.com/star-ga/mind-mem and also the /home/user/mind-mem directory."
        entities = extract_entities(text)
        slugs = [e["slug"] for e in entities]
        assert slugs.count("mind-mem") == 1

    def test_empty_text(self):
        """Empty string yields no entities."""
        assert extract_entities("") == []

    def test_unicode_text(self):
        """Unicode content does not crash extraction."""
        entities = extract_entities("Project at /home/user/caf\u00e9-app works fine \u2014 @bob-\u00fc")
        # Should at least extract the path; exact behavior on unicode handles is secondary
        assert isinstance(entities, list)

    def test_very_long_text(self):
        """Large input is handled without error and excerpts are bounded."""
        text = "prefix " * 5000 + "https://github.com/org/big-repo" + " suffix" * 5000
        entities = extract_entities(text)
        assert any(e["slug"] == "big-repo" for e in entities)
        for e in entities:
            # Excerpts should be bounded (30 before + match + 50 after)
            assert len(e["excerpt"]) < 300

    def test_excerpt_context(self):
        """Excerpt contains surrounding context around the match."""
        entities = extract_entities("Check out https://github.com/star-ga/mind-mem for details.")
        match = next(e for e in entities if e["slug"] == "mind-mem")
        assert "https://github.com/star-ga/mind-mem" in match["excerpt"]


# ---------------------------------------------------------------------------
# filter_new_entities
# ---------------------------------------------------------------------------


class TestFilterNewEntities:
    """Test filtering extracted entities against existing registry."""

    def test_all_new(self):
        """All entities pass when registry is empty."""
        entities = [
            {"entity_type": "project", "slug": "foo", "source_pattern": "github_repo", "excerpt": "x"},
            {"entity_type": "tool", "slug": "bar", "source_pattern": "cli_tool", "excerpt": "x"},
        ]
        existing = {"projects": set(), "tools": set(), "people": set()}
        result = filter_new_entities(entities, existing)
        assert len(result) == 2

    def test_all_existing(self):
        """All entities are filtered when already tracked."""
        entities = [
            {"entity_type": "project", "slug": "foo", "source_pattern": "github_repo", "excerpt": "x"},
            {"entity_type": "person", "slug": "alice", "source_pattern": "at_mention", "excerpt": "x"},
        ]
        existing = {"projects": {"foo"}, "tools": set(), "people": {"alice"}}
        result = filter_new_entities(entities, existing)
        assert len(result) == 0

    def test_mixed(self):
        """Only new entities pass through."""
        entities = [
            {"entity_type": "project", "slug": "old-proj", "source_pattern": "github_repo", "excerpt": "x"},
            {"entity_type": "project", "slug": "new-proj", "source_pattern": "github_repo", "excerpt": "x"},
        ]
        existing = {"projects": {"old-proj"}, "tools": set(), "people": set()}
        result = filter_new_entities(entities, existing)
        assert len(result) == 1
        assert result[0]["slug"] == "new-proj"

    def test_tool_alias_filtering(self):
        """Tool aliases are resolved for filtering.

        For example, 'claude' is aliased to 'codex-cli', so if 'codex-cli'
        is already tracked, 'claude' should be filtered out.
        """
        entities = [
            {"entity_type": "tool", "slug": "claude", "source_pattern": "cli_tool", "excerpt": "x"},
        ]
        existing = {"projects": set(), "tools": {"codex-cli"}, "people": set()}
        result = filter_new_entities(entities, existing)
        assert len(result) == 0

    def test_tool_alias_raw_slug_match(self):
        """Raw slug also matches even without alias resolution."""
        entities = [
            {"entity_type": "tool", "slug": "docker", "source_pattern": "cli_tool", "excerpt": "x"},
        ]
        existing = {"projects": set(), "tools": {"docker"}, "people": set()}
        result = filter_new_entities(entities, existing)
        assert len(result) == 0

    def test_empty_entities_list(self):
        """Empty input yields empty output."""
        existing = {"projects": {"foo"}, "tools": set(), "people": set()}
        result = filter_new_entities([], existing)
        assert result == []


# ---------------------------------------------------------------------------
# entities_to_signals
# ---------------------------------------------------------------------------


class TestEntitiesToSignals:
    """Test signal dict generation from extracted entities."""

    def test_project_signal_format(self):
        """Project entity produces correct signal structure."""
        entities = [
            {
                "entity_type": "project",
                "slug": "mind-mem",
                "source_pattern": "github_repo",
                "excerpt": "star-ga/mind-mem repo",
            },
        ]
        signals = entities_to_signals(entities, "test.md")
        assert len(signals) == 1
        sig = signals[0]
        assert sig["type"] == "entity"
        assert sig["pattern"] == "auto-capture-entity"
        assert sig["confidence"] == "medium"
        assert sig["priority"] == "P2"
        assert "PRJ-mind-mem" in sig["text"]
        assert sig["structure"]["subject"] == "PRJ-mind-mem"
        assert sig["structure"]["object"] == "project"
        assert "entity-project" in sig["structure"]["tags"]
        assert "auto-ingest" in sig["structure"]["tags"]

    def test_tool_signal_format(self):
        """Tool entity uses TOOL- prefix."""
        entities = [
            {"entity_type": "tool", "slug": "docker", "source_pattern": "cli_tool", "excerpt": "docker build"},
        ]
        signals = entities_to_signals(entities, "test.md")
        assert signals[0]["structure"]["subject"] == "TOOL-docker"

    def test_person_signal_format(self):
        """Person entity uses PER- prefix."""
        entities = [
            {"entity_type": "person", "slug": "alice", "source_pattern": "at_mention", "excerpt": "@alice review"},
        ]
        signals = entities_to_signals(entities, "test.md")
        assert signals[0]["structure"]["subject"] == "PER-alice"

    def test_excerpt_truncation_in_text(self):
        """Excerpt in signal text is truncated to 100 chars."""
        long_excerpt = "x" * 200
        entities = [
            {"entity_type": "project", "slug": "foo", "source_pattern": "prj_ref", "excerpt": long_excerpt},
        ]
        signals = entities_to_signals(entities, "test.md")
        # The text field should contain at most 100 chars of excerpt
        assert long_excerpt[:100] in signals[0]["text"]
        assert long_excerpt[:101] not in signals[0]["text"]

    def test_multiple_entities(self):
        """Multiple entities produce multiple signals."""
        entities = [
            {"entity_type": "project", "slug": "a", "source_pattern": "prj_ref", "excerpt": "a"},
            {"entity_type": "tool", "slug": "b", "source_pattern": "tool_ref", "excerpt": "b"},
            {"entity_type": "person", "slug": "c", "source_pattern": "per_ref", "excerpt": "c"},
        ]
        signals = entities_to_signals(entities, "src.md")
        assert len(signals) == 3
        prefixes = [s["structure"]["subject"].split("-")[0] for s in signals]
        assert prefixes == ["PRJ", "TOOL", "PER"]

    def test_empty_entities(self):
        """Empty input yields empty output."""
        assert entities_to_signals([], "test.md") == []
