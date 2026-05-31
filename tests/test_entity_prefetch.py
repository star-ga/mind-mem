"""v3.3.0 Tier 3 #8 — entity-graph prefetch.

When a question mentions a Person / Project / Tool / Incident, pre-fetch
the entity block and its 1-hop graph neighbourhood so the downstream
RRF fusion has that evidence even when BM25 misses the entity tokens.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mind_mem.entity_prefetch import (
    extract_entity_candidates,
    is_entity_prefetch_enabled,
    prefetch_entity_blocks,
    resolve_entity_prefetch_config,
)


@pytest.fixture
def entities_workspace(tmp_path: Path) -> str:
    """Workspace with one person + one project entity block."""
    for d in ("decisions", "tasks", "entities", "intelligence", "memory", ".mind-mem-index"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)

    (tmp_path / "entities" / "people.md").write_text(
        "[PER-001]\ntype: person\nName: Alice Johnson\nAliases: Alice\n"
        "Statement: Engineer on the PostgreSQL migration. See D-20260420-001 for the decision.\n"
        "---\n\n",
        encoding="utf-8",
    )
    (tmp_path / "entities" / "projects.md").write_text(
        "[PRJ-042]\ntype: project\nName: PostgreSQL Migration\nStatement: Migrate from SQLite to PostgreSQL.\n---\n\n",
        encoding="utf-8",
    )
    (tmp_path / "decisions" / "DECISIONS.md").write_text(
        "[D-20260420-001]\ntype: decision\nStatement: Use PostgreSQL for the storage tier.\n---\n\n",
        encoding="utf-8",
    )
    return str(tmp_path)


class TestCandidateExtraction:
    def test_capitalized_tokens(self) -> None:
        cands = extract_entity_candidates("What did Alice say about PostgreSQL?")
        # Extraction is intentionally broad (regex-level) — filtering to
        # real entities happens when each candidate is checked against
        # the entity corpus in prefetch_entity_blocks.
        assert "Alice" in cands
        assert "PostgreSQL" in cands

    def test_empty_query_returns_empty(self) -> None:
        assert extract_entity_candidates("") == []
        assert extract_entity_candidates("   ") == []

    def test_all_lowercase_returns_empty(self) -> None:
        assert extract_entity_candidates("what did she say about it?") == []

    def test_dedupes_repeated_tokens(self) -> None:
        """Repeated tokens from the same query produce one candidate."""
        cands = extract_entity_candidates("Alice and Alice and Alice")
        assert cands.count("Alice") == 1


class TestPrefetch:
    def test_no_match_returns_empty(self, entities_workspace: str) -> None:
        """Query with no entity-matching tokens → empty."""
        assert prefetch_entity_blocks("quantum gravity thermodynamics", entities_workspace) == []

    def test_matches_person_by_name(self, entities_workspace: str) -> None:
        """'Alice' matches PER-001 (Name+Aliases)."""
        out = prefetch_entity_blocks("What did Alice say about the migration?", entities_workspace, max_hops=0)
        ids = [b["_id"] for b in out]
        assert "PER-001" in ids
        assert all(b["_prefetch"] in ("entity", "entity_neighbour") for b in out)

    def test_matches_project_by_statement_keyword(self, entities_workspace: str) -> None:
        """'PostgreSQL' in Statement field matches PRJ-042."""
        out = prefetch_entity_blocks("PostgreSQL migration timeline?", entities_workspace, max_hops=0)
        ids = [b["_id"] for b in out]
        assert "PRJ-042" in ids

    def test_no_entities_directory_returns_empty(self, tmp_path: Path) -> None:
        """Fresh workspace without entities/ returns nothing (not an error)."""
        ws = tmp_path / "fresh"
        ws.mkdir()
        assert prefetch_entity_blocks("Alice", str(ws)) == []

    def test_max_entities_cap(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        (ws / "entities").mkdir(parents=True)
        (ws / "decisions").mkdir()
        # Five matching entities; cap at 2.
        people = ""
        for i in range(5):
            people += f"[PER-{i:03d}]\ntype: person\nName: Alice Clone {i}\nAliases: Alice\nStatement: Copy number {i}.\n---\n\n"
        (ws / "entities" / "people.md").write_text(people, encoding="utf-8")

        out = prefetch_entity_blocks("Alice notes", str(ws), max_entities=2, max_hops=0)
        # Only entity blocks (no graph hop), so exactly 2 returned
        assert len([b for b in out if b.get("_prefetch") == "entity"]) == 2

    def test_one_hop_walk_via_entity(self, entities_workspace: str) -> None:
        """Entity block cross-references a decision → graph-walk surfaces it."""
        out = prefetch_entity_blocks(
            "What did Alice say?",
            entities_workspace,
            max_hops=1,
        )
        ids = [b["_id"] for b in out]
        # Alice's block references D-20260420-001
        assert "PER-001" in ids
        assert "D-20260420-001" in ids

    def test_every_result_is_annotated(self, entities_workspace: str) -> None:
        out = prefetch_entity_blocks("Alice", entities_workspace, max_hops=0)
        assert out
        for b in out:
            assert "_prefetch" in b


class TestEnableResolution:
    def test_default_off_without_config(self) -> None:
        assert is_entity_prefetch_enabled(None) is False
        assert is_entity_prefetch_enabled({}) is False

    def test_enabled_explicit(self) -> None:
        assert is_entity_prefetch_enabled({"retrieval": {"entity_prefetch": {"enabled": True}}}) is True

    def test_auto_enable_default_on_when_section_present(self) -> None:
        """When retrieval.entity_prefetch exists but enabled is unset, auto_enable defaults True."""
        assert is_entity_prefetch_enabled({"retrieval": {"entity_prefetch": {}}}) is True

    def test_auto_enable_false(self) -> None:
        cfg = {"retrieval": {"entity_prefetch": {"enabled": False, "auto_enable": False}}}
        assert is_entity_prefetch_enabled(cfg) is False


class TestConfigResolution:
    def test_defaults(self) -> None:
        cfg = resolve_entity_prefetch_config(None)
        assert cfg == {"max_entities": 3, "max_hops": 1, "entity_score": 5.0}

    def test_custom_values(self) -> None:
        cfg = resolve_entity_prefetch_config({"retrieval": {"entity_prefetch": {"max_entities": 5, "max_hops": 2, "entity_score": 8.0}}})
        assert cfg["max_entities"] == 5
        assert cfg["max_hops"] == 2
        assert cfg["entity_score"] == pytest.approx(8.0)

    def test_invalid_values_fall_back(self) -> None:
        cfg = resolve_entity_prefetch_config({"retrieval": {"entity_prefetch": {"max_entities": -1, "max_hops": "x", "entity_score": 0}}})
        assert cfg == {"max_entities": 3, "max_hops": 1, "entity_score": 5.0}
