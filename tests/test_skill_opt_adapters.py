# Copyright 2026 STARGA, Inc.
"""Regression tests for skill discovery.

Agent discovery accepted any ``.md`` file in the agents directory, so
prose documents that merely live there (constitutions, rubrics) became
valid ``mm skill optimize`` targets — while ``can_handle``, the declared
predicate for "is this mine?", was never consulted. Separately, a file
that could not be read or decoded was dropped from discovery in
silence, and resurfaced downstream as "Skill not found".
"""

from __future__ import annotations

from mind_mem.skill_opt.adapters import ClaudeAgentAdapter, discover_all

_AGENT = "---\nname: reviewer\ndescription: Reviews code\n---\n\nYou are a reviewer.\n"
_PROSE = "# THE CONSTITUTION\n\nInvariants every agent obeys.\n"


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def warning(self, event: str, **kwargs) -> None:
        self.events.append((event, kwargs))

    def info(self, event: str, **kwargs) -> None:
        pass

    def debug(self, event: str, **kwargs) -> None:
        pass


class TestAgentDiscoveryIsNarrow:
    def test_prose_in_the_agents_dir_is_not_an_agent(self, tmp_path) -> None:
        agents = tmp_path / "agents"
        agents.mkdir()
        (agents / "reviewer.md").write_text(_AGENT, encoding="utf-8")
        (agents / "CONSTITUTION.md").write_text(_PROSE, encoding="utf-8")

        found = ClaudeAgentAdapter().discover(str(agents))
        assert [p.rsplit("/", 1)[-1] for p in found] == ["reviewer.md"]

    def test_can_handle_matches_what_discover_returns(self, tmp_path) -> None:
        agents = tmp_path / "agents"
        agents.mkdir()
        agent = agents / "reviewer.md"
        prose = agents / "CONSTITUTION.md"
        agent.write_text(_AGENT, encoding="utf-8")
        prose.write_text(_PROSE, encoding="utf-8")

        adapter = ClaudeAgentAdapter()
        assert adapter.can_handle(str(agent)) is True
        assert adapter.can_handle(str(prose)) is False

    def test_discovered_agents_still_parse(self, tmp_path) -> None:
        agents = tmp_path / "agents"
        agents.mkdir()
        (agents / "reviewer.md").write_text(_AGENT, encoding="utf-8")
        specs = discover_all({"claude_agents": str(agents)})
        assert [s.skill_id for s in specs] == ["claude:reviewer"]


class TestDiscoveryFailuresAreReported:
    def test_undecodable_agent_file_is_named(self, tmp_path, monkeypatch) -> None:
        from mind_mem.skill_opt import adapters

        agents = tmp_path / "agents"
        agents.mkdir()
        (agents / "broken.md").write_bytes(b"---\nname: broken\n---\n\nca\xfd va\n")
        recorder = _Recorder()
        monkeypatch.setattr(adapters, "_log", recorder)

        assert ClaudeAgentAdapter().discover(str(agents)) == []
        assert [e for e in recorder.events if e[0] == "skill_file_unreadable"]

    def test_unparseable_skill_is_named(self, tmp_path, monkeypatch) -> None:
        from mind_mem.skill_opt import adapters

        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_bytes(b"---\nname: my-skill\n---\n\nca\xfd va\n")
        recorder = _Recorder()
        monkeypatch.setattr(adapters, "_log", recorder)

        assert discover_all({"openclaw": str(tmp_path)}) == []
        events = [e for e in recorder.events if e[0] == "skill_discovery_failed"]
        assert events and events[0][1]["error_type"] == "UnicodeDecodeError"
        assert events[0][1]["source"] == "openclaw"
