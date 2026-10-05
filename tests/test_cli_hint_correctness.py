# Copyright 2026 STARGA, Inc.
"""Commands and config values the product tells users about must exist.

Each case is a string that used to point at something that did not work:
* ``mind-mem-init`` wrote ``recall.backend = "bm25"``, which recall reported as
  an unknown backend on every call;
* client hints said ``mm inject --agent <x>`` without the required query;
* the dashboard suggested ``mm compact``, which is not a subcommand;
* docs/install-guide.md told users to run ``mm migrate-store --check``.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from mind_mem import _recall_core as core
from mind_mem import hook_installer
from mind_mem.init_workspace import _BACKEND_RECALL, DEFAULT_CONFIG
from mind_mem.mm_cli import build_parser

REPO_ROOT = Path(__file__).resolve().parent.parent


def _parses(command: str) -> None:
    argv = shlex.split(command.replace("<question>", "q"))
    assert argv[0] == "mm", command
    try:
        build_parser().parse_args(argv[1:])
    except SystemExit as exc:  # argparse exits 2 on a bad command line
        pytest.fail(f"not a valid mm command line: {command!r} (exit {exc.code})")


class TestInitWritesAKnownRecallBackend:
    def test_default_config_backend_is_known(self) -> None:
        assert DEFAULT_CONFIG["recall"]["backend"] in core._KNOWN_RECALL_BACKENDS

    @pytest.mark.parametrize("store", sorted(_BACKEND_RECALL))
    def test_every_store_default_is_known(self, store: str) -> None:
        assert _BACKEND_RECALL[store] in core._KNOWN_RECALL_BACKENDS | core._POSTGRES_DELEGATED_RECALL_BACKENDS

    def test_bm25_logs_no_unknown_backend_warning(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "mind-mem.json").write_text('{"recall": {"backend": "bm25"}}', encoding="utf-8")
        seen: list[str] = []

        class _Spy:
            def __getattr__(self, _level: str):
                return lambda event, **_kw: seen.append(event)

        monkeypatch.setattr(core, "_log", _Spy())
        assert core._load_backend(str(tmp_path)) is None  # the built-in BM25 scan
        assert "unknown_recall_backend" not in seen

    def test_a_typo_still_warns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Positive control: the warning still fires for a value nobody handles."""
        (tmp_path / "mind-mem.json").write_text('{"recall": {"backend": "bm52"}}', encoding="utf-8")
        seen: list[str] = []

        class _Spy:
            def __getattr__(self, _level: str):
                return lambda event, **_kw: seen.append(event)

        monkeypatch.setattr(core, "_log", _Spy())
        core._load_backend(str(tmp_path))
        assert "unknown_recall_backend" in seen


def _strings(node: object) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [s for v in node.values() for s in _strings(v)]
    if isinstance(node, list):
        return [s for v in node for s in _strings(v)]
    return []


def _hint_commands(text: str) -> list[str]:
    return [c for c in re.findall(r"`(mm [^`]+)`", text) if c.startswith("mm inject")]


class TestClientHintsAreRunnable:
    @pytest.mark.parametrize("agent", ["gemini", "continue", "zed"])
    def test_json_hint_inject_command_parses(self, agent: str, tmp_path: Path) -> None:
        spec = hook_installer.AGENT_REGISTRY[agent]
        merged, _ = hook_installer._JSON_MERGERS[spec.config_fmt]({}, str(tmp_path))
        commands = [c for text in _strings(merged) for c in _hint_commands(text)]
        assert commands, f"{agent}: no `mm inject` hint found"
        for command in commands:
            _parses(command)

    def test_copilot_hint_inject_command_parses(self) -> None:
        commands = _hint_commands(hook_installer.AGENT_REGISTRY["copilot"].content_tmpl)
        assert commands
        for command in commands:
            _parses(command)

    def test_a_query_less_hint_would_be_caught(self) -> None:
        """Mutation twin: the old hint text fails the same check."""
        with pytest.raises(pytest.fail.Exception):
            _parses("mm inject --agent gemini")


class TestDocumentedCommandsExist:
    def test_dashboard_names_a_real_console_script(self) -> None:
        from mind_mem import accountability_dashboard

        source = Path(accountability_dashboard.__file__).read_text(encoding="utf-8")
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "`mm compact`" not in source
        assert re.search(r"^mind-mem-compact = ", pyproject, flags=re.M)
        assert "`mind-mem-compact` runs one" in source

    def test_install_guide_mm_commands_parse(self) -> None:
        text = (REPO_ROOT / "docs" / "install-guide.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```bash\n(.*?)```", text, flags=re.S)
        commands = []
        for block in blocks:
            for line in block.splitlines():
                line = line.split("#", 1)[0].strip()
                if line.startswith("mm "):
                    commands.append(line)
        assert any(c.startswith("mm doctor --migrate-recall-log") for c in commands)
        subcommands = set(build_parser()._subparsers._group_actions[0].choices)  # type: ignore[union-attr]
        for command in commands:
            assert shlex.split(command)[1] in subcommands, command
        assert "migrate-store --check" not in text
