# Copyright 2026 STARGA, Inc.
"""The ``skills/mind-mem`` agent skill (CLI user manual) cannot rot.

An agent follows a skill file literally, so a documented command that no
longer exists is worse than no documentation: the agent runs it, fails, and
has nothing else to go on. These tests pin the manual to the code:

* ``references/cli.md`` is generated from ``mm_cli.build_parser()`` and must
  match a fresh render byte for byte.
* every leaf command has an example, and every example parses.
* every ``mm ...`` line in any fenced shell block of any skill file parses
  with the real parser -- subcommand AND flags.
* every console script, ``MIND_MEM_*`` variable, client key and config key the
  manual names exists in the code.
* the table of contents links every reference file and every link resolves.
* ``mm skill install`` copies the bundle, is idempotent and never clobbers a
  different copy without ``--force``.

Each check has a positive control so an empty scan cannot pass vacuously.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "skills" / "mind-mem"
SKILL_MD = SKILL_DIR / "SKILL.md"
REF_DIR = SKILL_DIR / "references"
GEN = ROOT / "scripts" / "gen_skill_cli_reference.py"

sys.path.insert(0, str(ROOT / "scripts"))
import gen_skill_cli_reference as gen  # noqa: E402

from mind_mem.mm_cli import build_parser  # noqa: E402

_FENCE_RE = re.compile(r"```(?:bash|sh|shell)\n(.*?)```", re.DOTALL)


def _skill_files() -> list[Path]:
    return [SKILL_MD, *sorted(REF_DIR.glob("*.md"))]


def _parses(argv: list[str]) -> tuple[bool, str]:
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            build_parser().parse_args(argv)
    except SystemExit as exc:
        return exc.code == 0, err.getvalue().strip()
    return True, ""


def _documented_mm_lines() -> list[tuple[Path, str, bool]]:
    """Every documented ``mm ...`` invocation as ``(file, line, is_fenced)``.

    Fenced shell lines are commands meant to run and must parse completely.
    Inline spans are often a bare verb mention (`mm import`), so for them a
    missing *required* argument is fine -- an unknown verb, flag or choice is not.
    """
    found: list[tuple[Path, str, bool]] = []
    for path in _skill_files():
        for block in _FENCE_RE.findall(path.read_text(encoding="utf-8")):
            for raw in block.splitlines():
                line = raw.split(" #", 1)[0].strip()
                # tolerate a leading env assignment such as MIND_MEM_SCOPE=admin
                line = re.sub(r"^(?:[A-Z_][A-Z0-9_]*=\S+\s+)+", "", line)
                if line.startswith("mm ") or line == "mm":
                    found.append((path, line, True))
        # Inline code spans too (the routing tables are written this way).
        # Spans with "..." or a "<placeholder>" in verb position are
        # templates, not commands.
        for span in re.findall(r"`(mm [^`\n]+)`", path.read_text(encoding="utf-8")):
            span = re.sub(r"^(?:[A-Z_][A-Z0-9_]*=\S+\s+)+", "", span)
            if "..." in span or span.startswith("mm <"):
                continue
            found.append((path, span, False))
    return found


def _mm_argv(line: str) -> list[str]:
    argv = shlex.split(line)[1:]
    # stop at a shell pipe / redirection, which belongs to the shell, not mm
    for i, tok in enumerate(argv):
        if tok in ("|", "&&", ";") or tok.startswith(("2>", ">")):
            return argv[:i]
    return argv


# ---------------------------------------------------------------------------
# Generated CLI reference
# ---------------------------------------------------------------------------


class TestGeneratedCliReference:
    def test_renderer_sees_a_real_parser(self) -> None:
        leaves = gen.leaf_commands()
        assert len(leaves) >= 50, "positive control: the parser walk found almost nothing"
        assert any(path == "recall" for path, _, _ in leaves)
        assert any(path == "skill install" for path, _, _ in leaves)

    def test_committed_cli_md_matches_the_parser(self) -> None:
        committed = (REF_DIR / "cli.md").read_text(encoding="utf-8")
        assert committed == gen.render(), (
            "skills/mind-mem/references/cli.md has drifted from mm_cli.build_parser(). Run: python3 scripts/gen_skill_cli_reference.py"
        )

    def test_every_leaf_command_has_an_example(self) -> None:
        leaves = {path for path, _, _ in gen.leaf_commands()}
        missing = sorted(leaves - set(gen.EXAMPLES))
        stale = sorted(set(gen.EXAMPLES) - leaves)
        assert not missing, f"add an example to EXAMPLES in scripts/gen_skill_cli_reference.py for: {missing}"
        assert not stale, f"EXAMPLES names commands the parser no longer has: {stale}"

    @pytest.mark.parametrize("path", sorted(gen.EXAMPLES))
    def test_example_parses(self, path: str) -> None:
        example = gen.EXAMPLES[path]
        argv = _mm_argv(example)
        assert " ".join(argv[: len(path.split())]) == path, f"example for {path!r} runs a different command: {example}"
        ok, err = _parses(argv)
        assert ok, f"example for 'mm {path}' does not parse: {example}\n{err}"

    def test_check_mode_reports_up_to_date(self) -> None:
        proc = subprocess.run([sys.executable, str(GEN), "--check"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# Hand-written files: every command, flag, name must exist
# ---------------------------------------------------------------------------


class TestHandWrittenManual:
    def test_scanner_finds_commands(self) -> None:
        lines = _documented_mm_lines()
        assert len(lines) >= 25, f"positive control: only {len(lines)} documented mm lines found"
        assert any(line.startswith("mm recall") for _, line, fenced in lines if fenced)
        assert any(not fenced for _, _, fenced in lines), "positive control: no inline spans scanned"

    def test_scanner_rejects_an_invented_command_and_flag(self) -> None:
        """Mutation control: the parse check must fail on things that do not exist."""
        assert not _parses(["frobnicate"])[0]
        assert not _parses(["recall", "x", "--no-such-flag"])[0]
        assert _parses(["recall", "x", "--limit", "3"])[0]

    def test_every_documented_mm_line_parses(self) -> None:
        bad = []
        for path, line, fenced in _documented_mm_lines():
            ok, err = _parses(_mm_argv(line))
            if not ok and not fenced and "the following arguments are required" in err:
                continue
            if not ok:
                bad.append(f"{path.relative_to(ROOT)}: {line}\n    {err.splitlines()[-1] if err else ''}")
        assert not bad, "documented commands the CLI rejects:\n" + "\n".join(bad)

    def test_inline_mm_subcommands_exist(self) -> None:
        """`mm <verb>` in inline code anywhere in the manual names a real verb."""
        top = gen._subparsers(build_parser())
        assert top is not None
        verbs = set(top.choices)
        named: set[str] = set()
        for path in _skill_files():
            named.update(re.findall(r"`mm ([a-z][a-z-]*)", path.read_text(encoding="utf-8")))
        assert named, "positive control: no inline mm verbs found"
        assert not sorted(named - verbs), f"inline `mm <verb>` names that are not subcommands: {sorted(named - verbs)}"

    def test_console_scripts_exist(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        scripts = set(re.findall(r"^([a-z][a-z0-9-]*) = \"mind_mem[.\w]*:\w+\"", pyproject, re.MULTILINE))
        assert "mm" in scripts and "mind-mem-init" in scripts, "positive control: script table not parsed"
        named: set[str] = set()
        for path in _skill_files():
            named.update(re.findall(r"(?<![./\w-])(mind-mem-[a-z-]+)\b", path.read_text(encoding="utf-8")))
        named -= {"mind-mem-4b", "mind-mem-workspace", "mind-mem-inbox"}  # model / example paths, not scripts
        assert named, "positive control: no console scripts named"
        assert not sorted(named - scripts), f"documented console scripts not in pyproject: {sorted(named - scripts)}"

    def test_env_vars_exist_in_source(self) -> None:
        source = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in (ROOT / "src" / "mind_mem").rglob("*.py"))
        named: set[str] = set()
        for path in _skill_files():
            named.update(re.findall(r"\b(MIND_MEM_[A-Z0-9_]+)\b", path.read_text(encoding="utf-8")))
        assert "MIND_MEM_WORKSPACE" in named, "positive control"
        missing = sorted(v for v in named if v not in source)
        assert not missing, f"documented env vars no code reads: {missing}"

    def test_client_keys_exist(self) -> None:
        from mind_mem.hook_installer import AGENT_REGISTRY

        text = (REF_DIR / "install.md").read_text(encoding="utf-8")
        para = text.split("Client keys accepted by", 1)[1].split("\n\n", 1)[0]
        keys = set(re.findall(r"`([a-z][a-z-]*)`", para)) - {"mm", "install", "--agent"}
        assert len(keys) >= 10, f"positive control: only {sorted(keys)} parsed"
        # Subset, not equality: a client added to the registry by another
        # change leaves the list incomplete (still correct), but a key the
        # registry dropped would have an agent run a command that fails.
        invented = sorted(keys - set(AGENT_REGISTRY))
        assert not invented, f"install.md names client keys the installer does not know: {invented}"

    def test_config_keys_have_readers(self) -> None:
        source = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in (ROOT / "src" / "mind_mem").rglob("*.py"))
        hooks = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in (ROOT / "hooks").rglob("*.sh"))
        corpus = source + hooks
        text = (REF_DIR / "configuration.md").read_text(encoding="utf-8")
        table = text.split("## Frequently used keys", 1)[1].split("\n## ", 1)[0]
        keys: set[str] = set()
        for row in table.splitlines():
            if row.startswith("| `"):
                keys.update(re.findall(r"`([a-z_][a-z0-9_.]*)`", row.split("|")[1]))
        assert "governance_mode" in keys, "positive control"
        missing = []
        for key in sorted(keys):
            leaf = key.split(".")[-1]
            if f'"{leaf}"' not in corpus and f"'{leaf}'" not in corpus:
                missing.append(key)
        assert not missing, f"configuration.md documents keys nothing reads: {missing}"


# ---------------------------------------------------------------------------
# Table of contents
# ---------------------------------------------------------------------------


class TestTableOfContents:
    def test_frontmatter(self) -> None:
        text = SKILL_MD.read_text(encoding="utf-8")
        match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
        assert match, "SKILL.md needs YAML frontmatter"
        front = match.group(1)
        assert re.search(r"^name: mind-mem$", front, re.MULTILINE)
        desc = re.search(r"^description: (.+)$", front, re.MULTILINE)
        assert desc and 50 < len(desc.group(1)) <= 1024

    def test_every_reference_is_linked_and_every_link_resolves(self) -> None:
        refs = {p.name for p in REF_DIR.glob("*.md")}
        assert {"install.md", "cli.md", "configuration.md", "troubleshooting.md", "mcp-vs-cli.md", "faq.md"} <= refs
        toc = SKILL_MD.read_text(encoding="utf-8")
        unlinked = sorted(r for r in refs if f"references/{r}" not in toc)
        assert not unlinked, f"reference files the table of contents never links: {unlinked}"
        for path in _skill_files():
            for target in re.findall(r"\]\(([^)#\s]+\.md)(?:#[^)]*)?\)", path.read_text(encoding="utf-8")):
                if target.startswith("http"):
                    continue
                assert (path.parent / target).resolve().is_file(), f"{path.relative_to(ROOT)} links missing {target}"

    def test_shipped_in_the_wheel(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert '"share/mind-mem/skills/mind-mem" = ["skills/mind-mem/SKILL.md"]' in pyproject
        assert '"share/mind-mem/skills/mind-mem/references" = ["skills/mind-mem/references/*.md"]' in pyproject


# ---------------------------------------------------------------------------
# mm skill install
# ---------------------------------------------------------------------------


class TestSkillInstall:
    def test_bundle_is_found_from_a_checkout(self) -> None:
        from mind_mem.skill_bundle import bundled_skill_dir

        found = bundled_skill_dir()
        assert found is not None and (found / "SKILL.md").is_file()

    def test_install_then_idempotent_then_refuses_then_force(self, tmp_path: Path) -> None:
        from mind_mem.skill_bundle import install_skill

        dry = install_skill(tmp_path, dry_run=True)
        assert dry["status"] == "would_install"
        assert not (tmp_path / "mind-mem").exists(), "dry run wrote files"

        first = install_skill(tmp_path)
        assert first["status"] == "installed"
        dest = tmp_path / "mind-mem"
        assert (dest / "SKILL.md").read_bytes() == SKILL_MD.read_bytes()
        assert (dest / "references" / "cli.md").is_file()
        assert sorted(first["files"]) == sorted(str(p.relative_to(SKILL_DIR)) for p in SKILL_DIR.rglob("*.md"))

        assert install_skill(tmp_path)["status"] == "up_to_date"

        (dest / "SKILL.md").write_text("locally edited\n", encoding="utf-8")
        refused = install_skill(tmp_path)
        assert refused["status"] == "exists"
        assert (dest / "SKILL.md").read_text(encoding="utf-8") == "locally edited\n", "refusal still overwrote"

        forced = install_skill(tmp_path, force=True)
        assert forced["status"] == "installed"
        assert (dest / "SKILL.md").read_bytes() == SKILL_MD.read_bytes()

    def test_missing_bundle_is_reported(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem import skill_bundle

        monkeypatch.setattr(skill_bundle, "bundled_skill_dir", lambda: None)
        report = skill_bundle.install_skill(tmp_path)
        assert report["status"] == "missing_bundle"
        assert not any(tmp_path.iterdir())

    def test_cli_entry_point(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from mind_mem.mm_cli import main

        assert main(["skill", "install", "--target", str(tmp_path)]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == "installed"
        assert (tmp_path / "mind-mem" / "SKILL.md").is_file()
        (tmp_path / "mind-mem" / "SKILL.md").write_text("x", encoding="utf-8")
        assert main(["skill", "install", "--target", str(tmp_path)]) == 1
        assert json.loads(capsys.readouterr().out)["status"] == "exists"
