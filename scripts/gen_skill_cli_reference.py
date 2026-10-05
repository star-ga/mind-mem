#!/usr/bin/env python3
# Copyright 2026 STARGA, Inc.
"""Render ``skills/mind-mem/references/cli.md`` from the live ``mm`` parser.

The agent skill's CLI reference is generated, never hand-maintained: every
subcommand, positional and flag comes from ``mind_mem.mm_cli.build_parser()``,
and every leaf command carries one example from :data:`EXAMPLES` that the
test suite parses with the same parser. A renamed flag or a removed verb makes
``tests/test_skill_manual.py`` fail until this file is re-run.

Usage::

    python3 scripts/gen_skill_cli_reference.py          # rewrite cli.md
    python3 scripts/gen_skill_cli_reference.py --check  # exit 1 on drift
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "skills" / "mind-mem" / "references" / "cli.md"

# One runnable example per leaf command. Keys are the command path after
# ``mm``. Each value is parsed by build_parser() in the test suite, so an
# example can never name a flag the CLI does not accept.
EXAMPLES: dict[str, str] = {
    "recall": 'mm recall "database decision" --limit 5 --active-only',
    "context": 'mm context "billing service" --max-tokens 1500',
    "inject": 'mm inject "auth migration" --agent claude-code',
    "resume": "mm resume --json",
    "resume-on-start": "mm resume-on-start",
    "dead-ends": 'mm dead-ends --tool Bash --command "pip install foo"',
    "review": "mm review --json",
    "status": "mm status",
    "index": "mm index",
    "usage": "mm usage --json",
    "tool-run": "mm tool-run -- pytest -q",
    "tool-recall": "mm tool-recall to-e5700398f3f9b657",
    "migrate-store": 'mm migrate-store --from markdown --to postgres --dsn "$MIND_MEM_DSN" --dry-run',
    "migrate": "mm migrate --maintenance",
    "lint": "mm lint",
    "detect": "mm detect",
    "install": "mm install claude-code --dry-run",
    "install-all": "mm install-all --dry-run",
    "install-model": "mm install-model --dry-run",
    "doctor": "mm doctor",
    "token rotate": "mm token rotate",
    "import": "mm import --from markdown ~/notes --dry-run",
    "kinds backfill": "mm kinds backfill",
    "kinds list": "mm kinds list --kind concept --limit 20",
    "vault scan": "mm vault scan ~/vault",
    "vault write": 'mm vault write ~/vault notes/db.md --id D-20261005-001 --title "DB choice" --body "Use Postgres 16"',
    "lineage flag": "mm lineage flag D-20261005-002 D-20261005-001 --kind supersedes",
    "skill list": "mm skill list",
    "skill test": "mm skill test claude:code-reviewer",
    "skill analyze": "mm skill analyze claude:code-reviewer",
    "skill optimize": "mm skill optimize claude:code-reviewer",
    "skill history": "mm skill history claude:code-reviewer --limit 5",
    "skill score": "mm skill score claude:code-reviewer",
    "skill install": "mm skill install --target ~/.claude/skills",
    "serve": "mm serve --port 8080",
    "http-serve": "mm http-serve --port 8765",
    "view": "mm view --port 0",
    "daemon": "mm daemon --once --dry-run",
    "inbox-watch": "mm inbox-watch ~/mind-mem-inbox --once",
    "ingest-serve": "mm ingest-serve --replay-only",
    "send": 'mm send "Build is green on main" --from agent-a --to agent-b --subject ci',
    "inbox": "mm inbox --to agent-b --limit 10",
    "graph-backfill": "mm graph-backfill --limit 25",
    "graph-answer": "mm graph-answer PostgreSQL --hops 2 --json",
    "pipeline-status": "mm pipeline-status --json",
    "accountability": "mm accountability",
    "dashboard": "mm dashboard --json",
    "replay-check": "mm replay-check --attestation attestation.json",
    "audit-model": "mm audit-model ./checkpoints/my-model --json",
    "sign-model": "mm sign-model ./checkpoints/my-model --generate-key ./keys/release",
    "verify-model": "mm verify-model ./checkpoints/my-model",
    "gate check": "mm gate check ./checkpoints/my-model --json",
    "gate list": "mm gate list",
    "gate remove": "mm gate remove ./checkpoints/my-model",
    "bind": "mm bind --json",
    "config set": "mm config set governance_mode propose",
    "audit-pinned": "mm audit-pinned --config mind-mem.json",
    "anchor": 'mm anchor "$MIND_MEM_WORKSPACE" --json',
    "chain survey": 'mm chain survey "$MIND_MEM_WORKSPACE"',
    "chain verify-archive": 'mm chain verify-archive "$MIND_MEM_WORKSPACE"',
    "chain witness": 'mm chain witness "$MIND_MEM_WORKSPACE" --anchor-id EV-1 --anchor-hash abc123 --json',
    "chain recover": 'mm chain recover "$MIND_MEM_WORKSPACE"',
    "mic convert": "mm mic convert graph.mic --to micb -o graph.micb",
    "mic inspect": "mm mic inspect graph.micb --json",
    "inspect": "mm inspect D-20261005-001 --format json",
    "explain": 'mm explain "database decision" --limit 5',
    "trace": "mm trace --last 20",
    "export": "mm export --policy redacted --format jsonl --out bundle.jsonl",
    "receipt export": "mm receipt export --out receipt.json",
    "receipt verify": "mm receipt verify --input receipt.json",
    "recompact": "mm recompact D-20261005-001 --limit 5",
    "compliance detectors": "mm compliance detectors",
    "compliance scan": 'mm compliance scan --text "contact alice@example.com"',
    "compliance redact": "mm compliance redact --file notes.md",
    "compliance screen": "mm compliance screen --file notes.md --json",
    "compliance provenance": "mm compliance provenance --json",
    "self-update": "mm self-update --check",
}

HEADER = """# `mm` command reference

<!-- GENERATED by scripts/gen_skill_cli_reference.py from mind_mem.mm_cli.build_parser().
     Do not edit by hand: tests/test_skill_manual.py fails when this file and the parser disagree. -->

Every subcommand the installed `mm` accepts, with its flags and one example.
The one-line descriptions are the parser's own help text. For *which* command
to reach for, start at [../SKILL.md](../SKILL.md); this page answers *how*.

Conventions that hold for every command:

* The workspace is `$MIND_MEM_WORKSPACE`, or the current directory when it is
  unset. Commands that take a `workspace` positional (`anchor`, `chain ...`)
  default to the current directory, so pass `"$MIND_MEM_WORKSPACE"` explicitly.
* Results go to **stdout** (usually JSON); structured logs go to **stderr**.
  Use `2>/dev/null` when piping stdout into a JSON parser.
* `mm <command> --help` prints the same information from the installed version.
"""


def _subparsers(parser: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _metavar(action: argparse.Action) -> str:
    if action.choices is not None and action.metavar is None:
        return "{" + ",".join(str(c) for c in action.choices) + "}"
    if isinstance(action.metavar, str):
        return action.metavar
    return action.dest.upper()


def _signature(path: list[str], parser: argparse.ArgumentParser) -> str:
    parts = ["mm", *path]
    for action in parser._actions:
        if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)):
            continue
        if not action.option_strings:
            name = f"<{action.dest}>"
            if action.nargs == argparse.REMAINDER or action.nargs == "...":
                parts.append(f"-- <{action.dest}...>")
            elif action.nargs in ("?", "*"):
                parts.append(f"[{name}]")
            else:
                parts.append(name)
            continue
        flag = max(action.option_strings, key=len)
        token = flag if action.nargs == 0 else f"{flag} {_metavar(action)}"
        parts.append(token if action.required else f"[{token}]")
    return " ".join(parts)


def _arg_rows(parser: argparse.ArgumentParser) -> list[str]:
    rows: list[str] = []
    for action in parser._actions:
        if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)):
            continue
        name = ", ".join(action.option_strings) if action.option_strings else f"<{action.dest}>"
        desc = _clean(action.help)
        if action.choices is not None:
            choices = ", ".join(f"`{c}`" for c in action.choices)
            desc = f"{desc} Choices: {choices}." if desc else f"Choices: {choices}."
        if action.option_strings and action.required:
            desc = f"(required) {desc}"
        rows.append(f"| `{name}` | {desc.replace('|', '/') or '-'} |")
    return rows


def leaf_commands(parser: argparse.ArgumentParser | None = None) -> list[tuple[str, str, argparse.ArgumentParser]]:
    """Return ``(path, help, parser)`` for every runnable leaf, in parser order."""
    if parser is None:
        from mind_mem.mm_cli import build_parser

        parser = build_parser()
    out: list[tuple[str, str, argparse.ArgumentParser]] = []

    def walk(p: argparse.ArgumentParser, path: list[str], help_text: str) -> None:
        subs = _subparsers(p)
        if subs is None:
            out.append((" ".join(path), help_text, p))
            return
        helps = {choice.dest: _clean(choice.help) for choice in subs._choices_actions}
        for name, child in subs.choices.items():
            walk(child, [*path, name], helps.get(name, ""))

    walk(parser, [], "")
    return out


def render() -> str:
    from mind_mem.mm_cli import build_parser

    parser = build_parser()
    top = _subparsers(parser)
    assert top is not None, "mm parser registered no subcommands"
    top_help = {choice.dest: _clean(choice.help) for choice in top._choices_actions}
    lines = [HEADER, "## Index", "", "| Command | What it does |", "| --- | --- |"]
    for name in top.choices:
        anchor = f"mm-{name}"
        lines.append(f"| [`mm {name}`](#{anchor}) | {top_help.get(name, '').replace('|', '/')} |")
    lines.append("")
    leaves = leaf_commands(parser)
    seen_groups: set[str] = set()
    for path, help_text, sub in leaves:
        group = path.split(" ")[0]
        if " " in path and group not in seen_groups:
            seen_groups.add(group)
            lines += [f"## mm {group}", "", top_help.get(group, ""), ""]
        heading = "###" if " " in path else "##"
        lines += [f"{heading} mm {path}", ""]
        description = help_text
        if not description and sub.description:
            description = _clean(sub.description)
        if description:
            lines += [description, ""]
        lines += ["```text", _signature(path.split(" "), sub), "```", ""]
        rows = _arg_rows(sub)
        if rows:
            lines += ["| Argument | Meaning |", "| --- | --- |", *rows, ""]
        example = EXAMPLES.get(path)
        if example:
            lines += ["Example:", "", "```bash", example, "```", ""]
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="Exit 1 if cli.md is stale instead of rewriting it.")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    text = render()
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
        if current != text:
            print(f"{OUT.relative_to(ROOT)} is stale; run: python3 scripts/gen_skill_cli_reference.py", file=sys.stderr)
            return 1
        print(f"{OUT.relative_to(ROOT)} is up to date")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
