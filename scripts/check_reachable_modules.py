#!/usr/bin/env python3
# Copyright 2026 STARGA, Inc.
"""No caller, no tick — the reachability gate.

This repository's definition of "done" has been *merged with tests*, never
*reachable*. An architectural audit (2026-08-31) verified nine features that
shipped, were ticked on the roadmap, and are invoked by nothing:

    append_only, novel_term_gate, llm_noise_profile, smart_chunker,
    lint_autofix, ontology, kg_fusion, the tier ladder, and 4 of 8
    `mind/*.mind` kernels that do not compile under a ticked box.

The 84-vs-96 tool drift is the same disease from the other side: the surface
grows faster than anything downstream can track. The roadmap already holds the
epigram -- *"an expression index that never matches is invisible; an evidence
field that is never checked is decorative"* -- it had just never been applied to
MODULES.

This gate is the module-level analogue: a module under ``src/mind_mem/`` that
nothing outside its own tests imports is **shelfware**, and shelfware must not
be counted as delivered.

WHAT IT IS NOT. This is not a dead-code detector and it does not delete
anything. A module may be legitimately unreachable *for now* -- newly landed
behind a flag, or waiting on a consumer that is itself an open roadmap item.
Those go in ``ALLOWLIST`` **with the reason and the item that will consume
them**, so the debt is named rather than invisible. An allowlist entry is a
promise with an address, not an exemption.

Usage:
    python3 scripts/check_reachable_modules.py            # report
    python3 scripts/check_reachable_modules.py --check    # exit 1 if any
                                                          # unreachable module
                                                          # is not allowlisted
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "mind_mem"

# Modules that are reachable by a route this AST scan cannot see, or that are
# knowingly-unreachable with a named consumer. Each entry MUST say why, and a
# deferred one MUST name the roadmap item that will consume it.
ALLOWLIST: dict[str, str] = {
    "__init__": "package root",
    # ---- String-dispatched daemon jobs -------------------------------------
    # These three are LIVE, SHIPPED surfaces the AST scan structurally cannot
    # see: nothing imports them, because the daemon dispatches them by NAME.
    # `cron_runner.JOB_DEFS` maps a job string to a module and runs it as
    # `python -m mind_mem.<module>` in a subprocess, and `daemon._TASK_RUNNERS`
    # wires the same names into the scheduler.
    #
    # This is not a deferred-consumer exemption -- there is no debt to repay.
    # It is the gate declaring a known blind spot, and it was earned: the 5.0.0
    # sweep's cascade wave flagged `entity_ingest` as dead in wave 1 and
    # `transcript_capture` in wave 2, and deleting either would have left the
    # gate green while breaking the daemon at runtime.
    "entity_ingest": (
        "LIVE. Dispatched by string, not import: cron_runner.JOB_DEFS job "
        "'entity_ingest' -> `python -m mind_mem.entity_ingest`, wired into "
        "daemon._TASK_RUNNERS. No importer by design."
    ),
    "transcript_capture": (
        "LIVE. Backs cron_runner.JOB_DEFS job 'transcript_scan' (the job name "
        "and module name deliberately differ) and is in the init_workspace "
        "scaffold list. No importer by design."
    ),
    "intel_scan": (
        "LIVE. Reached through `importlib.util.find_spec('mind_mem.intel_scan')` "
        "in apply_engine -- a runtime probe, invisible to an AST import scan."
    ),
    "session_summarizer": (
        "LIVE, by THREE routes this AST scan cannot see, which is why it is "
        "named here rather than left to luck: (1) hooks/session-end.sh runs "
        "`python3 -m mind_mem.session_summarizer` -- a shell caller that has "
        "existed all along; (2) cron_runner.OPT_IN_JOB_DEFS job "
        "'session_summary' -> `python -m mind_mem.session_summarizer`, wired "
        "into daemon._TASK_RUNNERS (OFF by default); (3) bootstrap_corpus "
        "imports write_summary. Only (3) is an import, and bootstrap_corpus is "
        "itself in the reachability baseline -- so without this entry a LIVE "
        "shipped surface's visibility depends on an unwired module staying put."
    ),
    "__main__": "python -m mind_mem entry point",
    "mm_cli": "console_scripts entry point in pyproject.toml",
    "mcp_server": "MCP stdio server entry point",
    "append_only": (
        "T-007. Operator setup helper, invoked from the runbook rather than a "
        "write path. DEFERRED CONSUMER: surface via `mm doctor` -- until then "
        "this is honest shelfware, named here rather than hidden."
    ),
    "novel_term_gate": (
        "Group J. Its only designed consumer is the anticipation-cache "
        "consumer, which is itself an OPEN roadmap item -- the leaf shipped "
        "before the branch. Fable's audit further argues the cache is "
        "architecturally wrong for a governed store (a client-side TTL cache "
        "serves blocks without passing the governed read path, so a block "
        "contradicted since caching is served as truth). If that call stands, "
        "this module should be DELETED, not wired."
    ),
}

# Import forms this scan understands.
_SELF_PKG = "mind_mem"


def _module_name(path: pathlib.Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    return ".".join(rel.parts)


def _imported_names(tree: ast.AST) -> set[str]:
    """Every mind_mem module an AST references by import.

    Covers ``import mind_mem.x``, ``from mind_mem.x import y``, ``from .x
    import y`` and ``from ..pkg.x import y``. Relative levels are resolved by
    the caller, which knows the importing module's own package.
    """
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith(_SELF_PKG + "."):
                    out.add(a.name[len(_SELF_PKG) + 1 :])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                out.add(f"\x00{node.level}\x00{node.module or ''}")
            elif node.module == _SELF_PKG:
                # `from mind_mem import usage_meter` -- the module is EXACTLY
                # the package root, so each imported NAME may itself be a
                # submodule. This form was missed until 2026-08-31: the old
                # test required node.module to start with "mind_mem.", which
                # this never does, so mm_cli's `from mind_mem import
                # usage_meter` / `self_update` read as no reference at all and
                # both modules were deleted out from under a console_script.
                for a in node.names:
                    out.add(a.name)
            elif node.module and node.module.startswith(_SELF_PKG + "."):
                out.add(node.module[len(_SELF_PKG) + 1 :])
                for a in node.names:
                    out.add(f"{node.module[len(_SELF_PKG) + 1 :]}.{a.name}")
    return out


def _resolve(importer: str, raw: str) -> str:
    """Resolve a recorded import against the importing module's package."""
    if not raw.startswith("\x00"):
        return raw
    _, level_s, mod = raw.split("\x00", 2)
    level = int(level_s)
    parts = importer.split(".")[:-1]  # drop the module itself
    base = parts[: len(parts) - (level - 1)] if level > 1 else parts
    return ".".join([*base, mod]) if mod else ".".join(base)


def _entry_point_modules() -> set[str]:
    """Modules reachable as console_scripts, which no import graph can see.

    17 of them, declared in pyproject `[project.scripts]`. Counting these as
    unreachable would give the gate 17 false positives on day one, and a gate
    that cries wolf gets ignored -- which is worse than no gate.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - 3.10
        return set()
    try:
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover
        return set()
    out: set[str] = set()
    for target in data.get("project", {}).get("scripts", {}).values():
        mod = target.split(":", 1)[0]
        if mod.startswith(_SELF_PKG + "."):
            out.add(mod[len(_SELF_PKG) + 1 :])
    return out


# Consumer trees that live OUTSIDE the package but import it. Parsed as
# REFERENCERS only -- never gated as modules themselves.
CONSUMER_TREES = ("benchmarks", "train", "examples")


def _consumer_tree_references() -> set[str]:
    """mind_mem modules imported by benchmarks/, train/ and examples/.

    This gate originally walked only ``src/`` for referencers, so any module
    whose sole caller lives in one of these trees read as unreachable. That
    scope defect nearly cost four modules on the canonical LoCoMo reproduce
    path during the 5.0.0 sweep -- the gate's own blind spot became the
    argument for deleting live code.

    Files here are referencers, never gated modules: nothing in these trees is
    product code, so nothing in them is ever reported as unreachable.

    AST-only, so a string-built ``__import__(f"mind_mem.{name}")`` stays
    invisible. That is deliberate rather than a gap:
    ``benchmarks/local_stack_audit.py`` builds names that way, and it is a
    PARITY LIST that must track the product -- counting it as a referencer
    would let a stale audit list pin modules the product no longer ships.
    """
    out: set[str] = set()
    for tree_name in CONSUMER_TREES:
        tree = ROOT / tree_name
        if not tree.is_dir():
            continue
        for path in tree.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                parsed = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            for raw in _imported_names(parsed):
                if raw.startswith("\x00"):
                    # A relative import inside the consumer tree itself, not
                    # an import of mind_mem. Not resolvable against SRC.
                    continue
                out.add(raw)
                out.add(raw.rsplit(".", 1)[0])
    return out


def _workflow_path_references() -> set[str]:
    """Modules a CI workflow invokes by PATH rather than by import.

    `.github/workflows/ci.yml` runs `python src/mind_mem/check_version.py` as
    the version-consistency gate. No Python file imports it, so an
    import-graph scan of any scope reports it dead -- and the 5.0.0 sweep duly
    deleted it, turning the gate into "can't open file". A YAML job step is a
    caller; it just is not an import.

    Text scan rather than a YAML parse on purpose: the reference can appear in
    a `run:` block, a composite action, or a shell one-liner, and all that
    matters is that the path is named somewhere in the workflow.
    """
    wf = ROOT / ".github" / "workflows"
    if not wf.is_dir():
        return set()
    out: set[str] = set()
    text = "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in sorted(wf.rglob("*.y*ml")))
    for path in SRC.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in text:
            out.add(_module_name(path))
    return out


def scan() -> tuple[list[str], dict[str, int]]:
    modules = {_module_name(p) for p in SRC.rglob("*.py") if "__pycache__" not in p.parts}
    referenced: set[str] = set()
    for p in SRC.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        me = _module_name(p)
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for raw in _imported_names(tree):
            target = _resolve(me, raw)
            if target and target != me:
                referenced.add(target)
                # `from .pkg import mod` references pkg.mod as a module too.
                referenced.add(target.rsplit(".", 1)[0])

    entry_points = _entry_point_modules()
    referenced |= entry_points

    consumer_refs = _consumer_tree_references()
    referenced |= consumer_refs

    workflow_refs = _workflow_path_references()
    referenced |= workflow_refs

    unreachable = sorted(
        m
        for m in modules
        if m not in referenced and m.rsplit(".", 1)[-1] != "__init__" and not any(m == a or m.endswith("." + a) for a in ALLOWLIST)
    )
    return unreachable, {
        "modules": len(modules),
        "referenced": len(referenced),
        "consumer_tree_refs": len(consumer_refs),
        "workflow_path_refs": len(workflow_refs),
        "entry_points": len(entry_points),
    }


BASELINE = ROOT / "scripts" / "reachability_baseline.txt"


def _baseline() -> set[str]:
    """Today's known-unreachable set, frozen so the gate can RATCHET.

    49 modules are unreachable as of 2026-08-31 -- roughly 15% of
    src/mind_mem, tested but never invoked by the product. Failing CI on all
    of them on day one would simply get the gate disabled, so the debt is
    frozen here by name and the gate fails only on ADDITIONS.

    The baseline is a debt register, not an amnesty: entries leave it by
    being wired or deleted, and the file only ever shrinks.
    """
    if not BASELINE.is_file():
        return set()
    out = set()
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line.split("#", 1)[0].strip())
    return out


def _waiting() -> dict[str, str]:
    """Modules deliberately held back, mapped to their stated trigger.

    A line of the form ``module  # waiting: <trigger>`` says a human decided
    this module is not wired YET and named the condition that should flip it.
    That annotation is the whole point: without it a reachability report says
    only "unreachable", and "unreachable" is what got 47 working modules
    deleted on the theory that no caller means no worth. It does not. An
    unwired module with a recorded trigger is a decision; an unwired module
    with no note is an open question. The report must not confuse the two.
    """
    if not BASELINE.is_file():
        return {}
    out: dict[str, str] = {}
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "#" not in line:
            continue
        mod, _, note = line.partition("#")
        note = note.strip()
        if note.lower().startswith("waiting:"):
            out[mod.strip()] = note[len("waiting:") :].strip()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 on any NEW unreachable module")
    ap.add_argument("--update-baseline", action="store_true", help="rewrite the baseline (only ever to shrink it)")
    ap.add_argument("--list", action="store_true", help="print every currently-unreachable module")
    args = ap.parse_args(argv)

    unreachable, stats = scan()
    base = _baseline()
    new_debt = [m for m in unreachable if m not in base]
    fixed = sorted(base - set(unreachable))

    print(f"modules under src/mind_mem: {stats['modules']}")
    print(f"console_script entry points: {stats['entry_points']}")
    print(f"allowlisted (reason recorded): {len(ALLOWLIST)}")
    print(f"unreachable total: {len(unreachable)}  (baseline {len(base)})")
    if fixed:
        print(f"NEWLY REACHABLE (wire or delete confirmed) -- shrink the baseline: {len(fixed)}")
        for m in fixed:
            print(f"  + {m}")
    if args.list:
        waiting = _waiting()
        held = [m for m in sorted(unreachable) if m in waiting]
        open_q = [m for m in sorted(unreachable) if m not in waiting]
        print(f"NOT WIRED, DELIBERATELY WAITING ({len(held)}) -- decided, not debt:")
        for m in held:
            print(f"  ~ {m}  <- flips when: {waiting[m]}")
        print(f"NOT WIRED, NO RECORDED DECISION ({len(open_q)}) -- each needs a call:")
        for m in open_q:
            print(f"  - {m}")
    if new_debt:
        print(f"NEW unreachable modules: {len(new_debt)}")
        for m in new_debt:
            print(f"  ! {m}")

    if args.update_baseline:
        if len(unreachable) > len(base):
            print("refusing to grow the baseline; wire or delete the new module instead", file=sys.stderr)
            return 1
        waiting = _waiting()
        lines = []
        for m in sorted(unreachable):
            note = waiting.get(m)
            lines.append(f"{m}  # waiting: {note}" if note else m)
        BASELINE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"baseline updated: {len(base)} -> {len(unreachable)}")
        return 0

    if new_debt and args.check:
        print(
            "\n::error::NEW unreachable module(s) with no caller. Wire it, delete it, "
            "or add it to ALLOWLIST WITH the reason and the roadmap item that will "
            "consume it. 'No caller, no tick.'",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
