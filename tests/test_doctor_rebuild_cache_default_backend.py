"""``mm doctor --rebuild-cache`` must not crash on the DEFAULT markdown backend.

REGRESSION. `_cmd_doctor` bound `pg_only` only inside its
`if store_class == "PostgresBlockStore":` arm and then read it again ~130 lines later in
the `--rebuild-cache` block. On the default markdown/SQLite workspace that arm does not
run, so the later read raised ``UnboundLocalError`` — on the common path, and the one a
Postgres-focused test would never reach.

The test is STATIC on purpose. Driving the real command needs a workspace, a backend and
a cache, and each of those is a way for the test to pass for a reason unrelated to the
defect. The defect is a scope error, and scope is decidable from the source: assert that
the first binding of ``pg_only`` precedes every read of it inside the function. That
assertion fails on the pre-fix source and cannot be satisfied by accident.
"""

from __future__ import annotations

import ast
import pathlib

import mind_mem.mm_cli as mm_cli


def _doctor_function() -> ast.FunctionDef:
    src = pathlib.Path(mm_cli.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    candidates = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_cmd_doctor"]
    assert candidates, "_cmd_doctor not found — this test's premise is gone, not satisfied"
    return max(candidates, key=lambda n: n.lineno)


def test_pg_only_is_bound_unconditionally() -> None:
    """The binding must be UNCONDITIONAL, not merely earlier in the file.

    An earlier version of this test compared line numbers and PASSED on the broken
    source — the conditional binding inside the Postgres arm still precedes the read, so
    line order says nothing about whether the name is bound on the path that reaches it.
    The property that matters is that `pg_only` is assigned among the function's
    straight-line statements, so every path reaches a binding. Caught by reverting the
    fix and watching this test fail to notice.
    """
    fn = _doctor_function()

    reads = sorted(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == "pg_only")
    all_assigns = sorted(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == "pg_only")

    # Positive controls: a name nobody writes or nobody reads makes this vacuous.
    assert all_assigns, "pg_only is never assigned — the test can no longer see the defect"
    assert reads, "pg_only is never read — the test can no longer see the defect"

    # Straight-line statements of the function body: depth 0, so unconditional.
    # An `AnnAssign` with NO value binds nothing: `pg_only: set` is a type declaration and
    # leaves the name unbound, so the runtime still raises. The first version of this list
    # accepted it — verified by mutation, which passed while the mutated code still crashed.
    # Require an actual value.
    top_level_binds = [
        stmt.lineno
        for stmt in fn.body
        if (isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pg_only" for t in stmt.targets))
        or (
            isinstance(stmt, ast.AnnAssign) and stmt.value is not None and isinstance(stmt.target, ast.Name) and stmt.target.id == "pg_only"
        )
    ]

    assert top_level_binds, (
        f"pg_only is assigned only inside conditional branches (lines {all_assigns}) but "
        f"read at lines {reads}. On the default markdown backend the branch that binds it "
        "does not run, so `mm doctor --rebuild-cache` raises UnboundLocalError. It needs an "
        "unconditional binding in the function body."
    )
    assert min(top_level_binds) < min(reads), (
        f"pg_only's unconditional binding (line {min(top_level_binds)}) must precede its first read (line {min(reads)})"
    )


def test_supersedes_is_reachable_from_the_lineage_cli() -> None:
    """A supported edge kind must be selectable by an operator.

    ``supersedes`` is weighted by the lineage library and treated as a
    staleness-firing contradiction kind, but the ``mm lineage flag --kind`` parser
    omitted it — so the capability existed and the surface did not.
    """
    src = pathlib.Path(mm_cli.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)

    kind_choices: tuple[str, ...] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        names = {kw.arg for kw in node.keywords}
        if not {"choices", "help"} <= names:
            continue
        args = [a for a in node.args if isinstance(a, ast.Constant) and a.value == "--kind"]
        if not args:
            continue
        for kw in node.keywords:
            if kw.arg == "choices" and isinstance(kw.value, ast.Tuple):
                vals = tuple(e.value for e in kw.value.elts if isinstance(e, ast.Constant))
                if "contradicts" in vals:
                    kind_choices = vals

    assert kind_choices is not None, (
        "no `--kind` argument with a choices tuple containing 'contradicts' was found — "
        "the parser moved, so this test cannot see the surface it guards"
    )
    assert "supersedes" in kind_choices, (
        f"`mm lineage flag --kind` offers {kind_choices!r}; 'supersedes' is a supported edge kind and must be selectable"
    )
