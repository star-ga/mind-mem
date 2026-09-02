# Copyright 2026 STARGA, Inc.
"""Static scanners behind ``test_governed_write_paths``.

Pure AST — no imports of ``mind_mem``, so the invariant is checked
against the *source on disk* rather than against whatever a monkeypatched
runtime happens to expose. Every scanner returns a sorted tuple so
failures are stable and diffable.

Deliberately AST and not grep: ``grep -c '.write_block('`` over this repo
returns two docstring hits (``block_store.py`` lines 680 and 700) that are
not call sites at all, and grep cannot name the enclosing function, which
is the granularity the allowlist needs.
"""

from __future__ import annotations

import ast
import os

SRC_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "mind_mem")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Context managers on ``GovernanceGate`` that open an admission scope.
ADMIT_OPENERS: frozenset[str] = frozenset({"admit_block", "admit_batch", "admit_proposal"})

#: The DELETE-side scopes, kept as a SEPARATE set rather than folded into
#: :data:`ADMIT_OPENERS`. A receipt is not transferable between the two
#: operations — ``require_admission`` refuses a delete receipt spent on a
#: write — so a write surface that opened only a delete scope must still
#: fail the write-side check. Merging the sets would make that pass.
DELETE_ADMIT_OPENERS: frozenset[str] = frozenset({"admit_delete", "admit_delete_batch"})

#: Corpus files that hold recallable blocks (``_recall_constants.CORPUS_FILES``
#: values, by basename). A direct append to one of these mints a block
#: without ever touching ``BlockStore.write_block``.
#:
#: Hand-copied on purpose — this module imports no ``mind_mem``, so the
#: invariant is checked against the source on disk rather than against
#: whatever a monkeypatched runtime exposes. The cost of the copy is
#: drift, and it had already drifted: ``INGEST.md`` joined
#: ``CORPUS_FILES`` in 5.0.1 and was never added here, so for one release
#: a direct writer to ``memory/INGEST.md`` was invisible to
#: ``test_no_unpinned_direct_corpus_writers`` — the scan would have
#: reported a clean tree it never looked at. Nothing wrote that file by
#: literal path (the ``INGEST`` prefix routes through ``write_block``), so
#: the hole was latent rather than exploited.
#:
#: :func:`corpus_basenames_from_source` re-derives this set from
#: ``corpus_registry.CORPUS_TABLE`` by AST — the one definition of the
#: corpus, which ``_recall_constants.CORPUS_FILES`` is now derived from
#: rather than duplicating — and
#: ``test_scanner_corpus_basenames_match_the_registry`` fails the build
#: when the two disagree. The copy stays; silent drift does not.
CORPUS_BASENAMES: frozenset[str] = frozenset(
    {
        "DECISIONS.md",
        "TASKS.md",
        "projects.md",
        "people.md",
        "tools.md",
        "incidents.md",
        "CONTRADICTIONS.md",
        "DRIFT.md",
        "SIGNALS.md",
        "MESSAGES.md",
        "INBOX.md",
        "IMPORTED.md",
        "INGEST.md",
    }
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def corpus_basenames_from_source() -> frozenset[str]:
    """Re-derive :data:`CORPUS_BASENAMES` from ``_recall_constants.py``.

    By AST, not by import, so this module keeps its rule of never
    importing ``mind_mem``: the registry is read as the text on disk.
    Only ``ast.Constant`` values are collected — a computed entry would
    be silently skipped, so the count is asserted by the caller rather
    than trusted here.

    Reads ``corpus_registry.CORPUS_TABLE``, which is where the one
    definition of the corpus now lives. It used to read a
    ``CORPUS_FILES`` dict literal in ``_recall_constants``; that name is
    now DERIVED from this table, so parsing it as a literal failed —
    the guard broke because the code it guards got better. Following the
    data to its source is the fix; loosening the parse to accept a
    computed value would have made the guard unable to see drift at all.
    """
    path = os.path.join(SRC_ROOT, "corpus_registry.py")
    tree = parse(path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign):
            continue
        target = node.target
        if not isinstance(target, ast.Name) or target.id != "CORPUS_TABLE":
            continue
        if not isinstance(node.value, ast.Tuple):
            break
        names: set[str] = set()
        for row in node.value.elts:
            if not isinstance(row, ast.Call) or not row.args:
                continue
            last = row.args[-1]
            if isinstance(last, ast.Constant) and isinstance(last.value, str):
                names.add(os.path.basename(last.value))
        if names:
            return frozenset(names)
        break
    raise AssertionError(f"CORPUS_TABLE is not a tuple of literal rows in {path}; the drift guard cannot read it")


def iter_source_files(root: str = SRC_ROOT) -> tuple[str, ...]:
    """Every ``.py`` under *root*, deterministic order, ``__pycache__`` skipped."""
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                found.append(os.path.join(dirpath, name))
    return tuple(sorted(found))


def parse(path: str) -> ast.Module:
    with open(path, "r", encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=path)


def relpath(path: str) -> str:
    return os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")


def qualnames(tree: ast.AST) -> dict[ast.AST, str]:
    """Map every node to the dotted ``Class.func`` name enclosing it."""
    out: dict[ast.AST, str] = {}

    def walk(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nested = stack + [child.name]
                out[child] = ".".join(nested)
                walk(child, nested)
            else:
                out[child] = ".".join(stack)
                walk(child, stack)

    walk(tree, [])
    return out


def called_name(node: ast.Call) -> str | None:
    """``foo`` for ``x.foo(...)`` and ``foo(...)``; ``None`` otherwise."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


# ---------------------------------------------------------------------------
# Scan A — raw write_block call sites
# ---------------------------------------------------------------------------


def find_write_block_calls(tree: ast.AST, rel: str) -> list[tuple[str, str, int]]:
    """``(file, enclosing qualname, lineno)`` for every ``write_block`` call."""
    names = qualnames(tree)
    hits = [
        (rel, names.get(node, "") or "<module>", node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and called_name(node) == "write_block"
    ]
    return sorted(hits, key=lambda h: (h[0], h[2]))


def scan_write_block_calls(files: tuple[str, ...]) -> tuple[tuple[str, str, int], ...]:
    hits: list[tuple[str, str, int]] = []
    for path in files:
        hits.extend(find_write_block_calls(parse(path), relpath(path)))
    return tuple(sorted(hits))


# ---------------------------------------------------------------------------
# Scan A2 — does a function open an admission scope?
# ---------------------------------------------------------------------------


def function_node(tree: ast.AST, qualname: str) -> ast.AST | None:
    names = qualnames(tree)
    for node, name in names.items():
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and name == qualname:
            return node
    return None


def opens_admission(func: ast.AST, openers: frozenset[str] = ADMIT_OPENERS) -> bool:
    """True when *func* calls one of *openers* in its own body.

    Defaults to the write-side scopes, so every existing caller keeps the
    check it had. A delete surface passes :data:`DELETE_ADMIT_OPENERS`
    explicitly rather than widening the default.
    """
    return any(isinstance(n, ast.Call) and called_name(n) in openers for n in ast.walk(func))


def calls_require_admission(func: ast.AST) -> bool:
    return any(isinstance(n, ast.Call) and called_name(n) == "require_admission" for n in ast.walk(func))


def binds_status_to_require_admission(func: ast.AST) -> bool:
    """True when every ``require_admission`` call in *func* passes ``status=``.

    The tier check in ``require_admission`` can only refuse a status
    escalation if the write surface hands it the status. A backend that
    calls ``require_admission(block_id)`` alone still gets its receipt
    checked, but its tier row is unenforceable — so that omission is a
    build failure rather than a silent downgrade.
    """
    calls = [n for n in ast.walk(func) if isinstance(n, ast.Call) and called_name(n) == "require_admission"]
    if not calls:
        return False
    return all(any(kw.arg == "status" for kw in call.keywords) for call in calls)


# ---------------------------------------------------------------------------
# Scan B — write_block implementations
# ---------------------------------------------------------------------------


def scan_write_block_defs(files: tuple[str, ...]) -> tuple[tuple[str, str, int, bool], ...]:
    """``(file, qualname, lineno, enforces)`` for every ``def write_block``."""
    out: list[tuple[str, str, int, bool]] = []
    for path in files:
        tree = parse(path)
        names = qualnames(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "write_block":
                out.append((relpath(path), names.get(node, node.name), node.lineno, calls_require_admission(node)))
    return tuple(sorted(out))


def scan_write_block_status_binding(files: tuple[str, ...]) -> tuple[tuple[str, str, int, bool], ...]:
    """``(file, qualname, lineno, binds_status)`` for every ``def write_block``."""
    out: list[tuple[str, str, int, bool]] = []
    for path in files:
        tree = parse(path)
        names = qualnames(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "write_block":
                out.append((relpath(path), names.get(node, node.name), node.lineno, binds_status_to_require_admission(node)))
    return tuple(sorted(out))


# ---------------------------------------------------------------------------
# Scan C — the receipt contextvar must not be reachable outside its module
# ---------------------------------------------------------------------------


def scan_contextvar_references(files: tuple[str, ...], symbol: str) -> tuple[tuple[str, int], ...]:
    out: list[tuple[str, int]] = []
    for path in files:
        tree = parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == symbol:
                out.append((relpath(path), node.lineno))
            elif isinstance(node, ast.Attribute) and node.attr == symbol:
                out.append((relpath(path), node.lineno))
    return tuple(sorted(out))


# ---------------------------------------------------------------------------
# Scan D — direct corpus-file appends (block minting that skips write_block)
# ---------------------------------------------------------------------------


def _write_target(node: ast.Call) -> ast.expr | None:
    """Path argument of a write-mode ``open`` / ``_atomic_write`` call."""
    name = called_name(node)
    if name == "_atomic_write":
        return node.args[0] if node.args else None
    if name != "open":
        return None
    mode = None
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        mode = node.args[1].value
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            mode = keyword.value.value
    if isinstance(mode, str) and any(char in mode for char in "aw+"):
        return node.args[0] if node.args else None
    return None


def _assignments(scope: ast.AST) -> dict[str, list[ast.expr]]:
    """Name -> assigned expressions, over the whole subtree of *scope*."""
    env: dict[str, list[ast.expr]] = {}
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    env.setdefault(target.id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            env.setdefault(node.target.id, []).append(node.value)
    return env


def _module_constants(tree: ast.Module) -> dict[str, list[ast.expr]]:
    """Module-level assignments only.

    Deliberately NOT ``_assignments(tree)``: that walks into every
    function body, so a local ``path = ...`` in one function would resolve
    a same-named local in another. Cross-function bleed of exactly that
    shape made an earlier draft of this scanner report
    ``intel_scan.save_intel_state`` (which writes ``intel-state.json``)
    as a corpus write.
    """
    env: dict[str, list[ast.expr]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    env.setdefault(target.id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            env.setdefault(node.target.id, []).append(node.value)
    return env


def _enclosing_function(tree: ast.AST) -> dict[ast.AST, ast.AST | None]:
    """Map every node to the innermost ``FunctionDef`` containing it."""
    out: dict[ast.AST, ast.AST | None] = {}

    def walk(node: ast.AST, current: ast.AST | None) -> None:
        for child in ast.iter_child_nodes(node):
            nxt = child if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else current
            out[child] = nxt
            walk(child, nxt)

    walk(tree, None)
    return out


def _const_strings(expr: ast.expr | None, env: dict[str, list[ast.expr]], helpers: dict[str, ast.AST], depth: int = 0) -> set[str]:
    """String constants an expression can be built from (bounded depth)."""
    if expr is None or depth > 5:
        return set()
    if isinstance(expr, ast.Constant):
        return {expr.value} if isinstance(expr.value, str) else set()
    if isinstance(expr, ast.Name):
        out: set[str] = set()
        for value in env.get(expr.id, []):
            out |= _const_strings(value, env, helpers, depth + 1)
        return out
    if isinstance(expr, ast.JoinedStr):
        return set().union(*(_const_strings(v, env, helpers, depth + 1) for v in expr.values)) if expr.values else set()
    if isinstance(expr, ast.FormattedValue):
        return _const_strings(expr.value, env, helpers, depth + 1)
    if isinstance(expr, ast.BinOp):
        return _const_strings(expr.left, env, helpers, depth + 1) | _const_strings(expr.right, env, helpers, depth + 1)
    if isinstance(expr, ast.Call):
        out = set()
        for arg in expr.args:
            out |= _const_strings(arg, env, helpers, depth + 1)
        callee = helpers.get(called_name(expr) or "")
        if callee is not None:
            callee_env = _assignments(callee)
            for node in ast.walk(callee):
                if isinstance(node, ast.Return):
                    out |= _const_strings(node.value, callee_env, {}, depth + 1)
        return out
    if isinstance(expr, ast.Subscript):
        return _const_strings(expr.value, env, helpers, depth + 1)
    return set()


def scan_corpus_writes(files: tuple[str, ...]) -> tuple[tuple[str, str, int, str], ...]:
    """``(file, qualname, lineno, corpus basename)`` for direct corpus appends.

    Resolves the path expression through local/module assignments and one
    hop into a same-module helper's ``return``. It therefore CANNOT see a
    path that arrives as a function parameter or a loop variable over a
    mapping (``compaction.py`` builds its target that way) — those are
    covered by the ``write_block`` scans or by nothing, and this scanner
    does not claim otherwise.
    """
    out: list[tuple[str, str, int, str]] = []
    for path in files:
        tree = parse(path)
        rel = relpath(path)
        names = qualnames(tree)
        module_env = _module_constants(tree)
        enclosing = _enclosing_function(tree)
        helpers = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        local_envs: dict[ast.AST, dict[str, list[ast.expr]]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = _write_target(node)
            if target is None:
                continue
            scope = enclosing.get(node)
            if scope is None:
                env = module_env
            else:
                if scope not in local_envs:
                    merged = {k: list(v) for k, v in module_env.items()}
                    for key, values in _assignments(scope).items():
                        merged.setdefault(key, []).extend(values)
                    local_envs[scope] = merged
                env = local_envs[scope]
            consts = _const_strings(target, env, helpers)
            matched = sorted(b for b in CORPUS_BASENAMES if any(v == b or v.endswith("/" + b) for v in consts))
            if matched:
                out.append((rel, names.get(node, "") or "<module>", node.lineno, ",".join(matched)))
    return tuple(sorted(out))
