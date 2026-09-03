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

#: The ONE function in ``src/`` allowed to open a store restore, as the
#: ``(file, qualname)`` pair :func:`scan_restore_calls` reports.
#:
#: A pair and not a bare name: ``restore_snapshot`` is a module-level
#: function so its qualname carries no module prefix, and matching on the
#: name alone would accept a second ``restore_snapshot`` added to any other
#: file. The file is half the identity of the rule.
#:
#: ``apply_engine.restore_snapshot`` opens the ``admit_batch`` scope that
#: records the manifest digest, the reinstated ids and the withdrawn ids,
#: and then calls the store. Every other ``.restore(`` in ``src/`` is a
#: store forwarding to the store underneath it — those are listed in
#: :data:`RESTORE_DELEGATES`, and anything outside both sets is a new
#: door into the most destructive operation the product has.
RESTORE_SEAM_OPENER: tuple[str, str] = ("src/mind_mem/apply_engine.py", "restore_snapshot")

#: ``.restore(`` calls that forward to a store one layer down rather than
#: opening a scope: the encryption wrapper, the replica adapter and the
#: shard fan-out. Each is a ``BlockStore`` in its own right, so each is
#: itself gated (or delegates to a gated store) — they are sanctioned
#: *callers*, never sanctioned *openers*.
RESTORE_DELEGATES: frozenset[tuple[str, str]] = frozenset(
    {
        ("src/mind_mem/block_store_encrypted.py", "EncryptedBlockStore.restore"),
        ("src/mind_mem/block_store_postgres_replica.py", "ReplicatedPostgresBlockStore.restore"),
        ("src/mind_mem/storage/sharded_pg.py", "ShardedPostgresBlockStore.restore"),
    }
)

#: ``def restore`` implementations that legitimately do not call
#: ``require_restore_admission`` in their own body.
#:
#: Exactly one: ``BlockStore.restore`` is the Protocol declaration, an
#: ellipsis body that runs on no path and has no ``snap_dir`` to check.
#:
#: The replica adapter and the shard fan-out were exempt here until they
#: checked at their own seam. Delegation left the refusal *inherited* —
#: correct only while every inner store happened to enforce — and
#: measured, that was not equivalent: over an inner store that does not
#: check, ``ReplicatedPostgresBlockStore.restore`` and
#: ``ShardedPostgresBlockStore.restore`` both returned normally with no
#: scope open, while ``EncryptedBlockStore.restore`` over the same inner
#: store raised and never reached it. The shard fan-out also answered
#: "is this my snapshot?" (``BlockStoreError``) to a caller it had not
#: authorised. Both now call ``require_restore_admission`` first and
#: still delegate the work, so this set is the Protocol stub alone and
#: every future ``def restore`` in ``src/`` has to enforce or fail the
#: build.
RESTORE_ENFORCEMENT_EXEMPT: frozenset[tuple[str, str]] = frozenset(
    {
        ("src/mind_mem/block_store.py", "BlockStore.restore"),
    }
)

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


#: Workspace subdirectories whose ``.md`` files the block store READS.
#:
#: ``MarkdownBlockStore._discover_files`` lists each of these directories
#: (``os.listdir`` — one level, never a walk) and takes every ``.md`` in
#: it. So a ``.md`` written directly into one of these is in the store's
#: read set no matter what it is called, and :data:`CORPUS_BASENAMES` —
#: which only knows the thirteen NAMED corpus files — cannot see it.
#:
#: That gap was not hypothetical. ``dream_cycle._create_entity_file``
#: wrote ``entities/<PREFIX>-<slug>.md``; measured on a fresh workspace,
#: the file was in ``list_blocks`` (so it inflated ``GET /status``) with
#: ``get_by_id`` ``None``, evidence chain +0 and hash chain +0 — an
#: ungoverned write into the store's own read set that scan D was
#: structurally unable to report.
#:
#: Hand-copied for the same reason as :data:`CORPUS_BASENAMES` (this
#: module imports no ``mind_mem``), and pinned against the registry by
#: ``test_scanner_corpus_dirs_match_the_registry``.
CORPUS_DIRS: frozenset[str] = frozenset(
    {
        "decisions",
        "tasks",
        "entities",
        "intelligence",
    }
)

#: Stands in for a path segment the scanner cannot resolve to a constant —
#: an interpolated workspace root, a parameter, a computed name. Chosen
#: because it cannot occur in a real path, so it never collides with a
#: literal segment.
UNRESOLVED = "\x00"


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


def corpus_dirs_from_source() -> frozenset[str]:
    """Re-derive :data:`CORPUS_DIRS` from ``corpus_registry.py``.

    By AST for the same reason as :func:`corpus_basenames_from_source`:
    this module never imports ``mind_mem``, so the registry is read as the
    text on disk.

    ``CORPUS_DIRS`` is what ``MarkdownBlockStore._discover_files`` lists,
    so a directory added there joins the store's read set immediately. The
    scanner has to learn about it in the same commit or it goes back to
    being blind in exactly one more place.
    """
    path = os.path.join(SRC_ROOT, "corpus_registry.py")
    tree = parse(path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign):
            continue
        target = node.target
        if not isinstance(target, ast.Name) or target.id != "CORPUS_DIRS":
            continue
        if not isinstance(node.value, ast.Tuple):
            break
        names = {e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
        if len(names) == len(node.value.elts):
            return frozenset(names)
        break
    raise AssertionError(f"CORPUS_DIRS is not a tuple of string literals in {path}; the drift guard cannot read it")


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


def conditional_calls(func: ast.AST) -> set[int]:
    """``id()`` of every ``Call`` sitting inside a conditional expression.

    "Inside" means either arm of an :class:`ast.IfExp` — ``A if cond else
    B``. Both arms count: a scope in the ``body`` is skipped when the test
    is false, and one in the ``orelse`` when it is true, so neither is a
    scope the function always opens.
    """
    out: set[int] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.IfExp):
            continue
        for branch in (node.body, node.orelse):
            for child in ast.walk(branch):
                if isinstance(child, ast.Call):
                    out.add(id(child))
    return out


def opens_admission(func: ast.AST, openers: frozenset[str] = ADMIT_OPENERS) -> bool:
    """True when *func* UNCONDITIONALLY calls one of *openers* in its body.

    Defaults to the write-side scopes, so every existing caller keeps the
    check it had. A delete surface passes :data:`DELETE_ADMIT_OPENERS`
    explicitly rather than widening the default.

    **An opener that is one arm of a conditional expression does not
    count**, and that clause is the whole difference between this scanner
    and the one that shipped in 5.0.2. ``capture.append_signals`` held::

        _scope = gate.admit_batch(...) if _gate is not None else contextlib.nullcontext()
        with _scope, open(signals_path, "a", encoding="utf-8") as f:

    and a plain "does ``admit_batch`` appear anywhere in this function"
    walk answered *yes* — so the allowlist entry claiming the function ran
    under an admission was checked, passed, and was false. Measured: with
    ``memory/hash_chain_v2.db`` replaced by a directory the gate could not
    be constructed, the conditional substituted a no-op scope, the signal
    block landed in ``intelligence/SIGNALS.md`` and both ledgers stayed at
    +0, with all 18 tests in ``test_governed_write_paths`` green.

    The rule is deliberately blunt: ``X if c else Y`` fails even when BOTH
    arms are openers. A scope the source cannot be read as always opening
    is one this scanner reports as not opened — erring toward "ungated" is
    the only direction that is safe for a checker whose job is to refuse a
    fail-open write path, and the remedy (open the scope unconditionally,
    or fail before you reach it) is the shape every sanctioned caller
    already has.
    """
    conditional = conditional_calls(func)
    return any(isinstance(n, ast.Call) and called_name(n) in openers and id(n) not in conditional for n in ast.walk(func))


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


def calls_require_restore_admission(func: ast.AST) -> bool:
    """True when *func* calls ``require_restore_admission`` in its body.

    The restore-side twin of :func:`calls_require_admission`. Read as a
    call and not as a name, so an import that is never invoked does not
    count as enforcement — "imported" is not "wired".
    """
    return any(isinstance(n, ast.Call) and called_name(n) == "require_restore_admission" for n in ast.walk(func))


# ---------------------------------------------------------------------------
# Scan E — the restore seam
# ---------------------------------------------------------------------------


def scan_restore_calls(files: tuple[str, ...]) -> tuple[tuple[str, str, int], ...]:
    """``(file, enclosing qualname, lineno)`` for every ``.restore(`` call.

    Attribute calls only: ``called_name`` returns ``restore`` for both
    ``store.restore(x)`` and a bare ``restore(x)``, and only the first is
    a store seam, so bare ``ast.Name`` calls are excluded here. Nothing in
    ``src/`` currently calls a module-level ``restore``; if something
    does, it is not the store surface this scan is about.
    """
    hits: list[tuple[str, str, int]] = []
    for path in files:
        tree = parse(path)
        names = qualnames(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "restore":
                continue
            hits.append((relpath(path), names.get(node, "") or "<module>", node.lineno))
    return tuple(sorted(hits))


def scan_restore_defs(files: tuple[str, ...]) -> tuple[tuple[str, str, int, bool], ...]:
    """``(file, qualname, lineno, enforces)`` for every ``def restore``.

    Matched on the method name alone, so ``restore_workspace`` and
    ``restore_header_gaps`` are not in scope — they are not the
    ``BlockStore`` seam, and folding them in would make the enforcement
    allowlist a list of unrelated exceptions.
    """
    out: list[tuple[str, str, int, bool]] = []
    for path in files:
        tree = parse(path)
        names = qualnames(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "restore":
                out.append((relpath(path), names.get(node, node.name), node.lineno, calls_require_restore_admission(node)))
    return tuple(sorted(out))


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


#: Bound on how far :func:`_path_templates` will follow assignments and
#: helper returns. Mirrors the depth cap in :func:`_const_strings`.
_PATH_DEPTH = 5

#: Bound on how many alternative templates one expression may produce. A
#: name with several assignments multiplies out, and a scanner that can be
#: made to hang on a pathological source file is a scanner someone will
#: delete.
_PATH_FANOUT = 32


def _join_templates(parts: list[set[str]]) -> set[str]:
    """Cartesian product of *parts*, joined with ``/``, fanout-capped."""
    out = {""}
    for part in parts:
        if not part:
            return set()
        out = {(left + "/" + right if left else right) for left in out for right in part}
        if len(out) > _PATH_FANOUT:
            return set(sorted(out)[:_PATH_FANOUT])
    return out


def _concat_templates(parts: list[set[str]]) -> set[str]:
    """Cartesian product of *parts*, concatenated with no separator."""
    out = {""}
    for part in parts:
        if not part:
            return set()
        out = {left + right for left in out for right in part}
        if len(out) > _PATH_FANOUT:
            return set(sorted(out)[:_PATH_FANOUT])
    return out


def _path_templates(
    expr: ast.expr | None,
    env: dict[str, list[ast.expr]],
    helpers: dict[str, ast.AST],
    depth: int = 0,
) -> set[str]:
    """Path *shapes* an expression can denote, with unknowns as :data:`UNRESOLVED`.

    The difference from :func:`_const_strings`, and the reason both exist:
    ``_const_strings`` returns an unordered BAG of fragments, which is
    enough to ask "does a named corpus file appear in here" and useless
    for asking "where in the path does this segment sit". Measured on this
    tree, the bag for ``conflict_resolver.generate_resolution_proposals``
    is ``{'RESOLUTIONS_PROPOSED.md', 'intelligence', 'proposed'}`` and the
    bag for ``dream_cycle._create_entity_file`` is ``{'.md', 'entities'}``
    — from the bags alone, ``intelligence/proposed/X.md`` (nested, NOT in
    the store's read set) and ``entities/X.md`` (directly inside it, read
    and parsed) are indistinguishable. Ordering them is what lets scan D
    flag the second without also flagging the first.

    ``os.path.join(a, b)`` contributes a separator between its arguments;
    an f-string or ``+`` does not. An unresolvable piece becomes
    :data:`UNRESOLVED` rather than being dropped, so it still occupies its
    position and cannot let two literals collapse together.

    Returns an empty set when nothing is derivable.
    """
    if expr is None or depth > _PATH_DEPTH:
        return set()
    if isinstance(expr, ast.Constant):
        return {expr.value} if isinstance(expr.value, str) else {UNRESOLVED}
    if isinstance(expr, ast.Name):
        values = env.get(expr.id, [])
        if not values:
            return {UNRESOLVED}
        out: set[str] = set()
        for value in values:
            out |= _path_templates(value, env, helpers, depth + 1)
        return out or {UNRESOLVED}
    if isinstance(expr, ast.JoinedStr):
        return _concat_templates([_path_templates(v, env, helpers, depth + 1) for v in expr.values])
    if isinstance(expr, ast.FormattedValue):
        # The VALUE of an interpolation is deliberately not resolved: an
        # f-string splices text mid-segment, so resolving it could fuse a
        # literal directory name onto a neighbouring segment and move it.
        return {UNRESOLVED}
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        return _concat_templates(
            [
                _path_templates(expr.left, env, helpers, depth + 1),
                _path_templates(expr.right, env, helpers, depth + 1),
            ]
        )
    if isinstance(expr, ast.Call):
        name = called_name(expr)
        if name == "join":
            return _join_templates([_path_templates(a, env, helpers, depth + 1) for a in expr.args])
        callee = helpers.get(name or "")
        if callee is not None:
            callee_env = _assignments(callee)
            out = set()
            for node in ast.walk(callee):
                if isinstance(node, ast.Return):
                    out |= _path_templates(node.value, callee_env, {}, depth + 1)
            return out or {UNRESOLVED}
        return {UNRESOLVED}
    return {UNRESOLVED}


def corpus_dir_hit(template: str, corpus_dirs: frozenset[str]) -> str | None:
    """The corpus directory *template* writes a ``.md`` DIRECTLY into, if any.

    Encodes ``MarkdownBlockStore._discover_files`` rather than a heuristic:
    that method calls ``os.listdir`` on each corpus directory — one level,
    never a walk — and takes every name ending ``.md``. So a path is in the
    store's read set exactly when its last segment ends ``.md`` and the
    segment before it is a corpus directory.

    The corpus directory must additionally sit at the path root or directly
    under an :data:`UNRESOLVED` segment, because the workspace root is
    always an unresolved value at scan time. Without that clause
    ``namespaces.init_multi_agent_workspace`` — which writes
    ``shared/intelligence/LEDGER.md``, a *different* tree that the store
    never lists — would be reported, and a scanner that cries wolf is one
    an allowlist quietly grows to silence.
    """
    segments = [seg for seg in template.replace("\\", "/").split("/") if seg not in ("", ".")]
    if len(segments) < 2:
        return None
    if not segments[-1].endswith(".md"):
        return None
    parent = segments[-2]
    if parent not in corpus_dirs:
        return None
    grandparent = segments[-3] if len(segments) >= 3 else None
    if grandparent is not None and UNRESOLVED not in grandparent:
        return None
    return parent


def scan_corpus_writes(files: tuple[str, ...]) -> tuple[tuple[str, str, int, str], ...]:
    """``(file, qualname, lineno, what matched)`` for direct corpus writes.

    Two rules, unioned, because the corpus has two shapes:

    * **by name** — the path names one of the thirteen files in
      :data:`CORPUS_BASENAMES`; and
    * **by location** — the path is a ``.md`` written DIRECTLY into one of
      :data:`CORPUS_DIRS`, whatever it is called
      (:func:`corpus_dir_hit`).

    The second rule is the one added in 5.0.2, and it exists because the
    first cannot be made to cover the store's read set. ``_discover_files``
    lists every ``.md`` in a corpus directory, so the set of files the store
    reads is open-ended while ``CORPUS_BASENAMES`` is a fixed thirteen. A
    writer that minted its own filename was therefore invisible *by
    construction*, not by oversight — which is precisely what
    ``dream_cycle._create_entity_file`` did with
    ``entities/<PREFIX>-<slug>.md``.

    Resolves the path expression through local/module assignments and one
    hop into a same-module helper's ``return``. It therefore CANNOT see a
    path that arrives as a function parameter or a loop variable over a
    mapping (``compaction.py`` builds its target that way), and it cannot
    see one assembled inside a helper from that helper's own parameters
    (``apply_engine`` resolves its receipt path through ``_safe_resolve``
    that way). Those are covered by the ``write_block`` scans or by
    nothing, and this scanner does not claim otherwise.
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
            if not matched:
                dirs = {hit for tpl in _path_templates(target, env, helpers) if (hit := corpus_dir_hit(tpl, CORPUS_DIRS))}
                matched = sorted(f"{d}/*.md" for d in dirs)
            if matched:
                out.append((rel, names.get(node, "") or "<module>", node.lineno, ",".join(matched)))
    return tuple(sorted(out))
