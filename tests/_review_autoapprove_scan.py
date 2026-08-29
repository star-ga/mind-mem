# Copyright 2026 STARGA, Inc.
"""Static scanner behind ``test_review_no_autoapprove``.

Pure AST over the ``review_*`` sources on disk — no ``mind_mem`` import,
so the invariant is checked against the shipped source rather than
against whatever a runtime happens to expose.

Deliberately AST and not grep: the docstrings in these modules *say*
"never auto-approve", so a grep for ``auto_approve`` matches the very
prose that forbids it. Only an AST walk can tell an identifier from a
sentence, and only an AST walk can name the enclosing function of a
call site — which is the granularity the approval choke point needs.
"""

from __future__ import annotations

import ast
import os
import re

SRC_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "mind_mem")

#: Every module that can reach an approval.
REVIEW_MODULES: tuple[str, ...] = (
    "review_batch.py",
    "review_cli.py",
    "review_evidence.py",
    "review_metrics.py",
    "review_preview.py",
    "review_queue.py",
    "review_render.py",
    "review_session.py",
)

#: The only function permitted to invoke the governed approval tool.
APPROVAL_CHOKE_POINT = "governed_approve"

#: The only function permitted to invoke the governed rejection tool.
REJECTION_CHOKE_POINT = "governed_reject"

#: Names of the governance entry points that mutate the corpus.
GOVERNED_WRITE_CALLS: frozenset[str] = frozenset({"approve_apply", "reject_proposal", "apply_proposal"})

#: Identifier shapes that describe approving without an operator present.
BANNED_IDENTIFIER_RE = re.compile(
    r"auto_?approve|approve_?all_?(low|safe|auto)|autoapply|auto_?apply|"
    r"unattended|no_?confirm|skip_?review|silent_?approve|trust_?level|risk_?threshold",
    re.IGNORECASE,
)

#: Risk vocabulary. A review front end that branches on these is one
#: refactor away from "auto-approve the low-risk ones".
RISK_LITERALS: frozenset[str] = frozenset({"low", "medium", "high"})


def module_paths() -> tuple[str, ...]:
    return tuple(os.path.join(SRC_ROOT, name) for name in REVIEW_MODULES)


def _parse(path: str) -> ast.Module:
    with open(path, encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=path)


def _enclosing_functions(tree: ast.Module) -> dict[ast.AST, str]:
    """Map every node to the name of the function that lexically contains it."""
    owner: dict[ast.AST, str] = {}

    def walk(node: ast.AST, current: str) -> None:
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else current
            owner[child] = name
            walk(child, name)

    walk(tree, "<module>")
    return owner


def governed_call_sites() -> tuple[tuple[str, str, str], ...]:
    """Every call to a governed write, as ``(module, enclosing_function, callee)``."""
    found: list[tuple[str, str, str]] = []
    for path in module_paths():
        if not os.path.isfile(path):
            continue
        tree = _parse(path)
        owner = _enclosing_functions(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = _callee_name(node.func)
            if callee in GOVERNED_WRITE_CALLS:
                found.append((os.path.basename(path), owner.get(node, "<module>"), callee))
    return tuple(sorted(found))


def banned_identifiers() -> tuple[tuple[str, str], ...]:
    """Identifiers matching an auto-approval shape, as ``(module, identifier)``.

    Docstrings, comments and plain strings are ignored on purpose: prose
    that forbids auto-approval must not trip the check that enforces it.
    """
    found: list[tuple[str, str]] = []
    for path in module_paths():
        if not os.path.isfile(path):
            continue
        module = os.path.basename(path)
        for node in ast.walk(_parse(path)):
            for name in _identifiers(node):
                if BANNED_IDENTIFIER_RE.search(name):
                    found.append((module, name))
    return tuple(sorted(set(found)))


def risk_branches() -> tuple[tuple[str, str, int], ...]:
    """Comparisons against a risk literal, as ``(module, function, lineno)``."""
    found: list[tuple[str, str, int]] = []
    for path in module_paths():
        if not os.path.isfile(path):
            continue
        module = os.path.basename(path)
        tree = _parse(path)
        owner = _enclosing_functions(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            for operand in [node.left, *node.comparators]:
                if isinstance(operand, ast.Constant) and operand.value in RISK_LITERALS:
                    found.append((module, owner.get(node, "<module>"), node.lineno))
    return tuple(sorted(set(found)))


def _identifiers(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (node.attr,)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return (node.name,)
    if isinstance(node, ast.arg):
        return (node.arg,)
    if isinstance(node, ast.keyword) and node.arg:
        return (node.arg,)
    return ()


def _callee_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""
