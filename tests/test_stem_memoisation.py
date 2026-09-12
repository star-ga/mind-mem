"""`_stem` is memoised — a pure function called 124,203 times per 15 recalls.

PROFILED 2026-09-11, after the corpus-parse cache removed the previous 81%
(PERF-PLAN.md P1b). The new top cost is tokenisation:

    _recall_tokenization._stem     124,203 calls / 0.632s cumulative
    str.endswith                 2,328,980 calls / 0.253s

~19 `endswith` per `_stem` call, which is what a suffix-rule stemmer does. The words
themselves repeat heavily across a corpus, so almost all of that work is recomputation.

Memoising is only safe because `_stem` is PURE — one `str` in, one `str` out, no clock, no
I/O, no mutable state — and that purity is what these tests pin. The cache is the easy part;
the risk is that a later edit makes the function context-dependent and the memo then serves
a stale answer forever, which is silent and corpus-wide.
"""

from __future__ import annotations

import ast
import pathlib

import mind_mem._recall_tokenization as _tok_module
from mind_mem._recall_tokenization import _stem


def test_stemming_is_stable_under_repetition():
    """The property the memo relies on: same input, same output, always."""
    for word in ("running", "bodies", "management", "attention", "ies", "a"):
        assert _stem(word) == _stem(word)


def test_known_reductions_still_hold():
    """Memoisation must not change WHAT the stemmer returns. Pinned against the rules
    the docstring claims, so a cache that accidentally shadowed the logic would fail."""
    assert _stem("bodies") == "body"
    assert _stem("happiness") == "happi" or _stem("happiness") == "happiness"
    assert _stem("management") == "manage"
    assert _stem("attention") == "attent"
    assert _stem("cat") == "cat"          # <= 3 chars: untouched
    assert _stem("") == ""                # empty is not a crash


def test_it_is_actually_memoised():
    """POSITIVE CONTROL. Every test above passes with no cache at all, so without this
    the file proves only that the stemmer is correct."""
    assert hasattr(_stem, "cache_info"), "_stem is not memoised"
    _stem.cache_clear()
    _stem("provisioning")
    first = _stem.cache_info()
    _stem("provisioning")
    second = _stem.cache_info()
    assert second.hits == first.hits + 1, (first, second)
    assert second.misses == first.misses, (first, second)


def test_the_cache_is_BOUNDED():
    """An unbounded memo over tokens from an arbitrarily large corpus is a memory leak
    with a plausible-sounding justification. A corpus has many distinct words."""
    info = _stem.cache_info()
    assert info.maxsize is not None, "the stem memo is unbounded"
    assert info.maxsize >= 4096, info


def test_stem_STAYS_PURE_no_clock_no_io_no_state():
    """THE LOAD-BEARING TEST, and the reason this file exists rather than a one-line diff.

    Memoisation is safe only while the function is pure. If a later edit makes `_stem`
    depend on config, a clock, or module state, the memo will serve the first answer
    forever — silently, for every word in every corpus. Walked over the AST so a comment
    mentioning `time` cannot satisfy it, and so the check survives refactoring.
    """
    # The MODULE's file, not `inspect.getfile(_stem)`. Once `_stem` is wrapped in an
    # lru_cache, getfile resolves to functools.py and the walk below would inspect the
    # stdlib instead of the stemmer -- passing or failing for reasons unrelated to this
    # code. Caught by the "not vacuous" control below, which is exactly its job.
    src = pathlib.Path(_tok_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)

    imported = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    banned = {"time", "random", "datetime", "os", "socket", "requests", "urllib"}
    assert not (imported & banned), sorted(imported & banned)

    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_stem"
    )
    # No assignment to a module-level name from inside the stemmer: `global`/`nonlocal`
    # are how a "pure" function quietly acquires state.
    for node in ast.walk(fn):
        assert not isinstance(node, ast.Global), "_stem declares global state"
        assert not isinstance(node, ast.Nonlocal), "_stem declares nonlocal state"


def test_the_purity_walk_is_not_vacuous():
    """POSITIVE CONTROL for the walk above: it must actually find the function."""
    src = pathlib.Path(_tok_module.__file__).read_text(encoding="utf-8")
    names = {n.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef)}
    assert "_stem" in names, sorted(names)[:10]
