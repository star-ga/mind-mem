# Copyright 2026 STARGA, Inc.
"""Every text-mode ``open`` in shipped code must name its encoding.

Python picks the *locale* encoding when you omit it. On Linux that is UTF-8
and nothing goes wrong; on Windows it is cp1252, and reading a corpus file
containing any non-ASCII byte raises ``UnicodeDecodeError``.

This was not hypothetical. ``apply_engine._mark_proposal_status`` WROTE with
``encoding="utf-8"`` and READ without it, so on Windows mind-mem wrote UTF-8
and then tried to read it back as cp1252 — every ``rollback_proposal`` against
a proposal containing so much as a ✅ died with *"'charmap' codec can't decode
byte 0x9d"*. 35 sites across 10 modules had the same latent defect. The Linux
gate could never see any of it.

Parsed with ``ast`` rather than grepped, so a mention inside a docstring (of
which there are two, deliberately describing the dangerous pattern) is not
mistaken for a call.
"""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "mind_mem"

#: Modes that carry bytes, where an encoding is meaningless (and an error).
_BINARY = ("rb", "wb", "ab", "r+b", "w+b", "a+b", "xb")


def _text_opens_without_encoding(tree: ast.AST) -> list[int]:
    """Line numbers of text-mode open()/fdopen() calls with no encoding."""
    bad: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        # Bare ``open(...)`` and ``os.fdopen(...)`` only. NOT every attribute
        # called "open": ``tarfile.open(x, "w:gz")`` and ``gzip.open`` take a
        # compression mode, not a text mode, and have no ``encoding`` in the
        # same sense -- flagging them would push someone to "fix" a call that
        # was never broken.
        if isinstance(fn, ast.Name):
            name = fn.id
        elif isinstance(fn, ast.Attribute) and fn.attr == "fdopen":
            name = "fdopen"
        else:
            continue
        if name not in ("open", "fdopen"):
            continue
        if any(k.arg == "encoding" for k in node.keywords):
            continue
        mode = None
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            mode = node.args[1].value
        for k in node.keywords:
            if k.arg == "mode" and isinstance(k.value, ast.Constant):
                mode = k.value.value
        if isinstance(mode, str) and ("b" in mode):
            continue  # binary: encoding would be an error
        if mode is None:
            continue  # default "r" with no explicit mode: covered below
        bad.append(node.lineno)
    return bad


def _modules() -> list[pathlib.Path]:
    return sorted(SRC.rglob("*.py"))


def test_the_scan_finds_the_modules() -> None:
    """Positive control: a scan over zero files passes everything."""
    mods = _modules()
    assert len(mods) > 100, f"only found {len(mods)} modules under {SRC}"


def test_no_text_open_omits_its_encoding() -> None:
    offenders: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line in _text_opens_without_encoding(tree):
            offenders.append(f"{path.relative_to(SRC)}:{line}")
    assert offenders == [], (
        "text-mode open() without encoding= — these read as cp1252 on Windows "
        "and raise UnicodeDecodeError on any non-ASCII byte:\n  " + "\n  ".join(offenders)
    )


def test_the_detector_actually_detects() -> None:
    """Mutation control: the scanner must flag a known-bad call.

    Without this, `offenders == []` above would also pass against a detector
    that never returns anything — which is how a guard silently stops
    guarding.
    """
    bad = ast.parse('with open(p, "r") as f:\n    pass\n')
    assert _text_opens_without_encoding(bad) == [1]

    good = ast.parse('with open(p, "r", encoding="utf-8") as f:\n    pass\n')
    assert _text_opens_without_encoding(good) == []

    binary = ast.parse('with open(p, "rb") as f:\n    pass\n')
    assert _text_opens_without_encoding(binary) == [], "binary must not be flagged"


#: ``importlib.metadata`` Distribution.read_text(NAME) takes a FILENAME and has
#: no encoding kwarg -- "fixing" it would be an error, so it is exempt by
#: location rather than by guessing from the call shape.
_READ_TEXT_EXEMPT = {("self_update.py", "read_text")}


def _path_text_calls_without_encoding(tree: ast.AST, filename: str) -> list[int]:
    """``Path.read_text()`` / ``.write_text()`` with no encoding.

    Same defect as a bare ``open``: both default to the locale encoding, so
    both read cp1252 on Windows. ``model_signing`` wrote a signing manifest
    this way -- on Windows that verifies against different bytes than it wrote.
    """
    bad: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("read_text", "write_text"):
            continue
        if (filename, node.func.attr) in _READ_TEXT_EXEMPT:
            continue
        if any(k.arg == "encoding" for k in node.keywords):
            continue
        bad.append(node.lineno)
    return bad


def test_no_path_text_helper_omits_its_encoding() -> None:
    offenders: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line in _path_text_calls_without_encoding(tree, path.name):
            offenders.append(f"{path.relative_to(SRC)}:{line}")
    assert offenders == [], (
        "Path.read_text()/write_text() without encoding= — same locale-default defect as a bare open():\n  " + "\n  ".join(offenders)
    )


def test_the_path_detector_actually_detects() -> None:
    """Mutation control for the scanner above."""
    bad = ast.parse("p.read_text()\n")
    assert _path_text_calls_without_encoding(bad, "x.py") == [1]
    good = ast.parse('p.read_text(encoding="utf-8")\n')
    assert _path_text_calls_without_encoding(good, "x.py") == []
