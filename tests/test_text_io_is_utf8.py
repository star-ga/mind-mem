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

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "mind_mem"

#: Directories that hold no source of ours. Everything else is scanned, so a
#: directory added tomorrow is covered the day it appears -- an include-list
#: is what let this gate miss ``tests/`` for as long as it existed.
#:
#: ``.wt``           git worktrees: other checkouts of this same repo, which
#:                   would be scanned twice and could be at any revision.
#: ``build``/``dist``/``*.egg-info``  packaging output, not authored here.
#: ``.venv``/``venv``/``node_modules``  third-party code.
#: the caches         ``__pycache__``, ``.mypy_cache``, ``.pytest_cache``,
#:                   ``.ruff_cache``, ``.hypothesis`` -- generated.
_NOT_OURS = frozenset(
    {
        ".git",
        ".wt",
        "build",
        "dist",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".hypothesis",
        ".tox",
        ".eggs",
    }
)

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
        # ``mode is None`` means NO mode argument, i.e. the default "r" --
        # text mode, locale encoding, exactly the defect. An earlier version
        # of this scanner skipped that case with the comment "covered below"
        # and then did not cover it, so 20 mode-less reads passed the guard
        # while the commit claimed every text open was fixed. A guard with a
        # hole is worse than no guard: it certifies the thing it missed.
        bad.append(node.lineno)
    return bad


def _is_ours(path: pathlib.Path) -> bool:
    parts = path.relative_to(REPO).parts
    return not any(p in _NOT_OURS or p.endswith(".egg-info") for p in parts)


def _modules() -> list[pathlib.Path]:
    """Every Python file in the repository, not just the shipped package.

    Scoped to ``src/mind_mem`` when this gate was written, which is exactly
    how the defect it exists to stop reached CI anyway: the HTTP transport's
    ``workspace`` fixture seeded ``DECISIONS.md`` with a bare
    ``write_text(...)``, the em dash in it was encoded as cp1252 byte 0x97 on
    the Windows runners, and ``delete_block`` -- which reads the corpus
    strictly as UTF-8 -- died on it. `DELETE /memories/<missing-id>` answered
    500 on all five Windows rows and 404 everywhere else. The product code was
    correct and this gate was green; the file that wrote the bytes was simply
    outside the only directory it looked at.

    A test fixture that seeds a corpus is writing corpus bytes, so it is held
    to the corpus rule. So is a script, a benchmark and a training helper.
    """
    return sorted(p for p in REPO.rglob("*.py") if _is_ours(p))


def test_the_scan_finds_the_modules() -> None:
    """Positive control: a scan over zero files passes everything."""
    mods = _modules()
    assert len(mods) > 800, f"only found {len(mods)} python files under {REPO}"
    names = {m.relative_to(REPO).parts[0] for m in mods}
    # Named individually: "more than 800 files" would still be satisfied by a
    # scan that swept src/ and tests/ and silently dropped everything else,
    # which is the shape of the miss this widening repairs.
    for expected in ("src", "tests", "scripts", "benchmarks", "train", "examples"):
        assert expected in names, f"{expected}/ is not being scanned: {sorted(names)}"
    assert any(m.parent == REPO for m in mods), "root-level modules are not being scanned"


def test_no_text_open_omits_its_encoding() -> None:
    offenders: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line in _text_opens_without_encoding(tree):
            offenders.append(f"{path.relative_to(REPO)}:{line}")
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

    # Pin the hole that ACTUALLY happened, not a cousin of it. The scanner
    # once skipped mode-less opens with `if mode is None: continue`, and 20
    # sites escaped while the guard went green. An explicit-mode control
    # would still pass if someone reintroduced that line -- so the historical
    # mutation gets its own assertion. A mutation control that does not pin
    # the real mutation certifies the return of the bug it exists to stop.
    modeless = ast.parse("with open(p) as f:\n    pass\n")
    assert _text_opens_without_encoding(modeless) == [1], "the historical hole"

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
            offenders.append(f"{path.relative_to(REPO)}:{line}")
    assert offenders == [], (
        "Path.read_text()/write_text() without encoding= — same locale-default defect as a bare open():\n  " + "\n  ".join(offenders)
    )


def test_the_path_detector_actually_detects() -> None:
    """Mutation control for the scanner above."""
    bad = ast.parse("p.read_text()\n")
    assert _path_text_calls_without_encoding(bad, "x.py") == [1]
    good = ast.parse('p.read_text(encoding="utf-8")\n')
    assert _path_text_calls_without_encoding(good, "x.py") == []


def _subprocess_decodes_without_encoding(tree: ast.AST) -> list[int]:
    """``subprocess`` calls that DECODE output with the locale codec.

    ``text=True`` (and its old spelling ``universal_newlines=True``, and
    ``check_output`` in text mode) turns the child's bytes into ``str`` using
    ``locale.getpreferredencoding()``. That is UTF-8 on the Linux and macOS
    runners and cp1252 on the Windows ones, so the same helper reads a git
    author, a validator's TOTAL line or a probe's output correctly on two
    platforms and returns mojibake -- or raises -- on the third.

    ``apply_engine`` already carries the comment naming this ("encoding is
    explicit: `text=True` alone decodes with the locale preferred encoding,
    which is cp1252 on Windows"), on one call out of two in the same
    function. A convention held by one site and a comment is not a gate.
    """
    bad: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "subprocess"):
            continue
        if fn.attr not in ("run", "Popen", "check_output", "call", "check_call"):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        if "encoding" in kw:
            continue
        # ``check_output`` decodes only when asked to; without text/
        # universal_newlines it returns bytes, which need no encoding.
        #
        # And the VALUE matters, not the presence of the keyword. A call
        # written ``text=False,  # bytes -- mic-b output is binary`` is asking
        # for bytes on purpose, and ``encoding=`` would force it back into
        # text mode: every ``b"..." in result.stderr`` against it then raises
        # ``TypeError: 'in <string>' requires string as left operand``. An
        # earlier version of this scanner tested ``"text" in kw`` and reddened
        # seven tests by "fixing" exactly that call.
        if any(_asks_for_text(kw.get(name)) for name in ("text", "universal_newlines")):
            bad.append(node.lineno)
    return bad


def _asks_for_text(value: ast.expr | None) -> bool:
    """Whether a ``text=``/``universal_newlines=`` argument means text mode.

    ``None`` (absent) and a literal false value mean bytes. Anything else --
    ``True``, or an expression this scanner cannot evaluate -- is treated as
    text, so a computed flag is flagged rather than waved through.
    """
    if value is None:
        return False
    if isinstance(value, ast.Constant):
        return bool(value.value)
    return True


def test_no_subprocess_decodes_with_the_locale_codec() -> None:
    offenders: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line in _subprocess_decodes_without_encoding(tree):
            offenders.append(f"{path.relative_to(REPO)}:{line}")
    assert offenders == [], (
        "subprocess text=True without encoding= — these decode the child's output as cp1252 on Windows:\n  " + "\n  ".join(offenders)
    )


def test_the_subprocess_detector_actually_detects() -> None:
    """Mutation control for the scanner above."""
    bad = ast.parse("subprocess.run(cmd, capture_output=True, text=True)\n")
    assert _subprocess_decodes_without_encoding(bad) == [1]

    legacy = ast.parse("subprocess.run(cmd, universal_newlines=True)\n")
    assert _subprocess_decodes_without_encoding(legacy) == [1], "the old spelling of text="

    good = ast.parse('subprocess.run(cmd, text=True, encoding="utf-8")\n')
    assert _subprocess_decodes_without_encoding(good) == []

    binary = ast.parse("subprocess.run(cmd, capture_output=True)\n")
    assert _subprocess_decodes_without_encoding(binary) == [], "bytes mode must not be flagged"

    # The mutation that actually happened: the scanner read the PRESENCE of
    # the keyword instead of its value, and an explicit request for bytes was
    # reported as a missing encoding.
    explicit_bytes = ast.parse("subprocess.run(cmd, text=False)\n")
    assert _subprocess_decodes_without_encoding(explicit_bytes) == [], "text=False is a request for bytes"
    assert _subprocess_decodes_without_encoding(ast.parse("subprocess.run(cmd, universal_newlines=False)\n")) == []

    # A flag this scanner cannot evaluate is flagged, not waved through.
    computed = ast.parse("subprocess.run(cmd, text=want_text)\n")
    assert _subprocess_decodes_without_encoding(computed) == [1], "an unevaluable flag must not be assumed binary"


def _tempfile_text_without_encoding(tree: ast.AST) -> list[int]:
    """``tempfile.NamedTemporaryFile(mode="w")`` with no encoding.

    The same locale default as :func:`open`, reached through a different
    door. ``mode`` defaults to ``"w+b"``, so only an explicitly textual mode
    is flagged -- adding ``encoding=`` to a binary temp file is an error.
    """
    bad: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "tempfile"):
            continue
        if fn.attr not in ("NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile"):
            continue
        if any(k.arg == "encoding" for k in node.keywords):
            continue
        mode = None
        for k in node.keywords:
            if k.arg == "mode" and isinstance(k.value, ast.Constant):
                mode = k.value.value
        if isinstance(mode, str) and "b" not in mode:
            bad.append(node.lineno)
    return bad


def test_no_text_mode_tempfile_omits_its_encoding() -> None:
    offenders: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line in _tempfile_text_without_encoding(tree):
            offenders.append(f"{path.relative_to(REPO)}:{line}")
    assert offenders == [], (
        'tempfile.NamedTemporaryFile(mode="w") without encoding= — same locale default as a bare open():\n  ' + "\n  ".join(offenders)
    )


def test_the_tempfile_detector_actually_detects() -> None:
    """Mutation control for the scanner above."""
    bad = ast.parse('tempfile.NamedTemporaryFile(mode="w", suffix=".md")\n')
    assert _tempfile_text_without_encoding(bad) == [1]

    good = ast.parse('tempfile.NamedTemporaryFile(mode="w", encoding="utf-8")\n')
    assert _tempfile_text_without_encoding(good) == []

    binary = ast.parse('tempfile.NamedTemporaryFile(mode="wb")\n')
    assert _tempfile_text_without_encoding(binary) == [], "binary must not be flagged"

    default = ast.parse("tempfile.NamedTemporaryFile()\n")
    assert _tempfile_text_without_encoding(default) == [], "the default mode is binary"
