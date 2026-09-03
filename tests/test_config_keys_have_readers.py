# Copyright 2026 STARGA, Inc.
"""Every key ``init_workspace`` ships must be a setting, not a claim.

``DEFAULT_CONFIG`` is the file an operator edits to change how this
product behaves. A key in it that no code reads is worse than a missing
feature: it is a lever wired to nothing, and the operator who moves it
believes they have changed something. ``mcp_acl`` shipped that way --
six tool names under ``admin_tools``, three of which (``write_memory``,
``apply_proposal``, ``reindex_vectors``) were not registered tool names
at all -- while the real ACL lived in ``mcp/infra/acl.py``, code-defined
and total. Editing the config to lock a door did nothing, twice removed.

This module is the gate that keeps that from recurring.

**What the scan can and cannot see.** It is a literal search for
``"key"`` / ``'key'`` across ``src/mind_mem/**/*.py`` (excluding
``init_workspace.py``, the writer) **and ``hooks/*.sh``**. Enumerated
before relying on it:

* a key read by a *computed* name (``config.get(name)`` in a loop over
  key names) would be invisible. Checked: no such loop over top-level
  config keys exists in ``src/``.
* a key read only by a non-Python consumer used to be invisible, and
  that blind spot MIS-CLASSIFIED a live setting: ``auto_capture`` is read
  by ``hooks/session-end.sh``, has been for as long as the hook has
  existed, and this gate listed it as reader-less debt. The hooks are
  shipped product -- ``hook_installer`` wires them into every client --
  so scanning them is not a widened net, it is the correct corpus. The
  Go/JS SDKs and templates remain out of scope: a key only they honour is
  still a defect by this gate's standard, because the Python package that
  ships the default must be able to honour it.
* a key whose name is a common English word could collect false
  *positives*, never false negatives. A false positive weakens the
  gate's reach, so :func:`test_scanner_sees_a_key_that_is_really_read`
  and :func:`test_scanner_reports_zero_for_a_key_nothing_reads` pin both
  directions of the method before any verdict is trusted, and
  :func:`test_the_hooks_leg_actually_scanned_the_hooks` pins the new leg
  the same way -- an empty corpus finds nothing and calls it clean.
* **the scan cannot tell a reader from a mention.** The needle is a
  quoted literal anywhere in a file, so a comment, a docstring or a
  dead branch satisfies it exactly as well as a live read -- and a file
  that exists in a working tree satisfies it without existing in the
  commit. Both halves of that bit us at once in a4a1759 (see the
  behavioural section at the foot of this module), so the one key whose
  wiring this release claims is additionally EXECUTED:
  :func:`test_auto_recall_false_actually_silences_the_hook` runs the
  shipped hook and requires the observable output to change. Text is
  the cheap check and stays; behaviour is the one that cannot be
  satisfied by a comment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from mind_mem.init_workspace import DEFAULT_CONFIG

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src", "mind_mem")

#: The writer of the defaults. Its own mention of a key is not a reader.
_WRITER = "init_workspace.py"

#: The shipped hooks. ``hook_installer`` copies these into client configs,
#: so a key one of them reads is honoured on every install.
_HOOKS = os.path.join(os.path.dirname(_HERE), "hooks")

#: **Empty, and that is the point.** Five keys sat here through 5.0.1;
#: 5.0.2 closed every one, by the ladder rather than by the shortcut:
#:
#: * ``auto_capture``   -- was never unread. ``hooks/session-end.sh``
#:                         reads it; the SCANNER was wrong, not the key.
#:                         Fixed by scanning ``hooks/`` (see the module
#:                         docstring), not by excusing it.
#: * ``auto_recall``    -- WIRED. ``hooks/session-start.sh`` now honours
#:                         it, which is what the docs always claimed.
#:                         The wiring is EXECUTED by the behavioural
#:                         tests at the foot of this module, not merely
#:                         matched as text: the first attempt at this
#:                         entry shipped every claim about the wiring
#:                         and left the hook itself uncommitted.
#: * ``workspace_path`` -- REMOVED, substitute named: the workspace is an
#:                         argument / ``MIND_MEM_WORKSPACE``.
#: * ``scan_schedule``  -- REMOVED, substitute named: ``cron_runner``'s
#:                         job table, or ``mind-mem-scan`` by hand.
#: * ``mcp_rate_limit`` -- REMOVED, substitute named: it was a duplicate
#:                         spelling of ``limits.rate_limit_calls_per_minute``
#:                         / ``limits.query_timeout_seconds``.
#:
#: This is a **ratchet, not an exemption list**: the assertion is exact
#: equality, so a new reader-less key fails the build, and so does a
#: stale entry left behind after one is wired. Now that it is empty, the
#: only legal move is to keep it empty -- there is no debt left to hold.
#: ``mcp_acl`` was the sixth entry and went the same way.
KNOWN_UNREAD: frozenset[str] = frozenset()


def _source_blobs() -> dict[str, str]:
    """Every shipped consumer of the config: the package, and the hooks.

    The hooks leg is not a courtesy. ``hooks/session-end.sh`` has read
    ``auto_capture`` all along, and a src-only scan reported that live
    setting as a lever wired to nothing -- the gate manufacturing the
    exact defect it exists to detect. A gate whose corpus is smaller than
    the product does not under-report politely; it reports the wrong
    thing confidently.
    """
    blobs: dict[str, str] = {}
    for root, _dirs, files in os.walk(_SRC):
        for name in files:
            if not name.endswith(".py") or name == _WRITER:
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as handle:
                blobs[path] = handle.read()
    for root, _dirs, files in os.walk(_HOOKS):
        for name in files:
            if not name.endswith(".sh"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as handle:
                blobs[path] = handle.read()
    return blobs


def _readers_of(key: str, blobs: dict[str, str]) -> list[str]:
    """Modules mentioning *key* as a string literal."""
    needles = (f'"{key}"', f"'{key}'")
    return sorted(path for path, blob in blobs.items() if any(n in blob for n in needles))


@pytest.fixture(scope="module")
def blobs() -> dict[str, str]:
    return _source_blobs()


def test_the_scan_actually_scanned_something(blobs: dict[str, str]) -> None:
    """A gate whose corpus is empty passes everything it is asked."""
    assert len(blobs) > 100, f"only {len(blobs)} modules scanned — the walk is wrong"


def test_scanner_sees_a_key_that_is_really_read(blobs: dict[str, str]) -> None:
    """Negative control: the method finds a reader that exists.

    Without this, a zero from :func:`_readers_of` could mean "nothing
    reads it" or "the scanner is broken", and the two are not the same
    finding.

    The named file is ``governance_gate.py`` — where
    :func:`~mind_mem.governance_gate.read_governance_mode` lives, "the
    single parse behind every governance-mode decision in this package".
    It was ``apply_engine.py`` until that module was refactored to call
    the shared reader instead of re-parsing the config itself, which took
    the literal with it and turned this control red for a reason that had
    nothing to do with the scanner. Pinning the key's DEFINITIONAL owner
    rather than one of its consumers is what makes the control survive a
    consumer moving; it is not a looser assertion, it still names exactly
    one file.
    """
    readers = _readers_of("governance_mode", blobs)
    assert readers, "scanner found no reader for governance_mode — the method is broken"
    assert any(p.endswith("governance_gate.py") for p in readers), readers


def test_scanner_reports_zero_for_a_key_nothing_reads(blobs: dict[str, str]) -> None:
    """Positive control: the method can see absence.

    ``mcp_acl`` is the exact key this gate was built for. It is gone from
    the defaults and from the source, so a scanner that can detect
    absence must report zero here — and one that cannot would have let
    ``mcp_acl`` ship unnoticed all over again.
    """
    assert _readers_of("mcp_acl", blobs) == []


def test_mcp_acl_is_not_shipped_as_a_setting() -> None:
    """The dead ACL key is out of the defaults for good."""
    assert "mcp_acl" not in DEFAULT_CONFIG


def test_the_acl_is_code_defined_and_total() -> None:
    """Why ``mcp_acl`` could never have worked: the ACL is not config.

    Every registered tool is classified in code, and a tool in neither
    set is refused before its body runs — so there is no decision a
    config list could have moved.
    """
    from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS

    assert ADMIN_TOOLS and USER_TOOLS
    assert not (ADMIN_TOOLS & USER_TOOLS), "a tool classified both ways has no single verdict"


def test_every_default_config_key_has_a_reader(blobs: dict[str, str]) -> None:
    """Exact-equality ratchet over the reader-less keys."""
    unread = {key for key in DEFAULT_CONFIG if not _readers_of(key, blobs)}

    new = sorted(unread - KNOWN_UNREAD)
    assert not new, (
        f"new DEFAULT_CONFIG key(s) with no reader in src/: {new}. "
        "A key nothing reads is a lever wired to nothing — wire it, or do not ship it."
    )

    fixed = sorted(KNOWN_UNREAD - unread)
    assert not fixed, (
        f"{fixed} now has a reader — remove it from KNOWN_UNREAD. The ratchet only moves down, and it has to be moved deliberately."
    )


def test_the_hooks_leg_actually_scanned_the_hooks(blobs: dict[str, str]) -> None:
    """Positive control for the corpus widening itself.

    ``KNOWN_UNREAD`` is now empty, so every assertion about a key having a
    reader rests on this walk finding files. If ``hooks/`` were mistyped or
    moved, the walk would silently contribute nothing, ``auto_capture`` and
    ``auto_recall`` would read as unread again, and the exact-equality
    ratchet would fail loudly — but only because of this leg, so it is
    pinned directly rather than inferred from a downstream failure.
    """
    hooks = [p for p in blobs if os.sep + "hooks" + os.sep in p]
    assert hooks, f"the hooks walk contributed nothing; is {_HOOKS} still there?"
    assert any(p.endswith("session-start.sh") for p in hooks), sorted(hooks)
    assert any(p.endswith("session-end.sh") for p in hooks), sorted(hooks)


def test_auto_capture_reader_is_the_session_end_hook(blobs: dict[str, str]) -> None:
    """The mis-classification that motivated widening the corpus.

    ``auto_capture`` was listed as reader-less debt for as long as this gate
    has existed while ``hooks/session-end.sh`` was reading it the whole time.
    Naming the file is what makes the fix checkable: a scan that widened to
    ``hooks/`` but stopped matching would pass a bare "has some reader".
    """
    readers = _readers_of("auto_capture", blobs)
    assert any(p.endswith("session-end.sh") for p in readers), readers


def test_auto_recall_is_wired_into_the_session_start_hook(blobs: dict[str, str]) -> None:
    """The key the docs described and no code honoured, now honoured.

    ``README.md`` and ``docs/configuration.md`` both promised that setting
    this false suppresses session-start context. Nothing read it, so the
    promise was false in two published documents at once. Wired rather than
    deleted: no caller is a fact about wiring, never about worth.
    """
    readers = _readers_of("auto_recall", blobs)
    assert any(p.endswith("session-start.sh") for p in readers), readers


def test_the_removed_keys_are_gone_from_the_defaults() -> None:
    """Three levers wired to nothing, removed with a substitute named.

    Each is documented in ``docs/configuration.md`` under "removed", with
    what to use instead — because a key vanishing with no forwarding address
    is its own kind of silent breakage.
    """
    for key in ("workspace_path", "scan_schedule", "mcp_rate_limit"):
        assert key not in DEFAULT_CONFIG, f"{key} is back in DEFAULT_CONFIG with no reader"


def test_the_rate_limit_substitute_is_real() -> None:
    """``mcp_rate_limit`` was removed as a DUPLICATE, so the twin must exist.

    Removing a key on the grounds that another one does the job is only
    honest if that other one is actually there and actually read. Both
    halves are checked: the default ships the key, and the live limiter
    config carries the same name.
    """
    from mind_mem.mcp.infra.config import _DEFAULT_LIMITS

    assert DEFAULT_CONFIG["limits"]["rate_limit_calls_per_minute"] == 120
    assert DEFAULT_CONFIG["limits"]["query_timeout_seconds"] == 30
    assert "rate_limit_calls_per_minute" in _DEFAULT_LIMITS
    assert "query_timeout_seconds" in _DEFAULT_LIMITS


# ---------------------------------------------------------------------------
# The string scan is necessary and NOT sufficient.
# ---------------------------------------------------------------------------
#
# Every assertion above this line is satisfied by the *characters*
# ``"auto_recall"`` appearing in ``hooks/session-start.sh``. That is a
# statement about text, not about behaviour, and the release that
# introduced these tests proved how far apart the two can drift: commit
# a4a1759 landed the emptied ``KNOWN_UNREAD``, the two assertions above,
# the ``init_workspace`` comment reading "now WIRED
# (hooks/session-start.sh)", and the README and ``docs/configuration.md``
# rows describing the switch -- and did not land ``hooks/session-start.sh``
# itself. The wiring existed only in a working tree. Locally the gate was
# green because the scan found a file no one had committed; CI, checking
# out the same SHA, reported ``['auto_recall']`` reader-less. Five
# artefacts asserted the wiring and none of them executed it.
#
# So the switch is exercised: run the shipped hook, flip the key, and
# require the observable behaviour to change. A comment mentioning the key
# satisfies the scan; it cannot satisfy this.


def _hook_is_runnable() -> tuple[bool, str]:
    """Can this platform actually run the hook, rather than merely find it?

    Two things are required and neither is implied by the other. ``bash``
    must run a command -- on a Windows runner a bare ``bash`` resolves to
    the WSL launcher, which with no distribution installed exits 1, so
    ``shutil.which`` says yes and is wrong (the same probe guards the
    ``validate.sh`` tests in ``test_apply_engine.py``). And ``python3``
    must parse JSON, because that is what the hook shells out to; without
    it the hook takes its ``|| echo "true"`` fallback and a test asserting
    "false silences it" would be testing the fallback, not the switch.

    Returned as a reason string rather than a bare bool so a skip on the
    canonical row names which half was missing.
    """
    try:
        bash = subprocess.run(["bash", "-c", "echo mm_bash_ok"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False, "bash could not be executed"
    if bash.returncode != 0 or "mm_bash_ok" not in (bash.stdout or ""):
        return False, "no usable bash (on Windows a bare `bash` is the WSL launcher)"
    try:
        py = subprocess.run(
            ["python3", "-c", "import json; print('mm_py_ok')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False, "python3 could not be executed (the hook shells out to it)"
    if py.returncode != 0 or "mm_py_ok" not in (py.stdout or ""):
        return False, "python3 present but not usable (the hook shells out to it)"
    return True, ""


_HOOK_RUNS, _HOOK_SKIP_REASON = _hook_is_runnable()

_SESSION_START = os.path.join(_HOOKS, "session-start.sh")


def _run_session_start(workspace: str) -> subprocess.CompletedProcess[str]:
    """Run the shipped hook exactly as ``hooks.json`` registers it."""
    return subprocess.run(
        ["bash", _SESSION_START],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "MIND_MEM_WORKSPACE": workspace},
    )


def _workspace(tmp_path: object, config: dict[str, object] | None) -> str:
    """A workspace the hook has something to say about.

    The state file is what makes the ON case produce output; without it
    the hook prints its "not initialized" line instead, and an assertion
    that OFF prints nothing would pass against a hook that prints nothing
    either way.
    """
    ws = str(tmp_path)
    os.makedirs(os.path.join(ws, "memory"), exist_ok=True)
    with open(os.path.join(ws, "memory", "intel-state.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "governance_mode": "detect_only",
                "last_scan": "2026-09-02",
                "counters": {"contradictions_open": 0},
            },
            handle,
        )
    if config is not None:
        with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as handle:
            json.dump(config, handle)
    return ws


def test_the_hook_probe_is_live_on_the_canonical_row() -> None:
    """A guard that always skips is a test that never ran.

    The behavioural tests below are the only ones in this module that
    execute anything, and they are conditional. On Linux -- every row
    where this gate is enforced -- the condition must hold, so a broken
    probe fails the build here instead of quietly turning the two tests
    that matter into no-ops.
    """
    if sys.platform.startswith("linux"):
        assert _HOOK_RUNS, f"the hook probe failed on Linux: {_HOOK_SKIP_REASON}"


@pytest.mark.skipif(not _HOOK_RUNS, reason=_HOOK_SKIP_REASON or "hook not runnable")
def test_auto_recall_false_actually_silences_the_hook(tmp_path: object) -> None:
    """The round trip, on one workspace, with only the key changed.

    The negative half -- "``false`` prints nothing" -- passes on its own
    against a hook that is broken, missing, or silent for some unrelated
    reason. It is only evidence next to the positive control immediately
    before it: the same hook, the same workspace, the same command, with
    ``true`` in place of ``false``, must print the health line. One of the
    two assertions proves the method can see output; the other proves the
    switch removes it.
    """
    on = _workspace(tmp_path, {"auto_recall": True})
    lit = _run_session_start(on)
    assert lit.returncode == 0, lit.stderr
    assert "mind-mem health" in lit.stdout, (
        f"positive control failed: the hook said nothing with auto_recall true — stdout={lit.stdout!r} stderr={lit.stderr!r}"
    )

    with open(os.path.join(on, "mind-mem.json"), "w", encoding="utf-8") as handle:
        json.dump({"auto_recall": False}, handle)

    off = _run_session_start(on)
    assert off.returncode == 0, off.stderr
    assert off.stdout == "", (
        f"auto_recall false did not silence the hook: {off.stdout!r}. "
        "The key is documented as the switch for this hook; if the literal is "
        "present but the behaviour is not, the scan above is measuring a comment."
    )


@pytest.mark.skipif(not _HOOK_RUNS, reason=_HOOK_SKIP_REASON or "hook not runnable")
def test_the_hook_still_speaks_when_no_config_names_the_key(tmp_path: object) -> None:
    """Wiring a key must not change what an existing workspace does.

    Every workspace initialised before the key was read has no opinion
    about it, and many have no ``mind-mem.json`` at all. The default is
    ``True`` in ``DEFAULT_CONFIG`` and the hook defaults the same way, so
    the silent majority keeps its context. A wiring that switched a
    documented feature off for everyone who never asked would be a
    regression wearing a fix's clothes.
    """
    assert DEFAULT_CONFIG["auto_recall"] is True, "the default this test pins has moved"

    bare = _run_session_start(_workspace(tmp_path, None))
    assert bare.returncode == 0, bare.stderr
    assert "mind-mem health" in bare.stdout, bare.stdout
