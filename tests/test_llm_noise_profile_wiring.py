# Copyright 2026 STARGA, Inc.
"""Acceptance gate for the ``llm_noise_profile`` wiring (5.0.1 restoration, slice 5).

``llm_noise_profile`` shipped a complete per-provider / per-domain EMA and a
JSON persistence layer, kept 40-odd unit tests of its own, and was imported by
**nothing in the product** — which is why 5.0.0 deleted it. Its tests proved
the arithmetic; none of them proved a caller existed, and that is the whole
lesson of the sweep.

The event source it was missing was already in the tree: ``report_outcome``
carries a verdict ("did acting on these blocks actually work?") plus the
provenance of who reported it. That is a ``was_correct`` observation about a
noisy sensor. This file pins the join.

Five contracts, one class each:

1. the derivation rules are real functions with real answers, not guesses
   hidden inside the call site;
2. a recorded outcome **moves per-domain reliability**, and the profile
   **survives a restart** — proved in a genuinely fresh interpreter, not by
   re-reading an object this process still holds;
3. influence is **bounded**: a replayed report moves nothing, and one report
   naming fifty blocks is one observation, not fifty;
4. ``calibration_stats`` exposes it when the flag is on;
5. with the flag OFF the behaviour is byte-identical to the wiring not being
   there at all — same envelope, same workspace bytes, same log lines — and
   the flag PROBE is unobservable even against a malformed config.

Every assertion in classes 2-4 fails if the ``_fold_into_noise_profile`` call
is removed from ``outcome_store.record_outcome`` or the module body is
stubbed; class 5 fails if the leg runs when it should not. The teeth of the
class-5 comparison are checked by an explicit positive control
(``test_the_off_comparison_has_teeth``) so a vacuous "identical" cannot pass.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mind_mem import llm_noise_profile
from mind_mem.calibration import CalibrationManager
from mind_mem.init_workspace import init
from mind_mem.llm_noise_profile import (
    DEFAULT_DOMAIN,
    NOISE_PROFILE_FLAG,
    PROFILE_REL_PATH,
    UNATTRIBUTED_PROVIDER,
    LLMNoiseProfiler,
    block_domain,
    profile_path,
    provider_for,
    report_domains,
)
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mm_cli import config_set
from mind_mem.outcome_attribution import report_outcome

_ID_A = "D-20260401-001"
_ID_B = "D-20260401-002"
_ID_T = "T-20260401-001"
_STAMP = "2026-04-02T09:00:00Z"
_TOOL = "gpt-4"

#: The module's own starting point for an unseen provider. Hard-coded rather
#: than imported so a silent change to the default is caught here too.
_DEFAULT_RELIABILITY = 0.7
_ALPHA = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _point_handler_at(handler: logging.StreamHandler, stream: object) -> None:
    """``handler.setStream(stream)``, tolerating an already-closed old stream.

    ``setStream`` flushes the stream it is replacing, which is right and
    which raises ``ValueError: I/O operation on closed file`` when the
    stream it finds is a pytest capture object that was closed at the end
    of some *earlier* test. ``observability.StructuredLogger`` captures
    ``sys.stderr`` once, when the logger is first built, so whichever test
    in the session happened to build it decides what this handler holds
    for every test after — and this file's own passing or failing then
    depends on what ran before it, which is not a property of the code
    under test.

    The fallback runs only when the flush actually fails, and a closed
    stream has no buffered output left to lose. Everything else —
    including the flush — is unchanged, so nothing is skipped on a stream
    that is alive.
    """
    try:
        handler.setStream(stream)  # type: ignore[arg-type]
    except ValueError:
        handler.acquire()
        try:
            handler.stream = stream  # type: ignore[assignment]
        finally:
            handler.release()


@contextlib.contextmanager
def _mind_mem_stderr():
    """Collect everything mind-mem's own loggers write, as an operator sees it.

    Same shape as ``tests/test_granularity_align_wiring.py``: neither
    ``caplog`` nor ``capfd`` works, because ``observability.StructuredLogger``
    sets ``propagate = False`` and captures the ``sys.stderr`` object that
    existed when the logger was first built.
    """
    buffer = io.StringIO()
    restore: list[tuple[logging.StreamHandler, object]] = []
    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        if not name.startswith("mind-mem") or not isinstance(logger, logging.Logger):
            continue
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                restore.append((handler, handler.stream))
                _point_handler_at(handler, buffer)
    saved_stderr = sys.stderr
    sys.stderr = buffer
    try:
        yield buffer
    finally:
        sys.stderr = saved_stderr
        for handler, stream in restore:
            _point_handler_at(handler, stream)


def _events(buffer: io.StringIO) -> list[str]:
    """Event names mind-mem logged; timestamps would differ run to run."""
    names: list[str] = []
    for line in buffer.getvalue().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            names.append(line)
            continue
        names.append(str(entry.get("event", "")))
    return names


#: Written by ``init`` since 5.0.2 — the spec binding that arms the gate.
_BINDING_FILE = ".spec_binding.json"


def _digest_bytes(root: Path, path: Path) -> bytes:
    """The file's bytes, with location- and clock-derived fields normalised.

    Exactly one file needs this, for reasons that have nothing to do with
    the flag under test. ``init`` arms every workspace it creates, and the
    binding it writes records the *absolute path* of the config it
    attested and the *instant* it did so. Two workspaces, in two
    directories, created microseconds apart, therefore differ in those
    two fields no matter what the flag says — so comparing them raw makes
    this test fail permanently on a difference the wiring cannot cause.

    This is normalisation, not a narrowed scan. The file stays in the
    digest and every other field stays in it byte for byte — including
    ``spec_hash``, which is the SHA3-512 of the config the flag lives in,
    so a flag that reached the config would still move this digest.
    ``config_path`` is kept, made workspace-relative rather than dropped.
    A binding that will not parse is digested raw: corruption must show
    up as a difference, never be normalised away.
    """
    raw = path.read_bytes()
    if path.name != _BINDING_FILE:
        return raw
    try:
        record = json.loads(raw)
        record["bound_at"] = "<normalised: bind instant>"
        record["config_path"] = Path(os.path.relpath(record["config_path"], root)).as_posix()
    except (ValueError, KeyError, TypeError):
        return raw
    return json.dumps(record, sort_keys=True).encode("utf-8")


def _tree_digest(root: Path) -> dict[str, str]:
    """SHA-256 of every regular file under *root*, keyed by POSIX relative path.

    SQLite's ``-wal`` / ``-shm`` sidecars are skipped: opening a WAL database
    creates them, so their presence is a property of having connected at all.
    The bytes that matter — ``recall.db`` and any profile file — are covered.
    ``.spec_binding.json`` goes through :func:`_digest_bytes` first; see there.
    """
    digest: dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.endswith(("-wal", "-shm")):
                continue
            path = Path(dirpath) / name
            # ``as_posix()``, never ``str()``: os-native separators would key
            # this map with backslashes on Windows, so every membership test
            # against a forward-slash constant would miss and the comparison
            # would silently lose its teeth.
            digest[path.relative_to(root).as_posix()] = hashlib.sha256(_digest_bytes(root, path)).hexdigest()
    return digest


def _make_workspace(tmp_path: Path, name: str, flag: object) -> str:
    """A seeded workspace whose ``mind-mem.json`` carries *flag*.

    ``flag`` is written verbatim under ``v4.llm_noise_profile``; pass the
    sentinel ``...`` to leave the key out entirely, which is what an
    untouched 5.0.0 config looks like.
    """
    workspace = str(tmp_path / name)
    os.makedirs(workspace)
    init(workspace)
    # Through ``mm config set``: ``init`` arms the gate against the config
    # it wrote, so a hand edit is drift rather than configuration.
    if flag is not ...:
        config_set(os.path.join(workspace, "mind-mem.json"), f"v4.{NOISE_PROFILE_FLAG}", flag)
    return workspace


@pytest.fixture()
def pinned_config(monkeypatch, tmp_path):
    """Point the flag resolver at a per-test config path.

    ``_config_path`` falls back to the cwd and then to ``~/.mind-mem``; an
    operator config elsewhere on this machine must not decide whether these
    tests pass. Each test re-points it at the workspace it is exercising.
    """
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "absent-mind-mem.json"))

    def _use(workspace: str) -> str:
        monkeypatch.setenv("MIND_MEM_CONFIG", os.path.join(workspace, "mind-mem.json"))
        return workspace

    return _use


def _profile(workspace: str) -> LLMNoiseProfiler:
    """Load the persisted profile the way a brand-new process would."""
    profiler = LLMNoiseProfiler()
    profiler.load(profile_path(workspace))
    return profiler


# ---------------------------------------------------------------------------
# 1. The derivation rules
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDerivation:
    def test_flag_registered(self) -> None:
        """Guard: an unregistered flag reads as OFF forever, silently."""
        from mind_mem.v4 import feature_flags

        assert NOISE_PROFILE_FLAG in feature_flags.ALL_V4_FLAGS

    @pytest.mark.parametrize(
        ("block_id", "expected"),
        [
            ("D-20260401-001", "D"),
            ("T-20260401-001", "T"),
            ("inbox-2026-01", "INBOX"),
            ("DIA-D1-3", "DIA"),
            ("  C-1  ", "C"),
            ("42", DEFAULT_DOMAIN),
            ("", DEFAULT_DOMAIN),
        ],
    )
    def test_domain_is_the_block_id_family(self, block_id: str, expected: str) -> None:
        assert block_domain(block_id) == expected

    def test_domains_are_distinct_and_sorted(self) -> None:
        """Distinct: fifty decision blocks are one verdict about ``D``.

        Sorted: the EMA is order-dependent, so two identical reports have to
        move the profile identically.
        """
        assert report_domains([_ID_B, _ID_A, _ID_T, _ID_A]) == ["D", "T"]
        assert report_domains([f"D-2026-{i:03d}" for i in range(50)]) == ["D"]

    @pytest.mark.parametrize(
        ("tool_id", "actor_id", "expected"),
        [
            ("gpt-4", "ci", "gpt-4"),
            ("", "ci", "ci"),
            ("  ", "  ", UNATTRIBUTED_PROVIDER),
            ("", "", UNATTRIBUTED_PROVIDER),
        ],
    )
    def test_provider_precedence(self, tool_id: str, actor_id: str, expected: str) -> None:
        assert provider_for(tool_id=tool_id, actor_id=actor_id) == expected

    def test_stamp_parses_to_a_timezone_independent_epoch(self) -> None:
        from mind_mem.llm_noise_profile import stamp_to_epoch

        assert stamp_to_epoch("1970-01-01T00:00:00Z") == 0.0
        assert stamp_to_epoch("not a stamp") is None
        assert stamp_to_epoch(None) is None


# ---------------------------------------------------------------------------
# 2. Outcomes move reliability, and the movement survives a restart
# ---------------------------------------------------------------------------


class TestOutcomesMoveReliability:
    def test_a_failure_lowers_the_domain_reliability(self, tmp_path, pinned_config) -> None:
        ws = pinned_config(_make_workspace(tmp_path, "on", {"enabled": True}))

        assert not os.path.exists(profile_path(ws))
        report_outcome(ws, [_ID_A], "failure", task_id="build-1", tool_id=_TOOL, recorded_at=_STAMP)

        assert os.path.isfile(profile_path(ws)), "the profile was never persisted"
        profiler = _profile(ws)
        # One failure: global EMA 0.7 -> 0.665, and a new domain seeds from
        # the freshly updated global (the module's stated seeding rule).
        assert profiler.get_reliability(_TOOL) == pytest.approx(_DEFAULT_RELIABILITY * _ALPHA)
        assert profiler.get_reliability(_TOOL, "D") == pytest.approx(_DEFAULT_RELIABILITY * _ALPHA)
        assert profiler.get_reliability(_TOOL, "D") < _DEFAULT_RELIABILITY

    def test_success_and_failure_move_in_opposite_directions(self, tmp_path, pinned_config) -> None:
        good = pinned_config(_make_workspace(tmp_path, "good", {"enabled": True}))
        report_outcome(good, [_ID_A], "success", task_id="b", tool_id=_TOOL, recorded_at=_STAMP)
        up = _profile(good).get_reliability(_TOOL, "D")

        bad = pinned_config(_make_workspace(tmp_path, "bad", {"enabled": True}))
        report_outcome(bad, [_ID_A], "failure", task_id="b", tool_id=_TOOL, recorded_at=_STAMP)
        down = _profile(bad).get_reliability(_TOOL, "D")

        assert up > _DEFAULT_RELIABILITY > down

    def test_neutral_moves_nothing(self, tmp_path, pinned_config) -> None:
        """``neutral`` means "not attributable" — that is not evidence either."""
        ws = pinned_config(_make_workspace(tmp_path, "neutral", {"enabled": True}))
        result = report_outcome(ws, [_ID_A], "neutral", task_id="b", tool_id=_TOOL, recorded_at=_STAMP)

        assert result["recorded"] == 1  # the outcome row itself was written
        assert not os.path.exists(profile_path(ws)), "a neutral verdict must not create a profile"

    def test_each_domain_in_a_mixed_report_is_observed(self, tmp_path, pinned_config) -> None:
        ws = pinned_config(_make_workspace(tmp_path, "mixed", {"enabled": True}))
        report_outcome(ws, [_ID_A, _ID_T], "failure", task_id="b", tool_id=_TOOL, recorded_at=_STAMP)

        profiler = _profile(ws)
        assert set(profiler._profiles[_TOOL].domain_reliability) == {"D", "T"}
        assert profiler._profiles[_TOOL].total_observations == 2
        assert profiler._profiles[_TOOL].error_count == 2

    def test_provider_falls_back_to_actor_then_unattributed(self, tmp_path, pinned_config) -> None:
        ws = pinned_config(_make_workspace(tmp_path, "prov", {"enabled": True}))
        report_outcome(ws, [_ID_A], "failure", task_id="b", actor_id="ci", recorded_at=_STAMP)
        report_outcome(ws, [_ID_A], "failure", task_id="c", recorded_at=_STAMP)

        assert set(_profile(ws)._profiles) == {"ci", UNATTRIBUTED_PROVIDER}

    def test_profile_survives_a_real_restart(self, tmp_path, pinned_config) -> None:
        """The second observation must build on the first, across processes.

        A fresh interpreter is the point. Re-reading the file in *this*
        process would also pass if the state lived in memory; only a new
        process proves the file is the state.
        """
        ws = pinned_config(_make_workspace(tmp_path, "restart", {"enabled": True}))
        report_outcome(ws, [_ID_A], "failure", task_id="build-1", tool_id=_TOOL, recorded_at=_STAMP)
        after_first = _profile(ws).get_reliability(_TOOL, "D")

        script = (
            "from mind_mem.outcome_attribution import report_outcome\n"
            f"report_outcome({ws!r}, [{_ID_A!r}], 'failure', task_id='build-2',"
            f" tool_id={_TOOL!r}, recorded_at={_STAMP!r})\n"
        )
        env = dict(os.environ, MIND_MEM_CONFIG=os.path.join(ws, "mind-mem.json"))
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=180, encoding="utf-8", errors="replace"
        )
        assert proc.returncode == 0, proc.stderr

        after_second = _profile(ws).get_reliability(_TOOL, "D")
        # Continued from the persisted value, not restarted from the default.
        assert after_second == pytest.approx(after_first * _ALPHA)
        assert after_second < after_first
        assert _profile(ws)._profiles[_TOOL].total_observations == 2

    def test_the_mcp_report_outcome_tool_feeds_the_profile(self, tmp_path, pinned_config) -> None:
        """The plug point is the tool an agent actually calls, not just the lib."""
        from mind_mem.mcp.tools.calibration import report_outcome as mcp_report_outcome

        ws = pinned_config(_make_workspace(tmp_path, "mcp", {"enabled": True}))
        with use_workspace(ws):
            payload = json.loads(mcp_report_outcome(block_ids=[_ID_A], outcome="failure", task_id="b", tool_id=_TOOL))
        assert payload["status"] == "recorded"
        assert _profile(ws).get_reliability(_TOOL, "D") < _DEFAULT_RELIABILITY

    def test_the_persisted_file_is_a_pure_function_of_the_report(self, tmp_path, pinned_config) -> None:
        """Same reports, same injected stamp, byte-identical file.

        ``recorded_at`` is injectable all the way through, so the profile is
        reproducible from the outcome stream rather than from when the stream
        was replayed. Two workspaces, two moments, one set of bytes.
        """
        one = pinned_config(_make_workspace(tmp_path, "det-1", {"enabled": True}))
        report_outcome(one, [_ID_A, _ID_T], "success", task_id="b", tool_id=_TOOL, recorded_at=_STAMP)
        two = pinned_config(_make_workspace(tmp_path, "det-2", {"enabled": True}))
        report_outcome(two, [_ID_A, _ID_T], "success", task_id="b", tool_id=_TOOL, recorded_at=_STAMP)

        assert Path(profile_path(one)).read_bytes() == Path(profile_path(two)).read_bytes()


# ---------------------------------------------------------------------------
# 3. Bounded influence
# ---------------------------------------------------------------------------


class TestBoundedInfluence:
    def test_a_replayed_report_moves_nothing(self, tmp_path, pinned_config) -> None:
        """Idempotency has to reach the profile, not just the store.

        The outcome id is the SHA-256 of the canonical payload, so a replay
        conflicts on the primary key and changes no row. If the profile
        updated anyway, a reporter could drive its own reliability anywhere
        by re-sending one report.
        """
        ws = pinned_config(_make_workspace(tmp_path, "replay", {"enabled": True}))
        kwargs = {"task_id": "build-1", "tool_id": _TOOL, "recorded_at": _STAMP}

        first = report_outcome(ws, [_ID_A], "failure", **kwargs)
        after_first = Path(profile_path(ws)).read_bytes()

        for _ in range(5):
            replay = report_outcome(ws, [_ID_A], "failure", **kwargs)
            assert replay["outcome_id"] == first["outcome_id"]
            assert replay["idempotent"] is True

        assert Path(profile_path(ws)).read_bytes() == after_first
        assert _profile(ws)._profiles[_TOOL].total_observations == 1

    def test_listing_more_blocks_buys_no_extra_influence(self, tmp_path, pinned_config) -> None:
        """One report about fifty decisions is one observation about ``D``."""
        few = pinned_config(_make_workspace(tmp_path, "few", {"enabled": True}))
        report_outcome(few, [_ID_A], "failure", task_id="b", tool_id=_TOOL, recorded_at=_STAMP)

        many = pinned_config(_make_workspace(tmp_path, "many", {"enabled": True}))
        report_outcome(
            many,
            [f"D-20260401-{i:03d}" for i in range(1, 51)],
            "failure",
            task_id="b",
            tool_id=_TOOL,
            recorded_at=_STAMP,
        )

        assert _profile(many).get_reliability(_TOOL, "D") == pytest.approx(_profile(few).get_reliability(_TOOL, "D"))
        assert _profile(many)._profiles[_TOOL].total_observations == 1

    def test_a_write_failure_never_breaks_the_outcome_write(self, tmp_path, pinned_config, monkeypatch) -> None:
        """The row is already committed; the profile is a sidecar."""
        ws = pinned_config(_make_workspace(tmp_path, "readonly", {"enabled": True}))

        def _boom(self, path, *, at=None):
            raise OSError("read-only file system")

        monkeypatch.setattr(LLMNoiseProfiler, "save", _boom)
        result = report_outcome(ws, [_ID_A], "failure", task_id="b", tool_id=_TOOL, recorded_at=_STAMP)

        assert result["recorded"] == 1
        assert result["outcome_id"]
        assert not os.path.exists(profile_path(ws))


# ---------------------------------------------------------------------------
# 4. calibration_stats exposure
# ---------------------------------------------------------------------------


class TestCalibrationStatsExposure:
    def test_reliability_appears_when_the_flag_is_on(self, tmp_path, pinned_config) -> None:
        ws = pinned_config(_make_workspace(tmp_path, "stats-on", {"enabled": True}))
        report_outcome(ws, [_ID_A], "failure", task_id="b", tool_id=_TOOL, recorded_at=_STAMP)
        report_outcome(ws, [_ID_T], "success", task_id="c", tool_id="mistral", recorded_at=_STAMP)

        section = CalibrationManager(ws).get_calibration_stats()["llm_reliability"]

        assert section["flag"] == f"v4.{NOISE_PROFILE_FLAG}"
        assert section["path"] == PROFILE_REL_PATH
        assert section["provider_count"] == 2
        by_id = {entry["provider_id"]: entry for entry in section["providers"]}
        assert by_id[_TOOL]["reliability"] < _DEFAULT_RELIABILITY < by_id["mistral"]["reliability"]
        assert by_id[_TOOL]["errors"] == 1
        assert by_id["mistral"]["errors"] == 0
        assert by_id[_TOOL]["observation_noise"] == pytest.approx(1.0 - by_id[_TOOL]["reliability"])
        assert list(by_id[_TOOL]["domains"]) == ["D"]
        # Sorted most-reliable first, ties broken by provider id.
        assert [e["provider_id"] for e in section["providers"]] == ["mistral", _TOOL]

    def test_flag_on_with_no_reports_is_an_empty_roster_not_a_missing_key(self, tmp_path, pinned_config) -> None:
        ws = pinned_config(_make_workspace(tmp_path, "stats-empty", {"enabled": True}))
        section = CalibrationManager(ws).get_calibration_stats()["llm_reliability"]
        assert section["providers"] == []
        assert section["provider_count"] == 0

    def test_the_mcp_tool_carries_it(self, tmp_path, pinned_config) -> None:
        from mind_mem.mcp.tools.calibration import calibration_stats

        ws = pinned_config(_make_workspace(tmp_path, "stats-mcp", {"enabled": True}))
        report_outcome(ws, [_ID_A], "failure", task_id="b", tool_id=_TOOL, recorded_at=_STAMP)
        with use_workspace(ws):
            payload = json.loads(calibration_stats())
        assert payload["llm_reliability"]["providers"][0]["provider_id"] == _TOOL


# ---------------------------------------------------------------------------
# 5. Flag OFF is indistinguishable from the wiring not existing
# ---------------------------------------------------------------------------


_REPORT_SCRIPT = """
import json, sys

ws, dewire = sys.argv[1], sys.argv[2] == "1"
if dewire:
    # The control: this tree with the slice-5 call site removed, i.e. the
    # code exactly as it stood before the wiring landed.
    from mind_mem import outcome_store

    outcome_store._fold_into_noise_profile = lambda *a, **k: None

from mind_mem.outcome_attribution import report_outcome

envelope = report_outcome(
    ws,
    [%(id_a)r, %(id_t)r],
    "failure",
    task_id="build-1",
    tool_id=%(tool)r,
    actor_id="ci",
    recorded_at=%(stamp)r,
)
sys.stdout.write(json.dumps(envelope, sort_keys=True))
""" % {"id_a": _ID_A, "id_t": _ID_T, "tool": _TOOL, "stamp": _STAMP}


def _run_report(workspace: str, *, dewired: bool = False) -> tuple[dict, dict[str, str], list[str]]:
    """One report against *workspace*, in a fresh process: envelope, bytes, logs.

    A subprocess, not an in-process call, and deliberately so. The claim
    being tested is that a flag-off build emits nothing a de-wired build
    would not, and the only honest way to read "emits nothing" is a whole
    process's stderr. Reading it in-process means sharing a logging tree with
    every other test module pytest has imported, whose background threads can
    drop a line into the buffer that has nothing to do with this call — which
    is a false failure waiting to happen, and did happen once before this was
    hermetic.
    """
    script = Path(workspace).parent / f"report_{Path(workspace).name}.py"
    script.write_text(_REPORT_SCRIPT, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script), workspace, "1" if dewired else "0"],
        capture_output=True,
        text=True,
        env=dict(os.environ, MIND_MEM_CONFIG=os.path.join(workspace, "mind-mem.json")),
        timeout=180,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, proc.stderr
    script.unlink()
    return json.loads(proc.stdout), _tree_digest(Path(workspace)), _events(io.StringIO(proc.stderr))


class TestFlagOffIsByteIdentical:
    @pytest.mark.parametrize("flag", [..., {"enabled": False}, {}, True, "yes", None])
    def test_no_profile_is_written(self, tmp_path, pinned_config, flag) -> None:
        """Absent, false, and every non-canonical truthy spelling all mean OFF.

        A bare ``true`` must not switch the surface on — the canonical
        interpretation is ``{"enabled": true}`` and nothing else.
        """
        ws = pinned_config(_make_workspace(tmp_path, f"off-{abs(hash(repr(flag)))}", flag))
        result = report_outcome(ws, [_ID_A], "failure", task_id="b", tool_id=_TOOL, recorded_at=_STAMP)

        assert result["recorded"] == 1
        assert not os.path.exists(profile_path(ws))
        assert "llm_reliability" not in CalibrationManager(ws).get_calibration_stats()

    def test_off_equals_the_wiring_being_absent(self, tmp_path, pinned_config) -> None:
        """Envelope, workspace bytes and log lines, against a de-wired build.

        The control is this same tree with ``_fold_into_noise_profile``
        replaced by a no-op — i.e. the code as it stood before slice 5. If
        the flag-off path differs from that in any byte, the flag is
        observable.
        """
        wired = pinned_config(_make_workspace(tmp_path, "off-a", {"enabled": False}))
        envelope_a, digest_a, events_a = _run_report(wired)

        control = pinned_config(_make_workspace(tmp_path, "off-b", {"enabled": False}))
        envelope_b, digest_b, events_b = _run_report(control, dewired=True)

        assert envelope_a == envelope_b
        assert sorted(digest_a) == sorted(digest_b)
        assert digest_a == digest_b
        assert events_a == events_b

    def test_the_binding_normalisation_has_teeth(self, tmp_path, pinned_config) -> None:
        """Positive control for ``_digest_bytes``: it normalises two fields, not the file.

        Without this, replacing the binding's digest with a constant would
        look exactly the same from ``test_off_equals_the_wiring_being_absent``,
        and a real change to the attestation would stop being compared.
        """
        ws = Path(pinned_config(_make_workspace(tmp_path, "norm", {"enabled": False})))
        binding = ws / _BINDING_FILE
        assert binding.is_file(), "init did not arm the workspace; this control proves nothing"
        before = _tree_digest(ws)[_BINDING_FILE]

        record = json.loads(binding.read_text(encoding="utf-8"))
        record["spec_hash"] = "0" * len(record["spec_hash"])
        binding.write_text(json.dumps(record), encoding="utf-8")

        assert _tree_digest(ws)[_BINDING_FILE] != before, "a changed spec_hash left the digest unmoved"

    def test_the_binding_normalisation_survives_a_corrupt_file(self, tmp_path, pinned_config) -> None:
        """A binding that will not parse is digested raw, never normalised away."""
        ws = Path(pinned_config(_make_workspace(tmp_path, "corrupt", {"enabled": False})))
        binding = ws / _BINDING_FILE
        before = _tree_digest(ws)[_BINDING_FILE]

        binding.write_text("{ not json", encoding="utf-8")

        assert _tree_digest(ws)[_BINDING_FILE] != before

    def test_the_off_comparison_has_teeth(self, tmp_path, pinned_config) -> None:
        """Positive control: the same comparison FAILS when the flag is on.

        Without this, ``test_off_equals_the_wiring_being_absent`` would pass
        just as happily against a leg that never runs at all.
        """
        on = pinned_config(_make_workspace(tmp_path, "teeth-a", {"enabled": True}))
        _envelope_a, digest_a, _events_a = _run_report(on)

        control = pinned_config(_make_workspace(tmp_path, "teeth-b", {"enabled": True}))
        _envelope_b, digest_b, _events_b = _run_report(control, dewired=True)

        assert digest_a != digest_b
        assert PROFILE_REL_PATH in digest_a
        assert PROFILE_REL_PATH not in digest_b

    def test_the_probe_is_silent_on_a_malformed_config(self, tmp_path, monkeypatch) -> None:
        """The slice-1 lesson, re-pinned on this leg.

        ``is_enabled`` warns ``v4_config_unreadable`` on a config that does
        not parse. A probe that decides an OFF-by-default surface does not
        run must emit nothing, or the flag-off build is observably different
        from the build that never had the feature.
        """
        workspace = str(tmp_path / "malformed")
        os.makedirs(workspace)
        init(workspace)
        cfg = os.path.join(workspace, "mind-mem.json")
        Path(cfg).write_text('{"v4": {"llm_noise_profile": {"enabled": tru', encoding="utf-8")
        monkeypatch.setenv("MIND_MEM_CONFIG", cfg)

        with _mind_mem_stderr() as buffer:
            assert llm_noise_profile.profiling_enabled() is False
            assert llm_noise_profile.reliability_report(workspace) is None
        assert _events(buffer) == []

        # And the loud path still warns, so the quiet one is a real choice
        # rather than a warning that was globally lost.
        from mind_mem.v4 import feature_flags

        feature_flags._last_config_warning = None
        with _mind_mem_stderr() as loud:
            feature_flags.is_enabled(NOISE_PROFILE_FLAG)
        assert "v4_config_unreadable" in _events(loud)
