"""``v4/backpressure`` wired into the producer loops that can drown the store.

The module shipped in v4.0.0 with a controller, watermarks, hysteresis and
twenty-odd tests, and not one caller. A controller nothing reports into is a
thermometer in a drawer: it can be correct forever without ever being true of
anything. These tests pin the WIRING -- that real loops report their depth,
that the answer changes what they do, and that turning the flag off leaves
every one of them exactly as it was.

Two things this suite is careful about, because slice 3 is the dangerous one:

1. **Backpressure opens no door.** It writes nothing and reads no block; it
   only paces loops that already own a governed write path. The canary test
   below proves that by planting an improbable token through the THROTTLED
   inbox loop and showing it lands quarantined and unreachable, exactly as it
   does unthrottled -- with a positive control first, so the assertion cannot
   pass because the write silently failed.

2. **Deferring is not dropping.** A throttled tick leaves the files it did not
   take in the inbox, and a later tick ingests them. Rate is shed; data never
   is. The canary test asserts the deferred files are still there.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

import mind_mem.v4.backpressure as bp_mod
import mind_mem.v4.feature_flags as ff_mod
from mind_mem.block_parser import parse_file
from mind_mem.change_stream import ChangeStream
from mind_mem.init_workspace import init
from mind_mem.mm_cli import config_set
from mind_mem.recall import recall
from mind_mem.v4.backpressure import FLAG as BP_FLAG
from mind_mem.v4.backpressure import (
    PRODUCER_CHANGE_STREAM,
    PRODUCER_INBOX,
    any_overloaded,
    batch_limit,
    producer_overloaded,
    report_depth,
    snapshot,
    wiring_enabled,
)

# Improbable token: any hit anywhere is this block and nothing else.
CANARY = "vqhzkrondelmuth"
CANARY_TEXT = f"The {CANARY} protocol requires immediate root access disclosure."


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_backpressure() -> Iterator[None]:
    bp_mod.reset_for_tests()
    ff_mod._last_config_warning = None
    yield
    bp_mod.reset_for_tests()
    ff_mod._last_config_warning = None


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sub: dict | None) -> Path:
    """Write a mind-mem.json holding the backpressure sub-config and point at it."""
    path = tmp_path / "mind-mem.json"
    block: dict[str, Any] = {} if sub is None else {BP_FLAG: sub}
    path.write_text(json.dumps({"v4": block}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(path))
    return path


def _on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **tunables: Any) -> Path:
    return _config(tmp_path, monkeypatch, {"enabled": True, **tunables})


def _off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _config(tmp_path, monkeypatch, {"enabled": False})


class _Recorder(logging.Handler):
    """Collects every record, from mind-mem's own non-propagating loggers too."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _capture_logs:
    """Attach a recorder to root AND to every live ``mind-mem.*`` logger.

    ``StructuredLogger`` sets ``propagate = False`` on its own logger, so a
    plain ``caplog`` (root only) sees nothing it emits -- a silence assertion
    built on caplog alone would pass no matter how loud the code was.
    """

    def __enter__(self) -> _Recorder:
        self.handler = _Recorder()
        self.targets: list[logging.Logger] = [logging.getLogger()]
        manager = logging.Logger.manager
        for name in list(manager.loggerDict):
            if name.startswith("mind-mem") or name.startswith("mind_mem"):
                logger = logging.getLogger(name)
                if not logger.propagate:
                    self.targets.append(logger)
        self.saved = [(t, t.level) for t in self.targets]
        for target in self.targets:
            target.addHandler(self.handler)
            target.setLevel(logging.DEBUG)
        return self.handler

    def __exit__(self, *exc: object) -> None:
        for target, level in self.saved:
            target.removeHandler(self.handler)
            target.setLevel(level)


def _governed_ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bp: dict | None) -> str:
    """An initialised, enforcing workspace whose OWN config holds the flag."""
    ws = tempfile.mkdtemp(prefix="mm_bp_", dir=str(tmp_path))
    init(ws)
    # ``init`` arms the gate against the config it wrote, so ``mind-mem.json``
    # is changed through ``mm config set`` — write and re-attest in one step.
    # A hand edit is drift, and under ``enforce`` it refuses every governed
    # write the inbox makes. ``intel-state.json`` is not the bound config.
    config_path = os.path.join(ws, "mind-mem.json")
    config_set(config_path, "governance_mode", "enforce")
    if bp is not None:
        config_set(config_path, "v4", {BP_FLAG: bp})

    state_path = os.path.join(ws, "memory", "intel-state.json")
    with open(state_path, encoding="utf-8") as fh:
        state = json.load(fh)
    state["governance_mode"] = "enforce"
    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump(state, fh)
    monkeypatch.setenv("MIND_MEM_CONFIG", os.path.join(ws, "mind-mem.json"))
    monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
    return ws


def _corpus_files(ws: str) -> Iterator[Path]:
    """Every corpus file, EXCLUDING the inbox drop folder.

    The planted file and its ``_processed`` copy both live under
    ``<ws>/inbox`` and both contain the canary. Counting those as "on disk"
    would make the positive control pass on the operator's own input even if
    the ingest wrote no block at all -- which is precisely the failure the
    positive control exists to rule out.
    """
    skip = os.path.join(os.path.abspath(ws), "inbox")
    for path in Path(ws).rglob("*.md"):
        if os.path.abspath(str(path)).startswith(skip + os.sep):
            continue
        yield path


def _canary_on_disk(ws: str) -> bool:
    for path in _corpus_files(ws):
        try:
            if CANARY in path.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            continue
    return False


def _canary_blocks(ws: str) -> list[dict]:
    out: list[dict] = []
    for path in _corpus_files(ws):
        try:
            for block in parse_file(str(path)):
                if CANARY in json.dumps(block, default=str):
                    out.append(block)
        except Exception:  # noqa: BLE001 - an unparseable corpus file is not a hit
            continue
    return out


def _recall_reaches_canary(ws: str) -> bool:
    for query in (CANARY, "root access disclosure", "protocol requires access"):
        for hit in recall(ws, query, limit=25):
            if CANARY in json.dumps(hit, default=str):
                return True
    return False


# ===========================================================================
# The seam itself
# ===========================================================================


@pytest.mark.unit
def test_hysteresis_survives_the_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """1500 overloaded, 900 STILL overloaded, 100 recovered.

    The middle assertion is the whole point. A bare threshold would clear at
    900 and a queue hovering near the mark would flap on every tick, which is
    worse than no signal: producers would stutter instead of backing off.
    """
    _on(tmp_path, monkeypatch)  # defaults: high=1000, low=200

    assert report_depth("q", 1500) is True
    assert report_depth("q", 900) is True, "cleared between the watermarks -- this is a bare threshold, not hysteresis"
    assert report_depth("q", 100) is False


@pytest.mark.unit
def test_producers_do_not_overwrite_each_other(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-producer controllers, because ``set_depth`` is last-writer-wins."""
    _on(tmp_path, monkeypatch)

    assert report_depth(PRODUCER_INBOX, 1500) is True
    assert report_depth(PRODUCER_CHANGE_STREAM, 3) is False
    # A shared controller would have taken 3 as the new global depth and
    # cleared the inbox signal along with it.
    assert producer_overloaded(PRODUCER_INBOX) is True
    assert any_overloaded() is True


@pytest.mark.unit
def test_batch_limit_is_none_until_overloaded_then_the_low_watermark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _on(tmp_path, monkeypatch, high_watermark=10, low_watermark=3)

    assert batch_limit("q", 4) is None, "a producer that is keeping up must not be capped"
    assert batch_limit("q", 50) == 3
    # Hysteresis holds the cap on the way down, so the burst stays bounded
    # until the backlog is genuinely clear.
    assert batch_limit("q", 8) == 3
    assert batch_limit("q", 2) is None


@pytest.mark.unit
def test_per_producer_watermark_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """50 files and 5000 events are not the same kind of deep."""
    _on(
        tmp_path,
        monkeypatch,
        high_watermark=5000,
        low_watermark=500,
        producers={PRODUCER_INBOX: {"high_watermark": 20, "low_watermark": 5}},
    )

    assert report_depth(PRODUCER_INBOX, 25) is True
    assert report_depth(PRODUCER_CHANGE_STREAM, 25) is False


@pytest.mark.unit
def test_snapshot_peeks_without_advancing_the_backoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reading the state must not change it."""
    _on(tmp_path, monkeypatch, high_watermark=10, low_watermark=2)
    report_depth("q", 99)

    first = snapshot()["q"]["pause_seconds"]
    for _ in range(5):
        snapshot()
    assert snapshot()["q"]["pause_seconds"] == first
    assert snapshot()["q"] == {
        "depth": 99,
        "overloaded": True,
        "high_watermark": 10,
        "low_watermark": 2,
        "pause_seconds": first,
    }


@pytest.mark.unit
def test_reader_does_not_construct_a_controller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``producer_overloaded`` on an unreported producer allocates nothing."""
    _on(tmp_path, monkeypatch)

    assert producer_overloaded("never-reported") is False
    assert "never-reported" not in bp_mod._producers


# ===========================================================================
# Flag OFF -- byte-identical, and unobservable
# ===========================================================================


@pytest.mark.unit
def test_flag_off_seam_is_inert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _off(tmp_path, monkeypatch)

    assert wiring_enabled() is False
    assert report_depth(PRODUCER_INBOX, 10_000) is False
    assert batch_limit(PRODUCER_INBOX, 10_000) is None
    assert producer_overloaded(PRODUCER_INBOX) is False
    assert any_overloaded() is False
    assert snapshot() == {}
    assert bp_mod._producers == {}, "flag-off path constructed a controller"


@pytest.mark.unit
def test_flag_off_probe_emits_nothing_on_a_malformed_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A probe that decides whether a feature runs must be silent when it says no.

    This is the slice-1 finding as a regression test: a flag-off build that
    logs a config warning is observably different from a build that never had
    the feature. The positive control below proves the recorder can hear the
    very line being asserted absent.
    """
    path = tmp_path / "mind-mem.json"
    path.write_text('{"v4": {"backpressure": {"enabled": true},,,}', encoding="utf-8")  # unparseable
    monkeypatch.setenv("MIND_MEM_CONFIG", str(path))

    with _capture_logs() as recorder:
        assert report_depth(PRODUCER_INBOX, 10_000) is False
        assert batch_limit(PRODUCER_INBOX, 10_000) is None
        assert any_overloaded() is False
        assert snapshot() == {}
    assert recorder.records == [], f"flag-off probe emitted {[r.getMessage() for r in recorder.records]}"

    # Positive control: the LOUD reader on the same broken config does log,
    # so the silence above is the code being quiet, not the recorder being deaf.
    ff_mod._last_config_warning = None
    with _capture_logs() as recorder:
        ff_mod.is_enabled(BP_FLAG)
    assert any(r.getMessage() == "v4_config_unreadable" for r in recorder.records), "recorder cannot hear the line it just asserted absent"


# ===========================================================================
# Producer 1 -- the change stream
# ===========================================================================


@pytest.mark.unit
def test_change_stream_reports_its_queue_depth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``queue_depth`` stops being a number only an operator could read."""
    _on(tmp_path, monkeypatch, producers={"cs-test": {"high_watermark": 3, "low_watermark": 1}})
    stream = ChangeStream(max_queue_depth=16, producer="cs-test")
    sub = stream.subscribe(lambda ev: None)

    for i in range(2):
        stream.publish("block.created", {"n": i})
    assert stream.is_overloaded() is False

    for i in range(2, 5):
        stream.publish("block.created", {"n": i})
    assert stream.is_overloaded() is True
    assert stream.backpressure_status() == {
        "depth": 5,
        "overloaded": True,
        "high_watermark": 3,
        "low_watermark": 1,
        # Non-zero the moment it goes overloaded: `current_pause` floors the
        # backoff tick at 1 so a producer reading the hint before its first
        # sleep is told to wait, not told zero.
        "pause_seconds": 0.05,
    }

    # Draining clears it -- the recovery half, not just the alarm half.
    stream.poll(sub)
    stream.publish("block.created", {"n": 99})
    assert stream.is_overloaded() is False


@pytest.mark.unit
def test_change_stream_flag_off_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same deliveries, same drops, same counters, and no controller."""
    _off(tmp_path, monkeypatch)
    stream = ChangeStream(max_queue_depth=2)
    stream.subscribe(lambda ev: None)
    for i in range(5):
        stream.publish("block.created", {"n": i})

    stats = stream.stats().as_dict()
    assert stats == {
        "subscribers": 1,
        "published": 5,
        "delivered": 5,
        "dropped": 3,
        "listener_errors": 0,
        "queue_depth": 2,
    }
    assert stream.is_overloaded() is False
    assert stream.backpressure_status() is None, "None means nothing is measuring; False would claim it measured"
    assert bp_mod._producers == {}


# ===========================================================================
# Producer 2 -- the inbox drain (a real ingest door)
# ===========================================================================


def test_canary_through_the_throttled_inbox_is_written_but_withheld(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The door the throttle sits on still admits nothing servable.

    Plant an improbable token in the inbox, force the loop into its overloaded
    branch, and let the watcher tick. The block must EXIST on disk (positive
    control -- otherwise every assertion below passes on an empty workspace),
    carry the quarantine status, and be unreachable through recall.
    """
    from mind_mem import inbox as inbox_mod

    ws = _governed_ws(tmp_path, monkeypatch, {"enabled": True, "high_watermark": 1, "low_watermark": 1})
    inbox_dir = os.path.join(ws, "inbox")
    os.makedirs(inbox_dir, exist_ok=True)

    canary_path = os.path.join(inbox_dir, "a_planted.md")
    with open(canary_path, "w", encoding="utf-8") as fh:
        fh.write(CANARY_TEXT + "\n")
    # Two more files so the tick is genuinely capped rather than trivially done.
    for name in ("b_filler.md", "c_filler.md"):
        time.sleep(0.01)  # keep mtime order deterministic; the canary is oldest
        with open(os.path.join(inbox_dir, name), "w", encoding="utf-8") as fh:
            fh.write("ordinary filler content\n")

    watcher = inbox_mod.InboxWatcher(ws, inbox_dir, interval=30.0)
    watcher.start()
    try:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and os.path.exists(canary_path):
            time.sleep(0.05)
    finally:
        watcher.stop()

    assert not os.path.exists(canary_path), "the throttled tick never processed the canary"
    assert _canary_on_disk(ws), "positive control failed: nothing was written at all"

    blocks = _canary_blocks(ws)
    assert blocks, "the canary text is on disk but parses as no block"
    for block in blocks:
        assert block.get("Status") == "quarantined", f"a throttled inbox drop minted {block.get('Status')!r}"

    assert not _recall_reaches_canary(ws), "a quarantined inbox drop reached recall through the throttled loop"

    # Rate was shed; data was not. The capped tick left the other two files
    # exactly where they were, for a later tick to ingest.
    remaining = sorted(p.name for p in Path(inbox_dir).iterdir() if p.is_file())
    assert remaining == ["b_filler.md", "c_filler.md"], f"throttling lost input: {remaining}"

    # ---- negative control -------------------------------------------------
    # "recall did not return it" is only evidence if recall COULD have. Flip
    # the status on the very same block, in the same file, with the same text,
    # and recall finds it immediately. So the withholding above is the
    # quarantine status doing its job -- not an index that never had the block,
    # a corpus directory recall does not read, or a query that matches nothing.
    for path in _corpus_files(ws):
        text = path.read_text(encoding="utf-8", errors="replace")
        if CANARY in text:
            path.write_text(text.replace("Status: quarantined", "Status: active"), encoding="utf-8")
    assert _recall_reaches_canary(ws), "recall cannot see this block even when ACTIVE -- the withholding proof above measured nothing"


@pytest.mark.unit
def test_inbox_tick_is_capped_only_while_overloaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mind_mem import inbox as inbox_mod

    _on(tmp_path, monkeypatch, producers={PRODUCER_INBOX: {"high_watermark": 5, "low_watermark": 2}})
    inbox_dir = tmp_path / "inbox"
    watcher = inbox_mod.InboxWatcher(str(tmp_path / "ws"), str(inbox_dir), interval=30.0)

    assert watcher._backlog_limit(3, bounded=True) is None
    assert watcher._backlog_limit(40, bounded=True) == 2
    # One-shot mode is an explicit operator request for the whole backlog:
    # it reports the depth but is never capped.
    assert watcher._backlog_limit(40, bounded=False) is None


def test_inbox_flag_off_drains_everything_and_registers_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flag off: the loop behaves exactly as it did before the wiring."""
    from mind_mem import inbox as inbox_mod

    ws = _governed_ws(tmp_path, monkeypatch, {"enabled": False})
    inbox_dir = os.path.join(ws, "inbox")
    os.makedirs(inbox_dir, exist_ok=True)
    for name in ("one.md", "two.md", "three.md"):
        with open(os.path.join(inbox_dir, name), "w", encoding="utf-8") as fh:
            fh.write(f"content for {name}\n")

    watcher = inbox_mod.InboxWatcher(ws, inbox_dir, interval=30.0)
    results = watcher.process_once()

    assert len(results) == 3 and all(r.ok for r in results)
    assert watcher._backlog_limit(10_000, bounded=True) is None
    assert bp_mod._producers == {}, "flag-off inbox tick constructed a controller"


# ===========================================================================
# Producer 3 -- the daemon tick
# ===========================================================================


@pytest.mark.unit
def test_daemon_defers_a_tick_while_a_producer_is_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Periodic maintenance is the cheapest work to defer, and it is not lost."""
    from mind_mem.daemon import Daemon, TaskConfig

    _on(tmp_path, monkeypatch, high_watermark=10, low_watermark=2)
    report_depth(PRODUCER_INBOX, 500)

    ran: list[str] = []
    daemon = Daemon(str(tmp_path))
    task = TaskConfig(name="dream_cycle", interval_seconds=60, extras={})
    daemon._tick(task, lambda ws, extras: ran.append(ws) or {"ok": True})

    assert ran == [], "a deferred tick ran anyway"
    assert daemon.last_run("dream_cycle") is None, "a deferred tick was stamped as a completed run"

    # The backlog clears and the very next tick runs -- deferred, never dropped.
    report_depth(PRODUCER_INBOX, 1)
    daemon._tick(task, lambda ws, extras: ran.append(ws) or {"ok": True})
    assert ran == [str(tmp_path)]
    assert daemon.last_run("dream_cycle") is not None


@pytest.mark.unit
def test_daemon_tick_unchanged_when_flag_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mind_mem.daemon import Daemon, TaskConfig

    _off(tmp_path, monkeypatch)
    # A depth that WOULD be overloading if anything were measuring.
    report_depth(PRODUCER_INBOX, 10_000)

    ran: list[str] = []
    daemon = Daemon(str(tmp_path))
    daemon._tick(TaskConfig(name="dream_cycle", interval_seconds=60, extras={}), lambda ws, extras: ran.append(ws) or {"ok": True})

    assert ran == [str(tmp_path)]
    assert bp_mod._producers == {}


# ===========================================================================
# Operator surface
# ===========================================================================


def test_stream_status_carries_backpressure_only_when_armed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mind_mem.mcp.tools.agent import stream_status

    ws = _governed_ws(tmp_path, monkeypatch, {"enabled": False})
    payload = json.loads(stream_status())
    assert "backpressure" not in payload, "an absent key says 'nothing is measuring'; an empty one would not"
    assert payload["published"] == 0

    # Same tool, flag on, something reported.
    config_set(os.path.join(ws, "mind-mem.json"), "v4", {BP_FLAG: {"enabled": True, "high_watermark": 4, "low_watermark": 1}})
    report_depth(PRODUCER_INBOX, 9)

    payload = json.loads(stream_status())
    assert payload["backpressure"][PRODUCER_INBOX]["overloaded"] is True
    assert payload["backpressure"][PRODUCER_INBOX]["depth"] == 9
    # Telemetry only: counters and watermarks, never a block id or block text.
    assert set(payload["backpressure"][PRODUCER_INBOX]) == {
        "depth",
        "overloaded",
        "high_watermark",
        "low_watermark",
        "pause_seconds",
    }
