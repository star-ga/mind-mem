"""Two defects the slice-3 verifier reproduced, pinned so they cannot return.

1. **Flag-off was not inert.** ``change_stream.publish`` probes the
   backpressure flag on every event, and the silent probe re-read AND
   re-parsed ``mind-mem.json`` on every call -- 1000 config reads per 1000
   flag-OFF publishes, 2.5x slower than a build with no config at all. A
   default-OFF deployment must not pay a cost the unwired build never paid.

2. **Recovery was unreachable.** Only ``publish()`` reported depth, so the
   controller could learn the backlog grew but never that it shrank. A stream
   whose queue cap (1024) exceeds the high watermark (1000) tripped overload
   and then pinned there forever once publishing went idle, deferring every
   daemon task on every tick indefinitely.

Each test carries a positive control: a negative assertion whose fixture never
contained the bad case proves nothing.
"""

from __future__ import annotations

import importlib
import json
import pathlib

import pytest


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    path = tmp_path / "mind-mem.json"
    path.write_text(json.dumps({"v4": {}}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(path))
    import mind_mem.v4.feature_flags as ff

    ff._QUIET_CACHE.clear()
    return path


def _write(cfg: pathlib.Path, enabled: bool) -> None:
    cfg.write_text(json.dumps({"v4": {"backpressure": {"enabled": enabled}}}), encoding="utf-8")
    import mind_mem.v4.feature_flags as ff

    ff._QUIET_CACHE.clear()


def _count_reads(cfg, monkeypatch):
    """Count read_text() calls against the config file."""
    seen = {"n": 0}
    orig = pathlib.Path.read_text

    def counting(self, *a, **k):
        if self == cfg:
            seen["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "read_text", counting)
    return seen


class TestFlagOffProbeIsCheap:
    """Defect 1: the silent probe must not re-parse the config per call."""

    def test_a_thousand_flag_off_publishes_cause_at_most_one_config_read(self, cfg, monkeypatch) -> None:
        _write(cfg, False)
        from mind_mem.change_stream import ChangeStream

        stream = ChangeStream()
        stream.subscribe(lambda e: None)
        reads = _count_reads(cfg, monkeypatch)

        for i in range(1000):
            stream.publish("evt", {"i": i})

        assert reads["n"] <= 1, (
            f"flag-OFF publish re-read the config {reads['n']} times; the probe must be cached so a default-OFF build stays inert"
        )

    def test_positive_control_an_edited_config_is_still_picked_up(self, cfg) -> None:
        """The cache may change the COST, never the ANSWER.

        Without this control the test above is satisfiable by a probe that
        caches forever and can never see a flag turn on.
        """
        from mind_mem.v4 import backpressure as bp

        _write(cfg, False)
        assert bp.wiring_enabled() is False

        _write(cfg, True)
        assert bp.wiring_enabled() is True, "the probe cache went stale: an edited config must take effect"


class TestOverloadCanRecover:
    """Defect 2: the drain must report, or overload is a one-way door."""

    def test_a_full_drain_clears_overload(self, cfg) -> None:
        _write(cfg, True)
        import mind_mem.change_stream as csm
        import mind_mem.v4.backpressure as bp

        importlib.reload(bp)
        importlib.reload(csm)

        stream = csm.ChangeStream()
        sub_id = stream.subscribe(lambda e: None)
        for i in range(1200):
            stream.publish("evt", {"i": i})

        # Positive control: the burst really did trip overload, so the
        # assertion below is about recovery and not about an empty fixture.
        assert stream.is_overloaded() is True, "fixture never reached overload"
        assert bp.any_overloaded() is True

        drained = stream.poll(sub_id)
        assert drained, "positive control: the drain returned nothing"

        assert stream.is_overloaded() is False, (
            "still overloaded after a full drain -- the controller never hears "
            "that the backlog shrank, so the daemon defers every task forever"
        )
        assert bp.any_overloaded() is False, "daemon._tick reads any_overloaded(); a stuck True stalls maintenance"
