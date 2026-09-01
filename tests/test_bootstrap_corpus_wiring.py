"""Wiring + quarantine proof for the ``mind-mem-bootstrap`` ingest door.

``bootstrap_corpus`` mines session transcripts, daily logs and the well-known
markdown files into the corpus. Everything it reads originates outside the
workspace, so it is an untrusted door and nothing it writes may be servable.

Three things are proved here, and the third is the one that matters:

1. **Wiring.** The flag is registered, the console script is declared, and the
   four phases actually run and count.
2. **Flag-off.** With the flag off the door writes nothing, reads nothing, and
   its probe emits nothing — the OFF build is indistinguishable from the build
   that never had the door.
3. **Quarantine, with the vacuity controls.** A "recall does not return it"
   assertion is worthless on its own: it passes when the write silently failed,
   and it passes when recall is dead in the fixture. Both controls are here —
   the canary is proved present on disk, and a sibling ACTIVE block carrying a
   different token is proved *retrievable from the same workspace by the same
   call* before the canary is asserted absent.

There is a fourth control, because this door has a property the inbox door does
not: the ``SIG-`` blocks it mints have no scorable text field, so a bare recall
assertion would hold even if the status filter were deleted. The withholding is
therefore proved at ``admit_corpus`` — the exact function every read leg calls —
against the blocks the door really wrote, with a status-flipped twin of the same
block proved admissible. That is a discrimination test, not a tautology.
"""

from __future__ import annotations

import io
import json
import os
import pathlib
from typing import Any

import pytest

from mind_mem.admissibility import admit_corpus
from mind_mem.block_parser import parse_file
from mind_mem.bootstrap_corpus import (
    FLAG,
    INGEST_TIER,
    BootstrapReport,
    _write_summary_admitted,
    flag_enabled,
    main,
    run_bootstrap,
)
from mind_mem.enums import INITIAL_STATUS, is_servable
from mind_mem.init_workspace import init

#: Improbable token planted through the door. Any hit is this content.
CANARY = "qvhzlarkspurmoss"
#: Planted directly as an ACTIVE decision. Proves recall is ALIVE in the
#: fixture, so "the canary was not returned" cannot mean "nothing ever is".
LIVE_TOKEN = "wbrtfennelquartz"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixture — a governed workspace plus a fake HOME holding one transcript
# ---------------------------------------------------------------------------


def _transcript(path: str, token: str, n: int = 8) -> None:
    lines = [
        json.dumps(
            {
                "message": {
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"We decided to adopt the {token} protocol for message {i}.",
                }
            }
        )
        for i in range(n)
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _set_flag(ws: str, on: bool) -> str:
    cfg_path = os.path.join(ws, "mind-mem.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg.setdefault("v4", {})[FLAG] = {"enabled": on}
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    return cfg_path


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point HOME at an empty directory for EVERY test in this file.

    Not hygiene — safety, and it was found the hard way. Phase 1 walks
    ``~/.claude/projects`` and phase 3 reads ``~/CLAUDE.md``. Mutating the flag
    check away to prove the flag-off tests catch it did not fail them: it sent
    the suite off to scan the operator's real transcript corpus, and the run
    hung instead of going red. A test whose failure mode is "read the whole
    home directory" is a worse outcome than the bug it was hunting, so the
    isolation is unconditional and the flag check is no longer the only thing
    standing between this suite and a real filesystem.
    """
    empty = tmp_path / "empty_home"
    empty.mkdir()
    monkeypatch.setenv("HOME", str(empty))


@pytest.fixture
def door(tmp_path, monkeypatch):
    """A workspace, a fake HOME with one canary transcript, flag ON."""
    home = tmp_path / "home"
    (home / ".claude" / "projects" / "p1").mkdir(parents=True)
    ws = str(tmp_path / "ws")
    init(ws)

    _transcript(str(home / ".claude" / "projects" / "p1" / "session.jsonl"), CANARY)
    # A daily log (phase 2) and a CLAUDE.md (phase 3) so all four phases have
    # something to find rather than short-circuiting on an empty directory.
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")
    with open(os.path.join(ws, "memory", f"{today}.md"), "w", encoding="utf-8") as fh:
        fh.write(f"We decided to keep the {CANARY} log line.\n")
    with open(home / "CLAUDE.md", "w", encoding="utf-8") as fh:
        fh.write(f"We need to document the {CANARY} convention.\n")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("MIND_MEM_CONFIG", _set_flag(ws, True))
    return ws


def _corpus_blocks(ws: str) -> list[dict[str, Any]]:
    """Every parsed block in the workspace, from every markdown file."""
    out: list[dict[str, Any]] = []
    for path in pathlib.Path(ws).rglob("*.md"):
        if not path.is_file():
            continue
        try:
            out.extend(parse_file(str(path)))
        except Exception:  # noqa: BLE001 - an unparseable file is not a hit
            continue
    return out


def _canary_blocks(ws: str) -> list[dict[str, Any]]:
    return [b for b in _corpus_blocks(ws) if CANARY in json.dumps(b, default=str)]


def _canary_on_disk(ws: str) -> bool:
    for path in pathlib.Path(ws).rglob("*"):
        if not path.is_file():
            continue
        try:
            if CANARY in path.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            continue
    return False


def _seed_live_block(ws: str) -> None:
    """An ACTIVE, scorable block. The anti-vacuity control for recall."""
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as fh:
        fh.write(
            f"\n\n[D-20260901-900]\nDate: 2026-09-01\nStatus: active\nScope: global\n"
            f"Statement: We adopt the {LIVE_TOKEN} protocol for verification\n"
            f"Rationale: control block\nTags: control\n\n"
        )


# ---------------------------------------------------------------------------
# 1 — wiring
# ---------------------------------------------------------------------------


def test_the_flag_is_registered() -> None:
    """An unregistered name is fail-closed in feature_flags, i.e. dead."""
    from mind_mem.v4.feature_flags import ALL_V4_FLAGS

    assert FLAG in ALL_V4_FLAGS


def test_the_console_script_is_declared() -> None:
    """The plug point. Without it the module has no caller and no reachability."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'mind-mem-bootstrap = "mind_mem.bootstrap_corpus:main"' in text


def test_the_door_is_documented() -> None:
    """A surface nothing teaches is a surface nobody can use."""
    plan = (REPO_ROOT / "docs" / "plans" / "RESTORE-44-WIRING-PLAN.md").read_text(encoding="utf-8")
    assert "mind-mem-bootstrap" in plan

    cli = (REPO_ROOT / "docs" / "cli-reference.md").read_text(encoding="utf-8")
    assert "mind-mem-bootstrap" in cli, "the console script has no user-facing documentation"
    # The two facts an operator must not have to discover by experiment.
    assert f'"{FLAG}"' in cli, "the docs do not name the flag that turns the door on"
    assert "pending" in cli.split("mind-mem-bootstrap")[-1], "the docs do not say what status the door mints"


# ---------------------------------------------------------------------------
# 2 — flag OFF is byte-identical to not having the door
# ---------------------------------------------------------------------------


def test_main_refuses_when_the_flag_is_off(tmp_path, monkeypatch, capsys) -> None:
    ws = str(tmp_path / "ws")
    init(ws)
    monkeypatch.setenv("MIND_MEM_CONFIG", _set_flag(ws, False))

    before = {p: p.read_bytes() for p in pathlib.Path(ws).rglob("*") if p.is_file()}
    assert main([ws]) == 2
    after = {p: p.read_bytes() for p in pathlib.Path(ws).rglob("*") if p.is_file()}

    assert after == before, "an OFF door changed the workspace"
    assert FLAG in capsys.readouterr().err


def test_flag_off_reads_nothing_at_all(tmp_path, monkeypatch) -> None:
    """The check runs BEFORE any source is opened, not after the scan."""
    ws = str(tmp_path / "ws")
    init(ws)
    monkeypatch.setenv("MIND_MEM_CONFIG", _set_flag(ws, False))

    def _boom(*_a, **_k):
        raise AssertionError("an OFF door read a transcript")

    monkeypatch.setattr("mind_mem.bootstrap_corpus.find_recent_transcripts", _boom)
    monkeypatch.setattr("mind_mem.bootstrap_corpus.find_all_logs", _boom)
    assert main([ws]) == 2


def test_a_missing_flag_is_the_same_as_a_false_one(tmp_path, monkeypatch) -> None:
    """Default-OFF means absent, not just explicitly false."""
    ws = str(tmp_path / "ws")
    init(ws)
    cfg = os.path.join(ws, "mind-mem.json")
    monkeypatch.setenv("MIND_MEM_CONFIG", cfg)
    assert not flag_enabled()
    assert main([ws]) == 2


def test_the_off_probe_is_silent(tmp_path, monkeypatch, capsys) -> None:
    """A probe that logs makes the OFF build observably different.

    The loud resolver warns ``v4_config_unreadable`` on a malformed config.
    This door probes through the quiet one, so a broken config on an OFF path
    still emits nothing.
    """
    bad = tmp_path / "mind-mem.json"
    bad.write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(bad))
    capsys.readouterr()

    assert flag_enabled() is False

    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == "", f"the OFF probe spoke: {captured}"


# ---------------------------------------------------------------------------
# 3 — the four phases actually run (the smoke test the plan asked for)
# ---------------------------------------------------------------------------


def test_dry_run_reports_signals_and_writes_nothing(door) -> None:
    """``--dry-run reports N signals`` — the plan's first acceptance clause."""
    before = {p: p.read_bytes() for p in pathlib.Path(door).rglob("*") if p.is_file()}
    buf = io.StringIO()
    report = run_bootstrap(door, dry_run=True, out=buf)
    after = {p: p.read_bytes() for p in pathlib.Path(door).rglob("*") if p.is_file()}

    assert isinstance(report, BootstrapReport)
    assert report.signals_detected > 0, "a dry run over a seeded workspace found nothing"
    assert report.signals_written == 0
    assert after == before, "a dry run wrote to the corpus"
    assert "DRY RUN" in buf.getvalue()


def test_all_four_phases_report_a_source(door) -> None:
    """Each phase found its seeded input, so no phase is silently a no-op."""
    buf = io.StringIO()
    report = run_bootstrap(door, out=buf)
    text = buf.getvalue()

    assert report.transcripts == 1, "phase 1 found no transcript"
    assert report.logs == 1, "phase 2 found no daily log"
    assert report.markdown_files == 1, "phase 3 found no markdown file"
    assert report.summaries_created == 1
    assert report.signals_written > 0
    for phase in ("Phase 1:", "Phase 2:", "Phase 3:", "Phase 4:"):
        assert phase in text


def test_the_backfill_is_re_runnable(door) -> None:
    """Content-hash dedup: a second pass mints nothing new."""
    buf = io.StringIO()
    first = run_bootstrap(door, out=buf)
    second = run_bootstrap(door, out=buf)
    assert first.signals_written > 0
    assert second.signals_written == 0, "re-running the backfill duplicated signals"


def test_main_returns_zero_when_the_flag_is_on(door) -> None:
    assert main([door, "--dry-run"]) == 0


# ---------------------------------------------------------------------------
# 4 — the canary. Positive controls first, then the withholding.
# ---------------------------------------------------------------------------


def test_the_canary_reaches_disk(door) -> None:
    """POSITIVE CONTROL. Everything below is vacuous without this."""
    run_bootstrap(door, out=io.StringIO())
    assert _canary_on_disk(door), "nothing was written at all; the door is a no-op"
    assert _canary_blocks(door), "the canary is on disk but not inside a parsed block"


def test_every_canary_block_is_pending(door) -> None:
    """Withheld by STATUS, not by an index that has not caught up."""
    run_bootstrap(door, out=io.StringIO())
    blocks = _canary_blocks(door)
    assert blocks
    statuses = {str(b.get("Status", "")).strip().lower() for b in blocks}
    assert statuses == {"pending"}, f"expected every block pending, got {statuses}"
    assert not any(is_servable(s) for s in statuses)


def test_admit_corpus_withholds_every_block_the_door_wrote(door) -> None:
    """The read-side filter, run on the real blocks, with a discrimination twin.

    ``admit_corpus`` is what every block-reading leg calls. Feeding it the
    door's own output proves the withholding; feeding it the same blocks with
    the status flipped proves the filter is doing the work, rather than the
    blocks being unreachable for some unrelated reason.
    """
    run_bootstrap(door, out=io.StringIO())
    blocks = _canary_blocks(door)
    assert blocks, "positive control failed: no canary blocks to filter"

    assert admit_corpus(blocks) == [], "a pending block the door wrote was admitted"

    flipped = [{**b, "Status": "active"} for b in blocks]
    assert len(admit_corpus(flipped)) == len(blocks), (
        "the discrimination control failed: these blocks are inadmissible even "
        "as ACTIVE, so the empty result above proves nothing about the status filter"
    )


def test_recall_withholds_the_canary_while_it_still_serves(door) -> None:
    """End to end, with the anti-vacuity control in the SAME call shape."""
    from mind_mem._recall_core import recall

    run_bootstrap(door, out=io.StringIO())
    _seed_live_block(door)

    live = recall(door, LIVE_TOKEN, limit=25)
    assert any(LIVE_TOKEN in json.dumps(h, default=str) for h in live), (
        "recall is dead in this fixture; a 'canary not found' assertion would pass trivially"
    )

    for query in (CANARY, f"{CANARY} protocol", "decided adopt protocol"):
        for hit in recall(door, query, limit=25):
            assert CANARY not in json.dumps(hit, default=str), f"the door's content reached recall via {query!r}"


def test_the_summary_sink_is_outside_the_recall_corpus(door) -> None:
    """Closure, not vigilance: ``summaries/`` is not a corpus directory."""
    from mind_mem.corpus_registry import CORPUS_DIRS

    assert "summaries" not in CORPUS_DIRS
    run_bootstrap(door, out=io.StringIO())
    daily = pathlib.Path(door) / "summaries" / "daily"
    written = [p for p in daily.glob("*.md") if CANARY in p.read_text(encoding="utf-8")]
    assert written, "positive control failed: no summary was written"
    for block in (b for p in written for b in parse_file(str(p))):
        assert not is_servable(block.get("Status")), f"a servable summary block: {block.get('Status')}"


# ---------------------------------------------------------------------------
# 5 — the gate: every write leg is admitted, under a quarantining tier
# ---------------------------------------------------------------------------


def test_the_doors_tier_cannot_mint_a_servable_status() -> None:
    """The one table that decides this. PROPOSAL_APPLY is the only exception."""
    assert not is_servable(INITIAL_STATUS[INGEST_TIER])


class _RecordingGate:
    """Delegates to the real gate, and records what it was asked to admit."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.admissions: list[dict] = []

    def admit_block(self, **kwargs):
        self.admissions.append(kwargs)
        return self._inner.admit_block(**kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_the_summary_write_happens_inside_an_admission(door, monkeypatch) -> None:
    """Not merely 'a receipt was minted' — the write runs while it is open."""
    from mind_mem.admission import current_admission
    from mind_mem.governance_gate import get_gate as _real_get_gate

    recorder: dict[str, Any] = {}

    def _wrapped(ws: str):
        gate = recorder.setdefault("gate", _RecordingGate(_real_get_gate(ws)))
        return gate

    seen: list[Any] = []
    real_write = __import__("mind_mem.session_summarizer", fromlist=["write_summary"]).write_summary

    def _spy(*args, **kwargs):
        seen.append(current_admission())
        return real_write(*args, **kwargs)

    monkeypatch.setattr("mind_mem.governance_gate.get_gate", _wrapped)
    monkeypatch.setattr("mind_mem.bootstrap_corpus.write_summary", _spy)

    run_bootstrap(door, out=io.StringIO())

    gate = recorder["gate"]
    assert gate.admissions, "the summary leg opened no admission"
    assert all(a["tier"] is INGEST_TIER for a in gate.admissions)
    assert seen and all(r is not None for r in seen), "write_summary ran with no admission open"
    assert all(r.tier is INGEST_TIER for r in seen if r is not None)


def test_the_summary_leg_is_fail_closed_without_a_gate(door, monkeypatch) -> None:
    """No gate, no write. An ingest door must not fall through to a raw append."""

    def _no_gate(_ws: str):
        raise RuntimeError("gate unavailable")

    monkeypatch.setattr("mind_mem.governance_gate.get_gate", _no_gate)
    assert _write_summary_admitted(door, "/nonexistent/t.jsonl", [{"role": "user", "content": "x"}]) is None
    assert not list((pathlib.Path(door) / "summaries" / "daily").glob("*.md"))


def test_the_signal_leg_admits_before_the_bytes_land(door, monkeypatch) -> None:
    """A refused admission must abort the append, not annotate it.

    ``capture.append_signals`` opens the scope above the file open. Making the
    gate refuse proves the ordering: the corpus is untouched.
    """
    from mind_mem.governance_gate import GovernanceBypassError

    signals_path = pathlib.Path(door) / "intelligence" / "SIGNALS.md"
    before = signals_path.read_bytes()

    class _RefusingGate:
        def admit_batch(self, **_kwargs):
            raise GovernanceBypassError("refused for the test")

    monkeypatch.setattr("mind_mem.capture._get_gate", lambda _ws: _RefusingGate())
    with pytest.raises(GovernanceBypassError):
        run_bootstrap(door, out=io.StringIO())

    assert signals_path.read_bytes() == before, "a refused admission still wrote signals"
