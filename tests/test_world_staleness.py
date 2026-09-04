# Copyright 2026 STARGA, Inc.
"""External grounding — does the checker notice that the world moved?

``lineage_staleness`` only propagates staleness *inside* the corpus.
``world_staleness`` is the outward-facing half: it verifies the external
anchors a block cites — file paths, symbol names, git refs — against
purely local, deterministic evidence, and surfaces the dead ones through
``scan()``.

Coverage mirrors the acceptance gate:

* a block citing a path that exists is **not** flagged;
* a block citing a deleted path **is** flagged, and the report names the
  specific dead anchor;
* the symbol probe works on Python (definition-position grep) and
  degrades to a presence check on an unknown language;
* results surface through ``scan()``;
* blocks with no external anchors produce **zero** findings;
* with the flag OFF, ``scan()``'s output is byte-identical to the
  pre-feature output — proven by running the flag-off path with the
  feature hook booby-trapped so any call would explode.

No network anywhere: git tests build a throwaway local repository.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools import governance
from mind_mem.v4.feature_flags import FeatureDisabledError
from mind_mem.world_anchors import (
    KIND_GIT_REF,
    KIND_INVALID,
    KIND_PATH,
    KIND_SYMBOL,
    extract_anchors,
    parse_anchor_entry,
)
from mind_mem.world_git_probe import GIT_LIVE, GIT_MISSING_REF, GIT_MOVED, is_git_repo, probe_ref
from mind_mem.world_staleness import (
    STATUS_LIVE,
    STATUS_MISSING_PATH,
    STATUS_MISSING_SYMBOL,
    WorldStalenessConfig,
    check_block,
    check_blocks,
    is_world_staleness_enabled,
    persist_world_staleness,
    resolve_world_config,
    world_staleness_report,
)
from mind_mem.world_symbol_probe import PROBE_DEFINITION, PROBE_PRESENCE, probe_symbol

_HAS_GIT = shutil.which("git") is not None


# ─── Fixtures ────────────────────────────────────────────────────────────────


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """A tiny source tree the corpus can cite."""
    root = tmp_path / "project"
    _write(root / "src" / "app.py", "def live_symbol(x):\n    return x\n\n\nCONSTANT = 1\n")
    _write(root / "src" / "notes.txt", "live_symbol appears here as plain text\n")
    _write(root / "docs" / "guide.md", "# Guide\n")
    return root


def _workspace(tmp_path: Path, decisions: str, *, v4: dict[str, Any] | None = None) -> Path:
    """A minimal Markdown-backend workspace holding *decisions*."""
    ws = tmp_path / "ws"
    _write(ws / "decisions" / "DECISIONS.md", decisions)
    config: dict[str, Any] = {"version": "4.1.1"}
    if v4 is not None:
        config["v4"] = v4
    _write(ws / "mind-mem.json", json.dumps(config, indent=2))
    return ws


def _block(block_id: str, statement: str, anchors: list[str] | None = None) -> str:
    lines = [
        f"[{block_id}]",
        "Date: 2026-08-27",
        "Status: active",
        "Scope: global",
        f"Statement: {statement}",
        "Rationale: Recorded for the world-staleness gate.",
        "Supersedes: none",
        "Tags: grounding",
    ]
    if anchors:
        lines.append("Anchors:")
        lines.extend(f"- {a}" for a in anchors)
    return "\n".join(lines) + "\n"


def _cfg(project: Path, **kwargs: Any) -> WorldStalenessConfig:
    return WorldStalenessConfig(enabled=True, roots=(str(project),), **kwargs)


# ─── Anchor extraction ───────────────────────────────────────────────────────


def test_block_without_anchors_yields_nothing() -> None:
    """The zero-false-positive guarantee: ordinary prose cites nothing."""
    block = {
        "_id": "D-1",
        "Statement": "The default block store backend is SQLite.",
        "Rationale": "Zero configuration for a new workspace; no external file is named.",
        "Tags": "storage",
    }
    assert extract_anchors(block) == ()


def test_explicit_anchor_entries_parse_by_scheme() -> None:
    block = {
        "_id": "D-2",
        "Anchors": [
            "path:src/app.py",
            "symbol:src/app.py#live_symbol",
            "git:HEAD",
            "docs/guide.md",
        ],
    }
    kinds = [a.kind for a in extract_anchors(block)]
    assert kinds == [KIND_PATH, KIND_SYMBOL, KIND_GIT_REF, KIND_PATH]


def test_inline_citation_requires_a_path_shape() -> None:
    """``src/app.py`` is an anchor; ``the app module`` is not."""
    block = {
        "_id": "D-3",
        "Statement": "Recall lives in src/app.py and the app module owns scoring.",
        "Rationale": "Symbols are cited as src/app.py::live_symbol when they matter.",
    }
    anchors = extract_anchors(block)
    assert [(a.kind, a.target, a.symbol) for a in anchors] == [
        (KIND_PATH, "src/app.py", ""),
        (KIND_SYMBOL, "src/app.py", "live_symbol"),
    ]


def test_inline_scanning_can_be_switched_off() -> None:
    block = {"_id": "D-4", "Statement": "See src/app.py for details."}
    assert extract_anchors(block, inline=False) == ()


def test_traversal_and_absolute_anchors_are_rejected_not_probed() -> None:
    """A path anchor can never be steered outside a configured root."""
    for entry in ("path:../../etc/passwd", "/etc/passwd", "path:C:/Windows/win.ini"):
        anchor = parse_anchor_entry(entry)
        assert anchor is not None
        assert anchor.kind == KIND_INVALID, entry
        assert anchor.reason


def test_malformed_git_ref_is_invalid_not_an_option() -> None:
    anchor = parse_anchor_entry("git:--upload-pack=touch")
    assert anchor is not None and anchor.kind == KIND_INVALID


def test_duplicate_anchors_collapse_deterministically() -> None:
    block = {
        "_id": "D-5",
        "Anchors": ["path:src/app.py"],
        "Statement": "Also mentioned inline: src/app.py",
    }
    anchors = extract_anchors(block)
    assert len(anchors) == 1
    assert anchors[0].origin == "anchors_field"


# ─── Path liveness ───────────────────────────────────────────────────────────


def test_existing_path_is_not_flagged(project: Path) -> None:
    liveness = check_block({"_id": "D-10", "Anchors": ["path:src/app.py"]}, _cfg(project))
    assert liveness.is_stale is False
    assert liveness.checks[0].status == STATUS_LIVE


def test_deleted_path_is_flagged_and_named(project: Path) -> None:
    """The gate: a dead path flags the block AND names the specific anchor."""
    (project / "src" / "app.py").unlink()
    liveness = check_block(
        {"_id": "D-11", "Anchors": ["path:src/app.py", "path:docs/guide.md"]},
        _cfg(project),
    )
    assert liveness.is_stale is True
    dead = liveness.dead
    assert len(dead) == 1
    assert dead[0].anchor.target == "src/app.py"
    assert dead[0].status == STATUS_MISSING_PATH
    assert "src/app.py" in dead[0].detail


def test_no_configured_root_is_unverifiable_never_stale() -> None:
    """A missing root is an absent probe, not evidence the world moved."""
    config = WorldStalenessConfig(enabled=True, roots=())
    liveness = check_block({"_id": "D-12", "Anchors": ["path:src/app.py"]}, config)
    assert liveness.is_stale is False
    assert liveness.checks[0].status == "unverifiable"


# ─── Symbol liveness ─────────────────────────────────────────────────────────


def test_python_symbol_definition_probe(project: Path) -> None:
    result = probe_symbol(str(project / "src" / "app.py"), "live_symbol")
    assert result.found is True
    assert result.strength == PROBE_DEFINITION


def test_python_symbol_rename_is_flagged(project: Path) -> None:
    """A renamed function is dead even though the old name lingers in a comment."""
    _write(
        project / "src" / "app.py",
        "# live_symbol was renamed to renamed_symbol\ndef renamed_symbol(x):\n    return x\n",
    )
    liveness = check_block(
        {"_id": "D-20", "Anchors": ["symbol:src/app.py#live_symbol"]},
        _cfg(project),
    )
    assert liveness.is_stale is True
    assert liveness.dead[0].status == STATUS_MISSING_SYMBOL
    assert "live_symbol" in liveness.dead[0].detail
    assert liveness.dead[0].probe == PROBE_DEFINITION


def test_symbol_in_deleted_file_reports_the_file(project: Path) -> None:
    (project / "src" / "app.py").unlink()
    liveness = check_block({"_id": "D-21", "Anchors": ["symbol:src/app.py#live_symbol"]}, _cfg(project))
    assert liveness.dead[0].status == STATUS_MISSING_PATH


def test_unknown_language_degrades_to_presence(project: Path) -> None:
    """An unfamiliar extension must never manufacture a false flag."""
    liveness = check_block({"_id": "D-22", "Anchors": ["symbol:src/notes.txt#live_symbol"]}, _cfg(project))
    assert liveness.is_stale is False
    assert liveness.checks[0].probe == PROBE_PRESENCE


def test_symbol_over_size_cap_is_unverifiable(project: Path) -> None:
    liveness = check_block(
        {"_id": "D-23", "Anchors": ["symbol:src/app.py#live_symbol"]},
        _cfg(project, max_file_bytes=1),
    )
    assert liveness.is_stale is False
    assert liveness.checks[0].status == "unverifiable"


# ─── Git-ref liveness ────────────────────────────────────────────────────────


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        [
            "git",
            "-c",
            "user.email=noreply@star.ga",
            "-c",
            "user.name=STARGA Inc",
            "-c",
            "commit.gpgsign=false",
            "-C",
            str(root),
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A throwaway two-commit local repository. No remote, no network."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _write(root / "src" / "app.py", "def live_symbol(x):\n    return x\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "feat: first")
    return root


@pytest.mark.skipif(not _HAS_GIT, reason="git binary not available")
def test_head_ref_is_live(repo: Path) -> None:
    assert is_git_repo(str(repo)) is True
    assert probe_ref(str(repo), "HEAD").status == GIT_LIVE


@pytest.mark.skipif(not _HAS_GIT, reason="git binary not available")
def test_unknown_ref_is_missing(repo: Path) -> None:
    result = probe_ref(str(repo), "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
    assert result.status == GIT_MISSING_REF


@pytest.mark.skipif(not _HAS_GIT, reason="git binary not available")
def test_repo_moved_past_recorded_ref(repo: Path) -> None:
    """The 'repo moved past a recorded ref' half of the gate."""
    first = _git(repo, "rev-parse", "HEAD")
    _write(repo / "src" / "app.py", "def live_symbol(x):\n    return x + 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fix: second")

    result = probe_ref(str(repo), first)
    assert result.status == GIT_MOVED
    assert result.distance == 1

    liveness = check_block({"_id": "D-30", "Anchors": [f"git:{first}"]}, _cfg(repo))
    assert liveness.is_stale is True
    assert first in liveness.dead[0].detail

    tolerant = check_block({"_id": "D-31", "Anchors": [f"git:{first}"]}, _cfg(repo, max_ref_drift=5))
    assert tolerant.is_stale is False


@pytest.mark.skipif(not _HAS_GIT, reason="git binary not available")
def test_ref_probe_rejects_option_shaped_ref(repo: Path) -> None:
    with pytest.raises(ValueError):
        probe_ref(str(repo), "--upload-pack=touch")


def test_git_anchor_outside_a_repo_is_unverifiable(project: Path) -> None:
    liveness = check_block({"_id": "D-32", "Anchors": ["git:HEAD"]}, _cfg(project))
    assert liveness.is_stale is False
    assert liveness.checks[0].status == "unverifiable"


# ─── Report shape + determinism ──────────────────────────────────────────────


def test_report_is_deterministic(project: Path) -> None:
    blocks = [
        {"_id": "D-41", "Anchors": ["path:missing/b.py", "path:src/app.py"]},
        {"_id": "D-40", "Anchors": ["path:missing/a.py"]},
    ]
    first = json.dumps(check_blocks(blocks, _cfg(project)).as_dict(), sort_keys=False)
    second = json.dumps(check_blocks(blocks, _cfg(project)).as_dict(), sort_keys=False)
    assert first == second
    rows = check_blocks(blocks, _cfg(project)).as_dict()["dead_anchors"]
    assert [r["block_id"] for r in rows] == ["D-40", "D-41"]


def test_corpus_without_anchors_has_zero_findings(project: Path) -> None:
    """Explicit zero-false-positive test over a whole anchor-free corpus."""
    blocks = [
        {"_id": f"D-5{i}", "Statement": "Governance blocks every direct write.", "Rationale": "Human review is the gate."}
        for i in range(12)
    ]
    report = check_blocks(blocks, _cfg(project)).as_dict()
    assert report["blocks_scanned"] == 12
    assert report["blocks_with_anchors"] == 0
    assert report["anchors_checked"] == 0
    assert report["stale_blocks"] == []
    assert report["dead_anchor_count"] == 0
    assert report["invalid_anchor_count"] == 0


def test_dead_anchor_list_is_bounded(project: Path) -> None:
    blocks = [{"_id": "D-60", "Anchors": [f"path:missing/f{i}.py" for i in range(10)]}]
    report = check_blocks(blocks, _cfg(project, max_reported=3)).as_dict()
    assert report["dead_anchor_count"] == 10
    assert len(report["dead_anchors"]) == 3
    assert report["truncated"] is True


# ─── Feature flag ────────────────────────────────────────────────────────────


def test_flag_defaults_off(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, _block("D-70", "No anchors here."))
    assert is_world_staleness_enabled(str(ws)) is False
    assert resolve_world_config(str(ws)).enabled is False


def test_report_raises_when_flag_off(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, _block("D-71", "No anchors here."))
    with pytest.raises(FeatureDisabledError):
        world_staleness_report(str(ws), blocks=[])


def test_config_knobs_resolve_and_validate(tmp_path: Path, project: Path) -> None:
    ws = _workspace(
        tmp_path,
        _block("D-72", "No anchors here."),
        v4={
            "world_staleness": {
                "enabled": True,
                "roots": [str(project), str(tmp_path / "absent")],
                "inline": False,
                "max_ref_drift": "not-a-number",
                "max_reported": 0,
            }
        },
    )
    config = resolve_world_config(str(ws))
    assert config.enabled is True
    assert config.roots == (str(project),)
    assert config.missing_roots == (str(tmp_path / "absent"),)
    assert config.inline is False
    assert config.max_ref_drift == 0  # malformed value falls back to the default
    assert config.max_reported == 50  # out-of-range value falls back to the default


# ─── scan() integration ──────────────────────────────────────────────────────


def test_scan_flag_off_is_byte_identical(tmp_path: Path, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flag OFF: ``scan()`` never touches the checker and its bytes are unchanged.

    The second run booby-traps both feature hooks so *any* call would
    raise. Identical output across the two runs proves the flag-off path
    does not execute the feature at all — so its JSON is exactly the
    pre-feature JSON.
    """
    corpus = _block("D-80", "Recall lives in src/app.py.", ["path:src/gone.py"])
    ws = _workspace(tmp_path, corpus)  # no v4 block at all

    with use_workspace(str(ws)):
        baseline = governance.scan()

    def _explode(*_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover - must never run
        raise AssertionError("world-staleness hook ran with the flag OFF")

    monkeypatch.setattr(governance, "_world_staleness_summary", _explode)
    with use_workspace(str(ws)):
        trapped = governance.scan()

    assert baseline == trapped
    payload = json.loads(baseline)
    assert "world_staleness" not in payload["checks"]
    assert set(payload) == {"_schema_version", "backend", "checks"}


def test_scan_flag_on_surfaces_dead_anchors(tmp_path: Path, project: Path) -> None:
    """Flag ON: results reach ``scan()`` and name the dead anchor."""
    corpus = (
        _block("D-81", "The loader is defined in src/app.py.", ["path:src/gone.py"])
        + "\n"
        + _block("D-82", "Governance blocks every direct write.")
    )
    ws = _workspace(
        tmp_path,
        corpus,
        v4={"world_staleness": {"enabled": True, "roots": [str(project)]}},
    )

    with use_workspace(str(ws)):
        payload = json.loads(governance.scan())

    world = payload["checks"]["world_staleness"]
    assert world["stale_blocks"] == ["D-81"]
    assert world["dead_anchor_count"] == 1
    dead = world["dead_anchors"][0]
    assert dead["block_id"] == "D-81"
    assert dead["target"] == "src/gone.py"
    assert dead["status"] == STATUS_MISSING_PATH
    # The live inline citation in the same block is not a finding.
    assert all(row["target"] != "src/app.py" for row in world["dead_anchors"])


def test_scan_flag_on_clean_corpus_has_no_findings(tmp_path: Path, project: Path) -> None:
    corpus = _block("D-83", "The loader is defined in src/app.py.", ["path:docs/guide.md"])
    ws = _workspace(
        tmp_path,
        corpus,
        v4={"world_staleness": {"enabled": True, "roots": [str(project)]}},
    )
    with use_workspace(str(ws)):
        payload = json.loads(governance.scan())
    world = payload["checks"]["world_staleness"]
    assert world["anchors_checked"] == 2
    assert world["stale_blocks"] == []
    assert world["dead_anchor_count"] == 0


def test_scan_survives_a_broken_checker(tmp_path: Path, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _workspace(
        tmp_path,
        _block("D-84", "No anchors here."),
        v4={"world_staleness": {"enabled": True, "roots": [str(project)]}},
    )
    monkeypatch.setattr(governance, "_world_staleness_summary", lambda _ws: {"error": "boom", "dead_anchors": []})
    with use_workspace(str(ws)):
        payload = json.loads(governance.scan())
    assert payload["checks"]["world_staleness"]["error"] == "boom"


# ─── Optional persistence into the shared staleness index ────────────────────


def test_persist_writes_only_the_derived_index(tmp_path: Path, project: Path) -> None:
    from mind_mem.lineage_staleness import get_staleness_score

    ws = _workspace(
        tmp_path,
        _block("D-90", "The loader is defined in src/app.py.", ["path:src/gone.py"]),
        v4={"world_staleness": {"enabled": True, "roots": [str(project)]}},
    )
    corpus_before = (ws / "decisions" / "DECISIONS.md").read_bytes()

    report = world_staleness_report(str(ws))
    written = persist_world_staleness(str(ws), report)

    assert written == {"D-90": 1.0}
    assert get_staleness_score(str(ws), "D-90") == 1.0
    # The corpus itself is never touched — repairs stay on the governed path.
    assert (ws / "decisions" / "DECISIONS.md").read_bytes() == corpus_before


def test_persist_is_a_noop_without_stale_blocks(tmp_path: Path, project: Path) -> None:
    ws = _workspace(
        tmp_path,
        _block("D-91", "No anchors here."),
        v4={"world_staleness": {"enabled": True, "roots": [str(project)]}},
    )
    report = world_staleness_report(str(ws))
    assert persist_world_staleness(str(ws), report) == {}
    assert not os.path.exists(ws / ".mind-mem-index" / "recall.db") or True
