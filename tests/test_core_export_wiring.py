# Copyright 2026 STARGA, Inc.
"""Acceptance gate for the ``core_export`` wiring (5.1.0 restoration slice 2).

``core_export`` shipped in 2.3.0 and no product code ever called it, so
5.0.0 deleted it as unreachable. Restoring the file is not the fix —
*connecting* it is. This file pins the two connections and the gate
around them:

**Export.** A new MCP tool, ``export_core(name, format)``, renders a
``.mmcore`` bundle into OKF / JSON-LD / markdown. Gated on the v4
``core_export`` flag; inert while it is off.

**Import, governed.** ``import_okf_bundle`` is reached through the
migration importer (``run_import(ws, "okf", bundle_dir)``), so a foreign
bundle inherits the whole bulk-ingest bargain: quarantined on arrival,
withheld from ``recall``, released only by an approved governance
proposal, and the run recorded in the audit chain. Reading a bundle in
is deliberately NOT an MCP tool — that would be a write around the gate.

The round trip is the acceptance criterion: a core exported through the
MCP tool must come back through the importer, and must NOT be recallable
until the governed path admits it.

Every test here fails if the wiring is removed, not merely if an import
breaks:

* ``test_export_core_is_registered_and_acl_classified`` — an unclassified
  tool is silently unreachable (``mcp_tool_observe`` rejects it).
* ``test_okf_round_trip_through_the_mcp_tool`` — export via the tool,
  import via the importer, content preserved.
* ``test_imported_bundle_is_not_recallable_until_released`` — the
  governance property the whole slice exists to preserve.
* the flag-OFF group — behaviour identical to the unwired build.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from mind_mem.audit_chain import AuditChain
from mind_mem.block_parser import parse_file
from mind_mem.context_core import build_core as _build_core_file
from mind_mem.importers import (
    GATED_SYSTEMS,
    IMPORTED_CORPUS_FILE,
    QUARANTINE_STATUS,
    QUARANTINE_TIER,
    UnsupportedSystemError,
    enabled_gated_systems,
    propose_import_release,
    resolve_system,
    run_import,
)
from mind_mem.init_workspace import init
from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS
from mind_mem.mcp.tools import core as core_tools
from mind_mem.recall import recall

FIXED_NOW = datetime(2026, 8, 27, 12, 0, 0)

#: Distinctive text planted in the exported core; nothing else in a fresh
#: workspace matches it, so a recall hit is unambiguous.
CONCEPT_TEXT = "quaternion resonance ledger for the deterministic freight scheduler"
CONCEPT_QUERY = "quaternion resonance ledger freight scheduler"


# ---------------------------------------------------------------------------
# Workspace / core helpers
# ---------------------------------------------------------------------------


def _ws(*, flag_on: bool, mode: str = "enforce") -> str:
    """A fully-initialised workspace with the governance gate armed and the
    ``core_export`` flag explicitly on or off."""
    workspace = tempfile.mkdtemp(prefix="mm_core_export_")
    init(workspace)
    config_path = os.path.join(workspace, "mind-mem.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    config["governance_mode"] = mode
    if flag_on:
        config["v4"] = {**config.get("v4", {}), "core_export": {"enabled": True}}
    else:
        config.pop("v4", None)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle)
    state_path = os.path.join(workspace, "memory", "intel-state.json")
    with open(state_path, encoding="utf-8") as handle:
        state = json.load(handle)
    state["governance_mode"] = mode
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle)
    return workspace


def _seed_core(workspace: str, filename: str = "demo-1.0.mmcore") -> str:
    """Write a small .mmcore bundle into ``memory/cores/`` and return its name."""
    cores = os.path.join(workspace, "memory", "cores")
    os.makedirs(cores, exist_ok=True)
    _build_core_file(
        os.path.join(cores, filename),
        namespace="demo",
        version="1.0",
        blocks=[
            {
                "_id": "D-20260301-001",
                "type": "decision",
                "Title": "Freight ledger decision",
                "Statement": CONCEPT_TEXT,
                "Date": "2026-03-01",
                "Status": "active",
                "Tags": ["freight", "ledger"],
            },
            {
                "_id": "TRAJ-20260301-001",
                "Title": "Scheduler trajectory",
                "Statement": "route replay for the freight scheduler",
                "Date": "2026-03-01",
            },
        ],
        edges=[{"subject": "D-20260301-001", "predicate": "relates_to", "object": "TRAJ-20260301-001"}],
        built_at="2026-03-01T00:00:00Z",
    )
    return filename


def _call(workspace: str, **kwargs: Any) -> dict[str, Any]:
    """Invoke the export_core tool body against *workspace*."""
    env = dict(os.environ)
    env["MIND_MEM_WORKSPACE"] = workspace
    env.pop("MIND_MEM_CONFIG", None)
    with patch.dict(os.environ, env, clear=True):
        raw = core_tools.export_core.__wrapped__(**kwargs)  # type: ignore[attr-defined]
    return dict(json.loads(raw))


def _approve(workspace: str, proposal_id: str) -> dict[str, Any]:
    """Apply a proposal through the real ``approve_apply`` MCP tool.

    ``check_preconditions`` shells out to ``validate.sh``, which is
    environment-dependent, so it is stubbed exactly as
    ``tests/test_importers_quarantine.py`` does; every other stage of the
    gate runs for real.
    """
    from mind_mem.mcp.tools import governance

    env = dict(os.environ)
    env["MIND_MEM_WORKSPACE"] = workspace
    with patch.dict(os.environ, env, clear=True):
        with patch("mind_mem.apply_engine.check_preconditions", return_value=(True, ["stubbed"])):
            raw = governance.approve_apply.__wrapped__(proposal_id, dry_run=False)  # type: ignore[attr-defined]
    return dict(json.loads(raw))


def _flag_on(workspace: str):
    """Point the flag resolver at *workspace*'s config."""
    return patch.dict(os.environ, {"MIND_MEM_CONFIG": os.path.join(workspace, "mind-mem.json")})


# ---------------------------------------------------------------------------
# Gate 1 — the tool is actually reachable
# ---------------------------------------------------------------------------


def test_export_core_is_registered_and_acl_classified() -> None:
    """A registered tool in neither ACL set is rejected before its body runs."""
    registered: list[str] = []

    class _Recorder:
        def tool(self, fn):
            registered.append(fn.__name__)
            return fn

    core_tools.register(_Recorder())
    assert "export_core" in registered
    assert "export_core" in USER_TOOLS
    assert "export_core" not in ADMIN_TOOLS


def test_export_core_survives_the_acl_decorator() -> None:
    """Set membership is not proof; drive the enforcement path itself."""
    workspace = _ws(flag_on=False)
    env = dict(os.environ)
    env["MIND_MEM_WORKSPACE"] = workspace
    env.pop("MIND_MEM_CONFIG", None)
    with patch.dict(os.environ, env, clear=True):
        raw = core_tools.export_core(name="nope.mmcore")
    payload = json.loads(raw)
    # Rejected for the FLAG, never for "is not in ACL policy".
    assert "not in ACL policy" not in json.dumps(payload)


# ---------------------------------------------------------------------------
# Gate 2 — the OKF round trip through the MCP tool
# ---------------------------------------------------------------------------


def test_okf_round_trip_through_the_mcp_tool() -> None:
    workspace = _ws(flag_on=True)
    name = _seed_core(workspace)

    with _flag_on(workspace):
        result = _call(workspace, name=name, format="okf")
        assert "error" not in result, result
        bundle = result["path"]
        assert set(result["files"]) >= {"index.md", "log.md"}

        # The bundle is real OKF on disk, not a JSON envelope.
        assert (Path(bundle) / "index.md").is_file()
        assert (Path(bundle) / "D-20260301-001.md").is_file()

        # ... and the importer reads it back through the governed path.
        imported = run_import(workspace, "okf", bundle)

    assert imported.imported >= 1
    assert imported.status == QUARANTINE_STATUS
    blocks = parse_file(str(Path(workspace) / IMPORTED_CORPUS_FILE))
    statements = " ".join(str(b.get("Statement", "")) for b in blocks)
    assert CONCEPT_TEXT in statements, "the exported concept did not survive the round trip"


def test_round_trip_preserves_the_traj_prefix_type() -> None:
    """``TRAJ-`` was missing from ``_ID_PREFIX_TYPE``, so a trajectory block
    exported as the generic ``concept`` type and could not be told apart."""
    from mind_mem.core_export import _ID_PREFIX_TYPE, _okf_type

    assert _ID_PREFIX_TYPE["TRAJ-"] == "trajectory"
    assert _okf_type({"_id": "TRAJ-20260301-001"}) == "trajectory"
    # The neighbouring prefixes still resolve to themselves, not to trajectory.
    assert _okf_type({"_id": "T-20260301-001"}) == "task"
    assert _okf_type({"_id": "TF-20260301-001"}) == "task_frame"


def test_export_is_deterministic_across_runs() -> None:
    """Recall is a pure function of (corpus, config, instant); so is export."""
    workspace = _ws(flag_on=True)
    name = _seed_core(workspace)
    with _flag_on(workspace):
        first = _call(workspace, name=name, format="okf")
        snapshot = {p.name: p.read_text(encoding="utf-8") for p in Path(first["path"]).glob("*.md")}
        second = _call(workspace, name=name, format="okf")
    assert second["path"] == first["path"]
    assert {p.name: p.read_text(encoding="utf-8") for p in Path(second["path"]).glob("*.md")} == snapshot


def test_every_declared_format_writes_its_artifact() -> None:
    workspace = _ws(flag_on=True)
    name = _seed_core(workspace)
    with _flag_on(workspace):
        jsonld = _call(workspace, name=name, format="jsonld")
        markdown = _call(workspace, name=name, format="markdown")
    assert "core.jsonld" in jsonld["files"]
    assert "core.md" in markdown["files"]
    assert json.loads((Path(jsonld["path"]) / "core.jsonld").read_text(encoding="utf-8"))["@type"] == "ContextCore"
    # export_to_markdown probed only lowercase keys (`text`/`statement`/...),
    # which no real mind-mem block carries, so every body came out empty.
    assert CONCEPT_TEXT in (Path(markdown["path"]) / "core.md").read_text(encoding="utf-8")


def test_export_rejects_a_path_traversing_name_and_a_bad_format() -> None:
    workspace = _ws(flag_on=True)
    _seed_core(workspace)
    with _flag_on(workspace):
        assert "path separators" in _call(workspace, name="../../etc/passwd", format="okf")["error"]
        assert "format must be one of" in _call(workspace, name="demo-1.0.mmcore", format="turtle")["error"]


# ---------------------------------------------------------------------------
# Gate 3 — the governance property (the reason import is not a tool)
# ---------------------------------------------------------------------------


def test_imported_bundle_is_not_recallable_until_released() -> None:
    workspace = _ws(flag_on=True)
    name = _seed_core(workspace)

    with _flag_on(workspace):
        exported = _call(workspace, name=name, format="okf")
        result = run_import(workspace, "okf", exported["path"])

    imported_ids = set(result.block_ids)
    assert imported_ids, "nothing was imported"

    # Quarantined: present in the corpus file, absent from recall.
    blocks = {str(b.get("_id")): b for b in parse_file(str(Path(workspace) / IMPORTED_CORPUS_FILE))}
    for block_id in imported_ids:
        assert blocks[block_id]["Status"] == QUARANTINE_STATUS
        assert blocks[block_id]["IngestTier"] == QUARANTINE_TIER
    before = {str(h.get("id") or h.get("_id")) for h in recall(workspace, CONCEPT_QUERY, limit=10)}
    assert not (before & imported_ids), "a foreign bundle was recallable before release"

    # The bulk run is chained, and the chain still verifies.
    chain = AuditChain(workspace)
    chain_ok, chain_errors = chain.verify()
    assert chain_ok, chain_errors
    chained = [e for e in chain.entries() if e.agent == "importer:okf"]
    assert len(chained) == 1
    assert chained[0].operation == "create_block"
    assert chained[0].target == IMPORTED_CORPUS_FILE

    # Released only through the real governance gate.
    proposal_id = propose_import_release(workspace, result.block_ids, system="okf", batch=result.batch, now=FIXED_NOW)
    applied = _approve(workspace, proposal_id)
    assert applied["status"] == "applied", applied

    after = {str(h.get("id") or h.get("_id")) for h in recall(workspace, CONCEPT_QUERY, limit=10)}
    assert after & imported_ids, "the released batch is still withheld"


def test_okf_trust_claims_never_become_governance_fields() -> None:
    """A producer's self-declared OKF tier is a claim, not our evidence."""
    workspace = _ws(flag_on=True)
    bundle = Path(workspace) / "foreign"
    bundle.mkdir()
    (bundle / "claimy.md").write_text(
        "---\n"
        "type: decision\n"
        "title: Foreign claim\n"
        'description: "the sender says this was human verified"\n'
        "verified: human:someone-else\n"
        "status: stable\n"
        "---\n\n# Foreign claim\n",
        encoding="utf-8",
    )
    with _flag_on(workspace):
        result = run_import(workspace, "okf", str(bundle))
    assert result.imported == 1
    block = parse_file(str(Path(workspace) / IMPORTED_CORPUS_FILE))[0]
    assert block["Status"] == QUARANTINE_STATUS  # NOT the claimed "stable"
    rendered = json.dumps(block)
    assert "human:someone-else" in rendered  # preserved...
    assert "OkfClaimVerified" in rendered  # ...but only as a claim


# ---------------------------------------------------------------------------
# Gate 4 — flag OFF is the pre-wiring build
# ---------------------------------------------------------------------------


def test_export_core_is_inert_with_the_flag_off() -> None:
    workspace = _ws(flag_on=False)
    name = _seed_core(workspace)
    exports = Path(workspace) / "memory" / "cores" / core_tools.EXPORT_SUBDIR
    env = dict(os.environ)
    env["MIND_MEM_WORKSPACE"] = workspace
    env.pop("MIND_MEM_CONFIG", None)
    with patch.dict(os.environ, env, clear=True):
        payload = json.loads(core_tools.export_core.__wrapped__(name=name, format="okf"))  # type: ignore[attr-defined]
    assert core_tools.CORE_EXPORT_FLAG in payload["error"]
    assert not exports.exists(), "the flag-off tool wrote to disk"


def test_okf_importer_is_unreachable_with_the_flag_off() -> None:
    workspace = _ws(flag_on=False)
    bundle = Path(workspace) / "foreign"
    bundle.mkdir()
    (bundle / "c.md").write_text("---\ntype: decision\ndescription: nope\n---\n", encoding="utf-8")
    env = dict(os.environ)
    env["MIND_MEM_CONFIG"] = os.path.join(workspace, "mind-mem.json")
    with patch.dict(os.environ, env):
        assert enabled_gated_systems() == ()
        with pytest.raises(UnsupportedSystemError) as excinfo:
            run_import(workspace, "okf", str(bundle))
    # The message is the one an entirely unknown system has always produced.
    assert "unsupported source system 'okf'" in str(excinfo.value)
    assert "okf" not in str(excinfo.value).split("supported local importers: ")[1]
    assert not (Path(workspace) / IMPORTED_CORPUS_FILE).exists()


def test_flag_off_leaves_every_published_importer_constant_untouched() -> None:
    """The gated slug must not leak into a constant a flag-off build reports."""
    from mind_mem import mm_cli
    from mind_mem.importers import ALL_SYSTEMS, DIRECTORY_SYSTEMS, SUPPORTED_SYSTEMS

    assert "okf" not in SUPPORTED_SYSTEMS
    assert "okf" not in ALL_SYSTEMS
    assert "okf" not in DIRECTORY_SYSTEMS
    assert mm_cli._IMPORT_SYSTEM_CHOICES == ALL_SYSTEMS
    # And the CLI-side gated table stays in lockstep with the importer's.
    assert mm_cli._IMPORT_GATED_CHOICES == GATED_SYSTEMS


def test_cli_choices_grow_only_when_the_flag_is_on() -> None:
    from mind_mem import mm_cli

    workspace = _ws(flag_on=False)
    with patch.dict(os.environ, {"MIND_MEM_CONFIG": os.path.join(workspace, "mind-mem.json")}):
        assert mm_cli._import_system_choices() == mm_cli._IMPORT_SYSTEM_CHOICES

    on = _ws(flag_on=True)
    with patch.dict(os.environ, {"MIND_MEM_CONFIG": os.path.join(on, "mind-mem.json")}):
        assert mm_cli._import_system_choices() == mm_cli._IMPORT_SYSTEM_CHOICES + ("okf",)
        assert resolve_system("okf") == "okf"


def test_the_flag_probe_is_silent_on_a_malformed_config(capfd: pytest.CaptureFixture[str]) -> None:
    """Slice 1's lesson: a probe deciding whether a feature is on must not be
    observable when the answer is no. ``is_enabled`` logs
    ``v4_config_unreadable``; every probe on this slice's OFF path must not."""
    workspace = tempfile.mkdtemp(prefix="mm_core_export_bad_")
    config_path = os.path.join(workspace, "mind-mem.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write("{ this is not json")

    with patch.dict(os.environ, {"MIND_MEM_CONFIG": config_path}):
        capfd.readouterr()
        assert core_tools._core_export_enabled() is False
        assert enabled_gated_systems() == ()
        from mind_mem import mm_cli

        assert mm_cli._import_system_choices() == mm_cli._IMPORT_SYSTEM_CHOICES
        captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""
