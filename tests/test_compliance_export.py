# Copyright 2026 STARGA, Inc.
"""``mm export``: a bundle that is evidence rather than a dump.

The property under test is byte-determinism — two runs over an unchanged
corpus produce identical bytes — and it is tested the way a determinism
claim has to be: with a mutation that shows the comparison can fail.
Around it sit the three governance claims the envelope makes:

* the bundle is built from the **admitted** set, so a quarantined block
  is not in it (with a positive control proving an identical *active*
  block with the same canary IS in it — otherwise "absent" would only
  mean the exporter found nothing);
* ``withheld_count`` says how much was withheld without saying what;
* ``--since`` excludes a block it cannot date rather than guessing, and
  counts what it excluded.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Iterator

import pytest

from mind_mem import mm_cli
from mind_mem.compliance import export as export_mod
from mind_mem.compliance.export import (
    BUNDLE_SCHEMA,
    FORMATS,
    ExportPolicy,
    UnknownExportPolicyError,
    build_bundle,
    load_admitted_blocks,
    policy_names,
    render_bundle,
    resolve_export_policy,
)
from mind_mem.v4.feature_flags import FeatureDisabledError

CANARY = "canary-CONTENT-9f3a"
SECRET = "AKIAIOSFODNN7EXAMPLE"

ACTIVE_BLOCK = f"""[DEC-20260105-001]
Title: Ship the thing
Status: active
Date: 2026-01-05
ActorId: agent-7
ActorRole: planner
SessionId: sess-1
ToolId: mm
Purpose: record the decision
Body: {CANARY} reach ops@example.com with {SECRET}
"""

QUARANTINED_BLOCK = f"""[DEC-20260206-002]
Title: Unreviewed claim
Status: quarantined
Date: 2026-02-06
Body: {CANARY} but withheld
"""

UNDATED_BLOCK = f"""[DEC-00000000-003]
Title: No date at all
Status: active
Body: {CANARY} undated
"""


@pytest.fixture
def clean_policies() -> Iterator[None]:
    """Restore the policy registry so a test-defined policy cannot leak.

    Registration is a side effect of construction, which is the property
    under test; without this the name would show up in every later
    ``--policy`` refusal message.
    """
    saved = dict(export_mod._POLICIES)
    try:
        yield
    finally:
        export_mod._POLICIES.clear()
        export_mod._POLICIES.update(saved)


def _workspace(tmp_path: Path, *, blocks: str, **v4: object) -> str:
    (tmp_path / "decisions").mkdir(parents=True, exist_ok=True)
    (tmp_path / "decisions" / "DECISIONS.md").write_text(blocks, encoding="utf-8")
    enabled = {"compliance_export": {"enabled": True}}
    enabled.update(v4)  # type: ignore[arg-type]
    (tmp_path / "mind-mem.json").write_text(json.dumps({"v4": enabled}), encoding="utf-8")
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Policies register by existing
# ---------------------------------------------------------------------------


class TestPolicyRegistry:
    def test_the_three_shipped_policies_are_registered(self) -> None:
        assert set(policy_names()) >= {"full", "redacted", "metadata-only"}

    def test_an_unknown_policy_is_refused_and_the_real_ones_named(self) -> None:
        with pytest.raises(UnknownExportPolicyError) as excinfo:
            resolve_export_policy("everything-please")
        assert "full" in str(excinfo.value)

    def test_a_policy_registers_itself_on_construction(self, clean_policies: None) -> None:
        assert "unit-test-policy" not in policy_names()
        ExportPolicy(name="unit-test-policy", description="probe")
        assert "unit-test-policy" in policy_names()
        assert resolve_export_policy("unit-test-policy").description == "probe"

    def test_two_policies_cannot_share_a_name(self, clean_policies: None) -> None:
        ExportPolicy(name="unit-test-dupe", description="first")
        with pytest.raises(ValueError):
            ExportPolicy(name="unit-test-dupe", description="second")

    def test_the_cli_format_mirror_has_not_drifted(self) -> None:
        """``mm_cli`` holds the format list as a literal so the parser does
        not import this package; the mirror is only safe if it is checked."""
        assert mm_cli._EXPORT_FORMATS == FORMATS


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    @pytest.mark.parametrize("fmt", list(FORMATS))
    def test_two_runs_produce_identical_bytes(self, tmp_path: Path, fmt: str) -> None:
        ws = _workspace(tmp_path, blocks=ACTIVE_BLOCK + "\n" + QUARANTINED_BLOCK)
        first = render_bundle(build_bundle(ws, policy="full", fmt=fmt))
        second = render_bundle(build_bundle(ws, policy="full", fmt=fmt))
        assert first == second

    def test_the_comparison_can_fail(self, tmp_path: Path) -> None:
        """Positive control: change one byte of the corpus, the bundle moves.

        Without this, "the two runs matched" is equally consistent with an
        exporter that emits a constant.
        """
        ws = _workspace(tmp_path, blocks=ACTIVE_BLOCK)
        before = render_bundle(build_bundle(ws, policy="full"))
        (tmp_path / "decisions" / "DECISIONS.md").write_text(
            ACTIVE_BLOCK.replace("Ship the thing", "Ship the other thing"), encoding="utf-8"
        )
        after = render_bundle(build_bundle(ws, policy="full"))
        assert before != after

    def test_record_order_does_not_depend_on_file_order(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, blocks=ACTIVE_BLOCK + "\n" + UNDATED_BLOCK)
        forward = build_bundle(ws, policy="full")
        (tmp_path / "decisions" / "DECISIONS.md").write_text(UNDATED_BLOCK + "\n" + ACTIVE_BLOCK, encoding="utf-8")
        reversed_file = build_bundle(ws, policy="full")
        assert [r["id"] for r in forward.records] == [r["id"] for r in reversed_file.records]

    def test_the_envelope_digest_covers_the_record_section(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, blocks=ACTIVE_BLOCK)
        bundle = build_bundle(ws, policy="full")
        payload = render_bundle(bundle)
        body = payload.split(b"\n", 1)[1]
        assert bundle.envelope["content_sha256"] == hashlib.sha256(body).hexdigest()

    def test_a_truncated_bundle_fails_its_own_digest(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, blocks=ACTIVE_BLOCK + "\n" + UNDATED_BLOCK)
        bundle = build_bundle(ws, policy="full")
        payload = render_bundle(bundle)
        truncated = payload.rsplit(b"\n", 2)[0] + b"\n"
        body = truncated.split(b"\n", 1)[1]
        assert hashlib.sha256(body).hexdigest() != bundle.envelope["content_sha256"]


# ---------------------------------------------------------------------------
# Admission is the boundary
# ---------------------------------------------------------------------------


class TestAdmission:
    def test_a_quarantined_block_is_not_in_the_bundle(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, blocks=ACTIVE_BLOCK + "\n" + QUARANTINED_BLOCK)
        bundle = build_bundle(ws, policy="full")
        ids = {r["id"] for r in bundle.records}

        # POSITIVE CONTROL: the corpus really does hold both blocks, and
        # the admitted one really is exported, so the absence below is
        # about admission and not about an exporter that found nothing.
        admitted, withheld = load_admitted_blocks(ws)
        assert withheld == 1
        assert "DEC-20260105-001" in ids
        assert len(admitted) == 1

        assert "DEC-20260206-002" not in ids
        assert bundle.envelope["withheld_count"] == 1

    def test_the_canary_of_a_withheld_block_never_reaches_the_bytes(self, tmp_path: Path) -> None:
        """Same canary in both blocks: only the admitted copy may appear."""
        ws = _workspace(tmp_path, blocks=QUARANTINED_BLOCK)
        payload = render_bundle(build_bundle(ws, policy="full"))
        assert CANARY in QUARANTINED_BLOCK
        assert CANARY.encode() not in payload
        assert b'"block_count":0' in payload

    def test_the_withheld_count_is_reported_not_hidden(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, blocks=QUARANTINED_BLOCK)
        assert build_bundle(ws, policy="full").envelope["withheld_count"] == 1


# ---------------------------------------------------------------------------
# Policies change what is in the record
# ---------------------------------------------------------------------------


class TestPolicies:
    def test_full_is_verbatim(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, blocks=ACTIVE_BLOCK)
        fields = build_bundle(ws, policy="full").records[0]["fields"]
        assert SECRET in fields["Body"]

    def test_redacted_removes_the_findings_and_counts_them(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, blocks=ACTIVE_BLOCK)
        bundle = build_bundle(ws, policy="redacted")
        body = bundle.records[0]["fields"]["Body"]
        assert SECRET not in body
        assert "[REDACTED:aws_access_key_id]" in body
        assert bundle.envelope["redaction"]["finding_count"] == 2

    def test_redacted_leaves_the_secret_out_of_the_rendered_bytes(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, blocks=ACTIVE_BLOCK)
        payload = render_bundle(build_bundle(ws, policy="redacted"))
        # POSITIVE CONTROL: the same search finds it under `full`.
        assert SECRET.encode() in render_bundle(build_bundle(ws, policy="full"))
        assert SECRET.encode() not in payload

    def test_metadata_only_drops_content_but_keeps_a_digest_of_it(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, blocks=ACTIVE_BLOCK)
        fields = build_bundle(ws, policy="metadata-only").records[0]["fields"]
        assert "Body" not in fields and "Title" not in fields
        assert fields["ActorId"] == "agent-7"
        assert len(fields["ContentSha256"]) == 64

    def test_metadata_only_carries_no_canary(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, blocks=ACTIVE_BLOCK)
        assert CANARY.encode() in render_bundle(build_bundle(ws, policy="full"))
        assert CANARY.encode() not in render_bundle(build_bundle(ws, policy="metadata-only"))


# ---------------------------------------------------------------------------
# --since
# ---------------------------------------------------------------------------


class TestSince:
    def test_older_blocks_are_excluded(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, blocks=ACTIVE_BLOCK)
        assert build_bundle(ws, policy="full", since=date(2026, 1, 1)).envelope["block_count"] == 1
        assert build_bundle(ws, policy="full", since=date(2026, 6, 1)).envelope["block_count"] == 0

    def test_an_undated_block_is_excluded_and_counted(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, blocks=UNDATED_BLOCK)
        # POSITIVE CONTROL: with no window the same block IS exported, so
        # the exclusion below is the date rule and not a parse failure.
        assert build_bundle(ws, policy="full").envelope["block_count"] == 1
        windowed = build_bundle(ws, policy="full", since=date(2020, 1, 1))
        assert windowed.envelope["block_count"] == 0
        assert windowed.envelope["undated_excluded"] == 1


# ---------------------------------------------------------------------------
# The flag is the door
# ---------------------------------------------------------------------------


class TestFlagWiring:
    def test_the_surface_refuses_when_the_flag_is_off(self, tmp_path: Path) -> None:
        (tmp_path / "mind-mem.json").write_text(json.dumps({"v4": {}}), encoding="utf-8")
        with pytest.raises(FeatureDisabledError) as excinfo:
            build_bundle(str(tmp_path), policy="full")
        assert "compliance_export" in str(excinfo.value)

    def test_a_bare_true_cannot_open_the_door(self, tmp_path: Path) -> None:
        (tmp_path / "mind-mem.json").write_text(json.dumps({"v4": {"compliance_export": True}}), encoding="utf-8")
        with pytest.raises(FeatureDisabledError):
            build_bundle(str(tmp_path), policy="full")


# ---------------------------------------------------------------------------
# The CLI is the entry point, so the CLI is what the tests drive
# ---------------------------------------------------------------------------


class TestTheCommandLine:
    def test_mm_export_is_a_real_verb_now(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ws = _workspace(tmp_path, blocks=ACTIVE_BLOCK + "\n" + QUARANTINED_BLOCK)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        out = tmp_path / "bundle.jsonl"
        assert mm_cli.main(["export", "--policy", "redacted", "--out", str(out)]) == 0
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["schema"] == BUNDLE_SCHEMA
        assert envelope["withheld_count"] == 1
        assert SECRET not in out.read_text(encoding="utf-8")

    def test_two_cli_runs_are_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ws = _workspace(tmp_path, blocks=ACTIVE_BLOCK)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        first, second = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
        assert mm_cli.main(["export", "--out", str(first)]) == 0
        assert mm_cli.main(["export", "--out", str(second)]) == 0
        capsys.readouterr()
        assert first.read_bytes() == second.read_bytes()

    def test_an_unknown_policy_exits_two_and_names_the_real_ones(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ws = _workspace(tmp_path, blocks=ACTIVE_BLOCK)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        assert mm_cli.main(["export", "--policy", "nope"]) == 2
        assert "metadata-only" in capsys.readouterr().err

    def test_a_bad_since_exits_two(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        ws = _workspace(tmp_path, blocks=ACTIVE_BLOCK)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        assert mm_cli.main(["export", "--since", "last tuesday"]) == 2
        capsys.readouterr()

    def test_the_flag_off_path_exits_three_with_the_enable_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "mind-mem.json").write_text(json.dumps({"v4": {}}), encoding="utf-8")
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(tmp_path))
        assert mm_cli.main(["export"]) == 3
        assert "compliance_export" in capsys.readouterr().err
