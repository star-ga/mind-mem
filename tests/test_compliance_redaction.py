# Copyright 2026 STARGA, Inc.
"""The pluggable redaction layer: registry, chain, modes, and the ledger.

Three claims are on trial here, and each one is paired with the control
that proves the test could have caught its opposite:

1. **A detector cannot be added without being registered.** Proven by
   defining one inside the test and finding it in the registry, and by an
   AST walk of the shipped package that is itself driven over a tree
   containing an unregistered class to show it can fail.
2. **The OFF path does not run.** Not "runs and finds nothing" — the
   scanner is replaced with one that raises, and the OFF path still
   returns.
3. **A redaction is auditable and leaks nothing.** The canary secret is
   absent from the ledger, against a positive control proving the same
   search finds both the ledger entry and the secret in the document it
   came from. A "not found" assertion over an empty file passes for the
   wrong reason; this one cannot.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Iterator

import pytest

from mind_mem.audit_chain import AuditChain
from mind_mem.compliance import detectors as det
from mind_mem.compliance.audit import REDACTION_OPERATION, record_redaction
from mind_mem.compliance.detectors import (
    CATEGORY_PII,
    CATEGORY_SECRET,
    Detector,
    DetectorSpecError,
    DuplicateDetectorError,
    Finding,
    RegexDetector,
    detector_names,
    get_detector,
    registered_detectors,
    scan_text,
)
from mind_mem.compliance.redaction import (
    MODE_FLAG,
    MODE_OFF,
    MODE_REDACT,
    MODE_REJECT,
    RedactionConfigError,
    RedactionRefused,
    redact,
    redaction_chain_for_workspace,
    resolve_mode,
)

CANARY_SECRET = "ghp_canary000000000000000000000000000000"
SHIPPED_DETECTORS = (
    "aws_access_key_id",
    "credit_card",
    "email",
    "github_token",
    "google_api_key",
    "private_key_block",
    "secret_key_prefix",
    "slack_token",
)


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """Snapshot the detector registry so a test-defined detector cannot leak.

    Registration is a global side effect of a class statement, which is
    exactly the property under test; without this fixture the tests that
    exercise it would silently change what every other test's chain runs.
    """
    saved = dict(det._REGISTRY)
    try:
        yield
    finally:
        det._REGISTRY.clear()
        det._REGISTRY.update(saved)


def _workspace(tmp_path: Path, **v4: object) -> str:
    (tmp_path / "mind-mem.json").write_text(json.dumps({"v4": v4}), encoding="utf-8")
    return str(tmp_path)


# ---------------------------------------------------------------------------
# 1. Registration is structural
# ---------------------------------------------------------------------------


class TestRegistrationCannotBeForgotten:
    def test_the_shipped_pack_is_all_present(self) -> None:
        assert set(SHIPPED_DETECTORS) <= set(detector_names())

    def test_a_concrete_detector_is_registered_by_existing(self, clean_registry: None) -> None:
        assert "unit_test_probe" not in detector_names()

        class _Probe(RegexDetector):
            name = "unit_test_probe"
            category = CATEGORY_SECRET
            pattern = re.compile(r"PROBE-[0-9]+")

        assert "unit_test_probe" in detector_names()
        assert isinstance(get_detector("unit_test_probe"), _Probe)

    def test_an_abstract_base_is_not_registered(self, clean_registry: None) -> None:
        """The exemption is the language's, not an author's flag."""
        assert "Detector" not in detector_names()
        assert RegexDetector.__abstractmethods__, "RegexDetector must stay abstract or it would self-register"

        class _StillAbstract(Detector):
            name = "never_registered"
            category = CATEGORY_PII

        assert _StillAbstract.__abstractmethods__ == frozenset({"scan"})
        assert "never_registered" not in detector_names()

    def test_a_nameless_detector_cannot_be_created(self, clean_registry: None) -> None:
        with pytest.raises(DetectorSpecError) as excinfo:

            class _Nameless(RegexDetector):
                category = CATEGORY_PII
                pattern = re.compile(r"x")

        assert "no 'name'" in str(excinfo.value)

    def test_a_detector_with_no_category_cannot_be_created(self, clean_registry: None) -> None:
        with pytest.raises(DetectorSpecError):

            class _Uncategorised(RegexDetector):
                name = "uncategorised"
                pattern = re.compile(r"x")

    def test_two_detectors_cannot_share_a_name(self, clean_registry: None) -> None:
        class _First(RegexDetector):
            name = "collide"
            category = CATEGORY_PII
            pattern = re.compile(r"a")

        with pytest.raises(DuplicateDetectorError) as excinfo:

            class _Second(RegexDetector):
                name = "collide"
                category = CATEGORY_PII
                pattern = re.compile(r"b")

        assert "collide" in str(excinfo.value)

    def test_an_unknown_detector_name_is_a_refusal_not_a_shorter_chain(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            get_detector("no_such_detector")
        assert "registered:" in str(excinfo.value)


def _detector_classes(root: Path) -> list[str]:
    """Class names in *root* whose bases name a detector base class."""
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            base_names = {b.id for b in node.bases if isinstance(b, ast.Name)}
            base_names |= {b.attr for b in node.bases if isinstance(b, ast.Attribute)}
            if base_names & {"Detector", "RegexDetector"}:
                found.append(node.name)
    return found


class TestThePackageHasNoUnregisteredDetector:
    def test_every_detector_class_in_the_package_is_registered(self) -> None:
        package = Path(det.__file__).parent
        classes = _detector_classes(package)
        # Positive control: a walker that found nothing would pass the
        # assertion below for the wrong reason.
        assert len(classes) >= len(SHIPPED_DETECTORS), f"the walker found only {classes}"
        registered = {type(d).__name__ for d in registered_detectors()}
        abstract = {"RegexDetector"}
        assert set(classes) - abstract <= registered

    def test_the_walker_can_actually_fail(self, tmp_path: Path) -> None:
        """Drive the same walker over a tree that does have an orphan."""
        (tmp_path / "orphan.py").write_text(
            "from mind_mem.compliance.detectors import RegexDetector\n\n\nclass NotInTheRegistry(RegexDetector):\n    pass\n",
            encoding="utf-8",
        )
        classes = _detector_classes(tmp_path)
        registered = {type(d).__name__ for d in registered_detectors()}
        assert classes == ["NotInTheRegistry"]
        assert set(classes) - {"RegexDetector"} - registered, "the walker reported clean over a tree with an orphan"


# ---------------------------------------------------------------------------
# 2. The detectors detect, and do not over-detect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "hit", "miss"),
    [
        ("email", "write to ops.team+x@sub.example.co.uk now", "write to ops at example"),
        ("aws_access_key_id", "key AKIAIOSFODNN7EXAMPLE here", "key AKIASHORT here"),
        ("github_token", f"token {CANARY_SECRET} here", "token ghp_short here"),
        ("slack_token", "token xoxb-1234567890-abcdef here", "token xoxb-1 here"),
        ("google_api_key", "key AIza" + "b" * 35 + " here", "key AIzaShort here"),
        ("secret_key_prefix", "key sk-ant-" + "c" * 24 + " here", "key sk-tiny here"),
        (
            "private_key_block",
            "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n-----END OPENSSH PRIVATE KEY-----",
            "-----BEGIN CERTIFICATE-----\nabc\n-----END CERTIFICATE-----",
        ),
        ("credit_card", "card 4111 1111 1111 1111 ok", "card 4111111111111112 ok"),
    ],
)
def test_each_detector_finds_its_shape_and_not_its_near_miss(name: str, hit: str, miss: str) -> None:
    detector = get_detector(name)
    assert detector.scan(hit), f"{name} missed its own shape"
    assert detector.scan(miss) == [], f"{name} fired on a near miss"


def test_luhn_is_what_separates_a_card_from_a_long_number() -> None:
    """Positive control for the negative above: the check is the only difference."""
    detector = get_detector("credit_card")
    assert detector.scan("4111111111111111")
    assert detector.scan("4111111111111112") == []


def test_findings_are_canonically_ordered_and_do_not_overlap() -> None:
    text = "a@b.com then AKIAIOSFODNN7EXAMPLE then c@d.com"
    once = scan_text(text)
    twice = scan_text(text)
    assert once == twice
    assert [f.start for f in once] == sorted(f.start for f in once)
    for earlier, later in zip(once, once[1:]):
        assert earlier.end <= later.start


def test_a_finding_never_carries_the_value_or_a_digest_of_it() -> None:
    """The whole record shape, asserted — a hash of a low-entropy secret is the secret."""
    finding = Finding(start=0, end=4, detector="email", category=CATEGORY_PII)
    assert set(finding.to_dict()) == {"detector", "category", "start", "end", "length"}


# ---------------------------------------------------------------------------
# 3. Modes, including an OFF path that provably does not run
# ---------------------------------------------------------------------------


class TestModes:
    def test_off_changes_nothing_and_reports_nothing(self) -> None:
        result = redact(f"secret {CANARY_SECRET}", mode=MODE_OFF)
        assert result.text == f"secret {CANARY_SECRET}"
        assert result.findings == ()
        assert result.original_sha256 == "" and result.redacted_sha256 == ""
        assert result.changed is False

    def test_off_does_not_even_scan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inertness is a claim about work done, not about output.

        Replacing the scanner with one that raises turns "the OFF path
        scanned anyway" from invisible into a failure.
        """

        def _explode(*_args: object, **_kwargs: object) -> list[Finding]:
            raise AssertionError("the OFF path ran the detector chain")

        monkeypatch.setattr("mind_mem.compliance.redaction.scan_text", _explode)
        assert redact("anything", mode=MODE_OFF).text == "anything"
        # Positive control: the same patch makes an ON mode fail.
        with pytest.raises(AssertionError):
            redact("anything", mode=MODE_FLAG)

    def test_flag_reports_without_rewriting(self) -> None:
        result = redact(f"secret {CANARY_SECRET}", mode=MODE_FLAG)
        assert result.findings
        assert CANARY_SECRET in result.text
        assert result.changed is False

    def test_redact_rewrites_and_the_digests_move(self) -> None:
        result = redact(f"secret {CANARY_SECRET}", mode=MODE_REDACT)
        assert CANARY_SECRET not in result.text
        assert "[REDACTED:github_token]" in result.text
        assert result.changed is True
        assert result.original_sha256 != result.redacted_sha256

    def test_redaction_is_idempotent(self) -> None:
        once = redact(f"secret {CANARY_SECRET}", mode=MODE_REDACT).text
        twice = redact(once, mode=MODE_REDACT).text
        assert once == twice

    def test_reject_refuses_and_carries_the_kinds_not_the_values(self) -> None:
        with pytest.raises(RedactionRefused) as excinfo:
            redact(f"secret {CANARY_SECRET}", mode=MODE_REJECT)
        message = str(excinfo.value)
        assert "github_token" in message
        assert CANARY_SECRET not in message

    def test_reject_has_a_way_out(self) -> None:
        """A refusal state with no exit is a one-way door; clean text passes."""
        result = redact("nothing to see", mode=MODE_REJECT)
        assert result.findings == ()
        assert result.text == "nothing to see"

    def test_an_unknown_mode_is_refused(self) -> None:
        with pytest.raises(RedactionConfigError):
            redact("x", mode="quietly-ignore")


# ---------------------------------------------------------------------------
# 4. The flag is the door
# ---------------------------------------------------------------------------


class TestFlagWiring:
    def test_absent_flag_means_off(self, tmp_path: Path) -> None:
        assert resolve_mode(_workspace(tmp_path)) == MODE_OFF

    def test_enabled_defaults_to_redact(self, tmp_path: Path) -> None:
        assert resolve_mode(_workspace(tmp_path, redaction={"enabled": True})) == MODE_REDACT

    def test_the_mode_is_read_from_the_workspace(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, redaction={"enabled": True, "mode": MODE_REJECT})
        assert resolve_mode(ws) == MODE_REJECT

    def test_a_bare_true_cannot_switch_it_on(self, tmp_path: Path) -> None:
        assert resolve_mode(_workspace(tmp_path, redaction=True)) == MODE_OFF

    def test_an_unknown_mode_in_config_refuses(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, redaction={"enabled": True, "mode": "sometimes"})
        with pytest.raises(RedactionConfigError):
            resolve_mode(ws)

    def test_a_declared_subset_is_the_whole_chain(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, redaction={"enabled": True, "detectors": ["email"]})
        assert [d.name for d in redaction_chain_for_workspace(ws)] == ["email"]

    def test_an_unknown_detector_name_refuses_rather_than_shrinking_the_chain(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, redaction={"enabled": True, "detectors": ["email", "typo_detector"]})
        with pytest.raises(RedactionConfigError) as excinfo:
            redaction_chain_for_workspace(ws)
        assert "typo_detector" in str(excinfo.value)

    def test_no_declared_subset_means_every_detector(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, redaction={"enabled": True})
        assert [d.name for d in redaction_chain_for_workspace(ws)] == list(detector_names())


# ---------------------------------------------------------------------------
# 5. Every pass reaches the ledger, and the ledger holds no secret
# ---------------------------------------------------------------------------


def _chain_text(workspace: str) -> str:
    path = Path(workspace) / ".mind-mem-audit" / "chain.jsonl"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


class TestTheLedger:
    def test_a_redaction_lands_in_the_chain(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        result = redact(f"secret {CANARY_SECRET}", mode=MODE_REDACT)
        entry = record_redaction(ws, result, target="notes/one.md", agent="tester")

        assert entry is not None
        assert entry.operation == REDACTION_OPERATION
        assert entry.target == "notes/one.md"
        assert entry.agent == "tester"
        assert entry.fields_changed == ["github_token"]
        assert "github_token x1" in entry.reason
        ok, errors = AuditChain(ws).verify()
        assert ok, errors

    def test_a_clean_pass_is_recorded_too(self, tmp_path: Path) -> None:
        """ "Nothing was found" is a claim an auditor needs evidence for."""
        ws = str(tmp_path)
        entry = record_redaction(ws, redact("all clear", mode=MODE_REDACT), target="notes/clean.md")
        assert entry is not None
        assert "no findings" in entry.reason

    def test_the_ledger_never_holds_the_secret(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        document = f"please use {CANARY_SECRET} for the deploy"
        result = redact(document, mode=MODE_REDACT)
        entry = record_redaction(ws, result, target="notes/one.md", agent="tester")
        assert entry is not None

        ledger = _chain_text(ws)
        # POSITIVE CONTROLS, both required: the file exists and holds the
        # event (so "not found" is not a statement about an empty file),
        # and the same substring search does find the secret in the
        # document it came from (so the method can see what it is looking
        # for).
        assert "notes/one.md" in ledger
        assert entry.entry_hash in ledger
        assert CANARY_SECRET in document

        assert CANARY_SECRET not in ledger

    def test_the_off_path_writes_no_ledger_at_all(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        assert record_redaction(ws, redact("x", mode=MODE_OFF), target="notes/one.md") is None
        assert not (Path(ws) / ".mind-mem-audit").exists()


# ---------------------------------------------------------------------------
# 6. The CLI is the entry point, so the CLI is what the tests drive
# ---------------------------------------------------------------------------


class TestTheCommandLine:
    def test_mm_compliance_detectors_lists_the_registry(self, capsys: pytest.CaptureFixture[str]) -> None:
        from mind_mem import mm_cli

        assert mm_cli.main(["compliance", "detectors", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert {row["name"] for row in payload["detectors"]} >= set(SHIPPED_DETECTORS)

    def test_mm_is_the_console_script_that_reaches_them(self) -> None:
        """`imported` is not `wired`: the verb must hang off the real entry point."""
        from mind_mem import mm_cli

        parser = mm_cli.build_parser()
        args = parser.parse_args(["compliance", "redact", "--text", "x"])
        assert args.func is mm_cli._cmd_compliance_redact
        assert parser.parse_args(["export"]).func is mm_cli._cmd_export

    def test_redact_refuses_when_the_flag_is_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from mind_mem import mm_cli

        monkeypatch.setenv("MIND_MEM_WORKSPACE", _workspace(tmp_path))
        assert mm_cli.main(["compliance", "redact", "--text", CANARY_SECRET]) == 3
        assert "redaction" in capsys.readouterr().err

    def test_redact_rewrites_and_records(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        from mind_mem import mm_cli

        ws = _workspace(tmp_path, redaction={"enabled": True})
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        source = tmp_path / "note.md"
        source.write_text(f"deploy with {CANARY_SECRET}\n", encoding="utf-8")

        assert mm_cli.main(["compliance", "redact", "--file", str(source), "--in-place", "--agent", "tester"]) == 0
        capsys.readouterr()

        rewritten = source.read_text(encoding="utf-8")
        assert CANARY_SECRET not in rewritten
        assert "[REDACTED:github_token]" in rewritten
        ledger = _chain_text(ws)
        entry = json.loads(ledger.splitlines()[0])
        assert entry["target"] == "note.md", "an absolute path in the ledger would not survive a moved workspace"
        assert entry["fields_changed"] == ["github_token"]
        assert CANARY_SECRET not in ledger

    def test_scan_is_read_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        from mind_mem import mm_cli

        ws = _workspace(tmp_path, redaction={"enabled": True})
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        assert mm_cli.main(["compliance", "scan", "--text", f"x {CANARY_SECRET}", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["finding_count"] == 1
        assert not (Path(ws) / ".mind-mem-audit").exists()

        # POSITIVE CONTROL: `redact` on the same text DOES write the ledger,
        # so "no ledger" is a property of `scan` and not of the workspace.
        assert mm_cli.main(["compliance", "redact", "--text", f"x {CANARY_SECRET}", "--target", "t.md"]) == 0
        capsys.readouterr()
        assert (Path(ws) / ".mind-mem-audit" / "chain.jsonl").is_file()

    def test_a_reject_workspace_refuses_and_still_records_the_refusal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from mind_mem import mm_cli

        ws = _workspace(tmp_path, redaction={"enabled": True, "mode": MODE_REJECT})
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        assert mm_cli.main(["compliance", "redact", "--text", f"x {CANARY_SECRET}", "--target", "t.md"]) == 4
        capsys.readouterr()
        ledger = _chain_text(ws)
        assert "t.md" in ledger
        assert CANARY_SECRET not in ledger
