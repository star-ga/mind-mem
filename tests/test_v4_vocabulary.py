"""Tests for v4 vocabulary-bound fields (Group E, ``v4.vocabulary``).

Covers:

    vocabulary.py       declaration loading (mind-mem.json +
                        vocabularies.json workspace file), list/dict
                        declaration shapes, strict vs tolerant parsing,
                        check_fields semantics (case folding, list
                        values, coercion, undeclared-field passthrough)
    block_metadata.py   wiring — validate_block workspace vocabulary
                        gate (reject + flag modes, implicit block_kind)
                        and set_block_metadata tag gate
                        (OutOfVocabularyError / flag-mode write-through)

Backward-compat contract under test throughout: no vocabulary declared
(or flag off) means no restriction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.v4 import FeatureDisabledError
from mind_mem.v4.block_metadata import FLAG as BM_FLAG
from mind_mem.v4.block_metadata import (
    SchemaValidationResult,
    get_block_metadata,
    register_schema_validator,
    set_block_metadata,
    validate_block,
)
from mind_mem.v4.vocabulary import FLAG as VOCAB_FLAG
from mind_mem.v4.vocabulary import (
    MODES,
    WORKSPACE_FILE,
    FieldVocabulary,
    OutOfVocabularyError,
    VocabularyConfigError,
    VocabularyViolation,
    check_fields,
    flagged,
    load_vocabularies,
    rejections,
    validate_workspace_fields,
)


def _ws(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    flags: dict[str, bool],
    vocabularies: dict | None = None,
    ws_file: dict | None = None,
) -> Path:
    """Write a workspace mind-mem.json (v4 flags + optional vocabularies
    key), an optional vocabularies.json, and point MIND_MEM_CONFIG at it."""
    cfg: dict = {"v4": {k: {"enabled": v} for k, v in flags.items()}}
    if vocabularies is not None:
        cfg["vocabularies"] = vocabularies
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    if ws_file is not None:
        (tmp_path / WORKSPACE_FILE).write_text(json.dumps(ws_file), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


@pytest.fixture
def vocab_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _ws(
        tmp_path,
        monkeypatch,
        flags={VOCAB_FLAG: True},
        vocabularies={
            "block_kind": ["decision", "fact", "reference"],
            "category": {"values": ["project", "user"], "mode": "flag", "case_sensitive": False},
        },
    )


# ===========================================================================
# Flag gating
# ===========================================================================


@pytest.mark.unit
def test_flag_off_blocks_all_entry_points(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _ws(tmp_path, monkeypatch, flags={VOCAB_FLAG: False}, vocabularies={"block_kind": ["decision"]})
    with pytest.raises(FeatureDisabledError):
        load_vocabularies(ws)
    with pytest.raises(FeatureDisabledError):
        check_fields({"block_kind": "x"}, {})
    with pytest.raises(FeatureDisabledError):
        validate_workspace_fields(ws, {"block_kind": "x"})


# ===========================================================================
# Declaration loading
# ===========================================================================


@pytest.mark.unit
def test_no_vocabulary_declared_means_no_restriction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _ws(tmp_path, monkeypatch, flags={VOCAB_FLAG: True})
    assert load_vocabularies(ws) == {}
    assert validate_workspace_fields(ws, {"block_kind": "anything-goes"}) == []


@pytest.mark.unit
def test_missing_config_files_mean_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Flag config lives elsewhere; the *workspace* has no files at all.
    _ws(tmp_path, monkeypatch, flags={VOCAB_FLAG: True})
    empty_ws = tmp_path / "empty"
    empty_ws.mkdir()
    assert load_vocabularies(empty_ws) == {}


@pytest.mark.unit
def test_list_shorthand_defaults(vocab_on: Path) -> None:
    vocabs = load_vocabularies(vocab_on)
    bk = vocabs["block_kind"]
    assert bk.values == ("decision", "fact", "reference")
    assert bk.mode == "reject"
    assert bk.case_sensitive is True


@pytest.mark.unit
def test_dict_form_mode_and_case(vocab_on: Path) -> None:
    cat = load_vocabularies(vocab_on)["category"]
    assert cat.mode == "flag"
    assert cat.case_sensitive is False
    assert cat.values == ("project", "user")


@pytest.mark.unit
def test_values_deduped_order_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _ws(tmp_path, monkeypatch, flags={VOCAB_FLAG: True}, vocabularies={"f": ["b", "a", "b", "c", "a"]})
    assert load_vocabularies(ws)["f"].values == ("b", "a", "c")


@pytest.mark.unit
def test_workspace_file_overrides_per_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _ws(
        tmp_path,
        monkeypatch,
        flags={VOCAB_FLAG: True},
        vocabularies={"block_kind": ["decision"], "status": ["open", "closed"]},
        ws_file={"block_kind": ["fact", "reference"]},
    )
    vocabs = load_vocabularies(ws)
    assert vocabs["block_kind"].values == ("fact", "reference")  # workspace file wins
    assert vocabs["status"].values == ("open", "closed")  # untouched fields survive


@pytest.mark.unit
def test_workspace_file_alone_is_sufficient(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _ws(tmp_path, monkeypatch, flags={VOCAB_FLAG: True}, ws_file={"block_kind": ["decision"]})
    assert load_vocabularies(ws)["block_kind"].values == ("decision",)


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_decl",
    [
        [],  # empty values list
        {"values": []},  # empty values list, dict form
        {"values": ["ok", 7]},  # non-string value
        {"values": ["ok"], "mode": "explode"},  # unknown mode
        {"values": ["ok"], "mode": 3},  # non-string mode
        {"values": ["ok"], "case_sensitive": "yes"},  # non-bool case flag
        "decision",  # bare string, not list/dict
        {"no_values_key": True},  # missing values
    ],
)
def test_malformed_declaration_skipped_tolerant_raises_strict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_decl: object) -> None:
    ws = _ws(
        tmp_path,
        monkeypatch,
        flags={VOCAB_FLAG: True},
        vocabularies={"bad": bad_decl, "good": ["a"]},
    )
    vocabs = load_vocabularies(ws)  # tolerant: bad skipped, good kept
    assert "bad" not in vocabs
    assert vocabs["good"].values == ("a",)
    with pytest.raises(VocabularyConfigError):
        load_vocabularies(ws, strict=True)


@pytest.mark.unit
def test_non_mapping_vocabularies_key_tolerated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _ws(tmp_path, monkeypatch, flags={VOCAB_FLAG: True}, vocabularies=None)
    (ws / "mind-mem.json").write_text(
        json.dumps({"v4": {VOCAB_FLAG: {"enabled": True}}, "vocabularies": ["not", "a", "mapping"]}),
        encoding="utf-8",
    )
    assert load_vocabularies(ws) == {}
    with pytest.raises(VocabularyConfigError):
        load_vocabularies(ws, strict=True)


@pytest.mark.unit
def test_unreadable_workspace_file_tolerated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _ws(tmp_path, monkeypatch, flags={VOCAB_FLAG: True}, vocabularies={"f": ["a"]})
    (ws / WORKSPACE_FILE).write_text("{not json", encoding="utf-8")
    assert load_vocabularies(ws)["f"].values == ("a",)
    with pytest.raises(VocabularyConfigError):
        load_vocabularies(ws, strict=True)


# ===========================================================================
# FieldVocabulary invariants
# ===========================================================================


@pytest.mark.unit
def test_field_vocabulary_rejects_bad_construction() -> None:
    with pytest.raises(VocabularyConfigError):
        FieldVocabulary(field="", values=("a",))
    with pytest.raises(VocabularyConfigError):
        FieldVocabulary(field="f", values=())
    with pytest.raises(VocabularyConfigError):
        FieldVocabulary(field="f", values=("a",), mode="maybe")
    assert set(MODES) == {"reject", "flag"}


# ===========================================================================
# check_fields semantics
# ===========================================================================


@pytest.mark.unit
def test_check_fields_in_vocab_passes(vocab_on: Path) -> None:
    vocabs = load_vocabularies(vocab_on)
    assert check_fields({"block_kind": "decision"}, vocabs) == []


@pytest.mark.unit
def test_check_fields_out_of_vocab_violation(vocab_on: Path) -> None:
    vocabs = load_vocabularies(vocab_on)
    violations = check_fields({"block_kind": "rumour"}, vocabs)
    assert len(violations) == 1
    v = violations[0]
    assert isinstance(v, VocabularyViolation)
    assert v.field == "block_kind"
    assert v.value == "rumour"
    assert v.mode == "reject"
    assert v.allowed == ("decision", "fact", "reference")
    assert "rumour" in v.message()


@pytest.mark.unit
def test_check_fields_undeclared_field_passes(vocab_on: Path) -> None:
    vocabs = load_vocabularies(vocab_on)
    assert check_fields({"free_text": "anything at all"}, vocabs) == []


@pytest.mark.unit
def test_check_fields_case_insensitive_matching(vocab_on: Path) -> None:
    vocabs = load_vocabularies(vocab_on)
    assert check_fields({"category": "PROJECT"}, vocabs) == []  # case-insensitive vocab
    violations = check_fields({"block_kind": "Decision"}, vocabs)  # case-sensitive vocab
    assert len(violations) == 1


@pytest.mark.unit
def test_check_fields_list_values_checked_elementwise(vocab_on: Path) -> None:
    vocabs = load_vocabularies(vocab_on)
    assert check_fields({"block_kind": ["decision", "fact"]}, vocabs) == []
    violations = check_fields({"block_kind": ["decision", "rumour", "gossip"]}, vocabs)
    assert [v.value for v in violations] == ["rumour", "gossip"]


@pytest.mark.unit
def test_check_fields_none_skipped_and_scalars_coerced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _ws(tmp_path, monkeypatch, flags={VOCAB_FLAG: True}, vocabularies={"priority": ["1", "2"]})
    vocabs = load_vocabularies(ws)
    assert check_fields({"priority": None}, vocabs) == []  # absence is not a violation
    assert check_fields({"priority": 1}, vocabs) == []  # int coerced to "1"
    assert len(check_fields({"priority": 3}, vocabs)) == 1


@pytest.mark.unit
def test_rejections_and_flagged_split(vocab_on: Path) -> None:
    vocabs = load_vocabularies(vocab_on)
    violations = check_fields({"block_kind": "rumour", "category": "ops"}, vocabs)
    assert [v.field for v in rejections(violations)] == ["block_kind"]
    assert [v.field for v in flagged(violations)] == ["category"]


# ===========================================================================
# Wiring — validate_block
# ===========================================================================


@pytest.fixture
def wired_ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _ws(
        tmp_path,
        monkeypatch,
        flags={BM_FLAG: True, VOCAB_FLAG: True},
        vocabularies={
            "block_kind": ["decision", "fact"],
            "category": {"values": ["project", "user"], "mode": "flag"},
        },
    )


@pytest.mark.unit
def test_validate_block_rejects_out_of_vocab_kind(wired_ws: Path) -> None:
    result = validate_block("rumour", {"text": "hi"}, workspace=wired_ws)
    assert result.ok is False
    assert "vocabulary" in result.reason
    assert "rumour" in result.reason


@pytest.mark.unit
def test_validate_block_accepts_in_vocab_kind(wired_ws: Path) -> None:
    result = validate_block("decision", {"text": "hi"}, workspace=wired_ws)
    assert result.ok is True
    assert result.reason == "no_validator"


@pytest.mark.unit
def test_validate_block_flag_mode_keeps_ok_with_reason(wired_ws: Path) -> None:
    result = validate_block("decision", {"category": "ops"}, workspace=wired_ws)
    assert result.ok is True
    assert result.reason.startswith("vocabulary_flagged:")
    assert "ops" in result.reason


@pytest.mark.unit
def test_validate_block_payload_block_kind_wins_over_kind_arg(wired_ws: Path) -> None:
    # Payload carries its own block_kind — that value is what gets checked.
    result = validate_block("rumour", {"block_kind": "fact"}, workspace=wired_ws)
    assert result.ok is True
    result = validate_block("decision", {"block_kind": "rumour"}, workspace=wired_ws)
    assert result.ok is False


@pytest.mark.unit
def test_validate_block_without_workspace_unchanged(wired_ws: Path) -> None:
    # Legacy call shape: no workspace, vocabulary layer never consulted.
    result = validate_block("rumour", {"text": "hi"})
    assert result.ok is True
    assert result.reason == "no_validator"


@pytest.mark.unit
def test_validate_block_vocab_flag_off_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _ws(
        tmp_path,
        monkeypatch,
        flags={BM_FLAG: True, VOCAB_FLAG: False},
        vocabularies={"block_kind": ["decision"]},
    )
    result = validate_block("rumour", {"text": "hi"}, workspace=ws)
    assert result.ok is True
    assert result.reason == "no_validator"


@pytest.mark.unit
def test_validate_block_schema_validator_failure_takes_precedence(wired_ws: Path) -> None:
    register_schema_validator(
        "decision",
        lambda payload: SchemaValidationResult(ok="text" in payload, reason="text required"),
    )
    try:
        result = validate_block("decision", {}, workspace=wired_ws)
        assert result.ok is False
        assert result.reason == "text required"
        # Validator passes AND vocabulary passes.
        assert validate_block("decision", {"text": "hi"}, workspace=wired_ws).ok is True
    finally:
        # Registry is process-global — leave no validator behind.
        from mind_mem.v4.block_metadata import _validators

        _validators.pop("decision", None)


# ===========================================================================
# Wiring — set_block_metadata
# ===========================================================================


@pytest.fixture
def tag_ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _ws(
        tmp_path,
        monkeypatch,
        flags={BM_FLAG: True, VOCAB_FLAG: True},
        vocabularies={
            "env": ["prod", "dev"],
            "team": {"values": ["core", "infra"], "mode": "flag"},
        },
    )


@pytest.mark.unit
def test_set_block_metadata_rejects_out_of_vocab_tag(tag_ws: Path) -> None:
    with pytest.raises(OutOfVocabularyError) as excinfo:
        set_block_metadata(tag_ws, "b1", tags={"env": "staging"})
    assert excinfo.value.violations[0].field == "env"
    assert "staging" in str(excinfo.value)
    assert get_block_metadata(tag_ws, "b1") is None  # nothing written


@pytest.mark.unit
def test_set_block_metadata_accepts_in_vocab_tag(tag_ws: Path) -> None:
    md = set_block_metadata(tag_ws, "b1", tags={"env": "prod"})
    assert md.tags == {"env": "prod"}
    fetched = get_block_metadata(tag_ws, "b1")
    assert fetched is not None and fetched.tags == {"env": "prod"}


@pytest.mark.unit
def test_set_block_metadata_flag_mode_writes_through(tag_ws: Path) -> None:
    md = set_block_metadata(tag_ws, "b2", tags={"team": "growth"})
    assert md.tags == {"team": "growth"}
    fetched = get_block_metadata(tag_ws, "b2")
    assert fetched is not None and fetched.tags == {"team": "growth"}


@pytest.mark.unit
def test_set_block_metadata_undeclared_tag_unrestricted(tag_ws: Path) -> None:
    md = set_block_metadata(tag_ws, "b3", tags={"anything": "goes"})
    assert md.tags == {"anything": "goes"}


@pytest.mark.unit
def test_set_block_metadata_vocab_flag_off_unrestricted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _ws(
        tmp_path,
        monkeypatch,
        flags={BM_FLAG: True, VOCAB_FLAG: False},
        vocabularies={"env": ["prod"]},
    )
    md = set_block_metadata(ws, "b1", tags={"env": "staging"})
    assert md.tags == {"env": "staging"}
