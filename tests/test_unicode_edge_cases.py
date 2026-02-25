#!/usr/bin/env python3
"""Tests for Unicode and edge case handling across mind-mem modules."""

import os
import tempfile

import pytest

from mind_mem.block_parser import parse_blocks
from mind_mem.init_workspace import init
from mind_mem.recall import recall, tokenize
from mind_mem.entity_ingest import extract_entities
from mind_mem.session_summarizer import extract_summary


class TestUnicodeBlockParsing:
    """Test that blocks with Unicode content are parsed correctly."""

    def test_chinese_block_content(self):
        text = "[DEC-001]\nType: Decision\nStatement: 我们决定使用Python进行开发\n\n"
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert "我们决定" in blocks[0].get("Statement", "")

    def test_emoji_block_content(self):
        text = "[DEC-002]\nType: Decision\nStatement: Deploy to production 🚀\n\n"
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert "🚀" in blocks[0].get("Statement", "")

    def test_arabic_block_content(self):
        text = "[DEC-003]\nType: Decision\nStatement: قرار بشأن الهندسة المعمارية\n\n"
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert "قرار" in blocks[0].get("Statement", "")

    def test_japanese_katakana(self):
        text = "[DEC-004]\nType: Decision\nStatement: テスト駆動開発を採用する\n\n"
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert "テスト" in blocks[0].get("Statement", "")

    def test_mixed_script_content(self):
        text = "[DEC-005]\nType: Decision\nStatement: Use Python3 для разработки (développement)\n\n"
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        stmt = blocks[0].get("Statement", "")
        assert "для" in stmt
        assert "développement" in stmt

    def test_korean_block_content(self):
        text = "[DEC-006]\nType: Decision\nStatement: 데이터베이스로 PostgreSQL을 선택합니다\n\n"
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert "데이터베이스" in blocks[0].get("Statement", "")

    def test_thai_block_content(self):
        text = "[DEC-007]\nType: Decision\nStatement: ใช้ภาษาไพทอนในการพัฒนา\n\n"
        blocks = parse_blocks(text)
        assert len(blocks) == 1

    def test_devanagari_block_content(self):
        text = "[DEC-008]\nType: Decision\nStatement: डेटाबेस के लिए PostgreSQL चुनें\n\n"
        blocks = parse_blocks(text)
        assert len(blocks) == 1


class TestUnicodeRecall:
    """Test recall with Unicode queries and content."""

    def test_recall_chinese_query(self, tmp_path):
        ws = str(tmp_path)
        init(ws)
        decisions = os.path.join(ws, "decisions", "cn.md")
        with open(decisions, "w", encoding="utf-8") as f:
            f.write("[DEC-010]\nType: Decision\nStatement: 数据库选择PostgreSQL\n\n")
        results = recall(ws, "数据库", limit=5)
        assert isinstance(results, list)

    def test_recall_emoji_query(self, tmp_path):
        ws = str(tmp_path)
        init(ws)
        decisions = os.path.join(ws, "decisions", "emoji.md")
        with open(decisions, "w", encoding="utf-8") as f:
            f.write("[DEC-011]\nType: Decision\nStatement: Ship fast 🚀 and iterate\n\n")
        results = recall(ws, "🚀", limit=5)
        assert isinstance(results, list)

    def test_recall_empty_query(self, tmp_path):
        ws = str(tmp_path)
        init(ws)
        results = recall(ws, "", limit=5)
        assert isinstance(results, list)

    def test_recall_japanese_query(self, tmp_path):
        ws = str(tmp_path)
        init(ws)
        decisions = os.path.join(ws, "decisions", "jp.md")
        with open(decisions, "w", encoding="utf-8") as f:
            f.write("[DEC-012]\nType: Decision\nStatement: テスト駆動開発を採用する\n\n")
        results = recall(ws, "テスト駆動開発", limit=5)
        assert isinstance(results, list)

    def test_recall_cyrillic_query(self, tmp_path):
        ws = str(tmp_path)
        init(ws)
        decisions = os.path.join(ws, "decisions", "ru.md")
        with open(decisions, "w", encoding="utf-8") as f:
            f.write("[DEC-013]\nType: Decision\nStatement: Использовать PostgreSQL для базы данных\n\n")
        results = recall(ws, "базы данных", limit=5)
        assert isinstance(results, list)

    def test_tokenize_unicode(self):
        tokens = tokenize("数据库 PostgreSQL テスト")
        assert isinstance(tokens, list)

    def test_tokenize_emoji(self):
        tokens = tokenize("deploy 🚀 ship 🎉")
        assert isinstance(tokens, list)


class TestEdgeCaseInputs:
    """Test edge cases in various modules."""

    def test_extract_entities_with_unicode(self):
        text = "Check https://github.com/star-ga/日本語 and @用户 mentions"
        entities = extract_entities(text)
        assert isinstance(entities, list)

    def test_extract_entities_very_long_text(self):
        text = "word " * 100000
        entities = extract_entities(text)
        assert isinstance(entities, list)

    def test_extract_entities_null_bytes(self):
        text = "normal text\x00with null bytes"
        entities = extract_entities(text)
        assert isinstance(entities, list)

    def test_extract_entities_empty_string(self):
        entities = extract_entities("")
        assert isinstance(entities, list)
        assert len(entities) == 0

    def test_extract_summary_empty_messages(self):
        result = extract_summary([])
        assert result["message_count"] == 0
        assert result["topics"] == []
        assert result["files"] == []

    def test_extract_summary_unicode_messages(self):
        messages = [
            {"role": "user", "content": "使用Python开发"},
            {"role": "assistant", "content": "好的，我来帮你"},
        ]
        result = extract_summary(messages)
        assert result["message_count"] == 2

    def test_extract_summary_only_whitespace(self):
        messages = [
            {"role": "user", "content": "   "},
            {"role": "assistant", "content": "\n\n"},
        ]
        result = extract_summary(messages)
        assert result["message_count"] == 2

    def test_extract_summary_emoji_content(self):
        messages = [
            {"role": "user", "content": "🚀 Deploy the app 🎉"},
            {"role": "assistant", "content": "Done! ✅"},
        ]
        result = extract_summary(messages)
        assert result["message_count"] == 2

    def test_parse_blocks_with_bom(self):
        """UTF-8 BOM at start of text interferes with block ID regex."""
        text = "\ufeff[DEC-BOM]\nType: Decision\nStatement: BOM test\n\n"
        blocks = parse_blocks(text)
        # BOM prefix prevents regex match on first block; this is known behavior
        assert len(blocks) == 0

    def test_parse_blocks_bom_after_first_block(self):
        """BOM only affects the first block; subsequent blocks parse normally."""
        text = "\ufeff[DEC-BOM]\nStatement: skipped\n\n[DEC-OK]\nType: Decision\nStatement: parsed\n\n"
        blocks = parse_blocks(text)
        assert any(b["_id"] == "DEC-OK" for b in blocks)

    def test_parse_blocks_crlf_line_endings(self):
        text = "[DEC-CRLF]\r\nType: Decision\r\nStatement: CRLF test\r\n\r\n"
        blocks = parse_blocks(text)
        assert len(blocks) >= 1


class TestBoundaryConditions:
    """Test boundary and limit conditions."""

    def test_very_long_block_id(self):
        long_id = "A" * 200
        text = f"[{long_id}]\nType: Decision\nStatement: test\n\n"
        blocks = parse_blocks(text)
        assert isinstance(blocks, list)

    def test_block_with_no_fields(self):
        text = "[DEC-EMPTY]\n\n"
        blocks = parse_blocks(text)
        assert isinstance(blocks, list)

    def test_deeply_nested_workspace(self, tmp_path):
        deep = str(tmp_path / "a" / "b" / "c" / "d" / "workspace")
        os.makedirs(deep, exist_ok=True)
        init(deep)
        results = recall(deep, "test", limit=5)
        assert isinstance(results, list)

    def test_recall_with_special_chars(self, tmp_path):
        ws = str(tmp_path)
        init(ws)
        for query in [
            "SELECT * FROM",
            "rm -rf /",
            "<script>alert(1)</script>",
            "'; DROP TABLE",
            "..\\..\\etc\\passwd",
        ]:
            results = recall(ws, query, limit=5)
            assert isinstance(results, list)

    def test_recall_with_very_long_query(self, tmp_path):
        ws = str(tmp_path)
        init(ws)
        query = "database " * 5000
        results = recall(ws, query, limit=5)
        assert isinstance(results, list)

    def test_parse_blocks_maximum_fields(self):
        """Block with many fields should parse without error."""
        fields = "\n".join(f"Field{i}: value{i}" for i in range(100))
        text = f"[DEC-MANY]\n{fields}\n\n"
        blocks = parse_blocks(text)
        assert len(blocks) >= 1

    def test_parse_blocks_unicode_in_id(self):
        """Block ID with Unicode characters."""
        text = "[DEC-日本語]\nType: Decision\nStatement: Unicode ID test\n\n"
        blocks = parse_blocks(text)
        assert isinstance(blocks, list)

    def test_parse_blocks_newlines_only(self):
        text = "\n\n\n\n"
        blocks = parse_blocks(text)
        assert blocks == []

    def test_recall_zero_limit(self, tmp_path):
        ws = str(tmp_path)
        init(ws)
        results = recall(ws, "test", limit=0)
        assert isinstance(results, list)
        assert len(results) == 0
