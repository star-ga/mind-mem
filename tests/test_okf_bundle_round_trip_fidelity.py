"""An OKF bundle must survive its own writer, and a dropped concept must be loud.

Two halves of the same defect — an import that reports success while losing
content:

* The writer emitted a double-quoted scalar with a **raw newline** inside it.
  The reader is line-based, so ``description: "line one\\nline two"`` came back
  as ``'"line one'`` — first line only, opening quote retained. Multi-line
  values are ordinary corpus content: ``block_parser`` joins indented
  continuation lines into a scalar with ``\\n``, so any multi-line
  ``Statement`` was truncated by a write/read round trip.
* ``_parse_okf_frontmatter`` bailed to ``{}`` unless the very first line was
  exactly ``---``. ``str.strip()`` does not remove a UTF-8 BOM, so a
  BOM-prefixed bundle parsed to ``{}`` for every file, and
  ``import_okf_bundle`` dropped each concept on ``if not fm.get("type")`` with
  no log and no counter — an import that returned ``[]`` and looked fine.
"""

from __future__ import annotations

import logging

import pytest

from mind_mem.core_export import (
    _parse_okf_frontmatter,
    _unquote,
    _yaml_scalar,
    import_okf_bundle,
)

MULTILINE = "line one\nline two\nline three"


class _CapturingHandler(logging.Handler):
    """Collect structured events off the module's own logger.

    The structured logger sets ``propagate = False`` and binds its stream
    handler at import time, so neither ``caplog`` nor ``capfd`` sees it —
    attaching here reads the events themselves rather than their rendering.
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, dict]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.events.append((record.getMessage(), getattr(record, "data", None) or {}))


@pytest.fixture
def okf_log():
    logger = logging.getLogger("mind-mem.core_export")
    handler = _CapturingHandler()
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


def _frontmatter(**fields: str) -> str:
    body = "".join(f"{k}: {_yaml_scalar(v)}\n" for k, v in fields.items())
    return f"---\n{body}---\n\n# concept\n"


class TestScalarRoundTrip:
    """What the writer emits, the reader must read back unchanged."""

    def test_multiline_value_survives_the_round_trip(self) -> None:
        """Before the fix this came back as ``'"line one'``."""
        doc = _frontmatter(type="decision", description=MULTILINE)
        assert _parse_okf_frontmatter(doc)["description"] == MULTILINE

    def test_emitted_scalar_contains_no_raw_newline(self) -> None:
        """A raw newline inside a quoted scalar ends the value mid-string."""
        emitted = _yaml_scalar(MULTILINE)
        assert "\n" not in emitted
        assert emitted.startswith('"') and emitted.endswith('"')

    def test_a_multiline_value_does_not_swallow_the_next_key(self) -> None:
        """The truncated form also left the rest of the value as junk lines."""
        doc = _frontmatter(description=MULTILINE, type="decision")
        fm = _parse_okf_frontmatter(doc)
        assert fm["type"] == "decision"
        assert fm["description"] == MULTILINE

    @pytest.mark.parametrize(
        "value",
        [
            'quotes "inside" it',
            "back\\slash",
            "trailing backslash\\",
            "tab\tseparated",
            "carriage\r\nreturn",
            'mixed \\" escape',
            "plain unquoted value",
            "- leading dash",
        ],
    )
    def test_escaping_is_reversible(self, value: str) -> None:
        assert _unquote(_yaml_scalar(value)) == value

    def test_unquote_leaves_a_bare_scalar_alone(self) -> None:
        assert _unquote("bare value") == "bare value"


class TestBomTolerance:
    """A BOM is not whitespace; the reader must not treat it as no frontmatter."""

    def test_bom_prefixed_frontmatter_parses(self) -> None:
        doc = "﻿" + _frontmatter(type="decision", title="X")
        fm = _parse_okf_frontmatter(doc)
        assert fm.get("type") == "decision"
        assert fm.get("title") == "X"

    def test_a_file_with_no_frontmatter_still_parses_to_nothing(self) -> None:
        assert _parse_okf_frontmatter("# just a heading\n") == {}


class TestDroppedConceptsAreReported:
    """Silently returning ``[]`` is a failed import wearing a success shape."""

    def test_bom_bundle_is_imported_not_dropped(self, tmp_path) -> None:
        (tmp_path / "D-1.md").write_text(
            "﻿" + _frontmatter(type="decision", description=MULTILINE),
            encoding="utf-8",
        )
        blocks = import_okf_bundle(tmp_path)
        assert [b["type"] for b in blocks] == ["decision"]
        assert blocks[0]["Statement"] == MULTILINE

    def test_skipped_concept_is_logged(self, tmp_path, okf_log) -> None:
        """Before the fix the concept vanished with no log line and no counter."""
        (tmp_path / "junk.md").write_text("no frontmatter here\n", encoding="utf-8")
        blocks = import_okf_bundle(tmp_path)
        assert blocks == []
        skipped = [data for event, data in okf_log.events if event == "okf_concept_skipped"]
        assert len(skipped) == 1
        assert skipped[0]["path"].endswith("junk.md")
        assert skipped[0]["reason"] == "no_frontmatter"

    def test_a_concept_missing_only_type_is_reported_as_such(self, tmp_path, okf_log) -> None:
        (tmp_path / "b.md").write_text("---\ntitle: no type\n---\n", encoding="utf-8")
        assert import_okf_bundle(tmp_path) == []
        assert [data["reason"] for event, data in okf_log.events if event == "okf_concept_skipped"] == ["missing_type"]

    def test_an_import_that_dropped_everything_is_reported(self, tmp_path, okf_log) -> None:
        (tmp_path / "a.md").write_text("no frontmatter\n", encoding="utf-8")
        (tmp_path / "b.md").write_text("---\ntitle: no type\n---\n", encoding="utf-8")
        assert import_okf_bundle(tmp_path) == []
        empty = [data for event, data in okf_log.events if event == "okf_import_empty"]
        assert empty and empty[0]["skipped"] == 2

    def test_conformant_concept_is_not_reported_as_skipped(self, tmp_path, okf_log) -> None:
        (tmp_path / "D-1.md").write_text(_frontmatter(type="decision"), encoding="utf-8")
        blocks = import_okf_bundle(tmp_path)
        assert len(blocks) == 1
        assert okf_log.events == []
