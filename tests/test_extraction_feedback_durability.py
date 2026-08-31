# Copyright 2026 STARGA, Inc.
"""Durability of the extraction-quality feedback file.

It is optional telemetry, but it is the only record of which model was
measured to return nothing — losing it silently is what makes the caller
keep paying for that model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mind_mem import extraction_feedback as ef
from mind_mem.extraction_feedback import ExtractionFeedback


def _fill(fb: ExtractionFeedback, n: int, *, empty: bool = True) -> None:
    for _ in range(n):
        fb.record(model="m", operation="entities", input_length=10, output_count=0 if empty else 3, latency_ms=5.0)


class TestUnreadableFileIsKept:
    def test_corrupt_file_is_moved_aside_not_overwritten(self, tmp_path: Path) -> None:
        """Regression: a truncated file was silently reset to empty and
        then overwritten by the next save, erasing the history without a
        trace."""
        path = tmp_path / "fb.json"
        path.write_text('{"records": [{"model": "m"', encoding="utf-8")
        fb = ExtractionFeedback(path=str(path))
        assert fb.records == []
        backup = tmp_path / "fb.json.corrupt"
        assert backup.is_file()
        assert backup.read_text(encoding="utf-8") == '{"records": [{"model": "m"'

    def test_non_utf8_byte_does_not_raise_out_of_init(self, tmp_path: Path) -> None:
        """Regression: open() without encoding= raised UnicodeDecodeError
        straight out of __init__ on a locale-encoded byte."""
        path = tmp_path / "fb.json"
        path.write_bytes(b'{"records": [], "stats": {"caf\xe9": {}}}')
        fb = ExtractionFeedback(path=str(path))
        assert fb.records == []
        assert (tmp_path / "fb.json.corrupt").is_file()

    def test_json_scalar_file_is_quarantined(self, tmp_path: Path) -> None:
        path = tmp_path / "fb.json"
        path.write_text('"not an object"', encoding="utf-8")
        fb = ExtractionFeedback(path=str(path))
        assert fb.records == []
        assert fb._stats == {}
        assert (tmp_path / "fb.json.corrupt").is_file()

    def test_wrong_typed_sections_do_not_poison_state(self, tmp_path: Path) -> None:
        path = tmp_path / "fb.json"
        path.write_text(json.dumps({"records": "nope", "stats": []}), encoding="utf-8")
        fb = ExtractionFeedback(path=str(path))
        assert fb.records == []
        assert fb._stats == {}

    def test_good_file_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "fb.json"
        fb = ExtractionFeedback(path=str(path))
        _fill(fb, 10)
        assert path.is_file()
        again = ExtractionFeedback(path=str(path))
        assert len(again.records) == 10
        assert again.should_skip_extraction("m") is True
        assert not (tmp_path / "fb.json.corrupt").exists()

    def test_non_ascii_content_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "fb.json"
        fb = ExtractionFeedback(path=str(path))
        fb.record(model="modèle-é", operation="entities", input_length=1, output_count=1, latency_ms=1.0)
        fb.flush()
        again = ExtractionFeedback(path=str(path))
        assert again.records[0]["model"] == "modèle-é"


class TestSaveIsAtomic:
    def test_crash_mid_dump_leaves_the_previous_file_intact(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: _save truncate-wrote the live file, so a process
        killed mid-dump left a partial file that no longer parsed."""
        path = tmp_path / "fb.json"
        fb = ExtractionFeedback(path=str(path))
        _fill(fb, 10)
        before = path.read_text(encoding="utf-8")

        def _die(data: object, fh: object, **kwargs: object) -> None:
            fh.write('{"version": 1, "records": [{"mo')  # type: ignore[attr-defined]
            raise RuntimeError("killed mid-dump")

        monkeypatch.setattr(ef.json, "dump", _die)
        with pytest.raises(RuntimeError):
            fb.flush()
        monkeypatch.undo()

        assert path.read_text(encoding="utf-8") == before
        assert len(ExtractionFeedback(path=str(path)).records) == 10
        assert not (tmp_path / "fb.json.corrupt").exists()

    def test_failed_save_leaves_no_temp_file_behind(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "sub" / "fb.json"
        fb = ExtractionFeedback(path=str(path))
        _fill(fb, 9)

        def _die(data: object, fh: object, **kwargs: object) -> None:
            raise RuntimeError("killed mid-dump")

        monkeypatch.setattr(ef.json, "dump", _die)
        with pytest.raises(RuntimeError):
            fb.flush()
        monkeypatch.undo()
        assert os.listdir(path.parent) == []
