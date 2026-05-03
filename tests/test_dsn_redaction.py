"""Tests for DSN password redaction in mm_cli.

The migrate-store command writes a JSON receipt that includes the DSN.
v3.8.13 used a urlparse-only redactor that left the keyword form
(`host=… password=secret …`) intact, leaking the password to disk.
v3.8.14 ``_redact_dsn`` handles both forms.
"""

from __future__ import annotations

import pytest

from mind_mem.mm_cli import _redact_dsn


class TestRedactDSN:
    def test_url_form_password_redacted(self) -> None:
        out = _redact_dsn("postgresql://mindmem:supersecret@127.0.0.1:5432/mindmem")
        assert "supersecret" not in out
        assert "mindmem:***@" in out  # username preserved, password masked.
        assert "127.0.0.1" in out
        assert "5432" in out
        assert "/mindmem" in out

    def test_url_form_no_password(self) -> None:
        # No password -> no change beyond the optional default port handling.
        out = _redact_dsn("postgresql://mindmem@127.0.0.1:5432/mindmem")
        assert "mindmem" in out
        assert "***" not in out

    def test_keyword_form_password_redacted(self) -> None:
        out = _redact_dsn("host=localhost dbname=mindmem user=mindmem password=supersecret")
        assert "supersecret" not in out
        assert "password=***" in out
        assert "host=localhost" in out
        assert "dbname=mindmem" in out

    def test_keyword_form_password_redacted_quoted(self) -> None:
        # psycopg accepts unquoted values with no whitespace; check the
        # literal-value case where no spaces appear.
        out = _redact_dsn("host=h dbname=d user=u password=secret123")
        assert "secret123" not in out
        assert "password=***" in out

    def test_keyword_form_case_insensitive(self) -> None:
        out = _redact_dsn("HOST=h dbname=d user=u PASSWORD=secret123")
        assert "secret123" not in out

    def test_empty_dsn(self) -> None:
        assert _redact_dsn("") == ""

    def test_dsn_without_password(self) -> None:
        # Keyword form with no password must round-trip unchanged.
        out = _redact_dsn("host=localhost dbname=mindmem user=mindmem")
        assert out == "host=localhost dbname=mindmem user=mindmem"
