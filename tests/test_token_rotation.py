"""Regression tests for the token rotation primitive (roadmap v4.0.x).

``mm token rotate`` mints a new url-safe token (192-bit by default)
and emits the shell export to use it. The HTTP transport reads
``MIND_MEM_TOKENS`` (comma-separated) on every request so no restart
is needed. During the grace window both old and new tokens
authenticate; after the grace window the operator drops the old
entry.
"""

from __future__ import annotations

import argparse
import json

import pytest

from mind_mem import http_transport, mm_cli

# ---------------------------------------------------------------------------
# _active_tokens helper — fallback ladder
# ---------------------------------------------------------------------------


def test_active_tokens_reads_multi(monkeypatch: pytest.MonkeyPatch) -> None:
    """MIND_MEM_TOKENS takes precedence over MIND_MEM_TOKEN + fallback."""
    monkeypatch.setenv("MIND_MEM_TOKENS", "tok-a,tok-b , tok-c")
    monkeypatch.setenv("MIND_MEM_TOKEN", "stale-single")
    assert http_transport._active_tokens(fallback="ignored") == [
        "tok-a",
        "tok-b",
        "tok-c",
    ]


def test_active_tokens_falls_back_to_single(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When MIND_MEM_TOKENS unset, fall back to MIND_MEM_TOKEN."""
    monkeypatch.delenv("MIND_MEM_TOKENS", raising=False)
    monkeypatch.setenv("MIND_MEM_TOKEN", "only-token")
    assert http_transport._active_tokens(fallback=None) == ["only-token"]


def test_active_tokens_falls_back_to_handler_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When neither env var is set, fall back to the handler-bound token."""
    monkeypatch.delenv("MIND_MEM_TOKENS", raising=False)
    monkeypatch.delenv("MIND_MEM_TOKEN", raising=False)
    assert http_transport._active_tokens(fallback="startup-token") == ["startup-token"]


def test_active_tokens_empty_when_nothing_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env, no fallback → empty list (authentication fails closed)."""
    monkeypatch.delenv("MIND_MEM_TOKENS", raising=False)
    monkeypatch.delenv("MIND_MEM_TOKEN", raising=False)
    assert http_transport._active_tokens(fallback=None) == []


def test_active_tokens_strips_empty_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty entries (',,,', trailing comma) are stripped, not counted."""
    monkeypatch.setenv("MIND_MEM_TOKENS", ",,real-token,,")
    assert http_transport._active_tokens(fallback=None) == ["real-token"]


# ---------------------------------------------------------------------------
# `mm token rotate` CLI — output schema + entropy
# ---------------------------------------------------------------------------


def test_token_rotate_emits_required_fields(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`mm token rotate` prints a JSON report with the documented fields."""
    monkeypatch.delenv("MIND_MEM_TOKENS", raising=False)
    monkeypatch.delenv("MIND_MEM_TOKEN", raising=False)
    ns = argparse.Namespace(length=24, grace_seconds=3600)
    rc = mm_cli._cmd_token_rotate(ns)
    assert rc == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    for key in (
        "new_token",
        "active_tokens_after_rotate",
        "active_tokens_after_grace",
        "grace_seconds",
        "shell",
        "shell_final",
        "instructions",
    ):
        assert key in report, f"missing field {key!r}"
    # New token must be url-safe (no padding, no slashes) and length ≥ 24.
    assert len(report["new_token"]) >= 24
    assert "/" not in report["new_token"]
    assert "+" not in report["new_token"]
    # The shell export must add the new token to MIND_MEM_TOKENS.
    assert "MIND_MEM_TOKENS=" in report["shell"]
    assert report["new_token"] in report["shell"]


def test_token_rotate_grace_window_preserves_existing_tokens(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When MIND_MEM_TOKENS already has tokens, rotation appends — the
    new set MUST include both the new token AND every previous token
    so in-flight clients stay authenticated through the grace window."""
    monkeypatch.setenv("MIND_MEM_TOKENS", "old-token-1,old-token-2")
    monkeypatch.delenv("MIND_MEM_TOKEN", raising=False)
    ns = argparse.Namespace(length=24, grace_seconds=86400)
    mm_cli._cmd_token_rotate(ns)
    out = capsys.readouterr().out
    report = json.loads(out)
    rotated = report["active_tokens_after_rotate"]
    assert "old-token-1" in rotated
    assert "old-token-2" in rotated
    assert report["new_token"] in rotated
    # New token is first (canonical write token going forward).
    assert rotated[0] == report["new_token"]
    # After grace: only the new token remains.
    assert report["active_tokens_after_grace"] == [report["new_token"]]


def test_token_rotate_falls_back_to_single_token(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Single-token deployment (MIND_MEM_TOKEN only) rotates into the
    multi-token set so the operator can upgrade rotation discipline
    without a re-bootstrap."""
    monkeypatch.delenv("MIND_MEM_TOKENS", raising=False)
    monkeypatch.setenv("MIND_MEM_TOKEN", "legacy-single")
    ns = argparse.Namespace(length=24, grace_seconds=3600)
    mm_cli._cmd_token_rotate(ns)
    out = capsys.readouterr().out
    report = json.loads(out)
    assert "legacy-single" in report["active_tokens_after_rotate"]


# ---------------------------------------------------------------------------
# Token entropy floor
# ---------------------------------------------------------------------------


def test_token_rotate_enforces_min_length(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--length`` below 16 is silently floored to 16 (minimum entropy)."""
    ns = argparse.Namespace(length=4, grace_seconds=60)  # request too small
    mm_cli._cmd_token_rotate(ns)
    out = capsys.readouterr().out
    report = json.loads(out)
    # 16 url-safe bytes ≈ 22 chars after base64-url encoding.
    assert len(report["new_token"]) >= 22


# ---------------------------------------------------------------------------
# CLI integration: parser recognises `mm token rotate`
# ---------------------------------------------------------------------------


def test_cli_parser_recognises_token_rotate() -> None:
    """``mm token rotate`` must be a registered subcommand."""
    parser = mm_cli.build_parser()
    args = parser.parse_args(["token", "rotate"])
    assert args.func is mm_cli._cmd_token_rotate
    assert args.length == 24
    assert args.grace_seconds == 86_400


def test_cli_parser_accepts_token_rotate_flags() -> None:
    """``--length`` and ``--grace-seconds`` are wired."""
    parser = mm_cli.build_parser()
    args = parser.parse_args(["token", "rotate", "--length", "32", "--grace-seconds", "300"])
    assert args.length == 32
    assert args.grace_seconds == 300
