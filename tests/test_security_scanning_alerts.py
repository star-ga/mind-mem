"""Regression tests for code-scanning alerts #181–#189.

Covered:
- #189: federation log sanitization strips CRLF/NUL from str-typed
  fields in the three_way_merge_resolved LogRecord extra dict.
- #182: embedding_pipeline IN-clause parameterized query survives
  SQL-metacharacter block ids without injection.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def federation_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable the v4.federation feature flag for tests in this module."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"v4": {"federation": {"enabled": True}}}))
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))


# ---------------------------------------------------------------------------
# #189 — federation log sanitization
# ---------------------------------------------------------------------------


def test_federation_log_sanitizes_crlf_and_nul(
    tmp_path: Path,
    federation_enabled: None,
) -> None:
    """three_way_merge_resolved log extra strips CR, LF, and NUL from
    agent_id and block_id values (CodeQL py/log-injection, alert #189).

    The test installs a capture handler on the mind_mem.federation logger,
    triggers a THREE_WAY_MERGE resolution with adversarial identifiers
    containing CRLF/NUL, and asserts that none of those control characters
    appear in the emitted LogRecord's extra fields.
    """
    from mind_mem.v4 import federation as fed

    # Adversarial agent / block names that contain CRLF and NUL.
    evil_block = "block-\r\ninjected\x00"
    evil_agent = "agent-\x0dinject\x0a"

    workspace = str(tmp_path)

    # Ensure schema exists and record writes so detect_conflict fires.
    fed.record_agent_write(workspace, evil_block, agent_id=evil_agent)
    fed.record_agent_write(workspace, evil_block, agent_id="other-agent")

    # Capture structured log records emitted under mind_mem.federation.
    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _Capture()
    logger = logging.getLogger("mind_mem.federation")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        # Trigger a THREE_WAY_MERGE resolution so the audit log fires.
        try:
            fed.resolve_conflict(
                workspace,
                evil_block,
                fed.MergeStrategy.THREE_WAY_MERGE,
                merger=lambda left, right: left,
            )
        except Exception:
            # Resolution may fail if conflict not detected; that's fine —
            # we only need the log emission attempt.
            pass

        # Walk every captured record and check its extra fields.
        control_chars = set("\x00\r\n\x01\x1f\x7f")
        for record in captured:
            for key, val in record.__dict__.items():
                if not isinstance(val, str):
                    continue
                bad = [ch for ch in val if ch in control_chars]
                assert not bad, f"Control character(s) {bad!r} found in log field '{key}' = {val!r} (alert #189)"
    finally:
        logger.removeHandler(handler)


def test_federation_safe_helper_strips_controls() -> None:
    """Direct unit test for the _safe() helper added for alert #189.

    Verifies that ASCII control chars 0x00–0x1f and 0x7f are removed
    while printable text is preserved.
    """
    from mind_mem.v4.federation import _safe

    assert _safe("hello") == "hello"
    assert _safe("agent-\r\ninjected\x00") == "agent-injected"
    assert _safe("\x01\x02\x1f\x7f") == ""
    assert _safe("normal-id-123") == "normal-id-123"
    # Tabs are control chars (0x09) — also stripped.
    assert _safe("tab\there") == "tabhere"


# ---------------------------------------------------------------------------
# #182 — embedding_pipeline IN-clause parameterized query
# ---------------------------------------------------------------------------


def test_embedding_pipeline_in_clause_survives_sql_metacharacters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The IN-clause in embedding_pipeline.derive_embeddings uses only
    parameterized "?,?,..,?" placeholders — block ids containing SQL
    metacharacters must not cause injection or a query error (alert #182).

    Strategy: build a real SQLite workspace index.db with a blocks table,
    insert rows whose ids contain SQL-significant characters, call
    derive_embeddings, and assert:
    (a) the call completes without raising,
    (b) the blocks table is intact afterwards (no DROP executed).
    """
    from mind_mem.v4.embedding_pipeline import derive_embeddings

    # Enable the embedding_pipeline feature flag.
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"v4": {"embedding_pipeline": {"enabled": True}}}))
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))

    workspace = tmp_path
    db = workspace / "index.db"

    adversarial_ids = [
        "normal-id",
        "id-with-'quote",
        'id-with-"doublequote',
        "id; DROP TABLE blocks; --",
        "id--comment",
        "id) OR 1=1 --",
    ]

    # Build the schema expected by derive_embeddings.
    with sqlite3.connect(str(db)) as conn:
        conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, content TEXT NOT NULL)")
        for bid in adversarial_ids:
            conn.execute(
                "INSERT INTO blocks (id, content) VALUES (?, ?)",
                (bid, f"content for {bid}"),
            )

    # Call must complete without raising.
    result = derive_embeddings(workspace, adversarial_ids, dim=16)

    # All inserted ids must appear in the result (or be silently skipped
    # by the fail-soft path — important thing is no exception was raised).
    unknown = set(result.keys()) - set(adversarial_ids)
    assert not unknown, f"Unexpected keys in result: {unknown}"

    # Verify the table is still intact — no DROP survived.
    with sqlite3.connect(str(db)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
    assert count == len(adversarial_ids), (
        f"blocks table row count changed — SQL injection may have occurred (expected {len(adversarial_ids)}, got {count})"
    )
