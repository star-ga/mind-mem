"""Reader-only compatibility controls for mixed v1/v2 served ledgers.

The fixture is intentionally written in a temporary workspace. It exercises
the released v1 writer and a real v2 row shape without touching the live
served ledger or promoting the v2 serving callers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from mind_mem.accountability_dashboard import ledger_panel
from mind_mem.recall_digests import query_hash, served_set_digest
from mind_mem.served_ledger import (
    HEAD_RELPATH,
    ServedRun,
    ServedRunV2,
    _write_row,
    append_served_run,
    ledger_path,
    read_served_runs,
    row_hash,
    run_id,
    verify_served_chain,
)

PIPELINE = "b" * 64
ANCHOR = "c" * 64


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True)
    (ws / "mind-mem.json").write_text(json.dumps({"served_ledger": {"enabled": True}}), encoding="utf-8")
    return ws


def _append_v1(ws: Path, question: str, block_id: str) -> ServedRun:
    row = append_served_run(
        ws,
        query_hash=query_hash(question),
        served_digest=served_set_digest([block_id]),
        ids=[block_id],
        pipeline_hash=PIPELINE,
        index_anchor=ANCHOR,
        scoring_instant="2026-09-13",
    )
    assert isinstance(row, ServedRun)
    return row


def _append_v2_tail(ws: Path, predecessor: ServedRun) -> ServedRunV2:
    ids = ("D-V2", "D-V2-related")
    qh = query_hash("v2 compatibility")
    digest = served_set_digest(ids)
    row = ServedRunV2(
        seq=predecessor.seq + 1,
        prev_row_hash=row_hash(predecessor),
        run_id=run_id(query_hash=qh, served_digest=digest, pipeline_hash=PIPELINE),
        query_hash=qh,
        served_digest=digest,
        ids=ids,
        pipeline_hash=PIPELINE,
        index_anchor=ANCHOR,
        scoring_instant="2026-09-13",
        serve_kind="attested",
        context_digest="a" * 64,
    )
    _write_row(ws, row)
    return row


def _mixed_workspace(tmp_path: Path) -> Path:
    ws = _workspace(tmp_path)
    _append_v1(ws, "v1 first", "D-V1-first")
    second = _append_v1(ws, "v1 second", "D-V1-second")
    _append_v2_tail(ws, second)
    return ws


def _replace_last_row(ws: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = Path(ledger_path(ws))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3, "mutation fixture must contain both v1 history and a v2 tail"
    row = json.loads(lines[-1])
    mutate(row)
    lines[-1] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_mixed_reader_verifies_real_v1_history_and_v2_tail(tmp_path: Path) -> None:
    ws = _mixed_workspace(tmp_path)

    rows = read_served_runs(ws)
    verdict = verify_served_chain(ws)

    assert len(rows) == 3, "positive control: the fixture must contain three persisted rows"
    assert isinstance(rows[0], ServedRun)
    assert isinstance(rows[1], ServedRun)
    assert isinstance(rows[2], ServedRunV2)
    assert verdict.ok is True
    assert verdict.rows_checked == 3
    assert verdict.bad_seq is None


def test_unchanged_writer_appends_v1_after_v2_tail(tmp_path: Path) -> None:
    ws = _mixed_workspace(tmp_path)

    appended = _append_v1(ws, "post v2 old writer", "D-V1-after-v2")
    rows = read_served_runs(ws)

    assert appended.seq == 3
    assert isinstance(rows[-1], ServedRun)
    assert set(rows[-1].to_row()) == {
        "seq",
        "prev_row_hash",
        "run_id",
        "query_hash",
        "served_digest",
        "ids",
        "pipeline_hash",
        "index_anchor",
        "scoring_instant",
    }
    assert verify_served_chain(ws).ok is True


def test_shared_dashboard_reader_accepts_mixed_ledger(tmp_path: Path) -> None:
    ws = _mixed_workspace(tmp_path)

    panel = ledger_panel(str(ws))

    assert panel.present is True
    assert panel.rows == 3
    assert panel.ok is True


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("context", lambda row: row.__setitem__("context_digest", "d" * 64)),
        ("body", lambda row: row.__setitem__("query_hash", "e" * 64)),
        ("prev", lambda row: row.__setitem__("prev_row_hash", "f" * 64)),
    ],
)
def test_v2_context_body_and_prev_tampering_is_convicted(tmp_path: Path, name: str, mutate: Callable[[dict[str, Any]], None]) -> None:
    ws = _mixed_workspace(tmp_path / name)
    _replace_last_row(ws, mutate)

    verdict = verify_served_chain(ws)

    assert verdict.ok is False
    assert verdict.rows_checked < 3 or verdict.bad_seq is not None


def test_v2_invalid_serve_kind_is_refused(tmp_path: Path) -> None:
    ws = _mixed_workspace(tmp_path)
    _replace_last_row(ws, lambda row: row.__setitem__("serve_kind", "forged"))

    with pytest.raises(ValueError, match="serve_kind"):
        read_served_runs(ws)
    assert verify_served_chain(ws).ok is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.__setitem__("unexpected", True),
        lambda row: row.pop("context_digest"),
    ],
)
def test_v2_unknown_or_partial_schema_is_refused(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    ws = _mixed_workspace(tmp_path)
    _replace_last_row(ws, mutate)

    with pytest.raises(ValueError, match="schema"):
        read_served_runs(ws)
    assert verify_served_chain(ws).ok is False


def test_malformed_tail_is_not_an_empty_or_v1_fixture(tmp_path: Path) -> None:
    ws = _mixed_workspace(tmp_path)
    path = Path(ledger_path(ws))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    path.write_text("\n".join(lines[:2] + ["{not-json"]) + "\n", encoding="utf-8")

    assert verify_served_chain(ws).ok is False


@pytest.mark.parametrize("payload", ["1", "[]", '"not-a-row"'])
def test_non_object_json_row_is_refused_without_escaping_verifier(tmp_path: Path, payload: str) -> None:
    ws = _workspace(tmp_path)
    path = Path(ledger_path(ws))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        read_served_runs(ws)
    verdict = verify_served_chain(ws)
    assert verdict.ok is False
    assert "JSON object" in verdict.reason


@pytest.mark.parametrize("head", [None, ""])
def test_missing_or_empty_head_does_not_verify_mixed_rows(tmp_path: Path, head: str | None) -> None:
    ws = _mixed_workspace(tmp_path / ("missing" if head is None else "empty"))
    head_path = Path(ws, HEAD_RELPATH)
    if head is None:
        head_path.unlink()
    else:
        head_path.write_text("\n", encoding="utf-8")

    verdict = verify_served_chain(ws)

    assert verdict.ok is False
    assert verdict.rows_checked >= 2
    assert verdict.bad_seq == 2
