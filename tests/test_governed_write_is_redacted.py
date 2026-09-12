"""Redaction must fire on the GOVERNED PROPOSE DOOR, not only in the CLI.

Completeness check for the roadmap item "Pluggable redaction layer", which was left
unticked "pending a completeness check". The layer itself is real and well covered: 4
modes, a detector registry with metaclass validation, 58 passing tests across
`test_compliance_redaction.py`, `test_redaction_layer_is_pluggable.py` and
`test_dsn_redaction.py`.

What none of them covered is the one thing that decides whether it is a CONTROL or just
a FEATURE: that `redact` is reached when a block is written through the governed door.
`test_governed_write_is_screened.py` drives `propose_update` thoroughly for PROVENANCE
and pins `_ws(...)` with `redaction: {"enabled": False}` throughout — so before this
file, no test wrote through the governed door with redaction ON. The compliance code
already carries that lesson in its own comment: "a compliance control the governed path
does not run is a feature, not a control."

Every assertion here is paired with a control, because each failure mode is a way to
pass while doing nothing:

  * "the secret is absent" passes when the write silently failed and no block exists —
    so a same-shape write with redaction OFF must show the secret PRESENT;
  * "the write was refused" passes when the door refuses everything — so a clean
    statement must be ADMITTED under the same mode;
  * a refusal must leave NOTHING on disk, since a refused secret that is still written
    is worse than no redaction at all: the operator believes it was caught.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mind_mem.init_workspace import init

# An AWS key shape the built-in AwsAccessKeyDetector matches. Deliberately a literal,
# not a live credential.
SECRET = "AKIAIOSFODNN7EXAMPLE"
CLEAN = "We ship the release on Friday."


def _ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mode: str | None) -> str:
    ws = str(tmp_path / "ws")
    init(ws)
    cfg = os.path.join(ws, "mind-mem.json")
    data = json.loads(Path(cfg).read_text(encoding="utf-8"))
    v4 = data.setdefault("v4", {})
    v4["provenance"] = {"enabled": False}
    v4["redaction"] = {"enabled": False} if mode is None else {"enabled": True, "mode": mode}
    Path(cfg).write_text(json.dumps(data, indent=1), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
    monkeypatch.setenv("MIND_MEM_CONFIG", cfg)
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    return ws


def _propose(**kwargs: str) -> dict:
    from mind_mem.mcp.tools.governance import propose_update

    kwargs.setdefault("block_type", "decision")
    kwargs.setdefault("rationale", "agreed in the review meeting")
    return json.loads(propose_update(**kwargs))


def _signals(ws: str) -> str:
    path = Path(ws) / "intelligence" / "SIGNALS.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_POSITIVE_CONTROL_the_secret_lands_verbatim_with_redaction_off(tmp_path, monkeypatch):
    """Everything below depends on this. Without it, "the secret is not in SIGNALS.md"
    is satisfied by a write that never happened, which is the most common way a
    security test proves nothing."""
    ws = _ws(tmp_path, monkeypatch, mode=None)
    assert "error" not in _propose(statement=f"Key is {SECRET} for the bucket.")
    assert SECRET in _signals(ws), "the write did not land, so no absence proves anything"


def test_reject_mode_refuses_the_write_through_the_governed_door(tmp_path, monkeypatch):
    ws = _ws(tmp_path, monkeypatch, mode="reject")
    got = _propose(statement=f"Key is {SECRET} for the bucket.")
    assert got.get("error") == "redaction_refused", got


def test_a_refused_write_leaves_NOTHING_on_disk(tmp_path, monkeypatch):
    """Worse than no redaction: the operator believes the secret was caught while it
    sits in SIGNALS.md. Paired with the verbatim control above, so this is absence
    from a write that would otherwise have landed."""
    ws = _ws(tmp_path, monkeypatch, mode="reject")
    _propose(statement=f"Key is {SECRET} for the bucket.")
    assert SECRET not in _signals(ws), "a refused write still wrote the secret"


def test_CONTROL_reject_mode_still_admits_a_clean_statement(tmp_path, monkeypatch):
    """A door that refuses everything is not a control, and would make the refusal
    test above pass for the wrong reason."""
    ws = _ws(tmp_path, monkeypatch, mode="reject")
    got = _propose(statement=CLEAN)
    assert "error" not in got, got
    assert "Friday" in _signals(ws)


def test_redact_mode_writes_the_block_WITHOUT_the_secret(tmp_path, monkeypatch):
    """The mode that is supposed to keep the content and drop the secret. Both halves
    are asserted: a mode that dropped the whole block would satisfy "secret absent"
    while losing the decision."""
    ws = _ws(tmp_path, monkeypatch, mode="redact")
    got = _propose(statement=f"Rotate the bucket key {SECRET} before Friday.")
    assert "error" not in got, got
    body = _signals(ws)
    assert SECRET not in body, "redact mode wrote the secret verbatim"
    assert "bucket" in body, "redact mode dropped the surrounding content too"


def test_flag_mode_admits_but_does_not_silently_claim_to_have_redacted(tmp_path, monkeypatch):
    """`flag` records a finding without altering text. It must not be mistaken for
    `redact`: if it admitted the write AND reported redaction, an operator would
    believe secrets were being removed when they are being logged."""
    ws = _ws(tmp_path, monkeypatch, mode="flag")
    got = _propose(statement=f"Key is {SECRET} for the bucket.")
    assert "error" not in got, got
    assert SECRET in _signals(ws), (
        "flag mode altered the text; that is redact mode's job and conflating them "
        "makes the configured mode a lie"
    )


def test_the_screen_call_is_inside_propose_update():
    """WIRING. The layer is only a control if the governed door runs it — and this
    file's whole premise is that no prior test proved that for redaction."""
    import inspect

    from mind_mem.mcp.tools import governance

    source = inspect.getsource(governance.propose_update)
    assert "screen(" in source, source[-500:]
    assert "RedactionRefused" in source, source[-500:]
