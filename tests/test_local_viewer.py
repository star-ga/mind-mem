"""`mm view` — a read-only local viewer that cannot serve withheld content.

ROADMAP ("Local visual viewer"): "`mm view` web UI not yet shipped. Stack target: stdlib
HTTP + minimal JS/D3." And the flag registry recorded the honest open question: "No viewer
surface ships in this package. Wiring question: the flag may belong to a client, not to
mind-mem; if so it should move rather than be deleted." It belongs here — the viewer reads
the governed corpus, and the admission rules that decide what is servable live here.

**THE LOAD-BEARING PROPERTY IS THE ADMISSION FILTER, not the HTML.** This repo has already
paid for the alternative: a block loader ran `SELECT id, status, tags, json_blob FROM
blocks WHERE parent_id = ''` and never filtered on `status`, so quarantined and pending
content surfaced verbatim through a user-scope tool. The column was right there in the
SELECT, which is exactly why it read as safe. A viewer is a read surface, so every block it
shows goes through `admit_corpus` — and the test for that is paired with a control proving
an admitted block IS shown, because "the quarantined block is absent" passes just as well
when nothing was loaded at all.

Three more refusals, each fail-closed:
  * a NON-LOOPBACK bind is refused — an unauthenticated read surface on 0.0.0.0 publishes
    the governed corpus to the network;
  * the flag OFF means it does not start, probed with `is_enabled_quiet` so a flag-off
    build is not observably different from one without the feature;
  * there is NO write path at all, asserted over the import graph rather than trusted.
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib

import pytest

from mind_mem.viewer import (
    VIEWER_FLAG,
    ViewerRefused,
    build_payload,
    resolve_bind,
)

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "mind_mem"

ACTIVE = {"_id": "DEC-1", "Type": "decision", "Status": "active", "Excerpt": "We ship Friday."}
PENDING = {"_id": "SIG-9", "Type": "signal", "Status": "pending", "Excerpt": "unreviewed claim"}
QUARANTINED = {"_id": "DEC-2", "Type": "decision", "Status": "quarantined", "Excerpt": "poisoned"}


def test_an_admitted_block_IS_shown(tmp_path, monkeypatch):
    """POSITIVE CONTROL FIRST. Every absence assertion below is worthless without it:
    "the withheld block is not in the payload" is satisfied by a payload built from an
    empty corpus."""
    import mind_mem.viewer as v

    monkeypatch.setattr(v, "_load_corpus", lambda ws: [ACTIVE, PENDING, QUARANTINED])
    payload = build_payload(str(tmp_path))
    ids = {b["id"] for b in payload["blocks"]}
    assert "DEC-1" in ids, payload


def test_a_PENDING_block_is_NOT_shown(tmp_path, monkeypatch):
    import mind_mem.viewer as v

    monkeypatch.setattr(v, "_load_corpus", lambda ws: [ACTIVE, PENDING, QUARANTINED])
    ids = {b["id"] for b in build_payload(str(tmp_path))["blocks"]}
    assert "SIG-9" not in ids, "an unreviewed signal reached the viewer"


def test_a_QUARANTINED_block_is_NOT_shown(tmp_path, monkeypatch):
    """The exact failure this repo already paid for once."""
    import mind_mem.viewer as v

    monkeypatch.setattr(v, "_load_corpus", lambda ws: [ACTIVE, PENDING, QUARANTINED])
    ids = {b["id"] for b in build_payload(str(tmp_path))["blocks"]}
    assert "DEC-2" not in ids, "quarantined content reached the viewer"


def test_the_payload_never_carries_a_withheld_excerpt_anywhere(tmp_path, monkeypatch):
    """Not just absent from `blocks` — absent from the whole serialised payload. A count,
    a summary or a 'recent' list rebuilt from the unfiltered corpus would leak the text
    while the block list looked clean."""
    import mind_mem.viewer as v

    monkeypatch.setattr(v, "_load_corpus", lambda ws: [ACTIVE, PENDING, QUARANTINED])
    blob = json.dumps(build_payload(str(tmp_path)))
    assert "poisoned" not in blob, blob[:400]
    assert "unreviewed claim" not in blob, blob[:400]


def test_the_admission_filter_is_CALLED_not_reimplemented():
    """A status check hand-rolled here would drift from the shared rules the moment a
    status is added — which is why `admit_corpus` exists as the one authority."""
    source = inspect.getsource(pathlib.Path(SRC / "viewer.py").read_text.__self__.__class__) \
        if False else (SRC / "viewer.py").read_text(encoding="utf-8")
    assert "admit_corpus" in source, "the viewer does not call the shared admission filter"
    tree = ast.parse(source)
    imported = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert any("admissibility" in m for m in imported), sorted(imported)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.0.5", "example.com"])
def test_a_NON_LOOPBACK_bind_is_refused(host):
    """An unauthenticated read surface on a routable address publishes the governed
    corpus to the network. Fail closed, and name the host in the refusal."""
    with pytest.raises(ViewerRefused, match="loopback"):
        resolve_bind(host, 8900)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_binds_are_allowed(host):
    """POSITIVE CONTROL: a refusal that refused everything would make the viewer
    unusable and satisfy the test above."""
    assert resolve_bind(host, 8900)[1] == 8900


def test_a_privileged_or_out_of_range_port_is_refused():
    with pytest.raises(ViewerRefused):
        resolve_bind("127.0.0.1", 80)
    with pytest.raises(ViewerRefused):
        resolve_bind("127.0.0.1", 70000)


def test_the_flag_is_declared_and_probed_quietly():
    """`is_enabled_quiet`, never `is_enabled`: the latter warns on a malformed config, so
    a probe on an OFF path would make the flag-off build observably different from one
    that never had the feature."""
    from mind_mem.v4.feature_flags import ALL_V4_FLAGS

    assert VIEWER_FLAG in ALL_V4_FLAGS
    source = (SRC / "viewer.py").read_text(encoding="utf-8")
    assert "is_enabled_quiet" in source
    assert "import is_enabled\n" not in source


def test_THERE_IS_NO_WRITE_PATH(tmp_path):
    """Asserted over the import graph, not trusted. A viewer that can write is not a
    viewer, and the governed store has exactly one write door."""
    tree = ast.parse((SRC / "viewer.py").read_text(encoding="utf-8"))
    banned = {"block_store", "apply_engine", "governance_gate", "capture", "write_block"}
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            seen.add(node.names[0].name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            seen.add((node.module or "").split(".")[-1])
    assert not (seen & banned), sorted(seen & banned)


def test_the_no_write_check_is_not_vacuous():
    """POSITIVE CONTROL for the walk above: it must actually parse imports, or it passes
    by inspecting nothing."""
    tree = ast.parse((SRC / "viewer.py").read_text(encoding="utf-8"))
    names = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert len(names) >= 3, len(names)
