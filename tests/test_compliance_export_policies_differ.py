"""Group E — the export policies do not merely EXIST, they change the output.

ROADMAP carried this as `[~]`: "The surface exists; its behaviour against the full
item text is NOT verified." A `--policy` flag that accepted three values and
produced identical bundles would satisfy every existing test while shipping a
compliance feature that complies with nothing.

MEASURED 2026-09-11, one block containing an email address:

    policy          records  email present  bytes
    full                  1  YES              184
    metadata-only         1  no               192
    redacted              1  no               183

So the redaction is real: `full` carries the address through and the other two do
not. Pinned, because "the flag is accepted" and "the flag redacts" are different
claims and only the second one protects a data subject.

Note metadata-only is LARGER than redacted (192 vs 183). That is not a defect --
it keeps structural fields a redacted export drops -- and it is recorded so nobody
reads byte count as a proxy for how much was removed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mind_mem.compliance.export import build_bundle, policy_names
from mind_mem.init_workspace import init

_EMAIL = "alice@example.com"


@pytest.fixture
def ws(tmp_path):
    root = os.path.join(str(tmp_path), "ws")
    os.makedirs(root, exist_ok=True)
    init(root)
    Path(root, "mind-mem.json").write_text(
        json.dumps({"v4": {"compliance_export": {"enabled": True}}}), encoding="utf-8"
    )
    Path(root, "decisions", "DECISIONS.md").write_text(
        f"[D-001]\nType: Decision\nStatement: Contact {_EMAIL} about the plan\n"
        f"Status: Active\nDate: 2026-01-01\n\n",
        encoding="utf-8",
    )
    return root


def _blob(ws, policy):
    bundle = build_bundle(ws, policy=policy)
    return json.dumps(getattr(bundle, "records", None) or [])


def test_all_three_policies_are_offered():
    assert set(policy_names()) == {"full", "metadata-only", "redacted"}


def test_full_carries_the_sensitive_value_through(ws):
    """POSITIVE CONTROL. Every redaction assertion below is vacuous without it:
    if the email never reached any bundle there would be nothing to redact."""
    assert _EMAIL in _blob(ws, "full")


@pytest.mark.parametrize("policy", ["redacted", "metadata-only"])
def test_the_protective_policies_remove_it(ws, policy):
    assert _EMAIL not in _blob(ws, policy), (
        f"policy {policy!r} exported the raw address; the flag is accepted but "
        f"protects nothing"
    )


def test_the_policies_produce_genuinely_different_bundles(ws):
    """Three names with one output would pass every other test in this file."""
    blobs = {p: _blob(ws, p) for p in policy_names()}
    assert len(set(blobs.values())) == 3, (
        f"policies produced {len(set(blobs.values()))} distinct bundles, not 3: "
        f"{ {k: len(v) for k, v in blobs.items()} }"
    )


def test_every_policy_still_exports_the_block(ws):
    """Redaction must not be implemented by dropping the record entirely.

    An empty bundle also contains no email. That would pass the redaction tests
    above while destroying the export's purpose.
    """
    for policy in policy_names():
        bundle = build_bundle(ws, policy=policy)
        assert len(getattr(bundle, "records", None) or []) == 1, policy
