"""Group E — the provenance policy does not merely EXIST, it refuses (verified).

ROADMAP carried this as `[~]`: "That the policy EXISTS is verified; that it
actually makes the fields required is NOT." Three Group E items said the same
shape of thing -- surface present, behaviour unverified -- and that is precisely
the claim-outruns-code pattern, one step short of a false tick.

MEASURED 2026-09-11 through the real governed door (mcp.tools.governance.propose_update):

    v4.provenance.policy unset        -> accepted
    v4.provenance.policy=recommended  -> accepted
    v4.provenance.policy=required     -> REFUSED, error="provenance_required"

So the enforcement is real. Pinned here because "exists" and "enforces" are
different claims and only the second one protects anything.

Two harness errors of mine are recorded rather than hidden, because each produced
a confident WRONG answer:

  1. Without MIND_MEM_SCOPE=admin every call was refused for ACL scope, which
     reads exactly like the policy working -- a false POSITIVE.
  2. Configured under `compliance.provenance_policy`, which the code does not
     read; the real key is `v4.provenance.{enabled,policy}`. Under the wrong key
     `required` ACCEPTED a write with zero provenance, which reads exactly like
     the policy being broken -- a false NEGATIVE, and the more dangerous of the
     two, since it would have been filed as a defect.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mind_mem.init_workspace import init
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.governance import propose_update

_NO_PROVENANCE = dict(
    block_type="decision",
    statement="A decision proposed with no provenance fields at all",
    rationale="asserting whether the required policy actually refuses this",
)


@pytest.fixture(autouse=True)
def _admin(monkeypatch):
    """propose_update is admin-scoped; without this every case refuses on ACL.

    An ACL refusal is indistinguishable from a policy refusal in the response, so
    omitting this makes the enforcement tests pass for the wrong reason.
    """
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")


def _ws(tmp_path, policy):
    ws = os.path.join(str(tmp_path), "ws")
    os.makedirs(ws, exist_ok=True)
    init(ws)
    cfg = {} if policy is None else {"v4": {"provenance": {"enabled": True, "policy": policy}}}
    Path(ws, "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    return ws


def _propose(ws, **kw):
    with use_workspace(ws):
        return json.loads(propose_update(**{**_NO_PROVENANCE, **kw}))


def test_required_refuses_a_write_with_no_provenance(tmp_path):
    out = _propose(_ws(tmp_path, "required"))
    assert out.get("error"), f"required accepted a provenance-less write: {out}"
    assert "provenance" in str(out["error"]).lower(), out["error"]


def test_required_ACCEPTS_the_same_write_once_provenance_is_supplied(tmp_path):
    """POSITIVE CONTROL. Without it, a door that refused everything would pass."""
    out = _propose(
        _ws(tmp_path, "required"),
        actor_id="tester",
        actor_role="operator",
        session_id="s-1",
        tool_id="pytest",
        purpose="verify the policy accepts a complete write",
    )
    assert not out.get("error"), f"required refused a fully-provenanced write: {out}"


def test_recommended_does_not_refuse(tmp_path):
    """The two settable policies must differ, or 'required' means nothing."""
    out = _propose(_ws(tmp_path, "recommended"))
    assert not out.get("error"), f"recommended behaved like required: {out}"


def test_the_default_workspace_does_not_refuse(tmp_path):
    """Off by default: enabling enforcement must be a deliberate act."""
    out = _propose(_ws(tmp_path, None))
    assert not out.get("error"), f"an unconfigured workspace enforced: {out}"


def test_a_malformed_policy_fails_CLOSED(tmp_path):
    """A typo must not silently disable enforcement."""
    out = _propose(_ws(tmp_path, "REQUIRED_TYPO"))
    assert out.get("error"), (
        "an unrecognised policy value was ignored rather than refused; a typo in "
        "the config would silently turn enforcement off"
    )
