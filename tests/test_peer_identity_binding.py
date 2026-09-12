"""Per-peer identity, stage (a): a token may only write as the agent it is bound to.

ROADMAP ("Per-peer identity beyond bearer token"): "Today any holder of the shared
`X-MindMem-Token` can call any federation endpoint as any `agent_id`. There is no
cryptographic binding between the token and the agent identity the caller claims. A
leaked token gives full write authority over the federation surface. Two staged fixes:
**(a) per-peer tokens with a token→agent_id table; reject a write whose claimed
`agent_id` doesn't match the bound identity for the presented token.** (b) signed-write
envelopes ..."

This is (a). (b) needs Ed25519 envelopes and is not attempted here.

The groundwork was already right: `_handle_fed_write` receives `actor` (the credential
that passed auth) as a separate keyword from the body's `agent_id` (a CLAIM), and its
docstring says "a peer writing under someone else's `agent_id` is exactly the thing an
operator would want to be able to see afterwards." It logged that and allowed it. Now it
can refuse it.

**The table never holds a raw token.** Bindings are keyed on the same truncated digest
`_token_actor` already derives (`http:tok:<sha256[:12]>`), and that function is IMPORTED
rather than re-implemented — two derivations of one identity would drift, and the drift
would silently unbind every peer.

**A MALFORMED TABLE FAILS CLOSED.** This is the property that matters most and the
easiest to get backwards: if a bad entry degraded the verdict to "unbound", a typo in the
operator's config would silently restore exactly the impersonation this item exists to
close — a fail-OPEN produced by a config error, invisible in every log that only records
successes.

**An unbound token keeps working, and says so.** Most deployments have no table, and
breaking them would be worse than the hole. But "binding not enforced" must be
distinguishable from "binding checked and passed", or an operator cannot tell which one
they have.
"""

from __future__ import annotations

from mind_mem.http_transport import _token_actor
from mind_mem.peer_identity import (
    PEER_BINDINGS_ENV,
    ClaimVerdict,
    bound_agent_for_actor,
    check_agent_claim,
)

TOKEN_A = "tok-alpha-secret"
TOKEN_B = "tok-bravo-secret"


def _fed_ws(tmp_path, monkeypatch) -> str:
    """A workspace with the v4 federation flag ON.

    The refusal tests do NOT need this: the identity check runs BEFORE the
    feature-flag check, so an impersonating request is refused 403 even on a workspace
    where federation is disabled — which is the safer order, since telling an
    impersonator "the feature is off" is more than they need to know. The ALLOWED path
    does need it, or the positive control cannot reach a 200 and would "pass" against a
    503 for the wrong reason.
    """
    import json

    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    cfg = ws / "mind-mem.json"
    cfg.write_text(json.dumps({"v4": {"federation": {"enabled": True}}}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ws))
    return str(ws)


def _bind(monkeypatch, value: str) -> None:
    # No cache to clear: peer_identity reads the env on every call, the same way
    # _active_tokens does, so a rotated binding takes effect without a restart.
    monkeypatch.setenv(PEER_BINDINGS_ENV, value)


def test_no_table_configured_is_UNBOUND_not_a_refusal(monkeypatch):
    """Back-compat. Most deployments have no table and must keep working."""
    monkeypatch.delenv(PEER_BINDINGS_ENV, raising=False)
    got = check_agent_claim(_token_actor(TOKEN_A), "agent-alpha")
    assert got.verdict is ClaimVerdict.UNBOUND, got
    assert got.allowed is True


def test_a_bound_token_writing_as_its_own_agent_is_allowed(monkeypatch):
    _bind(monkeypatch, f"agent-alpha:{TOKEN_A}")
    got = check_agent_claim(_token_actor(TOKEN_A), "agent-alpha")
    assert got.verdict is ClaimVerdict.MATCH, got
    assert got.allowed is True


def test_a_bound_token_writing_as_SOMEONE_ELSE_is_REFUSED(monkeypatch):
    """THE ITEM. This is the impersonation the roadmap describes, and the only test
    here whose failure means the feature does nothing."""
    _bind(monkeypatch, f"agent-alpha:{TOKEN_A},agent-bravo:{TOKEN_B}")
    got = check_agent_claim(_token_actor(TOKEN_A), "agent-bravo")
    assert got.verdict is ClaimVerdict.MISMATCH, got
    assert got.allowed is False
    assert "agent-alpha" in got.reason and "agent-bravo" in got.reason


def test_a_token_absent_from_a_CONFIGURED_table_is_refused(monkeypatch):
    """Once an operator configures bindings they have declared that peers are known.
    An unlisted token then reads as an unknown peer, not as a legacy one — otherwise
    adding a table would leave the hole open for every token not in it, which is the
    whole population an attacker would use."""
    _bind(monkeypatch, f"agent-alpha:{TOKEN_A}")
    got = check_agent_claim(_token_actor("some-other-token"), "agent-alpha")
    assert got.allowed is False, got
    assert got.verdict is ClaimVerdict.UNKNOWN_PEER, got


def test_a_MALFORMED_table_fails_CLOSED(monkeypatch):
    """The property most easily got backwards. A bad entry must not degrade to
    "unbound": a typo would then silently restore the impersonation this closes, as a
    fail-OPEN caused by a config error and invisible in any success-only log."""
    for bad in ("agent-alpha", "agent-alpha:", ":tok", "=", "agent-alpha:tok:extra:bits"):
        _bind(monkeypatch, bad)
        got = check_agent_claim(_token_actor(TOKEN_A), "agent-alpha")
        assert got.allowed is False, (bad, got)
        assert got.verdict is ClaimVerdict.TABLE_INVALID, (bad, got)
        assert "MIND_MEM_PEER_AGENTS" in got.reason, got.reason


def test_the_reason_never_contains_the_token(monkeypatch):
    """A refusal is logged and read by everyone who can read the log. A credential that
    lands there stops being a credential — the same reason `_token_actor` exists."""
    _bind(monkeypatch, f"agent-alpha:{TOKEN_A}")
    got = check_agent_claim(_token_actor(TOKEN_A), "agent-bravo")
    assert TOKEN_A not in got.reason
    assert TOKEN_A not in str(got)


def test_the_binding_is_keyed_on_the_SAME_digest_the_door_derives(monkeypatch):
    """Two derivations of one identity drift, and the drift silently unbinds every
    peer. `peer_identity` imports `_token_actor` rather than re-deriving."""
    import ast
    import inspect

    import mind_mem.peer_identity as pi

    # Asserted on the IMPORT GRAPH, not on the word "sha256": the first version of this
    # test failed on a correct module because its own docstring says "sha256[:12]" while
    # explaining that it does not derive one. A test that matches its own prose reports
    # a defect that is not there. Deriving a digest requires hashlib, so its absence is
    # the precise, unambiguous check.
    tree = ast.parse(inspect.getsource(pi))
    imported = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "hashlib" not in imported, (
        "peer_identity imports hashlib, so it can derive its own digest; it must use "
        "_token_actor so the door and the table cannot disagree about a peer's identity"
    )
    assert "http_transport" in imported, (
        "peer_identity does not import the door's identity function at all"
    )
    _bind(monkeypatch, f"agent-alpha:{TOKEN_A}")
    assert bound_agent_for_actor(_token_actor(TOKEN_A)) == "agent-alpha"


def test_whitespace_and_ordering_do_not_change_the_answer(monkeypatch):
    _bind(monkeypatch, f"  agent-alpha : {TOKEN_A} , agent-bravo:{TOKEN_B}  ")
    assert check_agent_claim(_token_actor(TOKEN_B), "agent-bravo").allowed is True
    assert check_agent_claim(_token_actor(TOKEN_B), "agent-alpha").allowed is False


def test_two_agents_may_not_share_one_token(monkeypatch):
    """A token bound to two agents is an ambiguous identity, and resolving it either
    way invents authority the operator did not grant. Refused as a table error."""
    _bind(monkeypatch, f"agent-alpha:{TOKEN_A},agent-bravo:{TOKEN_A}")
    got = check_agent_claim(_token_actor(TOKEN_A), "agent-alpha")
    assert got.verdict is ClaimVerdict.TABLE_INVALID, got
    assert got.allowed is False


def test_the_unauthenticated_loopback_actor_is_never_silently_bound(monkeypatch):
    """`--allow-unauthenticated-localhost` presents no credential. With a table
    configured, that door must not pass a binding check it cannot satisfy."""
    from mind_mem.http_transport import HTTP_UNAUTHENTICATED_ACTOR

    _bind(monkeypatch, f"agent-alpha:{TOKEN_A}")
    got = check_agent_claim(HTTP_UNAUTHENTICATED_ACTOR, "agent-alpha")
    assert got.allowed is False, got


def test_the_verdict_set_is_closed():
    assert {v.name for v in ClaimVerdict} == {
        "UNBOUND", "MATCH", "MISMATCH", "UNKNOWN_PEER", "TABLE_INVALID"}


def test_the_module_reads_no_clock_and_no_randomness():
    import ast
    import inspect

    import mind_mem.peer_identity as pi

    tree = ast.parse(inspect.getsource(pi))
    names = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    names |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not (names & {"time", "random", "datetime", "uuid"}), sorted(names)


# ---------------------------------------------------------------------------
# Wiring: the DOOR must refuse. A binding module the handler never calls leaves
# the hole exactly where it was.
# ---------------------------------------------------------------------------


def test_the_federation_door_REFUSES_an_impersonating_write(monkeypatch, tmp_path):
    """THE SECURITY FIX, at the door. Before this, the handler logged the mismatch and
    allowed it, so a leaked token carried write authority over every agent."""
    from mind_mem.http_transport import _handle_fed_write

    _bind(monkeypatch, f"agent-alpha:{TOKEN_A},agent-bravo:{TOKEN_B}")
    status, body = _handle_fed_write(
        str(tmp_path),
        {"block_id": "DEC-1", "agent_id": "agent-bravo"},
        actor=_token_actor(TOKEN_A),
    )
    assert status == 403, (status, body)
    assert body["ok"] is False
    assert body["verdict"] == "mismatch", body
    assert TOKEN_A not in str(body), "the refusal leaked the token"


def test_the_refused_write_MUTATED_NOTHING(monkeypatch, tmp_path):
    """A refusal that already bumped the version vector would leave the impersonated
    agent's history altered — the check has to precede the write, not accompany it."""
    from mind_mem.http_transport import _handle_fed_write

    _bind(monkeypatch, f"agent-alpha:{TOKEN_A},agent-bravo:{TOKEN_B}")
    _handle_fed_write(
        str(tmp_path), {"block_id": "DEC-1", "agent_id": "agent-bravo"},
        actor=_token_actor(TOKEN_A),
    )
    from mind_mem.v4 import federation as fed

    # No version vector should exist for the impersonated pair at all.
    clock = fed.read_vclock(str(tmp_path)) if hasattr(fed, "read_vclock") else {}
    assert "agent-bravo" not in str(clock), clock


def test_CONTROL_the_bound_agent_may_still_write(monkeypatch, tmp_path):
    """POSITIVE CONTROL. A door that refused every federation write would satisfy the
    two tests above while breaking federation entirely — and the refusal is the kind of
    change that gets shipped without anyone noticing it refuses everything."""
    from mind_mem.http_transport import _handle_fed_write

    ws = _fed_ws(tmp_path, monkeypatch)
    _bind(monkeypatch, f"agent-alpha:{TOKEN_A}")
    status, body = _handle_fed_write(
        ws, {"block_id": "DEC-1", "agent_id": "agent-alpha"},
        actor=_token_actor(TOKEN_A),
    )
    assert status == 200, (status, body)
    assert body["identity_bound"] is True, body


def test_an_unbound_deployment_still_writes_and_SAYS_it_is_unbound(monkeypatch, tmp_path):
    """Back-compat plus honesty: the write succeeds, and the response distinguishes
    "checked and passed" from "not enforced here"."""
    from mind_mem.http_transport import _handle_fed_write

    ws = _fed_ws(tmp_path, monkeypatch)
    monkeypatch.delenv(PEER_BINDINGS_ENV, raising=False)
    status, body = _handle_fed_write(
        ws, {"block_id": "DEC-1", "agent_id": "anything-at-all"},
        actor=_token_actor(TOKEN_A),
    )
    assert status == 200, (status, body)
    assert body["identity_bound"] is False, body
