"""Per-peer identity, stage (a): bind a token to the agent it may write as.

ROADMAP ("Per-peer identity beyond bearer token"): "Today any holder of the shared
``X-MindMem-Token`` can call any federation endpoint as any ``agent_id``. There is no
cryptographic binding between the token and the agent identity the caller claims. A
leaked token gives full write authority over the federation surface. Two staged fixes:
**(a) per-peer tokens with a token->agent_id table; reject a write whose claimed
``agent_id`` doesn't match the bound identity for the presented token.** (b) signed-write
envelopes ..."

This module is (a). Stage (b) needs Ed25519 envelopes and is not attempted here.

The groundwork was already right: the federation write handler receives ``actor`` -- the
credential that passed auth -- separately from the body's ``agent_id``, which is a CLAIM,
and its docstring already noted that "a peer writing under someone else's ``agent_id`` is
exactly the thing an operator would want to be able to see afterwards." It could see it.
Now it can refuse it.

THE TABLE NEVER HOLDS A RAW TOKEN. Bindings are keyed on the truncated digest
``_token_actor`` already derives (``http:tok:<sha256[:12]>``), and that function is
IMPORTED rather than re-implemented: two derivations of one identity drift, and the drift
would silently unbind every peer while every log kept reporting success.

A MALFORMED TABLE FAILS CLOSED. This is the property most easily got backwards. If a bad
entry degraded to "unbound", a typo in the operator's config would silently restore the
impersonation this exists to close -- a fail-OPEN caused by a config error, invisible in
any log that records only what succeeded.

AN UNBOUND DEPLOYMENT KEEPS WORKING, AND SAYS SO. Most deployments have no table and
breaking them would be worse than the hole; but "binding not enforced" has to be
distinguishable from "binding checked and passed", or an operator cannot tell which one
they are running.

No clock, no randomness: the verdict is a pure function of the table and the two
identities.
"""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from enum import Enum

from .http_transport import HTTP_UNAUTHENTICATED_ACTOR, _token_actor

__all__ = [
    "PEER_BINDINGS_ENV",
    "ClaimVerdict",
    "ClaimCheck",
    "bound_agent_for_actor",
    "check_agent_claim",
]

#: ``agent_id:token[,agent_id:token...]``. The operator writes tokens because tokens are
#: what they have; only digests are ever stored or compared.
PEER_BINDINGS_ENV = "MIND_MEM_PEER_AGENTS"

_INVALID = "__invalid__"


class ClaimVerdict(Enum):
    """Closed verdict set. Only two arms allow the write, and both say why."""

    UNBOUND = "unbound"           # no table configured — legacy behaviour
    MATCH = "match"               # bound, and the claim matches
    MISMATCH = "mismatch"         # bound to a DIFFERENT agent — impersonation
    UNKNOWN_PEER = "unknown-peer"  # table configured, this credential is not in it
    TABLE_INVALID = "table-invalid"  # config error — fail CLOSED


@dataclass(frozen=True)
class ClaimCheck:
    """The verdict, whether it permits the write, and a token-free reason."""

    verdict: ClaimVerdict
    allowed: bool
    reason: str = ""


def _load_bindings() -> dict[str, str]:
    """Parse the binding table into ``{actor_key: agent_id}``.

    READ ON EVERY CALL, deliberately uncached, matching ``_active_tokens``' documented
    convention: it re-reads ``MIND_MEM_TOKENS`` per request so a rotation lands "without
    restart". A cached binding table would break that symmetry in the worse direction --
    an operator who rotates a peer's token would keep enforcing the OLD binding until
    someone restarted the server, and nothing would say so. Federation writes are not a
    hot path, so an env read plus a few digests per write is the right trade.

    (It was cached for exactly one revision. The cache also leaked across tests -- 8
    federation-wire tests failed on a binding table no longer in the environment -- which
    is the same staleness the production concern describes, surfacing early.)

    Returns ``{}`` when nothing is configured, and ``{_INVALID: reason}`` on any
    malformed entry -- never a partial table. A partially-parsed table is the fail-open:
    the peers that survived parsing stay bound, and the one whose line had a typo becomes
    unbound with nothing saying so.

    A token bound to two agents is also invalid: an ambiguous identity resolved either
    way invents authority the operator never granted.
    """
    raw = os.environ.get(PEER_BINDINGS_ENV, "").strip()
    if not raw:
        return {}

    table: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if entry.count(":") != 1:
            return {_INVALID: f"entry {entry.split(':')[0].strip()!r} is not 'agent_id:token'"}
        agent, token = (part.strip() for part in entry.split(":", 1))
        if not agent or not token:
            return {_INVALID: f"entry {entry.split(':')[0].strip()!r} has an empty agent_id or token"}
        key = _token_actor(token)
        if key in table and table[key] != agent:
            return {
                _INVALID: (
                    f"one token is bound to both {table[key]!r} and {agent!r}; an "
                    f"ambiguous identity resolved either way invents authority"
                )
            }
        table[key] = agent
    if not table:
        return {_INVALID: "the table parsed to no bindings at all"}
    return table


def bound_agent_for_actor(actor: str) -> str | None:
    """The agent this credential may write as, or ``None``.

    Compared with :func:`hmac.compare_digest` over the actor keys rather than a dict
    lookup on the presented value, so the comparison does not vary with how much of the
    key matched.
    """
    table = _load_bindings()
    if _INVALID in table:
        return None
    for key, agent in table.items():
        if hmac.compare_digest(key, actor):
            return agent
    return None


def check_agent_claim(actor: str, claimed_agent_id: str) -> ClaimCheck:
    """Judge a claimed ``agent_id`` against the credential that presented it."""
    table = _load_bindings()

    if _INVALID in table:
        return ClaimCheck(
            ClaimVerdict.TABLE_INVALID,
            False,
            f"{PEER_BINDINGS_ENV} is malformed ({table[_INVALID]}); refusing the write "
            f"rather than treating a config error as 'no bindings configured', which "
            f"would silently re-open agent impersonation",
        )

    if not table:
        return ClaimCheck(
            ClaimVerdict.UNBOUND,
            True,
            f"{PEER_BINDINGS_ENV} is not set, so identity binding is NOT enforced and "
            f"any valid token may write as any agent_id",
        )

    # A table exists, so the operator has declared that peers are known. An
    # unauthenticated door presents no credential and cannot satisfy a binding.
    if actor == HTTP_UNAUTHENTICATED_ACTOR:
        return ClaimCheck(
            ClaimVerdict.UNKNOWN_PEER,
            False,
            f"this door presented no credential ({HTTP_UNAUTHENTICATED_ACTOR}) and "
            f"{PEER_BINDINGS_ENV} is configured, so there is no binding it can satisfy",
        )

    bound = bound_agent_for_actor(actor)
    if bound is None:
        return ClaimCheck(
            ClaimVerdict.UNKNOWN_PEER,
            False,
            f"the presented credential is not in {PEER_BINDINGS_ENV}; once bindings are "
            f"configured an unlisted credential is an unknown peer, not a legacy one -- "
            f"otherwise adding a table would leave the hole open for every token not in it",
        )

    if hmac.compare_digest(bound, str(claimed_agent_id)):
        return ClaimCheck(ClaimVerdict.MATCH, True, f"credential is bound to {bound!r}")

    return ClaimCheck(
        ClaimVerdict.MISMATCH,
        False,
        f"the presented credential is bound to agent_id {bound!r} but the write claims "
        f"{str(claimed_agent_id)!r}",
    )
