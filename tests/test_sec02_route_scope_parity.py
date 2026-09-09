# Copyright 2026 STARGA, Inc.
"""SEC02 — every http-serve route declares a scope, mirrored to the MCP ACL.

MEASURED at 97fd765 before this change:

  * ``Route`` fields are method, path, handler, takes, verdict, mutates,
    empty_tail_error -- and NO scope. ``verdict`` classifies what an endpoint
    SERVES, which is a different question from WHO MAY CALL IT.
  * ``_authorized()`` returns a bare bool: a token either matches an active
    token or it does not. There is no admin/user axis anywhere in the
    transport.

So one flat bearer authorises every route, including ``POST /clear``, which
deletes every block ``store.get_all(active_only=False)`` returns. An agent
holding the USER-tier credential -- the one the MCP door correctly refuses
``delete_memory_item`` to -- can wipe the corpus through this door instead, and
the evidence chain records it as an authorised deletion.

THE AUTHORITY IS THE MCP ACL, not a second quieter classification invented
here. ``mcp/infra/acl.py`` already splits 26 ADMIN_TOOLS from 76 USER_TOOLS, so
each route declares the capability it mirrors and the parity is asserted rather
than assumed. Drift becomes a build failure, exactly as the existing content
sweep does for ``verdict``.

DIRECTION OF THE FIX: parity means this door denies what the MCP door already
denies. Never the reverse. No route may become MORE permissive here.
"""

from __future__ import annotations

import pytest

from mind_mem.http_transport import ROUTES
from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS

ADMIN = "admin"
USER = "user"


def test_every_route_declares_a_scope():
    """No default, for the same reason ``verdict`` has none: a new endpoint
    must not be routable until someone decides who may call it."""
    missing = [r.name for r in ROUTES if getattr(r, "scope", None) not in (ADMIN, USER)]
    assert not missing, f"routes with no declared scope: {missing}"


def test_scope_has_no_default_on_the_dataclass():
    import dataclasses

    from mind_mem.http_transport import Route

    field = {f.name: f for f in dataclasses.fields(Route)}["scope"]
    assert field.default is dataclasses.MISSING, "scope must not be defaultable"
    assert field.default_factory is dataclasses.MISSING


def test_no_mutating_route_is_user_scope():
    """The SEC02 defect in one assertion."""
    leaks = [r.name for r in ROUTES if r.mutates and r.scope != ADMIN]
    assert not leaks, f"mutating routes reachable at user scope: {leaks}"


def test_parity_is_enforced_at_IMPORT_not_only_asserted_here():
    """Root: do not claim import-time parity unless the code performs it.

    It does now -- Route.__post_init__ consults the MCP ACL -- so this control
    verifies the CONSTRUCTOR refuses, rather than re-walking the table and
    calling that a parity check.
    """
    from mind_mem.http_transport import NO_CONTENT, Route, _handle_status

    with pytest.raises(ValueError, match="MCP twin"):
        Route("GET", "/x", _handle_status, "workspace", NO_CONTENT, mutates=False, scope="user", mirrors="delete_memory_item")


def test_declared_scope_matches_the_mcp_authority():
    """Where a route names an MCP twin, the two must agree."""
    for r in ROUTES:
        twin = getattr(r, "mirrors", None)
        if not twin:
            continue
        assert twin in ADMIN_TOOLS or twin in USER_TOOLS, f"{r.name} mirrors {twin!r}, which is in neither ACL set"
        expected = ADMIN if twin in ADMIN_TOOLS else USER
        assert r.scope == expected, f"{r.name} declares {r.scope} but its MCP twin {twin} is {expected}"


def test_a_route_without_an_mcp_twin_still_declares_why():
    """A route with no MCP counterpart is the easy place to hide a permission
    hole, so it must carry a written justification rather than defaulting."""
    for r in ROUTES:
        if getattr(r, "mirrors", None):
            continue
        why = getattr(r, "scope_reason", "")
        assert isinstance(why, str) and len(why.strip()) >= 20, f"{r.name} has no MCP twin and no scope_reason explaining its scope"


def test_clear_and_delete_are_admin():
    """The two routes the finding names, pinned by name so a future edit that
    relaxes either fails here rather than silently."""
    by_path = {r.path: r for r in ROUTES}
    from mind_mem.http_transport import PATH_CLEAR

    assert by_path[PATH_CLEAR].scope == ADMIN
    deletes = [r for r in ROUTES if r.method == "DELETE"]
    assert deletes, "positive control: there is a DELETE route to check"
    assert all(r.scope == ADMIN for r in deletes)


def test_the_dispatcher_consults_the_scope():
    """`declared` is not `enforced`: a scope field nothing reads is decoration.

    This is the failure the RA.1 lane already shipped once -- a marker with no
    reader -- so it is asserted here rather than assumed.
    """
    import inspect

    from mind_mem import http_transport

    src = inspect.getsource(http_transport)
    assert "_caller_is_admin(" in src, "no admin predicate is called anywhere"
    assert src.count("route.scope") >= 1 or src.count(".scope ==") >= 1, "the dispatcher never reads route.scope"


def test_admin_scope_fails_closed_when_admin_identity_is_unavailable():
    """If the transport cannot establish admin identity it must DENY an admin
    route, never fall through to allow. Direction matters: stricter is a fix,
    looser is the vulnerability."""
    from mind_mem.http_transport import _caller_is_admin

    assert _caller_is_admin(presented=None, active_admin=[]) is False
    assert _caller_is_admin(presented="anything", active_admin=[]) is False
    assert _caller_is_admin(presented="user-token", active_admin=["admin-token"]) is False
    assert _caller_is_admin(presented="admin-token", active_admin=["admin-token"]) is True


# ---------------------------------------------------------------------------
# The compatibility rule, learned from a regression run rather than designed.
#
# Enforcing admin scope UNCONDITIONALLY turned 26 existing tests red, because
# every single-token deployment lost its admin routes. That is a public
# compatibility break to fix a hole those deployments do not have: with one
# credential there is no privilege separation to breach -- the operator issued
# one token with full access and got exactly that.
#
# The defect is narrower and sharper: an operator who DOES configure
# MIND_MEM_ADMIN_TOKEN, and hands out the user-tier token, finds this transport
# IGNORING the separation they set up. So enforcement follows the operator's
# own configuration.
# ---------------------------------------------------------------------------


def test_enforcement_follows_the_operator_configuration():
    import inspect

    from mind_mem import http_transport

    src = inspect.getsource(http_transport)
    assert "_capture_auth_snapshot(" in src
    assert 'route.scope == "admin" and auth_snapshot.admin_configured and not _caller_is_admin(' in src, (
        "admin enforcement must be conditional on an admin credential existing, "
        "or every single-token deployment loses its admin routes on upgrade"
    )


def test_the_admin_reader_is_separate_from_the_authentication_path():
    """Root owns _active_tokens. Authorisation must not reach into it."""
    import inspect

    from mind_mem.http_transport import _active_admin_tokens, _active_tokens

    admin_src = inspect.getsource(_active_admin_tokens)
    assert "MIND_MEM_ADMIN_TOKEN" in admin_src
    # Assert on a CALL, not a bare name. This exact assertion has now caught my
    # own explanatory prose four times this session (Stage 2.65,
    # attach_served_run, pipeline_hash, and here), so the rule is: a
    # source-inspection control must target syntax that only appears in CODE --
    # a call with its paren, or a keyword with its equals -- never an
    # identifier that a comment may legitimately mention.
    assert "_active_tokens(" not in admin_src, "the admin reader must not call the auth path"
    assert 'environ.get("MIND_MEM_ADMIN_TOKEN"' not in inspect.getsource(_active_tokens), (
        "authentication must not have grown an admin axis; that is a separate owner"
    )


def test_denial_does_not_disclose_which_admin_routes_exist():
    import inspect

    from mind_mem import http_transport

    src = inspect.getsource(http_transport)
    i = src.index('route.scope == "admin" and auth_snapshot.admin_configured')
    window = src[i : i + 900]
    assert "404" in window and "403" not in window, "an admin-route denial should read like an unmatched path, not confirm the route exists"
