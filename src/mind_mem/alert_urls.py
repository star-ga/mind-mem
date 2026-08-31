# Copyright 2026 STARGA, Inc.
"""Destination constraints for outbound alert URLs (T-004).

``alerting.py`` POSTs governance signals -- contradiction payloads,
block IDs, the absolute workspace path -- to operator-configured
webhooks. Those URLs come from ``mind-mem.json``, which is not reliably
operator-authored: workspaces get cloned, shared, and committed. An
attacker-influenced config previously turned every governance alert into
an arbitrary outbound request from the mind-mem host, which is both an
exfiltration channel and an SSRF primitive against endpoints only
reachable from that host.

Two layers:

1. **Default-on, zero-config.** SSRF-shaped destinations are refused:
   loopback, link-local (169.254.0.0/16 -- cloud metadata), RFC1918,
   CGNAT, multicast, reserved, unspecified. No legitimate SaaS webhook
   lives there, so this costs a correctly-configured operator nothing.
2. **Opt-in strict allowlist.** ``MIND_MEM_ALERT_URL_ALLOWLIST`` pins
   delivery to named hosts. When set it is authoritative: a publicly
   routable host that is not on it is refused.

``MIND_MEM_ALERT_ALLOW_ANY=true`` restores the pre-T-004 open behaviour
for the genuine case of a self-hosted receiver on a private network.
Env-only, matching ``MIND_MEM_VAULT_ALLOWLIST`` (T-006) and the
env-only token rule (T-005) -- a workspace file must not be able to
widen its own permissions.

Stdlib only; no new dependency.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlsplit

__all__ = [
    "AlertUrlError",
    "alert_allow_any",
    "alert_url_allowlist",
    "assert_destination_allowed",
    "validate_alert_url",
]


class AlertUrlError(ValueError):
    """An alert destination URL is malformed or not permitted.

    Subclasses :class:`ValueError` so existing sink constructors, which
    already document ``ValueError`` for a bad scheme, keep their
    contract.
    """


_ALLOWED_SCHEMES = ("http", "https")


def alert_url_allowlist() -> list[str]:
    """Configured host allowlist, lowercased. Empty means "not set"."""
    raw = os.environ.get("MIND_MEM_ALERT_URL_ALLOWLIST", "").strip()
    if not raw:
        return []
    sep = ";" if ";" in raw else ","
    return [h.strip().lower().rstrip(".") for h in raw.split(sep) if h.strip()]


def alert_allow_any() -> bool:
    """True when the operator has opted out of destination constraints."""
    return os.environ.get("MIND_MEM_ALERT_ALLOW_ANY", "").strip().lower() in ("1", "true", "yes")


def validate_alert_url(url: str) -> str:
    """Validate the *shape* of an alert URL. Returns it unchanged.

    Shape only -- no name resolution. Raises :class:`AlertUrlError`.
    """
    if not isinstance(url, str) or not url.strip():
        raise AlertUrlError("alert URL must be a non-empty string")
    url = url.strip()
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise AlertUrlError(f"unparseable alert URL: {exc}") from exc
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        raise AlertUrlError(f"alert URL scheme must be http or https, got {parts.scheme!r}")
    if not parts.hostname:
        raise AlertUrlError("alert URL has no host")
    if parts.username or parts.password:
        # A webhook receiver never needs inline credentials; their
        # presence is a redirect/exfil smell.
        raise AlertUrlError("alert URL must not embed credentials")
    return url


def _host_on_allowlist(host: str, allow: list[str]) -> bool:
    """Exact host match, or a true subdomain of an allowlist entry.

    Compares label-wise: ``notexample.com`` must NOT match an entry of
    ``example.com`` the way a bare ``str.endswith`` would let it.
    """
    host = host.lower().rstrip(".")
    return any(host == entry or host.endswith("." + entry) for entry in allow)


def _blocked_reason(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    """Name the range *ip* falls in, or "" when it is routable."""
    # ::ffff:169.254.169.254 must be judged as the IPv4 address it maps
    # to, not as an opaque IPv6 address.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.is_unspecified:
        return "unspecified"
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local (cloud metadata range)"
    if ip.is_multicast:
        return "multicast"
    if ip.is_reserved:
        return "reserved"
    if ip.is_private:
        return "private"
    # CGNAT 100.64.0.0/10 is NOT is_private on every supported CPython,
    # so it needs its own check rather than riding on is_private.
    if isinstance(ip, ipaddress.IPv4Address) and ip in ipaddress.ip_network("100.64.0.0/10"):
        return "carrier-grade NAT"
    return ""


def _resolve(host: str, port: int) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Every address *host* resolves to. Empty when it does not resolve.

    A literal address resolves to itself without touching DNS.
    """
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (OSError, UnicodeError):
        # Unresolvable here means unresolvable for urllib too, so the
        # request fails on its own -- this is not a bypass.
        return []
    out: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        try:
            out.append(ipaddress.ip_address(info[4][0]))
        except ValueError:  # pragma: no cover -- getaddrinfo returned a non-address
            continue
    return out


def assert_destination_allowed(url: str) -> None:
    """Raise :class:`AlertUrlError` unless *url* may be delivered to.

    Call immediately before sending, not only at construction: a name
    that resolved publicly when a sink was built can resolve into an
    internal range later.

    deferred: a resolve-then-connect gap remains -- urllib resolves the
    name again, so a sub-TTL DNS rebind between the two lookups is not
    covered. Closing it means resolving once and connecting to the
    pinned address, which requires a custom opener/connection class per
    sink -- upgrade path: replace urlopen with an opener whose
    HTTPConnection is constructed against the vetted IP with the
    original Host header preserved for SNI/vhost routing.
    """
    validate_alert_url(url)
    if alert_allow_any():
        return
    parts = urlsplit(url)
    host = (parts.hostname or "").lower().rstrip(".")

    allow = alert_url_allowlist()
    if allow and not _host_on_allowlist(host, allow):
        raise AlertUrlError(f"alert host {host!r} is not in MIND_MEM_ALERT_URL_ALLOWLIST")

    try:
        port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise AlertUrlError(f"alert URL has an invalid port: {exc}") from exc

    # The range check runs even for an allowlisted name: an allowlisted
    # NAME must not be able to smuggle in an internal ADDRESS.
    for ip in _resolve(host, port):
        reason = _blocked_reason(ip)
        if reason:
            raise AlertUrlError(
                f"alert host {host!r} resolves to {ip} ({reason}); "
                "refused as an SSRF-shaped destination. Set "
                "MIND_MEM_ALERT_ALLOW_ANY=true to deliver to internal "
                "addresses anyway."
            )
