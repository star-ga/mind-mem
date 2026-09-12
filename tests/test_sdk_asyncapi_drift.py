"""The AsyncAPI spec must describe the events mind-mem ACTUALLY publishes.

ROADMAP ("OpenAPI + AsyncAPI specs"): "OpenAPI 3.1.0 ships at `sdk/spec/openapi.json`
(13 paths, version-gated by `tests/test_sdk_openapi_drift.py`); **AsyncAPI is still
unpublished, which is why this stays open.**"

A declarative spec is a published CLAIM about the wire. An AsyncAPI file listing channels
nothing emits would be worse than no file: a client would implement handlers for messages
that never arrive and have no way to discover that from the spec. So this mirrors the
OpenAPI drift gate — the spec is checked against the CODE, in both directions:

  * every canonical event kind in `event_fanout` appears in the spec, or the spec
    under-reports and a client silently misses messages;
  * every event the spec advertises exists in the code, or the spec over-reports and a
    client waits forever for something that is never sent.

Both directions matter and only one of them is the obvious one. The OpenAPI gate already
learned this: it asserts served routes are documented AND documented routes are served.

The two surfaces are kept apart on purpose. `event_fanout` carries governance events to a
publisher (logging / Redis stream); `alerting` delivers operator alerts to a webhook or
Slack. They have different payload envelopes, and merging them into one channel list would
publish a shape neither side sends.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = REPO_ROOT / "sdk" / "spec" / "asyncapi.json"


@pytest.fixture(scope="module")
def spec() -> dict:
    assert ARTIFACT.exists(), f"{ARTIFACT} is missing — the AsyncAPI half is unpublished"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_it_is_a_parseable_asyncapi_3_document(spec):
    assert spec.get("asyncapi", "").startswith("3."), spec.get("asyncapi")
    assert spec.get("info", {}).get("title")
    assert spec.get("channels"), "a spec with no channels documents nothing"


def test_every_canonical_governance_EVENT_is_documented(spec):
    """UNDER-REPORTING direction. A canonical kind absent from the spec means a client
    written from the spec silently misses those messages."""
    from mind_mem.event_fanout import _CANONICAL_EVENTS

    documented = set(_documented_event_names(spec))
    missing = sorted(set(_CANONICAL_EVENTS) - documented)
    assert not missing, f"canonical events absent from asyncapi.json: {missing}"


def test_every_documented_event_EXISTS_in_the_code(spec):
    """OVER-REPORTING direction, and the one a spec author gets wrong. An advertised
    event that nothing emits makes a client wait forever for a message never sent, and
    the spec gives them no way to find that out."""
    from mind_mem import alerting, event_fanout

    real = set(event_fanout._CANONICAL_EVENTS) | _alert_events()
    phantom = sorted(set(_documented_event_names(spec)) - real)
    assert not phantom, (
        f"asyncapi.json advertises events that nothing in src/ publishes: {phantom}"
    )
    assert alerting.Alert  # the alert envelope is part of the documented surface


def test_the_alert_events_that_are_actually_FIRED_are_documented(spec):
    """Only three alerts are fired anywhere in the tree. Documenting the machinery's
    capacity rather than its actual emissions would be the same over-report."""
    documented = set(_documented_event_names(spec))
    for event in _alert_events():
        assert event in documented, f"fired alert {event!r} is not in asyncapi.json"


def test_the_two_envelopes_are_kept_distinct(spec):
    """`event_fanout` sends {kind, payload, workspace, ...}; `alerting` sends
    {severity, event, payload, workspace, timestamp}. One merged message schema would
    publish a shape neither side actually sends."""
    schemas = spec.get("components", {}).get("schemas", {})
    assert "GovernanceEvent" in schemas and "OperatorAlert" in schemas, sorted(schemas)
    gov = set(schemas["GovernanceEvent"].get("properties", {}))
    alert = set(schemas["OperatorAlert"].get("properties", {}))
    assert "kind" in gov and "severity" not in gov, sorted(gov)
    assert "severity" in alert and "event" in alert, sorted(alert)


def test_the_envelopes_match_the_dataclasses_they_describe(spec):
    """Checked against the code, not against the prose above it. A spec that drifted
    from `Event`/`Alert` would mis-describe every message on the wire."""
    import dataclasses

    from mind_mem.alerting import Alert
    from mind_mem.event_fanout import Event

    schemas = spec["components"]["schemas"]
    gov_props = set(schemas["GovernanceEvent"]["properties"])
    alert_props = set(schemas["OperatorAlert"]["properties"])

    event_fields = {f.name for f in dataclasses.fields(Event)}
    alert_fields = {f.name for f in dataclasses.fields(Alert)}

    # The spec documents the WIRE, so internal timing fields are legitimately absent;
    # what must not happen is the spec claiming a field the dataclass does not have.
    assert gov_props <= event_fields, sorted(gov_props - event_fields)
    assert alert_props <= alert_fields, sorted(alert_props - alert_fields)
    # ...and the identifying field of each must be present, or the message is unusable.
    assert {"kind", "payload"} <= gov_props
    assert {"severity", "event", "payload"} <= alert_props


def test_the_version_matches_the_package(spec):
    """Same gate the OpenAPI spec has: a spec advertising a different version than the
    package is a spec someone will trust for the wrong release."""
    from mind_mem import __version__

    assert spec["info"]["version"] == __version__, (
        f"asyncapi.json advertises {spec['info']['version']!r} but the package is "
        f"{__version__!r}"
    )


# ---------------------------------------------------------------------------


def _documented_event_names(spec: dict) -> list[str]:
    """Event names the spec advertises, read from its channel message names."""
    names: list[str] = []
    for channel in (spec.get("channels") or {}).values():
        for msg_name in (channel.get("messages") or {}):
            names.append(msg_name)
    return names


def _alert_events() -> set[str]:
    """Alert event strings ACTUALLY fired in src/, read from the source.

    Derived by walking the tree rather than restated here: a hand-kept list would drift
    from the emitters, and this test exists precisely to catch drift.
    """
    import ast

    found: set[str] = set()
    for path in (REPO_ROOT / "src" / "mind_mem").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "attr", "")
            if name != "fire":
                continue
            for kw in node.keywords:
                if kw.arg == "event" and isinstance(kw.value, ast.Constant):
                    if isinstance(kw.value.value, str):
                        found.add(kw.value.value)
    return found


def test_the_alert_event_walker_finds_something():
    """POSITIVE CONTROL for the helper two tests depend on. A walker that returned an
    empty set would make those tests vacuously pass."""
    events = _alert_events()
    assert events, "the AST walk found no fired alerts, so the checks above prove nothing"
    assert "drift_spike" in events, sorted(events)
