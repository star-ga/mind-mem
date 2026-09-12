#!/usr/bin/env python3
"""Generate ``sdk/spec/asyncapi.json`` FROM THE CODE.

A declarative spec is a published claim about the wire, so this reads the event
taxonomy and the envelopes out of ``mind_mem`` rather than restating them. Restating
them is how a spec comes to advertise a channel nothing emits — and a client written
from such a spec waits forever for a message never sent, with no way to discover that.

Two surfaces, deliberately kept apart because they send different envelopes:

* ``event_fanout`` — governance events (``kind``/``payload``/``workspace``) to a
  configured publisher (logging, Redis stream).
* ``alerting`` — operator alerts (``severity``/``event``/``payload``/``workspace``/
  ``timestamp``) to a webhook or Slack.

Merging them into one channel would publish a shape neither side actually sends.

Only alerts that are genuinely FIRED somewhere in ``src/`` are documented. The alerting
module can carry any event string; documenting its capacity rather than its emissions
would be the same over-report in a different place.

LIVES INSIDE THE PACKAGE, not under ``sdk/``, and that placement is load-bearing:
``sdk`` must not depend on ``mind_mem`` (arch-mind rule ``NO_CROSS_PKG``), and this
generator necessarily imports the taxonomy it documents. The first version sat in
``sdk/spec/`` and the rules gate caught three forbidden edges. The ARTIFACT still lands
under ``sdk/spec/`` where a consumer looks for it -- same split ``export_openapi`` already
uses.

Run: ``python3 -m mind_mem.spec.export_asyncapi``.
Checked by ``tests/test_sdk_asyncapi_drift.py`` in BOTH directions.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import pathlib

from mind_mem import __version__
from mind_mem.alerting import Alert
from mind_mem.event_fanout import _CANONICAL_EVENTS, Event

# src/mind_mem/spec/ -> repo root is three parents up.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
#: The artifact lives under sdk/ where an SDK consumer looks for it, even though the
#: generator lives in the package -- the same split export_openapi uses.
OUT_PATH = REPO_ROOT / "sdk" / "spec" / "asyncapi.json"

#: Envelope fields that exist on the dataclass but are NOT part of the wire contract:
#: internal timing captured for ordering, never promised to a consumer. Named here so
#: their absence from the spec is a decision rather than an omission.
_INTERNAL_ONLY = {"ts_monotonic", "ts_wall"}


def _fired_alert_events() -> set[str]:
    """Alert event strings actually passed to ``AlertRouter.fire(event=...)``.

    Walks the AST for real call sites. A hand-kept list would drift from the emitters,
    which is exactly what the spec must not do.
    """
    found: set[str] = set()
    for path in (REPO_ROOT / "src" / "mind_mem").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "fire":
                for kw in node.keywords:
                    if kw.arg == "event" and isinstance(kw.value, ast.Constant):
                        if isinstance(kw.value.value, str):
                            found.add(kw.value.value)
    return found


def _schema_for(cls: type, *, required: tuple[str, ...]) -> dict:
    props: dict[str, dict] = {}
    for field in dataclasses.fields(cls):
        if field.name in _INTERNAL_ONLY:
            continue
        props[field.name] = {"type": "object" if field.name == "payload" else "string"}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(required),
        "properties": props,
    }


def build() -> dict:
    governance = sorted(_CANONICAL_EVENTS)
    alerts = sorted(_fired_alert_events())

    return {
        "asyncapi": "3.0.0",
        "info": {
            "title": "mind-mem event surface",
            "version": __version__,
            "description": (
                "Events mind-mem publishes. GENERATED from the source taxonomy by "
                "sdk/spec/gen_asyncapi.py and gated in both directions by "
                "tests/test_sdk_asyncapi_drift.py: every canonical event is documented, "
                "and every documented event exists in the code. An advertised event "
                "nothing emits would make a client wait forever for a message that is "
                "never sent."
            ),
        },
        "defaultContentType": "application/json",
        "channels": {
            "governanceEvents": {
                "address": "mind-mem/governance",
                "title": "Governance events",
                "description": (
                    "Emitted through event_fanout.emit_event to the configured "
                    "publisher (logging or Redis stream). Envelope is event_fanout.Event; "
                    "ts_monotonic/ts_wall are internal ordering fields and are not part "
                    "of the wire contract."
                ),
                "messages": {
                    name: {
                        "name": name,
                        "title": name.replace("_", " "),
                        "payload": {"$ref": "#/components/schemas/GovernanceEvent"},
                    }
                    for name in governance
                },
            },
            "operatorAlerts": {
                "address": "mind-mem/alerts",
                "title": "Operator alerts",
                "description": (
                    "Delivered by alerting.WebhookSink as an HTTP POST of Alert.as_dict(), "
                    "or by SlackSink in Slack's own attachment shape. Only alerts actually "
                    "fired in src/ are listed; the router accepts any event string, and "
                    "documenting that capacity rather than the real emissions would "
                    "over-report the surface."
                ),
                "messages": {
                    name: {
                        "name": name,
                        "title": name.replace("_", " "),
                        "payload": {"$ref": "#/components/schemas/OperatorAlert"},
                    }
                    for name in alerts
                },
            },
        },
        "operations": {
            "receiveGovernanceEvent": {
                "action": "receive",
                "channel": {"$ref": "#/channels/governanceEvents"},
            },
            "receiveOperatorAlert": {
                "action": "receive",
                "channel": {"$ref": "#/channels/operatorAlerts"},
            },
        },
        "components": {
            "schemas": {
                "GovernanceEvent": _schema_for(Event, required=("kind", "payload")),
                "OperatorAlert": _schema_for(
                    Alert, required=("severity", "event", "payload", "workspace")
                ),
            }
        },
    }


def main() -> int:
    out = OUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    spec = build()
    print(f"wrote {out}")
    for channel, body in spec["channels"].items():
        print(f"  {channel}: {len(body['messages'])} message(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
