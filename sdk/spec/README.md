# API specifications

## `openapi.json`

The OpenAPI 3.1 document for the REST API, exported from the live application
rather than written by hand. It is the contract both in-tree clients
(`sdk/go`, `sdk/js`) are checked against.

Regenerate after any change to the routes:

```bash
mind-mem-openapi --write     # rewrite the artifact
mind-mem-openapi --check     # exit 1 if it has drifted
```

(`python3 -m mind_mem.spec.export_openapi` is the same program; the console
script is the route the reachability gate can see.)

A committed spec is only worth having if something stops it disagreeing with
the server, so it ships with three gates in
`tests/test_sdk_openapi_drift.py`:

| Gate | What it catches |
|---|---|
| Structural diff against `create_app()` | any change to a path, verb, parameter, body, response, security scheme or component schema |
| Route census over `app.routes` | a route the schema generator drops — checked without going through `.openapi()`, so the generator cannot agree with itself |
| Version equality | an artifact advertising a release it was not exported from |

`tests/test_sdk_route_conformance.py` then joins this artifact to the two
clients' route tables (`sdk/go/routes.go`, `sdk/js/src/routes.ts`). That gate
exists because the clients had already drifted: both issued
`GET /v1/recall` with query parameters against a server that serves `POST`
with a JSON body, and `GET /v1/blocks/{id}` against a server that serves the
singular `/v1/block/{block_id}`. Each client's own suite passed, because each
was only ever compared with itself.

### Version and release

The artifact carries the package version in `info.version`, so **a version bump
must be followed by `--write`**. The structural comparison deliberately ignores
`info.version` — a release commit should not go red for a reason unrelated to
route drift — but `TestArtifactVersion` asserts it separately and names the
command in its failure message.

## `asyncapi.json`

The outbound event publisher has an opt-in cross-process transport: when
`events.enabled` and the Redis publisher are configured,
`mind_mem.event_fanout.RedisStreamPublisher` appends one JSON string in the
Redis Streams `data` field. The contract is deliberately limited to that
publisher boundary. It does not describe SSE, webhooks, a consumer service,
acknowledgements, retries, ordering, consumer groups, or at-least-once
delivery; publish failures are swallowed by the existing fail-open event seam.

The artifact records the eight canonical event names and the five literal
event kinds currently observed at product emit sites. Event kinds remain
extensible, and the payload is the existing `scrub_payload` output: bounded
IDs, hashes, enums, numbers and booleans may cross the boundary while prose is
dropped.

Regenerate or check it with the stdlib exporter:

```bash
mind-mem-asyncapi --write
mind-mem-asyncapi --check
```

(`python3 -m mind_mem.spec.export_asyncapi` is the equivalent module command.)
`tests/test_sdk_asyncapi_drift.py` compares the complete committed document
with the live source emitter inventory and validates a captured `XADD` record,
including duplicate-key, malformed-body and unsanitised-payload refusals.
