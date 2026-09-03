# API specifications

## `openapi.json`

The OpenAPI 3.1 document for the REST API, exported from the live application
rather than written by hand. It is the contract both in-tree clients
(`sdk/go`, `sdk/js`) are checked against.

Regenerate after any change to the routes:

```bash
python3 -m mind_mem.spec.export_openapi --write     # rewrite the artifact
python3 -m mind_mem.spec.export_openapi --check     # exit 1 if it has drifted
```

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

## AsyncAPI — deferred, with the reason

`AsyncAPI` is not published, and not because it was skipped. There is no
network event surface to describe. `src/mind_mem/change_stream.py` is an
**in-process** pub/sub bus; its own header records that "the HTTP webhook
endpoint + cross-process bus remain deferred". Writing an AsyncAPI document
today would mean publishing a transport contract no server honours — the same
failure mode the drift gate above exists to prevent, with no gate available to
catch it.

**Trigger:** the first time a change-stream event crosses a process boundary
(webhook, SSE, WebSocket, or the cross-process bus), that transport gets an
AsyncAPI document and a drift check built the same way as this one.
