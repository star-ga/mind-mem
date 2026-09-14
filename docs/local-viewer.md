# Local memory viewer

The next release candidate adds a packaged local browser interface:

```sh
MIND_MEM_WORKSPACE=/path/to/workspace mm view
```

Open the loopback URL printed by the command. Search admitted memory, inspect
its source/status fields, and query existing graph neighborhoods. The server
runs in the foreground and stops with Ctrl-C. Assets are packaged with MIND-Mem;
there are no remote scripts, telemetry calls or additional dependencies.

`--host` accepts `127.0.0.1`, `localhost` or `::1`; `--port` selects the port
(default 8765, or 0 to allocate one). Remote binding is refused. This local
operator interface does not add a remote authentication service. Host and
Origin checks restrict browser requests to the bound loopback authority.

`--agent-id NAME` applies the configured namespace admission rules to memory
reads. Graph viewing with an agent principal returns an explicit unsupported
response because graph edges do not yet carry namespace-bound source identity.
The unscoped local operator can inspect the existing graph.

The memory APIs are `GET /api/blocks?limit=N`,
`GET /api/search?q=TEXT&limit=N`, and `GET /api/block/ID`.
`GET /api/graph?entity=NAME&depth=N` returns the graph neighborhood as JSON.
Lists return at most 50 blocks; search operates over the first 10,000 admitted
blocks in source order. IDs outside that window are unavailable through this
viewer. The shared admission reader still enumerates the configured corpus
before limiting that window, so this limit does not cap backend I/O or memory.
Use the established recall/direct-read APIs when a larger corpus requires it.

The viewer offers no mutation routes. Graph reads use SQLite read-only mode and
never initialize/migrate a graph or create an unknown entity. A missing graph
is reported unavailable; an existing incomplete or corrupt graph returns a
bounded JSON 503 response. SQLite may update shared-memory coordination for an
existing live WAL database; this is separate from logical graph or schema
writes. Graph metadata and local search scores are diagnostics, without a
ranked recall attestation or an independent evidence claim.

Corpus strings render as text, with a restrictive Content Security Policy.
Absolute URL request targets and path traversal are refused. Common non-GET
methods return guarded refusal responses; HEAD sends headers without a body.
This candidate is implemented and reviewed separately from the pending 5.0.3
release source. Publication is still pending.
