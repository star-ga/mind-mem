# @mind-mem/sdk

JavaScript / TypeScript client for the [MIND-Mem](https://github.com/star-ga/mind-mem) REST API.

## Requirements

- Node.js 18+ (native `fetch`, `AbortSignal.timeout`)
- Or any modern browser

## Tests

```bash
npm test    # compiles src + test with tsconfig.test.json, then runs node --test
```

## Install

> **Not published yet.** The package name is an open decision — the manifest
> reads `@mind-mem/sdk`, the roadmap names `@star-ga/mind-mem-client`, and an
> npm name is not reclaimable once taken. Until it is settled,
> `sdk/js/package.json` carries `"private": true` so no accidental
> `npm publish` can claim either name. Build the publishable tarball with
> `python3 sdk/release/pack_js.py --stage <dir>`, which stamps the version
> from `pyproject.toml` rather than trusting the manifest.

For now, consume it from a checkout:

```bash
cd sdk/js && npm install && npm run build
```

## Quick start

```typescript
import { MindMemClient } from '@mind-mem/sdk';

const client = new MindMemClient('http://localhost:8080', {
  token: process.env.MIND_MEM_TOKEN,
});

const results = await client.recall('what did we decide about Postgres?', { limit: 5 });
console.log(results.results.map(r => r.block.content));
```

## API

### `new MindMemClient(baseUrl, options?)`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `token` | `string` | — | Bearer token. Sent as `Authorization: Bearer` and `X-MindMem-Token`. |
| `timeoutMs` | `number` | `30000` | Per-request abort timeout. |

### Methods (read-only surface)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `recall(query, opts?)` | `POST /v1/recall` | BM25/vector/hybrid recall against stored blocks. |
| `getBlock(blockId)` | `GET /v1/block/{block_id}` | Fetch a single block by ID. |
| `listContradictions()` | `GET /v1/contradictions` | List governance-detected contradictions. |
| `health()` | `GET /v1/health` | Check server health and version. |
| `scan()` | `GET /v1/scan` | Trigger a governance scan and return issues. |

The endpoint each method calls is declared in `src/routes.ts` and checked
against `sdk/spec/openapi.json` by `tests/test_sdk_route_conformance.py`, so a
client that calls something the server does not serve fails in CI rather than
at a user's first request.

### `RecallOptions`

```typescript
interface RecallOptions {
  limit?: number;                      // default: server-side default (10)
  activeOnly?: boolean;                // filter to active blocks only
  backend?: 'auto' | 'bm25' | 'hybrid';
}
```

## Errors

All errors extend `MindMemError` and carry `.statusCode` and `.responseBody`.

| Class | Status | Extra field |
|-------|--------|-------------|
| `MindMemAuthError` | 401 / 403 | — |
| `MindMemRateLimitError` | 429 | `.retryAfterSeconds` |
| `MindMemServerError` | 5xx | — |

```typescript
import { MindMemRateLimitError } from '@mind-mem/sdk';

try {
  const result = await client.recall('postgres decisions');
} catch (err) {
  if (err instanceof MindMemRateLimitError && err.retryAfterSeconds !== null) {
    await new Promise(r => setTimeout(r, err.retryAfterSeconds * 1000));
  }
  throw err;
}
```

## Write operations

Not covered by this client yet. The server *does* serve them —
`POST /v1/propose_update`, `POST /v1/approve_apply`,
`POST /v1/rollback_proposal` and the `/v1/admin/api_keys` surface are all in
`sdk/spec/openapi.json`, and all of them require an admin-scope token. Adding
one means a new entry in `src/routes.ts` plus its method; the conformance gate
then holds it to the spec like the rest.

## License

Apache-2.0 — STARGA Inc.
