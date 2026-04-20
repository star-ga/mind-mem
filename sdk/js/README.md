# @mind-mem/sdk

JavaScript / TypeScript client for the [mind-mem](https://github.com/star-ga/mind-mem) REST API.

## Requirements

- Node.js 18+ (native `fetch`, `AbortSignal.timeout`)
- Or any modern browser

## Install

```bash
npm install @mind-mem/sdk
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

### Methods (v0.1 — read-only surface)

| Method | Description |
|--------|-------------|
| `recall(query, opts?)` | BM25/vector/hybrid recall against stored blocks. |
| `getBlock(blockId)` | Fetch a single block by ID. |
| `listContradictions()` | List governance-detected contradictions. |
| `health()` | Check server health and version. |
| `scan()` | Trigger a governance scan and return issues. |

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

`propose_update` and `approve_apply` land in v0.2 once the REST API server ships.

## License

Apache-2.0 — STARGA Inc.
