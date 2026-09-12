# MIND-Mem web console

Thin Next.js client for the MIND-Mem REST API (v3.2.0+). Shows a
force-directed graph of blocks + their cross-references, a
chronological timeline of dated events, and a facts panel — all
derived from a single `recall(format="bundle")` call.

## Quick start

```bash
cd web
pnpm install   # or npm install
pnpm dev       # or npm run dev  → http://localhost:3000
```

Set the API URL if MIND-Mem isn't on localhost:8080:

```bash
NEXT_PUBLIC_MIND_MEM_API_URL=http://mind-mem.internal:8080 pnpm dev
```

## Architecture

- `app/page.tsx` — single-page console, submit a query, render the
  three panels.
- `components/GraphView.tsx` — d3-force simulation over the block
  graph. Nodes are coloured by `Status`; edges by predicate
  (`supersedes`/`depends_on`/`cites`/etc.).
- `components/TimelineView.tsx` — ordered dated events.
- `components/FactList.tsx` — extracted claims with confidence.
- `lib/api.ts` — typed client for `/v1/recall` (blocks + bundle
  formats) and `/v1/health`.

No state management library — React hooks are enough for the current
scope. When drift heatmap and contradiction graph views land they can
share a bundle via context or a fetcher like TanStack Query.

## Alternative to Obsidian

MIND-Mem v3.2.0 emits `[[wikilinks]]` on `vault_sync` so an Obsidian-
mounted vault gets graph + backlinks for free. This web app exists
for non-Obsidian deployments (headless servers, compliance-only
viewers, multi-tenant consoles in v4.0).

## Status

v3.3.0 scaffold. Landed in commits for the v3.3.0 "Other" section
of the MIND-Mem roadmap. Ship order for follow-up PRs:

1. Drift heatmap (reads `/v1/contradictions` + `/v1/scan`).
2. Per-tenant console (v4.0 — consumes `tenant_audit` summaries).
3. Evidence-bundle export to CSV / JSONL.
4. Keyboard shortcuts + search-within-bundle filter.

## Required: `MIND_MEM_CONSOLE_TOKEN`

The console will not proxy a single request until this is set. That is deliberate.

`app/v1/[...path]/route.ts` holds a bearer token for the mind-mem API and forwards with its
authority, which makes it a privilege boundary. An earlier version granted access to
"loopback callers", determined by reading the `Host` header — and `Host` is client-controlled,
so a request carrying `Host: localhost` received the token's full authority. That was
reproduced, not theorised.

There is no trustworthy peer address available to a Next.js route handler, so there is nothing
to put in the exemption's place. The exemption is therefore gone, and the route refuses
everything when no token is configured:

```bash
export MIND_MEM_CONSOLE_TOKEN="$(openssl rand -hex 32)"   # the console's own secret
export MIND_MEM_TOKEN="<the mind-mem API token>"          # forwarded upstream
export MIND_MEM_API_ORIGIN="http://127.0.0.1:8080"        # matches `mm serve --port`
npm run dev
```

Browser requests go through the proxy because `NEXT_PUBLIC_MIND_MEM_API_URL` is unset by
default, which makes them same-origin. Setting it to the API directly bypasses the proxy and
ships no token — fine for a bare loopback API with authentication off, wrong for anything else.

Path segments are decoded fully and then checked against an allowlist before being forwarded.
A single, double or triple-encoded `..` is rejected: `%252e%252e` survives one decode as
`%2e%2e` and then becomes `..`, which was a working escape past the `/v1/` prefix.
