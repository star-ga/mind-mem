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

## Who may call the proxy

`app/v1/[...path]/route.ts` holds a bearer token for the mind-mem API and forwards with its
authority, so it is a privilege boundary. The rules are:

- **Browser requests must be same-origin.** The route requires `Sec-Fetch-Site: same-origin`
  (or `none`, a direct navigation). Browsers set that header and page script cannot forge it,
  so it is trustworthy in a way the `Host` header is not — an earlier version granted access
  to "loopback callers" by reading `Host`, and a request carrying `Host: localhost` was
  reproduced receiving the token's full authority.
- **A cross-origin request is refused even with a valid token.** A token does not make a
  cross-origin request legitimate; accepting one would reopen the CSRF path this check exists
  to close. The real threat to a local console is a malicious page in the operator's browser
  reaching 127.0.0.1 — not a remote attacker, who cannot reach loopback at all.
- **Non-browser clients** (curl, scripts) send no `Sec-Fetch-Site`, so they must present
  `Authorization: Bearer $MIND_MEM_CONSOLE_TOKEN`.

**The primary control is the bind address, and this route cannot enforce it.** Bind Next.js to
loopback. If it must listen more widely, set `MIND_MEM_CONSOLE_TOKEN` and keep every
programmatic caller on it.

Do NOT put the API token in a `NEXT_PUBLIC_` variable — that ships it to the browser, which is
the one thing the proxy exists to prevent.

```bash
export MIND_MEM_TOKEN="<the mind-mem API token>"          # forwarded upstream
export MIND_MEM_API_ORIGIN="http://127.0.0.1:8080"        # matches `mm serve --port`
export MIND_MEM_CONSOLE_TOKEN="$(openssl rand -hex 32)"   # only for non-browser callers
npm run dev
```

Browser requests reach the proxy because `NEXT_PUBLIC_MIND_MEM_API_URL` is unset by default,
which makes them same-origin. Setting it to the API directly bypasses the proxy and ships no
token — fine for a bare loopback API with authentication off, wrong for anything else.

Path segments are decoded fully and then checked against an allowlist before being forwarded.
A single, double or triple-encoded `..` is rejected: `%252e%252e` survives one decode as
`%2e%2e` and then becomes `..`, which was a working escape past the `/v1/` prefix.
