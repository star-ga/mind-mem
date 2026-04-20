# mind-mem web console

Thin Next.js client for the mind-mem REST API (v3.2.0+). Shows a
force-directed graph of blocks + their cross-references, a
chronological timeline of dated events, and a facts panel — all
derived from a single `recall(format="bundle")` call.

## Quick start

```bash
cd web
pnpm install   # or npm install
pnpm dev       # or npm run dev  → http://localhost:3000
```

Set the API URL if mind-mem isn't on localhost:8080:

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

mind-mem v3.2.0 emits `[[wikilinks]]` on `vault_sync` so an Obsidian-
mounted vault gets graph + backlinks for free. This web app exists
for non-Obsidian deployments (headless servers, compliance-only
viewers, multi-tenant consoles in v4.0).

## Status

v3.3.0 scaffold. Landed in commits for the v3.3.0 "Other" section
of the mind-mem roadmap. Ship order for follow-up PRs:

1. Drift heatmap (reads `/v1/contradictions` + `/v1/scan`).
2. Per-tenant console (v4.0 — consumes `tenant_audit` summaries).
3. Evidence-bundle export to CSV / JSONL.
4. Keyboard shortcuts + search-within-bundle filter.
