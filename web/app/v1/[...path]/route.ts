/**
 * Server-side proxy to the mind-mem REST API.
 *
 * The browser client (lib/api.ts) sends no credentials, but a non-loopback
 * mind-mem bind refuses unauthenticated requests. Proxying here keeps the
 * bearer token server-side instead of shipping it to the browser, and makes
 * the API same-origin so no CORS handling is needed.
 *
 * Env: MIND_MEM_API_ORIGIN (default http://127.0.0.1:18795), MIND_MEM_TOKEN.
 */

const API_ORIGIN = process.env.MIND_MEM_API_ORIGIN ?? "http://127.0.0.1:18795";
const TOKEN = process.env.MIND_MEM_TOKEN ?? "";

async function proxy(req: Request, path: string[]): Promise<Response> {
  const incoming = new URL(req.url);
  const target = `${API_ORIGIN}/v1/${path.join("/")}${incoming.search}`;

  const headers = new Headers();
  const contentType = req.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  if (TOKEN) headers.set("authorization", `Bearer ${TOKEN}`);

  const body = req.method === "GET" || req.method === "HEAD" ? undefined : await req.text();

  const upstream = await fetch(target, { method: req.method, headers, body });
  const payload = await upstream.text();

  // The REST layer ignores `format: "bundle"` and always answers with
  // {results, attestation}, but lib/api.ts expects an EvidenceBundle.
  // Project one here so the console renders instead of showing blanks.
  if (upstream.ok && path.join("/") === "recall" && body && body.includes('"bundle"')) {
    try {
      return Response.json(toBundle(JSON.parse(payload)));
    } catch {
      /* fall through to the raw payload */
    }
  }

  return new Response(payload, {
    status: upstream.status,
    headers: { "content-type": upstream.headers.get("content-type") ?? "application/json" },
  });
}

type RecallResult = {
  _id?: string;
  type?: string;
  score?: number;
  excerpt?: string;
  tags?: string;
  Date?: string;
};

function headline(excerpt: string): string {
  for (const line of excerpt.split("\n")) {
    const text = line.replace(/^#+\s*/, "").trim();
    if (text) return text;
  }
  return "";
}

function toBundle(raw: { query?: string; results?: RecallResult[] }) {
  const results = raw.results ?? [];
  return {
    query: raw.query ?? "",
    facts: results.map((r) => ({
      claim: headline(r.excerpt ?? ""),
      source_id: r._id ?? "",
      confidence: r.score ?? 0,
    })),
    relations: [],
    timeline: results
      .filter((r) => r.Date)
      .map((r) => ({
        date: r.Date as string,
        event: headline(r.excerpt ?? ""),
        source_id: r._id ?? "",
      })),
    entities: results.map((r) => ({
      id: r._id ?? "",
      name: headline(r.excerpt ?? ""),
      type: r.type ?? "unknown",
    })),
    source_blocks: results.map((r) => ({
      _id: r._id ?? "",
      type: r.type,
      Statement: headline(r.excerpt ?? ""),
      Date: r.Date,
      score: r.score,
    })),
  };
}

export async function GET(req: Request, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, (await ctx.params).path);
}

export async function POST(req: Request, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, (await ctx.params).path);
}

export async function DELETE(req: Request, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, (await ctx.params).path);
}
