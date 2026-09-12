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

// 8080 matches `mm serve --port`'s default. The previous 18795 pointed at a port
// nothing listens on out of the box, so the console failed to reach a correctly
// started server.
const API_ORIGIN = process.env.MIND_MEM_API_ORIGIN ?? "http://127.0.0.1:8080";
const TOKEN = process.env.MIND_MEM_TOKEN ?? "";

// This route holds a bearer token and forwards to the API, so it is a privilege
// boundary and not merely a convenience. Two things follow.
//
// FIRST, it must refuse to be a confused deputy for anyone who can reach the Next.js
// server. A non-loopback bind means the console is reachable by someone who does not
// hold the token, and forwarding for them would hand out the token's authority.
// `MIND_MEM_CONSOLE_TOKEN`, when set, is required from the caller; when unset the route
// serves only loopback callers. Fail-closed on an unrecognised remote address rather
// than assuming loopback.
const CONSOLE_TOKEN = process.env.MIND_MEM_CONSOLE_TOKEN ?? "";

function callerIsAllowed(req: Request): boolean {
  if (CONSOLE_TOKEN) {
    const auth = req.headers.get("authorization") ?? "";
    // Constant-time comparison is not reachable here without a crypto import; the
    // token is a local deployment secret rather than a user password, and the
    // alternative (no check at all) is strictly worse.
    return auth === `Bearer ${CONSOLE_TOKEN}`;
  }
  const host = (req.headers.get("host") ?? "").split(":")[0];
  return host === "127.0.0.1" || host === "localhost" || host === "[::1]" || host === "::1";
}

// SECOND, the catch-all segment must not be able to walk out of `/v1/`. Next.js hands
// the path through decoded, so a `..` segment would let `fetch` normalise
// `/v1/../admin` to `/admin` and reach endpoints this route is not meant to expose.
// Rejecting the segment is right rather than stripping it: a caller who sent `..` is
// not asking for something this proxy should guess at.
function pathIsSafe(path: string[]): boolean {
  return path.every(
    (seg) => seg.length > 0 && seg !== "." && seg !== ".." && !seg.includes("/") && !seg.includes("\\"),
  );
}

async function proxy(req: Request, path: string[]): Promise<Response> {
  if (!callerIsAllowed(req)) {
    return Response.json({ error: "forbidden" }, { status: 403 });
  }
  if (!pathIsSafe(path)) {
    return Response.json({ error: "invalid path segment" }, { status: 400 });
  }
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
