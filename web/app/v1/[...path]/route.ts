import { timingSafeEqual } from "node:crypto";

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
// `MIND_MEM_CONSOLE_TOKEN` is REQUIRED from the caller, always. There is no loopback
// exemption — see `callerIsAllowed` for why the one this file used to have was unsound.
const CONSOLE_TOKEN = process.env.MIND_MEM_CONSOLE_TOKEN ?? "";

// A REQUIRED TOKEN, with NO loopback exemption — and the exemption is what was wrong.
//
// The first version of this check fell back to "serve loopback callers" by reading the
// `Host` header. `Host` is CLIENT-CONTROLLED: an adversarial-reproduced request carrying
// `Host: localhost` was granted the upstream token's full authority. There is no portable,
// trustworthy peer address in a Next.js route handler, so there is nothing to put in the
// exemption's place — which means the exemption has to go.
//
// With no token configured the route refuses EVERYTHING and says why. That is deliberately
// inconvenient: a console that silently proxies with privileged credentials is worse than a
// console that will not start until an operator names a secret.
function callerIsAllowed(req: Request): boolean {
  if (!CONSOLE_TOKEN) return false;
  const auth = req.headers.get("authorization") ?? "";
  const expected = `Bearer ${CONSOLE_TOKEN}`;
  // Constant-time: a `===` on a secret leaks the matching prefix length through timing.
  // Compare equal-length buffers only — the length itself is not the secret, so an early
  // length exit is fine, but the byte comparison must not short-circuit.
  const a = Buffer.from(auth);
  const b = Buffer.from(expected);
  if (a.length !== b.length) return false;
  return timingSafeEqual(a, b);
}

// Full decoding BEFORE validation, then a strict allowlist.
//
// The first version tested the segment for `.`/`..`/`/`/`\\` as handed over. That is not
// enough: a DOUBLE-ENCODED segment (`%252e%252e`) survives one decode as `%2e%2e` and then
// decodes again to `..`, which was reproduced as a working escape past the `/v1/` prefix.
// So decode repeatedly until the string stops changing (bounded, because a decode loop on
// attacker input is itself a denial-of-service surface), and only then check.
//
// The final check is an ALLOWLIST rather than a blocklist. A blocklist has to anticipate
// every encoding; an allowlist only has to describe what a legitimate path segment is.
function decodeFully(seg: string): string | null {
  let cur = seg;
  for (let i = 0; i < 4; i += 1) {
    let next: string;
    try {
      next = decodeURIComponent(cur);
    } catch {
      return null; // malformed escape: not a segment we should guess at
    }
    if (next === cur) return cur;
    cur = next;
  }
  return null; // still changing after 4 rounds — refuse rather than keep unwrapping
}

const SAFE_SEGMENT = /^[A-Za-z0-9._~-]+$/;

function pathIsSafe(path: string[]): boolean {
  return path.every((raw) => {
    const seg = decodeFully(raw);
    if (seg === null) return false;
    if (seg === "." || seg === "..") return false;
    return SAFE_SEGMENT.test(seg);
  });
}

async function proxy(req: Request, path: string[]): Promise<Response> {
  if (!callerIsAllowed(req)) {
    return Response.json({ error: "forbidden" }, { status: 403 });
  }
  if (!pathIsSafe(path)) {
    return Response.json({ error: "invalid path segment" }, { status: 400 });
  }
  const incoming = new URL(req.url);
  // Decoded segments, so the URL we build is the one we validated — forwarding the raw
  // form would let an encoding we accepted mean something different upstream.
  const safe = path.map((seg) => decodeFully(seg) as string);
  const target = `${API_ORIGIN}/v1/${safe.join("/")}${incoming.search}`;

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
  // PARSE the request and read its `format` field. The previous test was
  // `body.includes('"bundle"')` — a substring match over the RAW REQUEST BODY, so the
  // QUERY TEXT decided the response shape: `{"query":"bundle","format":"blocks"}`
  // returned a bundle despite asking for blocks. A caller's data must never select a
  // code path meant for its parameters.
  let wantsBundle = false;
  if (body) {
    try {
      wantsBundle = (JSON.parse(body) as { format?: unknown }).format === "bundle";
    } catch {
      wantsBundle = false; // unparseable body: do not guess at an intent
    }
  }
  if (upstream.ok && safe.join("/") === "recall" && wantsBundle) {
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
      // `score`, NOT `confidence`. A retrieval score is a ranking quantity (BM25/RRF,
      // unbounded, not on [0,1]), so projecting it into a field named `confidence`
      // invented a probability the system never computed -- and the console multiplied
      // it by 100, so a score of 17 displayed as "1700%". Passed through under its real
      // name so the UI shows what it actually is.
      score: r.score ?? 0,
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
