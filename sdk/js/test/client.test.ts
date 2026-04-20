import { describe, it, before, after, mock } from "node:test";
import assert from "node:assert/strict";
import { MindMemClient } from "../src/client.js";
import {
  MindMemAuthError,
  MindMemRateLimitError,
  MindMemServerError,
} from "../src/errors.js";
import type { RecallResult, HealthResult } from "../src/types.js";

// ---------------------------------------------------------------------------
// Minimal fetch mock helpers
// ---------------------------------------------------------------------------

type MockResponse = {
  status: number;
  body: unknown;
  headers?: Record<string, string>;
};

function makeFetch(response: MockResponse): typeof fetch {
  return async (_input: RequestInfo | URL, _init?: RequestInit): Promise<Response> => {
    const headersMap = new Map(Object.entries(response.headers ?? {}));
    return {
      ok: response.status >= 200 && response.status < 300,
      status: response.status,
      headers: {
        get: (name: string) => headersMap.get(name.toLowerCase()) ?? null,
      },
      json: async () => response.body,
    } as unknown as Response;
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("MindMemClient — construction", () => {
  it("normalises trailing slashes from baseUrl", () => {
    const c = new MindMemClient("http://localhost:8080///");
    assert.equal(c.baseUrl, "http://localhost:8080");
  });

  it("accepts a baseUrl without trailing slash unchanged", () => {
    const c = new MindMemClient("http://localhost:8080");
    assert.equal(c.baseUrl, "http://localhost:8080");
  });
});

describe("MindMemClient — URL composition", () => {
  it("composes /v1/recall with query params", async () => {
    let capturedUrl: string | undefined;

    const mockFetch: typeof fetch = async (input) => {
      capturedUrl = input.toString();
      return makeFetch({
        status: 200,
        body: { query: "test", results: [], total: 0, backend_used: "bm25", latency_ms: 1 },
      })(input);
    };

    const original = globalThis.fetch;
    globalThis.fetch = mockFetch;
    try {
      const client = new MindMemClient("http://localhost:8080");
      await client.recall("test query", { limit: 3, backend: "bm25" });
      assert.ok(capturedUrl?.includes("/v1/recall"), "path included");
      assert.ok(capturedUrl?.includes("limit=3"), "limit param");
      assert.ok(capturedUrl?.includes("backend=bm25"), "backend param");
    } finally {
      globalThis.fetch = original;
    }
  });

  it("URL-encodes block IDs in getBlock path", async () => {
    let capturedUrl: string | undefined;

    const mockFetch: typeof fetch = async (input) => {
      capturedUrl = input.toString();
      return makeFetch({ status: 200, body: { block: null } })(input);
    };

    const original = globalThis.fetch;
    globalThis.fetch = mockFetch;
    try {
      const client = new MindMemClient("http://localhost:8080");
      await client.getBlock("block/with spaces");
      assert.ok(capturedUrl?.includes("block%2Fwith%20spaces"), "block id encoded");
    } finally {
      globalThis.fetch = original;
    }
  });
});

describe("MindMemClient — auth header", () => {
  it("does NOT send Authorization when no token provided", async () => {
    let capturedHeaders: Record<string, string> = {};

    const mockFetch: typeof fetch = async (_input, init) => {
      capturedHeaders = Object.fromEntries(
        Object.entries((init?.headers as Record<string, string>) ?? {}),
      );
      return makeFetch({
        status: 200,
        body: { status: "ok", version: "3.2.0", block_count: 0, index_state: "ready", encryption_enabled: false, uptime_seconds: 1 } as HealthResult,
      })(_input, init);
    };

    const original = globalThis.fetch;
    globalThis.fetch = mockFetch;
    try {
      const client = new MindMemClient("http://localhost:8080");
      await client.health();
      assert.equal(capturedHeaders["Authorization"], undefined);
      assert.equal(capturedHeaders["X-MindMem-Token"], undefined);
    } finally {
      globalThis.fetch = original;
    }
  });

  it("sends Bearer token when token option provided", async () => {
    let capturedHeaders: Record<string, string> = {};

    const mockFetch: typeof fetch = async (_input, init) => {
      capturedHeaders = (init?.headers ?? {}) as Record<string, string>;
      return makeFetch({
        status: 200,
        body: { status: "ok", version: "3.2.0", block_count: 0, index_state: "ready", encryption_enabled: false, uptime_seconds: 1 },
      })(_input, init);
    };

    const original = globalThis.fetch;
    globalThis.fetch = mockFetch;
    try {
      const client = new MindMemClient("http://localhost:8080", { token: "secret-token" });
      await client.health();
      assert.equal(capturedHeaders["Authorization"], "Bearer secret-token");
      assert.equal(capturedHeaders["X-MindMem-Token"], "secret-token");
    } finally {
      globalThis.fetch = original;
    }
  });
});

describe("MindMemClient — typed results", () => {
  it("returns a typed RecallResult on 200", async () => {
    const envelope: RecallResult = {
      query: "postgres",
      results: [
        {
          block: {
            id: "b1",
            content: "Use pgvector",
            importance: 0.9,
            tier: "WORKING",
            created_at: "2026-01-01T00:00:00Z",
            updated_at: "2026-01-01T00:00:00Z",
            keywords: ["postgres"],
            category: "infra",
            namespace: null,
            active: true,
            access_count: 3,
            provenance: null,
          },
          score: 0.92,
          rank: 1,
        },
      ],
      total: 1,
      backend_used: "hybrid",
      latency_ms: 12,
    };

    const original = globalThis.fetch;
    globalThis.fetch = makeFetch({ status: 200, body: envelope });
    try {
      const client = new MindMemClient("http://localhost:8080");
      const result = await client.recall("postgres");
      assert.equal(result.total, 1);
      assert.equal(result.results[0]?.block.id, "b1");
      assert.equal(result.backend_used, "hybrid");
    } finally {
      globalThis.fetch = original;
    }
  });
});

describe("MindMemClient — error handling", () => {
  it("throws MindMemAuthError on 401", async () => {
    const original = globalThis.fetch;
    globalThis.fetch = makeFetch({ status: 401, body: { error: "Unauthorised" } });
    try {
      const client = new MindMemClient("http://localhost:8080");
      await assert.rejects(
        () => client.health(),
        (err: unknown) => {
          assert.ok(err instanceof MindMemAuthError, "is MindMemAuthError");
          assert.equal((err as MindMemAuthError).statusCode, 401);
          return true;
        },
      );
    } finally {
      globalThis.fetch = original;
    }
  });

  it("throws MindMemAuthError on 403", async () => {
    const original = globalThis.fetch;
    globalThis.fetch = makeFetch({ status: 403, body: { error: "Forbidden" } });
    try {
      const client = new MindMemClient("http://localhost:8080");
      await assert.rejects(
        () => client.health(),
        (err: unknown) => {
          assert.ok(err instanceof MindMemAuthError);
          assert.equal((err as MindMemAuthError).statusCode, 403);
          return true;
        },
      );
    } finally {
      globalThis.fetch = original;
    }
  });

  it("throws MindMemRateLimitError on 429 and parses Retry-After", async () => {
    const original = globalThis.fetch;
    globalThis.fetch = makeFetch({
      status: 429,
      body: { error: "Too Many Requests" },
      headers: { "retry-after": "60" },
    });
    try {
      const client = new MindMemClient("http://localhost:8080");
      await assert.rejects(
        () => client.recall("test"),
        (err: unknown) => {
          assert.ok(err instanceof MindMemRateLimitError);
          assert.equal((err as MindMemRateLimitError).retryAfterSeconds, 60);
          return true;
        },
      );
    } finally {
      globalThis.fetch = original;
    }
  });

  it("throws MindMemServerError on 500", async () => {
    const original = globalThis.fetch;
    globalThis.fetch = makeFetch({ status: 500, body: { error: "Internal Server Error" } });
    try {
      const client = new MindMemClient("http://localhost:8080");
      await assert.rejects(
        () => client.scan(),
        (err: unknown) => {
          assert.ok(err instanceof MindMemServerError);
          assert.equal((err as MindMemServerError).statusCode, 500);
          return true;
        },
      );
    } finally {
      globalThis.fetch = original;
    }
  });
});
