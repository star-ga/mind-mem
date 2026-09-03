import {
  MindMemAuthError,
  MindMemError,
  MindMemRateLimitError,
  MindMemServerError,
} from "./errors.js";
import {
  expandRoute,
  ROUTE_GET_BLOCK,
  ROUTE_HEALTH,
  ROUTE_LIST_CONTRADICTIONS,
  ROUTE_RECALL,
  ROUTE_SCAN,
} from "./routes.js";
import type {
  BlockResult,
  ClientOptions,
  ContradictionsResult,
  HealthResult,
  RecallOptions,
  RecallResult,
  ScanResult,
} from "./types.js";

const DEFAULT_TIMEOUT_MS = 30_000;

/**
 * HTTP client for the mind-mem REST API.
 *
 * @example
 * ```ts
 * import { MindMemClient } from '@mind-mem/sdk';
 *
 * const client = new MindMemClient('http://localhost:8080', {
 *   token: process.env.MIND_MEM_TOKEN,
 * });
 *
 * const results = await client.recall('what did we decide about Postgres?', { limit: 5 });
 * ```
 */
export class MindMemClient {
  readonly baseUrl: string;
  private readonly token: string | undefined;
  private readonly timeoutMs: number;

  constructor(baseUrl: string, options: ClientOptions = {}) {
    // Normalise: strip trailing slash so path joining is consistent.
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.token = options.token;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Full-text + semantic recall against stored memory blocks.
   *
   * Sent as `POST /v1/recall` with a JSON body — the shape the server has
   * always served. Fields the caller left unset are omitted so the server's
   * own defaults apply (see components/schemas/RecallRequest in
   * `sdk/spec/openapi.json`).
   */
  async recall(query: string, opts: RecallOptions = {}): Promise<RecallResult> {
    const body: Record<string, unknown> = { query };
    if (opts.limit !== undefined) body["limit"] = opts.limit;
    if (opts.activeOnly !== undefined) body["active_only"] = opts.activeOnly;
    if (opts.backend !== undefined) body["backend"] = opts.backend;
    return this.request<RecallResult>(ROUTE_RECALL.method, expandRoute(ROUTE_RECALL), {
      body,
    });
  }

  /**
   * Fetch a single memory block by its ID.
   */
  async getBlock(blockId: string): Promise<BlockResult> {
    return this.request<BlockResult>(
      ROUTE_GET_BLOCK.method,
      expandRoute(ROUTE_GET_BLOCK, blockId),
    );
  }

  /**
   * List all detected contradictions in the memory store.
   */
  async listContradictions(): Promise<ContradictionsResult> {
    return this.request<ContradictionsResult>(
      ROUTE_LIST_CONTRADICTIONS.method,
      expandRoute(ROUTE_LIST_CONTRADICTIONS),
    );
  }

  /**
   * Check the health / readiness of the running mind-mem instance.
   */
  async health(): Promise<HealthResult> {
    return this.request<HealthResult>(ROUTE_HEALTH.method, expandRoute(ROUTE_HEALTH));
  }

  /**
   * Run a governance scan and return any drift/conflict issues found.
   */
  async scan(): Promise<ScanResult> {
    return this.request<ScanResult>(ROUTE_SCAN.method, expandRoute(ROUTE_SCAN));
  }

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  private buildUrl(path: string, params?: Record<string, string>): string {
    const url = new URL(this.baseUrl + path);
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        url.searchParams.set(k, v);
      }
    }
    return url.toString();
  }

  private buildHeaders(): HeadersInit {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: "application/json",
    };
    if (this.token) {
      headers["Authorization"] = `Bearer ${this.token}`;
      headers["X-MindMem-Token"] = this.token;
    }
    return headers;
  }

  private async request<T>(
    method: string,
    path: string,
    options: { params?: Record<string, string>; body?: unknown } = {},
  ): Promise<T> {
    const url = this.buildUrl(path, options.params);
    const signal = AbortSignal.timeout(this.timeoutMs);

    const init: RequestInit = {
      method,
      headers: this.buildHeaders(),
      signal,
    };
    if (options.body !== undefined) {
      init.body = JSON.stringify(options.body);
    }

    let response: Response;
    try {
      response = await fetch(url, init);
    } catch (cause) {
      throw new MindMemError(
        `Network error reaching ${url}: ${(cause as Error).message}`,
        0,
        null,
      );
    }

    if (response.ok) {
      return (await response.json()) as T;
    }

    await this.throwForStatus(response);
    // Unreachable — throwForStatus always throws; satisfies TS control-flow.
    throw new MindMemError("Unexpected error", response.status);
  }

  private async throwForStatus(response: Response): Promise<never> {
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      body = null;
    }

    const message =
      typeof body === "object" && body !== null && "error" in body
        ? String((body as Record<string, unknown>)["error"])
        : `HTTP ${response.status}`;

    if (response.status === 401 || response.status === 403) {
      throw new MindMemAuthError(message, response.status as 401 | 403, body);
    }

    if (response.status === 429) {
      const retryHeader = response.headers.get("Retry-After");
      const retryAfterSeconds = retryHeader !== null ? Number(retryHeader) : null;
      throw new MindMemRateLimitError(message, retryAfterSeconds, body);
    }

    if (response.status >= 500) {
      throw new MindMemServerError(message, response.status, body);
    }

    throw new MindMemError(message, response.status, body);
  }
}
