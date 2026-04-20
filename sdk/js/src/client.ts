import {
  MindMemAuthError,
  MindMemError,
  MindMemRateLimitError,
  MindMemServerError,
} from "./errors.js";
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
   */
  async recall(query: string, opts: RecallOptions = {}): Promise<RecallResult> {
    const params: Record<string, string> = { q: query };
    if (opts.limit !== undefined) params["limit"] = String(opts.limit);
    if (opts.activeOnly !== undefined) params["active_only"] = String(opts.activeOnly);
    if (opts.backend !== undefined) params["backend"] = opts.backend;
    return this.get<RecallResult>("/v1/recall", params);
  }

  /**
   * Fetch a single memory block by its ID.
   */
  async getBlock(blockId: string): Promise<BlockResult> {
    return this.get<BlockResult>(`/v1/blocks/${encodeURIComponent(blockId)}`);
  }

  /**
   * List all detected contradictions in the memory store.
   */
  async listContradictions(): Promise<ContradictionsResult> {
    return this.get<ContradictionsResult>("/v1/contradictions");
  }

  /**
   * Check the health / readiness of the running mind-mem instance.
   */
  async health(): Promise<HealthResult> {
    return this.get<HealthResult>("/v1/health");
  }

  /**
   * Run a governance scan and return any drift/conflict issues found.
   */
  async scan(): Promise<ScanResult> {
    return this.get<ScanResult>("/v1/scan");
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

  private async get<T>(path: string, params?: Record<string, string>): Promise<T> {
    const url = this.buildUrl(path, params);
    const signal = AbortSignal.timeout(this.timeoutMs);

    let response: Response;
    try {
      response = await fetch(url, {
        method: "GET",
        headers: this.buildHeaders(),
        signal,
      });
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
