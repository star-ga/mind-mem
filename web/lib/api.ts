/**
 * Thin client for the mind-mem REST API (v3.2.0+).
 *
 * Pointed at http://127.0.0.1:8080 by default; override via
 * NEXT_PUBLIC_MIND_MEM_API_URL or (on the server side) the
 * MIND_MEM_API_URL env var.
 *
 * The REST API is documented at
 * /home/n/mind-mem/docs/v3.2.0-release-notes.md and served by
 * src/mind_mem/api/rest.py.
 */

export type MindMemBlock = {
  _id: string;
  type?: string;
  Status?: string;
  Statement?: string;
  Date?: string;
  Created?: string;
  score?: number;
  truth_score?: number;
  _tier?: number;
  _graph_hop?: number;
  _prefetch?: string;
  [key: string]: unknown;
};

export type Relation = {
  subject: string;
  predicate: string;
  object: string;
};

export type TimelineEvent = {
  date: string;
  event: string;
  source_id: string;
};

export type EntityRef = {
  id: string;
  name: string;
  type: string;
};

export type EvidenceBundle = {
  query: string;
  facts: Array<{ claim: string; source_id: string; confidence: number }>;
  relations: Relation[];
  timeline: TimelineEvent[];
  entities: EntityRef[];
  source_blocks: MindMemBlock[];
};

const API_BASE_URL =
  (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_MIND_MEM_API_URL) ||
  "http://127.0.0.1:8080";

export async function recallBundle(
  query: string,
  opts: { limit?: number; signal?: AbortSignal } = {},
): Promise<EvidenceBundle> {
  const res = await fetch(`${API_BASE_URL}/v1/recall`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      limit: opts.limit ?? 10,
      format: "bundle",
    }),
    signal: opts.signal,
  });
  if (!res.ok) {
    throw new Error(`recallBundle failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as EvidenceBundle;
}

export async function recallBlocks(
  query: string,
  opts: { limit?: number; signal?: AbortSignal } = {},
): Promise<MindMemBlock[]> {
  const res = await fetch(`${API_BASE_URL}/v1/recall`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      limit: opts.limit ?? 10,
      format: "blocks",
    }),
    signal: opts.signal,
  });
  if (!res.ok) {
    throw new Error(`recallBlocks failed: ${res.status} ${res.statusText}`);
  }
  const payload = (await res.json()) as { results?: MindMemBlock[] };
  return payload.results ?? [];
}

export async function getHealth(): Promise<{ status: string; workspace: string }> {
  const res = await fetch(`${API_BASE_URL}/v1/health`);
  if (!res.ok) {
    throw new Error(`health failed: ${res.status}`);
  }
  return (await res.json()) as { status: string; workspace: string };
}
