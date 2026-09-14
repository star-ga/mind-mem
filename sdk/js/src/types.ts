// Copyright 2026 STARGA, Inc.
// The REST server returns tool envelopes. Optional/new fields are retained;
// these types do not claim that an attached receipt was independently verified.
export type SearchBackend = "auto" | "bm25" | "hybrid";
export interface Envelope extends Record<string, unknown> {
  _schema_version: string;
}
export interface Block extends Record<string, unknown> {
  _id: string;
  Statement?: string;
  Status?: string;
  Date?: string;
  Type?: string;
}
export interface RecallItem extends Record<string, unknown> {
  _id: string;
  score: number;
  excerpt: string;
  type?: string;
  file?: string;
  line?: number;
  status?: string;
}
export interface RecallResult extends Envelope {
  query: string;
  query_id?: string;
  results: RecallItem[];
  count: number;
  backend: string;
  scoring_instant?: string;
  attestation?: Record<string, unknown> | null;
  warnings?: string[];
}
export interface BlockResult extends Envelope {
  block_id: string;
  found: true;
  block: Block;
}
export interface ContradictionsResult extends Envelope {
  status: string;
  contradictions: number;
  resolutions?: Record<string, unknown>[];
  message?: string;
}
export interface HealthResult extends Envelope {
  status: string;
  api_version: string;
  schema_version: string;
  workspace: string;
  workspace_exists: boolean;
}
export interface ScanResult extends Envelope {
  backend: string;
  checks: Record<string, unknown>;
}
export interface RecallOptions {
  limit?: number;
  activeOnly?: boolean;
  backend?: SearchBackend;
  /** UTC YYYY-MM-DD; omit to use the server's current scoring date. */
  scoringInstant?: string;
}
export interface ClientOptions {
  token?: string;
  timeoutMs?: number;
}
