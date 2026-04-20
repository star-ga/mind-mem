// ---------------------------------------------------------------------------
// Shared domain types
// ---------------------------------------------------------------------------

export type BlockTier = "WORKING" | "ARCHIVAL" | "COLD";
export type SearchBackend = "auto" | "bm25" | "hybrid";

export interface Block {
  id: string;
  content: string;
  importance: number;
  tier: BlockTier;
  created_at: string;
  updated_at: string;
  keywords: string[];
  category: string | null;
  namespace: string | null;
  active: boolean;
  access_count: number;
  provenance: string | null;
}

// ---------------------------------------------------------------------------
// Method-level result types
// ---------------------------------------------------------------------------

export interface RecallItem {
  block: Block;
  score: number;
  rank: number;
}

export interface RecallResult {
  query: string;
  results: RecallItem[];
  total: number;
  backend_used: string;
  latency_ms: number;
}

export interface BlockResult {
  block: Block;
}

export interface Contradiction {
  block_a_id: string;
  block_b_id: string;
  conflict_score: number;
  description: string;
  detected_at: string;
}

export interface ContradictionsResult {
  contradictions: Contradiction[];
  total: number;
}

export interface HealthResult {
  status: "ok" | "degraded" | "down";
  version: string;
  block_count: number;
  index_state: string;
  encryption_enabled: boolean;
  uptime_seconds: number;
}

export interface ScanIssue {
  type: string;
  severity: "low" | "medium" | "high" | "critical";
  block_id: string | null;
  message: string;
}

export interface ScanResult {
  issues: ScanIssue[];
  total_issues: number;
  scanned_blocks: number;
  duration_ms: number;
}

// ---------------------------------------------------------------------------
// Method option types
// ---------------------------------------------------------------------------

export interface RecallOptions {
  limit?: number;
  activeOnly?: boolean;
  backend?: SearchBackend;
}

export interface ClientOptions {
  token?: string;
  timeoutMs?: number;
}
