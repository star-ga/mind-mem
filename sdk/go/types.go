package mindmem

// BlockTier represents the storage tier of a memory block.
type BlockTier string

const (
	TierWorking  BlockTier = "WORKING"
	TierArchival BlockTier = "ARCHIVAL"
	TierCold     BlockTier = "COLD"
)

// SearchBackend selects the retrieval backend used for a recall query.
type SearchBackend string

const (
	BackendAuto   SearchBackend = "auto"
	BackendBM25   SearchBackend = "bm25"
	BackendHybrid SearchBackend = "hybrid"
)

// Block is a single persisted memory unit.
type Block struct {
	ID          string    `json:"id"`
	Content     string    `json:"content"`
	Importance  float64   `json:"importance"`
	Tier        BlockTier `json:"tier"`
	CreatedAt   string    `json:"created_at"`
	UpdatedAt   string    `json:"updated_at"`
	Keywords    []string  `json:"keywords"`
	Category    *string   `json:"category"`
	Namespace   *string   `json:"namespace"`
	Active      bool      `json:"active"`
	AccessCount int       `json:"access_count"`
	Provenance  *string   `json:"provenance"`
}

// RecallItem is one result entry returned by the Recall endpoint.
type RecallItem struct {
	Block Block   `json:"block"`
	Score float64 `json:"score"`
	Rank  int     `json:"rank"`
}

// RecallResult is the response envelope for POST /v1/recall.
type RecallResult struct {
	Query       string       `json:"query"`
	Results     []RecallItem `json:"results"`
	Total       int          `json:"total"`
	BackendUsed string       `json:"backend_used"`
	LatencyMs   float64      `json:"latency_ms"`
}

// BlockResult is the response envelope for GET /v1/blocks/{id}.
type BlockResult struct {
	Block Block `json:"block"`
}

// Contradiction describes a detected conflict between two memory blocks.
type Contradiction struct {
	BlockAID      string  `json:"block_a_id"`
	BlockBID      string  `json:"block_b_id"`
	ConflictScore float64 `json:"conflict_score"`
	Description   string  `json:"description"`
	DetectedAt    string  `json:"detected_at"`
}

// ContradictionsResult is the response envelope for GET /v1/contradictions.
type ContradictionsResult struct {
	Contradictions []Contradiction `json:"contradictions"`
	Total          int             `json:"total"`
}

// HealthResult is the response envelope for GET /v1/health.
type HealthResult struct {
	Status            string `json:"status"`
	Version           string `json:"version"`
	BlockCount        int    `json:"block_count"`
	IndexState        string `json:"index_state"`
	EncryptionEnabled bool   `json:"encryption_enabled"`
	UptimeSeconds     int    `json:"uptime_seconds"`
}

// ScanIssue is a single issue detected during a governance scan.
type ScanIssue struct {
	Type     string  `json:"type"`
	Severity string  `json:"severity"`
	BlockID  *string `json:"block_id"`
	Message  string  `json:"message"`
}

// ScanResult is the response envelope for GET /v1/scan.
type ScanResult struct {
	Issues        []ScanIssue `json:"issues"`
	TotalIssues   int         `json:"total_issues"`
	ScannedBlocks int         `json:"scanned_blocks"`
	DurationMs    float64     `json:"duration_ms"`
}

// RecallOptions controls the behaviour of a Recall call.
type RecallOptions struct {
	// Limit caps the number of results returned (0 means server default).
	Limit int
	// ActiveOnly restricts results to non-archived blocks.
	ActiveOnly bool
	// Backend selects the retrieval backend. Empty string means server default.
	Backend SearchBackend
}
