// Copyright 2026 STARGA, Inc.
package mindmem

import "encoding/json"

// RawResponse preserves the complete server JSON, including optional evidence
// and fields introduced after this SDK. It is not a verification verdict.
type RawResponse struct {
	Raw           json.RawMessage `json:"-"`
	SchemaVersion string          `json:"_schema_version,omitempty"`
}

func (r *RawResponse) captureJSON(raw json.RawMessage) { r.Raw = append(json.RawMessage(nil), raw...) }

type SearchBackend string

const (
	BackendAuto   SearchBackend = "auto"
	BackendBM25   SearchBackend = "bm25"
	BackendHybrid SearchBackend = "hybrid"
)

// Block carries governed fields with their case-sensitive server spelling.
// Arbitrary extra block fields remain available in BlockResult.Raw.
type Block struct {
	ID        string `json:"_id"`
	Statement string `json:"Statement,omitempty"`
	Status    string `json:"Status,omitempty"`
	Date      string `json:"Date,omitempty"`
	Type      string `json:"Type,omitempty"`
}

// RecallItem is a flat ranked hit, not a nested persisted block.
type RecallItem struct {
	ID      string  `json:"_id"`
	Type    string  `json:"type,omitempty"`
	Score   float64 `json:"score"`
	Excerpt string  `json:"excerpt"`
	File    string  `json:"file,omitempty"`
	Line    int     `json:"line,omitempty"`
	Status  string  `json:"status,omitempty"`
	Date    string  `json:"Date,omitempty"`
}

type RecallResult struct {
	RawResponse
	Query          string          `json:"query"`
	QueryID        string          `json:"query_id,omitempty"`
	Results        []RecallItem    `json:"results"`
	Count          int             `json:"count"`
	Backend        string          `json:"backend"`
	ScoringInstant string          `json:"scoring_instant,omitempty"`
	Attestation    json.RawMessage `json:"attestation,omitempty"`
	Warnings       []string        `json:"warnings,omitempty"`
}

type BlockResult struct {
	RawResponse
	BlockID string `json:"block_id"`
	Found   bool   `json:"found"`
	Block   Block  `json:"block"`
}

// Contradictions is a count; Resolutions holds the optional structured findings.
type ContradictionsResult struct {
	RawResponse
	Status         string            `json:"status"`
	Contradictions int               `json:"contradictions"`
	Resolutions    []json.RawMessage `json:"resolutions,omitempty"`
	Message        string            `json:"message,omitempty"`
}

type HealthResult struct {
	RawResponse
	Status                 string  `json:"status"`
	APIVersion             string  `json:"api_version"`
	Workspace              *string `json:"workspace,omitempty"`
	WorkspaceExists        *bool   `json:"workspace_exists,omitempty"`
	WorkspaceSchemaVersion string  `json:"schema_version"`
}

type ScanResult struct {
	RawResponse
	Backend string                     `json:"backend"`
	Checks  map[string]json.RawMessage `json:"checks"`
}

type RecallOptions struct {
	Limit      int
	ActiveOnly bool
	Backend    SearchBackend
	// UTC YYYY-MM-DD for deterministic replay; empty uses the server default.
	ScoringInstant string
}
