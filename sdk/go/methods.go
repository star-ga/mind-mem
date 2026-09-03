package mindmem

import (
	"context"
)

// recallRequest is the JSON body POST /v1/recall accepts. Field names and the
// omitempty choices mirror components/schemas/RecallRequest in
// sdk/spec/openapi.json: every field except query has a server-side default,
// so omitting it is how a caller asks for that default.
type recallRequest struct {
	Query      string `json:"query"`
	Limit      int    `json:"limit,omitempty"`
	ActiveOnly bool   `json:"active_only,omitempty"`
	Backend    string `json:"backend,omitempty"`
}

// Recall queries the memory store using full-text and semantic search.
// It maps to POST /v1/recall with a JSON body.
func (c *Client) Recall(ctx context.Context, query string, opts RecallOptions) (*RecallResult, error) {
	body := recallRequest{Query: query}
	if opts.Limit > 0 {
		body.Limit = opts.Limit
	}
	if opts.ActiveOnly {
		body.ActiveOnly = true
	}
	if opts.Backend != "" {
		body.Backend = string(opts.Backend)
	}

	var result RecallResult
	if err := c.post(ctx, RouteRecall.Expand(), body, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// GetBlock fetches a single memory block by its ID.
// It maps to GET /v1/block/{block_id}.
func (c *Client) GetBlock(ctx context.Context, blockID string) (*BlockResult, error) {
	var result BlockResult
	if err := c.get(ctx, RouteGetBlock.Expand(blockID), nil, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// ListContradictions returns all detected contradictions in the memory store.
// It maps to GET /v1/contradictions.
func (c *Client) ListContradictions(ctx context.Context) (*ContradictionsResult, error) {
	var result ContradictionsResult
	if err := c.get(ctx, RouteListContradictions.Expand(), nil, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// Health checks the readiness of the running mind-mem instance.
// It maps to GET /v1/health.
func (c *Client) Health(ctx context.Context) (*HealthResult, error) {
	var result HealthResult
	if err := c.get(ctx, RouteHealth.Expand(), nil, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// Scan runs a governance scan and returns any drift or conflict issues found.
// It maps to GET /v1/scan.
func (c *Client) Scan(ctx context.Context) (*ScanResult, error) {
	var result ScanResult
	if err := c.get(ctx, RouteScan.Expand(), nil, &result); err != nil {
		return nil, err
	}
	return &result, nil
}
