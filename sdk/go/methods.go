package mindmem

import (
	"context"
	"net/url"
	"strconv"
)

// Recall queries the memory store using full-text and semantic search.
// It maps to GET /v1/recall with query parameters.
func (c *Client) Recall(ctx context.Context, query string, opts RecallOptions) (*RecallResult, error) {
	params := map[string]string{
		"q": url.QueryEscape(query),
	}
	if opts.Limit > 0 {
		params["limit"] = strconv.Itoa(opts.Limit)
	}
	if opts.ActiveOnly {
		params["active_only"] = "true"
	}
	if opts.Backend != "" {
		params["backend"] = string(opts.Backend)
	}

	var result RecallResult
	if err := c.get(ctx, "/v1/recall", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// GetBlock fetches a single memory block by its ID.
// It maps to GET /v1/blocks/{id}.
func (c *Client) GetBlock(ctx context.Context, blockID string) (*BlockResult, error) {
	path := "/v1/blocks/" + url.PathEscape(blockID)

	var result BlockResult
	if err := c.get(ctx, path, nil, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// ListContradictions returns all detected contradictions in the memory store.
// It maps to GET /v1/contradictions.
func (c *Client) ListContradictions(ctx context.Context) (*ContradictionsResult, error) {
	var result ContradictionsResult
	if err := c.get(ctx, "/v1/contradictions", nil, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// Health checks the readiness of the running mind-mem instance.
// It maps to GET /v1/health.
func (c *Client) Health(ctx context.Context) (*HealthResult, error) {
	var result HealthResult
	if err := c.get(ctx, "/v1/health", nil, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// Scan runs a governance scan and returns any drift or conflict issues found.
// It maps to GET /v1/scan.
func (c *Client) Scan(ctx context.Context) (*ScanResult, error) {
	var result ScanResult
	if err := c.get(ctx, "/v1/scan", nil, &result); err != nil {
		return nil, err
	}
	return &result, nil
}
