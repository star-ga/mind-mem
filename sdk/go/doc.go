// Package mindmem is the official Go SDK for the mind-mem REST API.
//
// mind-mem is a persistent, auditable, contradiction-safe memory system for
// AI agents. This package provides a stdlib-only HTTP client covering the
// read-only API surface: Recall, GetBlock, ListContradictions, Health, and
// Scan.
//
// # Getting started
//
//	client := mindmem.NewClient("http://localhost:8080",
//	    mindmem.WithToken("your-token"),
//	)
//
//	result, err := client.Recall(ctx, "what did we decide?",
//	    mindmem.RecallOptions{Limit: 5},
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	for _, item := range result.Results {
//	    fmt.Println(item.Score, item.Block.Content)
//	}
//
// # Error handling
//
// All API errors are returned as *APIError. Use the sentinel helpers
// ErrUnauthorized, ErrRateLimit, and ErrServer together with errors.Is, or
// inspect the concrete *APIError for the status code and Retry-After value:
//
//	result, err := client.Recall(ctx, "query", mindmem.RecallOptions{})
//	if err != nil {
//	    var apiErr *mindmem.APIError
//	    if errors.As(err, &apiErr) {
//	        log.Printf("status %d: %s (request-id %s)",
//	            apiErr.StatusCode, apiErr.Message, apiErr.RequestID)
//	    }
//	    if mindmem.IsRetryable(err) {
//	        // back off and retry
//	    }
//	}
package mindmem
