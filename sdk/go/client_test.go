package mindmem_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	mindmem "github.com/star-ga/mind-mem/sdk/go"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// serve returns an httptest.Server that always responds with the given status
// code, optional headers, and a JSON body.
func serve(t *testing.T, status int, body interface{}, headers map[string]string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for k, v := range headers {
			w.Header().Set(k, v)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		if body != nil {
			_ = json.NewEncoder(w).Encode(body)
		}
	}))
}

// captureServer returns an httptest.Server that records the last request and
// responds with the given fixture.
func captureServer(t *testing.T, status int, body interface{}) (*httptest.Server, *http.Request) {
	t.Helper()
	var captured *http.Request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured = r
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		if body != nil {
			_ = json.NewEncoder(w).Encode(body)
		}
	}))
	t.Cleanup(func() { srv.Close() })
	return srv, captured
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

func TestNewClient_StripTrailingSlash(t *testing.T) {
	c := mindmem.NewClient("http://localhost:8080///")
	if c.BaseURL != "http://localhost:8080" {
		t.Fatalf("expected stripped URL, got %q", c.BaseURL)
	}
}

func TestNewClient_NoTrailingSlash(t *testing.T) {
	c := mindmem.NewClient("http://localhost:8080")
	if c.BaseURL != "http://localhost:8080" {
		t.Fatalf("unexpected BaseURL %q", c.BaseURL)
	}
}

func TestNewClient_DefaultTimeout(t *testing.T) {
	c := mindmem.NewClient("http://localhost:8080")
	if c.Timeout != 30*time.Second {
		t.Fatalf("expected 30s default timeout, got %v", c.Timeout)
	}
}

func TestWithToken(t *testing.T) {
	c := mindmem.NewClient("http://localhost:8080", mindmem.WithToken("secret"))
	if c.Token != "secret" {
		t.Fatalf("expected token 'secret', got %q", c.Token)
	}
}

func TestWithTimeout(t *testing.T) {
	c := mindmem.NewClient("http://localhost:8080", mindmem.WithTimeout(5*time.Second))
	if c.Timeout != 5*time.Second {
		t.Fatalf("expected 5s, got %v", c.Timeout)
	}
}

func TestWithHTTPClient(t *testing.T) {
	custom := &http.Client{Timeout: 99 * time.Second}
	c := mindmem.NewClient("http://localhost:8080", mindmem.WithHTTPClient(custom))
	if c.HTTPClient != custom {
		t.Fatalf("expected injected http.Client")
	}
}

// ---------------------------------------------------------------------------
// Auth header
// ---------------------------------------------------------------------------

func TestAuthHeader_WithToken(t *testing.T) {
	var gotAuth, gotToken string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotToken = r.Header.Get("X-MindMem-Token")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(mindmem.HealthResult{Status: "ok", Version: "3.2.0"})
	}))
	defer srv.Close()

	c := mindmem.NewClient(srv.URL, mindmem.WithToken("my-token"))
	_, err := c.Health(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotAuth != "Bearer my-token" {
		t.Errorf("Authorization header: got %q, want %q", gotAuth, "Bearer my-token")
	}
	if gotToken != "my-token" {
		t.Errorf("X-MindMem-Token header: got %q, want %q", gotToken, "my-token")
	}
}

func TestAuthHeader_WithoutToken(t *testing.T) {
	var gotAuth, gotToken string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotToken = r.Header.Get("X-MindMem-Token")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(mindmem.HealthResult{Status: "ok"})
	}))
	defer srv.Close()

	c := mindmem.NewClient(srv.URL)
	_, _ = c.Health(context.Background())
	if gotAuth != "" {
		t.Errorf("expected no Authorization header, got %q", gotAuth)
	}
	if gotToken != "" {
		t.Errorf("expected no X-MindMem-Token header, got %q", gotToken)
	}
}

// ---------------------------------------------------------------------------
// Recall round-trip
// ---------------------------------------------------------------------------

func TestRecall_RoundTrip(t *testing.T) {
	cat := "infra"
	fixture := mindmem.RecallResult{
		Query: "postgres",
		Results: []mindmem.RecallItem{
			{
				Block: mindmem.Block{
					ID:          "b1",
					Content:     "Use pgvector",
					Importance:  0.9,
					Tier:        mindmem.TierWorking,
					CreatedAt:   "2026-01-01T00:00:00Z",
					UpdatedAt:   "2026-01-01T00:00:00Z",
					Keywords:    []string{"postgres"},
					Category:    &cat,
					Active:      true,
					AccessCount: 3,
				},
				Score: 0.92,
				Rank:  1,
			},
		},
		Total:       1,
		BackendUsed: "hybrid",
		LatencyMs:   12,
	}

	var capturedPath, capturedQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		capturedQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(fixture)
	}))
	defer srv.Close()

	c := mindmem.NewClient(srv.URL)
	result, err := c.Recall(context.Background(), "postgres", mindmem.RecallOptions{
		Limit:   5,
		Backend: mindmem.BackendHybrid,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if capturedPath != "/v1/recall" {
		t.Errorf("path: got %q, want /v1/recall", capturedPath)
	}
	if !strings.Contains(capturedQuery, "limit=5") {
		t.Errorf("missing limit param in %q", capturedQuery)
	}
	if !strings.Contains(capturedQuery, "backend=hybrid") {
		t.Errorf("missing backend param in %q", capturedQuery)
	}
	if result.Total != 1 {
		t.Errorf("total: got %d, want 1", result.Total)
	}
	if result.Results[0].Block.ID != "b1" {
		t.Errorf("block ID: got %q, want b1", result.Results[0].Block.ID)
	}
}

// ---------------------------------------------------------------------------
// GetBlock path encoding
// ---------------------------------------------------------------------------

func TestGetBlock_PathEncoding(t *testing.T) {
	var capturedPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.RawPath
		if capturedPath == "" {
			capturedPath = r.URL.Path
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(mindmem.BlockResult{Block: mindmem.Block{ID: "x"}})
	}))
	defer srv.Close()

	c := mindmem.NewClient(srv.URL)
	_, err := c.GetBlock(context.Background(), "block/with spaces")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(capturedPath, "block%2Fwith%20spaces") {
		t.Errorf("expected encoded path, got %q", capturedPath)
	}
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

func TestError_401_ErrUnauthorized(t *testing.T) {
	srv := serve(t, 401, map[string]string{"error": "Unauthorised"}, nil)
	defer srv.Close()

	c := mindmem.NewClient(srv.URL)
	_, err := c.Health(context.Background())
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, mindmem.ErrUnauthorized) {
		t.Errorf("expected ErrUnauthorized, got %v", err)
	}
	var apiErr *mindmem.APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 401 {
		t.Errorf("expected *APIError with StatusCode 401")
	}
}

func TestError_403_ErrUnauthorized(t *testing.T) {
	srv := serve(t, 403, map[string]string{"error": "Forbidden"}, nil)
	defer srv.Close()

	c := mindmem.NewClient(srv.URL)
	_, err := c.Health(context.Background())
	if !errors.Is(err, mindmem.ErrUnauthorized) {
		t.Errorf("expected ErrUnauthorized for 403, got %v", err)
	}
}

func TestError_429_ErrRateLimitWithRetryAfter(t *testing.T) {
	srv := serve(t, 429,
		map[string]string{"error": "Too Many Requests"},
		map[string]string{"Retry-After": "60"},
	)
	defer srv.Close()

	c := mindmem.NewClient(srv.URL)
	_, err := c.Recall(context.Background(), "test", mindmem.RecallOptions{})
	if !errors.Is(err, mindmem.ErrRateLimit) {
		t.Errorf("expected ErrRateLimit, got %v", err)
	}
	var apiErr *mindmem.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError")
	}
	if apiErr.RetryAfter != 60 {
		t.Errorf("RetryAfter: got %d, want 60", apiErr.RetryAfter)
	}
}

func TestError_500_ErrServer(t *testing.T) {
	srv := serve(t, 500, map[string]string{"error": "Internal Server Error"}, nil)
	defer srv.Close()

	c := mindmem.NewClient(srv.URL)
	_, err := c.Scan(context.Background())
	if !errors.Is(err, mindmem.ErrServer) {
		t.Errorf("expected ErrServer, got %v", err)
	}
	var apiErr *mindmem.APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 500 {
		t.Errorf("expected *APIError with StatusCode 500")
	}
}

func TestIsRetryable(t *testing.T) {
	cases := []struct {
		name      string
		err       error
		retryable bool
	}{
		{"nil", nil, false},
		{"ErrRateLimit", mindmem.ErrRateLimit, true},
		{"ErrServer", mindmem.ErrServer, true},
		{"ErrUnauthorized", mindmem.ErrUnauthorized, false},
		{"429 APIError", &mindmem.APIError{StatusCode: 429}, true},
		{"500 APIError", &mindmem.APIError{StatusCode: 500}, true},
		{"401 APIError", &mindmem.APIError{StatusCode: 401}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := mindmem.IsRetryable(tc.err); got != tc.retryable {
				t.Errorf("IsRetryable(%v) = %v, want %v", tc.err, got, tc.retryable)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Context cancellation
// ---------------------------------------------------------------------------

func TestContextCancellation(t *testing.T) {
	blocked := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Block until the test is done, simulating a slow server.
		<-blocked
	}))
	defer srv.Close()
	defer close(blocked)

	ctx, cancel := context.WithCancel(context.Background())
	c := mindmem.NewClient(srv.URL)

	done := make(chan error, 1)
	go func() {
		_, err := c.Health(ctx)
		done <- err
	}()

	cancel()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("expected error after context cancellation, got nil")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for context cancellation to propagate")
	}
}

// ---------------------------------------------------------------------------
// Timeout
// ---------------------------------------------------------------------------

func TestTimeout(t *testing.T) {
	blocked := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-blocked
	}))
	defer srv.Close()
	defer close(blocked)

	c := mindmem.NewClient(srv.URL, mindmem.WithTimeout(50*time.Millisecond))
	start := time.Now()
	_, err := c.Health(context.Background())
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
	if elapsed > 2*time.Second {
		t.Errorf("timeout took too long: %v", elapsed)
	}
}
