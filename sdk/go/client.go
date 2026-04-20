package mindmem

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const defaultTimeout = 30 * time.Second

// Option is a functional option for NewClient.
type Option func(*Client)

// WithToken sets the Bearer token sent on every request as both the
// Authorization and X-MindMem-Token headers.
func WithToken(t string) Option {
	return func(c *Client) {
		c.Token = t
	}
}

// WithHTTPClient replaces the default *http.Client used for requests.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) {
		c.HTTPClient = hc
	}
}

// WithTimeout sets the per-request timeout. Defaults to 30 s.
func WithTimeout(d time.Duration) Option {
	return func(c *Client) {
		c.Timeout = d
		// Propagate to the embedded http.Client when using the default one.
		if c.HTTPClient != nil {
			c.HTTPClient.Timeout = d
		}
	}
}

// Client is an HTTP client for the mind-mem REST API.
//
//	client := mindmem.NewClient("http://localhost:8080",
//	    mindmem.WithToken(os.Getenv("MIND_MEM_TOKEN")),
//	)
type Client struct {
	// BaseURL is the scheme+host (and optional port) of the mind-mem server,
	// with no trailing slash.
	BaseURL string
	// HTTPClient is the underlying transport. Override with WithHTTPClient to
	// inject a custom transport (e.g. for testing).
	HTTPClient *http.Client
	// Token is the bearer token sent on every request. May be empty.
	Token string
	// Timeout is the per-request deadline.
	Timeout time.Duration
}

// NewClient constructs a Client pointed at baseURL.
// Trailing slashes on baseURL are stripped so path joining is consistent.
func NewClient(baseURL string, opts ...Option) *Client {
	c := &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		Timeout: defaultTimeout,
	}
	c.HTTPClient = &http.Client{Timeout: c.Timeout}

	for _, o := range opts {
		o(c)
	}
	return c
}

// buildURL joins the client's BaseURL with path and appends query pairs.
func (c *Client) buildURL(path string, params map[string]string) string {
	u := c.BaseURL + path
	if len(params) == 0 {
		return u
	}
	first := true
	for k, v := range params {
		if first {
			u += "?" + k + "=" + v
			first = false
		} else {
			u += "&" + k + "=" + v
		}
	}
	return u
}

// addHeaders sets the standard request headers including the auth token.
func (c *Client) addHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
		req.Header.Set("X-MindMem-Token", c.Token)
	}
}

// get performs a GET request and decodes the JSON response into dst.
func (c *Client) get(ctx context.Context, path string, params map[string]string, dst interface{}) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.buildURL(path, params), nil)
	if err != nil {
		return fmt.Errorf("mind-mem: build request: %w", err)
	}
	c.addHeaders(req)

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("mind-mem: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		if err := json.NewDecoder(resp.Body).Decode(dst); err != nil {
			return fmt.Errorf("mind-mem: decode response: %w", err)
		}
		return nil
	}

	return c.errorFromResponse(resp)
}

// errorFromResponse converts a non-2xx response into a typed *APIError.
func (c *Client) errorFromResponse(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)

	message := fmt.Sprintf("HTTP %d", resp.StatusCode)
	var envelope struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(body, &envelope) == nil && envelope.Error != "" {
		message = envelope.Error
	}

	apiErr := &APIError{
		StatusCode: resp.StatusCode,
		Message:    message,
		RequestID:  resp.Header.Get("X-Request-ID"),
	}

	if resp.StatusCode == 429 {
		if ra := resp.Header.Get("Retry-After"); ra != "" {
			var secs int
			if _, scanErr := fmt.Sscanf(ra, "%d", &secs); scanErr == nil {
				apiErr.RetryAfter = secs
			}
		}
	}

	return apiErr
}
