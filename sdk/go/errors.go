package mindmem

import (
	"context"
	"errors"
	"fmt"
)

// APIError is returned for any non-2xx response from the mind-mem server.
type APIError struct {
	// StatusCode is the HTTP status code of the response.
	StatusCode int
	// Message is the error description extracted from the response body or the
	// raw HTTP status line when the body cannot be decoded.
	Message string
	// RequestID is the value of the X-Request-ID response header, if present.
	RequestID string
	// RetryAfter is the parsed Retry-After header value in seconds. Non-zero
	// only for 429 responses that include the header.
	RetryAfter int
}

// Error implements the error interface.
func (e *APIError) Error() string {
	if e.RequestID != "" {
		return fmt.Sprintf("mind-mem: HTTP %d: %s (request-id %s)", e.StatusCode, e.Message, e.RequestID)
	}
	return fmt.Sprintf("mind-mem: HTTP %d: %s", e.StatusCode, e.Message)
}

// Is allows errors.Is comparisons against the sentinel errors.
func (e *APIError) Is(target error) bool {
	switch target {
	case ErrUnauthorized:
		return e.StatusCode == 401 || e.StatusCode == 403
	case ErrRateLimit:
		return e.StatusCode == 429
	case ErrServer:
		return e.StatusCode >= 500
	}
	return false
}

// Sentinel errors for the most common failure categories.
var (
	// ErrUnauthorized is matched by errors.Is when the server returns 401 or 403.
	ErrUnauthorized = errors.New("mind-mem: unauthorized")
	// ErrRateLimit is matched by errors.Is when the server returns 429.
	ErrRateLimit = errors.New("mind-mem: rate limit exceeded")
	// ErrServer is matched by errors.Is when the server returns a 5xx status.
	ErrServer = errors.New("mind-mem: server error")
)

// IsRetryable reports whether the error is transient and safe to retry after a
// back-off. Network errors and 429 / 5xx API errors are considered retryable.
func IsRetryable(err error) bool {
	if err == nil {
		return false
	}
	// Check known sentinel categories first so *APIError.Is() is exercised.
	if errors.Is(err, ErrRateLimit) || errors.Is(err, ErrServer) {
		return true
	}
	// Any other known sentinel (e.g. ErrUnauthorized) is not retryable.
	if errors.Is(err, ErrUnauthorized) {
		return false
	}
	// Concrete *APIError without a matching sentinel.
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode == 429 || apiErr.StatusCode >= 500
	}
	// Non-API error (e.g. network failure): retryable unless context was
	// explicitly cancelled or deadline exceeded.
	return !errors.Is(err, context.Canceled) &&
		!errors.Is(err, context.DeadlineExceeded)
}
