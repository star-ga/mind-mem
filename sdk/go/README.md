# mind-mem Go SDK

Go client for the [mind-mem](https://github.com/star-ga/mind-mem) REST API.
Stdlib-only — no external dependencies.

## Install

```sh
go get github.com/star-ga/mind-mem/sdk/go@latest
```

## Quick start

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    mindmem "github.com/star-ga/mind-mem/sdk/go"
)

func main() {
    client := mindmem.NewClient(
        "http://localhost:8080",
        mindmem.WithToken(os.Getenv("MIND_MEM_TOKEN")),
    )

    result, err := client.Recall(context.Background(), "what did we decide?",
        mindmem.RecallOptions{Limit: 5},
    )
    if err != nil {
        log.Fatal(err)
    }
    for _, item := range result.Results {
        fmt.Printf("[%.2f] %s\n", item.Score, item.Block.Content)
    }
}
```

## API

### Construction

```go
// Defaults: 30 s timeout, no auth token.
client := mindmem.NewClient("http://localhost:8080")

// With options:
client := mindmem.NewClient("http://localhost:8080",
    mindmem.WithToken("secret"),
    mindmem.WithTimeout(10 * time.Second),
    mindmem.WithHTTPClient(myHTTPClient),
)
```

### Methods

| Method | Endpoint |
|---|---|
| `Recall(ctx, query, RecallOptions)` | `GET /v1/recall` |
| `GetBlock(ctx, blockID)` | `GET /v1/blocks/{id}` |
| `ListContradictions(ctx)` | `GET /v1/contradictions` |
| `Health(ctx)` | `GET /v1/health` |
| `Scan(ctx)` | `GET /v1/scan` |

### Error handling

All API errors are returned as `*APIError`. Use the provided sentinels with
`errors.Is`:

```go
result, err := client.Recall(ctx, "query", mindmem.RecallOptions{})
if err != nil {
    switch {
    case errors.Is(err, mindmem.ErrUnauthorized):
        // token missing or invalid
    case errors.Is(err, mindmem.ErrRateLimit):
        var apiErr *mindmem.APIError
        errors.As(err, &apiErr)
        time.Sleep(time.Duration(apiErr.RetryAfter) * time.Second)
    case errors.Is(err, mindmem.ErrServer):
        // transient server error
    }
    if mindmem.IsRetryable(err) {
        // safe to retry after back-off
    }
}
```

## Requirements

- Go 1.21+
