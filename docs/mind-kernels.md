# MIND Kernels

## Overview

MIND kernels are configuration files (`.mind` extension) that customize scoring and retrieval behavior. They live in the `.mind/` subdirectory of the workspace.

## Kernel Format

Kernels use TOML-like syntax:

```toml
# recall.mind — customize BM25F scoring
[bm25]
k1 = 1.5
b = 0.8

[field_weights]
statement = 4.0
type = 1.5
tags = 2.5
context = 1.0

[boosts]
entity = 0.3
priority = 0.5
bigram = 0.2
```

## Available Kernels

### recall.mind
Controls the BM25F scoring pipeline.

| Section | Key | Type | Default | Description |
|---------|-----|------|---------|-------------|
| bm25 | k1 | float | 1.2 | Term frequency saturation |
| bm25 | b | float | 0.75 | Length normalization |
| field_weights | statement | float | 3.0 | Statement weight |
| field_weights | type | float | 1.0 | Type weight |
| field_weights | tags | float | 2.0 | Tags weight |
| boosts | entity | float | 0.2 | Entity match boost |
| boosts | priority | float | 0.4 | Priority boost |
| boosts | bigram | float | 0.15 | Bigram match boost |

## MCP Tools

- `list_mind_kernels` — List available kernels
- `get_mind_kernel` — Read kernel source

## Creating Custom Kernels

1. Create `.mind/` in your workspace
2. Add a `.mind` file with the kernel name
3. Use TOML sections and keys as documented
4. Restart the MCP server to load changes
