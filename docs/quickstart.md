# mind-mem Quickstart

Get mind-mem running in under 2 minutes.

---

## 1. Install

```bash
git clone https://github.com/star-ga/mind-mem.git
cd mind-mem
./install.sh --all
```

The installer auto-detects Claude Code, Codex CLI, Gemini CLI, Cursor, Windsurf, Zed, and Claude Desktop.

## 2. Verify

```bash
python3 scripts/validate_py.py /path/to/your/workspace
# Expected: 74 checks | 74 passed | 0 issues
```

## 3. First Recall

```bash
python3 scripts/recall.py --query "authentication" --workspace /path/to/your/workspace
```

No results yet -- your workspace is empty. Make some decisions first.

## 4. Add Your First Decision

Create or edit `decisions/DECISIONS.md` in your workspace:

```markdown
[D-20260218-001]
Date: 2026-02-18
Status: active
Statement: Use JWT tokens for API authentication
Tags: auth, security, api
Rationale: Stateless, works across microservices
```

Now recall finds it:

```bash
python3 scripts/recall.py --query "authentication" --workspace .
# -> [D-20260218-001] Use JWT tokens for API authentication (score: 12.4)
```

## 5. Run Integrity Scan

```bash
python3 scripts/intel_scan.py /path/to/your/workspace
# Expected: 0 critical | 0 warnings
```

## 6. Auto-Capture from Sessions

After each coding session, mind-mem auto-captures decision-like language from your daily logs:

```bash
python3 scripts/capture.py /path/to/your/workspace
```

Signals land in `intelligence/SIGNALS.md`. Review and promote with `/apply`.

## 7. Use via MCP

All MCP-compatible clients can now use mind-mem tools:

| Tool | What it does |
| ---- | ------------ |
| `recall` | Search memory with BM25 |
| `propose_update` | Propose a new decision/task |
| `scan` | Run integrity scan |
| `hybrid_search` | BM25 + Vector + RRF fusion |
| `intent_classify` | Show query routing strategy |
| `prefetch` | Pre-assemble likely context |

Full list: 16 tools, 8 resources. See [API Reference](api-reference.md).

## 8. Enable Hybrid Search (Optional)

Edit `mind-mem.json` in your workspace:

```json
{
  "recall": {
    "backend": "hybrid",
    "vector_enabled": true
  }
}
```

Requires `pip install mind-mem[embeddings]`.

## Next Steps

- [Architecture](architecture.md) -- how the recall pipeline works
- [Configuration](configuration.md) -- every setting explained
- [API Reference](api-reference.md) -- all 16 MCP tools
- [Migration Guide](migration.md) -- upgrading from mem-os
