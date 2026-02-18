# /recall — Memory Search

Search across all structured memory files. Default backend: BM25 scoring with Porter stemming and domain-aware query expansion. Optional: graph-based cross-reference boosting (`--graph`). Optional: vector/embedding backend (configure in mind-mem.json). Returns ranked results with block ID, type, score, excerpt, and file path.

## When to Use
- Before making decisions (check if a similar decision already exists)
- When asked about past events, decisions, or tasks
- To find related context for a current problem
- To check what's known about a person, project, or tool
- To explore connections between decisions and tasks (`--graph`)

## How to Run

### Basic Search
```bash
python3 maintenance/recall.py --query "authentication" --workspace "${MIND_MEM_WORKSPACE:-.}"
```

### Graph-Boosted Search (cross-reference neighbor discovery)
```bash
python3 maintenance/recall.py --query "database" --graph --workspace "${MIND_MEM_WORKSPACE:-.}"
```

### JSON Output (for programmatic use)
```bash
python3 maintenance/recall.py --query "auth" --workspace "${MIND_MEM_WORKSPACE:-.}" --json --limit 5
```

### Active Items Only
```bash
python3 maintenance/recall.py --query "deadline" --workspace "${MIND_MEM_WORKSPACE:-.}" --active-only
```

## What It Searches
- `decisions/DECISIONS.md` — All decisions
- `tasks/TASKS.md` — All tasks
- `entities/projects.md` — Projects
- `entities/people.md` — People
- `entities/tools.md` — Tools
- `entities/incidents.md` — Incidents
- `intelligence/CONTRADICTIONS.md` — Known contradictions
- `intelligence/DRIFT.md` — Drift detections
- `intelligence/SIGNALS.md` — Captured signals

## Scoring
Results are ranked by BM25 relevance (k1=1.2, b=0.75) with:
- **Stemming** — "queries" matches "query", "deployed" matches "deployment"
- **Query expansion** — "auth" expands to include "authentication", "login", "oauth", "jwt"
- **Recency** — Recent items score higher
- **Active status** — Active items get 1.2x boost
- **Priority** — P0/P1 items get 1.1x boost
- **Graph neighbors** — With `--graph`, blocks connected via cross-references to keyword matches get a 0.3x boost (tagged `[graph]` in output)
