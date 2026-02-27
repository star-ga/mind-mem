#!/bin/bash
# Pre-seed a demo workspace for VHS recording
set -e

WS="$HOME/demo-workspace"
rm -rf "$WS"
mind-mem-init "$WS" >/dev/null 2>&1

cat >> "$WS/decisions/DECISIONS.md" << 'EOF'

[D-20260215-001]
Type: Decision
Statement: Use async/await for all API endpoints
Status: active
Priority: P1
Tags: api, architecture
Rationale: Async improves throughput under concurrent load. FastAPI native.
Date: 2026-02-15

[D-20260210-003]
Type: Decision
Statement: REST over GraphQL for public API
Status: active
Priority: P2
Tags: api, design
Rationale: REST is simpler for consumers. GraphQL adds complexity without clear benefit.
Date: 2026-02-10

[D-20260220-002]
Type: Decision
Statement: SQLite WAL mode for concurrent read access
Status: active
Priority: P1
Tags: database, performance
Rationale: WAL allows multiple readers with one writer. Zero-infrastructure.
Date: 2026-02-20
EOF

cat >> "$WS/entities/tools.md" << 'EOF'

[E-TOOL-001]
Type: Entity
SubType: Tool
Statement: FastAPI — async Python web framework for API development
Status: active
Tags: python, api, framework

[E-TOOL-002]
Type: Entity
SubType: Tool
Statement: SQLite — embedded relational database with FTS5 full-text search
Status: active
Tags: database, search, local-first
EOF

echo "Demo workspace ready at $WS"
