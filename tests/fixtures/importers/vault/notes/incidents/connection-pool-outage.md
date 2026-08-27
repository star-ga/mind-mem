---
title: Connection pool outage
type: incident
severity: high
created: 2026-01-14T09:12:00Z
---
The failover left a stale connection pool behind. Health checks were
advisory only, so the pool kept handing out dead sockets for eleven
minutes.

Follow-up work is tracked against [[architecture]].
