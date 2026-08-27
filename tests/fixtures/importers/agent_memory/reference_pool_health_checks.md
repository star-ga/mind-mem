---
name: reference_pool_health_checks
description: "Advisory pool health checks cannot evict a dead socket, so a failover leaves the pool handing out unusable connections until a request times out."
metadata:
  node_type: memory
  type: reference
  originSessionId: 6f1c9d21-0b44-4f0e-9d17-2c1d5a9e77aa
---
Health checks that only log cannot evict a dead socket. After a failover
the pool keeps handing out unusable connections until a caller times
out, which is eleven minutes of silent failure.

Related: [[feedback_append_only_writes]].
