---
name: feedback_append_only_writes
description: "Append-only writes keep the on-disk layout auditable; in-place edits erase the evidence trail that recall depends on."
metadata:
  node_type: memory
  type: feedback
---
Append-only writes keep the on-disk layout auditable. An in-place edit
erases the evidence trail recall depends on, so repairs must be proposed
and reviewed rather than applied straight to the file.

See also [[reference_pool_health_checks]] and [[nested_retention_policy]].
