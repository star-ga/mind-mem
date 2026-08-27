---
name: nested_retention_policy
description: "Retention is decided per corpus file, not per block."
metadata:
  type: project
---
Retention is decided per corpus file, not per block. A note kept in a
nested directory is still part of the same corpus and is compacted on
the same schedule.
