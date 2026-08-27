---
title: Storage architecture
tags:
  - architecture
  - storage
status: current
---
The block store is append-only. Every write lands in a canonical file
chosen by the block-id prefix, which keeps recall and the on-disk layout
in lockstep.

Downstream of this decision: [[connection-pool-outage]] and the
[[nightly-compaction]] job both assume append-only semantics.
