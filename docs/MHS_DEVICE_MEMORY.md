# MHS / Device Memory Boundary

> Status: roadmap/specification work. Anthropic announced the Model Hardware Standard (MHS) research preview on 2026-08-27. The final open-source MHS specification is not public yet. This document therefore defines MIND's stable internal boundaries and an MHS compatibility seam without claiming conformance to an unpublished specification. When the normative MHS specification is released, the adapter and conformance layer MUST be reconciled before any interoperability claim is made.

`mind-mem` may make physical-device experiences more useful by remembering history and preferences.
It must never become the source of truth for current physical state.

## Good memory

- receipt references and verified action history;
- device aliases and user-friendly names;
- long-term maintenance/calibration facts;
- user preferences (for example a preferred room temperature range);
- recurring failure summaries;
- learned contextual associations that are clearly marked as memory, not live state.

## Forbidden authority substitution

A memory like `thermostat was 22 C ten minutes ago` cannot satisfy a current-state precondition for a
physical action. Safety/governance decisions read live state through the device gateway and bind the
resulting state hash.

```text
memory = context/history
MHS/device gateway = live physical observation authority
```

## Evidence linkage

Store content-addressed references to DeviceActionEvidence rather than copying/forking the signed
evidence schema. Verification remains owned by the canonical evidence primitive.

## Freshness

Memory records may include human timestamps/freshness metadata, but a caller requesting `live=true`
(or equivalent) must be routed to the device observation path rather than served from recall cache.

## Roadmap

1. Device/receipt reference record type.
2. Explicit `historical` vs `live-required` query semantics.
3. Prevent memory recall from satisfying physical pre-state hash requirements.
4. Preference retrieval integration for DeviceIntent while retaining governance as final authority.
