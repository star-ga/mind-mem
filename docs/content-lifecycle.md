# Content-category lifetimes

Content lifetimes are opt-in and separate from storage kinds, lineage edge
kinds, access frequency and trust tiers. They classify the meaning of a fact.
They do not delete corpus records or automatically renew them.

Configure `recall.validity_gate.content_categories` in `mind-mem.json`:

```json
{
  "recall": {
    "validity_gate": {
      "enabled": true,
      "content_categories": {
        "enabled": true,
        "ttl_days": {"infra": 2, "status": 1}
      }
    }
  }
}
```

The numbers above are an example, not product defaults. Enabling the policy
requires explicit positive integer day limits for both `infra` and `status`.
The supported range is 1–36,500 days. Unknown categories, booleans in numeric
fields, missing limits and attempts to configure time expiry for a durable
category are rejected. The validity gate must also be enabled.

| `ContentCategory` | Time policy |
| --- | --- |
| `infra`, `status` | Becomes stale at `age_days >= ttl_days`, regardless of reads or confirmations. |
| `decision`, `architecture` | No automatic time expiry. Contradiction, status and provenance rules still apply. |
| `credential` | No automatic time expiry. Explicit governed `Status: revoked` withdraws it from recall and shared read admission. |
| Field absent | Existing uncategorized-fact behavior. No inferred category. |

Write the category and the semantic date as ordinary governed fields:

```text
ContentCategory: status
ContentValidFrom: 2026-09-14
```

`ContentValidFrom` is exactly a UTC calendar date, `YYYY-MM-DD`. If absent,
the policy uses the block's `Date`, then its dated block ID. An explicitly
malformed date never falls through to a replacement. A missing, invalid or
future date requires review. Recall uses its existing pinned
`scoring_instant`; a replay does not consult the wall clock.

Recall annotates managed facts under `validity.content_lifecycle` and applies
the configured validity demotion when a short-lived fact expires or its
category/date is invalid. Expired status facts remain inspectable; expiry is
a review signal, not an assertion that the statement became false. Canonical
block fields are consulted so an old indexed value or a caller-supplied access
count cannot renew a fact. The dream-cycle stale pass uses the same policy,
and touching a Markdown file cannot reset semantic age.

For a `TierManager` with a workspace, managed categories bypass the idle
demotion/eviction sweep. Explicit demotion still works. A manager constructed
only with a database path retains its existing behavior because it has no
workspace policy to consult. No tier is promoted from usage or outcomes.

Category changes, renewal and revocation are corpus changes and must follow
the existing governed proposal/apply path. This feature creates no alternate
writer or ledger. Other revoked historical records remain available under
the existing status policy; the credential withdrawal rule is specific to
an explicitly categorized credential.

Acceptance controls are in `tests/test_content_lifecycle.py`: real scan and
SQLite recall, pinned-clock replay, stale cached metadata, file-touch
resistance, idle-tier preservation, explicit demotion, credential withdrawal
through recall and the public direct-fetch handler, and invalid configuration.
Namespace-specific source identity and a full transport sweep remain part of
the roadmap integration review; these controls do not establish them.
