# Local retrieval-receipt pilot — 2026-09-14

This operator pilot found a real reader-version mismatch and measured the local
receipt export and verification path. It advances **RE.4 to partial**. It does
not complete the benchmark matrix in the [receipt contract](../specs/retrieval-receipt-contract.md#8-benchmarks-and-determinism-claims).

## Useful debugging result

The same authentic, frozen operator history contained 3,695 rows: 3,694 V1 rows
and one V2 local-serving row. The installed 5.0.2 reader rejected the V2 schema
and reported an unreadable history. The 5.0.3 candidate reader validated all
3,695 rows and the head; its offline export verified as `locally_consistent`.
This distinguished an incompatible reader from broken ledger bytes. No history
was rewritten, deleted or re-anchored. The diagnosis preceded the performance
study and was not a blind experiment.

The private corpus and receipt payload are not published. The accompanying
[raw measurements](local-receipt-pilot-20260914.json) contain timings, resource
measurements, integrity hashes and control results, without queries or block
contents. They support inspection of the measurements, not independent
verification of the private events or an independently observed execution claim.

## Source, machine and procedure

- Source: `685aa573b6f2407d48ccd6c0093ffb5124fb8682`, clean before and after.
- CPU: Intel Core i7-5930K at 3.50 GHz; 12 logical CPUs reported.
- Host: Linux 7.0.0-30-generic, x86-64, glibc 2.43; Python 3.14.4, GCC 15.2.0.
- Measurements: 2026-09-14 16:27:18–16:27:45 UTC. Source, input ledger and head
  hashes were unchanged afterward.
- Thirty warm samples per API operation; ten fresh-process samples per CLI
  operation; two concurrent export clients with ten operations each. Fresh
  processes do not imply a cold OS page cache.
- Nearest-rank p50/p95/p99 include all samples. Memory was measured in a separate
  fresh process using Linux `ru_maxrss` and Python `tracemalloc`.

Before measurement, at 16:14:18 UTC, the protocol fixed these budgets: warm API
p95 at most 2 seconds; CLI p95 at most 5 seconds; peak RSS at most 128 MiB;
package size at most `1.5 * ledger_bytes + 16384`; zero operation errors and
zero input mutations. All six gates passed. The first instrument run combined
API operations and measured memory in a previously warmed process. Its method
was rejected; the corrected run retained the same inputs, budgets and sample
counts. Earlier raw results remain retained separately.

## Measurements

| Operation | Samples | p50 (s) | p95 (s) | p99 (s) | Mean CPU (s) |
|---|---:|---:|---:|---:|---:|
| Existing chain verification | 30 | 0.103431 | 0.124405 | 0.159661 | 0.106701 |
| Receipt export API | 30 | 0.189141 | 0.197638 | 0.219864 | 0.189109 |
| Receipt verification API | 30 | 0.163178 | 0.193866 | 0.194240 | 0.167011 |
| Receipt verification CLI | 10 | 0.315344 | 0.337432 | 0.337432 | 0.320930 |
| Receipt export CLI | 10 | 0.364029 | 0.397790 | 0.397790 | 0.338302 |
| Two-client receipt export | 20 | 0.396423 | 0.432527 | 0.433404 | — |

CLI CPU values are child-process CPU. The concurrent group took 4.102073 seconds
wall time and 7.244999 seconds aggregate child CPU, yielding **4.875584 exports
per second**. Dividing its sample count by summed per-request durations is not
aggregate throughput.

The separate memory control peaked at **72.01 MiB RSS** and **27,660,716 traced
Python bytes**. The package was 3,555,275 bytes for a 2,665,774-byte ledger:
**1.333675× package storage expansion**. This is not physical write amplification.
Every measured operation succeeded.

Existing chain verification performs less work than package export/verification.
It is a local operation baseline, not a matched recall-overhead or competitor
baseline. No ranking, model or provider ran in this study.

## Reproduction and evidence boundaries

At the measured source revision, the CLI operations are:

```bash
mm receipt export --out /new/output/receipt.json
mm receipt verify --input /new/output/receipt.json
```

Run them against an authorized local workspace and retain its source, ledger,
head and package hashes. Output paths must be new. Different histories and
machines will produce different sizes and timings. The private measurement
runner and protocol are identified below; exact replay of these historical
events requires access to the retained private input.

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `6eb6e61b5cfe868def066761cb036902c16ad773d96d587aecdf988ad40029bc` |
| Corrected measurement runner | `fac177108d7250cf023a88fa0f0927ac751f5e4b1422bfd307bac9ccefc39df6` |
| Published raw measurements | `2398fbd3cd7e478ee3d697800b968fb7006144bf1067488fbe92314fd7b7887c` |
| Private input receipt | `72534356464d19a8d0f8b2b5fa64d13be62d15d7712a12f3e0d7af02c1c73248` |
| `src/mind_mem/retrieval_receipts.py` | `293f10e42520cc202bc920a7038d4f95f826cd07619c4ff203087e40f725db86` |
| `src/mind_mem/served_ledger.py` | `680d019a1b4e0b644e3f45e28763f23909615db3bcd6df41ded396b3a91f8b4a` |

Ordinary recall versus recording overhead, controlled V1/V2/anticipation and
repeated-answer modes, concurrent live appends, cold storage, longer histories
and physical write amplification remain unmeasured here. Independent witness
trust, portable identity, billing and external performance comparisons also
remain outside this local pilot. No SOTA or full-roadmap score follows.
