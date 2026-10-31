# Public benchmark inventory

This kit reconstructs historical prompts and replays saved answers. It does
not constitute a fair new-release measurement. See
[the kit guide](benchmarks/public_release/README.md) for commands, limitations,
licenses, and the full/dev/held-out interpretation.

The shipped source text is limited to 96 corrected-key evidence records.
LoCoMo and LongMemEval are downloaded from immutable URLs with SHA-256 checks.
Question text and unmodified reference answers are loaded only from that cache.
Retrieval source IDs and hashes rebuild prompts; saved answer and verdict rows
carry no questions or conversation fields. The earlier mixed-selection key,
unchanged audit rows and excluded candidates are not shipped.

Official categories 1–4: 1002/1540, dev 531/802, held-out 471/738.
Corrected historical replay: 1003/1540, dev 536/802, held-out 467/738;
verification pending. The saved category-5 prompts exposed the dataset label.
Earlier development inspected both splits. No untouched benchmark has run.
The historical judge revision is unavailable and adjudications have not been
independently verified; replay is not rejudging. A new corrected-score claim
requires a pinned judge. The reference-answer configuration is unmeasured.

No historical timing, kernel or extraction-throughput claim is validated by
this kit. Its fresh runner uses the product's label-free answering boundary;
its historical renderer is retained solely to reconstruct archived inputs.
