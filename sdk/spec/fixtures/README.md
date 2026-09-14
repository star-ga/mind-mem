# REST response contract fixtures

These five JSON envelopes were captured from the actual authenticated REST
handlers against a temporary workspace containing one synthetic Orchid fact.
The health response's temporary workspace path is replaced by
`<fixture-workspace>`. No production memories, credentials or evidence are
included. Receipt hashes describe the temporary fixture, not a release proof.

`tests/test_sdk_response_contract.py` calls those handlers again and checks
the captured field types, nonempty recall/direct-fetch content and counts.
Dynamic request IDs, durations, paths and hashes are not pinned as constants.
Go and TypeScript client tests consume these same envelopes. Go's copy lives
inside its module at `testdata/contract`, with byte equality checked by the
Python gate, so a downloaded module can run its own tests.

Response changes require updating the clients, fixtures and this live-handler
gate together. Route conformance alone cannot establish response conformance.
