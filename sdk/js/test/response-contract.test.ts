// Copyright 2026 STARGA, Inc.
import { it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { MindMemClient } from "../src/client.js";

it("preserves actual REST envelopes, ranked hits and evidence", async () => {
  // Paths are resolved from sdk/js, the documented npm test working directory.
  const fixtures = Object.fromEntries(
    ["recall", "block", "health", "contradictions", "scan"].map(name => [
      name, JSON.parse(readFileSync(`../spec/fixtures/${name}.json`, "utf8")),
    ]),
  );
  const original = globalThis.fetch;
  const requested: Record<string, unknown>[] = [];
  globalThis.fetch = async (url, options) => {
    const path = new URL(String(url)).pathname;
    const name = path.startsWith("/v1/block/") ? "block" : path.split("/").at(-1)!;
    if (options?.body) requested.push(JSON.parse(String(options.body)));
    return new Response(JSON.stringify(fixtures[name]), { status: 200 });
  };
  try {
    const client = new MindMemClient("http://fixture.invalid");
    const recall = await client.recall("orchid", { scoringInstant: "2026-09-14" });
    assert.equal(recall.count, 1);
    assert.equal(recall.results[0]?._id, "D-20260914-001");
    assert.equal(recall.results[0]?.excerpt, "Orchid is the SDK contract sentinel.");
    assert.deepEqual(recall.attestation, fixtures["recall"].attestation);
    assert.equal(requested[0]?.["scoring_instant"], "2026-09-14");
    const block = await client.getBlock("D-20260914-001");
    assert.equal(block.found, true);
    assert.equal(block.block.Statement, "Orchid is the SDK contract sentinel.");
    const health = await client.health();
    assert.equal(health.workspace_exists, true);
    assert.ok(health.api_version);
    const contradictions = await client.listContradictions();
    assert.equal(contradictions.contradictions, 0);
    assert.equal(contradictions.status, "clean");
    const scan = await client.scan();
    assert.deepEqual(scan.checks["decisions"], { total: 1, active: 1 });
  } finally {
    globalThis.fetch = original;
  }
});


it("represents anonymous health without private workspace fields", async () => {
  const fixture = JSON.parse(readFileSync("../spec/fixtures/health_public.json", "utf8"));
  const original = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify(fixture), { status: 200 });
  try {
    const health = await new MindMemClient("http://fixture.invalid").health();
    assert.ok(health.api_version);
    assert.equal(health.workspace, undefined);
    assert.equal(health.workspace_exists, undefined);
  } finally {
    globalThis.fetch = original;
  }
});
