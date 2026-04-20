"use client";

import { useCallback, useState } from "react";

import FactList from "@/components/FactList";
import GraphView from "@/components/GraphView";
import TimelineView from "@/components/TimelineView";
import { recallBundle } from "@/lib/api";
import type { EvidenceBundle, MindMemBlock } from "@/lib/api";

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [bundle, setBundle] = useState<EvidenceBundle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<MindMemBlock | null>(null);

  const runQuery = useCallback(async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setSelected(null);
    try {
      const b = await recallBundle(query, { limit: 18 });
      setBundle(b);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [query]);

  return (
    <main style={{ maxWidth: 1400, margin: "0 auto", padding: "24px 24px 64px" }}>
      <header style={{ marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 28 }}>mind-mem governance console</h1>
        <p style={{ color: "#475569", margin: "4px 0 0" }}>
          Graph + timeline + facts over the mind-mem REST API (v3.2.0+).
        </p>
      </header>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void runQuery();
        }}
        style={{ display: "flex", gap: 8, marginBottom: 24 }}
      >
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask the memory something — e.g. 'PostgreSQL decision history'"
          style={{
            flex: 1,
            padding: "10px 14px",
            fontSize: 15,
            border: "1px solid #cbd5e1",
            borderRadius: 6,
            background: "#ffffff",
          }}
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          style={{
            padding: "10px 20px",
            fontSize: 15,
            border: "none",
            borderRadius: 6,
            background: loading ? "#94a3b8" : "#2563eb",
            color: "#ffffff",
            cursor: loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Querying…" : "Recall"}
        </button>
      </form>

      {error && (
        <div
          style={{
            padding: 12,
            marginBottom: 16,
            background: "#fef2f2",
            border: "1px solid #fecaca",
            borderRadius: 6,
            color: "#991b1b",
          }}
        >
          {error}
        </div>
      )}

      {bundle && (
        <section
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0, 2fr) minmax(0, 1fr)",
            gap: 24,
            alignItems: "start",
          }}
        >
          <div>
            <h2 style={{ fontSize: 20, marginTop: 0 }}>Graph</h2>
            <GraphView bundle={bundle} width={900} height={540} onSelect={setSelected} />
            <h2 style={{ fontSize: 20, marginTop: 24 }}>Timeline</h2>
            <TimelineView events={bundle.timeline} />
          </div>
          <aside>
            <h2 style={{ fontSize: 20, marginTop: 0 }}>Facts</h2>
            <FactList facts={bundle.facts} />
            {selected && (
              <div
                style={{
                  marginTop: 24,
                  padding: 16,
                  background: "#eff6ff",
                  border: "1px solid #bfdbfe",
                  borderRadius: 6,
                }}
              >
                <div style={{ fontSize: 13, color: "#1e40af", marginBottom: 8 }}>
                  selected: <code>{selected._id}</code>
                </div>
                <div style={{ fontSize: 14 }}>{selected.Statement ?? "(no statement field)"}</div>
              </div>
            )}
          </aside>
        </section>
      )}
    </main>
  );
}
