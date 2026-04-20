"use client";

import type { EvidenceBundle } from "@/lib/api";

type Props = {
  facts: EvidenceBundle["facts"];
};

export default function FactList({ facts }: Props) {
  if (!facts.length) {
    return <p style={{ color: "#64748b", fontStyle: "italic" }}>No facts extracted.</p>;
  }
  return (
    <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
      {facts.map((f, i) => (
        <li
          key={`${f.source_id}-${i}`}
          style={{
            padding: 12,
            marginBottom: 8,
            background: "#ffffff",
            border: "1px solid #e2e8f0",
            borderRadius: 4,
          }}
        >
          <div style={{ color: "#1e293b" }}>{f.claim}</div>
          <div
            style={{
              fontSize: 12,
              color: "#64748b",
              marginTop: 6,
              display: "flex",
              gap: 12,
            }}
          >
            <span>
              source: <code>{f.source_id}</code>
            </span>
            <span>confidence: {(f.confidence * 100).toFixed(0)}%</span>
          </div>
        </li>
      ))}
    </ul>
  );
}
