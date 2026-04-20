"use client";

/**
 * Chronological timeline of ``TimelineEvent`` records from the
 * :class:`EvidenceBundle`. Events are already sorted ascending by
 * the server-side builder.
 */

import type { TimelineEvent } from "@/lib/api";

type Props = {
  events: TimelineEvent[];
};

export default function TimelineView({ events }: Props) {
  if (!events.length) {
    return (
      <p style={{ color: "#64748b", fontStyle: "italic" }}>
        No dated events in this bundle.
      </p>
    );
  }
  return (
    <ol style={{ listStyle: "none", padding: 0, margin: 0 }}>
      {events.map((e, idx) => (
        <li
          key={`${e.source_id}-${idx}`}
          style={{
            padding: "12px 16px",
            borderLeft: "3px solid #3b82f6",
            marginBottom: 12,
            background: "#f8fafc",
            borderRadius: 4,
          }}
        >
          <div style={{ fontWeight: 600, color: "#1e293b" }}>{e.date}</div>
          <div style={{ color: "#475569", marginTop: 4 }}>{e.event}</div>
          <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 4 }}>
            source: <code>{e.source_id}</code>
          </div>
        </li>
      ))}
    </ol>
  );
}
