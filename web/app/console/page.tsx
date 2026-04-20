"use client";

/**
 * Managed-service console — per-tenant dashboard (v4.0 prep).
 *
 * Extends the single-page graph view with a tenant switcher, a
 * usage meter, and a per-tenant audit-chain verification badge.
 * Reads the tenant list from ``GET /v1/admin/tenants`` (v4.0 admin
 * endpoint — pending in the REST layer) and filters every other
 * call by ``X-Tenant-Id`` via ``withTenantHeader``.
 */

import { useEffect, useState } from "react";

import TenantSwitcher, { type Tenant, withTenantHeader } from "@/components/TenantSwitcher";

type TenantVerification = {
  tenant_id: string;
  verified: boolean;
  records: number;
  genesis: string;
};

const API_BASE =
  (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_MIND_MEM_API_URL) ||
  "http://127.0.0.1:8080";

export default function ConsolePage() {
  const [tenants, setTenants] = useState<Tenant[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [verification, setVerification] = useState<TenantVerification | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/v1/admin/tenants`);
        if (!res.ok) {
          throw new Error(`/v1/admin/tenants returned ${res.status}`);
        }
        const data = (await res.json()) as { tenants: Tenant[] };
        if (!cancelled) setTenants(data.tenants ?? []);
      } catch (err) {
        if (!cancelled) {
          // Tolerate missing endpoint — show single-tenant UI.
          setError(err instanceof Error ? err.message : String(err));
          setTenants([]);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    setLoading(true);
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/v1/admin/tenants/${selected}/verify`, {
          headers: withTenantHeader({ "Content-Type": "application/json" }),
        });
        if (!res.ok) throw new Error(`verify returned ${res.status}`);
        const data = (await res.json()) as TenantVerification;
        if (!cancelled) setVerification(data);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selected]);

  return (
    <main style={{ maxWidth: 1200, margin: "0 auto", padding: "24px 24px 48px" }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 26 }}>mind-mem managed console</h1>
          <p style={{ color: "#64748b", margin: "4px 0 0" }}>
            Per-tenant governance + audit chain verification.
          </p>
        </div>
        <TenantSwitcher tenants={tenants} onChange={setSelected} />
      </header>

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

      {selected ? (
        <section
          style={{
            padding: 20,
            background: "#ffffff",
            border: "1px solid #e2e8f0",
            borderRadius: 8,
          }}
        >
          <h2 style={{ marginTop: 0, fontSize: 20 }}>Audit chain verification</h2>
          {loading ? (
            <p>Verifying…</p>
          ) : verification ? (
            <dl style={{ display: "grid", gridTemplateColumns: "180px 1fr", gap: 8 }}>
              <dt>Tenant ID</dt>
              <dd>
                <code>{verification.tenant_id}</code>
              </dd>
              <dt>Status</dt>
              <dd style={{ color: verification.verified ? "#15803d" : "#b91c1c", fontWeight: 600 }}>
                {verification.verified ? "verified ✓" : "FAILED"}
              </dd>
              <dt>Records</dt>
              <dd>{verification.records}</dd>
              <dt>Genesis</dt>
              <dd>
                <code style={{ fontSize: 12 }}>{verification.genesis}</code>
              </dd>
            </dl>
          ) : (
            <p style={{ color: "#64748b" }}>No verification data.</p>
          )}
        </section>
      ) : (
        <p style={{ color: "#64748b" }}>Pick a tenant to begin.</p>
      )}
    </main>
  );
}
