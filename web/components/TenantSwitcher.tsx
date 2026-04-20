"use client";

/**
 * Tenant switcher for the managed-service console (v4.0 prep).
 *
 * Paired with the tenant_audit.py + tenant_kms.py backends. When the
 * operator runs mind-mem in multi-tenant mode, this dropdown picks
 * which tenant's governance surface the console shows.
 *
 * The selected tenant ID is persisted in localStorage and passed as
 * an X-Tenant-Id header on every API call — the REST layer's tenant
 * middleware (v4.0) reads it and routes to the right per-tenant
 * shard / chain / DEK.
 */

import { useEffect, useState } from "react";

const STORAGE_KEY = "mind-mem-tenant";

export type Tenant = {
  id: string;
  name: string;
  spec_hash?: string;
  records?: number;
};

type Props = {
  tenants: Tenant[];
  onChange?: (tenantId: string) => void;
};

export default function TenantSwitcher({ tenants, onChange }: Props) {
  const [current, setCurrent] = useState<string | null>(null);

  useEffect(() => {
    const stored = typeof window !== "undefined" ? window.localStorage.getItem(STORAGE_KEY) : null;
    if (stored && tenants.some((t) => t.id === stored)) {
      setCurrent(stored);
      onChange?.(stored);
    } else if (tenants.length > 0) {
      const first = tenants[0].id;
      setCurrent(first);
      onChange?.(first);
    }
  }, [tenants, onChange]);

  const handleChange = (tenantId: string) => {
    setCurrent(tenantId);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(STORAGE_KEY, tenantId);
    }
    onChange?.(tenantId);
  };

  if (tenants.length === 0) {
    return (
      <div style={{ color: "#94a3b8", fontSize: 13 }}>
        No tenants configured (single-tenant mode).
      </div>
    );
  }

  return (
    <div style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
      <label
        htmlFor="mm-tenant-switcher"
        style={{ fontSize: 13, color: "#475569", fontWeight: 500 }}
      >
        Tenant
      </label>
      <select
        id="mm-tenant-switcher"
        value={current ?? ""}
        onChange={(e) => handleChange(e.target.value)}
        style={{
          padding: "6px 10px",
          fontSize: 14,
          border: "1px solid #cbd5e1",
          borderRadius: 4,
          background: "#ffffff",
          minWidth: 160,
        }}
      >
        {tenants.map((t) => (
          <option key={t.id} value={t.id}>
            {t.name}
            {typeof t.records === "number" ? ` (${t.records} records)` : ""}
          </option>
        ))}
      </select>
    </div>
  );
}

/**
 * Attach the current tenant id as an X-Tenant-Id header.
 *
 * Usage::
 *
 *     const res = await fetch(url, {
 *       headers: withTenantHeader({ "Content-Type": "application/json" }),
 *       ...
 *     });
 */
export function withTenantHeader(init: HeadersInit = {}): HeadersInit {
  if (typeof window === "undefined") return init;
  const tid = window.localStorage.getItem(STORAGE_KEY);
  if (!tid) return init;
  // HeadersInit can be a Headers, a [string, string][], or a Record.
  // Normalise to a plain record so callers can spread.
  const base: Record<string, string> = {};
  if (init instanceof Headers) {
    init.forEach((v, k) => (base[k] = v));
  } else if (Array.isArray(init)) {
    for (const [k, v] of init) base[k] = v;
  } else {
    Object.assign(base, init);
  }
  base["X-Tenant-Id"] = tid;
  return base;
}
