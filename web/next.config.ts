import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // mind-mem-web is a thin client — the REST API lives on the
  // mind-mem process (default 127.0.0.1:8080). Set
  // NEXT_PUBLIC_MIND_MEM_API_URL to point elsewhere.
  reactStrictMode: true,
  eslint: {
    // CI runs ``next lint`` separately; keep builds unblocked.
    ignoreDuringBuilds: false,
  },
};

export default nextConfig;
