import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // mind-mem-web is a thin client — the REST API lives on the
  // mind-mem process (default 127.0.0.1:8080). Set
  // NEXT_PUBLIC_MIND_MEM_API_URL to point elsewhere.
  reactStrictMode: true,
  eslint: {
    // `next build` lints too, so a lint error fails the build. CI also runs
    // `npm run lint` (eslint --max-warnings=0) as its own step.
    ignoreDuringBuilds: false,
  },
};

export default nextConfig;
