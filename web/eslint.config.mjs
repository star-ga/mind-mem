// Flat ESLint config for the web console.
//
// `next lint` is deprecated as of Next.js 15.5 (and removed in 16), and with
// no config file in the tree it stopped at an interactive "how would you like
// to configure ESLint?" prompt instead of linting. This file is what makes
// `npm run lint` a real, non-interactive check; CI runs it with
// --max-warnings=0.
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { FlatCompat } from "@eslint/eslintrc";

const compat = new FlatCompat({ baseDirectory: dirname(fileURLToPath(import.meta.url)) });

const config = [
  { ignores: [".next/**", "node_modules/**", "out/**", "next-env.d.ts"] },
  ...compat.extends("next/core-web-vitals", "next/typescript"),
];

export default config;
