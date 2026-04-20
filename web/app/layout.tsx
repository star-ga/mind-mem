import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "mind-mem — governance console",
  description: "Graph + timeline + facts over the mind-mem REST API",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          padding: 0,
          fontFamily:
            "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Liberation Sans', sans-serif",
          background: "#f1f5f9",
          color: "#0f172a",
        }}
      >
        {children}
      </body>
    </html>
  );
}
