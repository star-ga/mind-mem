/**
 * Mind Mem hook handler for OpenClaw
 *
 * - agent:bootstrap  → injects health summary into bootstrap context
 * - command:new      → runs capture.py to extract session signals
 */

import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

// Resolve MIND_MEM_WORKSPACE from hook config env, process env, or default
function resolveWorkspace(event: any): string {
  const hookEnv = event.context?.cfg?.hooks?.internal?.entries?.["mind-mem"]?.env;
  return (
    hookEnv?.MIND_MEM_WORKSPACE ||
    process.env.MIND_MEM_WORKSPACE ||
    "."
  );
}

// Find mind-mem scripts directory (relative to workspace or standard locations)
function resolveScriptsDir(workspace: string): string | null {
  // Check .mind-mem/scripts/ (cloned into project)
  const dotMemOs = path.join(workspace, ".mind-mem", "scripts");
  if (fs.existsSync(dotMemOs)) return dotMemOs;

  // Check mind-mem/scripts/ in workspace parent
  const parentMemOs = path.join(path.dirname(workspace), "mind-mem", "scripts");
  if (fs.existsSync(parentMemOs)) return parentMemOs;

  // Check MIND_MEM_HOME env var
  const memOsHome = process.env.MIND_MEM_HOME;
  if (memOsHome) {
    const homeScripts = path.join(memOsHome, "scripts");
    if (fs.existsSync(homeScripts)) return homeScripts;
  }

  return null;
}

const handler = async (event: any): Promise<void> => {
  const workspace = resolveWorkspace(event);

  if (event.type === "agent" && event.action === "bootstrap") {
    // Inject health summary into bootstrap context
    const statePath = path.join(workspace, "memory", "intel-state.json");
    try {
      if (!fs.existsSync(statePath)) return;

      const raw = fs.readFileSync(statePath, "utf-8");
      const state = JSON.parse(raw);
      const mode = state.self_correcting_mode || state.governance_mode || "unknown";
      const lastScan = state.last_scan || "never";
      const contradictions = state.counters?.contradictions_open || 0;
      const drift = state.counters?.drift_signals_open || 0;

      const summary = `mind-mem health: mode=${mode} last_scan=${lastScan} contradictions=${contradictions} drift=${drift}`;

      // Push into bootstrap files if context supports it
      if (event.context?.bootstrapFiles && Array.isArray(event.context.bootstrapFiles)) {
        event.context.bootstrapFiles.push({
          path: "mind-mem-health",
          content: summary,
          type: "system",
        });
      } else {
        event.messages.push(summary);
      }
    } catch {
      // Non-fatal — workspace may not be initialized
    }
    return;
  }

  if (event.type === "command" && event.action === "new") {
    // Run capture.py on session end (/new)
    const scriptsDir = resolveScriptsDir(workspace);
    if (!scriptsDir) return;

    const capturePy = path.join(scriptsDir, "capture.py");
    if (!fs.existsSync(capturePy)) return;

    try {
      execSync(`python3 "${capturePy}" "${workspace}"`, {
        timeout: 10_000,
        stdio: "pipe",
        env: { ...process.env, MIND_MEM_WORKSPACE: workspace },
      });
    } catch {
      // Non-fatal — capture failures shouldn't block /new
    }
    return;
  }
};

export default handler;
