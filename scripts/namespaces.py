#!/usr/bin/env python3
"""mind-mem Multi-Agent Namespace & ACL Engine. Zero external deps.

Provides:
- Workspace-level + per-agent namespaces
- Simple JSON-based ACL (read/write permissions per agent pattern)
- Namespace-aware path resolution
- Shared ledger for cross-agent fact propagation

Namespace structure:
    workspace/
    ├── shared/                    # Visible to all agents
    │   ├── decisions/
    │   ├── tasks/
    │   └── entities/
    ├── agents/
    │   ├── <agent-id>/            # Private to this agent
    │   │   ├── decisions/
    │   │   ├── tasks/
    │   │   └── memory/
    │   └── <agent-id>/
    └── mind-mem-acl.json            # Access control

ACL format (mind-mem-acl.json):
    {
        "default_policy": "read",
        "agents": {
            "coder-1": {"namespaces": ["shared", "agents/coder-1"], "write": ["agents/coder-1"], "read": ["shared"]},
            "reviewer-*": {"namespaces": ["shared"], "write": [], "read": ["shared"]},
            "*": {"namespaces": ["shared"], "write": [], "read": ["shared"]}
        }
    }

Usage:
    from namespaces import NamespaceManager
    ns = NamespaceManager(workspace, agent_id="coder-1")
    ns.can_write("agents/coder-1/decisions/DECISIONS.md")  # True
    ns.can_write("shared/decisions/DECISIONS.md")            # False (read-only)
    ns.resolve_paths("decisions/DECISIONS.md")               # Returns paths in all accessible namespaces
"""

from __future__ import annotations

import fnmatch
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mind_filelock import FileLock
from observability import get_logger

_log = get_logger("namespaces")

# Standard directories that get replicated per namespace
NAMESPACE_DIRS = [
    "decisions",
    "tasks",
    "entities",
    "memory",
    "intelligence",
]

DEFAULT_ACL = {
    "default_policy": "read",
    "agents": {
        "*": {
            "namespaces": ["shared"],
            "write": [],
            "read": ["shared"],
        }
    },
}


class NamespaceManager:
    """Manages multi-agent namespaces and access control.

    Args:
        workspace: Root workspace path.
        agent_id: Current agent's identifier. None = workspace-level access (all perms).
    """

    def __init__(self, workspace: str, agent_id: str | None = None) -> None:
        self.workspace = os.path.abspath(workspace)
        self.agent_id = agent_id
        self._acl = self._load_acl()
        self._agent_policy = self._resolve_agent_policy()

    def _load_acl(self) -> dict:
        """Load ACL from workspace or return defaults."""
        acl_path = os.path.join(self.workspace, "mind-mem-acl.json")
        if os.path.isfile(acl_path):
            try:
                with open(acl_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                _log.warning("acl_load_failed", error=str(e))
        return DEFAULT_ACL

    def _resolve_agent_policy(self) -> dict:
        """Find the matching policy for the current agent_id."""
        if self.agent_id is None:
            # No agent_id = full access (workspace-level)
            return {
                "namespaces": ["shared", "agents/*"],
                "write": ["shared", "agents/*"],
                "read": ["shared", "agents/*"],
            }

        agents = self._acl.get("agents", {})

        # Exact match first
        if self.agent_id in agents:
            return agents[self.agent_id]

        # Pattern match (e.g., "reviewer-*" matches "reviewer-1")
        for pattern, policy in agents.items():
            if pattern != "*" and fnmatch.fnmatch(self.agent_id, pattern):
                return policy

        # Wildcard fallback
        if "*" in agents:
            return agents["*"]

        # Ultimate fallback: read-only shared
        return {"namespaces": ["shared"], "write": [], "read": ["shared"]}

    def can_read(self, rel_path: str) -> bool:
        """Check if current agent can read a path."""
        if self.agent_id is None:
            return True
        read_ns = self._agent_policy.get("read", [])
        return self._path_matches_namespaces(rel_path, read_ns)

    def can_write(self, rel_path: str) -> bool:
        """Check if current agent can write to a path."""
        if self.agent_id is None:
            return True
        write_ns = self._agent_policy.get("write", [])
        return self._path_matches_namespaces(rel_path, write_ns)

    def _path_matches_namespaces(self, rel_path: str, namespaces: list[str]) -> bool:
        """Check if a relative path falls under any of the given namespaces."""
        normalized = rel_path.replace("\\", "/").lstrip("/")
        for ns in namespaces:
            ns_normalized = ns.replace("\\", "/").rstrip("/")
            if normalized.startswith(ns_normalized + "/") or normalized == ns_normalized:
                return True
            # Support glob patterns like "agents/*"
            if "*" in ns_normalized:
                parts = normalized.split("/")
                ns_parts = ns_normalized.split("/")
                if len(parts) >= len(ns_parts):
                    prefix = "/".join(parts[:len(ns_parts)])
                    if fnmatch.fnmatch(prefix, ns_normalized):
                        return True
        return False

    def resolve_corpus_paths(self, rel_path: str) -> list[str]:
        """Resolve a relative path to all accessible copies across namespaces.

        For recall: returns absolute paths to the file in each namespace
        the agent can read.

        Example:
            resolve_corpus_paths("decisions/DECISIONS.md")
            → ["/ws/shared/decisions/DECISIONS.md", "/ws/agents/coder-1/decisions/DECISIONS.md"]
        """
        paths = []
        accessible = self._agent_policy.get("namespaces", [])

        for ns in accessible:
            if "*" in ns:
                # Expand wildcard namespaces (e.g., "agents/*")
                ns_dir = os.path.join(self.workspace, os.path.dirname(ns))
                if os.path.isdir(ns_dir):
                    for entry in sorted(os.listdir(ns_dir)):
                        full_ns = os.path.join(os.path.dirname(ns), entry)
                        candidate = os.path.join(self.workspace, full_ns, rel_path)
                        if os.path.isfile(candidate) and self.can_read(os.path.join(full_ns, rel_path)):
                            paths.append(candidate)
            else:
                candidate = os.path.join(self.workspace, ns, rel_path)
                if os.path.isfile(candidate) and self.can_read(os.path.join(ns, rel_path)):
                    paths.append(candidate)

        return paths

    def init_namespace(self, namespace: str) -> list[str]:
        """Initialize a namespace directory structure. Returns list of created dirs."""
        created = []
        ns_root = os.path.join(self.workspace, namespace)
        for d in NAMESPACE_DIRS:
            path = os.path.join(ns_root, d)
            if not os.path.isdir(path):
                os.makedirs(path, exist_ok=True)
                created.append(path)
        return created

    def init_agent(self, agent_id: str) -> list[str]:
        """Initialize namespace for a new agent. Returns created dirs."""
        return self.init_namespace(f"agents/{agent_id}")

    def list_agents(self) -> list[str]:
        """List all agent IDs that have namespaces."""
        agents_dir = os.path.join(self.workspace, "agents")
        if not os.path.isdir(agents_dir):
            return []
        return sorted(
            d for d in os.listdir(agents_dir)
            if os.path.isdir(os.path.join(agents_dir, d)) and not d.startswith(".")
        )

    def get_agent_namespace(self) -> str | None:
        """Get the namespace path for the current agent."""
        if self.agent_id is None:
            return None
        return f"agents/{self.agent_id}"


class SharedLedger:
    """Append-only cross-agent fact ledger.

    High-confidence facts can be proposed to the shared ledger,
    where they become visible to all agents after review.

    Location: workspace/shared/intelligence/LEDGER.md
    """

    def __init__(self, workspace: str) -> None:
        self.workspace = os.path.abspath(workspace)
        self.ledger_path = os.path.join(workspace, "shared", "intelligence", "LEDGER.md")

    def append_fact(self, fact: dict, source_agent: str) -> bool:
        """Append a fact to the shared ledger.

        Args:
            fact: Dict with keys: text, confidence, type, source_block
            source_agent: Agent ID that originated the fact

        Returns:
            True if appended, False if ledger doesn't exist
        """
        ledger_dir = os.path.dirname(self.ledger_path)
        if not os.path.isdir(ledger_dir):
            return False

        # Dedup check
        if os.path.isfile(self.ledger_path):
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                existing = f.read()
            if fact.get("text", "")[:80] in existing:
                return False

        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        with FileLock(self.ledger_path):
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(f"\n[FACT-{datetime.now().strftime('%Y%m%d-%H%M%S')}]\n")
                f.write(f"Date: {ts}\n")
                f.write(f"Source: {source_agent}\n")
                f.write(f"Confidence: {fact.get('confidence', 'medium')}\n")
                f.write(f"Type: {fact.get('type', 'observation')}\n")
                f.write(f"Text: {fact.get('text', '')}\n")
                if fact.get("source_block"):
                    f.write(f"SourceBlock: {fact['source_block']}\n")
                f.write("Status: pending-review\n")
                f.write("\n---\n")

        _log.info("fact_appended", agent=source_agent, confidence=fact.get("confidence"))
        return True

    def get_facts(self, status: str | None = None) -> list[dict]:
        """Read facts from the ledger, optionally filtered by status."""
        if not os.path.isfile(self.ledger_path):
            return []

        from block_parser import parse_file
        blocks = parse_file(self.ledger_path)
        if status:
            blocks = [b for b in blocks if b.get("Status") == status]
        return blocks


def init_multi_agent_workspace(workspace: str, agents: list[str] | None = None) -> dict:
    """Initialize a multi-agent workspace with shared + agent namespaces.

    Args:
        workspace: Root workspace path
        agents: List of agent IDs to create namespaces for

    Returns:
        Summary dict with created paths
    """
    ns = NamespaceManager(workspace)
    result = {"shared": [], "agents": {}}

    # Init shared namespace
    result["shared"] = ns.init_namespace("shared")

    # Init shared ledger
    ledger_dir = os.path.join(workspace, "shared", "intelligence")
    os.makedirs(ledger_dir, exist_ok=True)
    ledger_path = os.path.join(ledger_dir, "LEDGER.md")
    if not os.path.isfile(ledger_path):
        with open(ledger_path, "w", encoding="utf-8") as f:
            f.write("# Shared Fact Ledger\n\nCross-agent facts pending review.\n\n")
        result["shared"].append(ledger_path)

    # Init agent namespaces
    if agents:
        for agent_id in agents:
            result["agents"][agent_id] = ns.init_agent(agent_id)

    # Write default ACL if none exists
    acl_path = os.path.join(workspace, "mind-mem-acl.json")
    if not os.path.isfile(acl_path):
        acl = dict(DEFAULT_ACL)
        if agents:
            for agent_id in agents:
                acl["agents"][agent_id] = {
                    "namespaces": ["shared", f"agents/{agent_id}"],
                    "write": [f"agents/{agent_id}"],
                    "read": ["shared"],
                }
        with open(acl_path, "w", encoding="utf-8") as f:
            json.dump(acl, f, indent=2)
        result["acl"] = acl_path

    _log.info("multi_agent_init", agents=agents or [], shared_dirs=len(result["shared"]))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="mind-mem Namespace Manager")
    parser.add_argument("workspace", nargs="?", default=".")
    parser.add_argument("--init", nargs="*", metavar="AGENT_ID",
                        help="Initialize multi-agent workspace with given agent IDs")
    parser.add_argument("--list-agents", action="store_true", help="List registered agents")
    parser.add_argument("--check", nargs=2, metavar=("AGENT_ID", "PATH"),
                        help="Check if agent can access path")
    args = parser.parse_args()

    ws = os.path.abspath(args.workspace)

    if args.init is not None:
        result = init_multi_agent_workspace(ws, args.init or None)
        print(f"Initialized multi-agent workspace: {ws}")
        if result.get("shared"):
            print(f"  Shared: {len(result['shared'])} dir(s) created")
        for agent_id, dirs in result.get("agents", {}).items():
            print(f"  Agent '{agent_id}': {len(dirs)} dir(s) created")
        if result.get("acl"):
            print(f"  ACL: {result['acl']}")

    elif args.list_agents:
        ns = NamespaceManager(ws)
        agents = ns.list_agents()
        if agents:
            print(f"Registered agents ({len(agents)}):")
            for a in agents:
                print(f"  - {a}")
        else:
            print("No agents registered. Use --init to set up multi-agent workspace.")

    elif args.check:
        agent_id, path = args.check
        ns = NamespaceManager(ws, agent_id=agent_id)
        can_r = ns.can_read(path)
        can_w = ns.can_write(path)
        print(f"Agent '{agent_id}' → {path}:")
        print(f"  Read:  {'ALLOW' if can_r else 'DENY'}")
        print(f"  Write: {'ALLOW' if can_w else 'DENY'}")
