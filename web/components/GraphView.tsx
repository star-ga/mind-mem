"use client";

/**
 * Force-directed graph of blocks + their cross-references.
 *
 * Nodes come from the ``source_blocks`` in an :class:`EvidenceBundle`;
 * edges come from ``relations`` (supersedes / depends_on / cites /
 * tested_by / derived_from / relates_to / superseded_by).
 *
 * Uses d3-force directly — no react-flow dependency needed for the
 * default view. react-flow is imported lazily from the timeline /
 * drift views so the graph page stays small.
 */

import { useEffect, useRef } from "react";
import * as d3 from "d3";

import type { EvidenceBundle, MindMemBlock, Relation } from "@/lib/api";

type Props = {
  bundle: EvidenceBundle;
  width?: number;
  height?: number;
  onSelect?: (block: MindMemBlock) => void;
};

type SimNode = MindMemBlock & d3.SimulationNodeDatum;
type SimLink = { source: string | SimNode; target: string | SimNode; predicate: string } & d3.SimulationLinkDatum<SimNode>;

function colorForPredicate(predicate: string): string {
  return (
    {
      supersedes: "#c0392b",
      superseded_by: "#c0392b",
      depends_on: "#2980b9",
      relates_to: "#27ae60",
      cites: "#8e44ad",
      tested_by: "#f39c12",
      derived_from: "#16a085",
    }[predicate] ?? "#7f8c8d"
  );
}

function colorForStatus(status?: string): string {
  return (
    {
      verified: "#2ecc71",
      active: "#3498db",
      superseded: "#95a5a6",
      draft: "#f1c40f",
      deprecated: "#e74c3c",
    }[(status ?? "").toLowerCase()] ?? "#34495e"
  );
}

export default function GraphView({ bundle, width = 960, height = 600, onSelect }: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const nodes: SimNode[] = bundle.source_blocks.map((b) => ({ ...b }));
    const byId = new Map(nodes.map((n) => [n._id, n]));
    const links: SimLink[] = bundle.relations
      .filter((r) => byId.has(r.subject) && byId.has(r.object))
      .map((r) => ({ source: r.subject, target: r.object, predicate: r.predicate }));

    const sim = d3
      .forceSimulation<SimNode, SimLink>(nodes)
      .force(
        "link",
        d3.forceLink<SimNode, SimLink>(links).id((d) => d._id).distance(90),
      )
      .force("charge", d3.forceManyBody().strength(-220))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide().radius(22));

    const link = svg
      .append("g")
      .attr("stroke-opacity", 0.7)
      .selectAll<SVGLineElement, SimLink>("line")
      .data(links)
      .join("line")
      .attr("stroke", (d) => colorForPredicate(d.predicate))
      .attr("stroke-width", 1.5);

    const node = svg
      .append("g")
      .selectAll<SVGGElement, SimNode>("g")
      .data(nodes)
      .join("g")
      .attr("cursor", "pointer")
      .on("click", (_, d) => onSelect?.(d));

    node
      .append("circle")
      .attr("r", 14)
      .attr("fill", (d) => colorForStatus(d.Status))
      .attr("stroke", "#ffffff")
      .attr("stroke-width", 2);

    node
      .append("text")
      .text((d) => d._id.replace(/^(D|T|PER|PRJ|TOOL|INC)-/, ""))
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "central")
      .attr("font-size", 9)
      .attr("fill", "#fff");

    node.append("title").text((d) => [
      d._id,
      d.Statement ?? "",
      d.Status ? `Status: ${d.Status}` : "",
      typeof d.truth_score === "number" ? `truth: ${d.truth_score.toFixed(2)}` : "",
    ].filter(Boolean).join("\n"));

    sim.on("tick", () => {
      link
        .attr("x1", (d) => (typeof d.source === "object" ? d.source.x ?? 0 : 0))
        .attr("y1", (d) => (typeof d.source === "object" ? d.source.y ?? 0 : 0))
        .attr("x2", (d) => (typeof d.target === "object" ? d.target.x ?? 0 : 0))
        .attr("y2", (d) => (typeof d.target === "object" ? d.target.y ?? 0 : 0));
      node.attr("transform", (d) => `translate(${d.x ?? 0}, ${d.y ?? 0})`);
    });

    return () => {
      sim.stop();
    };
  }, [bundle, width, height, onSelect]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      style={{ background: "#0f172a", borderRadius: 8 }}
    />
  );
}
