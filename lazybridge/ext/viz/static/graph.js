// D3 force-directed graph with live mutation: nodes and edges
// can appear at runtime (tool nodes are discovered from events).
//
// Visual encoding per node type:
//   agent  → circle
//   tool   → hexagon
//   router → diamond
//
// Nodes are pinnable: end of drag → pinned (fx/fy fixed).
// Double-click to unpin. A pin icon appears on pinned nodes.

import { state, emit } from "/static/state.js";
import { installDefs } from "/static/graph-defs.js";
import { spawnPulse } from "/static/graph-pulse.js";

const NODE_R = 22;
// Precomputed hexagon path (flat-top, r=NODE_R)
const HEX_PATH = (() => {
  const pts = Array.from({ length: 6 }, (_, i) => {
    const a = (Math.PI / 180) * (60 * i + 30);
    return [NODE_R * Math.cos(a), NODE_R * Math.sin(a)];
  });
  return "M" + pts.map(p => p.join(",")).join("L") + "Z";
})();
// Diamond path for routers
const DIAMOND_PATH =
  `M0,${-NODE_R - 4} L${NODE_R + 4},0 L0,${NODE_R + 4} L${-NODE_R - 4},0 Z`;

let svg, gLinks, gNodes, gPulses, sim, nodeSel, linkSel;

export function initGraph() {
  const el = document.getElementById("canvas");
  const rect = el.getBoundingClientRect();
  svg = d3.select(el).attr("viewBox", `0 0 ${rect.width} ${rect.height}`);
  installDefs(svg);

  const root = svg.append("g").attr("class", "viewport");
  gLinks = root.append("g").attr("class", "links");
  gNodes = root.append("g").attr("class", "nodes");
  gPulses = root.append("g").attr("class", "pulses");

  svg.call(d3.zoom().scaleExtent([0.2, 4]).on("zoom", (ev) => {
    root.attr("transform", ev.transform);
  }));

  sim = d3.forceSimulation()
    .force("link", d3.forceLink().id(d => d.id).distance(160).strength(0.5))
    .force("charge", d3.forceManyBody().strength(-420))
    .force("center", d3.forceCenter(rect.width / 2, rect.height / 2))
    .force("collide", d3.forceCollide(NODE_R + 12))
    .on("tick", onTick);
}

export function setGraph(nodes, links) {
  state.nodes = nodes;
  state.links = links;
  redraw();
}

export function ensureToolNode(toolName, agentId) {
  const id = `tool:${toolName}`;
  if (!state.nodes.find(n => n.id === id)) {
    state.nodes.push({ id, name: toolName, type: "tool" });
  }
  if (!state.links.find(l => l.source === agentId && l.target === id && l.kind === "tool")) {
    state.links.push({ source: agentId, target: id, kind: "tool" });
  }
  redraw();
  return id;
}

function redraw() {
  linkSel = gLinks.selectAll(".link").data(state.links, l => keyLink(l));
  linkSel.exit().remove();
  linkSel = linkSel.enter()
    .append("line")
    .attr("class", l => `link ${l.kind || "tool"}`)
    .merge(linkSel);

  nodeSel = gNodes.selectAll(".node").data(state.nodes, n => n.id);
  nodeSel.exit().remove();

  const enter = nodeSel.enter().append("g")
    .attr("class", n => `node ${n.type}`)
    .attr("data-id", n => n.id)
    .call(d3.drag()
      .on("start", (ev, d) => {
        if (!ev.active) sim.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
      })
      .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
      .on("end", (ev, d) => {
        if (!ev.active) sim.alphaTarget(0);
        // Pin node in place — keep fx/fy set
        d._pinned = true;
        updatePinIcon(d);
      }));

  // Double-click unpins
  enter.on("dblclick", (ev, d) => {
    ev.stopPropagation();
    d.fx = null; d.fy = null; d._pinned = false;
    updatePinIcon(d);
    sim.alpha(0.3).restart();
  });

  // Hit ring (animation layer)
  enter.append("circle").attr("class", "hit-ring");

  // Shape per type
  enter.each(function(d) {
    const g = d3.select(this);
    if (d.type === "tool") {
      g.append("path").attr("class", "node-bg").attr("d", HEX_PATH);
    } else if (d.type === "router") {
      g.append("path").attr("class", "node-bg").attr("d", DIAMOND_PATH);
    } else {
      // agent (default) — circle
      g.append("circle").attr("class", "node-bg").attr("r", NODE_R);
    }
  });

  enter.append("text").attr("class", "node-label").attr("dy", 4)
    .text(n => truncate(n.name, 12));
  enter.append("text").attr("class", "node-sub").attr("dy", NODE_R + 14)
    .text(n => sublabel(n));

  // Pin icon (hidden by default)
  enter.append("text").attr("class", "pin-icon")
    .attr("dy", -NODE_R - 6).attr("text-anchor", "middle")
    .style("display", "none").text("📌");

  enter.on("click", (_ev, d) => emit("nodeClick", d));

  nodeSel = enter.merge(nodeSel);

  sim.nodes(state.nodes);
  sim.force("link").links(state.links);
  sim.alpha(0.6).restart();
}

function updatePinIcon(d) {
  const node = gNodes.select(`.node[data-id="${cssEscape(d.id)}"]`);
  node.classed("pinned", !!d._pinned);
  node.select(".pin-icon").style("display", d._pinned ? "" : "none");
}

function onTick() {
  if (!linkSel || !nodeSel) return;
  linkSel
    .attr("x1", l => l.source.x).attr("y1", l => l.source.y)
    .attr("x2", l => l.target.x).attr("y2", l => l.target.y);
  nodeSel.attr("transform", n => `translate(${n.x ?? 0},${n.y ?? 0})`);
}

function keyLink(l) {
  const a = typeof l.source === "object" ? l.source.id : l.source;
  const b = typeof l.target === "object" ? l.target.id : l.target;
  return `${a}->${b}|${l.kind || ""}`;
}

function truncate(s, n) {
  if (!s) return "";
  return s.length <= n ? s : s.slice(0, n - 1) + "…";
}

function sublabel(n) {
  if (n.type === "tool") return "tool";
  if (n.type === "router") return "router";
  return n.model ? truncate(n.model, 18) : (n.provider || "");
}

// ---- Animation API ---------------------------------------------------

export function pulse(srcId, dstId, color) {
  const src = state.nodes.find(n => n.id === srcId);
  const dst = state.nodes.find(n => n.id === dstId);
  if (!src || !dst) return;
  spawnPulse(gPulses, src, dst, color);
  fireHitRing(dstId, color);
}

export function fireHitRing(nodeId, color) {
  const node = gNodes.select(`.node[data-id="${cssEscape(nodeId)}"]`);
  if (node.empty()) return;
  const ring = node.select(".hit-ring");
  ring.attr("stroke", color || "#00d4ff").classed("fire", false);
  void ring.node().getBoundingClientRect();
  ring.classed("fire", true);
}

export function setInFlight(nodeId, on) {
  const node = gNodes.select(`.node[data-id="${cssEscape(nodeId)}"]`);
  if (node.empty()) return;
  node.classed("in-flight", !!on);
  if (on) state.inFlight.add(nodeId); else state.inFlight.delete(nodeId);
}

export function flashError(nodeId) {
  const node = gNodes.select(`.node[data-id="${cssEscape(nodeId)}"]`);
  if (node.empty()) return;
  node.classed("error", true);
  setTimeout(() => node.classed("error", false), 600);
}

function cssEscape(s) {
  return String(s).replace(/(["\\])/g, "\\$1");
}
