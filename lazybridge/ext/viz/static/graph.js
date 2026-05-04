// D3 force-directed graph with live mutation.
//
// Visual encoding per node type:
//   agent  → 3-D parallelepiped (box with top + right face)
//   tool   → hexagon
//   router → diamond
//
// Drag to pin a node (📌); double-click to unpin.

import { state, emit } from "/static/state.js";
import { installDefs } from "/static/graph-defs.js";
import { spawnPulse } from "/static/graph-pulse.js";

// --- geometry constants ---------------------------------------------------

const NODE_R = 22;          // radius for tools / routers and collision

const BOX = { w: 60, h: 32, d: 10 }; // agent box (front face + 3-D offset)
const AGENT_COLL_R = BOX.w / 2 + BOX.d + 14; // collision radius for agents

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
  `M0,${-(NODE_R + 4)} L${NODE_R + 4},0 L0,${NODE_R + 4} L${-(NODE_R + 4)},0 Z`;

// Returns {front, top, right} polygon point-strings for an agent box
function boxFaces() {
  const hw = BOX.w / 2, hh = BOX.h / 2, d = BOX.d;
  const p = (pts) => pts.map(([x, y]) => `${x},${y}`).join(" ");
  const ftl = [-hw, -hh], ftr = [hw, -hh];
  const fbl = [-hw,  hh], fbr = [hw,  hh];
  const btl = [-hw + d, -hh - d], btr = [hw + d, -hh - d];
  const bbr = [ hw + d,  hh - d];
  return {
    front: p([ftl, ftr, fbr, fbl]),
    top:   p([ftl, ftr, btr, btl]),
    right: p([ftr, fbr, bbr, btr]),
  };
}

// -------------------------------------------------------------------------

let svg, gLinks, gNodes, gPulses, sim, nodeSel, linkSel;

export function initGraph() {
  const el   = document.getElementById("canvas");
  const rect = el.getBoundingClientRect();
  svg = d3.select(el).attr("viewBox", `0 0 ${rect.width} ${rect.height}`);
  installDefs(svg);

  const root = svg.append("g").attr("class", "viewport");
  gLinks  = root.append("g").attr("class", "links");
  gNodes  = root.append("g").attr("class", "nodes");
  gPulses = root.append("g").attr("class", "pulses");

  svg.call(d3.zoom().scaleExtent([0.15, 4]).on("zoom", (ev) => {
    root.attr("transform", ev.transform);
  }));

  sim = d3.forceSimulation()
    .force("link",   d3.forceLink().id(d => d.id).distance(170).strength(0.45))
    .force("charge", d3.forceManyBody().strength(-480))
    .force("center", d3.forceCenter(rect.width / 2, rect.height / 2))
    .force("collide", d3.forceCollide(n => n.type === "agent" ? AGENT_COLL_R : NODE_R + 14))
    .on("tick", onTick);
}

export function setGraph(nodes, links) {
  // Enrich agent nodes with their outgoing tool names (for the card panel)
  for (const n of nodes) {
    if (n.type === "agent") {
      n.tools = links
        .filter(l => {
          const src = typeof l.source === "object" ? l.source.id : l.source;
          return src === n.id && l.kind === "tool";
        })
        .map(l => {
          const tgt = typeof l.target === "object" ? l.target.id : l.target;
          return tgt.replace(/^tool:/, "");
        });
    }
  }
  state.nodes = nodes;
  state.links = links;
  redraw();
}

export function ensureToolNode(toolName, agentId) {
  const id = `tool:${toolName}`;
  if (!state.nodes.find(n => n.id === id)) {
    state.nodes.push({ id, name: toolName, type: "tool" });
  }
  if (!state.links.find(l => {
    const s = typeof l.source === "object" ? l.source.id : l.source;
    return s === agentId && l.target === id && l.kind === "tool";
  })) {
    state.links.push({ source: agentId, target: id, kind: "tool" });
  }
  // Keep agent's tools list in sync
  const agentNode = state.nodes.find(n => n.id === agentId);
  if (agentNode && agentNode.tools && !agentNode.tools.includes(toolName)) {
    agentNode.tools.push(toolName);
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
      .on("start", (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag",  (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
      .on("end",   (ev, d) => { if (!ev.active) sim.alphaTarget(0); d._pinned = true; updatePinIcon(d); }));

  // Double-click unpins
  enter.on("dblclick", (ev, d) => {
    ev.stopPropagation();
    d.fx = null; d.fy = null; d._pinned = false;
    updatePinIcon(d);
    sim.alpha(0.3).restart();
  });

  // Hit-ring circle (same for all types, rendered before shape)
  enter.append("circle").attr("class", "hit-ring");

  // Shape per type
  enter.each(function(d) {
    const g = d3.select(this);
    if (d.type === "agent") {
      const f = boxFaces();
      g.append("polygon").attr("class", "box-right").attr("points", f.right);
      g.append("polygon").attr("class", "box-top").attr("points", f.top);
      g.append("polygon").attr("class", "box-front").attr("points", f.front);
    } else if (d.type === "tool") {
      g.append("path").attr("class", "node-bg").attr("d", HEX_PATH);
    } else {
      g.append("path").attr("class", "node-bg").attr("d", DIAMOND_PATH);
    }
  });

  enter.append("text").attr("class", "node-label").attr("dy", 4)
    .text(n => truncate(n.name, n.type === "agent" ? 10 : 12));
  enter.append("text").attr("class", "node-sub")
    .attr("dy", d => d.type === "agent" ? BOX.h / 2 + BOX.d + 14 : NODE_R + 14)
    .text(n => sublabel(n));

  // Pin icon (hidden by default)
  enter.append("text").attr("class", "pin-icon")
    .attr("dy", d => d.type === "agent" ? -(BOX.h / 2 + BOX.d + 6) : -(NODE_R + 6))
    .attr("text-anchor", "middle")
    .style("display", "none").text("📌");

  enter.on("click", (_ev, d) => {
    state.selectedNode = d;
    emit("nodeClick", d);
  });

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

// ---- Animation API -------------------------------------------------------

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
