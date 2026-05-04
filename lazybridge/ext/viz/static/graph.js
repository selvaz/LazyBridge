// D3 force-directed graph with live mutation: nodes and edges
// can appear at runtime (tool nodes are discovered from events).
// The module exposes a small imperative API used by app.js.

import { state, emit } from "/static/state.js";
import { installDefs } from "/static/graph-defs.js";
import { spawnPulse } from "/static/graph-pulse.js";

const NODE_R = 22;

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

  svg.call(d3.zoom().scaleExtent([0.4, 3]).on("zoom", (ev) => {
    root.attr("transform", ev.transform);
  }));

  sim = d3.forceSimulation()
    .force("link", d3.forceLink().id(d => d.id).distance(140).strength(0.5))
    .force("charge", d3.forceManyBody().strength(-380))
    .force("center", d3.forceCenter(rect.width / 2, rect.height / 2))
    .force("collide", d3.forceCollide(NODE_R + 8))
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
      .on("start", (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag",  (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
      .on("end",   (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));

  enter.append("circle").attr("class", "hit-ring");
  enter.append("circle").attr("class", "node-bg").attr("r", NODE_R).attr("fill", "url(#node-grad)");
  enter.append("text").attr("class", "node-label").attr("dy", 4).text(n => truncate(n.name, 14));
  enter.append("text").attr("class", "node-sub").attr("dy", NODE_R + 14).text(n => sublabel(n));
  enter.on("click", (_ev, d) => emit("nodeClick", d));

  nodeSel = enter.merge(nodeSel);

  sim.nodes(state.nodes);
  sim.force("link").links(state.links);
  sim.alpha(0.6).restart();
}

function onTick() {
  if (!linkSel || !nodeSel) return;
  linkSel
    .attr("x1", l => l.source.x).attr("y1", l => l.source.y)
    .attr("x2", l => l.target.x).attr("y2", l => l.target.y);
  nodeSel.attr("transform", n => `translate(${n.x},${n.y})`);
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
  // restart animation
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
