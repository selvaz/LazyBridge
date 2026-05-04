// D3 force-directed graph with hierarchical layout.
//
// Layout: longest-path on context+nested-agent edges → layer per agent;
// tool-function nodes placed one sub-row below their parent.
// IO nodes (__START__, __END__) are injected as virtual nodes above/below.
//
// Node shapes:
//   __START__ / __END__ → rounded pill (io type)
//   Agent               → 3-D parallelepiped
//   Tool                → hexagon
//   Router              → diamond
//
// Exports: setGraph, ensureToolNode, pulse, fireHitRing, setInFlight,
//          flashError, zoomIn, zoomOut, zoomReset, setVisibleLayers, getLayers,
//          refreshNodeLabels

import { state, emit } from "/static/state.js";
import { installDefs } from "/static/graph-defs.js";
import { spawnPulse } from "/static/graph-pulse.js";

// ---- geometry ---------------------------------------------------------------

const NODE_R = 22;
const BOX    = { w: 60, h: 32, d: 10 };
const AGENT_COLL_R = BOX.w / 2 + BOX.d + 14;

const HEX_PATH = (() => {
  const pts = Array.from({ length: 6 }, (_, i) => {
    const a = (Math.PI / 180) * (60 * i + 30);
    return [NODE_R * Math.cos(a), NODE_R * Math.sin(a)];
  });
  return "M" + pts.map(p => p.join(",")).join("L") + "Z";
})();
const DIAMOND_PATH =
  `M0,${-(NODE_R + 4)} L${NODE_R + 4},0 L0,${NODE_R + 4} L${-(NODE_R + 4)},0 Z`;

function boxFaces() {
  const hw = BOX.w / 2, hh = BOX.h / 2, d = BOX.d;
  const p  = pts => pts.map(([x, y]) => `${x},${y}`).join(" ");
  const ftl = [-hw, -hh], ftr = [hw, -hh], fbl = [-hw, hh], fbr = [hw, hh];
  const btl = [-hw + d, -hh - d], btr = [hw + d, -hh - d], bbr = [hw + d, hh - d];
  return {
    front: p([ftl, ftr, fbr, fbl]),
    top:   p([ftl, ftr, btr, btl]),
    right: p([ftr, fbr, bbr, btr]),
  };
}

// ---- layout constants -------------------------------------------------------
const LH = 230; // vertical gap between agent rows (extra space for task/ctx text)
const TY = 100; // tool sub-row Y offset below parent agent
const AX = 220; // horizontal gap between parallel agents
const TX = 115; // horizontal gap between tools of same parent

function idOf(r) { return typeof r === "object" ? r.id : r; }

// ---- hierarchical layout ----------------------------------------------------

export function computeLayout(nodes, links) {
  const agentIds = new Set(nodes.filter(n => n.type !== "tool").map(n => n.id));

  // Predecessors map (agent→agent, context + nested-agent tool edges)
  const pred = new Map(nodes.map(n => [n.id, new Set()]));
  for (const l of links) {
    const s = idOf(l.source), t = idOf(l.target);
    if (agentIds.has(s) && agentIds.has(t)) pred.get(t)?.add(s);
  }

  // Longest-path layer
  const layer = new Map();
  function assignLayer(id) {
    if (layer.has(id)) return layer.get(id);
    layer.set(id, -Infinity);
    const preds = [...(pred.get(id) || [])];
    const l = preds.length ? 1 + Math.max(...preds.map(assignLayer)) : 0;
    layer.set(id, l);
    return l;
  }
  for (const id of agentIds) assignLayer(id);

  const maxLayer = Math.max(0, ...[...layer.values()].filter(isFinite));

  // Group agents by layer
  const byLayer = Array.from({ length: maxLayer + 1 }, () => []);
  for (const [id, l] of layer) {
    const n = nodes.find(n => n.id === id);
    if (n && isFinite(l)) byLayer[l].push(n);
  }

  // Identify start (no predecessors) and end (no successors in context)
  const hasContextSucc = new Set();
  for (const l of links) {
    if ((l.kind === "context" || l.kind === "router") && agentIds.has(idOf(l.source)))
      hasContextSucc.add(idOf(l.source));
  }
  const hasContextPred = new Set();
  for (const l of links) {
    if ((l.kind === "context" || l.kind === "router") && agentIds.has(idOf(l.target)))
      hasContextPred.add(idOf(l.target));
  }
  // Also treat nested (tool-edge agent→agent) as flow for start/end logic.
  // The sub-agent's result flows BACK to the caller via tool_result, so:
  //   - caller gets a "successor" (the sub-agent call)
  //   - sub-agent gets a "predecessor" (the caller)
  //   - sub-agent also gets a "successor" (its result is consumed by the caller)
  // This prevents tool-nested agents from being mis-classified as END.
  for (const l of links) {
    if (l.kind === "tool" && agentIds.has(idOf(l.source)) && agentIds.has(idOf(l.target))) {
      hasContextSucc.add(idOf(l.source));
      hasContextPred.add(idOf(l.target));
      hasContextSucc.add(idOf(l.target)); // result consumed by caller → not an end node
    }
  }

  // Position agents + compute raw start/end flags
  const agentPos = new Map();
  for (let i = 0; i <= maxLayer; i++) {
    const group = byLayer[i];
    group.forEach((n, j) => {
      const x = (j - (group.length - 1) / 2) * AX;
      const y = i * LH;
      agentPos.set(n.id, { x, y });
      n._tx = x; n._ty = y;
      n._layer = i;
      n._isStart = !hasContextPred.has(n.id);
      n._isEnd   = !hasContextSucc.has(n.id);
    });
  }

  // If a single node is both START and END (solo pipeline), drop START badge
  nodes.forEach(n => { if (n._isStart && n._isEnd) n._isStart = false; });

  // Tool-function nodes: cluster below their parent
  const toolsByParent = new Map();
  for (const l of links) {
    const s = idOf(l.source), t = idOf(l.target);
    if (agentIds.has(s) && !agentIds.has(t)) {
      if (!toolsByParent.has(s)) toolsByParent.set(s, []);
      const tn = nodes.find(n => n.id === t);
      if (tn && !toolsByParent.get(s).includes(tn)) toolsByParent.get(s).push(tn);
    }
  }
  for (const [parentId, tools] of toolsByParent) {
    const pos = agentPos.get(parentId);
    if (!pos) continue;
    const parentLayer = nodes.find(n => n.id === parentId)?._layer ?? 0;
    tools.forEach((t, j) => {
      t._tx = pos.x + (j - (tools.length - 1) / 2) * TX;
      t._ty = pos.y + TY;
      t._layer = parentLayer;  // same visual row as parent
    });
  }

  // Fallback for disconnected nodes
  let fxOff = 0;
  for (const n of nodes) {
    if (n._tx === undefined) {
      n._tx = fxOff; n._ty = (maxLayer + 1) * LH; n._layer = maxLayer + 1;
      fxOff += AX;
    }
  }
}

export function getLayers() {
  const layers = new Set();
  for (const n of state.nodes) if (n._layer !== undefined) layers.add(n._layer);
  return [...layers].sort((a, b) => a - b);
}

// ---- D3 internals -----------------------------------------------------------

let svg, zoomBehavior, viewport;
let gLinks, gNodes, gPulses, sim, nodeSel, linkSel;
let _visibleLayers = null; // null = all visible

export function initGraph() {
  const el   = document.getElementById("canvas");
  const rect = el.getBoundingClientRect();
  svg = d3.select(el).attr("viewBox", `0 0 ${rect.width} ${rect.height}`);
  installDefs(svg);

  zoomBehavior = d3.zoom().scaleExtent([0.1, 5])
    .on("zoom", ev => viewport.attr("transform", ev.transform));
  svg.call(zoomBehavior);

  viewport = svg.append("g").attr("class", "viewport");
  gLinks   = viewport.append("g").attr("class", "links");
  gNodes   = viewport.append("g").attr("class", "nodes");
  gPulses  = viewport.append("g").attr("class", "pulses");

  sim = d3.forceSimulation()
    .force("link",    d3.forceLink().id(d => d.id).distance(120).strength(0.15))
    .force("charge",  d3.forceManyBody().strength(-60))
    .force("collide", d3.forceCollide(n => n.type === "agent" ? AGENT_COLL_R + 6 : NODE_R + 10))
    .alphaDecay(0.06)
    .on("tick", onTick);
}

// ---- Zoom -------------------------------------------------------------------

export function zoomIn()    { svg.transition().duration(280).call(zoomBehavior.scaleBy, 1.35); }
export function zoomOut()   { svg.transition().duration(280).call(zoomBehavior.scaleBy, 1 / 1.35); }
export function zoomReset() {
  const el = document.getElementById("canvas");
  const r  = el.getBoundingClientRect();
  svg.transition().duration(420)
    .call(zoomBehavior.transform, d3.zoomIdentity.translate(r.width / 2, r.height / 2));
}

// ---- Level filter -----------------------------------------------------------

export function setVisibleLayers(layerSet) {
  _visibleLayers = layerSet; // null → show all
  _applyLayerFilter();
}

function _applyLayerFilter() {
  if (!gNodes) return;
  gNodes.selectAll(".node").each(function(d) {
    const vis = _visibleLayers === null || _visibleLayers.has(d._layer);
    d3.select(this).classed("layer-hidden", !vis);
  });
  gLinks.selectAll(".link").each(function(l) {
    const sLay = l.source?._layer; const tLay = l.target?._layer;
    const vis = _visibleLayers === null ||
      (_visibleLayers.has(sLay) && _visibleLayers.has(tLay));
    d3.select(this).classed("layer-hidden", !vis);
  });
}

// ---- Public graph API -------------------------------------------------------

export function setGraph(nodes, links) {
  // Remove tool nodes that are duplicates of existing agent nodes
  // (a nested agent registered as both agent and tool).
  const agentNames = new Set(nodes.filter(n => n.type === "agent").map(n => n.name));
  const deduped = nodes.filter(n => !(n.type === "tool" && agentNames.has(n.name)));
  // Remap links that pointed to the dropped tool node to the agent node
  const remapped = links.map(l => {
    const t = idOf(l.target);
    const tn = nodes.find(n => n.id === t && n.type === "tool" && agentNames.has(n.name));
    if (!tn) return l;
    const an = deduped.find(n => n.type === "agent" && n.name === tn.name);
    return an ? { ...l, target: an.id, kind: "context" } : l;
  });
  for (const n of deduped) {
    if (n.type !== "tool") {
      n.tools = remapped
        .filter(l => idOf(l.source) === n.id && l.kind === "tool")
        .map(l => idOf(l.target).replace(/^tool:/, ""));
    }
  }
  state.nodes = deduped;
  state.links = remapped;
  computeLayout(nodes, links);
  _injectIONodes();
  _anchorNodes(state.nodes);
  redraw();
  setTimeout(zoomReset, 700);
}

export function ensureToolNode(toolName, agentId) {
  // If this tool name corresponds to an existing agent node, use that node id directly
  // (nested agent — don't create a duplicate tool node).
  const existingAgent = state.nodes.find(n => n.type === "agent" && n.name === toolName);
  const id = existingAgent ? existingAgent.id : `tool:${toolName}`;

  if (!existingAgent && !state.nodes.find(n => n.id === id))
    state.nodes.push({ id, name: toolName, type: "tool" });
  if (!state.links.find(l => idOf(l.source) === agentId && idOf(l.target) === id))
    state.links.push({ source: agentId, target: id, kind: existingAgent ? "context" : "tool" });
  const an = state.nodes.find(n => n.id === agentId);
  if (an?.tools && !an.tools.includes(toolName)) an.tools.push(toolName);
  computeLayout(state.nodes, state.links);
  _injectIONodes();
  _anchorNodes(state.nodes);
  redraw();
  return id;
}

// Inject virtual __START__ and __END__ nodes + flow links into state.nodes/links.
// Called after every computeLayout(). Safe to call multiple times (removes old IO nodes first).
function _injectIONodes() {
  const IO_IDS = new Set(["__START__", "__END__"]);
  // Strip previous virtual nodes/links
  const base  = state.nodes.filter(n => !IO_IDS.has(n.id));
  const blink = state.links.filter(l => !IO_IDS.has(idOf(l.source)) && !IO_IDS.has(idOf(l.target)));
  state.nodes.length = 0; base.forEach(n => state.nodes.push(n));
  state.links.length = 0; blink.forEach(l => state.links.push(l));

  const agents = base.filter(n => n.type !== "tool" && n._layer !== undefined);
  if (!agents.length) return;

  const startAgents = agents.filter(n => n._isStart);
  const endAgents   = agents.filter(n => n._isEnd);
  const minLayer    = Math.min(...agents.map(n => n._layer));
  const maxLayer    = Math.max(...agents.map(n => n._layer));

  const avgX = arr => arr.reduce((s, n) => s + (n._tx ?? 0), 0) / (arr.length || 1);

  // START node — above layer 0
  const sAgents = startAgents.length ? startAgents : agents.filter(n => n._layer === minLayer);
  const sx = avgX(sAgents), sy = minLayer * LH - LH * 0.7;
  const startNode = { id: "__START__", name: "START", type: "io", _role: "start",
                      _virtual: true, _layer: -1, _tx: sx, _ty: sy };
  state.nodes.push(startNode);
  sAgents.forEach(a => state.links.push({ source: "__START__", target: a.id, kind: "flow" }));

  // END node — below deepest layer
  const eAgents = endAgents.length ? endAgents : agents.filter(n => n._layer === maxLayer);
  const ex = avgX(eAgents), ey = maxLayer * LH + LH * 0.7;
  const endNode = { id: "__END__", name: "END", type: "io", _role: "end",
                    _virtual: true, _layer: maxLayer + 1, _tx: ex, _ty: ey };
  state.nodes.push(endNode);
  eAgents.forEach(a => state.links.push({ source: a.id, target: "__END__", kind: "flow" }));
}

// Fix all non-pinned nodes to their computed layout positions.
function _anchorNodes(nodes) {
  for (const n of nodes) {
    if (!n._pinned) {
      n.x  = n._tx ?? 0;
      n.y  = n._ty ?? 0;
      n.fx = n._tx ?? 0;
      n.fy = n._ty ?? 0;
    }
  }
}

// ---- Rendering --------------------------------------------------------------

function redraw() {
  linkSel = gLinks.selectAll(".link").data(state.links, keyLink);
  linkSel.exit().remove();
  linkSel = linkSel.enter()
    .append("line").attr("class", l => `link ${l.kind || "tool"}`)
    .merge(linkSel);

  nodeSel = gNodes.selectAll(".node").data(state.nodes, n => n.id);
  nodeSel.exit().remove();

  const enter = nodeSel.enter().append("g")
    .attr("class", n => `node ${n.type}`)
    .attr("data-id", n => n.id)
    .call(d3.drag()
      .on("start", (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag",  (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
      .on("end",   (ev, d) => { if (!ev.active) sim.alphaTarget(0); d._pinned = true; _updatePin(d); }));

  enter.on("dblclick", (ev, d) => {
    ev.stopPropagation();
    d.fx = d._tx ?? 0; d.fy = d._ty ?? 0;
    d.x  = d._tx ?? 0; d.y  = d._ty ?? 0;
    d._pinned = false;
    _updatePin(d);
    sim.alpha(0.1).restart();
  });

  // Hit-ring (below shapes)
  enter.append("circle").attr("class", "hit-ring");

  // Shapes per type
  enter.each(function(d) {
    const g = d3.select(this);
    if (d.type === "agent") {
      const f = boxFaces();
      g.append("polygon").attr("class", "box-right").attr("points", f.right);
      g.append("polygon").attr("class", "box-top").attr("points", f.top);
      g.append("polygon").attr("class", "box-front").attr("points", f.front);
    } else if (d.type === "tool") {
      g.append("path").attr("class", "node-bg").attr("d", HEX_PATH);
    } else if (d.type === "io") {
      g.append("rect").attr("class", "io-bg")
        .attr("x", -44).attr("y", -16).attr("width", 88).attr("height", 32)
        .attr("rx", 16).attr("ry", 16);
    } else {
      g.append("path").attr("class", "node-bg").attr("d", DIAMOND_PATH);
    }
  });

  // Labels
  enter.append("text").attr("class", "node-label").attr("dy", 5)
    .text(n => n.type === "io" ? n.name : trunc(n.name, n.type === "agent" ? 10 : 12));
  enter.filter(d => d.type !== "io").append("text").attr("class", "node-sub")
    .attr("dy", d => d.type === "agent" ? BOX.h / 2 + BOX.d + 14 : NODE_R + 14)
    .text(n => sublabel(n));

  // Task text (agents only) — updated dynamically by refreshNodeLabels()
  enter.filter(d => d.type === "agent").append("text").attr("class", "node-task")
    .attr("dy", BOX.h / 2 + BOX.d + 27).text("");

  // Context sources (agents only) — e.g. "ctx: researcher_a, store"
  enter.filter(d => d.type === "agent").append("text").attr("class", "node-ctx")
    .attr("dy", BOX.h / 2 + BOX.d + 38).text(d => _ctxLabel(d.id));

  // IO node sub-label (click hint)
  enter.filter(d => d.type === "io").append("text").attr("class", "node-sub io-hint")
    .attr("dy", 28).text("click to inspect");

  // Pin icon (non-IO nodes only)
  enter.filter(d => d.type !== "io").append("text").attr("class", "pin-icon")
    .attr("dy", d => d.type === "agent" ? -(BOX.h / 2 + BOX.d + 22) : -(NODE_R + 22))
    .attr("text-anchor", "middle").style("display", "none").text("📌");

  enter.on("click", (_ev, d) => { state.selectedNode = d; emit("nodeClick", d); });

  nodeSel = enter.merge(nodeSel);

  // IO role class (for CSS)
  nodeSel.attr("data-role", d => d._role || null);
  // Refresh dynamic text (ctx sources may have changed with new links)
  nodeSel.select(".node-ctx").text(d => _ctxLabel(d.id));
  nodeSel.select(".node-task").text(d => {
    const t = state.agentTasks.get(d.name) || state.agentTasks.get(d.id) || "";
    return trunc(t, 24);
  });

  sim.nodes(state.nodes);
  sim.force("link").links(state.links);
  sim.alpha(0.3).restart();

  _applyLayerFilter();
}

function _updatePin(d) {
  const node = gNodes.select(`.node[data-id="${css(d.id)}"]`);
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

function keyLink(l) { return `${idOf(l.source)}->${idOf(l.target)}|${l.kind || ""}`; }
function trunc(s, n) { return s && s.length > n ? s.slice(0, n - 1) + "…" : (s || ""); }
function sublabel(n) {
  if (n.type === "tool") return "tool";
  if (n.type === "router") return "router";
  return n.model ? trunc(n.model, 18) : (n.provider || "");
}
function css(s) { return String(s).replace(/(["\\])/g, "\\$1"); }

// Returns "ctx: A, B" for context sources feeding into nodeId
function _ctxLabel(nodeId) {
  const sources = state.links
    .filter(l => {
      const t = idOf(l.target);
      return t === nodeId && (l.kind === "context" || l.kind === "router");
    })
    .map(l => {
      const src = idOf(l.source);
      return state.nodes.find(n => n.id === src)?.name || src;
    });
  return sources.length ? "ctx: " + sources.map(s => trunc(s, 10)).join(", ") : "";
}

// Update task + ctx text on all rendered agent nodes.
// Called by dispatch whenever agentTasks changes.
export function refreshNodeLabels() {
  if (!nodeSel) return;
  nodeSel.select(".node-task").text(d => {
    const t = state.agentTasks.get(d.name) || state.agentTasks.get(d.id) || "";
    return trunc(t, 24);
  });
  nodeSel.select(".node-ctx").text(d => _ctxLabel(d.id));
}

// ---- Animation API ----------------------------------------------------------

export function pulse(srcId, dstId, color) {
  const src = state.nodes.find(n => n.id === srcId);
  const dst = state.nodes.find(n => n.id === dstId);
  if (src && dst) { spawnPulse(gPulses, src, dst, color); fireHitRing(dstId, color); }
}

export function fireHitRing(nodeId, color) {
  const node = gNodes.select(`.node[data-id="${css(nodeId)}"]`);
  if (node.empty()) return;
  const ring = node.select(".hit-ring");
  ring.attr("stroke", color || "#00d4ff").classed("fire", false);
  void ring.node().getBoundingClientRect();
  ring.classed("fire", true);
}

export function setInFlight(nodeId, on) {
  const node = gNodes.select(`.node[data-id="${css(nodeId)}"]`);
  if (node.empty()) return;
  node.classed("in-flight", !!on);
  if (on) state.inFlight.add(nodeId); else state.inFlight.delete(nodeId);
}

export function flashError(nodeId) {
  const node = gNodes.select(`.node[data-id="${css(nodeId)}"]`);
  if (node.empty()) return;
  node.classed("error", true);
  setTimeout(() => node.classed("error", false), 600);
}
