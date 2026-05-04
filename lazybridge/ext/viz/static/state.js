// Shared in-memory state, plus a tiny pub/sub so panels can react
// when a new event arrives or the user picks one from the timeline.

export const state = {
  meta: {},
  events: [],            // chronological, capped at 5000
  byId: new Map(),       // _seq -> event
  selectedSeq: null,
  store: {},
  storeFresh: new Map(), // key -> ts when last updated, used for highlight
  nodes: [],             // graph nodes (mutable as tools appear)
  links: [],             // graph links
  inFlight: new Set(),   // node ids currently between MODEL_REQUEST and MODEL_RESPONSE
};

const EVENTS_CAP = 5000;
const subs = new Map();

export function on(name, fn) {
  if (!subs.has(name)) subs.set(name, new Set());
  subs.get(name).add(fn);
  return () => subs.get(name).delete(fn);
}

export function emit(name, payload) {
  const s = subs.get(name);
  if (!s) return;
  for (const fn of s) {
    try { fn(payload); } catch (e) { console.error(`viz: handler ${name}`, e); }
  }
}

export function pushEvent(ev) {
  state.events.push(ev);
  state.byId.set(ev._seq, ev);
  if (state.events.length > EVENTS_CAP) {
    const drop = state.events.splice(0, state.events.length - EVENTS_CAP);
    for (const d of drop) state.byId.delete(d._seq);
  }
}

export function selectEvent(seq) {
  state.selectedSeq = seq;
  emit("select", seq);
}
