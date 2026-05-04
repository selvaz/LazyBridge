# lazybridge.ext.viz — pipeline visualizer (alpha)

Live & replay UI that shows what is happening inside a LazyBridge
``Session`` in real time. Pulses travel along graph edges as agents
call tools, the inspector lets you read every event payload, and the
store viewer highlights writes as they happen.

## Quick start

### Live mode

```python
from lazybridge import Session
from lazybridge.ext.viz import Visualizer

sess = Session(db="run.db")

with Visualizer(sess) as viz:
    # ...build and run your pipeline here...
    pipeline.run({"task": "..."})
    # the browser is already open on viz.url
```

The browser opens on `http://127.0.0.1:<ephemeral>/#t=<token>`. The
token is generated per-process and gates every API call; the URL is
local-only.

### Replay mode

```python
from lazybridge.ext.viz import Visualizer

# Reads the most recent session from the SQLite event log
Visualizer.replay(db="run.db").open()
```

Use the play / pause / step controls and the speed selector to scrub
through the recorded run.

## What you see

- **Graph canvas** — agents, routers, and tools as nodes. Click a
  node to inspect the most recent event involving it. Drag nodes to
  rearrange; scroll to zoom.
- **Pulse animation** — every `tool_call` spawns a glowing packet
  that travels from the agent to the tool, with a fading trail.
  `tool_result` sends a packet back. Agents pulse softly while
  between `model_request` and `model_response`.
- **Inspector** — JSON tree of the selected event's full payload, or
  a live snapshot of the `Store` (entries flash green when newly
  written).
- **Timeline** — one tick per event, colored by type. Click a tick
  to jump to that event in the inspector.

## Architecture

```
Session.emit
   │
   ▼
HubExporter ──▶ EventHub ──▶ many SSE subscribers (browser tabs)
                   │
                   ▼
                ring buffer (last 500 events) — for late-joining tabs

VizServer  ── /api/graph    GraphSchema.to_dict()
           ── /api/store    Store.read_all()
           ── /api/snapshot ring buffer
           ── /api/events   SSE
           ── /api/control  POST replay actions (replay mode only)
           ── /static/...   HTML / CSS / JS
```

Backend is stdlib only. Frontend loads `d3.v7.min.js` from a CDN.

## Files

| File | Role |
|---|---|
| `visualizer.py` | Public `Visualizer` class — entry point. |
| `server.py` | `ThreadingHTTPServer` + SSE + token + static serving. |
| `exporter.py` | `EventHub` (pub/sub) + `HubExporter` (LazyBridge adapter). |
| `replay.py` | Reads SQLite event log; `ReplayController` pumps events. |
| `_normalizer.py` | JSON-safe coercion of arbitrary event payloads. |
| `static/index.html` | UI shell. |
| `static/styles*.css` | Theme + graph + panel styles (split for clarity). |
| `static/app.js` | Bootstrap. |
| `static/graph.js` | D3 force layout + node/edge rendering. |
| `static/graph-pulse.js` | Pulse + trail animation. |
| `static/graph-defs.js` | SVG `<defs>` (glow filter, gradients). |
| `static/dispatch.js` | Maps event types to visual gestures. |
| `static/inspector.js` | Right-pane: event detail + store viewer. |
| `static/timeline.js` | Bottom-pane: ticks + scrubber + replay controls. |
| `static/sse.js` | Reconnecting `EventSource` wrapper. |
| `static/state.js` | Shared in-memory store + tiny pub/sub. |
| `static/auth.js` | URL-hash token extraction + fetch helpers. |
| `static/json-tree.js` | Collapsible JSON renderer. |

## Testing

```
pytest tests/unit/ext/viz -q
```

33 unit tests, no network or LLM calls.
