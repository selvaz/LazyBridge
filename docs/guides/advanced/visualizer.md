# Visualizer

A local-only web UI that shows what's happening inside a LazyBridge
`Session` in real time. Pulses travel along graph edges as agents
call tools, the inspector lets you read every event payload, and the
store viewer highlights writes as they happen. The same UI replays a
finished session from the SQLite event log with speed control and
step-by-step navigation.

> **Status: alpha.** Bound to `127.0.0.1` with an ephemeral port —
> not designed for remote access. Backend is stdlib-only
> (`http.server` + Server-Sent Events + a token in the URL); the
> frontend loads D3.js v7 from a CDN, so there is no build step.

## Signature

```python
from lazybridge import Session
from lazybridge.ext.viz import Visualizer


# Live mode — instrument an active Session.
Visualizer(
    session,                       # required: the Session to observe
    *,
    store=None,                    # optional Store to render in the side panel
    host="127.0.0.1",              # bind address (do NOT change unless you understand the risk)
    port=0,                        # 0 = ephemeral; pick a fixed port to share the URL
    auto_open=True,                # webbrowser.open(...) on start()
)


# Replay mode — reconstruct from a finished SQLite event log.
Visualizer.replay(
    db,                            # SQLite file path
    *,
    session_id=None,               # specific session id; None = first one in the file
    speed=1.0,                     # playback multiplier (0.1 .. 100.0)
    host="127.0.0.1",
    port=0,
    auto_open=True,
) -> Visualizer


# Server lifecycle.
viz.start()                        # start HTTP server (and replay controller in replay mode)
viz.stop()                         # stop server, detach exporter, stop replay
viz.url                            # URL the server is bound to
viz.open()                         # block until Ctrl+C — useful for replay scripts


# Context-manager pattern (recommended for live mode).
with Visualizer(session) as viz:
    # ...run pipeline; browser is already open...
    ...
```

## Synopsis

`Visualizer` has two modes built from the same UI:

- **Live mode** (`Visualizer(session)`) installs a `HubExporter` on
  the session, serves the live `GraphSchema`, optional `Store`, and
  a Server-Sent Events stream of every emitted event. The browser
  receives events as they happen and animates pulses along graph
  edges in real time.
- **Replay mode** (`Visualizer.replay(db=...)`) opens a finished
  SQLite event log, reconstructs a graph from the event stream
  (`reconstruct_graph(events)`), and exposes pause / play / step /
  speed controls so you can walk through a recorded run at your
  own pace.

In both modes the UI runs in a normal browser tab and talks to a
stdlib HTTP server bound to `127.0.0.1` with an ephemeral port. The
URL includes a token so a stray port-scanner doesn't get to replay
your private events; that's the only authentication.

`Visualizer` is a context manager — `__enter__` calls `start()` and
`__exit__` calls `stop()`. Recommended pattern: wrap your pipeline
run in a `with Visualizer(session):` block so the server shuts down
cleanly when the run finishes (the browser tab stays open; close it
yourself).

## When to use it

- **Debugging a multi-agent pipeline.** Live mode shows which agent
  fired which tool, in which order, with timing. Easier to read
  than a JSONL event dump.
- **Demos and walkthroughs.** Replay mode plus a recorded session
  is the cleanest way to talk through what a pipeline does without
  re-running it (and without spending tokens on every retake).
- **Post-mortem analysis.** A failed production run logged to
  SQLite replays in the same UI; jump directly to the
  `TOOL_ERROR` event and inspect the offending payload.
- **Teaching.** New users grasp the agent-as-tool composition model
  faster from the live graph than from documentation.

## When NOT to use it

- **Production observability.** Use `OTelExporter` and a real
  observability backend (Datadog, Honeycomb, Tempo). The
  Visualizer is for local inspection, not aggregated dashboards.
- **Headless / CI environments.** `auto_open=True` calls
  `webbrowser.open(...)`; pass `auto_open=False` if you're
  running in a container or behind SSH. (You can still hit
  `viz.url` from a tunnel, but local-only binding means you'll
  need an SSH port-forward.)
- **Sensitive sessions on shared hosts.** Even with the URL token,
  the server binds to `127.0.0.1` and serves whatever the
  `Session` emits. Don't run it on a multi-tenant host where
  other users could discover the port.
- **Long-lived processes that don't end cleanly.** The server
  thread runs until `stop()` is called. If your process daemonises
  or has a non-trivial shutdown path, wire `stop()` into your
  signal handler.

## Example

```python
from lazybridge import Agent, LLMEngine, Session, Store
from lazybridge.ext.viz import Visualizer


# 1) Live mode — wrap a pipeline run.
sess = Session(db="demo.db")
researcher = Agent(
    engine=LLMEngine("deepseek-v4-flash"),
    name="research",
    session=sess,
)
writer = Agent(
    engine=LLMEngine("deepseek-v4-flash"),
    name="write",
    session=sess,
)
orchestrator = Agent(
    engine=LLMEngine("deepseek-v4-flash"),
    tools=[researcher, writer],
    session=sess,
)

with Visualizer(sess) as viz:
    print(f"viz at {viz.url}")
    orchestrator("AI trends April 2026")


# 2) Replay mode — walk through a finished run at half speed.
Visualizer.replay(
    db="demo.db",
    speed=0.5,
).open()                           # blocks until Ctrl+C


# 3) Headless / CI — capture the URL without opening a browser.
with Visualizer(sess, auto_open=False) as viz:
    print(f"forward this port to view: {viz.url}")
    orchestrator("...")


# 4) Render a Store alongside the graph.
shared_store = Store(db="run.sqlite")
sess = Session(db="demo.db")

with Visualizer(sess, store=shared_store):
    # The right-hand panel updates as keys land in `shared_store`.
    pipeline_with_writes("...")
```

### Replay controls

The replay UI exposes four control actions over a side channel:

| Action | Effect |
|---|---|
| `play` | Resume playback (after `pause`). |
| `pause` | Halt playback at the current event. |
| `step` | Advance one event and pause. |
| `speed` | Set playback speed; numeric value, must be in `[0.1, 100.0]`. |

The browser UI surfaces these as buttons; under the hood they POST
to `/control` with a JSON body. The handler returns
`{"ok": true, "idx": <event_index>, "total": <total_events>}` so the
frontend can update its progress bar.

## Pitfalls

- **`auto_open=True` calls `webbrowser.open(...)` synchronously
  on `start()`.** On systems without a browser configured (CI,
  containers, headless servers), this is a silent no-op rather
  than an error — but you still need to read `viz.url` and open
  it yourself. Pass `auto_open=False` for headless runs.
- **`port=0` produces an ephemeral port** that changes every run.
  Pin a port (`port=8765`) only when you need a stable URL —
  e.g. to share with a teammate over a tunnel. Two visualizers
  on the same fixed port collide.
- **Live mode uses the live `GraphSchema`** of the session, so an
  agent without `session=sess` doesn't appear in the topology
  even if it runs. Pass the session to every agent you want
  visible.
- **Replay reconstructs a minimal graph** from event payloads —
  the result is structurally similar to but not identical with
  the live `GraphSchema`. If you need pixel-perfect topology,
  serialise the graph yourself (`sess.graph.to_yaml()`) and load
  it alongside the events.
- **Speed bounds.** `speed` is clamped to `[0.1, 100.0]` server-
  side; values outside that range produce a JSON error response.
  Real-time playback is `1.0`; `0.5` is half-speed; `10.0` is
  ten times faster.
- **The browser stays open after `stop()`.** The HTTP server
  shuts down, but the user's browser tab keeps the now-stale UI.
  This is by design — closing the user's tab from the server
  side would be hostile. Add a banner or close instruction in
  your demo script if it matters.
- **Custom UIs.** The exporter (`HubExporter`) and event hub
  (`EventHub`) live in `lazybridge.ext.viz.exporter`; if you want
  to drive a different frontend, instantiate them yourself and
  consume events from the hub's queue rather than going through
  `Visualizer`.

## See also

- [Session](../mid/session.md) — the source of events the
  Visualizer renders; covers `db=`, `batched=`, exporters.
- [GraphSchema](../full/graph-schema.md) — the topology view the
  Visualizer reads in live mode.
- [Exporters](../full/exporters.md) — the broader exporter surface;
  Visualizer's `HubExporter` is one of many possible sinks.
- [OpenTelemetry](otel.md) — the production-observability
  counterpart; complements (rather than replaces) the local
  Visualizer.
