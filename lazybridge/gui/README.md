# lazybridge.gui — Browser-based inspector & live-test for LazyBridge objects

Core, stdlib-only GUI for every importable LazyBridge object. One shared HTTP
server hosts a single tab; each call to an object's `.gui()` adds one entry
to the sidebar.

> **Status** — alpha. Local-dev tool. Localhost-bound, token-gated. Do not
> expose to a network.

## Install

Nothing to install. The package is pure stdlib (`http.server`, `threading`,
`queue`, `webbrowser`, `secrets`). Import the package to activate it:

```python
import lazybridge.gui                           # monkey-patches .gui() onto LazyBridge core classes
```

If you prefer not to monkey-patch (e.g. to keep type-checkers happy):

```python
from lazybridge.gui import open_gui
open_gui(my_agent)                              # returns the URL; same behaviour as my_agent.gui()
```

## The `.gui()` method

After `import lazybridge.gui`, every core class has a `.gui()` method that:

1. Starts the shared HTTP server the first time it's called (ephemeral port,
   `127.0.0.1` only, random 24-byte urlsafe token).
2. Opens your browser once (pass `open_browser=False` to suppress).
3. Registers a panel for the object on the sidebar.
4. Returns the URL.

```python
import lazybridge.gui
from lazybridge import LazyAgent, LazySession, LazyTool, LazyRouter, LazyStore

sess = LazySession()
agent = LazyAgent("anthropic", name="researcher", session=sess)

def search(query: str) -> str:
    """Search the web."""
    return f"results for {query}"

tool = LazyTool.from_function(search)

agent.gui()                                     # opens the browser, sidebar has researcher
tool.gui()                                      # adds a panel for the search tool
sess.gui()                                      # adds a session panel AND pre-registers every
                                                # agent / tool / store already in the session
```

All `.gui()` calls share one tab — no port churn, no extra browser windows.

## Panels — what each one does

| Object | Panel | Inspect | Edit (live) | Test tab |
|---|---|---|---|---|
| `LazyAgent` | `AgentPanel` | name, provider, model, description, session | `system` prompt, `model`, enabled session tools, native tools | `chat` / `loop` / `text` against the real provider, with usage & cost |
| `LazyTool` (function-backed) | `ToolPanel` | name, description, guidance, schema mode, JSON Schema | read-only | schema-driven form, `tool.run(args)` |
| `LazyTool.chain` / `.parallel` | `PipelinePanel` | mode, combiner, concurrency, timeouts, participants (clickable) | read-only | `tool.run({"task": …})` |
| `LazyRouter` | `RouterPanel` | routes, default, condition source | read-only | `router.route(value)` plus optional `route_and_run(value, prompt)` |
| `LazyStore` | `StorePanel` | key list with value preview + agent + timestamp | read / write / delete / read\_all | same four operations |
| `LazySession` | `SessionPanel` | id, tracking level, agents, store keys | read-only (use the child panels) | — |
| Human-in-the-loop | `HumanInputPanel` (via `panel_input_fn`) | live REPL prompt from `HumanAgent`/`SupervisorAgent` | — | submit response |

Every test tab runs **live against real provider credentials** — there is no
dry-run mode.

## Editing behaviour

The `LazyAgent` properties exposed for live editing — `system`, `model`,
`tools`, `native_tools` — are read by `chat()` / `loop()` on every call. Edits
take effect on the **next** invocation; no restart needed. Edits are lost
when the Python process exits: use the **"Export as Python"** button on the
agent panel to copy a minimal Python snippet that reconstructs the agent's
current state.

## Tool scope for the agent panel

An agent's "tools" checklist draws from the tools currently bound to any
agent in the **same session** (deduped by tool name). For tools that aren't
attached to a session agent yet, pass them explicitly:

```python
agent.gui(available_tools=[my_scratch_tool, another_tool])
```

## Human-in-the-loop

To have a `HumanAgent` / `SupervisorAgent` prompt show up in the GUI
sidebar instead of its own dedicated tab, use the shared-server factory:

```python
from lazybridge.gui import panel_input_fn

fn = panel_input_fn(name="reviewer")
supervisor = SupervisorAgent(name="reviewer", input_fn=fn)
# ... run the pipeline; the prompt appears in the sidebar.
fn.panel.close()
```

The legacy dedicated-port implementation at `lazybridge.gui.human` still
works but emits a `DeprecationWarning` on import; prefer the shared server.

## Public API

```python
from lazybridge.gui import (
    # High-level
    open_gui,                                   # open_gui(obj) → url
    get_server, close_server, is_running,       # shared server lifecycle

    # Panels (build your own extensions)
    Panel, AgentPanel, ToolPanel, PipelinePanel,
    SessionPanel, RouterPanel, StorePanel,
    HumanInputPanel, panel_input_fn,

    # Low-level
    GuiServer,

    # Monkey-patch control (rare)
    install_gui_methods, uninstall_gui_methods,
)
```

## Security model

- **Localhost only.** Bound to `127.0.0.1` on an ephemeral port.
- **Token-gated.** Every `/api/*` request requires a random 24-byte urlsafe
  token, inlined into the page JavaScript on first GET.
- **No persistence.** Panels' state is read from live Python objects on
  every request; GUI edits mutate those objects but are never serialised
  to disk.

This is a **developer tool**. Do not expose the server outside localhost.

## Known limitations

- Polling-based: the client re-fetches `/api/panels` every 2 s. SSE
  replacement is planned.
- `mypy` does not see the monkey-patched `.gui()` method directly. Use
  `open_gui(obj)` for type-checked call sites.
- No edit-conflict detection between tabs; last-write-wins.
- `LazyTool` metadata (name, description, guidance) is read-only — the
  compiled schema is cached on first call.

## Extending: write your own panel

```python
from lazybridge.gui import Panel, get_server

class MyThingPanel(Panel):
    kind = "agent"          # reuses the built-in agent renderer, or use "generic"

    def __init__(self, thing):
        self._thing = thing

    @property
    def id(self) -> str:
        return f"thing-{id(self._thing):x}"

    def render_state(self) -> dict:
        return {"name": self._thing.name, ...}

    def handle_action(self, action, args):
        if action == "do_something":
            return {"ok": True}
        return super().handle_action(action, args)

get_server().register(MyThingPanel(thing))
```

To register a new `kind`, add a matching renderer in
`lazybridge/gui/_templates.py` (`PANEL_RENDERERS[kind] = renderYourThing`).
