# GUI — Complete Reference

## 1. Overview

`lazybridge.gui` is a stdlib-only browser UI for inspecting, editing, and live-testing every LazyBridge object. One shared localhost server; one browser tab; multiple panel types in the sidebar.

```python
import lazybridge.gui   # activates .gui() method on all core classes
```

**Security:** localhost-only (`127.0.0.1`, ephemeral port), token-gated (random 24-byte urlsafe token inlined into page JS). Never expose outside localhost.

---

## 2. Activation & top-level API

```python
from lazybridge.gui import (
    # High-level entry points
    open_gui,                    # open_gui(obj) → url str
    panel_input_fn,              # factory for HumanAgent/SupervisorAgent browser input

    # Server lifecycle
    get_server,                  # → GuiServer singleton
    close_server,                # shut down the HTTP server
    is_running,                  # → bool

    # Panel classes (for custom extensions)
    Panel, AgentPanel, ToolPanel, PipelinePanel,
    SessionPanel, RouterPanel, StorePanel, HumanInputPanel,

    # Low-level
    GuiServer,

    # Monkey-patch control
    install_gui_methods,         # adds .gui() to core classes
    uninstall_gui_methods,       # removes .gui()
)
```

After `import lazybridge.gui`, every core object gains `.gui()`:

```python
url = obj.gui()                  # opens browser (first call only), adds panel, returns url
url = obj.gui(open_browser=False) # add panel without opening a tab
```

Equivalent without monkey-patching:

```python
from lazybridge.gui import open_gui
url = open_gui(my_agent)
```

All calls share **one browser tab** — no port churn, no extra windows.

---

## 3. Panel types

| Object | Panel | Inspect | Edit (live) | Test |
|---|---|---|---|---|
| `LazyAgent` | `AgentPanel` | name, provider, model, description, session | system prompt, model, enabled tools, native tools | `chat` / `loop` / `text` with usage + cost |
| `LazyTool` (function) | `ToolPanel` | name, description, guidance, JSON schema | read-only | schema-driven form → `tool.run(args)` |
| `LazyTool.chain / .parallel` | `PipelinePanel` | mode, combiner, concurrency, timeouts, participants | read-only | `tool.run({"task": …})` with per-step timeline |
| `LazyRouter` | `RouterPanel` | routes table, default route, condition repr | read-only | `router.route(value)` + optional agent invocation |
| `LazyStore` | `StorePanel` | all keys with value preview, writing agent, timestamp | read / write / delete / read-all | same four operations |
| `LazySession` | `SessionPanel` | session id, tracking level, registered agents, store keys | read-only | — (use child panels) |
| `HumanAgent` / `SupervisorAgent` | `HumanInputPanel` | live prompt from the running pipeline | — | submit response via browser |

---

## 4. LazyAgent panel

```python
import lazybridge.gui
from lazybridge import LazyAgent, LazySession, LazyTool

sess = LazySession()
agent = LazyAgent("anthropic", name="researcher", session=sess,
                  system="You are a researcher.")

agent.gui()  # opens browser, shows AgentPanel for "researcher"
```

**Inspect tab:** provider, model, name, description, session id — read-only.

**Edit tab (live):**
- `system` prompt — edits take effect on the **next** `chat()`/`loop()` call; no restart needed.
- `model` — change model at runtime; accepts literal model names or tier strings.
- Tools checklist — enable/disable tools from the session tool pool.
- Native tools checklist — toggle `NativeTool.WEB_SEARCH`, `CODE_EXECUTION`, etc.
- **"Export as Python"** button — copies a minimal snippet that reconstructs the agent's current state.

**Test tab:**
- Choose method: `text` / `chat` / `loop`
- Enter a prompt, click Run
- Shows `resp.content`, `usage.input_tokens`, `usage.output_tokens`, `usage.cost_usd`

**Tool scope for the checklist:**

By default, the checklist draws from all tools bound to agents in the same session. To add tools that aren't in a session yet:

```python
agent.gui(available_tools=[my_scratch_tool, another_tool])
```

---

## 5. LazyTool panel (function-backed)

```python
tool = LazyTool.from_function(search_web)
tool.gui()
```

**Inspect:** name, description, guidance text, schema mode, full JSON schema.

**Test:** a form auto-generated from the JSON schema — fill fields, click Run, see the return value.

---

## 6. PipelinePanel — chain / parallel tools

```python
pipeline = LazyTool.chain(researcher, writer, editor, name="article", description="Full pipeline")
pipeline.gui()
```

**Inspect:** mode (`chain` / `parallel`), combiner, concurrency limit, step timeout, participant list (each participant is clickable — opens its own panel).

**Test:**
- Enter a task string → `pipeline.run({"task": "..."})`
- Per-step timeline appears live as the pipeline runs:
  - `AGENT_START` event (agent name, step index)
  - `AGENT_FINISH` event (duration, first 200 chars of output)
- Final combined output shown when done
- Session-aware: if pipeline agents are in a `LazySession`, events are captured from the session event log and displayed chronologically.

---

## 7. SessionPanel

```python
sess = LazySession(db="pipeline.db", tracking="verbose")
# ... add agents ...
sess.gui()
```

**Shows:** session id, db path, tracking level, list of all registered agents (clickable → AgentPanel), current store keys.

Calling `sess.gui()` also automatically pre-registers panels for every agent, tool, and store already in the session — you don't need to call `.gui()` on each one separately.

---

## 8. StorePanel

```python
from lazybridge import LazyStore
store = LazyStore("data.db")
store.gui()
```

**Inspect:** all keys with truncated value preview, writing agent id, `written_at` timestamp.

**Test operations:**
- **Read** — enter a key, see the full value
- **Write** — enter key + value (JSON), click Write
- **Delete** — enter a key, click Delete
- **Read all** — dumps entire store as a JSON snapshot

---

## 9. RouterPanel

```python
from lazybridge import LazyRouter
router = LazyRouter(condition=classify, routes={"research": researcher, "code": coder})
router.gui()
```

**Inspect:** routes table (key → agent name), default key, condition function source (if introspectable).

**Test:**
- Enter any value → see which route is selected
- Optional: also invoke the selected agent with a prompt

---

## 10. HumanInputPanel — browser input for human agents

The legacy `web_input_fn()` from `lazybridge.gui.human` still works but emits `DeprecationWarning`. Use the shared-server factory instead:

```python
from lazybridge.gui import panel_input_fn
from lazybridge import HumanAgent, SupervisorAgent

fn = panel_input_fn(name="reviewer")  # creates a panel slot in the sidebar
human = HumanAgent(name="reviewer", input_fn=fn)

# When the pipeline reaches the human step, the GUI sidebar shows the prompt
# and a text field + Submit button. The pipeline pauses until Submit is clicked.

# Clean up when done
fn.panel.close()
```

The panel shows the task context (what the LLM produced so far) and a free-text input. Works with `SupervisorAgent` too — REPL commands (`continue`, `retry <agent>`, `store <key>`, `<tool>(args)`) are typed into the same browser field.

---

## 11. Editing behaviour & limitations

- Edits to `system`, `model`, `tools`, `native_tools` on `AgentPanel` take effect on the **next** call to the agent. No restart needed.
- Edits are **not persisted to disk** — use "Export as Python" to save the current state as a code snippet.
- `LazyTool` metadata (name, description, schema) is **read-only** in the GUI — the compiled schema is cached on first call.
- No edit-conflict detection between multiple browser tabs; last-write-wins.
- `mypy` does not see the monkey-patched `.gui()` method. Use `open_gui(obj)` or `cast(GuiEnabled, obj).gui()` for type-safe call sites.

---

## 12. Extending: write your own panel

```python
from lazybridge.gui import Panel, get_server

class MyThingPanel(Panel):
    kind = "generic"     # or "agent" to reuse the agent renderer

    def __init__(self, thing):
        self._thing = thing

    @property
    def id(self) -> str:
        return f"thing-{id(self._thing):x}"

    def render_state(self) -> dict:
        return {"name": self._thing.name, "value": self._thing.current_value}

    def handle_action(self, action: str, args: dict):
        if action == "refresh":
            return self.render_state()
        return super().handle_action(action, args)

get_server().register(MyThingPanel(my_thing))
```

To add a new `kind` with a custom renderer, add a matching entry to `lazybridge/gui/_templates.py`:

```python
PANEL_RENDERERS["my_kind"] = "renderMyThing"  # JS function name in _static/
```

---

## 13. Server lifecycle

```python
from lazybridge.gui import get_server, close_server, is_running

# First .gui() call starts the server automatically
server = get_server()      # GuiServer singleton; creates if not running
print(server.url)          # e.g. "http://127.0.0.1:54321?token=abc..."
print(is_running())        # True

close_server()             # clean shutdown; panels are de-registered
print(is_running())        # False
```

The server uses Server-Sent Events (`/api/events`) for live push updates (sidebar refresh, panel state changes). A 2-second polling fallback activates when SSE is unavailable (e.g. proxied environments).

---

## 14. Quick workflow: debug a pipeline in the browser

```python
import lazybridge.gui
from lazybridge import LazyAgent, LazySession, LazyTool

sess = LazySession(db="debug.db", tracking="verbose")

researcher = LazyAgent("anthropic", name="researcher", session=sess)
writer     = LazyAgent("openai",    name="writer",     session=sess)

pipeline = LazyTool.chain(researcher, writer,
                          name="article", description="Research then write")

# Register everything in one call
sess.gui()          # SessionPanel + auto-registers AgentPanels for researcher + writer

pipeline.gui()      # PipelinePanel with live per-step timeline

# Now run — watch the PipelinePanel timeline update in real time
result = pipeline.run({"task": "Write about fusion energy breakthroughs in 2025"})
print(result)
```
