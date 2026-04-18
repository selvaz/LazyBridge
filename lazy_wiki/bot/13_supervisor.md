# HumanAgent & SupervisorAgent — Complete Reference

## 1. Overview

Two classes let a human participate in a pipeline as a first-class agent — no callbacks, no hooks. Both expose the same surface as `LazyAgent` (`chat`, `achat`, `text`, `atext`, `loop`, `as_tool`, `.result`) so they drop into `LazyTool.chain(...)`, `LazyTool.parallel(...)`, `agent.loop(verify=...)`, and `LazyContext.from_agent(...)` without special-casing.

| Class | Module | Role |
|---|---|---|
| `HumanAgent` | `lazybridge/human.py` | Simple approval / review / dialogue — covers 90% of cases |
| `SupervisorAgent` | `lazybridge/supervisor.py` | REPL with superpowers: call tools, retry upstream agents with feedback, read the session store |

Both set `_is_human = True`, which `LazyTool.chain(step_timeout=...)` checks to skip the timeout on human participants — humans get unlimited time even when upstream/downstream agents don't.

Canonical API stub: [`00_quickref.md` §HumanAgent + §SupervisorAgent](00_quickref.md). Full tutorial with screenshots of the REPL: [`docs/course/13-human-in-the-loop.md`](../../docs/course/13-human-in-the-loop.md).

---

## 2. HumanAgent

### Constructor

```python
HumanAgent(
    name: str = "human",
    *,
    description: str | None = None,
    input_fn: Callable[[str], str] | None = None,     # default: builtin input()
    ainput_fn: Callable[[str], Awaitable[str]] | None = None,
    mode: Literal["single", "dialogue"] = "single",
    end_token: str = "done",                          # dialogue exit word
    prompt_template: str = "{task}",
    timeout: float | None = None,                     # seconds; None = forever
    default: str | None = None,                       # response on timeout
    session: LazySession | None = None,
)
```

### Methods

```python
HumanAgent.chat(messages, **kw) -> CompletionResponse
HumanAgent.achat(messages, **kw) -> CompletionResponse   # uses asyncio.to_thread
HumanAgent.text(messages, **kw) -> str
HumanAgent.atext(messages, **kw) -> str
HumanAgent.loop(messages, **kw) -> CompletionResponse    # alias for chat — present so verify= and chain() work uniformly
HumanAgent.as_tool(name, description, **kw) -> LazyTool
HumanAgent.result: str | None                            # last output, also readable via LazyContext.from_agent(human)
```

### Minimal usage

```python
from lazybridge import HumanAgent

human = HumanAgent(name="reviewer")
resp = human.chat("What do you think?")
print(resp.content)  # whatever the human typed
```

### As a chain participant

```python
from lazybridge import LazyAgent, LazyTool, HumanAgent

pipeline = LazyTool.chain(
    LazyAgent("anthropic", name="researcher"),
    HumanAgent(name="reviewer"),
    LazyAgent("openai", name="writer"),
    name="reviewed_pipeline",
    description="Research, get human review, then write",
)
pipeline.run({"task": "AI safety report"})
```

### As a verify judge

```python
agent.loop("Write code", verify=HumanAgent(name="approver"), max_verify=3)
```

### Dialogue mode

```python
human = HumanAgent(name="reviewer", mode="dialogue", end_token="done")
# Human types multiple lines; typing "done" finishes the turn.
```

### Timeouts and defaults

```python
# Wait 5 minutes, auto-approve on timeout
human = HumanAgent(name="approver", timeout=300, default="approved")
```

### Custom I/O (web UI, websocket, queue)

```python
human = HumanAgent(name="reviewer", input_fn=lambda prompt: my_web.wait(prompt))

async def ws_input(prompt: str) -> str:
    return await websocket.receive_text()

human = HumanAgent(name="reviewer", ainput_fn=ws_input)
```

---

## 3. SupervisorAgent

### Constructor

```python
SupervisorAgent(
    name: str = "supervisor",
    *,
    description: str | None = None,
    input_fn: Callable[[str], str] | None = None,      # default: builtin input()
    ainput_fn: Callable[[str], Awaitable[str]] | None = None,
    tools: list[LazyTool] | None = None,               # tools the human can call interactively
    agents: list | None = None,                        # agents the human can retry with feedback
    session: LazySession | None = None,                # for store/events access
    timeout: float | None = None,
    default: str | None = None,
)
```

### Methods

```python
SupervisorAgent.chat(messages, **kw) -> CompletionResponse   # runs the REPL
SupervisorAgent.achat(messages, **kw) -> CompletionResponse  # asyncio.to_thread wrapper
SupervisorAgent.text(messages, **kw) -> str
SupervisorAgent.atext(messages, **kw) -> str
SupervisorAgent.loop(messages, **kw) -> CompletionResponse   # alias for chat
SupervisorAgent.aloop(messages, **kw) -> CompletionResponse
SupervisorAgent.as_tool(name, description, **kw) -> LazyTool
SupervisorAgent.result: str | None
SupervisorAgent.id: str                                      # "supervisor-{name}"
```

### REPL commands

When the supervisor's turn comes, it prints the previous output, the available tools, the retryable agents, the current session-store keys (first 10), and waits for input. Each command is matched case-insensitively on the prefix.

| Command | Effect | Implementation |
|---|---|---|
| `continue` | Pass the current output forward unchanged | returns `last_output` |
| `continue: <msg>` | Replace the output with `<msg>` | returns the custom string |
| `retry <agent>: <feedback>` | Re-run `<agent>` with `"{task}\n\nFeedback: {feedback}"`; if the agent has tools, calls `.loop()`, else `.chat()` | `_parse_retry` + `_find_agent` |
| `retry <agent>` | Re-run without extra feedback | same as above with empty feedback |
| `store <key>` | Print `session.store.read(key)` | requires `session=` |
| `<tool>(<args>)` | Invoke a `LazyTool` interactively; the single positional argument is passed as the first required param (or `"task"` if no required params) | `_try_tool_call` regex `(\w+)\((.+)\)$` |

The supervisor captures the most recent non-`continue` output (tool result or agent retry) as `last_output`, so `continue` after a tool call forwards the tool's result — not the original task.

### Minimal standalone usage

```python
from lazybridge import SupervisorAgent

sup = SupervisorAgent(name="sup")
resp = sup.chat("draft: should we ship?")  # opens the REPL
print(resp.content)
```

### Chain usage

```python
from lazybridge import LazyAgent, LazyTool, LazySession, SupervisorAgent

sess = LazySession()

def search(query: str) -> str:
    """Search the web."""
    return f"results for {query}"

search_tool = LazyTool.from_function(search)
researcher  = LazyAgent("anthropic", name="researcher", tools=[search_tool], session=sess)
writer      = LazyAgent("openai",    name="writer",     session=sess)

supervisor = SupervisorAgent(
    name="supervisor",
    tools=[search_tool],
    agents=[researcher],
    session=sess,
)

pipeline = LazyTool.chain(
    researcher, supervisor, writer,
    name="supervised_pipeline",
    description="Research, supervise, write",
)
pipeline.run({"task": "AI safety report"})
```

### Scripted input (tests & non-interactive demos)

All human-in-the-loop tests in `tests/unit/test_human.py:179–268` drive the REPL with an iterator:

```python
inputs = iter(['search("AI safety")', 'retry researcher: add 2026 data', 'continue'])
supervisor = SupervisorAgent(
    name="sup",
    tools=[search_tool],
    agents=[researcher],
    input_fn=lambda prompt: next(inputs),
)
```

Every REPL prompt consumes one line from `inputs`. This pattern lets the same file run interactively in a terminal and non-interactively under `pytest` / CI.

### Async / event-loop safety

- `SupervisorAgent.achat()` runs the sync REPL inside `asyncio.to_thread`, so `await pipeline_tool.arun(...)` never blocks the event loop.
- Pass `ainput_fn=` when you plug into a websocket or async queue — the current implementation still wraps via `to_thread`, but the hook is reserved for a native async REPL.
- `LazyTool.parallel()` serializes human I/O through a module-level `_IO_LOCK` so concurrent prompts don't interleave on the terminal.

### Timeouts and KeyboardInterrupt

- `timeout=` + `default=` → on `TimeoutError`, logs a warning and returns the default.
- `timeout=` without `default=` → raises `TimeoutError`.
- CTRL+C → returns `default` if set, else re-raises `KeyboardInterrupt` for clean process exit.

---

## 4. HumanAgent vs SupervisorAgent — decision matrix

| I need… | Use |
|---|---|
| Simple approval gate | `HumanAgent(mode="single")` |
| Multi-turn review | `HumanAgent(mode="dialogue")` |
| Auto-approve after timeout | `HumanAgent(timeout=300, default="approved")` |
| Human as quality judge | `agent.loop(verify=HumanAgent(...))` |
| Human can investigate with tools | `SupervisorAgent(tools=[...])` |
| Human can retry previous steps | `SupervisorAgent(agents=[...])` |
| Human can inspect pipeline state | `SupervisorAgent(session=sess)` |
| Web UI instead of CLI | `HumanAgent(input_fn=my_callback)` / `SupervisorAgent(input_fn=my_callback)` |
| Async (websocket/queue) | `HumanAgent(ainput_fn=...)` / `SupervisorAgent(ainput_fn=...)` |

---

## 5. Cross-references

- Canonical API stub: [`00_quickref.md` §HumanAgent + §SupervisorAgent](00_quickref.md)
- Full tutorial with REPL screenshots: [`docs/course/13-human-in-the-loop.md`](../../docs/course/13-human-in-the-loop.md)
- Composition pattern 8 (HIL): [`docs/course/12-composition.md`](../../docs/course/12-composition.md)
- Tool wrapping (`as_tool()`): [`03_lazytool.md`](03_lazytool.md)
- Runnable example: [`examples/supervised_pipeline.py`](../../examples/supervised_pipeline.py)
- Scripted-input test pattern: `tests/unit/test_human.py:179–268`
