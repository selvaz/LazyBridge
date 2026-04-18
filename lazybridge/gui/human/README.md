# gui.human — Browser-based input for HumanAgent / SupervisorAgent

Stdlib-only, opt-in, zero extra dependencies. Provides a drop-in `input_fn`
that opens a local web page instead of reading from stdin, so a human can
review the previous output, pick commands, and submit a response from the
browser.

Works for both `HumanAgent` and `SupervisorAgent` because both already
accept an `input_fn=` callback — this module just supplies a nicer one.

## Quick start

```python
from lazybridge import SupervisorAgent, LazyTool, LazySession, LazyAgent
from lazybridge.gui.human import web_input_fn

sess = LazySession()
researcher = LazyAgent("anthropic", name="researcher", session=sess)

fn = web_input_fn()  # opens a browser tab on 127.0.0.1:<ephemeral>
supervisor = SupervisorAgent(
    name="supervisor",
    agents=[researcher],
    session=sess,
    input_fn=fn,
)

writer = LazyAgent("openai", name="writer", session=sess)
pipeline = LazyTool.chain(researcher, supervisor, writer,
                          name="supervised", description="Research, supervise, write")
pipeline.run({"task": "AI safety report"})

fn.server.close()  # optional: shut the server down when done
```

The page shows:

- The previous agent's output (pre-formatted).
- Quick-command chips (optional; pass `quick_commands=` on `ask()`).
- A textarea for your response and a Submit button (⌘/Ctrl-Enter also submits).

Each `HumanAgent`/`SupervisorAgent` prompt becomes one page update — the
tab polls `/prompt` every 500 ms and refreshes when a new prompt arrives.

## Public API

```python
from lazybridge.gui.human import WebInputServer, web_input_fn

WebInputServer(
    *,
    host: str = "127.0.0.1",   # do not change without understanding the implications
    port: int = 0,              # 0 = ephemeral
    open_browser: bool = True,
    title: str = "LazyBridge — Human Input",
)

server.url               # http://127.0.0.1:<port>/?t=<token>
server.port              # actual bound port
server.token             # random 24-byte urlsafe token
server.input_fn          # Callable[[str], str]                — for HumanAgent.input_fn
server.ainput_fn         # Callable[[str], Awaitable[str]]     — for HumanAgent.ainput_fn
server.ask(prompt, *, timeout=None, quick_commands=None) -> str
server.aask(prompt, *, timeout=None, quick_commands=None) -> str   # async
server.close()           # also unblocks any in-flight ask()

web_input_fn(**kwargs) -> Callable[[str], str]
    # one-shot factory; returned callable exposes `.server` for cleanup
```

## Security

- Binds to `127.0.0.1` by default — not reachable off the host.
- Every `/prompt` and `/submit` request requires a random 24-byte token
  passed as `?t=...` or `X-Token`. The token is generated at server start
  and inlined into the page JavaScript.
- No persistence: prompts and responses live only in memory while a call
  to `ask()` is pending.

This is a **developer tool**. Do not expose the server outside localhost.

## Testing / CI

The GUI is optional; CI runs `SupervisorAgent` with a scripted `input_fn`
(see `tests/unit/test_human.py`). The web server itself has its own
integration tests under `tests/unit/gui/human/test_web_input.py`.
