# HumanEngine

`HumanEngine` is the simplest form of human-in-the-loop: the engine
pauses execution and waits for the human to supply a response. No REPL,
no tool calling, no agent retry — just a prompt and a response.

Use it when the human is a **participant** in a pipeline (an approver, a
reviewer, an annotator) rather than an operator. The `Agent` wrapping
`HumanEngine` slots identically into `chain` / `parallel` / `Plan`.

Two built-in UI modes are available:

| `ui=` | How it works |
|-------|--------------|
| `"terminal"` (default) | Prints the task to stdout and reads from stdin |
| `"web"` | Serves a self-contained HTML form on `localhost` and opens the browser |

When `output=` is a Pydantic model, the terminal variant prompts
field-by-field with coercion — integers become `int`, `bool` accepts
`yes/no`, lists are comma-separated. The web variant renders a proper
HTML form with one input per field.

For a full interactive REPL with tool calls and agent retry, use
`SupervisorEngine` instead.

## Terminal UI example

```python
from lazybridge import Agent, HumanEngine
from pydantic import BaseModel

class Review(BaseModel):
    approved: bool
    comment: str
    rating: int

reviewer = Agent(
    engine=HumanEngine(timeout=120, default="no comment"),
    output=Review,
    name="reviewer",
)

# In a pipeline: draft → review → finalise.
pipeline = Agent.chain(drafter, reviewer, finaliser)
pipeline("draft the release notes")
```

## Web UI example

`ui="web"` is a drop-in replacement that works in scripts, notebooks, and
async servers. It uses only the standard library — no extra dependencies.

```python
from lazybridge import Agent, HumanEngine
from pydantic import BaseModel

class Approval(BaseModel):
    approved: bool
    reason: str

approver = Agent(
    engine=HumanEngine(ui="web", timeout=300),
    output=Approval,
    name="approver",
)

# Prints a localhost URL and opens it in the browser.
# Waits up to 5 minutes for the human to submit the form.
result = approver("Please review the attached proposal and approve or reject it.")
print(result.payload.approved, result.payload.reason)
```

The server binds to `127.0.0.1` only and uses an OS-assigned port, so it is
never reachable from outside the local machine.

## Custom UI adapter

For web apps and CI environments where `stdin` / browser are unavailable, pass
any object with `async def prompt(task, *, tools, output_type) -> str`:

```python
from lazybridge.engines.human import _UIProtocol

class SlackUI(_UIProtocol):
    async def prompt(self, task: str, *, tools, output_type) -> str:
        await post_slack_message(task)
        return await wait_for_slack_reply()

reviewer = Agent(engine=HumanEngine(ui=SlackUI()), name="reviewer")
```

## Pitfalls

- `timeout=` uses `asyncio.wait_for`, not OS signals — works in async contexts
  but may not interrupt a blocking `input()` call on some platforms.
- `ui="web"` opens the system browser automatically. Pass `timeout=` to avoid
  waiting forever if no one submits the form (default is 3600 s).
- The web form accepts any POST submission from localhost — add application-level
  authentication if the port is forwarded to external interfaces.

!!! note "API reference"

    HumanEngine(
        *,
        ui: Literal["terminal", "web"] | _UIProtocol = "terminal",
        timeout: float | None = None,
        default: str | None = None,
    ) -> Engine

    Usage: Agent(engine=HumanEngine(), tools=[...], output=PydanticModel)

    # ui="web" serves an HTML form on localhost and waits for POST submission.
    # ui="terminal" reads from stdin (default).
    # Custom UI: any object with async def prompt(task, *, tools, output_type) -> str.

!!! warning "Rules & invariants"

    - `HumanEngine` returns the human's raw string (or JSON for Pydantic models)
      wrapped in an Envelope.  It implements the same `Engine` protocol as
      `LLMEngine`, so `Agent(engine=HumanEngine())` is a drop-in replacement.
    - `output=SomeModel` switches to per-field prompting (terminal) or a
      structured HTML form (web).
    - `timeout` triggers `default` if set, else raises `TimeoutError`.
    - Tool invocation is NOT handled by HumanEngine — the human types a raw
      string, they do not call tools interactively. Use `SupervisorEngine` for
      that.

## See also

[supervisor](supervisor.md), [agent](agent.md),
decision tree: [human_engine_vs_supervisor](../decisions/human-engine-vs-supervisor.md)
