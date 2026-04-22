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

## What HumanEngine emits (audit trail)

Every human input produces a **`HIL_DECISION` event** on the active
`Session`'s event log, plus the usual `AGENT_START` / `AGENT_FINISH`
pair.  The decision payload records `kind="input"`, the task the human
saw, and the first 500 characters of the result — so an auditor can
reconstruct who was asked what, and what they answered, without
replaying the agent.

If the human's response fails to coerce into a requested Pydantic
`output_type`, HumanEngine does **not** hand back a raw string
pretending to be a model.  It emits a `TOOL_ERROR` event with
`kind="structured_output_coercion"` and returns an Envelope whose
`error.type == "StructuredOutputCoercionError"`.  Downstream code
sees `env.ok == False` and can retry, fall back, or surface the
problem — the usual Envelope error-path handling applies.

## Terminal UI example

```python
# What this shows: a reviewer step that wraps a human sign-off inside
# an otherwise-automated pipeline. HumanEngine is a drop-in for
# LLMEngine — the chain surrounding it doesn't care whether the
# reviewer is a human or a model.
# Why output=Review: the terminal UI then prompts EACH FIELD one at a
# time ("approved (bool):", "comment (str):", "rating (int):") and
# coerces the raw text via Pydantic TypeAdapter. If the human types
# "yes" for a bool it becomes True; if they type "abc" for an int the
# field is re-prompted (up to 3 retries) with the validation error
# shown inline. On final failure the run returns an error Envelope
# with type="StructuredOutputCoercionError" — the caller can retry,
# fall back, or surface the issue. No silent corruption.

from lazybridge import Agent, HumanEngine
from pydantic import BaseModel

class Review(BaseModel):
    approved: bool
    comment: str
    rating: int

reviewer = Agent(
    # timeout: wall-clock deadline on the prompt (asyncio.wait_for).
    # default: if the timeout fires, use this string instead of raising
    #          TimeoutError — essential for unattended CI runs.
    engine=HumanEngine(timeout=120, default="no comment"),
    output=Review,
    name="reviewer",
)

# In a pipeline: draft → review → finalise.
# The reviewer sees whatever the drafter produced as its task. When
# the human submits, the review payload becomes the finaliser's task.
pipeline = Agent.chain(drafter, reviewer, finaliser)
pipeline("draft the release notes")
```

What just happened: the chain paused at the reviewer step, presented
the drafter's output plus field-by-field prompts, and emitted a
`HIL_DECISION` event on the session before handing the coerced
`Review` instance back to the finaliser.  If no `Session` is attached,
the events are simply skipped — HumanEngine behaves identically to an
LLM engine without one.

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

For web apps, chat platforms, or CI runners where neither stdin nor a
local browser is available, pass **any** object with a coroutine
method matching `async def prompt(task, *, tools, output_type) -> str`.
The engine awaits it exactly like the built-in UIs.

```python
# What this shows: routing the prompt through Slack instead of stdin.
# The adapter returns whatever raw string the human types in reply —
# HumanEngine then runs the same Pydantic coercion / retry loop as if
# they had typed it at the terminal.
# Why the minimal surface: everything coercion-related (structured
# output, TOOL_ERROR on failure, HIL_DECISION on success) lives in
# HumanEngine itself. Adapters only worry about transport.

from lazybridge.engines.human import _UIProtocol

class SlackUI(_UIProtocol):
    async def prompt(self, task: str, *, tools, output_type) -> str:
        # tools is a list[Tool] (usually empty for HumanEngine) and
        # output_type is the Pydantic class / str you passed on the
        # Agent — useful if you want to render a typed form yourself
        # instead of letting HumanEngine prompt field-by-field.
        await post_slack_message(task)
        return await wait_for_slack_reply()   # raw string

reviewer = Agent(engine=HumanEngine(ui=SlackUI()), name="reviewer")
```

For **testing** a pipeline that includes a HumanEngine, the easiest
adapter is `scripted_inputs` from `lazybridge.testing` — see the
[testing guide](testing.md) for deterministic HIL harnesses.

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
