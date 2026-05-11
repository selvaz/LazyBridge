# HumanEngine

A drop-in replacement for `LLMEngine` whose "model" is a person at
a terminal (or a custom UI). The agent prompts the human, receives
their input, and treats it as the engine's response. Use it as an
approval gate or as a structured form filler.

## Signature

```python
from lazybridge import Agent
from lazybridge.ext.hil import HumanEngine, human_agent

# Canonical — Agent + HumanEngine
HumanEngine(
    *,
    ui="terminal",                 # "terminal" | "web" | _UIProtocol
    timeout=None,                  # seconds; on expiry triggers default= or raises TimeoutError
    default=None,                  # str returned on timeout
)

agent = Agent(
    engine=HumanEngine(timeout=120, default="no comment"),
    output=Review,                 # Pydantic model → field-by-field prompt
    name="reviewer",
)


# Sugar — same agent, less plumbing
agent = human_agent(
    timeout=120,
    default="no comment",
    output=Review,
    name="reviewer",
)
```

The sugar `human_agent(...)` lives in `lazybridge.ext.hil` and
forwards engine kwargs (`ui`, `timeout`, `default`) to `HumanEngine`
and the rest (`output=`, `name=`, `session=`, …) to `Agent`. See
[Canonical vs sugar](../../concepts/canonical-vs-sugar.md) for the
exhaustive comparison.

> **Status: ext.** Available out of the box once `lazybridge` is
> installed; lives under `lazybridge.ext.hil` to respect the
> core/ext import boundary.

## Synopsis

`HumanEngine` implements the same `Engine` protocol as `LLMEngine`,
so `Agent(engine=HumanEngine())` swaps in cleanly anywhere an LLM
agent fits. The terminal UI prompts the human with the task,
captures the typed response, and returns it as the
`Envelope.payload`. When `output=` is a Pydantic model, the prompt
goes field-by-field instead of asking for a single string.

There are two HIL engines in `lazybridge.ext.hil`:

- **`HumanEngine`** — approval gate / form filler. The human types
  a string (or fills fields). No tool calls, no agent retries —
  this engine is the lightweight variant.
- **`SupervisorEngine`** — full REPL with tool dispatch, agent
  retries, and store inspection. Lands in Phase 3 (Full tier);
  reach for it when the human needs to *do work*, not just decide.

## When to use it

- **Approval gates in pipelines.** Drop a `HumanEngine` agent into
  a `Plan` step or `Agent.chain` between drafting and finalising;
  the pipeline halts until the human types a verdict.
- **Structured human review.** When `output=` is a Pydantic model,
  the terminal UI prompts field-by-field — useful for review
  forms (rating, comment, approved boolean) without writing a
  per-field prompt loop.
- **Manual data entry inside an agent flow.** Sometimes the
  cheapest "tool" is a human: an agent that needs an OAuth code,
  a CAPTCHA solution, or a domain expert's call.
- **Tests / demos that need deterministic input.** Pass a custom
  `ui=` adapter implementing `prompt(task, *, tools, output_type)
  -> str` to script the human side.

## When NOT to use it

- **The human needs to call tools.** `HumanEngine` does not
  dispatch tools — the human can only type a raw string. Use
  `SupervisorEngine` (Full tier) for that.
- **Long-running async workflows where blocking on input is
  wrong.** The terminal UI blocks the current process. For web
  apps, supply a custom `ui=` adapter that wires into your event
  system (queue, websocket, …).
- **As a substitute for `verify=`.** `HumanEngine` is the engine
  itself, not a judge wrapping an LLM agent's output. If you want
  "LLM produces output → human approves before returning",
  combine an LLM agent inside a `Plan` with a human approval step
  — see [as-tool](as-tool.md) and [verify=](verify.md) for
  policy gating.

## Example

```python
from pydantic import BaseModel

from lazybridge import Agent, LLMEngine, Plan, Step
from lazybridge.ext.hil import HumanEngine


class Review(BaseModel):
    approved: bool
    comment: str
    rating: int                    # 1..5


# 1) Standalone — a single agent that prompts a person.
reviewer = Agent(
    engine=HumanEngine(timeout=120, default="no comment"),
    output=Review,
    name="reviewer",
)
result = reviewer("draft #42 — please review and rate")
if result.payload.approved:
    print("✓ approved", result.payload.comment)


# 2) Inside a pipeline — draft → review → finalise.
drafter = Agent(
    engine=LLMEngine("gpt-5.4-mini"),
    name="drafter",
)
finaliser = Agent(
    engine=LLMEngine("gpt-5.4-mini"),
    name="finaliser",
)

pipeline = Agent(
    engine=Plan(
        Step(target=drafter,   name=drafter.name),
        Step(target=reviewer,  name=reviewer.name),
        Step(target=finaliser, name=finaliser.name),
    ),
    name="release-pipeline",
)
pipeline("draft the v1.2 release notes")


# 3) Custom UI adapter for a web app — the prompt callable resolves
#    when the user submits a form.
class WebUI:
    def __init__(self, queue):
        self._queue = queue

    async def prompt(self, task, *, tools, output_type):
        await self._queue.publish({"task": task, "schema": output_type})
        return await self._queue.await_response()


reviewer_web = Agent(
    engine=HumanEngine(ui=WebUI(my_queue), timeout=600),
    output=Review,
    name="reviewer-web",
)
```

## Pitfalls

- **The terminal UI blocks the current process.** This is
  intended — a synchronous human-in-the-loop step shouldn't race
  the rest of the agent's tool loop. For non-blocking flows
  supply a custom `ui=` adapter or run the engine on a worker
  thread.
- **`timeout` uses the event loop, not signals.** It works in
  async contexts but may hang in tightly-blocking sync nests
  (a custom `ui` that calls `input()` inside a synchronous-only
  callsite). Pair with an `ainput_fn` adapter when you need
  cancellation.
- **`output=Model` switches the terminal UI to per-field
  prompting.** Without `output=`, the human types one free-form
  string. The model class is the trigger — there is no separate
  "form mode" flag.
- **`HumanEngine` is not a judge.** It produces the output, it
  doesn't grade an LLM's. To grade an LLM's output with a human
  in the loop, run the LLM in one step and a `HumanEngine` agent
  in the next; or use a custom callable as `verify=`.
- **`default=` is only applied on timeout.** If the human enters
  an empty string, that empty string is the response. If you need
  empty-input handling, validate after the call (or use a
  Pydantic `output=` model with a non-empty constraint).

## See also

- [Chain](chain.md) — typical pattern for inserting `HumanEngine`
  mid-pipeline.
- [Guards](guards.md) — hard input/output gates that don't need a
  human; complementary, not redundant.
- [verify=](verify.md) — judge-and-retry placement when an LLM
  judges instead of a human.
- *Guides → Full → SupervisorEngine* (Phase 3) — the heavier
  cousin: a full REPL with tools, agent retry, and store
  inspection.
- [Canonical vs sugar](../../concepts/canonical-vs-sugar.md) —
  `human_agent(...)` vs `Agent(engine=HumanEngine(...))`.
