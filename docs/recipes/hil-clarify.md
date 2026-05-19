# HIL as a clarifier

A multi-step pipeline pauses to ask the human for a missing piece of
information mid-flight. The `HumanEngine` step is just another node in
the `Plan` — indistinguishable, from the composition's point of view,
from any other `Agent`-shaped step.

This is the leaf-tool role of HIL: the agent is in charge, and the
human is consulted only when needed.

## Source

```python
--8<-- "examples/hil_app/01_clarify.py"
```

## Walkthrough

- **`human_agent(name="ask_city")`** returns an `Agent` whose engine is
  `HumanEngine`. The factory accepts the same `**agent_kwargs` as the
  unified `Agent(...)` constructor, so it composes with `Plan` like
  any other step.
- **`Step(ask_city, task="What city is the user in?")`** overrides
  the default `task=from_prev` so the prompt the human sees is fixed
  ("What city is the user in?"), independent of what the previous
  step produced. The human's response flows forward via `from_prev`
  as the input to the next step.
- **Terminal UI by default** — `human_agent()` without `ui=` writes
  the prompt to stdout and reads the response from stdin. No browser,
  no extra dependencies. Swap to `ui="web"` for a localhost form
  (see [HIL as a chat loop](hil-chat-loop.md)).

## Variations

- For typed input (e.g. ask the human for a Pydantic `Address`
  model), pass `output=Address` to `human_agent(...)`; the terminal
  UI prompts field-by-field with type-aware coercion.
- To gate the pipeline rather than enrich it ("is this OK?"), use
  the same shape but make the next step's `routes=` depend on the
  human's `yes`/`no` answer.

## See also

- [HumanEngine guide](../guides/mid/human-engine.md) — engine API and
  `_UIProtocol` extension hook.
- [HIL as an entrypoint](hil-entrypoint.md) — the same primitive,
  promoted to the front of the pipeline.
