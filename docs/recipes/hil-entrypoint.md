# HIL as a pipeline entrypoint

Same primitive as [HIL as a clarifier](hil-clarify.md), but moved to
the head of the plan: the human types the question, the agent answers,
the pipeline terminates. This is the minimal "ask-and-answer" web app
skeleton — what was a leaf in `01_clarify.py` becomes the front door
here.

## Source

```python
--8<-- "examples/hil_app/02_entrypoint.py"
```

## Walkthrough

- **`human_agent(ui="web", name="ask")`** uses the bundled stdlib
  web UI (`http.server`-based, no dependencies). When the pipeline
  runs, the engine prints a localhost URL and opens the user's
  default browser to it.
- **`Step(ask, task="What would you like to know?")`** is the
  pipeline's first step. The human's submission flows forward as the
  `from_prev` input for the next step (the MockAgent).
- **Single turn** — the pipeline ends after the agent step completes.
  For multi-turn loops where the agent's reply feeds back into the
  next prompt, see [HIL as a chat loop](hil-chat-loop.md).

## Variations

- Swap `MockAgent(...)` for an `Agent(engine=LLMEngine(...))` to get
  a real LLM-backed assistant. The web UI carries the human's
  question, the LLM answers, the browser sees the result on
  submit-redirect.
- For a typed reply schema, set `output=ReplyModel` on the answering
  agent; the engine validates and re-prompts on failure.

## See also

- [HumanEngine guide](../guides/mid/human-engine.md) — full engine API.
- [HIL as a chat loop](hil-chat-loop.md) — extends this pattern with
  a `Plan` routing cycle for multi-turn conversation.
- [HIL with a custom UI](hil-custom-ui.md) — replace the stdlib
  browser form with your own surface (Slack, queue, mobile app).
