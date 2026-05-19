# HIL as a chat loop

The pipeline routes from the agent step back to the HIL step, forming
a cycle that lasts as long as the human keeps responding. Multi-turn
chat with no chat-specific framework code.

Three properties combine to make this work:

- **`Plan` supports cycles natively** via `Step.routes` pointing at
  an earlier step (`engines/plan/_types.py:116` — *"Loops are simply
  routes back to an earlier step"*).
- **`task=from_prev` carries the agent's previous reply forward**,
  so after the first turn every HIL prompt automatically shows what
  the agent just said. No manual history wiring.
- **The web `HumanEngine` keeps its HTTP server alive** across
  successive `prompt()` calls, so the browser tab stays on the same
  URL for the entire session. Post-submit auto-redirects to the next
  form; an "Agent is thinking…" placeholder shows during agent
  processing.

## Source

```python
--8<-- "examples/hil_app/03_chat_loop.py"
```

## Walkthrough

- **`Step(ask, routes={"farewell": _is_exit})`** — the HIL step has
  no explicit `task=`, so the default `from_prev` sentinel propagates
  the agent's last reply (or the Plan's initial input on turn 1)
  into the HIL prompt. `_is_exit` checks whether the human typed
  "exit"; if so, routing jumps to the terminal `farewell` step.
- **`Step(answer, routes={"ask": lambda _e: True})`** — every time
  the answer step runs, the predicate routes back to `ask`, closing
  the cycle.
- **`max_iterations=10_000`** — `Plan` has a default cap of `100`
  iterations as a safety net for runaway cycles. For an open-ended
  chat we set it high; the loop terminates when the human types
  "exit", not when the cap is hit.

## Stale-form safety

`HumanEngine`'s web UI carries a hidden `_epoch` field on every
rendered form, bumped per `prompt()` call. If a previous prompt
timed out and a later turn published a new form, a late submission
from the stale tab is rejected with `410 Gone` instead of being
decoded against the new prompt's schema. The browser auto-refreshes
to the current form. No correctness hole on the timeout path.

## Variations

- **Real LLM**: swap `MockAgent(...)` for an
  `Agent(engine=LLMEngine(...))` to get a chat with an actual model.
- **Persistent transcript**: the in-process `HumanEngine` doesn't
  keep history across restarts; if you need that, write the
  `(prompt, response)` pair to a `Store` in a small post-HIL step
  and read it back via `from_step("...")` on the next iteration.
- **Different exit signal**: replace `_is_exit` with any
  predicate over the envelope — empty input, a Pydantic flag, a
  cost threshold.

## See also

- [HumanEngine guide](../guides/mid/human-engine.md) — full engine API.
- [Plan](../guides/full/plan.md) — `Step.routes` and cycle semantics.
- [HIL as an entrypoint](hil-entrypoint.md) — the single-turn
  precursor; this recipe adds the routing cycle.
