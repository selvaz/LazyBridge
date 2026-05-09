# HumanEngine vs SupervisorEngine

> **Which human-in-the-loop engine?**

Two HIL engines under `lazybridge.ext.hil`. The split is by *what
the human is allowed to do* during the gate, not by integration
shape.

## Decision tree

```text
Simple "wait for input / approve / fill a form"?
    → HumanEngine
      Agent(engine=HumanEngine(timeout=120, default="approve"),
            output=ReviewForm)
      # one prompt, one answer (or per-field prompts when output= is
      # a Pydantic model)

Interactive REPL where the operator can call tools, retry agents
with feedback, and inspect the store?
    → SupervisorEngine
      Agent(engine=SupervisorEngine(tools=[search],
                                    agents=[researcher],
                                    store=store))
      # commands: continue [text] | retry <agent>: <feedback> |
      #           store <key>     | <tool>(<args>)

Automated verification at runtime (no human, LLM judge)?
    → verify=judge_agent
      # NOT human-in-the-loop — see verify-placement.md
```

## Quick reference

| Need | Use |
|---|---|
| Approve / reject / fill form | **`HumanEngine`** |
| REPL with tool dispatch + agent retry + store inspection | **`SupervisorEngine`** |
| LLM-as-judge with retry feedback | `verify=judge_agent` (not HIL) |

## Notes

- **`HumanEngine` is the lighter variant.** One prompt, one
  answer; with `output=PydanticModel` the terminal UI prompts
  field-by-field for a structured form.
- **`SupervisorEngine` is the REPL.** The operator calls tools
  registered on the engine, retries any agent passed in
  `agents=[…]` with feedback, and inspects `store=` keys via the
  `store <key>` command. Heavier but interactive.
- **Both are drop-in `Engine`s.** `Agent(engine=HumanEngine(...))`
  / `Agent(engine=SupervisorEngine(...))` plug into any pipeline
  that takes an `Engine`. Use them as a `Plan` step like any
  other agent.
- **Sugar exists.** `human_agent(...)` and `supervisor_agent(...)`
  in `lazybridge.ext.hil` skip the explicit `Agent(engine=…)`
  wrap. Use the sugar for one-liners; the canonical form when
  the engine choice should be visible at the call site.
- **Set `timeout=` for unattended pipelines.** Both engines hang
  forever on `timeout=None` (the default) when no human shows
  up. Pair `timeout=` with `default=` for graceful fallback.

## See also

- [HumanEngine](../guides/mid/human-engine.md) — full reference,
  custom UI adapter via `ui=`.
- [SupervisorEngine](../guides/full/supervisor.md) — REPL
  command surface, scripted-input tests, async UI via `ainput_fn`.
- [`verify=` placement](verify-placement.md) — automated LLM
  judge, distinct from HIL.
- [Canonical vs sugar](../concepts/canonical-vs-sugar.md) —
  `human_agent(...)` / `supervisor_agent(...)` mapped to their
  `Agent(engine=...)` canonical equivalents.
