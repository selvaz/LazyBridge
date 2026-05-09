# `verify=` placement

> **Agent level, tool level, or Plan step level?**

`verify=` retries with judge feedback. Three placements, same judge
contract — pick the narrowest scope that covers the policy.

## Decision tree

```text
Gating the final output of a single agent?
    → Agent(engine=LLMEngine("…"), verify=judge, max_verify=3)

One specific sub-agent (used as a tool by a parent) is the risky
one; rest of the run is fine?
    → risky_subagent.as_tool(verify=judge, max_verify=2)
      Agent(engine=LLMEngine("…"),
            tools=[risky_subagent.as_tool(verify=judge)])

One step of a declared Plan needs a judge; other steps don't?
    → Plan(
          Step(target=Agent(engine=LLMEngine("…"),
                            verify=judge,
                            name="summarise")),
          …,
      )

Want a judge on every tool call (not output)?
    → That isn't what verify= does — see Guards instead.
      Agent(engine=LLMEngine("…"),
            tools=[…],
            guard=ContentGuard(input_fn=…, output_fn=…))
```

## Quick reference

| Scope | Placement |
|---|---|
| Whole agent's final output | `Agent(verify=judge)` |
| One sub-agent's invocations only | `subagent.as_tool(verify=judge)` |
| One Plan step only | `Step(target=Agent(verify=judge, name=…))` |
| Every tool invocation (input/output filter) | **Not `verify=`** — use a [Guard](../guides/mid/guards.md) |

## Notes

- **`verify=` is the soft sibling of Guards.** A `Guard` is a
  hard yes / no gate that ends the run on failure; `verify=`
  re-prompts with the judge's reason and retries up to
  `max_verify` times. Pick by what should happen on failure.
- **The judge contract.** Returns a string starting with
  `"approved"` (case-insensitive) to accept; anything else is
  rejection, and the verdict text is appended to the next
  attempt's task as feedback.
- **Callable judges that return `bool` don't produce feedback.**
  Retries reuse the same task. Return a string verdict if you
  want the feedback loop.
- **Nested verify is allowed but expensive.** Agent-level +
  tool-level + Plan-level on the same path stacks the loops;
  pick one per agent unless you're intentionally building
  defence in depth.
- **Keep judges cheap and specific.** Use a smaller / faster
  model and one criterion per judge — multi-criteria judges
  conflate failure modes and produce vague feedback.

## See also

- [`verify=`](../guides/mid/verify.md) — full reference for the
  three placements, retry-budget tradeoffs, callable vs Agent
  judges.
- [Guards](../guides/mid/guards.md) — the hard-gate alternative
  for input / output filtering.
- [Evals](../guides/mid/evals.md) — the offline / CI sibling of
  `verify=` (batch grading instead of live retries).
