# verify=

A judge-and-retry loop wrapped around an agent's output (or around a
specific tool call). Each output is graded; rejection feeds the
judge's reason back into the next attempt as feedback, capped at
`max_verify` attempts.

The hard sibling of `verify=` is [Guards](guards.md): a guard blocks
on policy violation; `verify=` accepts feedback and retries. Pick
the right tool for the policy.

## Signature

```python
# Three placements, same judge contract.

# 1. Agent-level — final output gate
Agent(
    engine=LLMEngine("claude-haiku-4-5"),
    verify=judge_agent,            # Agent or Callable[[str], Any]
    max_verify=3,                  # max attempts; default 3, must be >= 1
)

# 2. Tool-level — every call through the wrapped agent is gated
agent.as_tool(
    name="...",
    description="...",
    verify=judge_agent,
    max_verify=3,
)

# 3. Plan-level — wrap the step's agent with its own verify=
Plan(
    Step(
        target=Agent(
            engine=LLMEngine("claude-haiku-4-5"),
            verify=judge_agent,
            max_verify=3,
            name="summarise",
        ),
        name="summarise",
    ),
)
```

### Judge contract

- The judge receives the agent's output text (and the original task
  for context) and must respond with a string starting with
  `"approved"` (case-insensitive) to accept.
- Anything else is treated as rejection; the verdict text is
  injected as feedback on the next retry:
  `f"{original_task}\n\nFeedback: {judge_verdict}"`.
- Judges may be `Agent` instances or plain callables
  (`Callable[[str], Any]`).

## Synopsis

`verify=` is the soft, retryable counterpart to `Guards`. Where a
guard is a hard yes / no gate that ends the run on failure, a
verifier *re-asks* the agent with the judge's feedback baked into
the next prompt. After `max_verify` attempts the last result is
returned as-is, even if still rejected — there's no infinite loop.

There are three placements, all with the same contract:

- **Agent-level** (`Agent(engine=…, verify=judge)`) gates the
  agent's final output. Whatever tool chain the engine chose
  internally, the judge sees the eventual text and may force a
  retry.
- **Tool-level** (`agent.as_tool(verify=judge)`) gates every
  invocation of one specific wrapped agent — useful when one
  sub-agent is the risky one and the rest is fine. Already
  documented in passing in [as-tool](as-tool.md); `verify=` is
  the dedicated reference for the loop semantics.
- **Plan-level** is just a special case of agent-level: wrap the
  step's agent with its own `verify=`. No special primitive.

## When to use it

- **High-stakes outputs that should not ship "first try".**
  Drafts, summaries of regulated content, customer-facing replies.
- **Quality control with feedback.** Unlike a guard, `verify=`
  gives the agent a chance to fix what the judge complained
  about. Use it when "wrong answer" is recoverable, not just
  blockable.
- **Per-tool gating.** When one sub-agent in a hierarchy is
  noisier than the rest, gate just that one with
  `agent.as_tool(verify=judge)` instead of taxing the whole
  pipeline.
- **Targeted Plan steps.** When one step in a `Plan` is the
  quality-critical one (the summary, the final draft), wrap just
  its agent with `verify=` — leave the rest unchecked.

## When NOT to use it

- **Hard policy violations.** PII leakage, content-safety
  failures, schema violations — those need [Guards](guards.md).
  `verify=` retries; a guard refuses and ends the run.
- **CI / batch grading.** When you want to test "does this agent
  generally produce good output?" offline, use [Evals](evals.md)
  instead. `verify=` runs every single time the agent is invoked.
- **As a structured-output validator.** That's what `output=`
  does (with `max_output_retries=`). The framework already
  re-prompts on Pydantic validation errors; you don't need
  `verify=` on top.
- **Multi-criteria judges.** A judge that grades fluency,
  accuracy, and tone simultaneously produces vague feedback.
  Either run two separate `verify=` loops (expensive) or pick one
  criterion and accept the rest will need a different mechanism.

## Example

```python
from lazybridge import Agent, LLMEngine, Plan, Step


judge = Agent(
    engine=LLMEngine(
        "claude-haiku-4-5-20251001",   # cheap-tier judge
        system='Respond "approved" or "rejected: <short reason>".',
    ),
    name="judge",
)


# 1) Agent-level — final output gated, up to 2 attempts.
writer = Agent(
    engine=LLMEngine("claude-haiku-4-5"),
    verify=judge,
    max_verify=2,
    name="writer",
)
result = writer("write a haiku about bees")
print(result.text())


# 2) Tool-level (Option B) — every call of synthesizer is gated;
#    the orchestrator's other tools run unchecked.
synthesizer = Agent(
    engine=LLMEngine("claude-haiku-4-5"),
    name="synthesizer",
)
orchestrator = Agent(
    engine=LLMEngine("claude-haiku-4-5"),
    tools=[
        synthesizer.as_tool(
            name="synth",
            verify=judge,
            max_verify=2,
        ),
    ],
)


# 3) Plan-level — only the summarise step is gated.
fetcher = Agent(
    engine=LLMEngine("claude-haiku-4-5"),
    name="fetch",
)
publisher = Agent(
    engine=LLMEngine("claude-haiku-4-5"),
    name="publish",
)
summariser = Agent(
    engine=LLMEngine("claude-haiku-4-5"),
    verify=judge,
    max_verify=2,
    name="summarise",
)

plan = Agent(
    engine=Plan(
        Step(target=fetcher,    name=fetcher.name),
        Step(target=summariser, name=summariser.name),
        Step(target=publisher,  name=publisher.name),
    ),
)


# 4) Callable judge — boolean verdict, no feedback loop.
def at_least_three_lines(output: str) -> bool:
    return output.count("\n") >= 2

short_writer = Agent(
    engine=LLMEngine("claude-haiku-4-5"),
    verify=at_least_three_lines,
    max_verify=2,
)
```

## Pitfalls

- **A strict judge + small `max_verify` silently returns poor
  output.** After the cap is hit, the last attempt is returned
  even if still rejected. Log the retry feedback during
  development so you know when you're hitting the cap; consider
  raising `max_verify` or relaxing the policy.
- **Callable judges returning booleans don't produce feedback.**
  Retries reuse the same task. Return a string verdict
  (`"approved"` or `"rejected: <reason>"`) if you want the
  feedback loop. A `bool`-returning callable is acceptable when
  the failure mode is binary and "try again" alone is enough.
- **Nested verify is allowed but expensive.** Agent-level +
  tool-level + Plan-level on the same path stacks the loops.
  Pick one per agent unless you're intentionally building
  defence in depth.
- **Keep judges cheap and specific.** Use a smaller / faster
  model. One criterion per judge — multi-criteria judges
  conflate failure modes and produce vague feedback. Two
  single-criterion judges chained at different levels often
  produce better results than one multi-criterion judge at one
  level.
- **`verify=` operates on `Envelope.text()`.** When the agent
  has structured output (`output=Model`), the judge sees the JSON
  serialisation, not the model instance. Phrase the judge's
  policy accordingly, or write a callable judge that does
  `json.loads(...)` first.
- **`max_verify=1` disables the retry but keeps the gate.** The
  agent runs once, the judge grades, and the result is returned
  whatever the verdict. Effectively a non-blocking quality
  warning. Use a real `max_verify=2` or `3` if you actually want
  the retry semantics.
- **Cost rollup.** Each retry is a full agent run plus a judge
  run. With a `Session=`, the cost rollup includes every
  attempt; check `session.usage_summary()` for the true cost
  of a verify-heavy workflow.

## See also

- [Guards](guards.md) — the hard counterpart: blocks on policy
  failure with no retry. Pair them: hard guards for "must never
  pass", `verify=` for "should usually be good".
- [As tool](as-tool.md) — the surface for tool-level `verify=`
  placement.
- [Evals](evals.md) — offline / CI grading; `verify=` is the
  runtime sibling.
- [Agent](../basic/agent.md) — `verify=` is a first-class
  `Agent` kwarg alongside `guard=`, `output=`, and `tools=`.
