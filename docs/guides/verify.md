# verify=

**Use `verify=`** to wrap a run (or a tool call) in a judge/retry
loop: each output is scored, rejection feeds the judge's reason back
into the next attempt as context, capped at `max_verify`.  Two
placements:

* **Agent-level** (`Agent(..., verify=judge)`) — gates the whole run.
* **Tool-level** (`agent.as_tool(..., verify=judge)`) — gates each call
  through the wrapped agent, leaving the outer pipeline untouched.

## Example

```python
from lazybridge import Agent, Plan, Step

judge = Agent(
    "claude-opus-4-7",   # would typically be a cheaper model
    name="judge",
    system='Respond "approved" or "rejected: <short reason>".',
)

# Agent-level: final output gated.
writer = Agent("claude-opus-4-7", verify=judge, max_verify=2)
writer("write a haiku about bees")

# Tool-level (Option B): every call of synthesizer is gated.
synthesizer = Agent("claude-opus-4-7", name="synthesizer")
orchestrator = Agent(
    "claude-opus-4-7",
    tools=[synthesizer.as_tool("synth", verify=judge, max_verify=2)],
)

# Plan-level: one step gated, rest unchecked.
plan = Plan(
    Step(fetcher, name="fetch"),
    Step(Agent("claude-opus-4-7", verify=judge, name="summarise"),
         name="summarise"),
    Step(publisher, name="publish"),
)
```

## Pitfalls

- A strict judge + small ``max_verify`` silently returns poor output.
  Log the retry feedback during development so you know when you're
  hitting the cap.
- Judges as *callables* returning booleans don't produce feedback;
  retries reuse the same task. Return a string verdict if you want the
  feedback loop.
- Nested verify (Agent-level + tool-level + Plan-level all on the
  same path) is allowed but expensive. Pick one per agent unless
  you're intentionally stacking.
- Keep judges cheap (a smaller/faster model) and specific (one
  criterion per judge). Multi-criteria judges conflate failure modes
  and produce vague feedback.

!!! note "API reference"

    # Three placements, same judge contract.

    # 1. Agent-level (final output gate)
    Agent("model", verify=judge_agent, max_verify=3, ...)

    # 2. Tool-level (every call through the tool gated — "Option B")
    agent.as_tool(name, description, verify=judge_agent, max_verify=3)

    # 3. Plan-level (per-step, via agent-as-step with verify=)
    Plan(Step(Agent(..., verify=judge_agent), ...))

    # Judge contract
    # Judge receives the agent's output text (and the original task for
    # context) and must respond with a string starting with
    # "approved" (case-insensitive) to accept. Anything else is treated
    # as a rejection; its text is injected as feedback on the next retry.
    # Judges may be Agents or plain callables: `Callable[[str], Any]`.

!!! warning "Rules & invariants"

    - Retry loop: up to ``max_verify`` attempts. Final attempt is returned
      as-is even if still rejected (no infinite loop).
    - Rejection feedback is appended to the task string for the next
      attempt: ``f"{original_task}\n\nFeedback: {judge_verdict}"``.
    - Agent-level ``verify=`` gates the Agent's final output, regardless of
      which tool chain the engine chose internally.
    - Tool-level ``verify=`` (Option B via ``as_tool``) gates every
      invocation of that specific wrapped agent — useful when one
      sub-agent is the risky one and the rest is fine.
    - Plan-level is just a special case of agent-level: wrap the step's
      agent with its own ``verify=``.

## See also

- [Agent.as_tool](as-tool.md) — tool-level verify placement (Option B).
- [Plan](plan.md) — alternative: explicit retry steps via routing.
