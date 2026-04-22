# verify=

`verify=` is LazyBridge's LLM-as-judge primitive. A cheap judge agent
(or a plain callable) sits on the output path; if it says "approved",
pass through; if not, inject the judge's feedback as retry fuel and try
again up to `max_verify` times.

Three placements address three different failure modes:

* **Agent-level** is the broad-strokes gate: "I don't trust this
  agent's final output without a second opinion."
* **Tool-level (Option B)** is precision: "This one sub-agent is the
  risky one; gate it specifically, and let the rest run freely."
  Put it on the `as_tool` wrapper so every call is vetted.
* **Plan-level** is when a specific step needs its own gate and the
  rest of the pipeline doesn't — identical to agent-level, just
  applied to the Step's agent.

Judge design: keep the judge cheap (a smaller/faster model) and
specific (one criterion per judge). Multi-criteria judges conflate
failure modes and produce vague feedback.

## Example

```python
from lazybridge import Agent, LLMEngine, Plan, Step

# Judge needs a system prompt; ``system=`` belongs on the engine.
# Use a cheaper model in real code.
judge = Agent(
    engine=LLMEngine(
        "claude-opus-4-7",
        system='Respond "approved" or "rejected: <short reason>".',
    ),
    name="judge",                    # label used in session.usage_summary()
)

# Agent-level:
#   verify=judge   → after the engine produces its final output, hand it
#                    to ``judge``. If judge's verdict starts with
#                    "approved", pass through. Otherwise rerun with the
#                    verdict appended to the task as feedback.
#   max_verify=2   → give the writer at most 2 attempts before returning
#                    the last (possibly rejected) output as-is.
writer = Agent("claude-opus-4-7", verify=judge, max_verify=2)
writer("write a haiku about bees")

# Tool-level (Option B): every call of synthesizer is gated.
#   name="synthesizer" is the tool name seen by the orchestrator LLM
#   (and the default Tool.name when this Agent is wrapped via as_tool()).
synthesizer = Agent("claude-opus-4-7", name="synthesizer")
orchestrator = Agent(
    "claude-opus-4-7",
    # as_tool("synth", ...) overrides the tool name to "synth".
    # verify=/max_verify= here gate every call through this tool, not
    # the orchestrator's own final output.
    tools=[synthesizer.as_tool("synth", verify=judge, max_verify=2)],
)

# Plan-level: one step gated, rest unchecked.
# Step(target, name="…") — ``name`` is the step id used by sentinels
# (from_step("fetch")), checkpoints, and the graph view. ``target`` is
# the callable / Agent that actually runs the step.
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

[agent](agent.md), [as_tool](as-tool.md), [plan](plan.md),
[guards](guards.md), [evals](evals.md),
decision tree: [verify_placement](../decisions/verify-placement.md)
