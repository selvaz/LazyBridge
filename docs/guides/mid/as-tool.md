# Agent.as_tool

Wrap an agent as a `Tool` that another agent can call. The default
shape is the same as passing the agent directly into `tools=[...]`;
the explicit form unlocks two extras: a different surface name than
the agent's own `name=`, and a `verify=` judge-and-retry loop around
every call.

## Signature

```python
agent.as_tool(
    name=None,                     # surface tool name; defaults to agent.name
    description=None,              # LLM-facing description; defaults to agent.description
    *,
    verify=None,                   # Agent or callable[[str], Any] — judge-and-retry gate
    max_verify=3,                  # max attempts when verify=...
) -> Tool
```

The returned `Tool` has the schema `(task: str) -> Envelope` (the
framework sets `returns_envelope=True` automatically so engines roll
up cost / token / latency metadata correctly through the call).

The implicit form — `Agent(tools=[other_agent])` — is canonical for
the no-rename, no-verify case. See
[Canonical vs sugar](../../concepts/canonical-vs-sugar.md) for the
exhaustive comparison.

## Synopsis

`as_tool` does three things, in order of how often you'll reach for
them:

1. **Wraps an agent so it satisfies the `Tool` contract.** The
   wrapped tool's `name` becomes the LLM-facing tool name; the
   wrapped tool's `description` is what the model reads when
   deciding whether to call it.
2. **Optionally renames.** If you pass `name="alias"`, that's what
   the LLM sees; the source agent's own `name=` is unchanged. The
   alias is also what `from_agent("alias")` reads from the `Store`
   (see [Store](store.md)).
3. **Optionally adds a `verify=` judge.** Every invocation runs up
   to `max_verify` times; after each run the judge sees the output
   and replies `"approved"` (case-insensitive) or
   `"rejected: <reason>"`. On rejection the judge's feedback is
   threaded into the next attempt's task. This is the "Option B"
   placement — judge sits at the **tool-call boundary**, not
   around the agent's own engine.

The implicit form (`tools=[other_agent]`) covers (1). When you need
(2) or (3), construct explicitly with `as_tool(...)`.

## When to use it

- **You want a different surface name than `agent.name=`.** The
  agent's own `name=` is the authoritative key for `Store` writes,
  graph nodes, and cost rollup. If you want the LLM to see a
  different name (e.g. `research` instead of `senior_researcher_v2`),
  pass `name="research"` to `as_tool`.
- **You want a different LLM-facing description.** Override
  `description=` to give the model a more focused or
  context-specific cue without changing the source agent.
- **You want a judge-gated call.** `verify=judge` runs the judge
  after every call, retrying up to `max_verify` times with the
  judge's feedback. Use this for high-stakes outputs where the
  parent agent should not see "first try, possibly wrong" results.
- **You want to expose `Agent.parallel(...)` as a single tool.**
  The `_ParallelAgent` returned by `Agent.parallel(...)` has its
  own `as_tool()` that folds the `list[Envelope]` into one
  envelope (labelled-text join), so the outer agent reads it
  uniformly.

## When NOT to use it

- **For the simple "agent A calls agent B" case.** Just pass the
  agent: `Agent(tools=[other_agent])` is canonical and saves the
  noise. The framework wraps it the same way internally.
- **When the wrapped agent has structured output.** The default
  schema is `(task: str) -> str`. If the parent needs a typed
  payload from the child, orchestrate with `Plan` and
  `Step(target=child, output=Model)` instead — `Plan` preserves
  typed envelopes between steps; `as_tool` flattens them.
- **As a substitute for `verify=` on the parent agent.**
  `Agent(verify=judge)` wraps the parent's *whole run*; this is
  the right choice when the policy applies to every output. The
  per-tool `verify=` is for when only specific sub-tools need
  gating.

## Example

```python
from lazybridge import Agent, LLMEngine


def search(query: str) -> str:
    """Search the web for ``query`` and return the top three hits."""
    return "..."


researcher = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[search],
    name="senior_researcher_v2",
)
judge = Agent(
    engine=LLMEngine(
        "claude-opus-4-7",
        system='Respond "approved" or "rejected: <reason>".',
    ),
    name="judge",
)


# 1) Implicit — pass the agent directly. Tool name = "senior_researcher_v2".
orchestrator = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[researcher],
)


# 2) Explicit alias — tool name "research", source agent name unchanged.
orchestrator = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[
        researcher.as_tool(
            name="research",
            description="Find three high-quality sources for a topic.",
        ),
    ],
)


# 3) Verified call — judge gates every research invocation, up to two attempts.
orchestrator = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[
        researcher.as_tool(
            name="research",
            verify=judge,
            max_verify=2,
        ),
    ],
)
result = orchestrator("write a paragraph on bee population trends")
print(result.text())


# 4) Parallel fan-out as a single tool — folds list[Envelope] into one envelope.
fan_out = Agent.parallel(researcher_us, researcher_eu, researcher_asia)
orchestrator = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[fan_out.as_tool(name="multi_region_research")],
)
```

## Pitfalls

- **Default schema is `(task: str) -> str`.** The wrapped agent's
  `output=Model` is **not** preserved in the tool's signature —
  the LLM sees a string contract regardless. For typed downstream
  consumption use a `Plan` step with `Step(output=Model)` instead.
- **`max_verify` is the upper bound, not the target.** A judge
  that's too strict can fail every attempt, costing `max_verify`
  full agent runs per call. Pick a defensible verdict prompt and
  keep `max_verify=2` or `3` unless you have a reason.
- **Verify retries cost tokens on both the child and the judge.**
  Each attempt is a full child-agent run plus a judge run. Use
  `verify=` only on calls that genuinely warrant the cost.
- **The alias is the Store key.** `as_tool(name="research")` makes
  `__agent_output__:research` the auto-write key — not
  `__agent_output__:senior_researcher_v2`. `from_agent("research")`
  reads it back. Keep aliases stable across runs that share a
  Store.
- **Long nested chains share one Session.** Pass `session=sess` on
  the outer agent only — inner agents (whether wrapped via
  `as_tool` or passed directly) inherit it. `usage_summary()`
  aggregates cost across the whole tree.
- **Pre-existing `as_tool()` callsites are still supported.**
  `tools=[agent.as_tool("name")]` and `tools=[agent]` are
  equivalent when the agent already has `name="name"`. Don't
  rewrite working code mechanically; prefer the implicit form for
  new code, and reach for `as_tool` when you actually want one of
  its three jobs.

## See also

- [Chain](chain.md) — when you want a fixed sequential pipeline
  rather than an LLM-directed dispatch.
- [Parallel](parallel.md) — `Agent.parallel(...).as_tool()` folds
  scripted fan-out into a single tool.
- [Tool](../basic/tool.md) — the surface every `as_tool()` call
  produces.
- *Guides → Full → verify=* (Phase 3) — verify around the parent
  agent's whole run versus per-tool gating.
- [Canonical vs sugar](../../concepts/canonical-vs-sugar.md) —
  the rules for when to call `as_tool` explicitly versus relying
  on the implicit `tools=[agent]` form.
