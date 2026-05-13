# Step 5: Why multi-agent + sub-agent as a tool

In Step 4 you built a single agent with tools. That covers a lot. But as soon
as a task crosses two distinct concerns — *research* and *writing*,
*classification* and *response*, *plan* and *execute* — packing both jobs
into one prompt makes the model worse, not better.

The fix is **specialised agents**, each with a focused job. This step covers:

1. *Why* you'd split — the practical argument, with the same task tried two ways
2. *The first composition pattern* — **sub-agent as a tool**

The other three patterns (`Agent.chain`, `Agent.parallel`, `Plan`) get their
own steps. This one teaches the most flexible pattern: the parent LLM decides
when to delegate.

---

## The "one big agent" antipattern

Imagine: *"Write a one-paragraph note on a topic, with sources."*

The one-agent attempt looks tempting at first:

```python
from lazybridge import Agent, LLMEngine

def web_search(query: str) -> str:
    """Search the web for current information."""
    return f"[stub result for {query!r}]"

agent = Agent(
    engine=LLMEngine("claude-haiku-4-5", system="""
        You research topics and write notes.
        Use web_search to find facts. Then synthesise them into
        one tight paragraph. Cite sources. Don't pad. Don't hallucinate.
        Don't repeat yourself. Don't editorialise. Keep it under 120 words. ...
    """),
    tools=[web_search],
)

print(agent("AI agent frameworks in 2026").text())
```

Three things go wrong over time:

1. **The system prompt grows on every edge case.** "Also don't use jargon."
   "Also don't quote without citing." "Also keep the second sentence active
   voice." You end up with a 30-line system prompt that nobody can debug.
2. **The model conflates two jobs.** Research wants breadth ("look up many
   facts"). Writing wants compression ("pick the best three"). One model
   doing both produces mediocre output at both.
3. **You can't swap pieces.** Want to research with a cheap model and write
   with a smart one? Want to A/B-test two writing styles? You can't — it's
   one monolith.

---

## The split

Two agents, each with **one** job:

```python
researcher = Agent(
    engine=LLMEngine(
        "claude-haiku-4-5",
        system="You look up facts via web_search. "
               "Return 5–8 short bullet points. No prose.",
    ),
    tools=[web_search],
    name="researcher",
)

writer = Agent(
    engine=LLMEngine(
        "claude-haiku-4-5",
        system="You write a tight one-paragraph note "
               "from research bullets. 80–120 words.",
    ),
    name="writer",
)
```

Each agent has:

- A **short, focused system prompt** that a human can read in five seconds
- Its **own tool set** (researcher has `web_search`, writer has none)
- A `name` — the stable handle other agents will refer to

Once they exist, the question becomes: **how do they hand off work?**

---

## Pattern 1 — Sub-agent as a tool

In LazyBridge, **agents are tools**. You can pass any agent into another
agent's `tools=[...]`, and the parent treats it like any function:

```python
writer = Agent(
    engine=LLMEngine(
        "claude-haiku-4-5",
        system=(
            "You write a one-paragraph note on the topic the user provides. "
            "If you need current facts, call the researcher tool first."
        ),
    ),
    tools=[researcher],          # ← the researcher agent, passed as a tool
    name="writer",
)

print(writer("AI agent frameworks in 2026").text())
```

That's the entire composition. No supervisor object, no graph, no `add_node`.

When `writer` runs, the underlying model sees `researcher` in its tool list
with the description "researcher" (the agent's name) and decides — turn by
turn — whether to call it.

---

## What actually happens at runtime

Turn `verbose=True` on and run the example. You'll see exactly what the model
chose:

```text
[agent ▶ engine=LLMEngine model=claude-haiku-4-5 tools=[researcher]]
  user: AI agent frameworks in 2026
  assistant: ◆ tool_call researcher("agent frameworks 2026 overview")
    [agent ▶ engine=LLMEngine model=claude-haiku-4-5 tools=[web_search]]
      user: agent frameworks 2026 overview
      assistant: ◆ tool_call web_search(query="agent framework comparison 2026")
      tool[web_search]: [stub result for ...]
      assistant: - LangGraph: graph-based, ...
                 - CrewAI: role-based, ...
                 - LazyBridge: composition primitives, ...
    [done] turns=2  cost=$0.0006
  tool[researcher]: - LangGraph: ...
                    - CrewAI: ...
                    - LazyBridge: ...
  assistant: In 2026 the agent-framework landscape consolidates around three
             distinct approaches. LangGraph favours...
[done] turns=2  total_cost=$0.0012
```

Two things to notice:

- **Nested trace.** The `researcher` ran inside the `writer`'s loop. Its
  tools (`web_search`) appear indented underneath it. You always see the
  full chain of delegations.
- **Cost rolls up.** `total_cost=$0.0012` includes both the inner
  `researcher` call and the outer `writer` call. The envelope's
  `.metadata.cost_usd` is always the *aggregate* across nested agents — so
  you can budget by the top-level call.

---

## When the parent skips the delegation

Because the parent *decides* whether to call the sub-agent, it can skip the
delegation when it doesn't need it:

```python
print(writer("Define what a Python list comprehension is.").text())
```

Output (no tool call needed):

```text
[agent ▶ engine=LLMEngine model=claude-haiku-4-5 tools=[researcher]]
  user: Define what a Python list comprehension is.
  assistant: A list comprehension is a concise way to build a list...
[done] turns=1  total_cost=$0.0001
```

The writer recognised the question didn't require research and answered from
its own knowledge. No researcher cost paid. That's the strength of the
LLM-orchestrated pattern: *the workflow shape itself depends on the input.*

---

## Naming the sub-agent for the LLM

The model picks the sub-agent based on its **name** (and optionally its
description). The clearer the name, the better the routing.

By default, the agent's `name=` becomes the tool name. To override what the
parent LLM sees, use `.as_tool("...")` or attach a `description`:

```python
deep_research = researcher.as_tool(
    "deep_research",
    description="Use this when the user asks for facts that change over time "
                "(news, prices, current events). Costs ~1000 tokens per call.",
)

writer = Agent(
    engine=LLMEngine("claude-haiku-4-5", system="..."),
    tools=[deep_research],
    name="writer",
)
```

Treat the description like a contract: it's the only signal the parent has
when deciding whether the cost of delegating is worth it.

---

## Pros and cons of this pattern

| | Sub-agent as a tool |
|---|---|
| **Strength** | The parent decides at runtime. Workflow shape adapts to the input. |
| **Strength** | Zero glue code. Composition is one line. |
| **Cost** | An extra LLM turn at the parent level for each delegation decision. |
| **Cost** | Cost can balloon if the parent over-delegates (cap with `max_turns` on the parent). |

Use it when **the right sequence depends on the input**.

When the sequence is *always the same* — research, then write, then polish —
you don't want the parent paying tokens to "decide" the same routing every
time. That's what `Agent.chain` is for, coming up in Step 7.

But first: there's a more powerful use of sub-agents that LazyBridge makes
trivial — **using a different LLM as a judge** to verify the primary agent's
output. That's Step 6.

---

## Summary

| Concept | Syntax | What it does |
|---|---|---|
| Pass an agent as a tool | `tools=[other_agent]` | Parent LLM can call it like any function |
| Custom tool surface | `agent.as_tool("name", description="...")` | Override what the parent LLM sees |
| Cost rollup | `env.metadata.cost_usd` | Aggregate across all nested agents |
| Trace | `verbose=True` | Nested indented log of every delegation |

You've got the **flexible** composition primitive. Next: using a second agent
not as a tool, but as a **judge** — cross-model verification.

---

[**Step 6: Cross-model verification with `verify=` →**](06-verify.md){ .md-button .md-button--primary }

[← Step 4: Giving your agent tools](04-tools.md){ .md-button }
