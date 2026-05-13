# Step 5: Multiple agents working together

A single agent with tools (Step 4) handles a lot. But once a task crosses two
or three concerns — research, then writing, then fact-checking — packing
everything into one prompt makes the model worse, not better.

The fix is **specialised agents**: each one has a focused system prompt and a
small tool set, and they hand off work to each other.

This step shows the three main composition patterns in LazyBridge, when to use
each, and how they compare to the multi-agent approaches in CrewAI and
LangGraph.

---

## Why split agents at all?

A worked example: "Write a one-paragraph note on a topic, with sources."

**One-agent attempt:**

```python
agent = Agent(
    engine=LLMEngine("claude-haiku-4-5", system="""
        You research topics and write notes. Use web_search to find facts,
        then synthesise them into one paragraph. Always cite sources.
        Don't hallucinate. Don't pad. Be concise. Be accurate. ...
    """),
    tools=[web_search],
)
```

That system prompt grows every time you encounter a new edge case. The model
fights between two jobs (search well, write well), and the result is mediocre
at both.

**Two-agent split:**

```python
researcher = Agent(
    engine=LLMEngine("claude-haiku-4-5", system="You find facts via web_search."),
    tools=[web_search],
    name="researcher",
)
writer = Agent(
    engine=LLMEngine("claude-haiku-4-5", system="You write one tight paragraph."),
    name="writer",
)
```

Each agent has *one* job. Each system prompt is short. You can swap models per
agent (cheap model for research, smart model for writing). You can test each
agent in isolation.

This is the case for splitting. Now: how do you wire them together?

---

## Pattern 1 — Sub-agent as a tool (LLM-orchestrated)

You already saw this at the end of Step 4. Pass another agent into `tools=[...]`
and the parent agent calls it like any function:

```python
researcher = Agent(
    engine=LLMEngine("claude-haiku-4-5", system="You find facts via web_search."),
    tools=[web_search],
    name="researcher",
)

writer = Agent(
    engine=LLMEngine("claude-haiku-4-5", system="You write a one-paragraph note. Delegate research to the researcher tool."),
    tools=[researcher],            # ← sub-agent as a tool
    name="writer",
)

print(writer("Voynich Manuscript").text())
```

The `writer` decides *when* (and *whether*) to call `researcher`. If the model
already knows enough, it skips the delegation. If it needs facts, it calls.

**Pros:** flexible, dynamic, no rigid wiring. Each agent's `name` and
`description` become the tool surface the LLM sees.

**Cons:** the parent LLM has to decide the workflow at every step — that's
extra tokens and extra latency. Use it when *the workflow itself depends on
the input*.

---

## Pattern 2 — `Agent.chain(...)` (deterministic linear pipeline)

When the workflow is always the same — research, *then* write, *then* polish —
LLM-orchestrated routing is overhead. Use `Agent.chain` instead:

```python
pipeline = Agent.chain(researcher, writer)

print(pipeline("Voynich Manuscript").text())
```

`Agent.chain(a, b)` builds a linear pipeline: the output of `a` becomes the
input of `b`. No orchestrating LLM in the middle, no extra cost.

For a longer chain:

```python
draft = Agent.chain(researcher, writer, editor)
```

This is the cheapest multi-agent shape in LazyBridge. Use it whenever the
sequence is fixed.

---

## Pattern 3 — `Plan` with explicit `Step`s

Sometimes you need more control than a chain — branching, parallel work, named
data flow, conditional routing. That's what `Plan` is for.

```python
from lazybridge import Agent, LLMEngine, Plan, Step, from_step, from_prev


report = Agent(
    engine=Plan(
        Step("researcher"),
        Step("writer", task=from_prev, context=from_step("researcher")),
    ),
    tools=[researcher, writer],
    name="report",
)

print(report("Voynich Manuscript").text())
```

Two new ideas:

- **`Step("agent_name")`** — names which sub-agent runs at this step. The name
  matches `Agent(name="...")` from earlier.
- **Sentinels** (`from_prev`, `from_step`) — wire data flow explicitly. Here
  the writer's `task` is the researcher's output, and its `context` is also
  the researcher's output (named lookup, useful when steps fan out).

`Plan` is a deterministic engine — the DAG is fixed at construction time, so
you can lint and visualise it. The full Plan API is in
[Guides → Full → Plan](../guides/full/plan.md); for now treat it as "a chain,
plus branches and named data flow when you need them".

---

## Pattern 4 — `Agent.parallel(...)` for fan-out

When two agents can work independently, run them in parallel:

```python
anthropic_search = Agent(engine=..., tools=[search_anthropic_docs], name="search_a")
openai_search    = Agent(engine=..., tools=[search_openai_docs],    name="search_o")
google_search    = Agent(engine=..., tools=[search_google_docs],    name="search_g")

fan = Agent.parallel(anthropic_search, openai_search, google_search)

env = fan("How does each provider implement structured output?")
print(env.text())            # joined output across branches
for sub in env.payload:      # iterate individual branch envelopes
    print(sub.metadata.cost_usd)
```

`Agent.parallel` runs each child concurrently. The returned envelope's
`.text()` is a labelled join across branches; `.payload` is a list of the
per-branch envelopes (full metadata for each).

---

## Choosing between the patterns

| Pattern | Use when | Cost | Determinism |
|---|---|---|---|
| Sub-agent as a tool | Workflow depends on input — LLM picks the path | Extra LLM hops | None — the LLM routes |
| `Agent.chain` | Fixed linear pipeline (always a → b → c) | Cheapest | Full — same order every time |
| `Plan` | Branches, parallel sub-steps, named data flow | Moderate (deterministic, no extra LLM) | Full — DAG is compiled up front |
| `Agent.parallel` | Two or more agents truly independent | Concurrent, sums per-branch | Order-independent |

Mix freely: a `Plan` step's target can itself be a `Chain`; a chain stage can
contain an agent whose tools include sub-agents.

---

## How it looks in other frameworks

Brief comparison. Step 6 covers the full LangGraph / CrewAI breakdown.

??? example "CrewAI"

    ```python
    from crewai import Agent, Crew, Process, Task
    from crewai_tools import SerperDevTool

    search_tool = SerperDevTool()

    researcher = Agent(
        role="Researcher",
        goal="Find accurate facts.",
        backstory="You are a careful researcher.",
        tools=[search_tool],
        verbose=True,
    )
    writer = Agent(
        role="Writer",
        goal="Write a one-paragraph note.",
        backstory="You write tight, concrete prose.",
        verbose=True,
    )

    research_task = Task(
        description="Research {topic} and report findings.",
        agent=researcher,
        expected_output="A list of 5 verified facts.",
    )
    write_task = Task(
        description="Write a one-paragraph note from the research.",
        agent=writer,
        expected_output="One paragraph, 80-120 words.",
        context=[research_task],
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
    )

    print(crew.kickoff(inputs={"topic": "Voynich Manuscript"}))
    ```

    What CrewAI buys you: a strong opinion (roles, goals, backstories, tasks,
    crews). What it costs: more concepts to learn, less direct mapping to the
    underlying API, and you're locked into the Crew abstraction. For the same
    linear pipeline LazyBridge does it with one line:
    `pipeline = Agent.chain(researcher, writer)`.

??? example "LangGraph (supervisor pattern)"

    ```python
    from typing import Annotated, Literal
    from typing_extensions import TypedDict

    from langchain_anthropic import ChatAnthropic
    from langchain_core.tools import tool
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode


    @tool
    def web_search(query: str) -> str:
        """Search the web."""
        return f"[stub] facts about {query!r}"


    class State(TypedDict):
        messages: Annotated[list, add_messages]
        next_agent: str


    researcher_llm = ChatAnthropic(model="claude-haiku-4-5").bind_tools([web_search])
    writer_llm = ChatAnthropic(model="claude-haiku-4-5")


    def researcher_node(state: State):
        return {"messages": [researcher_llm.invoke(state["messages"])]}


    def writer_node(state: State):
        return {"messages": [writer_llm.invoke(state["messages"])], "next_agent": "end"}


    def supervisor(state: State) -> Literal["researcher", "writer", END]:
        # In real code: an LLM-driven supervisor decides next_agent
        if state.get("next_agent") == "end":
            return END
        last = state["messages"][-1]
        return "writer" if hasattr(last, "tool_calls") and not last.tool_calls else "researcher"


    builder = StateGraph(State)
    builder.add_node("researcher", researcher_node)
    builder.add_node("tools", ToolNode(tools=[web_search]))
    builder.add_node("writer", writer_node)
    builder.add_conditional_edges("researcher", supervisor)
    builder.add_edge("tools", "researcher")
    builder.add_edge("writer", END)
    builder.add_edge(START, "researcher")
    graph = builder.compile()

    print(graph.invoke({"messages": [{"role": "user", "content": "Voynich Manuscript"}]})["messages"][-1].content)
    ```

    LangGraph gives you full control over the graph — at the cost of writing
    it. The equivalent `Agent.chain(researcher, writer)` is one line, with the
    same `verbose=True` observability built in.

---

## The promised final example

The index page promised a research pipeline. Here it is:

```python
from lazybridge import Agent, LLMEngine


def web_search(query: str) -> str:
    """Look up current facts on the web (stub — wire to a real search API)."""
    return f"[stub web result for {query!r}]"


researcher = Agent(
    engine=LLMEngine(
        "claude-haiku-4-5",
        system="You look up facts via web_search. Return 5-8 short bullet points, no prose.",
    ),
    tools=[web_search],
    name="researcher",
)

writer = Agent(
    engine=LLMEngine(
        "claude-haiku-4-5",
        system="You write a tight one-paragraph note from research bullets. 80-120 words.",
    ),
    name="writer",
)

editor = Agent(
    engine=LLMEngine(
        "claude-haiku-4-5",
        system="You proofread for clarity. Keep the meaning; cut filler.",
    ),
    name="editor",
)

pipeline = Agent.chain(researcher, writer, editor)

env = pipeline("AI agent frameworks in 2026")
print(env.text())
print(f"\n[total cost ${env.metadata.cost_usd:.4f}, "
      f"{env.metadata.input_tokens}+{env.metadata.output_tokens} tokens]")
```

Notice what the envelope still gives you in a multi-agent run:

- **`env.text()`** — the final editor's output, the user-facing string
- **`env.metadata.cost_usd`** — the *aggregate* cost across all three agents
- **`env.metadata.nested_*`** — per-agent breakdown (LazyBridge rolls it up)

You did not write any orchestration code. `Agent.chain` handled the handoff,
prompt threading, and cost aggregation. Switching `researcher` to a different
model is one string change. Adding a fourth stage is one more positional
argument.

---

## Tracing the run

Turn on `verbose=True` on the chain (or on individual agents) to see every
delegation:

```python
pipeline = Agent.chain(researcher, writer, editor)
pipeline.verbose = True
pipeline("AI agent frameworks in 2026")
```

```text
[chain stage 1/3 ▶ agent=researcher]
  user: AI agent frameworks in 2026
  assistant: ◆ tool_call web_search(query="...")
  tool[web_search]: [stub] ...
  assistant: - LazyBridge ships v1 ...
             - LangGraph adds ...
[chain stage 2/3 ▶ agent=writer]
  user: - LazyBridge ships v1 ...  (passed from stage 1)
  assistant: In 2026 the agent-framework landscape ...
[chain stage 3/3 ▶ agent=editor]
  ...
[done] stages=3  total_tokens=812/410  cost=$0.0034
```

Every handoff is visible. When something goes wrong, you can see exactly which
agent produced the bad output.

---

## Summary

| Concept | Syntax | Use when |
|---|---|---|
| Sub-agent as a tool | `tools=[other_agent]` | The parent LLM decides when to delegate |
| Linear pipeline | `Agent.chain(a, b, c)` | Fixed workflow — always a → b → c |
| Explicit DAG | `Plan(Step("a"), Step("b", task=from_prev))` | Branches, named data flow, deterministic routing |
| Parallel fan-out | `Agent.parallel(x, y, z)` | Independent work, concurrent |
| Cost / token rollup | `env.metadata.cost_usd` | Aggregated across every nested agent |
| Tracing | `agent.verbose = True` | See every handoff |

You now have all the pieces to build real multi-agent systems with LazyBridge.
The last step compares it to LangGraph and CrewAI head-on, so you can pick the
right tool for *your* project.

---

[**Step 6: LazyBridge vs LangGraph vs CrewAI →**](06-vs-frameworks.md){ .md-button .md-button--primary }

[← Step 4: Giving your agent tools](04-tools.md){ .md-button }
