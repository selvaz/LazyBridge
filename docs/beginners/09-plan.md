# Step 9: Explicit DAGs with `Plan` & sentinels

`Agent.chain` (Step 7) is great when stage N+1 needs *exactly* stage N's
output, and nothing else. But the moment a stage needs *more than that* —
the original user query *plus* the previous draft, or the output of a
specific earlier step (not just the previous one) — chain isn't enough.

That's where `Plan` comes in. A `Plan` is an explicit, compiled DAG where
each step is **named** and **addressable**, and you wire data between steps
using little type-safe markers called **sentinels**.

This is LazyBridge's most powerful composition primitive. It's also the one
you'll reach for least often — most workflows fit `chain` or `parallel`.
This step is the foundation; the [full Plan guide](../guides/full/plan.md)
covers advanced features (branching, parallel bands, checkpoints, resume).

---

## When `chain` isn't enough

Take a familiar example: a writer that needs to consult research **and**
remember the original user question.

With `chain`, the writer sees only the researcher's output. The original
question gets lost:

```
USER QUERY ──► researcher ──► research_bullets ──► writer
                                                     ▲
                                       Where is the user query?  ✗
```

You can work around this with prompt engineering ("the user asked X — your
input below is the research...") but it's clunky and the model can lose
track. The clean fix is a Plan with two **explicit data flows**:

```
USER QUERY ──┬──► researcher ──► research_bullets ──┐
             │                                       ▼
             └──────────── (original) ─────────► writer
```

Two arrows feeding the writer. That shape doesn't fit `chain` — it fits
`Plan`.

---

## The minimum Plan

```python
from lazybridge import Agent, LLMEngine, Plan, Step, from_prev

researcher = Agent(engine=LLMEngine("claude-haiku-4-5", system="..."), name="researcher")
writer     = Agent(engine=LLMEngine("claude-haiku-4-5", system="..."), name="writer")

pipeline = Agent(
    engine=Plan(
        Step("researcher"),                          # 1st step: run "researcher"
        Step("writer", task=from_prev),              # 2nd step: writer gets researcher's output
    ),
    tools=[researcher, writer],                       # the agents Plan can look up by name
    name="pipeline",
)
```

This is **literally what `Agent.chain(researcher, writer)` does internally** —
`chain` is just sugar over a linear Plan. The interesting bits start once
you have *more than one* data flow to wire.

Notice three things:

- **`Step("researcher")`** names the step. The name must match an agent in
  `tools=[...]` (or you pass `target=agent` directly).
- **`tools=[researcher, writer]`** on the outer Agent is the *registry*
  Plan steps look up. Same `tools=` you've seen since Step 4 — agents are
  tools.
- **`task=from_prev`** says "the task for this step is the previous step's
  output". `from_prev` is your first sentinel.

---

## Sentinels — the data flow markers

Sentinels are tiny, type-safe markers that say *where* a step's input comes
from. They're imported from the top-level `lazybridge` namespace.

The four you'll use most:

| Sentinel | What it returns | Example |
|---|---|---|
| `from_prev` | The output of the **immediately preceding** step | `task=from_prev` |
| `from_step("name")` | The output of a **specific named** earlier step | `context=from_step("researcher")` |
| `from_start` | The **original task** that entered the Plan | `task=from_start` |
| `from_agent("name")` | The agent's **last persisted output** (cross-run, needs a `Store`) | Advanced — see Guides |

Sentinels don't carry data — they're *references*. The Plan compiler reads
them at construction time and validates them against the step names you've
declared. If you typo a name (`from_step("researcer")`) the Plan raises a
`PlanCompileError` immediately — not at runtime, not three minutes into a
billing run.

---

## Two channels per step — `task=` vs `context=`

This is the part that makes Plan *do something chain can't*. Each `Step`
takes up to two data inputs:

```python
Step("writer",
     task=from_start,                  # the user message the writer will see
     context=from_step("researcher"))  # extra info appended/system-attached
```

| Channel | Conceptually | Default |
|---|---|---|
| `task=` | The **prompt** the step's agent receives as its user message | The previous step's output (`from_prev`) |
| `context=` | **Additional information** attached to the agent's context | None |

For our researcher → writer example: the user's original question goes in
`task=` (so the writer sees the question directly), and the researcher's
bullets go in `context=` (extra info the writer can use).

---

## Putting it together

The full Plan that solved the motivating problem:

```python
from lazybridge import Agent, LLMEngine, Plan, Step, from_start, from_step


def web_search(query: str) -> str:
    """Stub web search."""
    return f"[stub web result for {query!r}]"


researcher = Agent(
    engine=LLMEngine(
        "claude-haiku-4-5",
        system="You look up facts via web_search. Return 5–8 bullet points.",
    ),
    tools=[web_search],
    name="researcher",
)

writer = Agent(
    engine=LLMEngine(
        "claude-opus-4-7",
        system="You answer the user's question using the research bullets in your context. "
               "Plain English, one paragraph (80–120 words).",
    ),
    name="writer",
)

pipeline = Agent(
    engine=Plan(
        Step("researcher"),                                       # no task= → uses from_start by default
        Step("writer",
             task=from_start,                                     # user's question (NOT the research)
             context=from_step("researcher")),                    # research as side info
    ),
    tools=[researcher, writer],
    name="pipeline",
)

print(pipeline("Why are GPUs better than CPUs for training neural networks?").text())
```

The writer now sees the **user's question** as its task and the **research
bullets** in its context. It can quote facts from the research while
answering the actual question. That's not possible with `chain`.

---

## Why "compiled" matters

Plans are **compiled at construction time**, not at run time. This is a
specific design choice with payoff:

```python
pipeline = Agent(
    engine=Plan(
        Step("researcher"),
        Step("write", task=from_step("research")),    # ← typo: "research" not "researcher"
    ),
    tools=[researcher, writer],
)
# raises PlanCompileError immediately:
#   Unknown step reference "research" in from_step.
#   Did you mean "researcher"?
```

Compare with raw orchestration code where the typo silently produces a wrong
output three steps into a 30-minute run. The compile-time check is the
reason Plan exists as a separate primitive instead of being "chain plus
some extra tricks".

---

## Tracing — see every step

`verbose=True` on the Plan-backed Agent shows step-by-step execution with
data flow visible:

```python
pipeline = Agent(engine=Plan(...), tools=[...], verbose=True)
pipeline("Why are GPUs better than CPUs for training neural networks?")
```

Output (abbreviated):

```text
[plan ▶ pipeline  steps=2]
  [step 1/2: researcher  model=claude-haiku-4-5]
    task: Why are GPUs better than CPUs ...   (from_start by default)
    assistant: ◆ tool_call web_search("GPU vs CPU neural net training")
    tool[web_search]: [stub result]
    assistant: - Massive parallel ALUs ...
               - High memory bandwidth ...
               - Tensor-core hardware ...
  [step 2/2: writer  model=claude-opus-4-7]
    task:    Why are GPUs better than CPUs ...   (from_start)
    context: - Massive parallel ALUs ...           (from_step("researcher"))
    assistant: GPUs outperform CPUs for neural network training because ...

[done] steps=2  total_cost=$0.0034
```

Each step prints both channels (`task:` and `context:`) so you can see
exactly what was wired in — and where it came from.

---

## A wider preview — what else `Plan` can do

This page is the foundation. The full Plan covers:

- **Parallel bands** — run several steps concurrently inside a Plan, then
  fan in via `from_parallel("band_name")`
- **Conditional routing** — `Step("a", routes={"b": predicate})` — covered
  in Step 10
- **Checkpoints & resume** — persist Plan state, replay from a step
- **GraphSchema** — typed payload contracts between steps

Each of these is documented in the [full Plan guide](../guides/full/plan.md).
For the beginner tutorial, the sentinel pattern above is enough to cover
about 80% of real-world DAG shapes.

---

## When to use `Plan` (and when not to)

| Symptom | Use |
|---|---|
| Linear pipeline, single-arrow data flow | `Agent.chain` (Step 7) — Plan is overkill |
| Concurrent independent branches on same input | `Agent.parallel` (Step 8) |
| Stage needs the original query *and* a previous step's output | **Plan + sentinels** |
| Step needs to fan in from N parallel sub-steps | Plan + `from_parallel` |
| Workflow depends on the user input at runtime | Sub-agent as a tool (Step 5) |
| Want a step to run only sometimes | Plan + `routes=` (Step 10) |

Plan is the "I need explicit wiring" tool. Reach for chain or parallel when
the data flow is simple — that's most of the time.

---

## How other frameworks express explicit DAGs

??? example "LangGraph (the closest equivalent)"

    LangGraph is itself a DAG framework, so this is the fair comparison:

    ```python
    from typing import Annotated, TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langchain_anthropic import ChatAnthropic

    class State(TypedDict):
        question: str          # user's original task
        research: str          # filled by the researcher node
        answer: str            # filled by the writer node

    researcher_llm = ChatAnthropic(model="claude-haiku-4-5").bind_tools([web_search])
    writer_llm     = ChatAnthropic(model="claude-opus-4-7")

    def researcher_node(state: State):
        reply = researcher_llm.invoke([{"role": "user", "content": state["question"]}])
        return {"research": reply.content}

    def writer_node(state: State):
        prompt = (f"Question: {state['question']}\n\n"
                  f"Research:\n{state['research']}\n\n"
                  f"Answer in one paragraph.")
        reply = writer_llm.invoke([{"role": "user", "content": prompt}])
        return {"answer": reply.content}

    builder = StateGraph(State)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", END)
    graph = builder.compile()

    out = graph.invoke({"question": "Why are GPUs better than CPUs for training neural networks?",
                        "research": "", "answer": ""})
    print(out["answer"])
    ```

    The data-flow logic is fully manual: you declare a `State` schema, you
    spell out which node writes which key, and you write `state["question"]`
    everywhere you want the original input. With LazyBridge that's
    `from_start`. With LangGraph it's State plumbing.

    Both compile and validate at construction time. LangGraph gives you
    finer-grained control (custom reducers, conditional edges, sub-graphs);
    LazyBridge gives you a smaller surface for the common case.

??? example "CrewAI"

    CrewAI's `Task.context=[other_task]` lets a task see a previous task's
    output:

    ```python
    research_task = Task(description="Research {topic}.", agent=researcher,
                        expected_output="...")
    write_task = Task(
        description="Answer the question {topic} using the research.",
        agent=writer,
        expected_output="One paragraph.",
        context=[research_task],
    )
    crew = Crew(agents=[researcher, writer],
                tasks=[research_task, write_task],
                process=Process.sequential)
    ```

    Works for simple "one previous task as context" cases. For more complex
    shapes (named steps, parallel fan-in, the original-query-plus-research
    case above), you're back to prompt engineering inside the `description`
    — there's no `from_step` equivalent.

---

## Summary

| Concept | Syntax | What it does |
|---|---|---|
| Build a DAG | `Plan(Step("a"), Step("b"), ...)` | Named, ordered steps |
| Use as engine | `Agent(engine=Plan(...), tools=[a, b], name="...")` | Plan needs the agent registry |
| Default flow | `Step("name")` with no `task=` | Same as `task=from_prev` |
| Previous output | `task=from_prev` | One-hop linear |
| Specific step | `task=from_step("name")` | Pick any earlier step's output |
| Original task | `task=from_start` | The Plan's input |
| Side info | `context=from_step("...")` or `from_start` | Goes alongside the task |
| Compile errors | `PlanCompileError` at construction time | Typos caught before any LLM call |
| Tracing | `verbose=True` shows `task:` + `context:` per step | See the wiring at runtime |

You've now seen four composition primitives — sub-agent-as-tool, chain,
parallel, and Plan. The next step adds the missing piece: **conditional
flow** — making a step run only under certain conditions.

---

[**Step 10: Routing — conditional branching →**](10-routing.md){ .md-button .md-button--primary }

[← Step 8: Parallel work](08-parallel.md){ .md-button }
