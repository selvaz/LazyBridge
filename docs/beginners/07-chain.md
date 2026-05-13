# Step 7: Sequential pipelines with `Agent.chain`

Step 5 gave you the **flexible** composition primitive — sub-agent as a tool,
where the parent LLM decides at runtime whether to delegate.

That's powerful, but it has a cost: every delegation decision is an extra LLM
hop, and the parent always has to think before routing. When the workflow is
**always the same** (research → write → edit, every time), you don't want the
LLM "deciding" the same routing on every call.

That's where `Agent.chain` comes in: a fixed, deterministic, linear pipeline
with no orchestrating model in the middle.

---

## The simplest possible chain

```python
from lazybridge import Agent, LLMEngine

step_a = Agent(engine=LLMEngine("claude-haiku-4-5"), name="step_a")
step_b = Agent(engine=LLMEngine("claude-haiku-4-5"), name="step_b")

pipeline = Agent.chain(step_a, step_b)

env = pipeline("a task")
print(env.text())
```

`Agent.chain(a, b)` returns **a single agent** — `pipeline` — that behaves
like any other agent: you call it, you get an envelope back.

Internally LazyBridge builds a deterministic `Plan` (the engine you'll see
properly in Step 9) where `step_a` runs first, its output feeds `step_b`,
and `step_b`'s output is the final answer.

---

## How data flows between stages

This is the only rule you need to remember:

> The **text** output of stage N becomes the **user prompt** of stage N+1.

There's no clever routing, no context object, no shared state. Each stage
sees one string — the previous stage's `.text()` — as its user message.
Whatever system prompt and tools the stage was constructed with stay
unchanged.

```
        ┌─────────────┐
input → │  step_a     │ → text_a
        └─────────────┘             ┌─────────────┐
                          text_a → │  step_b     │ → final_text
                                    └─────────────┘
```

If a stage needs more than just the previous output (e.g. *both* the
original user query and the prior step's draft), that's the case for
`Plan` + sentinels — Step 9. For now: linear, single-hop.

---

## A practical three-stage pipeline

The classic research → write → edit:

```python
from lazybridge import Agent, LLMEngine


def web_search(query: str) -> str:
    """Look up current facts on the web (stub)."""
    return f"[stub web result for {query!r}]"


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
        system="You write one tight paragraph (80–120 words) "
               "from research bullets. Plain English.",
    ),
    name="writer",
)

editor = Agent(
    engine=LLMEngine(
        "claude-haiku-4-5",
        system="You proofread for clarity and concision. "
               "Keep the meaning; cut filler.",
    ),
    name="editor",
)

pipeline = Agent.chain(researcher, writer, editor)

env = pipeline("AI agent frameworks in 2026")
print(env.text())
```

Three specialised agents, one composition line. No glue code, no orchestrator,
no state management. The pipeline is itself an `Agent` — you can store it,
pass it around, even nest it inside another chain.

---

## One model per stage — cost-aware composition

The chain doesn't force all stages to share a model. Use the cheapest model
that can do each job:

```python
researcher = Agent(engine=LLMEngine("claude-haiku-4-5", system="..."), tools=[web_search], name="researcher")
writer     = Agent(engine=LLMEngine("claude-opus-4-7",  system="..."), name="writer")   # smarter for prose
editor     = Agent(engine=LLMEngine("gpt-5.4-mini",     system="..."), name="editor")    # cheap for cleanup

pipeline = Agent.chain(researcher, writer, editor)
```

The research stage is bulk + structured; cheap model is fine. The writing
stage is the one humans actually read; pay for quality. Cleanup is
mechanical; cheap again. With raw SDKs this would mean three different
clients with three different request shapes — here it's three strings.

---

## Adding `verify=` to a stage

You learned `verify=` in Step 6. It composes with chains in the obvious way —
each stage can have its own judge:

```python
def length_judge(text: str) -> str:
    words = len(text.split())
    return "approved" if 80 <= words <= 120 else f"Length {words}; must be 80–120."


writer_verified = Agent(
    engine=LLMEngine("claude-opus-4-7", system="..."),
    verify=length_judge,
    max_verify=3,
    name="writer",
)

pipeline = Agent.chain(researcher, writer_verified, editor)
```

The judge fires only when `writer_verified` runs, not on the other stages.
`verify=` and `chain` are orthogonal — they compose freely.

You can also `verify=` the **whole pipeline** by passing it as a kwarg to
`Agent.chain` itself:

```python
overall_judge = Agent(
    engine=LLMEngine("gpt-5.4-mini",
                     system="Reply 'approved' if the note answers the question."),
    name="overall_judge",
)

pipeline = Agent.chain(researcher, writer, editor, verify=overall_judge)
```

That judges only the *final* output (editor's), but if it rejects, the
**entire chain reruns** with feedback baked into the input. Useful when the
quality issue could be at any stage.

---

## Tracing — see every stage

`verbose=True` on the pipeline reveals each stage:

```python
pipeline = Agent.chain(researcher, writer, editor, verbose=True)
pipeline("AI agent frameworks in 2026")
```

Output (abbreviated):

```text
[chain pipeline ▶ stage 1/3: researcher  model=claude-haiku-4-5]
  user: AI agent frameworks in 2026
  assistant: ◆ tool_call web_search(query="agent frameworks 2026")
  tool[web_search]: [stub] ...
  assistant: - LangGraph: graph DSL ...
             - CrewAI: role-based ...
             - LazyBridge: composition primitives ...

[chain pipeline ▶ stage 2/3: writer  model=claude-opus-4-7]
  user: - LangGraph: graph DSL ...   (← researcher's output)
  assistant: In 2026 the agent-framework landscape consolidates ...

[chain pipeline ▶ stage 3/3: editor  model=gpt-5.4-mini]
  user: In 2026 the agent-framework landscape consolidates ...
  assistant: The 2026 agent-framework landscape consolidates around three ...

[done] stages=3  total_cost=$0.0034  total_tokens=812/410
```

Three observations:

- **Stage boundaries are visible.** When something looks wrong, you can see
  *which stage* introduced it.
- **The hand-off is just a string.** Stage 2's `user:` message is exactly
  stage 1's assistant output. No magic.
- **One total cost.** The envelope's `.metadata.cost_usd` is the sum across
  all stages — including tools and (if you used them) judge calls.

---

## What you skip vs raw SDKs

To do the same thing without LazyBridge you'd write something like this
(Anthropic flavour):

```python
import anthropic
client = anthropic.Anthropic()

# Stage 1: research — plus the manual tool loop you wrote in Step 4
r1 = client.messages.create(
    model="claude-haiku-4-5",
    system="...researcher prompt...",
    messages=[{"role": "user", "content": user_task}],
    tools=[{"name": "web_search", ...}],
    max_tokens=1024,
)
# ... tool loop here ...
research_text = "..."

# Stage 2: write
r2 = client.messages.create(
    model="claude-opus-4-7",
    system="...writer prompt...",
    messages=[{"role": "user", "content": research_text}],
    max_tokens=1024,
)
draft_text = r2.content[0].text

# Stage 3: edit — need a SECOND client for OpenAI
from openai import OpenAI
oa = OpenAI()
r3 = oa.responses.create(
    model="gpt-5.4-mini",
    instructions="...editor prompt...",
    input=draft_text,
)
print(r3.output_text)
```

Three call sites, two SDKs, two different request/response shapes, manual
string threading, plus the tool loop in stage 1. Adding a fourth stage means
another copy-paste block.

LazyBridge's equivalent is one expression: `Agent.chain(researcher, writer, editor)`.

---

## How other frameworks express the same pipeline

??? example "CrewAI (sequential process)"

    ```python
    from crewai import Agent, Crew, Process, Task

    researcher = Agent(role="Researcher", goal="Find facts.",
                       backstory="Careful.", tools=[search_tool])
    writer = Agent(role="Writer", goal="Write a paragraph.",
                   backstory="Concrete prose.")
    editor = Agent(role="Editor", goal="Polish.",
                   backstory="Catches filler.")

    research_task = Task(
        description="Research {topic}.",
        agent=researcher,
        expected_output="Bullet points.",
    )
    write_task = Task(
        description="Write a paragraph from the research.",
        agent=writer,
        expected_output="80-120 word paragraph.",
        context=[research_task],
    )
    edit_task = Task(
        description="Polish the paragraph.",
        agent=editor,
        expected_output="Polished paragraph.",
        context=[write_task],
    )

    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, write_task, edit_task],
        process=Process.sequential,
    )

    print(crew.kickoff(inputs={"topic": "AI agent frameworks in 2026"}))
    ```

    Concept count: `Agent`, `role`, `goal`, `backstory`, `Task`,
    `description`, `expected_output`, `context`, `Crew`, `process`. Strong
    abstraction, lots of moving parts. The same chain in LazyBridge is
    **one line**.

??? example "LangGraph (sequential StateGraph)"

    ```python
    from typing import Annotated, TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langchain_anthropic import ChatAnthropic

    researcher_llm = ChatAnthropic(model="claude-haiku-4-5").bind_tools([web_search_tool])
    writer_llm     = ChatAnthropic(model="claude-opus-4-7")
    editor_llm     = ChatAnthropic(model="gpt-5.4-mini")  # + a langchain-openai dep

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def researcher_node(state): return {"messages": [researcher_llm.invoke(state["messages"])]}
    def writer_node(state):     return {"messages": [writer_llm.invoke(state["messages"])]}
    def editor_node(state):     return {"messages": [editor_llm.invoke(state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("editor", editor_node)
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "editor")
    builder.add_edge("editor", END)
    graph = builder.compile()

    result = graph.invoke({
        "messages": [{"role": "user", "content": "AI agent frameworks in 2026"}],
    })
    print(result["messages"][-1].content)
    ```

    For a *fixed* sequential pipeline, the explicit graph is overhead.
    LangGraph shines on complex routing — for a plain chain, `Agent.chain`
    is the right tool.

---

## When NOT to use `Agent.chain`

Reach for a different pattern when:

| Symptom | Use instead |
|---|---|
| Some stages should run in parallel (independent work) | `Agent.parallel` — Step 8 |
| The next stage needs more than just the previous output | `Plan` + sentinels — Step 9 |
| The workflow depends on the input (dynamic routing) | Sub-agent as a tool — Step 5 |
| You want a stage to run only under a condition | `Plan` with `routes=` — Step 9 / Guides |

Otherwise — when the sequence is fixed — `Agent.chain` is the right shape.
It's cheaper than LLM-orchestrated routing and simpler than a full Plan.

---

## Summary

| Concept | Syntax | What it does |
|---|---|---|
| Build a chain | `Agent.chain(a, b, c)` | Linear pipeline; output of N feeds input of N+1 |
| Result | The chain itself is an `Agent` | Use it everywhere a regular agent fits |
| Data flow | Stage N's `text()` → Stage N+1's user prompt | Single string per hop |
| Per-stage models | Each agent has its own `LLMEngine("...")` | Pay only what each stage needs |
| Per-stage verify | `Agent(..., verify=judge)` on individual stages | Local quality gate |
| Whole-chain verify | `Agent.chain(..., verify=judge)` | Rerun the whole chain on rejection |
| Tracing | `Agent.chain(..., verbose=True)` | See every stage's input and output |
| Cost rollup | `env.metadata.cost_usd` | Aggregate across all stages |

Sequential is the most common shape — most "production" pipelines look like
this. Next we'll cover the case where stages should fire **at the same time**,
not in sequence.

---

[**Step 8: Parallel work with `Agent.parallel` →**](08-parallel.md){ .md-button .md-button--primary }

[← Step 6: Cross-model verification](06-verify.md){ .md-button }
