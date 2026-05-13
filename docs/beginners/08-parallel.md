# Step 8: Parallel work with `Agent.parallel`

`Agent.chain` (Step 7) runs stages **in sequence**. That's right when each
stage depends on the previous one — research, *then* write, *then* edit.

But many real workflows have a different shape: several agents could run
**at the same time**, because they don't depend on each other. Asking three
different model families the same question. Fanning out a query across
three search backends. Extracting entities, sentiment, and topics from the
same document. None of these need to be sequential.

`Agent.parallel` is the primitive for that case.

---

## The simplest parallel fan-out

```python
from lazybridge import Agent, LLMEngine

researcher_a = Agent(engine=LLMEngine("claude-haiku-4-5"), name="claude_view")
researcher_b = Agent(engine=LLMEngine("gpt-5.4-mini"),     name="gpt_view")
researcher_c = Agent(engine=LLMEngine("gemini-3-flash-preview"), name="gemini_view")

panel = Agent.parallel(researcher_a, researcher_b, researcher_c)

env = panel("What are the trade-offs of agent frameworks in 2026?")
print(env.text())
```

`Agent.parallel(a, b, c)` returns **a single agent** (a `ParallelAgent`) that
takes one task, runs all branches **concurrently** on that task, and bundles
the results into one envelope.

Three things to notice:

1. **Same task, three branches.** Each agent receives the exact same input.
2. **Independent execution.** No branch sees any other branch's output.
   They're isolated.
3. **One envelope back.** From the caller's point of view it's still just an
   `agent(...).text()` call.

---

## What the returned envelope looks like

The result envelope is your hand-off to the rest of the program. Two things
are guaranteed:

### `.text()` — a labelled join

```text
[claude_view]
Agent frameworks fall into three camps: graph-based ...

[gpt_view]
The trade-offs centre on flexibility versus determinism ...

[gemini_view]
You can think of agent frameworks along two axes: ...
```

Each branch's output is prefixed with its agent name in square brackets. If
you just want a quick "show me everything", this is enough.

### `.payload` — list of per-branch envelopes

When you need typed access (per-branch cost, individual `.error`, per-branch
metadata), the payload is a list of `Envelope`s, one per branch, in input
order:

```python
env = panel("...")

for branch in env.payload:
    print(branch.metadata.run_id, branch.metadata.cost_usd, branch.text()[:60])
```

Output:

```text
run_01HX... 0.00009  Agent frameworks fall into three camps: graph-based ...
run_01HY... 0.00012  The trade-offs centre on flexibility versus determinism ...
run_01HZ... 0.00008  You can think of agent frameworks along two axes ...
```

The same envelope shape you've seen from a single agent — just N of them.

!!! tip "Async-typed access"
    If you're already in async code and want the per-branch envelopes as
    objects (not as a payload list), `await panel.run_branches(task)` returns
    the list directly. The synchronous `panel(task)` is the canonical entry.

---

## Before the next example — `Agent.parallel` vs parallel tool calls

This is the single most common source of confusion in this step. Read it
now, before the bigger example below.

The LLM can *already* emit parallel tool calls. You saw it in Step 4 —
the weather agent fetched Rome and Tokyo concurrently in a single turn.
So when you pass three sub-agents as `tools=[a, b, c]`, the model may
choose to call all three at once, *or* just one, *or* none. **The model
decides.**

`Agent.parallel(a, b, c)` is the *opposite*: **you decide**, at
composition time, that all three run on every call. No model in the
middle, no skipping.

| | `Agent.parallel(a, b, c)` | `tools=[a, b, c]` on a regular Agent |
|---|---|---|
| Who decides to fan out? | **You** — always runs every branch | **The LLM** — picks at runtime, may skip branches |
| Cost predictability | Predictable: N branches always | Variable: depends on the model's plan |
| Same input to each branch? | Yes (identical) | LLM crafts each tool call's arguments |
| Use when | "I know I want all of these" | "Let the model figure out what's needed" |

**Rule of thumb:** if you want all N branches *every time*, use
`Agent.parallel`. If the right subset depends on the query, expose them
as `tools=` on an orchestrating agent and let the model decide.

The example below uses `Agent.parallel` because we *want* every search
backend hit on every query — we're trying to get coverage, not be smart
about which one to ask.

---

## A concrete use case — multi-source search

A common production shape: fan a query out across several search backends and
have the LLM synthesise the results.

```python
def search_anthropic_docs(query: str) -> str:
    """Search Anthropic's docs portal."""
    return f"[anthropic docs result for {query!r}]"

def search_openai_docs(query: str) -> str:
    """Search OpenAI's platform docs."""
    return f"[openai docs result for {query!r}]"

def search_google_docs(query: str) -> str:
    """Search Google AI docs."""
    return f"[google ai docs result for {query!r}]"


anthropic_search = Agent(
    engine=LLMEngine("claude-haiku-4-5", system="Look up Anthropic-specific facts."),
    tools=[search_anthropic_docs],
    name="anthropic_search",
)
openai_search = Agent(
    engine=LLMEngine("claude-haiku-4-5", system="Look up OpenAI-specific facts."),
    tools=[search_openai_docs],
    name="openai_search",
)
google_search = Agent(
    engine=LLMEngine("claude-haiku-4-5", system="Look up Google/Gemini-specific facts."),
    tools=[search_google_docs],
    name="google_search",
)

# Step 1 — fan out the same query to three search agents in parallel
fanout = Agent.parallel(anthropic_search, openai_search, google_search)

# Step 2 — pass the combined result to a synthesiser
synthesiser = Agent(
    engine=LLMEngine(
        "claude-opus-4-7",
        system="You synthesise multi-source research into one tight paragraph. "
               "Cite each source by its label.",
    ),
    name="synthesiser",
)

pipeline = Agent.chain(fanout, synthesiser)
print(pipeline("How does each provider implement structured output?").text())
```

This is **parallel inside a chain**: the fan-out produces a single envelope
whose `text()` is the labelled join, which becomes the input to the
synthesiser. Two LazyBridge primitives, one expression each, the same
familiar shape.

The wall-clock saving is real: three searches that would take 3×*t* in series
take ≈ *t* in parallel (modulo network jitter). Cost still adds up — you pay
for every branch.

---

## Bounding concurrency and timeouts

Two control knobs you should set in production:

```python
panel = Agent.parallel(
    researcher_a,
    researcher_b,
    researcher_c,
    concurrency_limit=2,    # never run more than 2 branches at once
    step_timeout=30.0,      # any branch exceeding 30s is cancelled
)
```

- **`concurrency_limit`** caps the simultaneous runs. Useful when:
    - Each branch hits the same rate-limited API
    - You have many branches and want to be a polite neighbour
    - The default (`None` = unbounded) would saturate your provider quota
- **`step_timeout`** caps the wall clock of each branch individually. The
  envelope of a timed-out branch arrives with `branch.error` set and
  `branch.ok == False`; the other branches still complete normally.

Both are optional. For small fan-outs (2–5 branches) you usually leave them
unset.

---

## Cost and latency — the trade-off

Parallel saves time but not cost. With four branches:

| | Sequential chain | `Agent.parallel` |
|---|---|---|
| **Cost** | sum of all stage costs | sum of all branch costs (same) |
| **Latency** | sum of all stage latencies | max of all branch latencies |
| **Order guarantee** | strict — N+1 sees N's output | none — branches are isolated |
| **Composition** | next stage gets stage N's text | next stage gets the labelled join |

Parallel is the right choice when you'd otherwise wait on the same wall clock
*N* times. It's the wrong choice when stage N+1 *needs* stage N's output —
that's `chain`.

---

## Tracing — see all branches at once

`verbose=True` shows every branch's run interleaved:

```python
panel = Agent.parallel(researcher_a, researcher_b, researcher_c, verbose=True)
panel("What are the trade-offs of agent frameworks in 2026?")
```

Output (abbreviated):

```text
[parallel ▶ 3 branches]
  [branch claude_view  model=claude-haiku-4-5]   started
  [branch gpt_view     model=gpt-5.4-mini]       started
  [branch gemini_view  model=gemini-3-flash-preview] started

  [branch claude_view]   assistant: Agent frameworks fall into ...
  [branch claude_view]   done  cost=$0.00009
  [branch gpt_view]      assistant: The trade-offs centre on ...
  [branch gpt_view]      done  cost=$0.00012
  [branch gemini_view]   assistant: You can think of agent ...
  [branch gemini_view]   done  cost=$0.00008

[done] branches=3  total_cost=$0.00029  max_latency=812ms
```

Branch lines are interleaved (because they run concurrently) but each one is
prefixed with its name, so you can follow each branch's trace by filtering
on the prefix. The final summary shows aggregate cost and *max* latency
(the chain shows *sum* — that's the visible difference between the two).

---

## How other frameworks express parallel fan-out

??? example "CrewAI (hierarchical process)"

    CrewAI doesn't have a "parallel" process directly — its sequential and
    hierarchical modes are about delegation order, not concurrent execution.
    To run agents truly in parallel you'd reach for Python's `concurrent.futures`
    or `asyncio.gather`:

    ```python
    import asyncio
    from crewai import Agent, Task

    async def run_one(agent, task):
        return await asyncio.to_thread(agent.execute_task, task)

    async def fanout(task_text):
        tasks = [Task(description=task_text, agent=a) for a in (a1, a2, a3)]
        return await asyncio.gather(*(run_one(a, t) for a, t in zip([a1, a2, a3], tasks)))

    results = asyncio.run(fanout("How does each provider..."))
    ```

    The orchestration is your problem; you've left CrewAI's abstractions and
    you're back to plain Python concurrency.

??? example "LangGraph (parallel nodes via Send)"

    ```python
    from typing import Annotated, TypedDict
    from operator import add
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import Send
    from langchain_anthropic import ChatAnthropic

    claude_view = ChatAnthropic(model="claude-haiku-4-5")
    gpt_view    = ChatAnthropic(model="gpt-5.4-mini")
    gemini_view = ChatAnthropic(model="gemini-3-flash-preview")

    class State(TypedDict):
        query: str
        results: Annotated[list[str], add]   # reducer aggregates branch outputs

    def fanout(state: State):
        return [Send("branch", {"query": state["query"], "agent_id": i})
                for i in range(3)]

    def branch(state):
        models = [claude_view, gpt_view, gemini_view]
        reply = models[state["agent_id"]].invoke(state["query"])
        return {"results": [reply.content]}

    builder = StateGraph(State)
    builder.add_node("fanout", lambda s: s)
    builder.add_node("branch", branch)
    builder.add_conditional_edges("fanout", fanout, ["branch"])
    builder.add_edge("branch", END)
    builder.add_edge(START, "fanout")
    graph = builder.compile()

    out = graph.invoke({"query": "How does each provider...", "results": []})
    for r in out["results"]:
        print(r)
    ```

    Powerful — `Send` is one of LangGraph's strongest primitives. Cost: a
    typed State, a reducer to merge branch outputs, a conditional-edge
    function, three node definitions. For the simple "run these N agents
    on the same task" case, `Agent.parallel(a, b, c)` is one line.

---

## When NOT to use `Agent.parallel`

| Symptom | Use instead |
|---|---|
| Stage N+1 needs the output of stage N | `Agent.chain` — Step 7 |
| You want the LLM to pick which subset to run | `tools=[a, b, c]` — Step 4/5 |
| Each branch needs *different* input | `Plan` with named steps — Step 9 |
| Branches need to be conditional | `Plan` with `routes=` — Step 10 |

`Agent.parallel` is for the case where the *only* dimension of variation is
"which agent" — the input is the same for all of them. Anything more
structured belongs in `Plan`.

---

## Summary

| Concept | Syntax | What it does |
|---|---|---|
| Fan-out | `Agent.parallel(a, b, c)` | Runs all branches concurrently on the same task |
| Joined text | `env.text()` | Labelled join across branches (`[name]` prefixed) |
| Per-branch envelopes | `env.payload` | `list[Envelope]` in input order |
| Async typed access | `await panel.run_branches(task)` | Same list, returned directly |
| Concurrency cap | `concurrency_limit=N` | Never more than N branches at once |
| Per-branch timeout | `step_timeout=seconds` | Cancel any branch over the limit |
| Cost rollup | `env.metadata.cost_usd` | Sum across all branches |
| Latency rollup | `env.metadata.latency_ms` | Max across branches (not sum) |

You've got the three deterministic composition primitives now — chain,
parallel, and sub-agent-as-tool. The next step covers the most powerful
one: the explicit DAG via `Plan` + sentinels, for workflows that don't
fit the three so far.

---

[**Step 9: Explicit DAGs with `Plan` & sentinels →**](09-plan.md){ .md-button .md-button--primary }

[← Step 7: Sequential pipelines](07-chain.md){ .md-button }
