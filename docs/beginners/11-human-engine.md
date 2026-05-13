# Step 11: Human in the loop with `HumanEngine`

All ten steps so far have one thing in common: an **LLM is the engine**.
Even `verify=` (Step 6) is "an LLM checks another LLM". For some tasks
that's not appropriate:

- **Compliance / legal sign-off** — the law says a human has to approve
- **Irreversible actions** — sending money, deploying production, deleting data
- **High-stakes content** — public statements, contracts, healthcare info
- **Disagreement-mode** — when LLMs converge on a wrong answer and only a
  human will spot it

For these cases LazyBridge swaps in a different engine: **`HumanEngine`**.
It pauses the agent at the engine boundary, presents the task to a human
(terminal prompt, web form, or your own UI), and returns their response —
in the same `Envelope` shape every other agent returns.

The whole point: the human is *just another engine*. Everything you've
learned (chain, parallel, Plan, routing, sub-agent-as-tool, verify=) still
works around it.

---

## The simplest human gate

Two equivalent ways to build a human-input agent:

```python
# 1. Direct: pass HumanEngine to the Agent constructor
from lazybridge import Agent
from lazybridge.ext.hil import HumanEngine

approver = Agent(engine=HumanEngine(), name="approver")

# 2. Factory: same thing, slightly tighter
from lazybridge.ext.hil import human_agent

approver = human_agent(name="approver")
```

Calling it pauses for input:

```python
env = approver("Approve deploy to production?")
print(env.text())   # whatever the operator typed
```

In the terminal you'll see:

```text
═══ Human input required ═══
Task: Approve deploy to production?
> _
```

The operator types a response, hits enter, and the `Envelope` returns
carrying the text. From the calling code's perspective, this is
indistinguishable from an LLM agent — same `.text()`, same `.payload`,
same `.metadata`.

---

## Composing the human into a Plan

A typical production shape: an LLM drafts a deploy command, a **human
approves**, then a downstream agent (or plain function) executes:

```python
from lazybridge import Agent, LLMEngine, Plan, Step, from_step, from_start
from lazybridge.ext.hil import human_agent


drafter = Agent(
    engine=LLMEngine(
        "claude-opus-4-7",
        system="You translate a deploy request into a one-line shell command. "
               "Be conservative; flag anything risky.",
    ),
    name="drafter",
)

approver = human_agent(
    name="approver",
    # The user sees: "Approve this command? <draft>"
    # They type "yes", "no", or a corrected command.
)

executor = Agent(
    engine=LLMEngine("claude-haiku-4-5",
                     system="Execute the approved shell command via the run_shell tool."),
    tools=[run_shell],
    name="executor",
)


pipeline = Agent(
    engine=Plan(
        Step("drafter"),                                # LLM proposes
        Step("approver",
             task=from_step("drafter")),                # human reviews the draft
        Step("executor",
             task=from_step("approver")),               # only the human's approved text runs
    ),
    tools=[drafter, approver, executor],
    name="deploy_pipeline",
)

pipeline("Restart the production payments service.")
```

What this gets you:

- **Audit trail** — every step is in the envelope's metadata; you can
  prove the human's exact input was what got executed
- **No new framework** — the human is wired with the same `Step` and
  sentinels you saw in Step 9
- **Safe defaults** — if `approver` returns text that doesn't approve,
  add a `routes=` (Step 10) that branches to an "abort" agent

---

## `timeout=` and `default=` — production-friendly fallbacks

A human gate that blocks forever on an offline operator isn't a gate, it's
a deadlock. `HumanEngine` takes two safety knobs:

```python
approver = human_agent(
    timeout=300.0,          # 5 minutes
    default="reject",       # what to return if no one responds in time
)
```

- **`timeout=seconds`** — the maximum wall-clock the engine waits for
  input. After that, it either uses `default` or raises `TimeoutError`.
- **`default="..."`** — the response to return on timeout. Set this to
  the *safe* answer (usually `"reject"`, `"no"`, `"abort"` — *not*
  `"approve"`).

If you don't set `default`, a timeout raises `TimeoutError` — the agent's
envelope arrives with `env.ok == False`, and you can branch on that with
routing (Step 10).

---

## Structured input from a human — `output=PydanticModel`

The same `output=` parameter you used in Step 3 works on `HumanEngine`.
Terminal mode prompts each field; web mode renders an HTML form:

```python
from pydantic import BaseModel, Field


class Approval(BaseModel):
    decision: Literal["approve", "reject", "modify"]
    notes: str = Field(default="", description="Optional reason / amended text")


approver = human_agent(
    output=Approval,
    name="approver",
    timeout=300.0,
    default='{"decision":"reject","notes":"timeout"}',
)

env = approver("Approve deploy to production?")
approval: Approval = env.payload
print(approval.decision, approval.notes)
```

Terminal session:

```text
═══ Human input required ═══
Task: Approve deploy to production?
decision (approve|reject|modify)> approve
notes (optional)> rollback plan attached, off-hours
```

The Pydantic model becomes a structured form. The result is type-safe
downstream — exactly like a Pydantic-typed LLM agent. With `routes_by=`
(Step 10) this gives you typed human-driven branching out of the box:

```python
Step("approver",
     output=Approval,
     routes_by="decision",
     after_branches="archive"),
Step("approve"),
Step("reject"),
Step("modify"),
Step("archive"),
```

The human's typed answer routes the Plan — no glue code.

---

## Web mode — `ui="web"`

Terminal input doesn't fit every deployment. Set `ui="web"` and `HumanEngine`
spins up a local web form for the operator:

```python
approver = human_agent(
    ui="web",
    timeout=600.0,
    default="reject",
    name="approver",
)
```

Same `output=PydanticModel` support; same envelope contract. Useful when
the agent runs on a server and a remote operator approves via browser.

You can also plug your own UI (Slack bot, ticket system, mobile app)
through the engine's protocol — `ui=YourCustomUI()` accepts any object
that implements `prompt(task, tools, output_type)`. See the
[HumanEngine guide](../guides/mid/human-engine.md) for the full protocol.

---

## When `HumanEngine` vs `verify=` (Step 6)

Both involve "approval", but they're different in shape:

| | `verify=` | `HumanEngine` |
|---|---|---|
| Who approves | An **LLM** (or callable) judge | A **human** |
| Where it runs | After the producer agent, gating its output | In place of an agent — the human *is* the engine |
| What it does on rejection | Feeds back as feedback, retries up to `max_verify` | Returns the operator's response (which can itself be a no) |
| Use when | You want automated quality gating | You need a human's actual decision |

You can combine them. Common pattern: an LLM-driven `verify=` catches the
obvious problems cheaply; if the verifier approves, a `HumanEngine` step
asks a person for final sign-off. Two levels of gates, each catching what
the other can't.

---

## The bigger sibling — `SupervisorEngine`

`HumanEngine` is a one-shot prompt: present, wait, return. For long-running
ops/dev workflows where a human wants to **interact**, LazyBridge has
`SupervisorEngine`: a REPL that exposes the agent's tools to the operator
and lets them call sub-agents, retry steps, inspect the store, and steer
the run.

```python
from lazybridge.ext.hil import supervisor_agent

ops = supervisor_agent(
    tools=[deploy, rollback, check_health],
    agents=[researcher, executor],
    name="ops_supervisor",
)
ops("Promote release v2.4 across staging then production.")
```

This is more than a beginner step — it's the
[SupervisorEngine guide](../guides/full/supervisor.md) territory.

---

## How other frameworks handle human-in-the-loop

??? example "LangGraph (`interrupt`)"

    LangGraph's HIL primitive is `interrupt()` — call it from inside a node
    and the graph pauses, persisting state via a checkpointer. The caller
    resumes the graph with the human's response:

    ```python
    from langgraph.types import interrupt, Command
    from langgraph.checkpoint.memory import MemorySaver

    def approval_node(state):
        # Pause and wait for human input
        decision = interrupt("Approve deploy?")
        return {"approved": decision}

    builder = StateGraph(State)
    builder.add_node("draft", draft_node)
    builder.add_node("approval", approval_node)
    builder.add_node("execute", execute_node)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "approval")
    builder.add_edge("approval", "execute")
    builder.add_edge("execute", END)
    graph = builder.compile(checkpointer=MemorySaver())

    # First run pauses at the interrupt
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"task": "Deploy v2.4"}, config=config)

    # ... operator decides offline ...

    # Resume with the human's answer
    graph.invoke(Command(resume="approve"), config=config)
    ```

    Works well; requires the checkpointer machinery, thread IDs, and two
    `invoke` calls. LazyBridge wraps the same idea (it has checkpoints too)
    but for the simple "ask the human, get the answer" case the
    `HumanEngine` engine swap is a single line and works without thread
    management.

??? example "CrewAI (`human_input=True`)"

    CrewAI puts `human_input` on the Task:

    ```python
    from crewai import Task

    approval_task = Task(
        description="Review the deploy plan and approve or reject.",
        agent=reviewer,
        expected_output="approve | reject",
        human_input=True,        # ← the human reviews after the agent runs
    )
    ```

    After the agent produces its draft, CrewAI prompts the operator at the
    terminal to confirm or edit. Simple for terminal-only flows. No web
    UI, no `timeout`/`default` story for production deployment, no
    structured Pydantic input — it returns the operator's free-form text.

---

## When NOT to use `HumanEngine`

Some smells that suggest the human is in the wrong place:

| Symptom | Reach for |
|---|---|
| Approval just checks an LLM's output format / tone | `verify=` (Step 6) — cheaper and faster |
| Most of the workload is human; LLM is the side-show | A plain Python form or web app — agents are overkill |
| The human's response is purely yes/no and rarely no | Skip the human; log + monitor instead |
| The human needs to interactively use tools / retry | `SupervisorEngine` (see Guides) |

Use `HumanEngine` when there's a **specific decision point** where you
genuinely need a human, and the rest of the workflow stays automatic.

---

## Summary

| Concept | Syntax | What it does |
|---|---|---|
| Direct engine | `Agent(engine=HumanEngine())` | Drop-in replacement for `LLMEngine` |
| Factory sugar | `human_agent(timeout=..., default=...)` | Same shape, tighter call site |
| Timeout fallback | `timeout=seconds, default="..."` | Avoid deadlocks; pick a *safe* default |
| Typed input | `output=PydanticModel` | Terminal prompts each field; web renders a form |
| Web UI | `ui="web"` | Local web form instead of stdin |
| Custom UI | `ui=YourUIObject` | Plug Slack, mobile, etc. via the protocol |
| In a Plan | `Step("approver", task=from_step("drafter"))` | Sentinels work unchanged |
| Routed by human | `Step("approver", output=Model, routes_by="decision", after_branches="...")` | Typed human → routing |
| REPL sibling | `SupervisorEngine` / `supervisor_agent()` | Interactive ops; see Guides |
| Distinct from | `verify=` — LLM judge with retry | `HumanEngine` — actual person, no automatic retry |

---

You've now seen every basic LazyBridge primitive a developer encounters:
LLMs, tools, agents, envelopes, sub-agents, verify, chain, parallel, Plan,
routing, and now HumanEngine. The last two steps zoom out: comparing
LazyBridge to LangGraph and CrewAI head-on, then pointing you to the
guides that go deeper.

---

[**Step 12: LazyBridge vs LangGraph vs CrewAI →**](12-vs-frameworks.md){ .md-button .md-button--primary }

[← Step 10: Routing](10-routing.md){ .md-button }
