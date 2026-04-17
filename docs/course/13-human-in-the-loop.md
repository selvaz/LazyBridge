# Module 13: Human-in-the-Loop

Humans participate in agent pipelines as first-class agents — not as callbacks or hooks, but as composable building blocks that slot into chains, tools, parallel panels, and verify judges.

Two classes:

- **HumanAgent** — simple, covers 90% of cases
- **SupervisorAgent** — human with superpowers (tools, retry, store access)

---

## HumanAgent — the simple one

### Basic usage

```python
from lazybridge import HumanAgent

# CLI — uses input() by default
human = HumanAgent(name="reviewer")
resp = human.chat("What do you think about this report?")
print(resp.content)  # whatever the human typed
```

### Single mode (default)

The pipeline pauses, human responds once, pipeline continues:

```python
from lazybridge import LazyAgent, LazyTool, HumanAgent

researcher = LazyAgent("anthropic", name="researcher", tools=[search])
human = HumanAgent(name="reviewer")
writer = LazyAgent("openai", name="writer")

pipeline = LazyTool.chain(
    researcher, human, writer,
    name="reviewed_pipeline",
    description="Research, get human review, then write",
)
result = pipeline.run({"task": "AI safety report"})
```

What happens:
1. Researcher runs, produces output
2. Pipeline pauses — human sees the output, types a response
3. Writer receives the human's response as context, writes final output

### Dialogue mode

Human can go back and forth until typing "done":

```python
human = HumanAgent(name="reviewer", mode="dialogue", end_token="done")
```

```
[reviewer] Dialogue mode (type 'done' to finish):
──────────────────────────────────────────────────
The researcher found 3 papers on AI safety...
──────────────────────────────────────────────────
[reviewer] > The summary looks good but missing recent papers
[reviewer] > Also add a section about alignment
[reviewer] > done

→ Both lines become the output passed to the next agent
```

### Custom callbacks

Replace `input()` with any function — web UI, Slack bot, API endpoint:

```python
# Web UI callback
def web_input(prompt: str) -> str:
    return my_web_api.wait_for_response(prompt)

human = HumanAgent(name="reviewer", input_fn=web_input)

# Async callback (websocket, queue)
async def ws_input(prompt: str) -> str:
    return await websocket.receive_text()

human = HumanAgent(name="reviewer", ainput_fn=ws_input)
```

### Timeout and defaults

```python
# Wait 5 minutes, then use default
human = HumanAgent(
    name="approver",
    timeout=300,
    default="approved",  # auto-approve after 5 min
)

# Timeout without default → raises TimeoutError
human = HumanAgent(name="approver", timeout=60)
# After 60 seconds: TimeoutError: Human input timed out after 60s
```

### Prompt template

Control how the task is presented to the human:

```python
human = HumanAgent(
    name="reviewer",
    prompt_template="Please review the following and provide feedback:\n\n{task}",
)
```

### As a tool

```python
review_tool = human.as_tool("review", "Human reviews the output")
orchestrator = LazyAgent("anthropic")
orchestrator.loop("Write and review a report", tools=[research_tool, review_tool])
```

### As a verify judge

```python
human = HumanAgent(name="judge", input_fn=lambda p: "approved" if ok else "retry: fix X")
agent.loop("Write code", verify=human, max_verify=3)
```

### With context injection

```python
from lazybridge import LazyContext

human.chat("Provide your input")
writer.chat("Write based on human input", context=LazyContext.from_agent(human))
```

---

## SupervisorAgent — human with superpowers

The SupervisorAgent gives a human full interactive control inside a pipeline. They can:

- **Call tools** — search, calculate, fetch data
- **Retry agents** — re-run previous agents with feedback
- **Read the store** — inspect shared state
- **Continue or override** — pass output forward or replace it

### Setup

```python
from lazybridge import LazyAgent, LazyTool, LazySession, SupervisorAgent

sess = LazySession(db="project.db")

researcher = LazyAgent("anthropic", name="researcher", tools=[search], session=sess)
analyst = LazyAgent("openai", name="analyst", tools=[calculator], session=sess)

supervisor = SupervisorAgent(
    name="supervisor",
    tools=[search_tool, calculator],      # tools the human can call
    agents=[researcher, analyst],         # agents the human can retry
    session=sess,                         # for store/events access
)

writer = LazyAgent("anthropic", name="writer", session=sess)

pipeline = LazyTool.chain(
    researcher, analyst, supervisor, writer,
    name="supervised_pipeline",
    description="Research, analyze, supervise, write",
)
result = pipeline.run({"task": "Q1 2026 AI trends report"})
```

### The interactive REPL

When the supervisor's turn comes, they get a full command interface:

```
══════════════════════════════════════════════════════════════
[supervisor] Pipeline step — your turn
──────────────────────────────────────────────────────────────
Previous output:
  Analysis shows moderate growth in AI adoption...

Available tools: search, calculator
Retryable agents: researcher, analyst
Store keys: findings, raw_data

Commands: continue | retry <agent>: <feedback> | store <key> | <tool>(<args>)
──────────────────────────────────────────────────────────────
```

### Commands

**Call a tool:**
```
[supervisor] > search("AI regulation EU 2026")
Result: "EU AI Act Phase 2 enforcement began March 2026..."

[supervisor] > calculator("15.3 * 1.12")
Result: "17.136"
```

**Retry an agent with feedback:**
```
[supervisor] > retry researcher: include EU regulation data from 2026
[researcher re-running with feedback...]
New output: "Found 5 papers including EU AI Act analysis..."
```

**Read the session store:**
```
[supervisor] > store findings
Store[findings]: "Key finding: AI adoption grew 23% in Q1..."
```

**Continue (pass output forward):**
```
[supervisor] > continue
# passes the current output to the next agent (writer)

[supervisor] > continue: Here's a summary combining everything...
# passes a custom message instead
```

### Retry deep dive

When you retry an agent, the supervisor:
1. Finds the agent by name
2. Sends the original task + your feedback
3. If the agent has tools, uses `loop()` (agent runs its tools again)
4. If no tools, uses `chat()`
5. Shows you the new output
6. You can retry again or continue

```
[supervisor] > retry analyst: break down by region, not just global
[analyst re-running...]
New output: "Regional breakdown: US: 28% growth, EU: 19%, Asia: 31%..."

[supervisor] > retry analyst: add confidence intervals
[analyst re-running...]
New output: "Regional breakdown with CI: US: 28% (±3%), EU: 19% (±4%)..."

[supervisor] > continue
```

### Custom callbacks (web UI)

```python
supervisor = SupervisorAgent(
    name="supervisor",
    tools=[search_tool],
    agents=[researcher],
    input_fn=my_web_repl_callback,  # custom I/O
)
```

---

## Pipeline safety

Both HumanAgent and SupervisorAgent handle these runtime concerns:

### Async event loop

`input()` blocks the event loop. Both classes use `asyncio.to_thread()` for async paths:

```python
# This works — never blocks the event loop
await pipeline_tool.arun({"task": "..."})  # human input runs in a thread
```

### step_timeout

Chains with `step_timeout=30` would kill human input. Both classes set `_is_human=True`, and the chain builder skips the timeout for human participants:

```python
# Human gets unlimited time even with step_timeout
pipeline = LazyTool.chain(
    researcher, human, writer,
    step_timeout=30.0,  # applies to researcher and writer, NOT to human
)
```

### KeyboardInterrupt

If the human presses CTRL+C:
- With `default=` set → returns the default
- Without `default=` → raises KeyboardInterrupt (clean exit)

### Parallel I/O

In `LazyTool.parallel()`, terminal I/O is serialized with a lock so prompts don't overlap.

---

## Decision matrix

| I need... | Use |
|-----------|-----|
| Simple approval gate | `HumanAgent(mode="single")` |
| Multi-turn review | `HumanAgent(mode="dialogue")` |
| Auto-approve after timeout | `HumanAgent(timeout=300, default="approved")` |
| Human as quality judge | `agent.loop(verify=human)` |
| Human can investigate with tools | `SupervisorAgent(tools=[...])` |
| Human can retry previous steps | `SupervisorAgent(agents=[...])` |
| Human can inspect pipeline state | `SupervisorAgent(session=sess)` |
| Web UI instead of CLI | `HumanAgent(input_fn=my_callback)` |
| Async (websocket/queue) | `HumanAgent(ainput_fn=my_async_callback)` |

---

## Exercise

1. Create a chain: researcher → HumanAgent → writer. Mock the human with `input_fn` returning a fixed string.
2. Create a SupervisorAgent with a search tool. Simulate a session where the human searches, then continues.
3. Use HumanAgent as a `verify=` judge in a loop and test the approval/rejection flow.
