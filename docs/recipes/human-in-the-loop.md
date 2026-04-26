# Recipe: Human-in-the-loop

**Tier:** Mid / Full  
**Goal:** Pause the pipeline and ask a human to approve, redirect, or inspect before continuing.

Two levels of involvement:

| | HumanEngine | SupervisorEngine |
|---|---|---|
| Use when | Simple approve/reject gate | Full REPL — retry agents, call tools, inspect state |
| Interaction | Single prompt | Interactive session |
| Unattended? | Yes — set `default=` | Yes — set `timeout=` + `default=` |

## Approval gate (HumanEngine)

The lightest option. The human sees the previous agent's output and types to approve or redirect.

```python
from lazybridge import Agent
from lazybridge.ext.hil import HumanEngine

def search(query: str) -> str:
    """Search the web for ``query``."""
    return "..."

researcher = Agent("claude-opus-4-7", name="researcher", tools=[search])

# HumanEngine pauses here; whatever the human types becomes the next agent's task.
gatekeeper = Agent(engine=HumanEngine(), name="gatekeeper")

writer = Agent("claude-opus-4-7", name="writer")

agents = [researcher, gatekeeper, writer]
pipeline = Agent.chain(*agents)
pipeline("draft a policy brief on AI regulation")
```

To run unattended (e.g. in tests), set `default=`:

```python
gatekeeper = Agent(engine=HumanEngine(default="continue"), name="gatekeeper")
```

## Full REPL (SupervisorEngine)

The human gets an interactive session and can retry named agents with feedback, call
tools directly, and inspect stored state before deciding to continue.

```python
from lazybridge import Agent
from lazybridge.ext.hil import SupervisorEngine

researcher = Agent("claude-opus-4-7", name="researcher", tools=[search])

supervisor = Agent(
    engine=SupervisorEngine(
        agents=[researcher],   # enables "retry researcher: <feedback>" REPL command
        tools=[search],        # human can also call search directly in the REPL
        timeout=120,           # seconds before returning `default`
        default="continue",    # used when timeout fires (for unattended runs)
    ),
    name="supervisor",
)

writer = Agent("claude-opus-4-7", name="writer")

agents = [researcher, supervisor, writer]
pipeline = Agent.chain(*agents)
pipeline("draft a policy brief on AI regulation")
```

REPL commands available to the human:

| Command | What it does |
|---|---|
| `continue [text]` | Accept; pass optional text as context to the next agent |
| `retry <agent>: <feedback>` | Re-run the named agent with feedback appended to its task |
| `store <key>` | Print the value at `store[key]` |
| `<tool>(<args>)` | Call a registered tool directly |

## Next

- [HumanEngine guide](../guides/human-engine.md) — full signature and options
- [SupervisorEngine guide](../guides/supervisor.md) — full REPL reference
- [Decision: HumanEngine vs SupervisorEngine](../decisions/human-engine-vs-supervisor.md)
- [Plan with resume](plan-with-resume.md) — add checkpoint/resume to a supervised pipeline
