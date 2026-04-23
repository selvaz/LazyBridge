## signature
Agent.chain(*agents: Agent, name: str = "chain", **kwargs) -> Agent

Under the hood: Plan(*[Step(target=a, name=a.name) for a in agents]).
Sequential. Each agent's output becomes the next agent's task
(``from_prev`` default). Result: the last agent's Envelope.

Alternatives:
  Plan(Step(a), Step(b))                    # same thing, more explicit
  Plan(Step(a, name="a"),
       Step(b, name="b", task=from_start))  # b gets the ORIGINAL task, not a's output

## rules
- ``Agent.chain`` is sugar for a linear ``Plan``. Use it when you have
  a straight line and no need for typed hand-offs or routing.
- Memory / session / guards on the chain wrapper apply at the outer
  boundary; individual agents keep their own.
- For fan-out on the same task see ``Agent.parallel``. For typed /
  conditional flows see ``Plan``.

## narrative
**Use `Agent.chain` for** linear pipelines where each agent's text output feeds the next.
It's the right default for sequential multi-agent flows with no branching.

**Upgrade to `Plan`** when steps need Pydantic typed models to flow between them, you need
conditional routing (`next: Literal[...]`), or you want crash-resume with `resume=True`.

## example
```python
from lazybridge import Agent, Memory

researcher = Agent("claude-opus-4-7", name="researcher", tools=[search])
editor     = Agent("claude-opus-4-7", name="editor")
writer     = Agent("claude-opus-4-7", name="writer")

# Each agent's output becomes the next agent's task (from_prev default).
# Memory("auto") keeps the running transcript in the chain's context window.
agents = [researcher, editor, writer]
pipeline = Agent.chain(*agents, memory=Memory("auto"))   # construction
print(pipeline("AI trends April 2026").text())            # invocation → Envelope → text
```

## pitfalls
- ``Agent.chain`` does not preserve typed outputs between steps — the
  next step sees the previous step's ``Envelope.text()``. If step N
  produces a Pydantic model and step N+1 needs it as a model, use
  ``Plan`` instead so you can declare ``output=``.
- The outer chain Agent has its own name ("chain" by default); set
  ``name="…"`` if you want it to appear distinctly in ``Session.graph``.

