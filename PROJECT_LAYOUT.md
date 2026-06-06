# LazyBridge — Agent config file (design doc)

> **Status.** Proposal / design doc. Nothing ships yet. Captures the
> agreed format so we can build it next.
>
> **Idea in one line.** Keep every agent's **base LLM config + system
> prompt** in a single Markdown file, one section per agent, retrievable
> by name — and let `LLMEngine` auto-fill from it any parameter you don't
> pass explicitly.

This is an **opt-in convention** for projects with many agents. The core
stays code-first and zero-config; this is a thin layer on top.

---

## 1. `agents.md` — one file, one section per agent

Each agent is an `## <name>` section (the heading is the **name tag**).
Inside: a fenced config block (base LLM knobs) followed by the prompt
prose.

````markdown
## researcher
```yaml
model: claude-opus-4-8
thinking: true
max_tokens: 4096
temperature: 0.2
```
You are a research expert. Find primary sources, cross-check claims,
and return a structured summary with citations.

## writer
```yaml
model: claude-sonnet-4-6
max_tokens: 2048
temperature: 0.7
```
You are a writer. Turn research notes into clear prose.
````

- **Name tag:** the `## ` heading text is the agent name — same key as
  `Agent(name=...)`, so the LazyBridge Name Chain (`from_agent`, `Step`,
  routing) keeps working unchanged.
- **Config block:** YAML, data only — the base config of each LLM
  (`model`, `thinking`, `max_tokens`, `temperature`, …). `thinking` may
  carry its own model when needed (e.g. `thinking: {model: ...}`).
- **Prompt:** everything after the config block, as prose.

### Retrieval

```python
cfg = load_agents("agents.md")["researcher"]
# -> {"model": "...", "thinking": True, "max_tokens": 4096,
#     "temperature": 0.2, "prompt": "You are a research expert. ..."}
```

Parser is ~15 lines: split on `^## `, first line is the name, the first
fenced block is the YAML config, the remainder is `prompt`.

---

## 2. `schemas.py` — structured outputs

Structured outputs are Pydantic models, so they stay typed Python (kept
in code for autocomplete + mypy), with a name → model map mirroring the
agent names:

```python
from pydantic import BaseModel

class Research(BaseModel):
    summary: str
    citations: list[str]

OUTPUTS = {"researcher": Research}   # agent name -> output model
```

---

## 3. `LLMEngine` auto-fill (next phase)

Add a constructor that, given an agent name, fills from `agents.md` any
parameter not passed explicitly.

```python
# everything from the file:
engine = LLMEngine.for_agent("researcher")

# explicit override; the rest from the file:
engine = LLMEngine.for_agent("researcher", temperature=0.0)
```

**Precedence:** explicit argument > value in file > engine default.

**Note — the UNSET problem.** To honour that precedence we must
distinguish "not passed" from "passed as `None`" (since `None` can be a
valid explicit value, e.g. `temperature=None` → provider default). This
needs a module-level sentinel as the default for each fillable parameter,
not `None`.

Open sub-questions:

- **A1. File location.** How does `for_agent` find `agents.md`? Explicit
  path arg, a configured default, or convention (project root)?
- **A2. Prompt.** Should `for_agent` also pull the prompt into `system=`
  by default (likely yes), with an opt-out?
- **A3. Output wiring.** Does `for_agent` (or a sibling `Agent.for_agent`)
  also attach `OUTPUTS[name]` automatically, or stays manual?

---

## 4. Resulting project shape

```
project/
├── agents.md       # base LLM config + prompt per agent (## name sections)
├── schemas.py      # Pydantic output models + name -> model map
├── tools.py        # shared tools
└── fleet.py        # thin: LLMEngine.for_agent(name) + tools + OUTPUTS -> Agent
```

That's the whole convention: two files describe the agents (one data,
one types), and the engine knows how to read the data one.

---

## 5. Next step

Build the `agents.md` parser + `LLMEngine.for_agent(...)` (with the UNSET
sentinel) and one worked example. Resolve A1–A3 first.
