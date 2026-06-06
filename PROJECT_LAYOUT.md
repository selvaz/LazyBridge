# LazyBridge — Agent config file (design doc)

> **Status.** Proposal / design doc. Nothing ships yet. Captures the
> agreed format so we can build it next.
>
> **Idea in one line.** Keep every agent's **base LLM config, prompt, and
> output shape** in a *single* Markdown file, one section per agent,
> retrievable by name — and let `LLMEngine` auto-fill from it any
> parameter you don't pass explicitly.

This is an **opt-in convention** for projects with many agents. The core
stays code-first and zero-config; this is a thin layer on top.

---

## 1. `agents.md` — one file, everything per agent

Each agent starts at a **delimiter comment** `<!-- agent: <name> -->`
(the **name tag**). After it: a fenced config block (base LLM knobs +
output shape), then the prompt prose, running until the next delimiter.

````markdown
<!-- agent: researcher -->
```yaml
model: claude-opus-4-8
thinking: true
max_tokens: 4096
temperature: 0.2
output:
  summary: str
  citations: list[str]
```
You are a research expert. Find primary sources, cross-check claims,
and return a structured summary with citations.

## Output format    ← a heading inside the prompt; stays prompt content

<!-- agent: writer -->
```yaml
model: claude-sonnet-4-6
max_tokens: 2048
temperature: 0.7
```
You are a writer. Turn research notes into clear prose.
````

- **Name tag:** the `<!-- agent: <name> -->` comment is the agent name —
  same key as `Agent(name=...)`, so the LazyBridge Name Chain
  (`from_agent`, `Step`, routing) keeps working unchanged. Markdown never
  emits this comment by accident, so a prompt may contain **any** `##`
  headings or fenced examples without being mis-parsed.
- **Config block:** YAML, data only — the base config of each LLM
  (`model`, `thinking`, `max_tokens`, `temperature`, …). `thinking` may
  carry its own model when needed (e.g. `thinking: {model: ...}`).
- **`output`** (optional): the structured-output shape (see §2).
- **Prompt:** everything after the config block, up to the next
  delimiter, as prose.

### Retrieval

```python
cfg = load_agents("agents.md")["researcher"]
# -> {"model": "...", "thinking": True, "max_tokens": 4096,
#     "temperature": 0.2, "output": <pydantic model>,
#     "prompt": "You are a research expert. ..."}
```

Parser (~15 lines): split on `^<!-- agent:\s*(\S+)\s*-->`, the captured
group is the name, the first fenced block is the YAML config, the
remainder up to the next delimiter is the prompt. The delimiter is the
**only** boundary — chosen precisely because free-form Markdown prompts
cannot accidentally reproduce it (addresses the `^## `-ambiguity flagged
in review).

---

## 2. `output` — inline shape, with an escape hatch

Structured outputs are Pydantic models, but most agent outputs are flat,
so they can be declared **inline** as `field: type` pairs. The loader
builds a Pydantic model at load time via `pydantic.create_model`:

```yaml
output:
  summary: str
  citations: list[str]
```

When an output is complex (nested types, validators, reuse), point at a
real Python class instead — the **escape hatch**:

```yaml
output: schemas.Research      # dotted reference to a Pydantic model
```

So the common case (flat output) lives entirely in the single file; only
rich outputs reach into code. A `schemas.py` module is therefore
**optional** — just the destination for the complex cases.

**Trade-off of inline outputs.** Nested/complex types are clumsy in YAML,
the `.payload` is not statically typed (no mypy/autocomplete), and custom
validators can't live there. Reach for the escape hatch when any of those
bite.

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
  also attach the section's `output` model automatically?

---

## 4. Composition with Skills (SuperTool)

`agents.md` and a Skill/SuperTool (see `SUPERTOOL_PLAN.md`) **compose**
when both describe the same agent. They do **not** resolve every field
the same way — fields fall into two merge strategies.

### Two merge strategies

- **OVERRIDE** — single-value fields (`output`, `model`, `temperature`,
  `max_tokens`, `thinking`, `disclosure`). One value wins, chosen by the
  ladder below. Two output schemas can't be meaningfully merged, so we
  pick one.
- **COMPOSE** — the additive field (the prompt / `system`). **Nothing is
  lost:** the agent's prompt and the Skill body are concatenated. The
  agent keeps its identity *and* gains the Skill's expertise.

### Ladder for OVERRIDE fields (outer wins)

```
fleet policy  >  caller (explicit)  >  agents.md  >  Skill (suggested)  >  engine default
```

For owned single-value fields **the agent beats the Skill**: `agents.md`
is the operator's deliberate per-agent config, the Skill only *suggests*.
Caller and fleet still sit above both. Effective value = the first that
is **not `UNSET`**, scanning outer → inner.

### The prompt (COMPOSE field)

- A caller's explicit `system=` still overrides outright (top of ladder).
- Otherwise the prompt is **composed**: `agents.md` prompt **+** Skill
  body, both kept. Order (**confirmed**): agent prompt first (its
  role/identity), Skill body appended (its how-to expertise).
- A fleet-wide preamble, if any, composes too.

### Shared mechanism

- **One resolver**, parameterized by each field's strategy
  (`OVERRIDE | COMPOSE`) — *not* an identical pick-one for every field.
  This **refines** SuperTool invariant #4 ("no per-field branches"): there
  is still one resolver, but it now knows each field's merge strategy.
- **One `UNSET` sentinel**, never `None`, so `temperature=None` /
  `output=None` stay valid explicit values. Same sentinel as
  `LLMEngine.for_agent` (§3).
- **One output representation** — inline `field: type` (via
  `create_model`) or a dotted Pydantic reference — so OVERRIDE compares
  like with like.

> **Net effect.** `output` → agent wins; prompt → agent + Skill composed,
> nothing lost. All other scalar knobs follow `output` — agent wins
> (**confirmed**).

---

## 5. Resulting project shape

```
project/
├── agents.md       # base LLM config + prompt + output shape per agent
├── schemas.py      # OPTIONAL: Pydantic models for complex outputs only
├── tools.py        # shared tools
└── fleet.py        # thin: LLMEngine.for_agent(name) + tools -> Agent
```

That's the whole convention: one file describes the agents; the engine
knows how to read it; code is only needed for tools and complex outputs.

---

## 6. Next step

Build the `agents.md` parser (config + inline/escape-hatch output) and
`LLMEngine.for_agent(...)` (with the UNSET sentinel), plus one worked
example. Resolve A1–A3 first. The precedence resolver and `UNSET`
sentinel built here are the **same** ones the SuperTool/Skill work
(`SUPERTOOL_PLAN.md`) reuses — build them once, in one place.
