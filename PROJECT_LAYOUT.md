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

Each agent is an `## <name>` section (the heading is the **name tag**).
Inside: a fenced config block (base LLM knobs + output shape) followed by
the prompt prose.

````markdown
## researcher
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
- **`output`** (optional): the structured-output shape (see §2).
- **Prompt:** everything after the config block, as prose.

### Retrieval

```python
cfg = load_agents("agents.md")["researcher"]
# -> {"model": "...", "thinking": True, "max_tokens": 4096,
#     "temperature": 0.2, "output": <pydantic model>,
#     "prompt": "You are a research expert. ..."}
```

Parser is ~15 lines: split on `^## `, first line is the name, the first
fenced block is the YAML config, the remainder is `prompt`.

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

`agents.md` and a Skill/SuperTool (see `SUPERTOOL_PLAN.md`) are **not
rivals** — they attach at different levels and **compose** when both are
present. `agents.md` configures the *Agent* (its base model, prompt,
output); a Skill configures a *Tool* the agent calls (its disclosure,
suggested output, coupled safeguard). They share one mechanism so they
never drift apart.

### One authority ladder (outer wins)

Every fillable field — `model`, prompt/`system`, `output`, engine knobs,
`disclosure` — resolves through the **same** ladder, most-outer first:

```
fleet policy  >  caller (explicit)  >  Skill (suggested)  >  agents.md (default)  >  engine default
```

The effective value is the first one that is **not `UNSET`**, scanning
outer → inner.

### One resolver, one sentinel, one output shape

- **One resolver.** A single precedence function implements the ladder
  for every field — no per-field branches. (This is SuperTool
  invariant #4, with the fix below.)
- **One sentinel.** "Not provided" is a single module-level `UNSET`,
  never `None`, so `temperature=None` / `output=None` stay valid explicit
  values. This is the *same* `UNSET` that `LLMEngine.for_agent` needs
  (§3) and that SuperTool invariant #4 requires instead of its
  `is not None` check.
- **One output representation.** A Skill's suggested output and an
  `agents.md` `output` are the *same kind of value* — inline `field: type`
  (via `create_model`) or a dotted reference to a Pydantic class — so the
  resolver compares them directly.

### When both are present

- **Agent identity** (model, prompt, output) comes from `agents.md`
  unless a caller or fleet policy overrides it.
- A **SuperTool** the agent calls carries its own Skill-supplied defaults,
  overridable by the caller via the same ladder.
- The two contend for the same field only under `subagent` disclosure,
  where the Skill becomes a subagent's `system`. If that subagent is
  itself an `agents.md` agent, the ladder decides: Skill (suggested) sits
  above `agents.md` (default), so the Skill priming wins over the base
  prompt unless the caller says otherwise.

> **Debatable rung.** Skill-above-`agents.md` is the one ordering worth a
> second look: a Skill is a more specific, opt-in capability, so we treat
> it as "more outer" than the base per-agent default. If operators expect
> `agents.md` to be the final word per agent, flip these two rungs — but
> keep the ladder single and uniform either way.

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
