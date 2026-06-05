# LazyBridge — Large-project layout (design doc)

> **Status.** Proposal / design doc. Nothing here ships yet. This file
> captures the rationale, the recommended on-disk convention, the
> alternatives considered, and the open questions — so we can agree on
> the shape *before* writing any loader/scaffolder code.
>
> **Scope.** How a **large application built on LazyBridge** (and
> LazyTools) should organise its agents, prompts, and configuration on
> disk. This is a *convention for downstream projects*, not a change to
> the LazyBridge core API. The core stays code-first and zero-config;
> this layout is a thin, optional pattern layered on top.

---

## 1. Motivation

LazyBridge is deliberately **code-first and zero-config**. The mental
model is `Agent = Engine + Tools + State`, and everything — system
prompt, model choice, engine knobs, tool wiring — is expressed as plain
Python:

```python
researcher = Agent(
    name="researcher",
    engine=LLMEngine("claude-opus-4-8", system="You are a research expert."),
    tools=[search],
)
```

This is the right default. It gives IDE autocomplete, static type
checking, and frictionless composition, and it is unbeatable for small
projects and examples.

It starts to hurt at scale. In a project with dozens of agents, each
agent drags along:

- a **system prompt** that is often long, multi-paragraph, and edited
  frequently — sometimes by people who are not Python engineers;
- a **model / tier** choice that should differ per environment
  (cheap model in dev, top tier in prod);
- **engine knobs** (`temperature`, `max_turns`, `thinking`, `cache`,
  `timeout`, retries);
- **tools**, **guards**, **verify**, and an **output schema**.

When all of this lives inline in `.py` files, two problems appear:

1. **Mixed lifecycles.** Prompt text and model choice (declarative data,
   reviewed as prose, A/B-tested, tweaked by PM/prompt engineers) are
   tangled with tool wiring and composition (imperative code that needs
   Python and type-checking). They change for different reasons, by
   different people, on different cadences — but they live in the same
   file and the same diff.
2. **No discovery story.** There is no convention for "where do the
   agents live", "what is this fleet made of", or "how do I find the
   prompt for `researcher`". Every team invents its own ad-hoc layout.

This doc proposes a **standard directory architecture** that separates
those two lifecycles **without fighting the code-first core**.

---

## 2. Design principles

1. **Don't fight code-first.** Assembly stays in Python. We are not
   introducing an "agent from YAML" runtime. The layout is a convention
   plus a thin loader; the `Agent(...)` call is still real, typed Python.
2. **Separate declarative from imperative.** Prompt prose and pure-data
   knobs go in non-`.py` files (`prompt.md`, `config.toml`) so they get
   clean diffs, prose review, and non-dev editability. Tools, guards,
   schemas, and composition stay in `.py`.
3. **Reuse the Name Chain.** LazyBridge already makes `Agent(name=...)`
   the authoritative key that wires `Step`, `from_step`, `from_agent`,
   and routing. We make **directory name == agent name == registry key**
   so there is zero extra mapping to maintain.
4. **One agent, one folder.** Everything an agent owns is co-located.
   You can read, review, or delete an agent as a unit.
5. **Composition is separate from definition.** Agents don't know which
   pipeline they run in. `Plan`/`Step` graphs live in their own place.
6. **Environments are overlays, not forks.** One source of truth per
   agent, with thin per-environment overrides merged on top — the same
   "flat dict of fleet defaults" pattern LazyBridge already recommends,
   given an ordered home on disk.
7. **Thin config (decided).** `config.toml` holds **data only** —
   model/tier and engine knobs. It never describes tools, guards, or
   control flow. Those are Python, where they keep their type-safety.
   (See §7 for the alternative we rejected.)

---

## 3. Recommended layout

```
project/
├── agents/
│   ├── _loader.py             # reads prompt.md + config.toml + env overlay
│   ├── registry.py            # discovers folders, builds agents by name
│   ├── researcher/
│   │   ├── agent.py           # ASSEMBLY: prompt + config + tools + schema → Agent
│   │   ├── prompt.md          # system prompt (prose; versionable; PR-reviewable)
│   │   ├── config.toml        # model/tier + engine knobs (data only)
│   │   ├── tools.py           # agent-specific tools (or re-exports from /tools)
│   │   └── schema.py          # Pydantic output model (optional)
│   └── writer/
│       └── …                  # same shape
│
├── prompts/                   # SHARED prompt fragments (reused across agents)
│   ├── _house_style.md        # common tone / rules
│   └── _safety.md
│
├── tools/                     # shared tool library (wrapped via LazyTools)
│   ├── search.py
│   └── email.py
│
├── pipelines/                 # composition: Plan/Step graphs over the agents
│   └── ticket_flow.py
│
├── config/
│   ├── defaults.toml          # fleet-wide defaults (timeout, retries, cache…)
│   ├── dev.toml               # environment overlay
│   └── prod.toml
│
└── evals/                     # evals keyed by agent / prompt name
    └── researcher/
```

### Why this shape

- **`agents/<name>/` (folder per agent).** Co-location: the prompt, the
  knobs, the tools, and the schema for one agent sit together and move
  together. The folder name is the agent's identity.
- **`prompt.md` split out.** A prompt is prose. As Markdown it gets
  readable PR diffs, can be edited without touching code, supports
  side-by-side versions (`prompt.md` vs `prompt.v2.md`) for A/B and
  eval, and can `{{ include }}` shared fragments from `prompts/`.
- **`config.toml` = data only.** Model, tier, and engine knobs are pure
  data with no logic, so TOML is a better home than a Python literal:
  trivially mergeable for env overlays, diffable, and safe for non-devs.
- **`agent.py` = thin assembly.** This is where code-first lives. It
  loads the prose + data, wires real `tools`/`schema`, and returns a
  typed `Agent`. Usually ~10 lines.
- **`pipelines/` separate.** Definition vs orchestration. The same agent
  can appear in many pipelines; it shouldn't import any of them.
- **`config/{env}.toml` overlays.** One agent definition; environment
  changes the model/retries/etc. via a merged overlay, not a copy.

---

## 4. File-by-file specification

### `agents/<name>/prompt.md`

The system prompt as prose. May reference shared fragments:

```markdown
{% include "prompts/_house_style.md" %}

You are a research expert. Given a question, you find primary sources,
cross-check claims, and return a structured summary with citations.

- Prefer primary sources over blogs.
- Flag any claim you could not verify.
```

(Include syntax is illustrative — the loader decides the mechanism;
see open question Q3.)

### `agents/<name>/config.toml` (thin)

Data only. No tools, no guards, no control flow.

```toml
# Model selection — either a concrete id or a provider+tier alias.
model = "claude-opus-4-8"
# or:
# provider = "anthropic"
# tier     = "top"

[engine]
temperature = 0.2
max_turns   = 20
thinking    = true
cache       = true
timeout     = 120
```

### `agents/<name>/agent.py` (assembly)

Thin, typed, code-first. Loads prose + data, wires the imperative parts:

```python
# agents/researcher/agent.py
from lazybridge import Agent, LLMEngine
from agents._loader import load           # convention: same folder as caller
from .schema import Research
from .tools import search

def build(env: str = "dev") -> Agent:
    cfg, system = load(__file__, env=env) # reads config.toml + overlay + prompt.md
    return Agent(
        name="researcher",                # == folder name (lint-enforced)
        engine=LLMEngine(cfg.model, system=system, **cfg.engine),
        tools=[search],
        output=Research,
    )
```

### `agents/_loader.py`

A small helper (~50 lines) that, given a caller's `__file__`:

1. reads sibling `config.toml`;
2. deep-merges `config/defaults.toml` → `config/<env>.toml` →
   the agent's `config.toml` (later wins);
3. reads sibling `prompt.md`, resolving any shared-fragment includes;
4. returns a small typed object `(cfg, system_prompt)`.

It does **not** construct the `Agent` — that stays in `agent.py`, so the
code-first contract is preserved and type-checked.

### `agents/registry.py`

Discovers `agents/*/agent.py`, imports each `build()`, and exposes the
fleet by name:

```python
from agents.registry import get, all_agents

researcher = get("researcher", env="prod")   # one agent
fleet      = all_agents(env="prod")           # {name: Agent}
```

Because folder name == `Agent(name=...)` == registry key, the LazyBridge
Name Chain keeps working unchanged: `from_agent("researcher")`,
`Step(name="researcher")`, and routing all resolve against the same key.

### `pipelines/<flow>.py`

Pure composition over registry agents:

```python
from lazybridge import Agent, Plan, Step, from_prev, from_step
from agents.registry import get

def ticket_flow(env="prod") -> Agent:
    return Agent(
        name="ticket_flow",
        engine=Plan(
            Step(target=get("researcher", env), name="researcher"),
            Step(target=get("writer", env), name="writer",
                 task=from_prev, context=from_step("researcher")),
        ),
    )
```

---

## 5. Environment overlays

Precedence, lowest to highest:

```
config/defaults.toml  →  config/<env>.toml  →  agents/<name>/config.toml
```

So a project-wide `prod` overlay can bump every agent to the top tier and
raise retries, while an individual agent can still pin a specific model.
This is exactly the "flat dict of fleet defaults" LazyBridge already
endorses (`PROD_DEFAULTS = dict(timeout=60, max_retries=5, ...)`), only
given a named, diffable home on disk.

> **Note.** Whether an agent's own `config.toml` should be allowed to
> override a project-mandated `prod` policy is a governance question —
> see open question Q4.

---

## 6. Prompt versioning & evals

- **Versions.** `prompt.md` is canonical; `prompt.v2.md` (etc.) live
  alongside for A/B and staged rollout. `config.toml` can name the active
  one (`prompt = "prompt.v2.md"`); default is `prompt.md`.
- **Evals.** `evals/<name>/` mirrors agent names so a prompt change and
  its eval move together in the same PR. This dovetails with the existing
  `lazybridge.ext.evals` framework.

---

## 7. Alternatives considered

### A. Status quo — everything inline in Python (rejected for *large* projects)

The current zero-config approach. **Keep it as the default** for small
projects and all examples. We reject it only as the *recommended pattern
for large fleets*, for the mixed-lifecycle and discovery reasons in §1.

### B. Rich / declarative config — "agent from YAML" (rejected; this was the explicit fork)

`config.toml` (or YAML) would also describe tools, guards, output
schemas, and even composition — a near-complete agent serialised as data.

- **Pro:** maximally accessible to non-developers; a whole agent editable
  without Python.
- **Con:** it fights the code-first core head-on. Tool references become
  stringly-typed names resolved at runtime (no autocomplete, no mypy),
  guards/validators need a mini-expression-language or plugin registry,
  and you slowly reinvent a worse Python. It also duplicates a concept the
  framework intentionally keeps as code.
- **Decision: rejected.** We go **thin config** (§2.7): data-only knobs in
  TOML, everything imperative in Python.

### C. Single flat config file for the whole fleet (rejected)

One big `agents.toml`/`fleet.py` listing every agent.

- **Con:** poor locality (one agent's prompt is nowhere near its tools),
  painful merge conflicts on a hot file, and no natural home for the
  prose prompt. Folder-per-agent wins on locality and reviewability.

### D. Convention only, no loader (viable fallback)

Document the layout but ship no `_loader.py`/`registry.py`; each project
writes its own glue.

- **Pro:** zero new surface to maintain in the ecosystem.
- **Con:** every team rewrites the same merge/discovery glue and they
  drift apart. Better to ship a tiny, optional, well-tested helper.

---

## 8. Trade-offs of the recommended approach

| Benefit | Cost |
| --- | --- |
| Prose prompts → clean diffs, prose review, non-dev edits | One agent now spans ~4 files instead of one block |
| Data-only TOML → trivial env overlays | A loader/merge layer to build and maintain |
| Folder == name == registry key (Name Chain reuse) | A lint rule needed to enforce the equality |
| Definition vs composition separated | More indirection to trace one end-to-end run |
| Optional & layered — core stays zero-config | Two ways to do things (inline vs layout) to document |

The indirection cost is real, which is why this is **opt-in for large
projects only**, never the default and never something the core depends
on.

---

## 9. Open questions

- **Q1. Home & form.** Ship this as (a) a docs recipe + example repo,
  (b) a concrete `lazytools.project` module (loader + `registry` +
  `lazytools new-agent` scaffolder), or (c) keep as design doc for now.
  *(Current decision: design doc first.)*
- **Q2. Config format.** TOML (chosen here for native typing + comments)
  vs YAML (more familiar to ops, but type-ambiguous). Confirm TOML.
- **Q3. Prompt includes.** What mechanism resolves shared fragments —
  Jinja, a tiny custom `{{ include }}`, or plain string concatenation in
  the loader? Prefer the simplest that works.
- **Q4. Override governance.** May an agent's own `config.toml` override
  a project `prod` overlay, or should some policy keys be locked at the
  environment level?
- **Q5. Loader home.** If we build the loader, does it belong in
  LazyTools (toolkit layer, depends on LazyBridge) or as a documented
  snippet projects copy? LazyTools is the natural fit.

---

## 10. Proposed next step

Agree on §2 principles and the §3 layout, resolve Q1–Q5, then — if we
choose to build it — implement a minimal `_loader.py` + `registry.py`
(target ~100 lines, fully tested) plus one worked example under
`examples/`, most likely housed in LazyTools per Q5.
