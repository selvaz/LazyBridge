# doc_skills — Local documentation skill runtime

Index local documentation folders into a portable skill bundle, then expose
the bundle as a LazyBridge tool or pipeline that any agent can call.

Retrieval uses full BM25 (Robertson IDF, k1=1.5, b=0.75) with IDF weights
computed once at index time. No external dependencies beyond the standard library.

## Install

```bash
pip install lazybridge   # no extra deps needed
```

## Quick start

### Build and persist a skill bundle

```python
from lazybridge.external_tools.doc_skills import build_skill

meta = build_skill(
    source_dirs=["./docs", "./reference"],
    skill_name="my-project",
    description="API reference and guides for MyProject.",
    output_root="./generated_skills",
)
print(meta["skill_dir"])     # e.g. ./generated_skills/my-project
print(meta["total_chunks"])
```

Add `generated_skills/` to your `.gitignore`.

### Load and use a saved skill

```python
from lazybridge.external_tools.doc_skills import skill_tool
from lazybridge import Agent

tool = skill_tool("./generated_skills/my-project")

resp = Agent("anthropic", tools=[tool])("How do I configure retry behaviour?")
print(resp.text())
```

### Two-step pipeline (router + executor)

```python
from lazybridge.external_tools.doc_skills import skill_pipeline
from lazybridge import Agent

pipeline = skill_pipeline(
    skill_dir="./generated_skills/my-project",
    provider="anthropic",
)
orchestrator = Agent("anthropic", tools=[pipeline])
resp = orchestrator("What is the canonical pattern for agents running in sequence?")
print(resp.text())
```

### Let an agent build skills on demand

```python
from lazybridge.external_tools.doc_skills import skill_builder_tool
from lazybridge import Agent

builder = skill_builder_tool()
agent = Agent("anthropic", tools=[builder])
agent("Index the /company/docs folder as 'company-wiki'")
```

### Query retrieval modes

| Mode | When | Example query |
|---|---|---|
| `answer` | default | "How does X work?" |
| `extract` | need raw quotes | "Extract all examples of verify=" |
| `locate` | need file paths | "Where is Store documented?" |
| `summarize` | overview | "Summarise the pattern hierarchy" |

Mode is detected automatically from the query wording, or set it explicitly:

```python
from lazybridge.external_tools.doc_skills import query_skill

brief = query_skill(
    "./generated_skills/my-project",
    "Where is verify= documented?",
    mode="locate",
)
```

## Public API

| Function | Description |
|---|---|
| `build_skill(source_dirs, skill_name, ...)` | Index docs → skill bundle on disk |
| `query_skill(skill_dir, task, ...)` | Retrieve + grounded context brief |
| `skill_tool(skill_dir, ...)` | One-step `Tool` for an agent |
| `skill_builder_tool(...)` | Tool that builds skill bundles |
| `skill_pipeline(skill_dir, ...)` | Router + executor chain pipeline |
