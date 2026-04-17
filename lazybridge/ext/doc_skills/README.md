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
from lazybridge.ext.doc_skills import build_skill

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
from lazybridge.ext.doc_skills import skill_tool
from lazybridge import LazyAgent

tool = skill_tool("./generated_skills/my-project")

resp = LazyAgent("anthropic").loop(
    "How do I configure retry behaviour?",
    tools=[tool],
)
print(resp.content)
```

### Two-step pipeline (router + executor)

```python
from lazybridge.ext.doc_skills import skill_pipeline
from lazybridge import LazyAgent

pipeline = skill_pipeline(
    skill_dir="./generated_skills/my-project",
    provider="anthropic",
)
orchestrator = LazyAgent("anthropic")
resp = orchestrator.loop(
    "What is the canonical pattern for agents running in sequence?",
    tools=[pipeline],
)
```

### Let an agent build skills on demand

```python
from lazybridge.ext.doc_skills import skill_builder_tool
from lazybridge import LazyAgent

builder = skill_builder_tool()
agent = LazyAgent("anthropic")
agent.loop("Index the /company/docs folder as 'company-wiki'", tools=[builder])
```

### Query retrieval modes

| Mode | When | Example query |
|---|---|---|
| `answer` | default | "How does X work?" |
| `extract` | need raw quotes | "Extract all examples of verify=" |
| `locate` | need file paths | "Where is LazyStore documented?" |
| `summarize` | overview | "Summarise the pattern hierarchy" |

Mode is detected automatically from the query wording, or set it explicitly:

```python
from lazybridge.ext.doc_skills import query_skill

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
| `skill_tool(skill_dir, ...)` | One-step LazyTool for an agent |
| `skill_builder_tool(...)` | Tool that builds skill bundles |
| `skill_pipeline(skill_dir, ...)` | Router + executor chain pipeline |
