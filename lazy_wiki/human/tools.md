# LazyBridge Tools

`lazybridge.tools` ships ready-made tools for the most common agent tasks.
Each tool works as a plain Python function **and** as a `LazyTool` you can
pass directly to any agent.

Install:
```bash
pip install lazybridge          # doc_skills — no extra deps
pip install lazybridge[tools]   # read_docs — adds pypdf, python-docx, trafilatura
```

---

## Available tools

| Module | What it does | Extra deps? |
|---|---|---|
| `lazybridge.tools.doc_skills` | Index local docs with BM25, query from any agent | None |
| `lazybridge.tools.read_docs` | Read .txt .md .pdf .docx .html from file or folder | `lazybridge[tools]` |

---

## doc_skills — Local documentation skill

Index a folder of documentation once.  Query it from any agent using full
BM25 retrieval — no vector database, no external API, everything local.

### Build and persist a skill bundle

```python
from lazybridge.tools.doc_skills import build_skill

meta = build_skill(
    source_dirs=["./docs", "./reference"],
    skill_name="my-project",
    description="API reference and guides for MyProject.",
    output_root="./generated_skills",   # default — persisted to disk
)
print(meta["skill_dir"])     # e.g. ./generated_skills/my-project
print(meta["total_chunks"])
print(meta["avgdl"])
```

The bundle is **saved to disk** and survives restarts. Run `build_skill` once,
or again whenever your docs change.  Point `skill_tool` at the same directory
every time — no re-indexing needed between sessions.

Add `generated_skills/` to your `.gitignore`.

### Load and use a saved skill

```python
from lazybridge.tools.doc_skills import skill_tool
from lazybridge import LazyAgent

# skill_dir points to the folder created by build_skill — works across restarts
tool = skill_tool("./generated_skills/my-project")

resp = LazyAgent("anthropic").loop(
    "How do I configure retry behaviour?",
    tools=[tool],
)
print(resp.content)
```

### Use the two-step pipeline

For complex or ambiguous queries, the pipeline first sharpens the query
(router), then retrieves and synthesises the answer (executor):

```python
from lazybridge.tools.doc_skills import skill_pipeline
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
print(resp.content)
```

```
user query
    │
    ▼  router  — rewrites into a precise retrieval query
    ▼  executor — calls the skill (BM25) → synthesises grounded answer
    │
orchestrator
```

### Let an agent build skills on demand

```python
from lazybridge.tools.doc_skills import skill_builder_tool
from lazybridge import LazyAgent

builder = skill_builder_tool()
agent = LazyAgent("anthropic")
agent.loop(
    "Index the /company/docs folder as 'company-wiki'",
    tools=[builder],
)
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
from lazybridge.tools.doc_skills import query_skill

brief = query_skill(
    "./generated_skills/my-project",
    "Where is verify= documented?",
    mode="locate",
)
```

---

## read_docs — Multi-format document reader

Read one or more documents and return their text in a format ready for an LLM.

```bash
pip install lazybridge[tools]
```

### Plain function

```python
from lazybridge.tools.read_docs import read_folder_docs

# Single file
text = read_folder_docs("/reports/q4.pdf")

# Folder — only PDFs and Word docs
text = read_folder_docs("/reports", extensions="pdf,docx")

# Folder — recursive, all formats
text = read_folder_docs("/reports", recursive=True)

# JSON output with metadata
import json
records = json.loads(read_folder_docs("/reports", output_format="json"))
for r in records:
    print(r["filename"], r["char_count"])
```

### As a LazyTool in an agent

```python
from lazybridge import LazyAgent, LazyTool
from lazybridge.tools.read_docs import read_folder_docs

docs_tool = LazyTool.from_function(
    read_folder_docs,
    guidance="Call this tool to read files before summarising or analysing them.",
)
resp = LazyAgent("anthropic").loop(
    "Read all PDFs in /reports/q4 and give me a one-page executive summary.",
    tools=[docs_tool],
)
print(resp.content)
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `path` | required | Single file path **or** folder path |
| `extensions` | `"txt,md,pdf,docx,html"` | Comma-separated; ignored for single files |
| `html_mode` | `"parsed"` | `parsed` (clean text) / `full` (raw HTML) / `both` |
| `recursive` | `False` | Scan subfolders; ignored for single files |
| `output_format` | `"text"` | `"text"` (LLM-readable) or `"json"` (with metadata) |

### Supported formats

| Format | Library | Notes |
|---|---|---|
| `.txt` `.md` | built-in | UTF-8 |
| `.pdf` | `pypdf` | Text extracted per page |
| `.docx` | `python-docx` | Paragraphs and tables |
| `.html` `.htm` | `trafilatura` | `parsed` mode strips nav, ads, boilerplate |

If a dependency is missing the tool returns a clear message instead of crashing:
```
[PDF unavailable — pip install pypdf]
```

---

## Adding new tools

New tools go in `lazybridge/tools/` as standalone `.py` files.
Add a row to the tables in this file and in `lazy_wiki/bot/12_tools.md`.

Convention:
- One or more plain functions with full type annotations and docstrings
- `Annotated[type, "description"]` on parameters — `LazyTool.from_function` picks them up automatically
- Graceful degradation for optional dependencies (`try/except ImportError`)
- No mandatory dependencies beyond the standard library
