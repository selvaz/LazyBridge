# lazybridge.tools — Ready-made tools

## Overview

`lazybridge.tools` is a subpackage of ready-made LazyBridge-compatible tools.
Each tool is a standalone module exposing plain Python functions that can be
called directly or wrapped as a `LazyTool` for any agent or pipeline.

```
lazybridge/
  tools/
    __init__.py
    doc_skills.py   ← BM25 local documentation skill
    read_docs.py    ← multi-format document reader
```

Import pattern:
```python
from lazybridge.tools.doc_skills import build_skill, skill_tool, skill_pipeline
from lazybridge.tools.read_docs  import read_folder_docs
```

Optional dependencies for `read_docs`:
```bash
pip install lazybridge[tools]   # pypdf, python-docx, trafilatura
```

---

## tools.doc_skills

Local documentation skill runtime. Index a folder of docs once, query them
from any agent using full BM25 retrieval. No vector DB, no embeddings API.

### Public API

| Symbol | Type | Description |
|---|---|---|
| `build_skill(source_dirs, skill_name, ...)` | `→ dict` | Index docs → skill bundle on disk |
| `query_skill(skill_dir, task, ...)` | `→ str` | BM25 retrieval → grounded context brief |
| `skill_tool(skill_dir, ...)` | `→ LazyTool` | One-step tool for any agent |
| `skill_builder_tool(...)` | `→ LazyTool` | Tool that builds skill bundles on demand |
| `skill_pipeline(skill_dir, ...)` | `→ LazyTool` | Two-step router + executor chain |

### build_skill

```python
from lazybridge.tools.doc_skills import build_skill

meta = build_skill(
    source_dirs    = ["./docs", "./reference"],
    skill_name     = "my-project",
    output_root    = "./generated_skills",   # default
    description    = "API reference for MyProject.",
    usage_notes    = "",
    include_extensions = [".md", ".txt", ".py", ...],  # default: all common types
    chunk_size     = 1800,   # max chars per chunk
    chunk_overlap  = 180,    # overlap for char-mode chunks
    copy_sources   = False,  # copy original files into bundle
    overwrite      = True,
    max_chars_per_file = 200_000,
)
# Returns: {skill_dir, skill_name, description, indexed_files, total_chunks, avgdl}
```

**Skill bundle written to disk:**
```
generated_skills/my-project/
  SKILL.md        ← LLM operating instructions
  manifest.json   ← metadata + avgdl (needed for BM25 scoring)
  vocab.json      ← Robertson IDF weights for every term
  chunks.jsonl    ← one JSON object per chunk
```

**Chunking strategy:**
- `.md .mdx .rst .adoc` → heading-aware: each `#` section = one chunk
- Everything else → character-based, snapping to `\n\n` boundaries
- Oversized sections: sub-split; tiny adjacent sections: merged

### query_skill

```python
from lazybridge.tools.doc_skills import query_skill

brief = query_skill(
    skill_dir     = "./generated_skills/my-project",
    task          = "How do I configure retry behaviour?",
    mode          = "auto",   # auto | answer | extract | locate | summarize
    top_k         = 8,
    max_chars     = 10_000,
    include_quotes = True,
)
```

**Modes:**

| Mode | Triggered by (auto) | Output |
|---|---|---|
| `answer` | default | Evidence bullets + full excerpts |
| `extract` | "extract", "list all", "elenca" | Raw quoted sections |
| `locate` | "where", "find", "dove" | Ranked file paths only |
| `summarize` | "summarize", "overview", "riassumi" | Compact bullets |

**BM25 scoring:**
```
score(chunk, query) = Σ_t  IDF(t) · tf_norm(t, chunk)
                    + heading_match_boost (1.2 per token)
                    + exact_phrase_boost  (4.0 if phrase found)

IDF(t) = log( (N - df + 0.5) / (df + 0.5) + 1 )   # Robertson IDF
tf_norm = tf · (k1+1) / ( tf + k1·(1-b + b·|d|/avgdl) )
k1=1.5, b=0.75
```

### skill_tool

```python
from lazybridge.tools.doc_skills import skill_tool
from lazybridge import LazyAgent

tool = skill_tool(
    skill_dir   = "./generated_skills/my-project",
    name        = None,      # defaults to slugified skill name
    description = None,      # defaults to skill description
    guidance    = None,      # injected into calling agent's system prompt
    strict      = False,
)
resp = LazyAgent("anthropic").loop("How does X work?", tools=[tool])
```

### skill_builder_tool

```python
from lazybridge.tools.doc_skills import skill_builder_tool

builder = skill_builder_tool()
orchestrator.loop(
    "Index the /docs folder as 'api-reference' and save to ./generated_skills",
    tools=[builder],
)
```

### skill_pipeline

Two-step chain: `skill_router` → `skill_executor`.

```python
from lazybridge.tools.doc_skills import skill_pipeline
from lazybridge import LazyAgent

pipeline = skill_pipeline(
    skill_dir      = "./generated_skills/my-project",
    provider       = "anthropic",
    router_model   = None,   # defaults to provider default
    executor_model = None,
    session        = None,   # LazySession created if omitted
    native_tools   = None,
)
resp = LazyAgent("anthropic").loop(
    "What is the canonical pattern for a sequential pipeline?",
    tools=[pipeline],
)
```

**Chain wiring:**
```
user task
    │
    ▼  router (LazyAgent, no tools, chat())
    │  Rewrites query for BM25. Preserves all technical identifiers.
    │
    ▼  executor (LazyAgent, tools=[skill_tool], loop())
    │  Calls skill_tool → BM25 retrieval.
    │  Synthesises grounded answer citing source files.
    ▼
orchestrator
```

`executor` has `tools=[skill_tool]` at construction → `sess.as_tool(mode="chain")`
detects tools and calls `loop()` automatically.

---

## tools.read_docs

Multi-format document reader. Returns LLM-ready text from any combination of
`.txt`, `.md`, `.pdf`, `.docx`, `.html` files.

### Public API

| Symbol | Description |
|---|---|
| `read_folder_docs(path, ...)` | Read a file or folder, return text or JSON |

```python
from lazybridge.tools.read_docs import read_folder_docs

# Single file
text = read_folder_docs("/reports/q4.pdf")

# Folder — filtered
text = read_folder_docs("/reports", extensions="pdf,docx", recursive=True)

# As LazyTool
from lazybridge import LazyAgent, LazyTool
tool = LazyTool.from_function(read_folder_docs)
resp = LazyAgent("anthropic").loop("Summarise all PDFs in /reports", tools=[tool])
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `path` | required | Single file or folder |
| `extensions` | `"txt,md,pdf,docx,html"` | Comma-separated; ignored for single files |
| `html_mode` | `"parsed"` | `parsed` / `full` / `both` |
| `recursive` | `False` | Scan subfolders; ignored for single files |
| `output_format` | `"text"` | `text` (LLM-readable) or `json` (with metadata) |

### Supported formats

| Extension | Reader | Notes |
|---|---|---|
| `.txt` `.md` | built-in | UTF-8 |
| `.pdf` | `pypdf` | Text extraction per page |
| `.docx` | `python-docx` | Paragraphs + tables |
| `.html` `.htm` | `trafilatura` | Strips nav/ads (parsed mode) |

Graceful degradation: if a dependency is missing, the tool returns a message
instead of raising (`[PDF unavailable — pip install pypdf]`).

### Install optional deps

```bash
pip install lazybridge[tools]
# or individually:
pip install pypdf python-docx trafilatura
```
