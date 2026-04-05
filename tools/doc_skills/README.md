# doc_skills_tool

Local documentation skill runtime for LazyBridge.

Index any folder of documentation files once, then query them from any agent
or pipeline using full BM25 retrieval.  No external vector database, no
embeddings API, no extra services — everything runs locally.

---

## How it works

```
source docs                skill bundle                  agent
────────────               ────────────────              ──────────────────────
docs/          build_skill  SKILL.md        skill_tool   LazyAgent.loop(
reference/   ──────────►   manifest.json  ──────────►     task,
wiki/          (once)       vocab.json      (per query)    tools=[skill_tool],
                            chunks.jsonl                  )
```

**Build phase** (run once, or when docs change):
- Reads every supported file in the source folders
- Splits into chunks using heading-aware chunking for Markdown/RST
- Computes Robertson IDF weights across all chunks → saved in `vocab.json`
- Writes `manifest.json` with metadata including `avgdl` (needed for BM25)
- Generates `SKILL.md` — operating instructions injected into the LLM context

**Query phase** (every agent call):
- Tokenises the query
- Scores all chunks with BM25 (TF × IDF, length-normalised)
- Returns a grounded context brief: skill instructions + source paths + excerpts
- No LLM call — retrieval is pure Python

---

## Retrieval quality

BM25 (Robertson IDF, k1=1.5, b=0.75) vs the naïve TF approach:

| | TF-only | **BM25** |
|---|---|---|
| Rare technical terms (class names, param names) | Same weight as common words | **High weight** |
| Long vs short chunks | Long chunks always win | **Length-normalised** |
| IDF computation | Per query (slow) | **At index time (fast)** |
| Heading match | Not used | **+1.2 boost** |
| Exact phrase match | Not used | **+4.0 boost** |

---

## Quick start

```python
from doc_skills_tool import build_skill, skill_tool
from lazybridge import LazyAgent

# 1. Build once
meta = build_skill(
    source_dirs=["./docs", "./reference"],
    skill_name="my-project",
    description="API reference for MyProject.",
)

# 2. Use in an agent
tool = skill_tool(meta["skill_dir"])
resp = LazyAgent("anthropic").loop(
    "How do I configure retry behaviour?",
    tools=[tool],
)
print(resp.content)
```

---

## Public API

### `build_skill(source_dirs, skill_name, ...)`

Index documentation folders into a skill bundle.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source_dirs` | `list[str]` | required | Folders to index |
| `skill_name` | `str` | required | Bundle name / folder name |
| `output_root` | `str` | `./generated_skills` | Where to write the bundle |
| `description` | `str` | auto | What the skill covers |
| `usage_notes` | `str` | `""` | Extra rules appended to SKILL.md |
| `include_extensions` | `list[str]` | `.md .txt .rst .py .json …` | File types to index |
| `chunk_size` | `int` | `1800` | Max characters per chunk |
| `chunk_overlap` | `int` | `180` | Overlap for char-based splits |
| `copy_sources` | `bool` | `False` | Copy original files into bundle |
| `overwrite` | `bool` | `True` | Replace existing bundle |

Returns `dict` with `skill_dir`, `indexed_files`, `total_chunks`, `avgdl`.

---

### `query_skill(skill_dir, task, ...)`

Retrieve relevant chunks and return a grounded context brief.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `skill_dir` | `str` | required | Path to a skill bundle |
| `task` | `str` | required | Question or task |
| `mode` | `str` | `"auto"` | `auto` / `answer` / `extract` / `locate` / `summarize` |
| `top_k` | `int` | `8` | Chunks to retrieve |
| `max_chars` | `int` | `10000` | Max characters in the returned brief |
| `include_quotes` | `bool` | `True` | Append full excerpts (answer mode) |

**Modes:**
- `auto` — detects intent from the task wording
- `answer` — synthesise an answer from evidence bullets + full excerpts
- `extract` — return raw quoted sections grouped by source file
- `locate` — return only the ranked list of relevant file paths
- `summarize` — compact bullets from the top chunks

No LLM call — pure Python BM25 retrieval.

---

### `skill_tool(skill_dir, ...)`

Wrap `query_skill()` as a `LazyTool` ready for any agent.

```python
tool = skill_tool(
    skill_dir="./generated_skills/my-project",
    name="my_project_docs",          # optional — defaults to skill name
    description="...",               # optional — defaults to skill description
    guidance="Call this tool when …" # optional — injected into agent system prompt
)
agent.loop(task, tools=[tool])
```

---

### `skill_builder_tool(...)`

A `LazyTool` that builds skill bundles on demand.
Pass to an orchestrator to let it index new docs autonomously.

```python
builder = skill_builder_tool()
orchestrator.loop("Index the /docs folder as 'api-reference'", tools=[builder])
```

---

### `skill_pipeline(skill_dir, ...)`

Two-step pipeline exposed as a single `LazyTool`:

```
user task
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  skill_router  (LazyAgent)                              │
│  Rewrites the task into a retrieval-optimised query.    │
│  Preserves all technical identifiers.                   │
└───────────────────────┬─────────────────────────────────┘
                        │ sharpened query
                        ▼
┌─────────────────────────────────────────────────────────┐
│  skill_executor  (LazyAgent + skill_tool)               │
│  Calls the skill tool → BM25 retrieval.                 │
│  Synthesises a grounded answer citing source files.     │
└───────────────────────┬─────────────────────────────────┘
                        │ grounded answer
                        ▼
                  orchestrator
```

```python
pipeline     = skill_pipeline(skill_dir="./generated_skills/my-project")
orchestrator = LazyAgent("anthropic")
resp = orchestrator.loop("When should I use LazyStore?", tools=[pipeline])
print(resp.content)
```

| Parameter | Default | Description |
|---|---|---|
| `skill_dir` | required | Path to a skill bundle |
| `provider` | `"anthropic"` | Provider for both agents |
| `router_model` | `None` | Model override for the router |
| `executor_model` | `None` | Model override for the executor |
| `session` | `None` | Existing `LazySession` (created if omitted) |
| `native_tools` | `None` | Provider-native tools for the executor |

---

## Skill bundle structure

```
generated_skills/
  my-project/
    SKILL.md        ← operating instructions for the LLM
    manifest.json   ← metadata + avgdl (required for BM25 scoring)
    vocab.json      ← Robertson IDF weights for every indexed term
    chunks.jsonl    ← one JSON object per chunk
    sources/        ← optional copy of original docs (copy_sources=True)
```

Add `generated_skills/` to your `.gitignore` — bundles are generated artifacts.

---

## Chunking strategy

| File type | Strategy |
|---|---|
| `.md` `.mdx` `.rst` `.adoc` | **Heading-aware** — each `#` section becomes a chunk; large sections sub-split; tiny sections merged |
| `.txt` `.py` `.json` `.yaml` `.toml` | **Character-based** — splits at `chunk_size` chars, snaps to `\n\n` paragraph boundaries |

Heading text is stored in the `heading` field and used as a BM25 title-boost signal.

---

## Dependencies

No extra dependencies beyond the standard library.
LazyBridge is required only for `skill_tool()`, `skill_builder_tool()`, and `skill_pipeline()`.

```bash
pip install lazybridge
```

---

## Running the test

```bash
# From the repo root
python tools/doc_skills/test_doc_skills.py
```

Or open `test_doc_skills.py` in Spyder and press **F5**.
Sections 1–2 (build + retrieval) run without an API key.
Sections 3–4 (agent calls) require `ANTHROPIC_API_KEY` in `.env`.
