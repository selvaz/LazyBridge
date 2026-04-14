# read_docs

LazyBridge-friendly document reader.

Reads `.txt`, `.md`, `.pdf`, `.docx`, `.html` files from a folder or a single
file and returns their text content in a format ready for LLM consumption.
Works as a plain Python function or as a `LazyTool` passed to any agent.

---

## Quick start

```python
# As a plain function
from lazybridge.ext.tools.read_docs import read_folder_docs

text = read_folder_docs("/path/to/reports", extensions="pdf,docx")
print(text)

# As a LazyBridge tool
from lazybridge import LazyTool, LazyAgent
from lazybridge.ext.tools.read_docs import read_folder_docs

docs_tool = LazyTool.from_function(read_folder_docs)
resp = LazyAgent("anthropic").loop(
    "Summarise all documents in /reports",
    tools=[docs_tool],
)
print(resp.content)
```

---

## `read_folder_docs(path, ...)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | required | Path to a **single file** or a **folder** |
| `extensions` | `str` | `"txt,md,pdf,docx,html"` | Comma-separated extensions to include when scanning a folder. Ignored for single files. |
| `html_mode` | `str` | `"parsed"` | How to process HTML: `parsed` (clean text via trafilatura), `full` (raw HTML), `both` |
| `recursive` | `bool` | `False` | Scan subfolders recursively. Ignored for single files. |
| `output_format` | `str` | `"text"` | `text` (LLM-readable string) or `json` (array with metadata) |

### Single file vs folder

```python
# Single file — extensions and recursive are ignored
text = read_folder_docs("/reports/q4.pdf")

# Folder — scan all matching files
text = read_folder_docs("/reports", extensions="pdf,docx", recursive=True)
```

### Output formats

**`text`** (default) — one string, each document preceded by a header:
```
════════════════════════════════════════════════════════════════════════
FILE : q4.pdf
TYPE : PDF   SIZE : 142,301 bytes   CHARS : 48,203
════════════════════════════════════════════════════════════════════════

... document text ...
```

**`json`** — array of objects, useful for programmatic processing:
```json
[
  {
    "filename": "q4.pdf",
    "relative_path": "q4.pdf",
    "extension": "pdf",
    "size_bytes": 142301,
    "char_count": 48203,
    "content": "..."
  }
]
```

---

## Supported formats

| Extension | Reader | Notes |
|---|---|---|
| `.txt` `.md` | Built-in | UTF-8, errors replaced |
| `.pdf` | `pypdf` | Text extraction per page |
| `.docx` | `python-docx` | Paragraphs + tables |
| `.html` `.htm` | `trafilatura` | Strips nav, ads, boilerplate (parsed mode) |

If a dependency is missing, the tool returns a descriptive message instead of crashing:
```
[PDF unavailable — pip install pypdf]
```

---

## Dependencies

```bash
pip install pypdf python-docx trafilatura
```

All three are optional — the tool degrades gracefully if any are missing.

---

## CLI usage

```bash
python -m lazybridge.ext.tools.read_docs /path/to/folder --extensions pdf,docx --recursive
python -m lazybridge.ext.tools.read_docs /path/to/folder --format json
python -m lazybridge.ext.tools.read_docs /path/to/file.pdf
```

---

## Use as a LazyTool

```python
import sys


from lazybridge import LazyAgent, LazyTool
from lazybridge.ext.tools.read_docs import read_folder_docs

docs_tool = LazyTool.from_function(
    read_folder_docs,
    name="read_documents",
    guidance="Call this tool to read the content of files before summarising or analysing them.",
)

agent = LazyAgent("anthropic")
resp  = agent.loop(
    "Read all PDFs in /reports/q4 and give me a one-page executive summary.",
    tools=[docs_tool],
)
print(resp.content)
```
