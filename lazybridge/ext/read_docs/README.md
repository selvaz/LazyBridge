# read_docs — Multi-format document reader

Read one or more documents and return their text in a format ready for an LLM.
Supports `.txt`, `.md`, `.pdf`, `.docx`, `.html` files.

## Install

```bash
pip install lazybridge[tools]   # installs pypdf, python-docx, trafilatura
```

## Quick start

### Plain function

```python
from lazybridge.ext.read_docs import read_folder_docs

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
from lazybridge.ext.read_docs import read_folder_docs

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

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `path` | required | Single file path or folder path |
| `extensions` | `"txt,md,pdf,docx,html"` | Comma-separated; ignored for single files |
| `html_mode` | `"parsed"` | `parsed` (clean text) / `full` (raw HTML) / `both` |
| `recursive` | `False` | Scan subfolders; ignored for single files |
| `output_format` | `"text"` | `"text"` (LLM-readable) or `"json"` (with metadata) |

## Supported formats

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
