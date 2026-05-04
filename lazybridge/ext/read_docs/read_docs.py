"""
lazybridge.ext.read_docs  —  Multi-format document reader
===========================================================

Reads .txt, .md, .pdf, .docx, .html files from a folder or a single file
and returns their text content in a format ready for LLM consumption.

Works as a plain Python function or as a Tool passed to any agent.

Usage — plain function:
    from lazybridge.ext.read_docs import read_folder_docs
    text = read_folder_docs("/path/to/reports", extensions="pdf,docx")

Usage — as a Tool:
    from lazybridge import Agent, Tool
    from lazybridge.ext.read_docs import read_folder_docs

    docs_tool = Tool(read_folder_docs)
    resp = Agent("anthropic", tools=[docs_tool])(
        "Summarise all PDFs in /reports",
    )

Optional dependencies (graceful degradation if missing):
    pip install lazybridge[tools]   # installs pypdf, python-docx, trafilatura
"""

from __future__ import annotations

import json
from pathlib import Path

# ── Per-format readers ─────────────────────────────────────────────────────────


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    try:
        import pypdf
    except ImportError:
        return "[PDF unavailable — pip install pypdf]"
    reader = pypdf.PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def _read_docx(path: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError:
        return "[Word unavailable — pip install python-docx]"
    doc = docx.Document(str(path))
    blocks: list[str] = []
    for block in doc.element.body:
        tag = block.tag.split("}")[-1]
        if tag == "p":
            text = "".join(n.text or "" for n in block.iter() if hasattr(n, "text"))
            if text.strip():
                blocks.append(text.strip())
        elif tag == "tbl":
            rows = []
            for row in block:
                cells = [
                    "".join(n.text or "" for n in cell.iter() if hasattr(n, "text")).strip()
                    for cell in row
                    if row.tag.split("}")[-1] in ("tr",) or True
                ]
                rows.append(" | ".join(c for c in cells if c))
            if rows:
                blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def _read_html_parsed(path: Path) -> str:
    """Clean body text via trafilatura — strips nav, ads, boilerplate."""
    try:
        import trafilatura
    except ImportError:
        return "[trafilatura unavailable — pip install trafilatura]"
    raw = path.read_text(encoding="utf-8", errors="replace")
    result = trafilatura.extract(raw, include_comments=False, include_tables=True, no_fallback=False)
    return result or "[trafilatura could not extract text from this page]"


def _read_html_full(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_html(path: Path, mode: str) -> str:
    if mode == "full":
        return _read_html_full(path)
    if mode == "both":
        return f"[PARSED BODY]\n{_read_html_parsed(path)}\n\n[FULL HTML]\n{_read_html_full(path)}"
    return _read_html_parsed(path)


_EXT_READERS: dict[str, object] = {
    ".txt": lambda p, _: _read_txt(p),
    ".md": lambda p, _: _read_txt(p),
    ".pdf": lambda p, _: _read_pdf(p),
    ".docx": lambda p, _: _read_docx(p),
    ".html": _read_html,
    ".htm": _read_html,
}


# ── Public API ─────────────────────────────────────────────────────────────────


def read_folder_docs(
    path: str,
    extensions: str = "txt,md,pdf,docx,html",
    html_mode: str = "parsed",
    recursive: bool = False,
    output_format: str = "text",
    *,
    base_dir: str | None = None,
) -> str:
    """Read documents from a file or folder and return their text content.

    Accepts either a single file path or a folder path.
    When given a folder, scans for all matching files (optionally recursive).
    When given a file, reads that file directly regardless of the extensions filter.

    Supported formats: .txt, .md, .pdf, .docx, .html/.htm.
    HTML files can be returned as clean extracted body text, raw HTML, or both.

    Args:
        path: Path to a single file OR a folder to scan.
            File example:   "/reports/q4.pdf"
            Folder example: "/reports"
        extensions: Comma-separated list of file extensions to include when
            scanning a folder. Ignored when path points to a single file.
            Supported values: txt, md, pdf, docx, html.
            Default: "txt,md,pdf,docx,html" (all formats).
            Example: "pdf,docx" to read only PDFs and Word files.
        html_mode: How to process HTML and HTM files.
            "parsed" — clean readable text extracted by trafilatura (default).
            "full"   — raw HTML source, unmodified.
            "both"   — parsed body text first, then raw HTML source.
        recursive: Whether to search subfolders recursively when path is a folder.
            False (default) — top-level files only.
            True  — all files in all subfolders.
            Ignored when path points to a single file.
        output_format: How to format the combined output.
            "text" (default) — a single human/LLM-readable string with headers.
            "json" — a JSON array with per-file metadata and content.

    Returns:
        A single string containing the text of all matched documents, or an
        error description string if the path is not found.
    """
    target = Path(path).expanduser().resolve()

    # When exposed as an agent tool, `path` is LLM-controlled and therefore
    # untrusted.  If the caller supplies `base_dir`, refuse any path that
    # resolves outside that sandbox.
    if base_dir is not None:
        base = Path(base_dir).expanduser().resolve()
        try:
            target.relative_to(base)
        except ValueError:
            return f"[Error: refused — path {str(target)!r} escapes base_dir {str(base)!r}]"

    if not target.exists():
        return f"[Error: path not found — {path}]"

    if target.is_file():
        files = [target]
        root = target.parent
    elif target.is_dir():
        root = target
        exts: set[str] = set()
        for e in extensions.split(","):
            e = e.strip().lstrip(".").lower()
            if e:
                exts.add(f".{e}")
        if ".html" in exts:
            exts.add(".htm")
        glob_pattern = "**/*" if recursive else "*"
        # Walk the tree without following symlinks.  Doing so closes
        # symlink-loop hangs and prevents a symlink in the indexed
        # folder from silently widening the read surface to other
        # directories.
        files = sorted(
            f for f in root.glob(glob_pattern) if f.is_file() and not f.is_symlink() and f.suffix.lower() in exts
        )
        if not files:
            return f"[No documents found in '{path}' matching extensions: {extensions}]"
    else:
        return f"[Error: path is neither a file nor a directory — {path}]"

    records: list[dict] = []
    for fpath in files:
        suffix = fpath.suffix.lower()
        reader = _EXT_READERS.get(suffix)
        if reader is None:
            content = f"[Unsupported extension: {suffix}]"
        else:
            try:
                content = reader(fpath, html_mode)  # type: ignore[operator]
            except Exception as exc:
                content = f"[Error reading file: {exc}]"
        records.append(
            {
                "filename": fpath.name,
                "relative_path": str(fpath.relative_to(root)),
                "extension": suffix.lstrip("."),
                "size_bytes": fpath.stat().st_size,
                "char_count": len(content),
                "content": content,
            }
        )

    if output_format == "json":
        return json.dumps(records, ensure_ascii=False, indent=2)

    parts: list[str] = []
    for rec in records:
        header = (
            f"{'=' * 72}\n"
            f"FILE : {rec['relative_path']}\n"
            f"TYPE : {rec['extension'].upper()}   SIZE : {rec['size_bytes']:,} bytes   CHARS : {rec['char_count']:,}\n"
            f"{'=' * 72}"
        )
        parts.append(f"{header}\n\n{rec['content']}")

    summary = (
        f"[{len(records)} document(s) read from '{path}' | "
        f"extensions: {extensions} | html_mode: {html_mode} | recursive: {recursive}]\n"
        f"{'─' * 72}\n\n"
    )
    return summary + "\n\n".join(parts)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Read documents from a file or folder and print their text content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m lazybridge.ext.read_docs /path/to/folder
  python -m lazybridge.ext.read_docs /path/to/file.pdf
  python -m lazybridge.ext.read_docs /path/to/folder --extensions pdf,docx --recursive
  python -m lazybridge.ext.read_docs /path/to/folder --format json
""",
    )
    parser.add_argument("path", help="File or folder to read")
    parser.add_argument(
        "--extensions", default="txt,md,pdf,docx,html", help="Comma-separated extensions (folder mode only)"
    )
    parser.add_argument("--html-mode", default="parsed", dest="html_mode", choices=["parsed", "full", "both"])
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--format", default="text", dest="output_format", choices=["text", "json"])
    args = parser.parse_args()

    print(
        read_folder_docs(
            path=args.path,
            extensions=args.extensions,
            html_mode=args.html_mode,
            recursive=args.recursive,
            output_format=args.output_format,
        )
    )
