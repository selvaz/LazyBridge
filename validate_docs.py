#!/usr/bin/env python3
"""
LazyBridge Documentation Example Validator — Part 1/5
Header, imports, data types, constants.
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

SNIPPET_KINDS = frozenset({
    "local", "llm_chat", "llm_loop", "structured_output", "delegation",
    "session_chain", "session_parallel", "context", "memory", "verify",
    "native_tools", "async_code", "streaming", "persistence", "advanced", "unknown",
})

TESTABILITY_LEVELS = frozenset({
    "syntax_only", "local_exec", "smoke_exec", "full_exec", "skip",
})

STATUS_VALUES = frozenset({"pass", "fail", "skip", "syntax_error", "timeout"})

TRIAGE_CATEGORIES = frozenset({
    "docs_issue", "framework_issue", "runtime_issue", "missing_env",
    "non_runnable_example", "timeout", "unknown",
})

# Names that validate_docs knows how to auto-inject imports for
INJECTABLE_NAMES = {
    "LazyAgent", "LazyTool", "LazySession", "LazyContext",
    "LazyStore", "LazyRouter", "Memory",
    "NativeTool", "ThinkingConfig",
    "BaseModel",
}

PROVIDER_STRINGS = frozenset({
    "anthropic", "claude", "openai", "gpt", "google", "gemini", "deepseek",
})


@dataclass
class Snippet:
    """A single extracted Python code block with its metadata and result."""
    id: str                      # stable: relpath::section-slug::NN
    file: str                    # relative path to source markdown
    heading: str                 # nearest preceding heading text
    raw_code: str                # exact code as it appears in the markdown

    snippet_kind: str = "unknown"
    testability: str = "syntax_only"

    # populated during validation
    status: str = "skip"         # pass | fail | skip | syntax_error | timeout
    reason: str = ""
    triage: str = "unknown"
    mode_run: str = "none"
    stdout: str = ""
    stderr: str = ""
    duration_sec: float = 0.0
# ---------------------------------------------------------------------------
# Part 2/5 — Markdown extraction
# ---------------------------------------------------------------------------

HEADING_RE = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)

# Matches fenced code blocks: ```lang\n...code...\n```
FENCED_BLOCK_RE = re.compile(
    r'^```(?P<lang>[a-zA-Z0-9_]*)[^\n]*\n(?P<code>.*?)^```[ \t]*$',
    re.MULTILINE | re.DOTALL,
)

# Shell-like languages — never extract these
SHELL_LANGS = frozenset({"bash", "sh", "shell", "zsh", "fish", "console", "text", "yaml",
                          "json", "toml", "ini", "dockerfile", "sql", "xml", "html", "css"})

PYTHON_INDICATOR_PATTERNS = [
    r'\bimport\b', r'\bfrom\b.*\bimport\b', r'\bdef\b\s+\w',
    r'\bclass\b\s+\w', r'\bprint\b\s*\(', r'\bLazyAgent\b',
    r'\bLazyTool\b', r'\bLazySession\b', r'\bLazyContext\b', r'\bMemory\b',
]


def slugify(text: str) -> str:
    """Convert heading text to a stable URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    text = re.sub(r'^-+|-+$', '', text)
    return text or "unnamed"


def _looks_like_python(code: str) -> bool:
    """Heuristic for unlabeled code blocks: does this look like Python?"""
    hits = sum(1 for p in PYTHON_INDICATOR_PATTERNS if re.search(p, code))
    return hits >= 2


def _is_pseudo_code(code: str) -> bool:
    """Detect obvious pseudo-code / output-only blocks."""
    stripped = code.strip()
    # Pure output: starts with [ or { and has no assignments or imports
    if stripped and stripped[0] in ('[', '{') and 'import' not in code and '=' not in code:
        return True
    # Block that is only comments
    non_blank = [l for l in stripped.splitlines() if l.strip()]
    if non_blank and all(l.strip().startswith('#') for l in non_blank):
        return True
    return False


def extract_snippets(filepath: Path, repo_root: Path) -> list:
    """
    Parse a markdown file and return all Python Snippet objects.
    Each snippet is associated with the nearest preceding heading.
    """
    text = filepath.read_text(encoding="utf-8")
    rel_path = str(filepath.relative_to(repo_root))

    # Build ordered (position, heading_text) pairs
    headings = [(m.start(), m.group(2)) for m in HEADING_RE.finditer(text)]

    def heading_at(pos: int) -> str:
        best = "root"
        for h_pos, h_text in headings:
            if h_pos <= pos:
                best = h_text
            else:
                break
        return best

    snippets = []
    heading_counters: dict[str, int] = {}

    for match in FENCED_BLOCK_RE.finditer(text):
        lang = match.group("lang").lower().strip()
        code = match.group("code")

        # Determine if this is a Python block
        if lang in ("python", "py"):
            is_python = True
        elif lang in SHELL_LANGS:
            is_python = False
        elif lang == "":
            is_python = _looks_like_python(code)
        else:
            is_python = False

        if not is_python:
            continue

        # Skip pseudo-code / output blocks
        if _is_pseudo_code(code):
            continue

        heading = heading_at(match.start())
        slug = slugify(heading)
        idx = heading_counters.get(slug, 0)
        heading_counters[slug] = idx + 1
        example_id = f"{rel_path}::{slug}::{idx:02d}"

        snippets.append(Snippet(
            id=example_id,
            file=rel_path,
            heading=heading,
            raw_code=code,
        ))

    return snippets
# ---------------------------------------------------------------------------
# Part 3/5 — Classification + code transformation
# ---------------------------------------------------------------------------

def _has(code: str, *patterns: str) -> bool:
    """Return True if any pattern matches anywhere in code."""
    return any(re.search(p, code) for p in patterns)


def classify_snippet(snippet) -> None:
    """Classify snippet_kind and testability in-place using code heuristics."""
    code = snippet.raw_code

    # --- Non-runnable markers ---
    has_ellipsis = bool(re.search(r'^\s*\.\.\.\s*$', code, re.MULTILINE))
    has_placeholder = bool(re.search(
        r'<[A-Z][A-Z_]+>|YOUR_API_KEY|sk-ant-\.\.\.|REPLACE_ME', code
    ))
    # A function with no body beyond a bare `...` on the def line
    has_stub_fn = bool(re.search(r'def \w+\([^)]*\):\s*\n\s*\.\.\.\s*$', code, re.MULTILINE))

    # --- Feature flags ---
    uses_lazyagent    = _has(code, r'\bLazyAgent\b')
    uses_lazytool     = _has(code, r'\bLazyTool\b')
    uses_lazysession  = _has(code, r'\bLazySession\b')
    uses_lazycontext  = _has(code, r'\bLazyContext\b')
    uses_memory       = _has(code, r'\bMemory\b')
    uses_lazystore    = _has(code, r'\bLazyStore\b')
    uses_lazyrouter   = _has(code, r'\bLazyRouter\b')
    uses_chat         = _has(code, r'\.chat\s*\(', r'\.achat\s*\(')
    uses_loop         = _has(code, r'\.loop\s*\(', r'\.aloop\s*\(')
    uses_json_method  = _has(code, r'\.json\s*\(', r'\.ajson\s*\(')
    uses_streaming    = _has(code, r'stream\s*=\s*True')
    uses_async        = _has(code, r'\basync\s+def\b', r'\bawait\b', r'\basyncio\b')
    uses_as_tool      = _has(code, r'\.as_tool\s*\(', r'LazyTool\.from_agent\b')
    uses_mode_chain   = _has(code, r'mode\s*=\s*["\']chain["\']')
    uses_mode_parallel = _has(code, r'mode\s*=\s*["\']parallel["\']')
    uses_verify       = _has(code, r'\bverify\s*=')
    uses_native_tools = _has(code, r'\bNativeTool\b', r'native_tools\s*=')
    uses_output_schema = _has(code, r'\boutput_schema\b')
    uses_thinking     = _has(code, r'\bthinking\b\s*=')
    uses_db           = _has(code, r'db\s*=\s*["\']', r'LazyStore\s*\(\s*db')
    uses_gather       = _has(code, r'\.gather\s*\(')
    uses_any_framework = any([uses_lazyagent, uses_lazytool, uses_lazysession,
                               uses_lazycontext, uses_memory, uses_lazystore])

    # --- Kind ---
    if has_ellipsis or has_placeholder or has_stub_fn:
        kind = "unknown"
    elif uses_mode_chain:
        kind = "session_chain"
    elif uses_mode_parallel:
        kind = "session_parallel"
    elif uses_verify:
        kind = "verify"
    elif uses_streaming:
        kind = "streaming"
    elif uses_native_tools:
        kind = "native_tools"
    elif uses_output_schema or uses_json_method:
        kind = "structured_output"
    elif uses_as_tool:
        kind = "delegation"
    elif uses_memory:
        kind = "memory"
    elif uses_lazycontext:
        kind = "context"
    elif uses_lazystore or uses_db:
        kind = "persistence"
    elif uses_async or uses_gather:
        kind = "async_code"
    elif uses_lazyrouter:
        kind = "advanced"
    elif uses_loop:
        kind = "llm_loop"
    elif uses_chat or uses_lazyagent:
        kind = "llm_chat"
    elif uses_any_framework:
        kind = "local"
    else:
        kind = "local"
    snippet.snippet_kind = kind

    # --- Testability ---
    if has_ellipsis or has_placeholder or has_stub_fn:
        snippet.testability = "skip"
        snippet.reason = "Contains ellipsis / placeholder — not complete code"
        snippet.triage = "non_runnable_example"
        return

    stripped = code.strip()
    lines = [l for l in stripped.splitlines() if l.strip() and not l.strip().startswith('#')]

    if len(lines) == 0:
        snippet.testability = "skip"
        snippet.reason = "Empty or comment-only block"
        snippet.triage = "non_runnable_example"
        return

    if kind == "unknown":
        snippet.testability = "syntax_only"
        snippet.reason = "Unclassifiable pattern"
        snippet.triage = "non_runnable_example"
        return

    # Pure local Python (no framework imports needed)
    if kind == "local" and not uses_any_framework:
        snippet.testability = "local_exec"
        return

    # Import-only snippets — cheap to run
    is_import_only = all(
        l.startswith('import ') or l.startswith('from ') or l.startswith('#')
        for l in stripped.splitlines()
        if l.strip()
    )
    if is_import_only:
        snippet.testability = "local_exec"
        return

    # Streaming / async / native tools / persistence — expensive or complex
    if kind in ("streaming", "native_tools", "async_code", "persistence", "advanced"):
        snippet.testability = "full_exec"
        return
    if uses_thinking:
        snippet.testability = "full_exec"
        return

    # Session-based patterns — smoke-able
    if kind in ("session_chain", "session_parallel", "delegation",
                "verify", "structured_output", "context", "memory",
                "llm_chat", "llm_loop"):
        snippet.testability = "smoke_exec"
        return

    snippet.testability = "full_exec"


# ---------------------------------------------------------------------------
# Code transformation
# ---------------------------------------------------------------------------

def _needs_import(code: str, name: str) -> bool:
    """True if `name` is used in code but not already imported."""
    if not re.search(rf'\b{re.escape(name)}\b', code):
        return False
    if re.search(rf'import\s+.*\b{re.escape(name)}\b', code):
        return False
    if re.search(rf'from\s+\S+\s+import.*\b{re.escape(name)}\b', code):
        return False
    return True


def inject_imports(code: str) -> str:
    """Prepend missing LazyBridge/stdlib imports inferred from code usage."""
    lb_core = [n for n in ("LazyAgent","LazyTool","LazySession","LazyContext",
                            "LazyStore","LazyRouter","Memory")
               if _needs_import(code, n)]
    lb_types = [n for n in ("NativeTool","ThinkingConfig")
                if _needs_import(code, n)]

    lines = []
    if lb_core:
        lines.append(f"from lazybridge import {', '.join(lb_core)}")
    if lb_types:
        lines.append(f"from lazybridge.core.types import {', '.join(lb_types)}")
    if _needs_import(code, "BaseModel"):
        lines.append("from pydantic import BaseModel")
    if re.search(r'\bjson\b', code) and "import json" not in code:
        lines.append("import json")
    if re.search(r'\bos\b', code) and "import os" not in code:
        lines.append("import os")
    if re.search(r'\basyncio\b', code) and "import asyncio" not in code:
        lines.append("import asyncio")
    if re.search(r'\bdatetime\b|\bdate\b', code) and "from datetime" not in code and "import datetime" not in code:
        lines.append("from datetime import date, datetime")

    if lines:
        return "\n".join(lines) + "\n\n" + code
    return code


def apply_provider_override(code: str, provider: Optional[str], model: Optional[str]) -> str:
    """
    Conservatively replace provider strings in simple LazyAgent() patterns.
    Only touches patterns like LazyAgent("anthropic") or LazyAgent("openai", ...).
    Never breaks code — if rewrite looks unsafe, returns code unchanged.
    """
    if not provider and not model:
        return code

    def replace_match(m: re.Match) -> str:
        old_prov = m.group(1).strip("\"'")
        if old_prov.lower() not in PROVIDER_STRINGS:
            return m.group(0)
        new_prov = provider if provider else old_prov
        sep = m.group(2)   # either "," or ")"
        if model and "model=" not in m.group(0):
            if sep.strip() == ")":
                return f'LazyAgent("{new_prov}", model="{model}")'
            else:
                return f'LazyAgent("{new_prov}", model="{model}"{sep}'
        return f'LazyAgent("{new_prov}"{sep}'

    # Only match simple one-liner patterns: LazyAgent("prov") or LazyAgent("prov",
    pattern = re.compile(r'LazyAgent\(["\'](\w+)["\']\s*([,)])', re.MULTILINE)
    try:
        return pattern.sub(replace_match, code)
    except Exception:
        return code   # never break code trying to transform it


def prepare_snippet(snippet, provider: Optional[str], model: Optional[str]) -> str:
    """Return final executable code: transform + inject imports."""
    code = snippet.raw_code
    code = apply_provider_override(code, provider, model)
    code = inject_imports(code)
    return code
# ---------------------------------------------------------------------------
# Part 4/5 — Execution engine
# ---------------------------------------------------------------------------

def _check_api_keys(code: str) -> Optional[str]:
    """
    Check if required API keys are present for the given code.
    Returns a human-readable reason string if keys are missing, else None.
    """
    key_map = [
        (r'LazyAgent\s*\(\s*["\'](?:anthropic|claude)["\']', "ANTHROPIC_API_KEY"),
        (r'LazyAgent\s*\(\s*["\'](?:openai|gpt)["\']',       "OPENAI_API_KEY"),
        (r'LazyAgent\s*\(\s*["\'](?:google|gemini)["\']',    "GOOGLE_API_KEY"),
        (r'LazyAgent\s*\(\s*["\']deepseek["\']',              "DEEPSEEK_API_KEY"),
    ]
    missing = []
    for pattern, env_var in key_map:
        if re.search(pattern, code) and not os.environ.get(env_var):
            missing.append(env_var)
    if missing:
        return f"Missing env: {', '.join(missing)}"
    return None


def _triage_failure(stderr: str, stdout: str) -> str:
    """Classify a test failure from stderr/stdout content."""
    combined = (stderr + stdout).lower()
    if any(k in combined for k in ("apikey", "api_key", "authentication",
                                    "invalid_api_key", "unauthorized")):
        return "missing_env"
    if "modulenotfounderror" in combined or "importerror" in combined:
        return "framework_issue"
    if "syntaxerror" in combined:
        return "docs_issue"
    if "nameerror" in combined:
        return "docs_issue"
    if any(k in combined for k in ("attributeerror", "typeerror", "valueerror",
                                    "assertionerror")):
        return "framework_issue"
    if any(k in combined for k in ("connectionerror", "connecttimeout",
                                    "httpx", "requests.exceptions")):
        return "runtime_issue"
    if "timeouterror" in combined or "timed out" in combined:
        return "timeout"
    return "unknown"


def syntax_check(snippet, code: str) -> None:
    """AST-parse prepared code and record pass/syntax_error result."""
    start = time.monotonic()
    try:
        ast.parse(code)
        snippet.status = "pass"
        snippet.reason = "Syntax OK"
        snippet.triage = "unknown"
    except SyntaxError as e:
        snippet.status = "syntax_error"
        snippet.reason = f"SyntaxError line {e.lineno}: {e.msg}"
        snippet.triage = "docs_issue"
    snippet.duration_sec = time.monotonic() - start


def run_snippet(snippet, repo_root: Path, timeout: int,
                provider: Optional[str], model: Optional[str]) -> None:
    """
    Execute a snippet in an isolated subprocess.
    All results are written back to snippet in-place.
    """
    start = time.monotonic()

    # Build final code (transform + inject imports)
    code = prepare_snippet(snippet, provider, model)

    # Check for required API keys on the *transformed* code
    key_issue = _check_api_keys(code)
    if key_issue:
        snippet.status = "skip"
        snippet.reason = key_issue
        snippet.triage = "missing_env"
        snippet.duration_sec = 0.0
        return

    # Write code to an isolated temp directory
    with tempfile.TemporaryDirectory(prefix="lazydoc_") as tmpdir:
        script = Path(tmpdir) / "snippet.py"
        script.write_text(code, encoding="utf-8")

        env = os.environ.copy()
        # Ensure the repo root is importable
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(repo_root) + (os.pathsep + existing_pp if existing_pp else "")
        )

        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(repo_root),
                env=env,
            )
        except subprocess.TimeoutExpired:
            snippet.status = "timeout"
            snippet.reason = f"Exceeded {timeout}s timeout"
            snippet.triage = "timeout"
            snippet.duration_sec = time.monotonic() - start
            return
        except Exception as exc:
            snippet.status = "fail"
            snippet.reason = f"Subprocess error: {exc}"
            snippet.triage = "runtime_issue"
            snippet.duration_sec = time.monotonic() - start
            return

    snippet.duration_sec = time.monotonic() - start
    snippet.stdout = proc.stdout[:4000]
    snippet.stderr = proc.stderr[:4000]

    if proc.returncode == 0:
        snippet.status = "pass"
        snippet.triage = "unknown"
    else:
        snippet.status = "fail"
        snippet.reason = f"Exit {proc.returncode}"
        snippet.triage = _triage_failure(proc.stderr, proc.stdout)


# ---------------------------------------------------------------------------
# Mode dispatch helper
# ---------------------------------------------------------------------------

def should_run(snippet, mode: str) -> bool:
    """Return True if this snippet should be executed (not just syntax-checked)."""
    if snippet.testability in ("skip", "syntax_only"):
        return False
    if mode == "syntax":
        return False
    if mode == "local":
        return snippet.testability == "local_exec"
    if mode == "smoke":
        return snippet.testability in ("local_exec", "smoke_exec")
    if mode == "full":
        return snippet.testability in ("local_exec", "smoke_exec", "full_exec")
    return False
# ---------------------------------------------------------------------------
# Part 5/5 — Reporting + CLI + main
# ---------------------------------------------------------------------------

def dump_snippets(snippets: list, dump_root: Path,
                  provider: Optional[str], model: Optional[str]) -> None:
    """
    Save each snippet as a standalone .py file under dump_root.

    Layout:
        dump_root/
          agents.md/
            00_creating-an-agent.py
            01_chat-single-turn.py
            ...
          sessions.md/
            00_creating-a-session.py
            ...

    Skippable snippets (testability==skip) get a .py.skip extension and
    contain the raw code plus a comment explaining why they were skipped.

    Each file has a header comment with: source, heading, kind, testability.
    """
    dump_root.mkdir(parents=True, exist_ok=True)
    counters: dict[str, int] = {}  # per source-file counter for numbering

    written = skipped_dumped = 0

    for s in snippets:
        # Build sub-directory from source filename (e.g. "agents.md")
        src_file = Path(s.file).name          # "agents.md"
        subdir = dump_root / src_file
        subdir.mkdir(parents=True, exist_ok=True)

        # Per-file sequential index
        key = s.file
        idx = counters.get(key, 0)
        counters[key] = idx + 1

        slug = slugify(s.heading)[:50]        # cap length for filesystem
        stem = f"{idx:02d}_{slug}"

        # Build header comment
        header_lines = [
            f"# Source   : {s.file}",
            f"# Heading  : {s.heading}",
            f"# ID       : {s.id}",
            f"# Kind     : {s.snippet_kind}",
            f"# Testable : {s.testability}",
        ]
        if s.testability == "skip":
            header_lines.append(f"# Skip     : {s.reason}")

        header = "\n".join(header_lines) + "\n\n"

        if s.testability == "skip":
            # Save raw code with .skip extension so it's visible but won't
            # be picked up accidentally by test runners
            out_path = subdir / f"{stem}.py.skip"
            out_path.write_text(header + s.raw_code, encoding="utf-8")
            skipped_dumped += 1
        else:
            # Save prepared code (imports injected, provider overridden)
            out_path = subdir / f"{stem}.py"
            try:
                prepared = prepare_snippet(s, provider, model)
            except Exception:
                prepared = s.raw_code   # fallback: raw
            out_path.write_text(header + prepared, encoding="utf-8")
            written += 1

    print(f"[dump] {written} runnable + {skipped_dumped} skip files → {dump_root}")


def build_json_report(snippets: list, mode: str) -> dict:
    results = []
    for s in snippets:
        results.append({
            "id": s.id,
            "file": s.file,
            "heading": s.heading,
            "snippet_kind": s.snippet_kind,
            "testability": s.testability,
            "mode_run": s.mode_run,
            "status": s.status,
            "reason": s.reason,
            "triage": s.triage,
            "stdout": s.stdout,
            "stderr": s.stderr,
            "duration_sec": round(s.duration_sec, 3),
        })

    def _counts(attr: str) -> dict:
        c: dict[str, int] = {}
        for s in snippets:
            v = getattr(s, attr)
            c[v] = c.get(v, 0) + 1
        return dict(sorted(c.items(), key=lambda x: -x[1]))

    return {
        "mode": mode,
        "total": len(snippets),
        "status_counts": _counts("status"),
        "kind_counts": _counts("snippet_kind"),
        "testability_counts": _counts("testability"),
        "file_counts": {
            s.file: sum(1 for x in snippets if x.file == s.file)
            for s in snippets
        },
        "results": results,
    }


def build_text_summary(snippets: list, mode: str) -> str:
    SEP = "=" * 72
    lines = [SEP, f"LazyBridge Docs Example Validator — mode={mode}", SEP, ""]

    def _counts(attr: str) -> dict:
        c: dict[str, int] = {}
        for s in snippets:
            v = getattr(s, attr)
            c[v] = c.get(v, 0) + 1
        return dict(sorted(c.items(), key=lambda x: -x[1]))

    status_c = _counts("status")
    kind_c   = _counts("snippet_kind")
    file_c: dict[str, int] = {}
    for s in snippets:
        file_c[s.file] = file_c.get(s.file, 0) + 1

    skip_reasons: dict[str, int] = {}
    failures = []
    for s in snippets:
        if s.status == "skip" and s.reason:
            k = s.reason[:80]
            skip_reasons[k] = skip_reasons.get(k, 0) + 1
        if s.status in ("fail", "timeout", "syntax_error"):
            failures.append(s)

    lines.append(f"Total snippets : {len(snippets)}")
    lines.append("")
    lines.append("Status breakdown:")
    for st, n in status_c.items():
        bar = "█" * min(n, 40)
        lines.append(f"  {st:<20} {n:>4}  {bar}")
    lines.append("")

    lines.append("By snippet kind:")
    for k, n in kind_c.items():
        lines.append(f"  {k:<26} {n:>4}")
    lines.append("")

    lines.append("By file:")
    for f, n in sorted(file_c.items()):
        lines.append(f"  {f:<48} {n:>3}")
    lines.append("")

    if skip_reasons:
        lines.append("Skip reasons (top 10):")
        for reason, n in sorted(skip_reasons.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  [{n:>3}x] {reason}")
        lines.append("")

    if failures:
        lines.append(f"Failures ({len(failures)}):")
        for s in failures:
            lines.append(f"  [{s.status.upper():>12}]  {s.id}")
            if s.reason:
                lines.append(f"               reason : {s.reason[:100]}")
            if s.triage and s.triage != "unknown":
                lines.append(f"               triage : {s.triage}")
            if s.stderr:
                last_err = s.stderr.strip().splitlines()[-1][:120]
                if last_err:
                    lines.append(f"               stderr : {last_err}")
        lines.append("")
    else:
        lines.append("No failures. ✓")
        lines.append("")

    lines.append(SEP)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LazyBridge documentation example validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  syntax  — parse + ast.compile only (no execution, no API keys needed)
  local   — execute snippets with no LLM dependency
  smoke   — execute simple LLM snippets (requires API keys)
  full    — attempt all executable snippets

Examples:
  python validate_docs.py --mode syntax
  python validate_docs.py --mode smoke --provider anthropic --model claude-haiku-4-5-20251001
  python validate_docs.py --mode full --timeout 60 --output-dir artifacts
  python validate_docs.py --mode syntax --include-bot-docs
  python validate_docs.py --filter quickstart --mode smoke
""",
    )
    p.add_argument("--docs-root", default="lazy_wiki/human",
                   help="Root directory of markdown docs (default: lazy_wiki/human)")
    p.add_argument("--mode", default="syntax",
                   choices=["syntax", "local", "smoke", "full"],
                   help="Validation mode (default: syntax)")
    p.add_argument("--provider", default=None,
                   choices=["anthropic", "openai", "google", "auto"],
                   help="Override provider in simple LazyAgent() patterns")
    p.add_argument("--model", default=None,
                   help="Inject model= into simple LazyAgent() patterns")
    p.add_argument("--timeout", type=int, default=30,
                   help="Per-snippet timeout in seconds (default: 30)")
    p.add_argument("--output-dir", default="artifacts",
                   help="Output directory for reports (default: artifacts)")
    p.add_argument("--include-bot-docs", action="store_true",
                   help="Also scan lazy_wiki/bot/ markdown files")
    p.add_argument("--filter", default=None,
                   help="Only process snippets whose ID contains this substring")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Cap number of snippets processed (for quick runs)")
    p.add_argument("--dump-dir", default=None, metavar="DIR",
                   help="Dump each snippet as a standalone .py file under DIR "
                        "(e.g. --dump-dir artifacts/examples). "
                        "Files are named <source_file>/<index>_<slug>.py. "
                        "Skipped snippets get a .py.skip extension instead.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).parent.resolve()

    # Resolve 'auto' provider before anything else
    provider = args.provider
    if provider == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("GOOGLE_API_KEY"):
            provider = "google"
        else:
            print("WARNING: --provider auto but no API keys found. "
                  "LLM snippets will be skipped.", file=sys.stderr)
            provider = None

    # Build list of directories to scan
    scan_roots = []
    docs_root = repo_root / args.docs_root
    if docs_root.exists():
        scan_roots.append(docs_root)
    else:
        print(f"ERROR: --docs-root not found: {docs_root}", file=sys.stderr)
        return 1
    if args.include_bot_docs:
        bot_root = repo_root / "lazy_wiki" / "bot"
        if bot_root.exists():
            scan_roots.append(bot_root)
        else:
            print(f"WARNING: bot docs not found: {bot_root}", file=sys.stderr)

    # Collect markdown files
    md_files: list[Path] = []
    for root in scan_roots:
        md_files.extend(sorted(root.rglob("*.md")))
    if not md_files:
        print("ERROR: No markdown files found.", file=sys.stderr)
        return 1
    print(f"[scan] {len(md_files)} markdown files in "
          f"{[str(r.relative_to(repo_root)) for r in scan_roots]}")

    # Extract snippets
    all_snippets = []
    for md_file in md_files:
        try:
            all_snippets.extend(extract_snippets(md_file, repo_root))
        except Exception as exc:
            print(f"WARNING: failed to parse {md_file.name}: {exc}", file=sys.stderr)
    print(f"[extract] {len(all_snippets)} Python code blocks found")

    # Apply filter / cap
    if args.filter:
        all_snippets = [s for s in all_snippets if args.filter in s.id]
        print(f"[filter] {len(all_snippets)} snippets match '{args.filter}'")
    if args.max_examples is not None:
        all_snippets = all_snippets[: args.max_examples]
        print(f"[cap] limited to {len(all_snippets)} snippets (--max-examples)")

    if not all_snippets:
        print("Nothing to process.")
        return 0

    # Classify
    for s in all_snippets:
        classify_snippet(s)

    # Dump snippets to individual .py files if requested
    if args.dump_dir:
        dump_snippets(all_snippets, Path(args.dump_dir), provider, args.model)

    # Process each snippet
    n = len(all_snippets)
    passed = failed = skipped = 0

    for i, s in enumerate(all_snippets, 1):
        w = len(str(n))
        prefix = f"[{i:>{w}}/{n}]"

        # Pre-skipped by classifier
        if s.testability == "skip":
            s.mode_run = args.mode
            s.status = "skip"
            skipped += 1
            print(f"{prefix} SKIP      {s.id}")
            print(f"           {s.reason[:80]}")
            continue

        executing = should_run(s, args.mode)

        if executing:
            s.mode_run = args.mode
            run_snippet(s, repo_root, args.timeout, provider, args.model)
        else:
            # Syntax check only
            s.mode_run = "syntax"
            code = prepare_snippet(s, provider, args.model)
            syntax_check(s, code)

        if s.status == "pass":
            passed += 1
            verb = "EXEC-PASS " if executing else "SYNTAX-OK "
            print(f"{prefix} {verb} {s.id}  ({s.duration_sec:.2f}s)")
        elif s.status == "skip":
            skipped += 1
            print(f"{prefix} SKIP      {s.id}")
            if s.reason:
                print(f"           {s.reason[:80]}")
        elif s.status in ("syntax_error", "fail", "timeout"):
            failed += 1
            verb = {
                "syntax_error": "SYNTAX-ERR",
                "fail":         "FAIL      ",
                "timeout":      "TIMEOUT   ",
            }[s.status]
            print(f"{prefix} {verb} {s.id}  ({s.duration_sec:.2f}s)")
            if s.reason:
                print(f"           reason: {s.reason[:100]}")
            if s.stderr:
                last = s.stderr.strip().splitlines()[-1]
                print(f"           stderr: {last[:120]}")

    # Write reports
    out_dir = repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "docs_example_report.json"
    txt_path  = out_dir / "docs_example_summary.txt"

    report  = build_json_report(all_snippets, args.mode)
    summary = build_text_summary(all_snippets, args.mode)

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_path.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"JSON report : {json_path}")
    print(f"Text summary: {txt_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
