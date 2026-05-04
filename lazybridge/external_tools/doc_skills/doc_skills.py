"""
lazybridge.external_tools.doc_skills  —  Local documentation skill runtime
=================================================================

Index local documentation folders into a portable skill bundle, then expose
the bundle as a LazyBridge tool or pipeline that any agent can call.

Retrieval uses full BM25 (Robertson IDF, k1=1.5, b=0.75) with IDF weights
computed once at index time and stored in vocab.json.  This is substantially
more accurate than TF-only matching, especially for technical documentation
where rare terms (class names, parameter names, error codes) matter most.

Chunking is heading-aware for Markdown/RST/AsciiDoc: each section becomes its
own chunk, preserving semantic boundaries.  Large sections are sub-split by
character count; tiny adjacent sections are merged.  Non-markdown files use
character-based splitting with paragraph-boundary snapping.

Public API
----------
    build_skill(source_dirs, skill_name, ...)  → dict        index docs → skill bundle on disk
    query_skill(skill_dir, task, ...)          → str         retrieve + grounded context brief
    skill_tools(*, skill_dir, ...)             → list[Tool]  one-step tool for an agent
    skill_builder_tools(*, ...)                → list[Tool]  tool that builds skill bundles
    skill_pipeline(*, skill_dir, ...)          → Tool        router + executor chain pipeline

Quick start
-----------
    from lazybridge.external_tools.doc_skills import build_skill, skill_tools
    from lazybridge import Agent

    meta = build_skill(["./docs", "./reference"], "my-project")
    tools = skill_tools(skill_dir=meta["skill_dir"])
    resp = Agent("anthropic", tools=tools)("How does X work?")
    print(resp.text())

Add  generated_skills/  to your .gitignore — build_skill() writes there by default.
No extra dependencies required beyond the standard library.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from lazybridge import Agent, Session, Tool

__all__ = [
    "build_skill",
    "query_skill",
    "skill_builder_tools",
    "skill_pipeline",
    "skill_tools",
]


# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_EXTENSIONS: tuple[str, ...] = (
    ".md",
    ".mdx",
    ".txt",
    ".rst",
    ".adoc",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
)

HEADING_EXTENSIONS: frozenset[str] = frozenset({".md", ".mdx", ".rst", ".adoc"})

BM25_K1: float = 1.5
BM25_B: float = 0.75

STOPWORDS: frozenset[str] = frozenset(
    {
        # English
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "for",
        "in",
        "on",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "as",
        "that",
        "this",
        "it",
        "its",
        "at",
        "into",
        "how",
        "what",
        "why",
        "when",
        "which",
        "who",
        "can",
        "could",
        "should",
        "would",
        "use",
        "using",
        "used",
        "about",
        "than",
        "then",
        "them",
        "they",
        "their",
        "there",
        # Italian
        "de",
        "del",
        "della",
        "di",
        "e",
        "il",
        "la",
        "lo",
        "gli",
        "le",
        "un",
        "una",
        "uno",
        "che",
        "come",
        "per",
        "con",
        "su",
        "da",
        "nel",
        "nella",
        "delle",
        "dei",
        "degli",
    }
)

MODE = Literal["answer", "extract", "locate", "summarize"]


# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass(slots=True)
class DocChunk:
    path: str
    title: str  # document-level title (first H1 or filename)
    heading: str  # nearest section heading — used for BM25 title boost
    text: str
    tokens: list[str]
    doc_len: int  # token count — needed for BM25 length normalisation
    ordinal: int


@dataclass(slots=True)
class SkillManifest:
    name: str
    description: str
    source_dirs: list[str]
    indexed_files: list[str]
    total_chunks: int
    avgdl: float
    extensions: list[str]
    version: str = "3.0.0"
    created_by: str = "lazybridge.external_tools.doc_skills"


# ── Text utilities ─────────────────────────────────────────────────────────────


def _slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return re.sub(r"-+", "-", value).strip("-") or "skill"


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-zA-Z0-9_.]{2,}", text.lower())
    return [t for t in toks if t not in STOPWORDS]


def _doc_title(path: Path, text: str) -> str:
    for line in text.splitlines():
        s = line.strip().lstrip("#").strip()
        if s:
            return s
    return path.stem.replace("_", " ").replace("-", " ").strip() or path.name


def _trim(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


# ── Chunking ───────────────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


def _char_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    out, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        window = text[start:end]
        if end < n:
            split = window.rfind("\n\n")
            if split >= chunk_size // 2:
                window = window[:split]
                end = start + split
        if window.strip():
            out.append(window.strip())
        if end >= n:
            break
        start = max(start + 1, end - overlap)
    return out


def _heading_chunks(text: str, max_chunk: int, overlap: int) -> list[tuple[str, str]]:
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", chunk) for chunk in _char_chunks(text, max_chunk, overlap)]

    raw: list[tuple[str, str]] = []
    if matches[0].start() > 50:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            raw.append(("", preamble))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw.append((m.group(2).strip(), text[m.start() : end].strip()))

    result: list[tuple[str, str]] = []
    min_size: int = max_chunk // 4

    for heading, body in raw:
        if len(body) > max_chunk:
            for j, sub in enumerate(_char_chunks(body, max_chunk, overlap)):
                result.append((f"{heading} [{j + 1}]" if j else heading, sub))
        elif result and len(body) < min_size:
            prev_h, prev_b = result[-1]
            result[-1] = (prev_h, prev_b + "\n\n" + body)
        else:
            result.append((heading, body))

    return result


def _make_chunks(path: Path, text: str, chunk_size: int, overlap: int) -> list[DocChunk]:
    text = _normalize(text)
    if not text:
        return []

    doc_title = _doc_title(path, text)
    pairs: list[tuple[str, str]] = (
        _heading_chunks(text, chunk_size, overlap)
        if path.suffix.lower() in HEADING_EXTENSIONS
        else [("", c) for c in _char_chunks(text, chunk_size, overlap)]
    )

    results = []
    for i, (heading, body) in enumerate(pairs):
        stripped = body.strip()
        if not stripped:
            continue
        toks = _tokenize(stripped)
        results.append(
            DocChunk(
                path=str(path),
                title=doc_title,
                heading=heading or doc_title,
                text=stripped,
                tokens=toks,
                doc_len=len(toks),
                ordinal=i,
            )
        )
    return results


# ── BM25 ───────────────────────────────────────────────────────────────────────


def _build_idf(chunks: list[DocChunk]) -> dict[str, float]:
    """Robertson IDF:  log( (N - df + 0.5) / (df + 0.5) + 1 )"""
    N = len(chunks)
    df: Counter[str] = Counter()
    for chunk in chunks:
        for tok in set(chunk.tokens):
            df[tok] += 1
    return {tok: math.log((N - freq + 0.5) / (freq + 0.5) + 1) for tok, freq in df.items()}


def _bm25(chunk: DocChunk, q_tokens: list[str], idf: dict[str, float], avgdl: float) -> float:
    if not q_tokens:
        return 0.0
    tf_map = Counter(chunk.tokens)
    dl, score = chunk.doc_len or 1, 0.0
    for tok in q_tokens:
        tf = tf_map.get(tok, 0)
        if not tf:
            continue
        score += idf.get(tok, 0.5) * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl))
    heading_lower = chunk.heading.lower()
    for tok in q_tokens:
        if tok in heading_lower:
            score += 1.2
    phrase = " ".join(q_tokens[:6])
    if phrase and phrase in chunk.text.lower():
        score += 4.0
    return score


# ── File iteration ─────────────────────────────────────────────────────────────


def _iter_docs(roots: Sequence[Path], include_exts: Sequence[str]) -> Iterable[Path]:
    extset: set[str] = {e.lower() for e in include_exts}
    seen: set[Path] = set()
    for root in roots:
        for path in sorted(root.rglob("*")):
            # Skip symlinks outright.  Following symlinks can
            # create loops that hang the indexer and lets a symlink inside
            # the indexed directory silently widen the read surface to
            # unrelated files on disk. Callers that want symlinked content
            # indexed should pass the resolved target as an additional
            # root explicitly.
            if path.is_symlink():
                continue
            if not path.is_file() or path.name.startswith("."):
                continue
            if path.suffix.lower() not in extset:
                continue
            rp = path.resolve()
            if rp not in seen:
                seen.add(rp)
                yield path


# ── Bundle I/O ─────────────────────────────────────────────────────────────────


def _load_manifest(skill_dir: Path) -> SkillManifest:
    return SkillManifest(**json.loads((skill_dir / "manifest.json").read_text(encoding="utf-8")))


def _load_chunks(skill_dir: Path) -> list[DocChunk]:
    with (skill_dir / "chunks.jsonl").open(encoding="utf-8") as f:
        return [DocChunk(**json.loads(line)) for line in f]


def _load_idf(skill_dir: Path) -> dict[str, float]:
    p = skill_dir / "vocab.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _load_skill_md(skill_dir: Path) -> str:
    p = skill_dir / "SKILL.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


# ── SKILL.md ───────────────────────────────────────────────────────────────────


def _render_skill_md(
    *, name: str, description: str, usage_notes: str, file_count: int, total_chunks: int, source_dirs: Sequence[Path]
) -> str:
    roots = "\n".join(f"- {p}" for p in source_dirs)
    return f"""---
name: {name}
description: {description}
retrieval: BM25 (Robertson IDF, k1={BM25_K1}, b={BM25_B})
---

# {name}

## Purpose
{description}

## Scope
Grounded only in the documentation indexed from:
{roots}

## Operating rules
1. Classify the task intent before answering (answer / extract / locate / summarize).
2. Retrieve only the most relevant chunks — do not pad with peripheral material.
3. Answer exclusively from retrieved evidence.
4. Include source file paths in every answer.
5. If evidence is absent or weak, say so explicitly.
6. Do not invent APIs, parameters, classes, or behaviour not present in the indexed docs.

## Retrieval
- Algorithm : BM25 (Robertson IDF, k1={BM25_K1}, b={BM25_B})
- Chunking  : heading-aware for Markdown/RST, character-based fallback
- Indexed files  : {file_count}
- Indexed chunks : {total_chunks}

## Task modes
| Mode      | When to use                              | Output shape                      |
|-----------|------------------------------------------|-----------------------------------|
| answer    | Default — synthesise an answer           | Evidence bullets + source paths   |
| extract   | Need raw quotes from the docs            | Direct excerpts grouped by source |
| locate    | Just need the relevant file paths        | Ranked file list                  |
| summarize | High-level overview of a topic           | Compact bullets from top chunks   |

## Additional notes
{usage_notes.strip() or "No additional notes."}
"""


# ── Context brief ──────────────────────────────────────────────────────────────


def _build_brief(*, manifest: SkillManifest, skill_md: str, task: str, mode: MODE, selected: list[DocChunk]) -> str:
    sources = list(dict.fromkeys(c.path for c in selected))
    excerpts = [f"### {c.heading}  [{Path(c.path).name}]\n{_trim(c.text, 800)}" for c in selected[:6]]
    return "\n".join(
        [
            f"[skill]   {manifest.name}",
            f"[task]    {task}",
            f"[mode]    {mode}",
            "",
            "[skill_instructions]",
            _trim(skill_md, 2500) if skill_md else "No SKILL.md present.",
            "",
            "[sources_retrieved]",
            *[f"  • {p}" for p in sources],
            "",
            "[excerpts]",
            *excerpts,
            "",
            "[execution_policy]",
            "  1. Use only the evidence above.",
            "  2. Do not infer behaviour not shown in the excerpts.",
            "  3. Cite source file names in every answer.",
            "  4. If evidence is partial or missing, say so explicitly.",
        ]
    )


def _auto_mode(task: str) -> MODE:
    t = task.lower()
    if any(x in t for x in ("where", "find", "locate", "which file", "in quale", "dove", "trova")):
        return "locate"
    if any(x in t for x in ("extract", "list all", "enumerate", "all examples", "estrai", "elenca")):
        return "extract"
    if any(x in t for x in ("summarize", "summarise", "overview", "brief", "riassumi", "panoramica")):
        return "summarize"
    return "answer"


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════


def build_skill(
    source_dirs: Annotated[list[str], "One or more folders containing documentation to index."],
    skill_name: Annotated[str, "Skill name — used as the bundle folder name and title."],
    output_root: Annotated[str, "Parent directory for the generated bundle."] = "./generated_skills",
    description: Annotated[str, "What this skill covers (used in SKILL.md and tool description)."] = "",
    usage_notes: Annotated[str, "Extra operational rules appended to SKILL.md."] = "",
    include_extensions: Annotated[list[str], "File extensions to index."] = list(DEFAULT_EXTENSIONS),  # noqa: B006
    chunk_size: Annotated[int, "Maximum characters per chunk."] = 1800,
    chunk_overlap: Annotated[int, "Overlap between char-mode chunks."] = 180,
    copy_sources: Annotated[bool, "Copy original docs into the bundle under sources/."] = False,
    overwrite: Annotated[bool, "Replace an existing bundle with the same name."] = True,
    max_chars_per_file: Annotated[int, "Safety cap on characters read per file."] = 200_000,
) -> dict[str, Any]:
    """
    Index documentation folders and write a portable skill bundle to disk.

    The bundle contains SKILL.md (LLM instructions), manifest.json (metadata +
    avgdl for BM25), vocab.json (Robertson IDF weights), and chunks.jsonl.
    Returns a metadata dict: skill_dir, indexed_files, total_chunks, avgdl.
    """
    roots = [Path(p).expanduser().resolve() for p in source_dirs]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Not a directory: {root}")

    skill_dir = Path(output_root).expanduser().resolve() / _slugify(skill_name)
    if skill_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Skill already exists: {skill_dir}")
        shutil.rmtree(skill_dir)
    skill_dir.mkdir(parents=True)
    if copy_sources:
        (skill_dir / "sources").mkdir()

    indexed_files: list[str] = []
    all_chunks: list[DocChunk] = []

    for root in roots:
        for path in _iter_docs([root], include_extensions):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")[:max_chars_per_file]
            except Exception:
                continue
            chunks = _make_chunks(path, text, chunk_size, chunk_overlap)
            if not chunks:
                continue
            indexed_files.append(str(path))
            all_chunks.extend(chunks)
            if copy_sources:
                dest = skill_dir / "sources" / root.name / path.parent.relative_to(root)
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest / path.name)

    if not all_chunks:
        raise ValueError("No indexable documentation found in the provided folders.")

    description = description or f"Documentation skill built from {len(indexed_files)} files."
    avgdl = sum(c.doc_len for c in all_chunks) / len(all_chunks)
    idf = _build_idf(all_chunks)

    manifest = SkillManifest(
        name=skill_name,
        description=description,
        source_dirs=[str(p) for p in roots],
        indexed_files=indexed_files,
        total_chunks=len(all_chunks),
        avgdl=avgdl,
        extensions=list(include_extensions),
    )
    skill_md = _render_skill_md(
        name=skill_name,
        description=description,
        usage_notes=usage_notes,
        file_count=len(indexed_files),
        total_chunks=len(all_chunks),
        source_dirs=roots,
    )

    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
    (skill_dir / "manifest.json").write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (skill_dir / "vocab.json").write_text(json.dumps(idf, ensure_ascii=False, indent=2), encoding="utf-8")
    with (skill_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    return {
        "skill_dir": str(skill_dir),
        "skill_name": skill_name,
        "description": description,
        "indexed_files": indexed_files,
        "total_chunks": len(all_chunks),
        "avgdl": round(avgdl, 1),
    }


def query_skill(
    skill_dir: Annotated[str, "Path to a skill bundle created by build_skill()."],
    task: Annotated[str, "Question or task to answer from the indexed documentation."],
    mode: Annotated[
        Literal["auto", "answer", "extract", "locate", "summarize"],
        "Execution mode. 'auto' detects intent from the task wording.",
    ] = "auto",
    top_k: Annotated[int, "Number of chunks to retrieve."] = 8,
    max_chars: Annotated[int, "Maximum characters in the returned context brief."] = 10_000,
    include_quotes: Annotated[bool, "Append full excerpts after evidence bullets."] = True,
) -> str:
    """
    Retrieve the most relevant chunks via BM25 and return a grounded context
    brief ready to be injected into an LLM's context window.
    """
    sdir = Path(skill_dir).expanduser().resolve()
    if not sdir.exists():
        raise FileNotFoundError(f"Skill directory not found: {sdir}")

    manifest = _load_manifest(sdir)
    chunks = _load_chunks(sdir)
    idf = _load_idf(sdir)
    skill_md = _load_skill_md(sdir)
    resolved_mode: MODE = _auto_mode(task) if mode == "auto" else mode  # type: ignore[assignment]

    q_tokens = _tokenize(task)
    ranked = sorted(((c, _bm25(c, q_tokens, idf, manifest.avgdl)) for c in chunks), key=lambda x: x[1], reverse=True)
    selected = [c for c, score in ranked[: max(1, top_k)] if score > 0]

    if not selected:
        return (
            f"[skill] {manifest.name}\n[task] {task}\n\n"
            "No relevant documentation was retrieved for this task. "
            "Do not answer beyond the indexed evidence."
        )[:max_chars]

    brief = _build_brief(manifest=manifest, skill_md=skill_md, task=task, mode=resolved_mode, selected=selected)
    result_lines: list[str] = []

    if resolved_mode == "locate":
        result_lines = ["Relevant files:"] + [f"  • {p}" for p in dict.fromkeys(c.path for c in selected)]
    elif resolved_mode == "extract":
        result_lines = ["Excerpts:"]
        for c in selected:
            result_lines.append(f"\n### {c.heading}  [{Path(c.path).name}]\n{_trim(c.text, 1000)}")
    elif resolved_mode == "summarize":
        result_lines = ["Summary:"]
        for c in selected[:6]:
            condensed = re.sub(r"\s+", " ", c.text)
            result_lines.append(f"  - {_trim(condensed, 300)}  [{Path(c.path).name}]")
    else:
        result_lines = ["Best evidence:"]
        for c in selected[:5]:
            condensed = re.sub(r"\s+", " ", c.text)
            result_lines.append(f"  - {_trim(condensed, 400)}  [{Path(c.path).name}]")
        if include_quotes:
            result_lines.append("\nFull excerpts:")
            for c in selected[:3]:
                result_lines.append(f"\n### {c.heading}  [{Path(c.path).name}]\n{_trim(c.text, 800)}")

    return (brief + "\n\n[result]\n" + "\n".join(result_lines))[:max_chars]


def skill_tools(
    *,
    skill_dir: Annotated[str, "Path to a skill bundle created by build_skill()."],
    name: Annotated[str | None, "Tool name exposed to the agent."] = None,
    description: Annotated[str | None, "Tool description."] = None,
    strict: Annotated[bool, "Strict JSON schema validation."] = False,
) -> list[Tool]:
    """Return a single-element list containing a query_skill() Tool ready
    to be passed to any agent or pipeline."""
    sdir = Path(skill_dir).expanduser().resolve()
    manifest = _load_manifest(sdir)

    def _run(
        task: Annotated[str, "Question or task to answer from this skill."],
        mode: Annotated[Literal["auto", "answer", "extract", "locate", "summarize"], "Retrieval mode."] = "auto",
        top_k: Annotated[int, "Number of chunks to retrieve."] = 8,
        include_quotes: Annotated[bool, "Include full excerpts."] = True,
    ) -> str:
        """Query a local documentation skill and return a grounded context brief.

        Use when the task is about the documentation indexed by this skill;
        treat the result as grounded evidence and answer only from it.
        """
        return query_skill(str(sdir), task, mode=mode, top_k=top_k, include_quotes=include_quotes)

    return [
        Tool(
            _run,
            name=name or _slugify(manifest.name),
            description=description or manifest.description,
            strict=strict,
        )
    ]


def skill_builder_tools(
    *,
    name: Annotated[str, "Tool name."] = "build_doc_skill",
    description: Annotated[str, "Tool description."] = (
        "Index documentation folders into a reusable local skill bundle. "
        "Call this to transform one or more documentation folders into a queryable local skill."
    ),
    strict: Annotated[bool, "Strict JSON schema validation."] = False,
) -> list[Tool]:
    """Return a single-element list containing a Tool that builds skill bundles."""
    return [Tool(build_skill, name=name, description=description, strict=strict)]


def skill_pipeline(
    *,
    skill_dir: Annotated[str, "Path to a skill bundle."],
    provider: Annotated[str | Any, "LazyBridge provider alias or instance."] = "anthropic",
    router_model: Annotated[str | None, "Model for the task-sharpening router."] = None,
    executor_model: Annotated[str | None, "Model for the grounded-answer executor."] = None,
    session: Annotated[Any, "Optional Session. Created if omitted."] = None,
    native_tools: Annotated[list | None, "Provider-native tools for the executor."] = None,
) -> Tool:
    """
    Two-step pipeline exposed as a single Tool.

      1. Router   — rewrites the user task into a retrieval-optimised query.
      2. Executor — calls skill_tools() and synthesises a grounded answer.

    Returns an Agent.chain(router, executor).as_tool().
    """
    from lazybridge.engines.llm import LLMEngine

    sdir = Path(skill_dir).expanduser().resolve()
    manifest = _load_manifest(sdir)
    sess = session or Session()
    s_tool = skill_tools(skill_dir=str(sdir))[0]

    # Resolve model strings: use the explicit model override if given, else the provider alias.
    router_model_str = router_model or (provider if isinstance(provider, str) else "anthropic")
    executor_model_str = executor_model or (provider if isinstance(provider, str) else "anthropic")

    router = Agent(
        engine=LLMEngine(
            router_model_str,
            system=(
                "You sharpen user queries for a local documentation retrieval system. "
                "Return a single concise retrieval query. "
                "Preserve every technical identifier: class names, method names, "
                "parameter names, error codes, configuration keys. "
                "Do not answer the question. Do not add facts."
            ),
        ),
        name="skill_router",
        session=sess,
    )
    executor = Agent(
        engine=LLMEngine(
            executor_model_str,
            system=(
                "You answer from the local skill tool only. Always call the skill tool first. "
                "Build your answer exclusively from the tool result. "
                "Name every source file you use. "
                "If the skill returns weak or absent evidence, say so explicitly."
            ),
            native_tools=native_tools,
        ),
        name="skill_executor",
        session=sess,
        tools=[s_tool],
    )

    pipeline = Agent.chain(router, executor, name="doc_skill_pipeline", session=sess)
    return pipeline.as_tool(
        "doc_skill_pipeline",
        description=(
            f"Grounded local-docs pipeline: {manifest.description}. "
            "Use for questions grounded in the indexed documentation. "
            "The pipeline sharpens the query then retrieves and synthesises the answer."
        ),
    )
