"""Daily Global News Report — configurable parallel pipeline.

Usage
-----
    python examples/daily_news_report.py                      # 5 stories, standard depth
    python examples/daily_news_report.py --stories 7 --depth deep
    python examples/daily_news_report.py --stories 3 --depth brief

Architecture
------------

build_pipeline(n_stories, depth) constructs the full graph at call time:

  Outer Plan  (store=SQLite)

    Step(us_pipeline, parallel) ─┐
    Step(cn_pipeline, parallel)  ├─ from_parallel_all → final_orchestrator → HTML
    Step(in_pipeline, parallel) ─┘

  Each region_pipeline is Agent(engine=Plan(...)):

    Step(discovery_agent)                          ← cheap, web search: top N headlines
    Step(article_writer_1, parallel) ─┐
    Step(article_writer_2, parallel)  ├─ from_parallel_all → region_assembler
    ...                               │
    Step(article_writer_N, parallel) ─┘

  article_writer (one per story, all run in parallel):
    - Extracts its assigned story from discovery output
    - 2-5 targeted web searches for depth, stats, expert context
    - fetch_image (article URL) → search_wikimedia_image fallback
    - Markdown table if story has meaningful statistics
    - Returns a complete professional article

  region_assembler (DeepSeek medium):
    - Combines N parallel articles into clean regional MD

  final_orchestrator (DeepSeek medium):
    - Parses local Image: paths → charts list
    - Saves global MD → generates HTML

Depth presets
-------------
  brief    →  4–5 paragraphs, 1 image, table if obvious
  standard →  6–8 paragraphs, 1 image, table if data exists
  deep     →  10–14 paragraphs, 2 images, table always

Requires: ANTHROPIC_API_KEY, OPENAI_API_KEY (discovery only), GOOGLE_API_KEY,
          DEEPSEEK_API_KEY
          pip install "lazybridge[anthropic,openai,google,deepseek,report]"
"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Annotated

from lazybridge import Agent, LLMEngine, NativeTool, Plan, Session, Step, Store, Tool, from_parallel_all
from lazybridge.external_tools.report_builder import report_tools

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TODAY      = date.today().isoformat()
OUTPUT_DIR = Path("./news_reports")
IMAGES_DIR = OUTPUT_DIR / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
session = Session(console=True)
store   = Store(db=str(OUTPUT_DIR / "pipeline.sqlite"))

# Each region uses a different provider + model for article writers to spread
# token-per-minute rate limits across multiple orgs.
# Google is excluded from article writers: mixing native_tools + custom tools
# requires tool_config.include_server_side_tool_invocations, which the current
# Google AI SDK version does not support. Discovery-only agents (no custom
# tools) can still use Google.
REGION_ARTICLE_PROVIDER = {
    "United States": "anthropic",
    "China":         "openai",
    "India":         "openai",    # was "google" — Google can't mix tools
}

REGION_ARTICLE_MODEL = {
    "United States": "cheap",
    "China":         "cheap",
    "India":         "o4-mini",   # literal model name
}

# Semaphore: max concurrent Wikimedia API calls (Commons throttles aggressively)
_WIKIMEDIA_SEM = threading.Semaphore(3)

DEPTH_CONFIG = {
    "brief": {
        "paragraphs": "4–5",
        "searches":   "2",
        "images":     "1",
        "table":      "only if the story is explicitly about numbers (economic data, election results, etc.)",
    },
    "standard": {
        "paragraphs": "6–8",
        "searches":   "3",
        "images":     "1",
        "table":      "include whenever meaningful statistics or comparisons exist",
    },
    "deep": {
        "paragraphs": "10–14",
        "searches":   "4–5",
        "images":     "2",
        "table":      "always research and include — if no hard data, use a timeline or key-facts table",
    },
}

REGIONS = [
    ("United States", "🇺🇸", "us"),
    ("China",         "🇨🇳", "cn"),
    ("India",         "🇮🇳", "in"),
]

# ---------------------------------------------------------------------------
# Shared image tools
# ---------------------------------------------------------------------------


def fetch_image(
    url: Annotated[str, "Direct URL of the image (JPG or PNG)"],
    filename: Annotated[str, "Local filename, e.g. 'us_iran.jpg'"],
) -> dict:
    """Download an image from a URL, save to images dir, return the local path."""
    dest = (IMAGES_DIR / Path(filename).name).resolve()
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (compatible; LazyBridge-NewsBot/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            dest.write_bytes(resp.read())
        return {"saved": True, "path": str(dest)}
    except (urllib.error.URLError, OSError) as exc:
        return {"error": True, "message": str(exc)}


def search_wikimedia_image(
    query: Annotated[str, "Search query, e.g. 'Iran nuclear talks 2026'"],
    filename: Annotated[str, "Local filename, e.g. 'us_iran.jpg'"],
) -> dict:
    """Search Wikimedia Commons for a freely-licensed image and download it.

    Uses a shared semaphore (max 3 concurrent) and exponential backoff on 429.
    """
    dest = (IMAGES_DIR / Path(filename).name).resolve()
    search_url = (
        "https://en.wikipedia.org/w/api.php?action=query&format=json"
        f"&generator=search&gsrsearch={urllib.request.quote(query)}&gsrnamespace=6"
        "&prop=imageinfo&iiprop=url|mime&gsrlimit=8"
    )
    headers = {"User-Agent": "LazyBridge-NewsBot/1.0 (educational; github.com/selvaz/LazyBridge)"}

    with _WIKIMEDIA_SEM:
        # Jitter to spread burst traffic
        time.sleep(random.uniform(0.3, 1.0))

        for attempt in range(4):
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(search_url, headers=headers), timeout=12
                ) as r:
                    pages = json.loads(r.read()).get("query", {}).get("pages", {})
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < 3:
                    time.sleep(2 ** attempt + random.uniform(0, 1))
                    continue
                return {"error": True, "message": str(exc)}
            except (urllib.error.URLError, OSError) as exc:
                return {"error": True, "message": str(exc)}
        else:
            return {"error": True, "message": "Wikimedia search: too many retries"}

        image_url = next(
            (
                p["imageinfo"][0]["url"]
                for p in pages.values()
                if p.get("imageinfo")
                and p["imageinfo"][0].get("mime") in ("image/jpeg", "image/png")
            ),
            None,
        )
        if not image_url:
            return {"error": True, "message": "No suitable image found on Wikimedia"}

        for attempt in range(3):
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(image_url, headers=headers), timeout=14
                ) as r:
                    dest.write_bytes(r.read())
                return {"saved": True, "path": str(dest), "source": "wikimedia"}
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < 2:
                    time.sleep(2 ** attempt + random.uniform(0, 1))
                    continue
                return {"error": True, "message": str(exc)}
            except (urllib.error.URLError, OSError) as exc:
                return {"error": True, "message": str(exc)}

        return {"error": True, "message": "Image download: too many retries"}


fetch_image_tool     = Tool(fetch_image,            name="fetch_image")
wikimedia_image_tool = Tool(search_wikimedia_image, name="search_wikimedia_image")

# ---------------------------------------------------------------------------
# save_markdown tool
# ---------------------------------------------------------------------------


def save_markdown(
    content: Annotated[str, "Full Markdown content to write"],
    filename: Annotated[str, "File name"] = "daily_news.md",
) -> dict:
    """Write a Markdown string to the reports directory and return the path."""
    import re as _re
    # Strip markdown link syntax that models sometimes inject: [text](url) → text
    filename = _re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", filename)
    path = (OUTPUT_DIR / Path(filename).name).resolve()
    path.write_text(content, encoding="utf-8")
    return {"saved": True, "path": str(path)}


save_md_tool = Tool(save_markdown, name="save_markdown")

# ---------------------------------------------------------------------------
# Discovery agent  — cheap, finds top N headlines with brief context
# ---------------------------------------------------------------------------

_DISCOVERY_SYSTEM = """\
You are a news discovery agent for the {region} region.
Search for today's ({today}) most important breaking and developing stories.

Return a NUMBERED LIST of exactly {n_stories} stories — nothing else:

1. <Headline>
   Context: <2–3 sentences: what happened, who is involved, why it matters today>
   Source URL: <article URL if available>
   Image URL: <direct image URL from the article if visible in search results, else blank>

2. <Headline>
   ...

Order by newsworthiness. No preamble, no sign-off.
"""


_DISCOVERY_PROVIDER = {
    "United States": "anthropic",
    "China":         "google",     # Google search quality for CN news
    "India":         "openai",     # spreads discovery load off anthropic
}


def _discovery_agent(region: str, n_stories: int) -> Agent:
    provider = _DISCOVERY_PROVIDER.get(region, "anthropic")
    return Agent(
        engine=LLMEngine(
            "cheap",
            provider=provider,
            system=_DISCOVERY_SYSTEM.format(region=region, n_stories=n_stories, today=TODAY),
            native_tools=[NativeTool.WEB_SEARCH],
        ),
        name=f"{region.lower().split()[0]}_discovery",
        session=session,
    )


# ---------------------------------------------------------------------------
# Article writer  — one per story, runs in parallel, does deep research
# ---------------------------------------------------------------------------

_ARTICLE_WRITER_SYSTEM = """\
You are a professional journalist covering the {region} region for a global news digest.
Today is {today}.

You receive a numbered list of today's top stories. Your assignment: write the complete \
article for story #{index}.

━━ RESEARCH ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Extract story #{index} from the input list.
2. Perform {searches} targeted web searches to gather:
   - Specific facts, figures, dates, names
   - Expert reactions or official statements
   - Historical context and background
   - Latest developments (what happened in the last 24h)
   - Statistics or data relevant to the story

━━ IMAGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Acquire {images} image(s) per this priority order:
  a) If the discovery entry had a non-blank Image URL → call fetch_image(url, filename)
  b) If fetch_image fails or URL was blank → call search_wikimedia_image(query, filename)
     where query is a concise keyword phrase about the story topic
  c) For "deep" depth, repeat for a second image on a related subtopic
  d) If all attempts fail → no image (do not invent paths)

━━ ARTICLE FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write {paragraphs} paragraphs of professional journalism. Then optionally a table. \
Then Image lines. Use this EXACT format:

## <Headline>

<Lead paragraph: the most important fact, written for a general audience.>

<Body paragraphs: background, context, key figures, expert quotes, timeline, implications.>

<Closing paragraph: what happens next, outstanding questions, significance.>

{table_instruction}

Image: <local path returned by fetch_image or search_wikimedia_image>
[If 2 images: add a second Image: line]

━━ RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- All facts must come from your web searches — do NOT invent statistics or quotes
- Image paths must be local paths returned by tools — do NOT invent or guess paths
- No preamble ("Here is the article..."), no sign-off, return only the article block
"""

_TABLE_INSTRUCTIONS = {
    "brief":    "Table: only if the story is explicitly about numbers (economic data, elections, casualties). Format:\n| Metric | Value |\n|--------|-------|",
    "standard": "Table (if meaningful data exists — search for it):\n| Metric | Value |\n|--------|-------|\nSkip if no concrete statistics found.",
    "deep":     "Table (always include — if no hard data, use a key-facts or timeline table):\n| Date / Metric | Event / Value |\n|---------------|---------------|",
}


def _article_writer(index: int, region: str, depth: str) -> Agent:
    cfg = DEPTH_CONFIG[depth]
    provider = REGION_ARTICLE_PROVIDER.get(region, "anthropic")
    model    = REGION_ARTICLE_MODEL.get(region, "cheap")
    return Agent(
        engine=LLMEngine(
            model,
            provider=provider,
            system=_ARTICLE_WRITER_SYSTEM.format(
                region=region,
                today=TODAY,
                index=index,
                searches=cfg["searches"],
                images=cfg["images"],
                paragraphs=cfg["paragraphs"],
                table_instruction=_TABLE_INSTRUCTIONS[depth],
            ),
            native_tools=[NativeTool.WEB_SEARCH],
        ),
        tools=[fetch_image_tool, wikimedia_image_tool],
        name=f"{region.lower().split()[0]}_article_{index}",
        session=session,
    )


# ---------------------------------------------------------------------------
# Region assembler  — combines N parallel articles into clean regional MD
# ---------------------------------------------------------------------------

_REGION_ASSEMBLER_SYSTEM = """\
You receive {n_stories} parallel article outputs. Each is preceded by a label \
like [art1], [art2], etc. — these labels are routing metadata, NOT part of the articles.

Your output must be a single clean Markdown document that:
  1. Strips every [artN] label entirely
  2. Concatenates the articles in order: art1 first, art{n_stories} last
  3. Preserves every article verbatim — headlines, paragraphs, tables, Image: lines
  4. Separates articles with a blank line

Return the combined Markdown only. No labels, no preamble, no sign-off.
"""


def _region_assembler(region: str, n_stories: int) -> Agent:
    return Agent(
        engine=LLMEngine(
            "medium",
            provider="deepseek",
            system=_REGION_ASSEMBLER_SYSTEM.format(n_stories=n_stories),
        ),
        name=f"{region.lower().split()[0]}_assembler",
        session=session,
    )


# ---------------------------------------------------------------------------
# HTML designer  — post-processes the generated HTML for visual impact
# ---------------------------------------------------------------------------

_HTML_DESIGNER_SYSTEM = """\
You are a senior digital news designer. Your task input contains the path of the generated HTML report.

FIRST: extract the html_path from your input. It is a file path ending in ".html"
(e.g. "D:\\LazyBridge\\examples\\news_reports\\daily_news_2026-05-03.html").
Scan the input text for any string ending in ".html" — that is the html_path.
If you cannot find a .html path, return "ERROR: no html_path found in input".

The report may be large (many articles + embedded images). You work SECTION BY SECTION
using specialised tools — you never need to read the full file at once.

━━ WORKFLOW ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1  — call inject_designer_css(html_path)
          This adds the designer stylesheet to <head> once. Do this first.

Step 2  — call extract_html_sections(html_path)
          Returns a list of sections: [{index, headline, html}]
          Each section is one article or one major document block.

Step 3  — for EACH section, apply editorial enhancements:

   LEAD PARAGRAPH — wrap the first <p> in:
     <p class="article-lead">…</p>

   KEY FIGURES — wrap numbers, %, currencies, dates in text nodes:
     <strong class="key-figure">…</strong>
     Examples: "42%", "$1.2 billion", "30 days", "May 3"

   NAMED ENTITIES — first mention of country names, organisation names, people:
     <em class="entity">…</em>

   PULL QUOTE — pick one striking sentence per article (≤ 20 words).
   Insert before the last <p>:
     <blockquote class="pull-quote">…</blockquote>

   Then call patch_html_section(html_path, index, styled_html) for that section.

━━ RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Process one section at a time — never hold multiple sections simultaneously
- Do NOT change any factual text — only wrap existing text with HTML tags
- Do NOT add tags inside <th>, <td>, <figcaption>, or <blockquote class="pull-quote">
- Do NOT touch <figure> or <img> blocks
- When all sections are done, reply with exactly: DONE <absolute_html_path> <sections_styled_count>
"""

_DESIGNER_CSS = """
/* ── HTML Designer enhancements ── */
.article-lead  { font-size:1.13em; font-weight:500; color:#1a1a2e; line-height:1.75; margin-bottom:1.1em; }
.key-figure    { font-weight:700; color:#1f4e79; }
.entity        { font-style:italic; color:#3d3d5c; }
.pull-quote    { border-left:4px solid #1f4e79; margin:1.6em 0; padding:.8em 1.4em;
                 font-size:1.08em; font-style:italic; color:#2c3e50; background:#f4f7fb;
                 border-radius:0 6px 6px 0; }
"""


# Regex that matches <figure>...</figure> blocks and bare self-closing <img> tags.
# Used to strip image payloads from designer context to prevent base64 garbling.
_FIG_RE = re.compile(r'<figure\b[^>]*>.*?</figure>|<img\b[^>]*/>', re.DOTALL)


def _strip_figs(html: str) -> str:
    """Remove figure blocks and bare img tags from an HTML string."""
    return _FIG_RE.sub('', html)


def _raw_at_stripped(raw: str, n: int) -> int:
    """Return the index in *raw* where *n* figure-stripped chars end.

    Walks *raw* skipping over figure/img blocks so we can map a stripped-char
    boundary back to the correct position in the original HTML.
    """
    pos = 0
    counted = 0
    for m in _FIG_RE.finditer(raw):
        before = m.start() - pos
        if counted + before >= n:
            return pos + (n - counted)
        counted += before
        pos = m.end()
    return min(pos + (n - counted), len(raw))


def _make_designer_tools(output_dir: Path) -> list[Tool]:
    # Sentinel injected once so we can split/patch sections reliably
    _SECTION_SENTINEL = "data-designer-section"

    def inject_designer_css(
        html_path: Annotated[str, "Absolute path to the HTML report file"],
    ) -> dict:
        """Inject the designer CSS and section sentinels into the HTML file (call once)."""
        p = Path(html_path)
        try:
            html = p.read_text(encoding="utf-8")
        except OSError as exc:
            return {"error": True, "message": str(exc)}

        # Already injected — idempotent
        if _SECTION_SENTINEL in html:
            return {"injected": False, "reason": "already done"}

        # Add CSS before </head>
        html = html.replace("</head>", f"<style>{_DESIGNER_CSS}</style>\n</head>", 1)

        # Tag every <h2> with a sequential data-designer-section index
        counter = 0
        def _tag_h2(m: re.Match) -> str:
            nonlocal counter
            result = m.group(0).replace("<h2", f'<h2 {_SECTION_SENTINEL}="{counter}"', 1)
            counter += 1
            return result

        html = re.sub(r"<h2[^>]*>", _tag_h2, html)
        p.write_text(html, encoding="utf-8")
        return {"injected": True, "sections_tagged": counter, "path": str(p)}

    def extract_html_sections(
        html_path: Annotated[str, "Absolute path to the HTML report file"],
    ) -> dict:
        """Return list of article sections: [{index, headline, html}].

        Each section spans from one <h2 data-designer-section="N"> up to
        (but not including) the next tagged <h2> or end of article.
        The html field contains only the section's inner HTML — safe to fit in context.
        """
        p = Path(html_path)
        try:
            html = p.read_text(encoding="utf-8")
        except OSError as exc:
            return {"error": True, "message": str(exc)}

        # Find article body
        body_m = re.search(r'<article[^>]*class="markdown-body"[^>]*>(.*?)</article>', html, re.DOTALL)
        if not body_m:
            return {"error": True, "message": "Could not find <article class='markdown-body'>"}
        body = body_m.group(1)

        # Split on tagged h2s
        pattern = re.compile(rf'(<h2[^>]*{re.escape(_SECTION_SENTINEL)}="(\d+)"[^>]*>.*?</h2>)', re.DOTALL)
        parts = pattern.split(body)

        sections = []
        i = 0
        while i < len(parts):
            chunk = parts[i]
            if i + 2 < len(parts) and re.search(rf'{re.escape(_SECTION_SENTINEL)}="\d+"', chunk):
                idx = int(parts[i + 1])
                heading_html = chunk
                body_chunk = parts[i + 2] if i + 2 < len(parts) else ""
                headline_text = re.sub(r"<[^>]+>", "", heading_html).strip()
                # Strip figure/img blocks before truncating — embedded base64 payloads
                # can be hundreds of KB and cause the designer model to output garbled data
                # when they're split mid-string by the 4000-char limit.
                body_no_figs = _strip_figs(body_chunk)
                preview = body_no_figs[:4000]
                sections.append({
                    "index": idx,
                    "headline": headline_text,
                    "html": heading_html + preview,
                    "truncated": len(body_no_figs) > 4000,
                })
                i += 3
            else:
                i += 1

        return {"sections": sections, "total": len(sections)}

    def patch_html_section(
        html_path: Annotated[str, "Absolute path to the HTML report file"],
        index: Annotated[int, "Section index from extract_html_sections"],
        styled_html: Annotated[str, "The enhanced HTML for this section (heading + body, up to 4000 chars)"],
    ) -> dict:
        """Replace one section's HTML in the file. Patches only the matching h2 block."""
        p = Path(html_path)
        try:
            html = p.read_text(encoding="utf-8")
        except OSError as exc:
            return {"error": True, "message": str(exc)}

        # Find the h2 with this index
        h2_pattern = re.compile(
            rf'(<h2[^>]*{re.escape(_SECTION_SENTINEL)}="{index}"[^>]*>.*?</h2>)',
            re.DOTALL,
        )
        m = h2_pattern.search(html)
        if not m:
            return {"error": True, "message": f"Section {index} not found"}

        # Find where the next tagged h2 starts (or article end)
        next_h2 = re.search(
            rf'<h2[^>]*{re.escape(_SECTION_SENTINEL)}="{index + 1}"',
            html[m.end():],
        )
        section_end = m.end() + (next_h2.start() if next_h2 else len(html[m.end():]))
        original_section = html[m.start():section_end]

        # Re-inject figure/img blocks that were stripped before sending to the designer.
        # Images are placed immediately after the closing </h2> tag.
        orig_figs = ''.join(_FIG_RE.findall(original_section))
        if orig_figs:
            styled_html = re.sub(
                r'</h2>',
                lambda _m: _m.group(0) + orig_figs,
                styled_html,
                count=1,
            )

        # Tail: original body content beyond the 4000 figure-stripped-char boundary.
        # We must map that boundary back to a position in the raw (un-stripped) body
        # because figure blocks occupy zero stripped chars but many raw chars.
        h2_end = re.search(r'</h2>', original_section)
        body_start = h2_end.end() if h2_end else 0
        body_raw = original_section[body_start:]
        tail_offset = _raw_at_stripped(body_raw, 4000)
        tail = body_raw[tail_offset:]

        html = html[:m.start()] + styled_html + tail + html[section_end:]
        p.write_text(html, encoding="utf-8")
        return {"patched": True, "index": index, "path": str(p)}

    return [
        Tool(inject_designer_css, name="inject_designer_css"),
        Tool(extract_html_sections, name="extract_html_sections"),
        Tool(patch_html_section, name="patch_html_section"),
    ]


html_designer = Agent(
    engine=LLMEngine("medium", provider="deepseek", system=_HTML_DESIGNER_SYSTEM),
    tools=_make_designer_tools(OUTPUT_DIR),
    name="html_designer",
    session=session,
)

# ---------------------------------------------------------------------------
# Final orchestrator  — assembly + HTML generation
# ---------------------------------------------------------------------------

_FINAL_ORCHESTRATOR_SYSTEM = f"""\
You receive enriched regional news Markdown from 3 pipelines, labelled:
  [🇺🇸_united] → United States
  [🇨🇳_china]  → China
  [🇮🇳_india]  → India

Each article block ends with one or more "Image: /local/path/..." lines.
Today: {TODAY}    Output dir: {OUTPUT_DIR.resolve()}

─── STEP 1 — Build charts list ─────────────────────────────────────────────
Scan all articles. For every "Image: /local/path" line collect:
  path  : the local path
  name  : the ## headline of the article that contains this Image: line
  title : same as name
Ignore lines where the path does not start with "/" or a drive letter.

─── STEP 2 — call save_markdown ────────────────────────────────────────────
filename: `daily_news_{TODAY}.md`  (use this exact string, no markdown formatting)
Build the global document — paste ALL article content VERBATIM (including
tables) but REMOVE every "Image: ..." line from the text:

    # 🌍 Daily Global News — {TODAY}

    ## Global Highlights
    <Select 5–7 most significant stories across all regions.
    One bullet: flag emoji + headline + one sentence summary.>

    ---

    ## 🇺🇸 United States
    <full US content — Image lines removed>

    ## 🇨🇳 China
    <full China content — Image lines removed>

    ## 🇮🇳 India
    <full India content — Image lines removed>

─── STEP 3 — call generate_report ─────────────────────────────────────────
  markdown_path  : path returned by save_markdown
  title          : "🌍 Daily Global News — {TODAY}"
  theme          : "executive"
  template       : "deep_dive"
  output_filename: "daily_news_{TODAY}.html"
  charts         : the list from Step 1

When done, reply with only the absolute html_path on a single line.
"""

final_orchestrator = Agent(
    engine=LLMEngine("medium", provider="deepseek", system=_FINAL_ORCHESTRATOR_SYSTEM),
    tools=[save_md_tool, *report_tools(OUTPUT_DIR)],
    name="final_orchestrator",
    session=session,
)

# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def _region_pipeline(region: str, flag: str, key: str, n_stories: int, depth: str) -> tuple[Agent, str]:
    """Build one region pipeline and return (agent, store_key)."""
    writer_steps = [
        Step(_article_writer(i + 1, region, depth), parallel=True, name=f"art{i + 1}")
        for i in range(n_stories)
    ]
    return (
        Agent(
            engine=Plan(
                Step(_discovery_agent(region, n_stories)),
                *writer_steps,
                Step(_region_assembler(region, n_stories), task=from_parallel_all("art1")),
            ),
            name=f"{flag}_{region.lower().split()[0]}",
            session=session,
        ),
        f"{key}_md",
    )


def build_pipeline(n_stories: int = 5, depth: str = "standard") -> Agent:
    """Construct the full news pipeline with the given parameters."""
    assert depth in DEPTH_CONFIG, f"depth must be one of {list(DEPTH_CONFIG)}"

    region_steps = []
    for region, flag, key in REGIONS:
        agent, store_key = _region_pipeline(region, flag, key, n_stories, depth)
        region_steps.append(
            Step(agent, parallel=True, name=f"{key}_report", writes=store_key)
        )

    return Agent(
        engine=Plan(
            *region_steps,
            Step(final_orchestrator, task=from_parallel_all(region_steps[0].name)),
            Step(html_designer),
            store=store,
        ),
        name="daily_news_pipeline",
        session=session,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily Global News Report pipeline")
    parser.add_argument("--stories", type=int, default=5, help="Number of stories per region (default: 5)")
    parser.add_argument(
        "--depth",
        choices=list(DEPTH_CONFIG),
        default="standard",
        help="Article depth: brief | standard | deep (default: standard)",
    )
    args = parser.parse_args()

    print(f"Building pipeline: {args.stories} stories × 3 regions, depth={args.depth}")
    print(f"Article writers: {args.stories * 3} agents running in parallel\n")

    pipeline = build_pipeline(n_stories=args.stories, depth=args.depth)
    result = pipeline(f"Produce today's ({TODAY}) global news digest.")

    # Locate the HTML file — deterministic path produced by final_orchestrator
    import re as _re
    html_file = OUTPUT_DIR / f"daily_news_{TODAY}.html"
    raw = result.text() or ""
    if html_file.exists():
        # Try to extract styled-section count from designer's last text reply
        m = _re.search(r"DONE\s+\S+\s+(\d+)", raw)
        styled = int(m.group(1)) if m else "?"
        print(f"\nReport ready → {html_file}")
        print(f"Sections styled: {styled}")
    else:
        # Check if orchestrator wrote a path to text (fallback)
        m = _re.search(r'[\w/\\:][^\s"\']*\.html', raw)
        html_path = m.group(0) if m else "(not found)"
        print(f"\nReport path: {html_path}")
        print(f"  [raw output]: {raw[:400]}")
    print(f"Store keys: {store.keys()}")

    usage = session.usage_summary()
    print(f"\n── Usage ──────────────────────────────────────────────────────")
    print(f"  Total  in={usage['total']['input_tokens']:,}  out={usage['total']['output_tokens']:,}  cost=${usage['total']['cost_usd']:.4f}")
    print(f"\n  By agent:")
    for agent_name, u in sorted(usage["by_agent"].items(), key=lambda x: -x[1]["cost_usd"]):
        print(f"    {agent_name:<40} in={u['input_tokens']:>8,}  out={u['output_tokens']:>7,}  ${u['cost_usd']:.4f}")
