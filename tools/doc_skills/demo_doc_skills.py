"""
demo_doc_skills.py  —  manual demo for lazybridge.tools.doc_skills
Run with F5 in Spyder or: python tools/doc_skills/demo_doc_skills.py

NOT a pytest test — intentionally named demo_* to avoid automatic test discovery.
Sections 1-2 run without an API key. Sections 3-4 require ANTHROPIC_API_KEY in .env.
"""

import os
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Load .env — prints key names only, no values ──────────────────────────────
_env = REPO_ROOT / ".env"
if _env.exists():
    _loaded = []
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _k, _v = _k.strip(), _v.strip().strip("'\"")
            os.environ.setdefault(_k, _v)
            _loaded.append(_k)
    print(f"[.env] {len(_loaded)} key(s) loaded: {', '.join(_loaded)}")
else:
    print(f"[.env] not found at {_env}")

from lazybridge.tools.doc_skills import build_skill, query_skill, skill_tool, skill_pipeline  # noqa: E402

WIKI_DIR   = str(REPO_ROOT / "lazy_wiki" / "bot")
SKILL_ROOT = str(REPO_ROOT / "generated_skills")
SEP        = "=" * 60

# ═════════════════════════════════════════════════════════════════════════════
# 1. BUILD — index the bot wiki into a skill bundle
# ═════════════════════════════════════════════════════════════════════════════
print("\nBuilding skill…")
meta = build_skill(
    source_dirs=[WIKI_DIR],
    skill_name="lazybridge-docs",
    description="LazyBridge framework reference — agents, sessions, tools, patterns.",
    usage_notes="Prefer concrete API names, method signatures, and canonical pattern names.",
    output_root=SKILL_ROOT,
)
print(f"  skill_dir : {meta['skill_dir']}")
print(f"  files     : {len(meta['indexed_files'])}")
print(f"  chunks    : {meta['total_chunks']}")
print(f"  avgdl     : {meta['avgdl']} tokens")

# ═════════════════════════════════════════════════════════════════════════════
# 2. PLAIN RETRIEVAL — no LLM, no API key needed
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{SEP}\nQUERY 1 — answer\n{SEP}")
print(query_skill(
    meta["skill_dir"],
    "How does sess.as_tool(mode='chain') wire context between agents?",
))

print(f"\n{SEP}\nQUERY 2 — locate\n{SEP}")
print(query_skill(
    meta["skill_dir"],
    "Where is verify= documented?",
    mode="locate",
))

print(f"\n{SEP}\nQUERY 3 — summarize\n{SEP}")
print(query_skill(
    meta["skill_dir"],
    "Summarize the pattern hierarchy in LazyBridge.",
    mode="summarize",
    top_k=5,
))

# ═════════════════════════════════════════════════════════════════════════════
# 3. AGENT + SKILL TOOL — requires ANTHROPIC_API_KEY
# ═════════════════════════════════════════════════════════════════════════════
if not os.environ.get("ANTHROPIC_API_KEY"):
    print(f"\n[skip] Sections 3-4 require ANTHROPIC_API_KEY in .env")
else:
    from lazybridge import LazyAgent

    tool  = skill_tool(meta["skill_dir"])
    agent = LazyAgent("anthropic")

    print(f"\n{SEP}\nAGENT — skill_tool\n{SEP}")
    resp = agent.loop(
        "What is the canonical pattern for a sequential pipeline where each agent "
        "feeds the next? Show code.",
        tools=[tool],
    )
    print(resp.content)

# ═════════════════════════════════════════════════════════════════════════════
# 4. FULL PIPELINE — router + executor chain, requires ANTHROPIC_API_KEY
# ═════════════════════════════════════════════════════════════════════════════
#   pipeline     = skill_pipeline(skill_dir=meta["skill_dir"], provider="anthropic")
#   orchestrator = LazyAgent("anthropic")
#   resp = orchestrator.loop(
#       "When should I use LazyStore instead of LazyContext.from_agent?",
#       tools=[pipeline],
#   )
#   print(resp.content)
