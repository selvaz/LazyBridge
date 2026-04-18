#!/usr/bin/env python3
"""Emit the model-tier matrix as markdown by reading each provider's
``_TIER_ALIASES`` attribute.

Run::

    python tools/generate_tier_matrix.py

Prints the canonical table to stdout — copy into
``lazy_wiki/human/agents.md`` / ``README.md`` / ``lazy_wiki/bot/00_quickref.md``.
A drift-guard test in ``tests/unit/gui/test_audit_c2_tier_matrix.py``
re-parses ``agents.md`` and asserts it matches this script's output.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly from a checkout without
# ``pip install -e .`` — prepend the repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lazybridge.core.providers.anthropic import AnthropicProvider
from lazybridge.core.providers.deepseek import DeepSeekProvider
from lazybridge.core.providers.google import GoogleProvider
from lazybridge.core.providers.openai import OpenAIProvider

# Display order per tier (top → super_cheap).
_TIER_ORDER = ["top", "expensive", "medium", "cheap", "super_cheap"]

# Provider column order and header label.
_PROVIDER_COLUMNS: list[tuple[str, type]] = [
    ("`anthropic` / `claude`",    AnthropicProvider),
    ("`openai` / `chatgpt` / `gpt`", OpenAIProvider),
    ("`google` / `gemini`",       GoogleProvider),
    ("`deepseek`",                DeepSeekProvider),
]


def render() -> str:
    header = "| tier | " + " | ".join(label for label, _ in _PROVIDER_COLUMNS) + " |"
    sep = "| --- | " + " | ".join("---" for _ in _PROVIDER_COLUMNS) + " |"
    rows = [header, sep]
    for tier in _TIER_ORDER:
        cells = [f"`{tier}`"]
        for _label, cls in _PROVIDER_COLUMNS:
            model = cls._TIER_ALIASES.get(tier, "?")
            # Note cases where a tier shares a model with another tier.
            duplicate_with = [
                t for t in _TIER_ORDER
                if t != tier and cls._TIER_ALIASES.get(t) == model
            ]
            if duplicate_with:
                cells.append(f"{model} *(same as {', '.join(duplicate_with)})*")
            else:
                cells.append(model)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    print(render(), end="")
