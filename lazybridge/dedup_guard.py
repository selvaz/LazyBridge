"""
DeduplicateGuard — input deduplication guard for LazyBridge agents.

Removes repeated text blocks from task/context before they reach the LLM.
Designed for multi-agent dialogue chains where conversation history
gets copy-pasted recursively, inflating token usage.

Usage::

    from lazybridge.dedup_guard import DeduplicateGuard

    agent = Agent(
        engine=LLMEngine(...),
        guard=DeduplicateGuard(),
    )

The guard fires on check_input only (output is left untouched).
"""

from __future__ import annotations

import re
from lazybridge.guardrails import Guard, GuardAction


def _normalise(text: str) -> str:
    """Collapse whitespace and lowercase for comparison."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _split_blocks(text: str) -> list[str]:
    """Split text into meaningful blocks.

    Tries in order:
    1. [Turn N] markers  — dialogue turns
    2. Double newlines   — paragraphs
    3. Single newlines   — lines
    """
    # Try dialogue turn markers first
    turns = re.split(r"(?=\[Turn\s+\d+\])", text)
    if len(turns) > 1:
        return [t for t in turns if t.strip()]
    # Paragraph split
    paras = re.split(r"\n{2,}", text)
    if len(paras) > 1:
        return [p for p in paras if p.strip()]
    # Line split as last resort
    return [ln for ln in text.splitlines() if ln.strip()]


def deduplicate(text: str, *, similarity_chars: int = 60) -> tuple[str, int]:
    """Remove duplicate blocks from text.

    Parameters
    ----------
    text:
        The raw input string to clean.
    similarity_chars:
        Number of leading characters used to detect near-duplicate blocks.
        Blocks whose first N chars match a previously seen block are dropped.

    Returns
    -------
    (cleaned_text, n_removed)
        The deduplicated string and how many blocks were removed.
    """
    if not text:
        return text, 0

    blocks   = _split_blocks(text)
    seen_full: set[str] = set()
    seen_prefix: set[str] = set()
    kept: list[str] = []
    removed = 0

    for block in blocks:
        norm   = _normalise(block)
        prefix = norm[:similarity_chars]

        if norm in seen_full or (prefix and prefix in seen_prefix):
            removed += 1
            continue

        seen_full.add(norm)
        if prefix:
            seen_prefix.add(prefix)
        kept.append(block)

    # Re-join with the same separator that best fits the original
    if re.search(r"\[Turn\s+\d+\]", text):
        cleaned = "\n".join(kept)
    elif "\n\n" in text:
        cleaned = "\n\n".join(kept)
    else:
        cleaned = "\n".join(kept)

    return cleaned.strip(), removed


class DeduplicateGuard(Guard):
    """Input guard that removes repeated text blocks before the LLM sees them.

    Parameters
    ----------
    similarity_chars:
        Length of the prefix fingerprint used to detect near-duplicates.
        Lower = more aggressive dedup.  Default 60 is good for dialogue turns.
    min_block_chars:
        Blocks shorter than this are never deduplicated (avoids removing
        short repeated phrases like "Yes" or "Certo").  Default 40.
    verbose:
        If True, prints a one-line summary when blocks are removed.
    """

    def __init__(
        self,
        *,
        similarity_chars: int = 60,
        min_block_chars: int = 40,
        verbose: bool = True,
    ) -> None:
        self._sim_chars  = similarity_chars
        self._min_chars  = min_block_chars
        self._verbose    = verbose

    def check_input(self, text: str) -> GuardAction:
        if not text or len(text) < self._min_chars:
            return GuardAction.allow()

        cleaned, n_removed = deduplicate(text, similarity_chars=self._sim_chars)

        if n_removed == 0:
            return GuardAction.allow()

        reduction_pct = round((1 - len(cleaned) / max(len(text), 1)) * 100)
        if self._verbose:
            print(
                f"  [DeduplicateGuard] removed {n_removed} duplicate block(s) "
                f"— {len(text):,} → {len(cleaned):,} chars (-{reduction_pct}%)"
            )

        return GuardAction.modify(cleaned)
