"""Regression tests for the v1-stabilization support-module audit fixes.

* F8 — repeated Memory compression overwrote the prior summary instead of
  folding it in: the oldest context was silently lost on the second
  compression.
* tool_schema — unannotated parameters in signature mode emitted an empty
  ``{}`` subschema (rejected/dropped by strict-mode validators) instead of
  the documented ``{"type": "string"}`` fallback.
* tool_schema — ``$defs`` name collisions across distinct models were
  silently resolved last-write-wins during ``flatten_refs``.
* guardrails — LLMGuard's async path enforced the judge timeout twice
  (outer ``wait_for`` + inner daemon thread), leaking a daemon thread per
  timed-out call.
* dedup_guard — ``verbose=True`` by default ``print()``-ed to stdout from
  library code; and the module had no behavioral coverage.
"""

from __future__ import annotations

import asyncio

from lazybridge.dedup_guard import DeduplicateGuard, deduplicate
from lazybridge.memory import Memory

# ---------------------------------------------------------------------------
# F8 — summary accumulation across compressions
# ---------------------------------------------------------------------------


def test_rule_summary_folds_prior_topics():
    mem = Memory(strategy="sliding", max_tokens=None)
    # First compression: 11 turns → summary of the first (11-10)=1 turn.
    for i in range(11):
        mem.add(f"question about quantum{i}", f"answer about quantum{i}")
    first_summary = mem._summary
    assert "quantum0" in first_summary

    # Refill past the window to trigger a second compression.
    for i in range(11):
        mem.add(f"question about finance{i}", f"answer about finance{i}")
    second_summary = mem._summary
    # The topics captured by the first compression must survive the second.
    assert "quantum0" in second_summary


def test_llm_summary_receives_prior_summary():
    prompts: list[str] = []

    def summarizer(prompt: str) -> str:
        prompts.append(prompt)
        return f"summary#{len(prompts)}"

    mem = Memory(strategy="summary", summarizer=summarizer)
    for i in range(11):
        mem.add(f"u{i}", f"a{i}")
    assert mem._summary == "summary#1"
    mem.add("x", "y")  # strategy="summary" compresses on every add past the window
    assert mem._summary == "summary#2"
    # The second summarizer call must include the first summary as
    # material to fold in.
    assert "summary#1" in prompts[1]


# ---------------------------------------------------------------------------
# tool_schema — unannotated params in strict signature mode
# ---------------------------------------------------------------------------


def test_unannotated_param_gets_string_schema_in_strict_mode():
    from lazybridge.tools import Tool

    def fn(x):  # deliberately unannotated
        return str(x)

    t = Tool(fn, name="fn", description="d")
    schema = t.definition().parameters
    assert schema["properties"]["x"] == {"type": "string"}


# ---------------------------------------------------------------------------
# tool_schema — $defs collisions fail loud
# ---------------------------------------------------------------------------


def test_flatten_refs_rejects_conflicting_defs():
    import pytest

    from lazybridge.core.tool_schema import _flatten_refs

    schema = {
        "type": "object",
        "properties": {
            "a": {
                "$defs": {"Config": {"type": "object", "properties": {"x": {"type": "integer"}}}},
                "$ref": "#/$defs/Config",
            },
            "b": {
                "$defs": {"Config": {"type": "object", "properties": {"y": {"type": "string"}}}},
                "$ref": "#/$defs/Config",
            },
        },
    }
    with pytest.raises(ValueError, match="Config"):
        _flatten_refs(schema)


def test_flatten_refs_accepts_identical_duplicate_defs():
    from lazybridge.core.tool_schema import _flatten_refs

    shared = {"type": "object", "properties": {"x": {"type": "integer"}}}
    schema = {
        "type": "object",
        "properties": {
            "a": {"$defs": {"Config": dict(shared)}, "$ref": "#/$defs/Config"},
            "b": {"$defs": {"Config": dict(shared)}, "$ref": "#/$defs/Config"},
        },
    }
    out = _flatten_refs(schema)
    assert "$ref" not in str(out)


# ---------------------------------------------------------------------------
# guardrails — single timeout enforcement on the async path
# ---------------------------------------------------------------------------


def test_ajudge_sync_fallback_uses_untimed_judge(monkeypatch):
    from lazybridge.guardrails import LLMGuard

    calls: dict[str, int] = {"once": 0, "judge": 0}

    class _SyncJudge:
        def __call__(self, prompt):
            class _E:
                @staticmethod
                def text():
                    return "allow"

            return _E()

    guard = LLMGuard(_SyncJudge(), timeout=5.0)

    orig_once = guard._judge_once
    orig_judge = guard._judge

    def _spy_once(text):
        calls["once"] += 1
        return orig_once(text)

    def _spy_judge(text):
        calls["judge"] += 1
        return orig_judge(text)

    monkeypatch.setattr(guard, "_judge_once", _spy_once)
    monkeypatch.setattr(guard, "_judge", _spy_judge)

    action = asyncio.run(guard.acheck_input("hello"))
    assert action.allowed
    # The async path must route through the untimed single round-trip —
    # not through _judge, which would enforce the timeout a second time
    # on its own daemon thread.
    assert calls["once"] == 1
    assert calls["judge"] == 0


def test_ajudge_timeout_fails_closed():
    import time

    from lazybridge.guardrails import LLMGuard

    class _HungJudge:
        def __call__(self, prompt):
            time.sleep(2)

            class _E:
                @staticmethod
                def text():
                    return "allow"

            return _E()

    guard = LLMGuard(_HungJudge(), timeout=0.1)

    async def _go():
        # Measure inside the loop: asyncio.run's shutdown joins the
        # default executor (whose worker is still sleeping), so wall
        # time around asyncio.run would include that join.
        t0 = time.monotonic()
        action = await guard.acheck_input("hello")
        return action, time.monotonic() - t0

    action, elapsed = asyncio.run(_go())
    assert not action.allowed
    assert elapsed < 1.5


# ---------------------------------------------------------------------------
# dedup_guard — behavior + no stdout by default
# ---------------------------------------------------------------------------


def test_deduplicate_removes_repeated_paragraphs():
    text = "First paragraph with plenty of unique content here.\n\nSecond block that is repeated verbatim for sure.\n\nSecond block that is repeated verbatim for sure."
    cleaned, removed = deduplicate(text)
    assert removed == 1
    assert cleaned.count("Second block") == 1


def test_deduplicate_near_duplicate_prefix():
    a = "Common shared prefix that is quite long and identical between blocks — variant one."
    b = "Common shared prefix that is quite long and identical between blocks — variant two."
    cleaned, removed = deduplicate(f"{a}\n\n{b}", similarity_chars=40)
    assert removed == 1


def test_deduplicate_keeps_short_blocks():
    text = "Yes\n\nYes\n\nYes"
    cleaned, removed = deduplicate(text, min_block_chars=10)
    assert removed == 0
    assert cleaned.count("Yes") == 3


def test_guard_modifies_input_and_is_silent_by_default(capsys):
    guard = DeduplicateGuard(min_block_chars=0)
    block = "A very repetitive block of content that goes on for a while."
    action = guard.check_input(f"{block}\n\n{block}")
    assert action.allowed
    assert action.modified_text is not None
    assert action.modified_text.count("repetitive") == 1
    # Library code must not print to stdout unsolicited.
    assert capsys.readouterr().out == ""


def test_guard_allows_clean_input_unchanged():
    guard = DeduplicateGuard()
    action = guard.check_input("Totally unique content.")
    assert action.allowed
    assert action.modified_text is None


def test_guard_dialogue_turn_dedup():
    text = "[Turn 1] Alice: hello there my friend, how are you today?\n[Turn 2] Bob: fine thanks for asking!\n[Turn 1] Alice: hello there my friend, how are you today?"
    guard = DeduplicateGuard(min_block_chars=0)
    action = guard.check_input(text)
    assert action.modified_text is not None
    assert action.modified_text.count("[Turn 1]") == 1


# ---------------------------------------------------------------------------
# __version__ fallback matches pyproject
# ---------------------------------------------------------------------------


def test_version_fallback_matches_pyproject():
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2]
    pyproject = (root / "pyproject.toml").read_text()
    declared = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
    init_src = (root / "lazybridge" / "__init__.py").read_text()
    fallbacks = set(re.findall(r'__version__ = "([^"]+)"', init_src))
    assert fallbacks == {declared}, (
        f"__version__ fallback(s) {fallbacks} diverge from pyproject version {declared!r} — keep them in sync"
    )
