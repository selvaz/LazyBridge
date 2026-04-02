"""Unit tests for LazyContext — no API calls."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from lazybridge.lazy_context import LazyContext
from lazybridge.lazy_store import LazyStore


# ── T3.01 — from_text: build() returns verbatim text ─────────────────────────

def test_from_text_build():
    ctx = LazyContext.from_text("you are an analyst")
    assert ctx.build() == "you are an analyst"


def test_from_text_strips_whitespace():
    ctx = LazyContext.from_text("  hello  ")
    assert ctx.build() == "hello"


# ── T3.02 — from_function: fn is called at build time, not at creation time ──

def test_from_function_lazy():
    calls = []

    def source() -> str:
        calls.append(1)
        return "dynamic content"

    ctx = LazyContext.from_function(source)
    assert calls == []          # not called yet
    result = ctx.build()
    assert len(calls) == 1      # called exactly once
    assert result == "dynamic content"


# ── T3.03 — from_agent: agent not yet run → empty string ─────────────────────

def test_from_agent_not_run():
    agent = MagicMock()
    agent.name = "researcher"
    agent._last_output = None
    ctx = LazyContext.from_agent(agent)
    assert ctx.build() == ""


# ── T3.04 — from_agent: agent has run → output included with label ────────────

def test_from_agent_with_output():
    agent = MagicMock()
    agent.name = "researcher"
    agent._last_output = "AI sector is booming"
    ctx = LazyContext.from_agent(agent)
    result = ctx.build()
    assert "AI sector is booming" in result
    assert "researcher" in result


def test_from_agent_custom_prefix():
    agent = MagicMock()
    agent.name = "analyst"
    agent._last_output = "bullish"
    ctx = LazyContext.from_agent(agent, prefix="[prev output]")
    result = ctx.build()
    assert result.startswith("[prev output]")
    assert "bullish" in result


# ── T3.05 — from_store: reads store keys at build time ───────────────────────

def test_from_store_reads_at_build_time():
    store = LazyStore()
    ctx = LazyContext.from_store(store, keys=["findings"])

    # Nothing in store yet → empty
    assert ctx.build() == ""

    # Write after context creation → now visible at build time
    store.write("findings", "CRISPR is promising")
    result = ctx.build()
    assert "findings" in result
    assert "CRISPR is promising" in result


def test_from_store_filtered_keys():
    store = LazyStore()
    store.write("a", "include me")
    store.write("b", "exclude me")
    ctx = LazyContext.from_store(store, keys=["a"])
    result = ctx.build()
    assert "include me" in result
    assert "exclude me" not in result


# ── T3.06 — merge / __add__: sources concatenated in order ───────────────────

def test_merge_two_contexts():
    ctx1 = LazyContext.from_text("first")
    ctx2 = LazyContext.from_text("second")
    merged = LazyContext.merge(ctx1, ctx2)
    result = merged.build()
    assert "first" in result
    assert "second" in result
    assert result.index("first") < result.index("second")


def test_add_operator():
    ctx = LazyContext.from_text("part A") + LazyContext.from_text("part B")
    result = ctx.build()
    assert "part A" in result
    assert "part B" in result


# ── T3.07 — failing source is skipped, build() does not raise ────────────────

def test_failing_source_skipped():
    def bad_source() -> str:
        raise RuntimeError("source exploded")

    ctx = LazyContext.from_function(bad_source)
    ctx._sources.append(lambda: "good content")  # add a working source too

    result = ctx.build()   # should not raise
    assert "good content" in result


# ── T3.08 — __bool__: True with sources, False when empty ────────────────────

def test_bool_with_sources():
    ctx = LazyContext.from_text("something")
    assert bool(ctx) is True


def test_bool_empty():
    ctx = LazyContext()
    assert bool(ctx) is False
