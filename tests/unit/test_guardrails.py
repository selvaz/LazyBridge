"""Tests for the v1.0 guardrails."""

from __future__ import annotations

import asyncio
import pytest

from lazybridge.guardrails import (
    ContentGuard,
    Guard,
    GuardAction,
    GuardChain,
    GuardError,
    LLMGuard,
)


# ── GuardAction factories ─────────────────────────────────────────────────────

def test_guard_action_allow():
    a = GuardAction.allow()
    assert a.allowed is True
    assert a.message is None

def test_guard_action_block():
    a = GuardAction.block("blocked")
    assert a.allowed is False
    assert a.message == "blocked"

def test_guard_action_modify():
    a = GuardAction.modify("clean text", message="sanitized")
    assert a.allowed is True
    assert a.modified_text == "clean text"
    assert a.message == "sanitized"

def test_guard_action_metadata():
    a = GuardAction.allow(score=0.9)
    assert a.metadata["score"] == 0.9


# ── Guard base (default allows everything) ────────────────────────────────────

def test_base_guard_allows_by_default():
    g = Guard()
    assert g.check_input("anything").allowed is True
    assert g.check_output("anything").allowed is True

def test_base_guard_async():
    g = Guard()
    assert asyncio.run(g.acheck_input("x")).allowed is True
    assert asyncio.run(g.acheck_output("x")).allowed is True


# ── ContentGuard ──────────────────────────────────────────────────────────────

def test_content_guard_input_allow():
    g = ContentGuard(input_fn=lambda t: GuardAction.allow())
    assert g.check_input("hello").allowed is True

def test_content_guard_input_block():
    g = ContentGuard(input_fn=lambda t: GuardAction.block("no"))
    action = g.check_input("bad")
    assert not action.allowed
    assert action.message == "no"

def test_content_guard_output_modify():
    g = ContentGuard(output_fn=lambda t: GuardAction.modify(t.replace("bad", "***")))
    action = g.check_output("bad word")
    assert action.allowed
    assert action.modified_text == "*** word"

def test_content_guard_no_fn_allows():
    g = ContentGuard()
    assert g.check_input("x").allowed is True
    assert g.check_output("x").allowed is True


# ── GuardChain ────────────────────────────────────────────────────────────────

def test_chain_first_block_wins():
    chain = GuardChain(
        ContentGuard(input_fn=lambda t: GuardAction.block("first")),
        ContentGuard(input_fn=lambda t: GuardAction.block("second")),
    )
    action = chain.check_input("x")
    assert not action.allowed
    assert action.message == "first"

def test_chain_all_allow():
    chain = GuardChain(
        ContentGuard(input_fn=lambda t: GuardAction.allow()),
        ContentGuard(input_fn=lambda t: GuardAction.allow()),
    )
    assert chain.check_input("x").allowed is True

def test_chain_modification_propagates():
    chain = GuardChain(
        ContentGuard(input_fn=lambda t: GuardAction.modify(t.upper())),
        ContentGuard(input_fn=lambda t: GuardAction.allow()),
    )
    # Modified text propagates to next guard but final result is allow
    action = chain.check_input("hello")
    assert action.allowed

def test_chain_async():
    chain = GuardChain(
        ContentGuard(input_fn=lambda t: GuardAction.allow()),
    )
    action = asyncio.run(chain.acheck_input("x"))
    assert action.allowed

def test_chain_empty():
    chain = GuardChain()
    assert chain.check_input("x").allowed is True


# ── GuardError ────────────────────────────────────────────────────────────────

def test_guard_error_is_exception():
    with pytest.raises(GuardError):
        raise GuardError("test guard error")
