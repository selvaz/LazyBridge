"""Unit tests for guardrails — input/output validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lazybridge.core.types import CompletionResponse, UsageStats
from lazybridge.guardrails import ContentGuard, GuardAction, GuardChain, GuardError


def _make_agent():
    from lazybridge.lazy_agent import LazyAgent

    with patch("lazybridge.core.executor.Executor.__init__", return_value=None):
        agent = LazyAgent.__new__(LazyAgent)
    mock_exec = MagicMock()
    mock_exec.provider.get_default_max_tokens.return_value = 4096
    mock_exec.model = "test-model"
    agent._executor = mock_exec
    import uuid

    agent.id = str(uuid.uuid4())
    agent.name = "test"
    agent.description = None
    agent.system = None
    agent.context = None
    agent.tools = []
    agent.native_tools = []
    agent.output_schema = None
    agent._last_output = None
    agent._last_response = None
    agent.session = None
    agent._log = None
    return agent


# ---------------------------------------------------------------------------
# GuardAction
# ---------------------------------------------------------------------------


def test_guard_action_allow():
    a = GuardAction.allow()
    assert a.allowed is True
    assert a.message is None


def test_guard_action_block():
    a = GuardAction.block("bad content")
    assert a.allowed is False
    assert a.message == "bad content"


def test_guard_action_modify():
    a = GuardAction.modify("cleaned text", message="PII removed")
    assert a.allowed is True
    assert a.modified_text == "cleaned text"


# ---------------------------------------------------------------------------
# ContentGuard
# ---------------------------------------------------------------------------


def test_content_guard_input_blocks():
    def block_bad(text: str) -> GuardAction:
        if "blocked" in text:
            return GuardAction.block("contains blocked word")
        return GuardAction.allow()

    guard = ContentGuard(input_fn=block_bad)
    agent = _make_agent()
    agent._executor.execute.return_value = CompletionResponse(content="ok", usage=UsageStats())

    with pytest.raises(GuardError, match="blocked word"):
        agent.chat("this is blocked content", guard=guard)


def test_content_guard_output_blocks():
    def block_toxic(text: str) -> GuardAction:
        if "toxic" in text:
            return GuardAction.block("toxic output detected")
        return GuardAction.allow()

    guard = ContentGuard(output_fn=block_toxic)
    agent = _make_agent()
    agent._executor.execute.return_value = CompletionResponse(content="this is toxic output", usage=UsageStats())

    with pytest.raises(GuardError, match="toxic output"):
        agent.chat("hello", guard=guard)


def test_content_guard_allows_clean_content():
    def check(text: str) -> GuardAction:
        if "bad" in text:
            return GuardAction.block("bad")
        return GuardAction.allow()

    guard = ContentGuard(input_fn=check, output_fn=check)
    agent = _make_agent()
    agent._executor.execute.return_value = CompletionResponse(content="clean response", usage=UsageStats())

    resp = agent.chat("clean input", guard=guard)
    assert resp.content == "clean response"


def test_content_guard_modifies_input():
    def redact(text: str) -> GuardAction:
        if "secret" in text:
            return GuardAction.modify(text.replace("secret", "[REDACTED]"))
        return GuardAction.allow()

    guard = ContentGuard(input_fn=redact)
    agent = _make_agent()
    agent._executor.execute.return_value = CompletionResponse(content="ok", usage=UsageStats())

    agent.chat("my secret password", guard=guard)
    call_args = agent._executor.execute.call_args[0][0]
    assert "[REDACTED]" in call_args.messages[0].content


# ---------------------------------------------------------------------------
# GuardChain
# ---------------------------------------------------------------------------


def test_guard_chain_first_block_wins():
    g1 = ContentGuard(input_fn=lambda t: GuardAction.allow())
    g2 = ContentGuard(input_fn=lambda t: GuardAction.block("g2 blocked"))
    g3 = ContentGuard(input_fn=lambda t: GuardAction.block("g3 blocked"))

    chain = GuardChain([g1, g2, g3])
    action = chain.check_input("test")
    assert not action.allowed
    assert action.message == "g2 blocked"


def test_guard_chain_all_allow():
    g1 = ContentGuard(input_fn=lambda t: GuardAction.allow())
    g2 = ContentGuard(input_fn=lambda t: GuardAction.allow())

    chain = GuardChain([g1, g2])
    action = chain.check_input("test")
    assert action.allowed


# ---------------------------------------------------------------------------
# Guard on loop()
# ---------------------------------------------------------------------------


def test_guard_blocks_loop_input():
    guard = ContentGuard(input_fn=lambda t: GuardAction.block("nope"))
    agent = _make_agent()

    with pytest.raises(GuardError, match="nope"):
        agent.loop("bad task", guard=guard)


def test_guard_blocks_loop_output():
    def block_output(text: str) -> GuardAction:
        if "unsafe" in text:
            return GuardAction.block("unsafe output")
        return GuardAction.allow()

    guard = ContentGuard(output_fn=block_output)
    agent = _make_agent()
    agent._executor.execute.return_value = CompletionResponse(content="unsafe result", usage=UsageStats())

    with pytest.raises(GuardError, match="unsafe output"):
        agent.loop("task", guard=guard)


# ---------------------------------------------------------------------------
# GuardError
# ---------------------------------------------------------------------------


def test_guard_error_has_action():
    action = GuardAction.block("test reason", score=0.95)
    err = GuardError(action)
    assert err.action is action
    assert "test reason" in str(err)
    assert err.action.metadata["score"] == 0.95
