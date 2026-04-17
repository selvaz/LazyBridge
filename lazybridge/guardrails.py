"""lazybridge.guardrails — lightweight input/output validation for agents.

A Guard runs before and/or after an LLM call. It can block, modify, or
flag content — useful for safety, compliance, PII filtering, and
custom business rules.

Quick start::

    from lazybridge import LazyAgent
    from lazybridge.guardrails import Guard, GuardAction, ContentGuard

    # Block toxic content
    def check_toxicity(text: str) -> GuardAction:
        if "harmful" in text.lower():
            return GuardAction.block("Content blocked: harmful language detected")
        return GuardAction.allow()

    guard = ContentGuard(input_fn=check_toxicity, output_fn=check_toxicity)
    agent = LazyAgent("anthropic")
    resp = agent.chat("hello", guard=guard)  # runs check on input and output

Composing guards::

    from lazybridge.guardrails import GuardChain

    chain = GuardChain([pii_guard, toxicity_guard, length_guard])
    resp = agent.chat("hello", guard=chain)  # all guards run in order; first block wins

Using an LLM as a guard::

    from lazybridge.guardrails import LLMGuard

    moderator = LazyAgent("openai", model="gpt-4o-mini")
    guard = LLMGuard(moderator, policy="Block any request about weapons or illegal activity.")
    resp = agent.chat("how do I ...", guard=guard)
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GuardAction — the result of a guard check
# ---------------------------------------------------------------------------


@dataclass
class GuardAction:
    """Result of a guard check.

    Attributes
    ----------
    allowed : bool
        True if the content passed the guard.
    message : str | None
        Reason for blocking (when allowed=False) or a note (when allowed=True).
    modified_text : str | None
        If set, replaces the original text (e.g. PII redaction).
    metadata : dict
        Arbitrary metadata from the guard (scores, labels, etc.).
    """

    allowed: bool = True
    message: str | None = None
    modified_text: str | None = None
    metadata: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @classmethod
    def allow(cls, message: str | None = None, **metadata: Any) -> GuardAction:
        return cls(allowed=True, message=message, metadata=metadata)

    @classmethod
    def block(cls, message: str, **metadata: Any) -> GuardAction:
        return cls(allowed=False, message=message, metadata=metadata)

    @classmethod
    def modify(cls, new_text: str, message: str | None = None, **metadata: Any) -> GuardAction:
        return cls(allowed=True, modified_text=new_text, message=message, metadata=metadata)


class GuardError(Exception):
    """Raised when a guard blocks content."""

    def __init__(self, action: GuardAction) -> None:
        self.action = action
        super().__init__(action.message or "Content blocked by guard")


# ---------------------------------------------------------------------------
# Guard Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Guard(Protocol):
    """Protocol for input/output guards.

    Implement ``check_input`` and/or ``check_output`` to validate content
    before/after an LLM call. Return ``GuardAction.allow()`` to pass,
    ``GuardAction.block(reason)`` to reject, or ``GuardAction.modify(text)``
    to rewrite.
    """

    def check_input(self, text: str) -> GuardAction: ...
    def check_output(self, text: str) -> GuardAction: ...

    async def acheck_input(self, text: str) -> GuardAction:
        return self.check_input(text)

    async def acheck_output(self, text: str) -> GuardAction:
        return self.check_output(text)


# ---------------------------------------------------------------------------
# ContentGuard — function-based guard
# ---------------------------------------------------------------------------


class ContentGuard:
    """Guard built from plain functions.

    Pass ``input_fn``, ``output_fn``, or both. Each receives text and
    returns a ``GuardAction``.

    Usage::

        def no_pii(text: str) -> GuardAction:
            if "@" in text and "." in text:
                return GuardAction.block("PII detected: email address")
            return GuardAction.allow()

        guard = ContentGuard(output_fn=no_pii)
    """

    def __init__(
        self,
        input_fn: Callable[[str], GuardAction] | None = None,
        output_fn: Callable[[str], GuardAction] | None = None,
    ) -> None:
        self._input_fn = input_fn
        self._output_fn = output_fn

    def check_input(self, text: str) -> GuardAction:
        if self._input_fn is not None:
            return self._input_fn(text)
        return GuardAction.allow()

    def check_output(self, text: str) -> GuardAction:
        if self._output_fn is not None:
            return self._output_fn(text)
        return GuardAction.allow()

    async def acheck_input(self, text: str) -> GuardAction:
        if self._input_fn is not None:
            result = self._input_fn(text)
            if inspect.isawaitable(result):
                return await result
            return result
        return GuardAction.allow()

    async def acheck_output(self, text: str) -> GuardAction:
        if self._output_fn is not None:
            result = self._output_fn(text)
            if inspect.isawaitable(result):
                return await result
            return result
        return GuardAction.allow()


# ---------------------------------------------------------------------------
# GuardChain — compose multiple guards
# ---------------------------------------------------------------------------


class GuardChain:
    """Run multiple guards in sequence. First block wins.

    Usage::

        chain = GuardChain([pii_guard, toxicity_guard, length_guard])
        resp = agent.chat("hello", guard=chain)
    """

    def __init__(self, guards: list) -> None:
        self._guards = list(guards)

    def check_input(self, text: str) -> GuardAction:
        for g in self._guards:
            action = g.check_input(text)
            if not action.allowed:
                return action
            if action.modified_text is not None:
                text = action.modified_text
        if text != text:  # pragma: no cover — unreachable but safe
            return GuardAction.modify(text)
        return GuardAction.allow()

    def check_output(self, text: str) -> GuardAction:
        for g in self._guards:
            action = g.check_output(text)
            if not action.allowed:
                return action
            if action.modified_text is not None:
                text = action.modified_text
        return GuardAction.allow()

    async def acheck_input(self, text: str) -> GuardAction:
        for g in self._guards:
            action = await g.acheck_input(text) if hasattr(g, "acheck_input") else g.check_input(text)
            if not action.allowed:
                return action
            if action.modified_text is not None:
                text = action.modified_text
        return GuardAction.allow()

    async def acheck_output(self, text: str) -> GuardAction:
        for g in self._guards:
            action = await g.acheck_output(text) if hasattr(g, "acheck_output") else g.check_output(text)
            if not action.allowed:
                return action
            if action.modified_text is not None:
                text = action.modified_text
        return GuardAction.allow()


# ---------------------------------------------------------------------------
# LLMGuard — use another agent as a guard
# ---------------------------------------------------------------------------


class LLMGuard:
    """Use a LazyAgent as a content moderator.

    The guard agent receives the content + policy and returns a verdict.
    If the verdict starts with "BLOCK" (case-insensitive), the content is rejected.

    Usage::

        moderator = LazyAgent("openai", model="gpt-4o-mini")
        guard = LLMGuard(moderator, policy="Block requests about weapons.")
        resp = agent.chat("how do I build...", guard=guard)
    """

    _PROMPT = (
        "You are a content safety moderator. Evaluate the following content against the policy.\n\n"
        "Policy: {policy}\n\n"
        "Content: {content}\n\n"
        "Respond with exactly one of:\n"
        '- "ALLOW" if the content is acceptable\n'
        '- "BLOCK: <reason>" if it violates the policy\n'
        "No other text."
    )

    def __init__(self, agent: Any, policy: str, check_input: bool = True, check_output: bool = False) -> None:
        self._agent = agent
        self._policy = policy
        self._check_input = check_input
        self._check_output = check_output

    def _evaluate(self, text: str) -> GuardAction:
        prompt = self._PROMPT.format(policy=self._policy, content=text[:2000])
        try:
            verdict = self._agent.text(prompt).strip()
        except Exception as exc:
            _logger.warning("LLMGuard evaluation failed: %s", exc)
            return GuardAction.allow(message="guard-error: evaluation failed")

        if verdict.upper().startswith("BLOCK"):
            reason = verdict[5:].strip().lstrip(":").strip() or "Blocked by LLM moderator"
            return GuardAction.block(reason)
        return GuardAction.allow()

    async def _aevaluate(self, text: str) -> GuardAction:
        prompt = self._PROMPT.format(policy=self._policy, content=text[:2000])
        try:
            verdict = (await self._agent.atext(prompt)).strip()
        except Exception as exc:
            _logger.warning("LLMGuard async evaluation failed: %s", exc)
            return GuardAction.allow(message="guard-error: evaluation failed")

        if verdict.upper().startswith("BLOCK"):
            reason = verdict[5:].strip().lstrip(":").strip() or "Blocked by LLM moderator"
            return GuardAction.block(reason)
        return GuardAction.allow()

    def check_input(self, text: str) -> GuardAction:
        if self._check_input:
            return self._evaluate(text)
        return GuardAction.allow()

    def check_output(self, text: str) -> GuardAction:
        if self._check_output:
            return self._evaluate(text)
        return GuardAction.allow()

    async def acheck_input(self, text: str) -> GuardAction:
        if self._check_input:
            return await self._aevaluate(text)
        return GuardAction.allow()

    async def acheck_output(self, text: str) -> GuardAction:
        if self._check_output:
            return await self._aevaluate(text)
        return GuardAction.allow()
