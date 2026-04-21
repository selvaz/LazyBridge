"""Guardrails — input/output filtering with block/allow/modify semantics."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GuardAction:
    allowed: bool = True
    message: str | None = None
    modified_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls, message: str | None = None, **metadata: Any) -> "GuardAction":
        return cls(allowed=True, message=message, metadata=metadata)

    @classmethod
    def block(cls, message: str, **metadata: Any) -> "GuardAction":
        return cls(allowed=False, message=message, metadata=metadata)

    @classmethod
    def modify(cls, new_text: str, message: str | None = None, **metadata: Any) -> "GuardAction":
        return cls(allowed=True, modified_text=new_text, message=message, metadata=metadata)


class Guard:
    """Base guard. Override check_input and/or check_output."""

    def check_input(self, text: str) -> GuardAction:
        return GuardAction.allow()

    def check_output(self, text: str) -> GuardAction:
        return GuardAction.allow()

    async def acheck_input(self, text: str) -> GuardAction:
        return self.check_input(text)

    async def acheck_output(self, text: str) -> GuardAction:
        return self.check_output(text)


class ContentGuard(Guard):
    """Function-based guard."""

    def __init__(
        self,
        input_fn: Callable[[str], GuardAction] | None = None,
        output_fn: Callable[[str], GuardAction] | None = None,
    ) -> None:
        self._input_fn = input_fn
        self._output_fn = output_fn

    def check_input(self, text: str) -> GuardAction:
        return self._input_fn(text) if self._input_fn else GuardAction.allow()

    def check_output(self, text: str) -> GuardAction:
        return self._output_fn(text) if self._output_fn else GuardAction.allow()


class GuardChain(Guard):
    """Run multiple guards in sequence; first block wins."""

    def __init__(self, *guards: Guard) -> None:
        self._guards = list(guards)

    def check_input(self, text: str) -> GuardAction:
        for g in self._guards:
            action = g.check_input(text)
            if not action.allowed:
                return action
            if action.modified_text is not None:
                text = action.modified_text
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
            action = await g.acheck_input(text)
            if not action.allowed:
                return action
            if action.modified_text is not None:
                text = action.modified_text
        return GuardAction.allow()

    async def acheck_output(self, text: str) -> GuardAction:
        for g in self._guards:
            action = await g.acheck_output(text)
            if not action.allowed:
                return action
            if action.modified_text is not None:
                text = action.modified_text
        return GuardAction.allow()


class LLMGuard(Guard):
    """Use an Agent as a judge. Returns block if judge says 'block' or 'deny'."""

    def __init__(self, agent: Any, policy: str = "block harmful content") -> None:
        self._agent = agent
        self._policy = policy

    def check_input(self, text: str) -> GuardAction:
        verdict = self._agent(f"Policy: {self._policy}\nContent: {text}\nVerdict (allow/block):").text()
        if "block" in verdict.lower() or "deny" in verdict.lower():
            return GuardAction.block(f"LLMGuard blocked: {verdict}")
        return GuardAction.allow()

    def check_output(self, text: str) -> GuardAction:
        return self.check_input(text)
