"""lazybridge.human — Human participant in agent pipelines.

A HumanAgent wraps a callback (default: ``input()``) so a human can be
used anywhere a LazyAgent is expected: in chains, as tools, in parallel,
or as a verify judge.

Quick start::

    from lazybridge import HumanAgent

    human = HumanAgent(name="reviewer")
    resp = human.chat("Review this report and provide feedback")
    print(resp.content)  # whatever the human typed

In a pipeline::

    from lazybridge import LazyAgent, LazyTool, HumanAgent

    researcher = LazyAgent("anthropic", name="researcher")
    human = HumanAgent(name="reviewer")
    writer = LazyAgent("openai", name="writer")

    pipeline = LazyTool.chain(researcher, human, writer,
                              name="reviewed_pipeline", description="Research, review, write")
    result = pipeline.run({"task": "AI safety report"})

Dialogue mode::

    human = HumanAgent(name="reviewer", mode="dialogue", end_token="done")
    # Human can go back and forth until typing "done"
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from lazybridge.core.types import CompletionResponse, UsageStats

_logger = logging.getLogger(__name__)

_IO_LOCK = threading.Lock()


class HumanAgent:
    """A human participant in agent pipelines.

    Satisfies the same duck-type interface as LazyAgent, so it works in
    chains, parallel panels, as tools, and as verify judges.

    Parameters
    ----------
    name:
        Human-readable name (shown in prompts and graph).
    description:
        Description when exposed as a tool via as_tool().
    input_fn:
        Sync callback ``(prompt: str) -> str``. Default: formatted ``input()``.
    ainput_fn:
        Async callback. Default: runs input_fn in a thread.
    mode:
        ``"single"`` (default) — one response per turn.
        ``"dialogue"`` — multi-turn until human types end_token.
    end_token:
        Word that ends dialogue mode. Default: ``"done"``.
    prompt_template:
        Format string with ``{task}`` placeholder.
    timeout:
        Seconds to wait for input. None = wait forever.
    default:
        Response returned on timeout. None = raise TimeoutError.
    session:
        Optional LazySession for tracking.
    """

    _is_human = True

    def __init__(
        self,
        name: str = "human",
        *,
        description: str | None = None,
        input_fn: Callable[[str], str] | None = None,
        ainput_fn: Callable[[str], Awaitable[str]] | None = None,
        mode: Literal["single", "dialogue"] = "single",
        end_token: str = "done",
        prompt_template: str = "{task}",
        timeout: float | None = None,
        default: str | None = None,
        session: Any = None,
    ) -> None:
        self.name = name
        self.description = description or f"Human input from {name}"
        self._input_fn = input_fn or self._default_input
        self._ainput_fn = ainput_fn
        self._mode = mode
        self._end_token = end_token.lower()
        self._prompt_template = prompt_template
        self.timeout = timeout
        self.default = default
        self.session = session

        self.output_schema = None
        self.tools: list = []
        self.native_tools: list = []
        self._last_output: str | None = None
        self._last_response: CompletionResponse | None = None

        if session:
            session._register_agent(self)

    @property
    def id(self) -> str:
        return f"human-{self.name}"

    def _default_input(self, prompt: str) -> str:
        with _IO_LOCK:
            print(f"\n{'─' * 60}")
            print(f"[{self.name}] Input requested:")
            print(f"{'─' * 60}")
            print(prompt)
            print(f"{'─' * 60}")
            return input("Your response: ")

    def _get_input(self, prompt: str) -> str:
        try:
            if self.timeout is None:
                return self._input_fn(prompt)
            result_holder: list[str | None] = [self.default]

            def _ask():
                result_holder[0] = self._input_fn(prompt)

            t = threading.Thread(target=_ask, daemon=True)
            t.start()
            t.join(timeout=self.timeout)
            if t.is_alive():
                if self.default is None:
                    raise TimeoutError(f"Human input timed out after {self.timeout}s")
                _logger.warning("Human input timed out, using default: %r", self.default)
            return result_holder[0] or ""
        except (KeyboardInterrupt, EOFError):
            if self.default is not None:
                return self.default
            raise

    def _extract_task(self, messages: str | list) -> str:
        if isinstance(messages, str):
            return messages
        for m in messages:
            if hasattr(m, "content"):
                return m.content if isinstance(m.content, str) else str(m.content)
            if isinstance(m, dict):
                return str(m.get("content", ""))
        return str(messages)

    def _run_single(self, task: str) -> str:
        prompt = self._prompt_template.format(task=task)
        return self._get_input(prompt)

    def _run_dialogue(self, task: str) -> str:
        prompt = self._prompt_template.format(task=task)
        with _IO_LOCK:
            print(f"\n{'─' * 60}")
            print(f"[{self.name}] Dialogue mode (type '{self._end_token}' to finish):")
            print(f"{'─' * 60}")
            print(prompt)
            print(f"{'─' * 60}")
        turns: list[str] = []
        while True:
            response = self._get_input(f"[{self.name}] > ")
            if response.strip().lower() == self._end_token:
                break
            turns.append(response)
        return "\n".join(turns) if turns else task

    def chat(self, messages: str | list, **kw: Any) -> CompletionResponse:
        task = self._extract_task(messages)
        if self._mode == "dialogue":
            response = self._run_dialogue(task)
        else:
            response = self._run_single(task)
        self._last_output = response
        resp = CompletionResponse(content=response, usage=UsageStats())
        self._last_response = resp
        return resp

    async def achat(self, messages: str | list, **kw: Any) -> CompletionResponse:
        if self._ainput_fn is not None:
            task = self._extract_task(messages)
            prompt = self._prompt_template.format(task=task)
            response = await self._ainput_fn(prompt)
            self._last_output = response
            resp = CompletionResponse(content=response, usage=UsageStats())
            self._last_response = resp
            return resp
        return await asyncio.to_thread(self.chat, messages, **kw)

    def text(self, messages: str | list, **kw: Any) -> str:
        return self.chat(messages, **kw).content

    async def atext(self, messages: str | list, **kw: Any) -> str:
        return (await self.achat(messages, **kw)).content

    def loop(self, messages: str | list, **kw: Any) -> CompletionResponse:
        return self.chat(messages, **kw)

    async def aloop(self, messages: str | list, **kw: Any) -> CompletionResponse:
        return await self.achat(messages, **kw)

    def as_tool(
        self,
        name: str | None = None,
        description: str | None = None,
        **kw: Any,
    ):
        from lazybridge.lazy_tool import LazyTool

        return LazyTool.from_agent(
            self,
            name=name or self.name,
            description=description or self.description,
            **kw,
        )

    @property
    def result(self) -> Any:
        return self._last_output

    @property
    def _provider_name(self) -> str:
        return "human"

    @property
    def _model_name(self) -> str:
        return "human"

    def __repr__(self) -> str:
        return f"HumanAgent(name={self.name!r}, mode={self._mode!r})"
