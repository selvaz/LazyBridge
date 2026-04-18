"""lazybridge.supervisor — Human with superpowers in agent pipelines.

A SupervisorAgent gives a human full control inside a pipeline: they can
use tools, inspect the session store, retry previous agents with feedback,
and decide when to continue.

Quick start::

    from lazybridge import LazyAgent, LazyTool, SupervisorAgent, LazySession

    sess = LazySession()
    researcher = LazyAgent("anthropic", name="researcher", tools=[search], session=sess)
    supervisor = SupervisorAgent(
        name="supervisor",
        tools=[search_tool],
        agents=[researcher],
        session=sess,
    )
    writer = LazyAgent("openai", name="writer", session=sess)

    pipeline = LazyTool.chain(researcher, supervisor, writer,
                              name="supervised", description="Research, supervise, write")
    result = pipeline.run({"task": "AI safety report"})

The supervisor gets an interactive REPL::

    [supervisor] Pipeline step
    ──────────────────────────
    Previous output: "Found 3 papers..."

    Commands: continue | retry <agent>: <feedback> | store <key> | <tool>(<args>)

    > search("AI safety 2026")
    Result: "New paper found..."
    > retry researcher: include the 2026 paper
    [researcher re-running...]
    New output: "Found 4 papers..."
    > continue
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from lazybridge.core.types import CompletionResponse, UsageStats

_logger = logging.getLogger(__name__)

_IO_LOCK = threading.Lock()


class SupervisorAgent:
    """Human with superpowers: tools, store access, and agent retry.

    Parameters
    ----------
    name:
        Human-readable name.
    tools:
        LazyTool instances the human can call interactively.
    agents:
        Agents the human can retry with feedback.
    session:
        LazySession for store/events access.
    input_fn:
        Sync input callback. Default: ``input()``.
    ainput_fn:
        Async input callback. Default: runs input_fn in thread.
    timeout:
        Seconds to wait per input. None = forever.
    default:
        Default response on timeout. None = raise TimeoutError.
    """

    _is_human = True

    def __init__(
        self,
        name: str = "supervisor",
        *,
        description: str | None = None,
        input_fn: Callable[[str], str] | None = None,
        ainput_fn: Callable[[str], Awaitable[str]] | None = None,
        tools: list | None = None,
        agents: list | None = None,
        session: Any = None,
        timeout: float | None = None,
        default: str | None = None,
    ) -> None:
        self.name = name
        self.description = description or f"Human supervisor: {name}"
        self._input_fn = input_fn or (lambda prompt: input(prompt))
        self._ainput_fn = ainput_fn
        self._tools = {t.name: t for t in (tools or [])}
        self._agents = {getattr(a, "name", str(i)): a for i, a in enumerate(agents or [])}
        self.session = session
        self.timeout = timeout
        self.default = default

        self.output_schema = None
        self.tools_list: list = list(tools or [])
        self.native_tools: list = []
        self._last_output: str | None = None
        self._last_response: CompletionResponse | None = None

        self._id: str | None = None

        if session:
            session._register_agent(self)

    @property
    def id(self) -> str:
        return self._id if self._id is not None else f"supervisor-{self.name}"

    @id.setter
    def id(self, value: str) -> None:
        self._id = value

    @property
    def tools(self) -> list:
        return self.tools_list

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
                    raise TimeoutError(f"Supervisor input timed out after {self.timeout}s")
                _logger.warning("Supervisor input timed out, using default: %r", self.default)
            return result_holder[0] or ""
        except (KeyboardInterrupt, EOFError):
            if self.default is not None:
                return self.default
            raise

    def _show_header(self, task: str) -> None:
        with _IO_LOCK:
            print(f"\n{'═' * 60}")
            print(f"[{self.name}] Pipeline step — your turn")
            print(f"{'─' * 60}")
            print(f"Previous output:\n  {task[:500]}")
            if self._tools:
                print(f"\nAvailable tools: {', '.join(self._tools.keys())}")
            if self._agents:
                print(f"Retryable agents: {', '.join(self._agents.keys())}")
            if self.session:
                keys = self.session.store.keys()
                if keys:
                    print(f"Store keys: {', '.join(keys[:10])}")
            print("\nCommands: continue | retry <agent>: <feedback> | store <key> | <tool>(<args>)")
            print(f"{'─' * 60}")

    def _try_tool_call(self, user_input: str) -> str | None:
        match = re.match(r"(\w+)\((.+)\)$", user_input.strip())
        if not match:
            return None
        tool_name = match.group(1)
        if tool_name not in self._tools:
            return None
        args_str = match.group(2).strip().strip("'\"")
        tool = self._tools[tool_name]
        try:
            defn = tool.definition() if callable(tool.definition) else tool.definition
            schema = getattr(defn, "parameters", {}) or {}
            if isinstance(schema, dict):
                required = schema.get("required", [])
            else:
                required = getattr(schema, "required", []) or []
            first_param = required[0] if required else "task"
            args = {first_param: args_str}
            result = tool.run(args)
            return str(result)
        except Exception as exc:
            return f"Tool error: {exc}"

    def _parse_retry(self, user_input: str) -> tuple[str, str]:
        rest = user_input[6:].strip()
        if ":" in rest:
            agent_name, feedback = rest.split(":", 1)
            return agent_name.strip(), feedback.strip()
        return rest.strip(), ""

    def _find_agent(self, name: str) -> Any:
        if name in self._agents:
            return self._agents[name]
        for key in self._agents:
            if key.lower() == name.lower():
                return self._agents[key]
        raise ValueError(f"Unknown agent '{name}'. Available: {', '.join(self._agents.keys())}")

    def _run_repl(self, task: str) -> str:
        self._show_header(task)
        last_output = task

        while True:
            user_input = self._get_input(f"[{self.name}] > ").strip()
            if not user_input:
                continue

            if user_input.lower().startswith("continue"):
                custom = user_input[8:].strip().lstrip(":").strip()
                return custom if custom else last_output

            if user_input.lower().startswith("retry "):
                try:
                    agent_name, feedback = self._parse_retry(user_input)
                    agent = self._find_agent(agent_name)
                    prompt = f"{task}\n\nFeedback: {feedback}" if feedback else task
                    has_tools = bool(getattr(agent, "tools", None) or getattr(agent, "native_tools", None))
                    if has_tools and hasattr(agent, "loop"):
                        resp = agent.loop(prompt)
                    else:
                        resp = agent.chat(prompt)
                    last_output = resp.content if hasattr(resp, "content") else str(resp)
                    with _IO_LOCK:
                        print(f"\n[{agent_name} re-run] {last_output[:300]}")
                except Exception as exc:
                    with _IO_LOCK:
                        print(f"Retry failed: {exc}")
                continue

            if user_input.lower().startswith("store "):
                key = user_input[6:].strip()
                if self.session:
                    value = self.session.store.read(key)
                    with _IO_LOCK:
                        print(f"Store[{key}]: {value}")
                else:
                    with _IO_LOCK:
                        print("No session configured — store not available.")
                continue

            tool_result = self._try_tool_call(user_input)
            if tool_result is not None:
                with _IO_LOCK:
                    print(f"Result: {tool_result[:500]}")
                last_output = tool_result
                continue

            with _IO_LOCK:
                print(f"Unknown command: {user_input}")
                print("Available: continue | retry <agent>: <feedback> | store <key> | <tool>(<args>)")

    def _extract_task(self, messages: str | list) -> str:
        if isinstance(messages, str):
            return messages
        for m in messages:
            if hasattr(m, "content"):
                return m.content if isinstance(m.content, str) else str(m.content)
            if isinstance(m, dict):
                return str(m.get("content", ""))
        return str(messages)

    def chat(self, messages: str | list, **kw: Any) -> CompletionResponse:
        task = self._extract_task(messages)
        response = self._run_repl(task)
        self._last_output = response
        resp = CompletionResponse(content=response, usage=UsageStats())
        self._last_response = resp
        return resp

    async def achat(self, messages: str | list, **kw: Any) -> CompletionResponse:
        # Current implementation offloads the sync REPL to a worker thread
        # regardless of whether ``ainput_fn`` is set — the two branches
        # used to be separate in anticipation of a native async REPL but
        # were bit-for-bit identical; collapsed here.  A future async
        # REPL implementation should use ``self._ainput_fn`` when
        # available and await directly on the event loop.
        return await asyncio.to_thread(self.chat, messages, **kw)

    def text(self, messages: str | list, **kw: Any) -> str:
        return self.chat(messages, **kw).content

    async def atext(self, messages: str | list, **kw: Any) -> str:
        return (await self.achat(messages, **kw)).content

    def loop(self, messages: str | list, **kw: Any) -> CompletionResponse:
        return self.chat(messages, **kw)

    async def aloop(self, messages: str | list, **kw: Any) -> CompletionResponse:
        return await self.achat(messages, **kw)

    def as_tool(self, name: str | None = None, description: str | None = None, **kw: Any):
        from lazybridge.lazy_tool import LazyTool

        return LazyTool.from_agent(
            self,  # type: ignore[arg-type]
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
        return "supervisor"

    def __repr__(self) -> str:
        return (
            f"SupervisorAgent(name={self.name!r}, tools={list(self._tools.keys())}, agents={list(self._agents.keys())})"
        )
