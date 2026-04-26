"""SupervisorEngine — human-in-the-loop with tool-calling and agent retry.

A supervisor presents the incoming task to a human via a REPL and lets
them do one of four things before returning control to the pipeline:

* ``continue [custom text]`` — accept the task (or override with custom
  text) and return to the caller.
* ``retry <agent>: <feedback>`` — re-run a previously-registered agent
  with optional feedback appended to the task. The new output becomes
  the current output.
* ``store <key>`` — read ``session.store[<key>]`` (requires ``store=``).
* ``<tool>(<args>)`` — invoke one of the registered tools interactively.
  The first required parameter receives the raw argument string.

Usage::

    from lazybridge import Agent, Session, Store, Tool
    from lazybridge.ext.hil import SupervisorEngine

    sess = Session()
    researcher = Agent("claude-opus-4-7", name="researcher", session=sess)
    writer = Agent("claude-opus-4-7", name="writer", session=sess)

    supervisor = Agent(
        engine=SupervisorEngine(
            tools=[search_tool],
            agents=[researcher],
            store=sess_store,
        ),
        name="supervisor",
        session=sess,
    )

    pipeline = Agent.chain(researcher, supervisor, writer)
    pipeline("AI safety report")

The engine is synchronous under the hood (terminal ``input()``) but
exposes the standard async Engine protocol so it can be mixed with LLM
agents in ``Agent.chain`` / ``Plan``.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from lazybridge.envelope import Envelope, EnvelopeMetadata
from lazybridge.session import EventType

if TYPE_CHECKING:
    from lazybridge.memory import Memory
    from lazybridge.session import Session
    from lazybridge.store import Store
    from lazybridge.tools import Tool


_IO_LOCK = threading.Lock()


class SupervisorEngine:
    """Human-in-the-loop engine with tool-calling and agent retry."""

    def __init__(
        self,
        *,
        tools: list[Tool | Callable | Any] | None = None,
        agents: list[Any] | None = None,
        store: Store | None = None,
        input_fn: Callable[[str], str] | None = None,
        ainput_fn: Callable[[str], Awaitable[str]] | None = None,
        timeout: float | None = None,
        default: str | None = None,
    ) -> None:
        # Tool-is-Tool: accept plain functions and Agents too, not just Tool
        # instances.  Matches the contract of ``Agent(tools=[...])`` so the
        # same tools list can be handed to either surface.
        from lazybridge.tools import wrap_tool

        wrapped = [wrap_tool(t) for t in (tools or [])]
        self._tools = {t.name: t for t in wrapped}
        self._agents = {getattr(a, "name", f"agent-{i}"): a for i, a in enumerate(agents or [])}
        self._store = store
        self._input_fn = input_fn or (lambda prompt: input(prompt))
        self._ainput_fn = ainput_fn
        self.timeout = timeout
        self.default = default

    # ------------------------------------------------------------------
    # Engine protocol
    # ------------------------------------------------------------------

    async def run(
        self,
        env: Envelope,
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
    ) -> Envelope:
        # Agents can pass additional tools via tools=; merge with engine-level.
        effective_tools = dict(self._tools)
        for t in tools or []:
            effective_tools.setdefault(t.name, t)

        run_id = str(uuid.uuid4())
        agent_name = getattr(self, "_agent_name", "supervisor")
        t_start = time.monotonic()
        if session:
            session.emit(EventType.AGENT_START, {"agent_name": agent_name, "task": env.task}, run_id=run_id)

        try:
            task_text = env.task or env.text()
            if env.context:
                task_text = f"{task_text}\n\nContext:\n{env.context}"

            # If the caller supplied an ``ainput_fn`` we use a native
            # async REPL — prompts go through the event loop instead of
            # a worker thread, so asyncio cancel scope works correctly.
            # Otherwise fall back to the sync REPL on a worker thread.
            if self._ainput_fn is not None:
                final = await self._run_repl_async(
                    task_text, effective_tools, agent_name,
                    session=session, run_id=run_id,
                )
            else:
                final = await asyncio.to_thread(
                    self._run_repl, task_text, effective_tools, agent_name,
                    session, run_id,
                )
        except Exception as exc:
            if session:
                session.emit(EventType.AGENT_FINISH, {"agent_name": agent_name, "error": str(exc)}, run_id=run_id)
            return Envelope.error_envelope(exc)

        latency_ms = (time.monotonic() - t_start) * 1000
        result = Envelope(
            task=env.task,
            context=env.context,
            payload=final,
            metadata=EnvelopeMetadata(latency_ms=latency_ms, run_id=run_id),
        )
        if session:
            session.emit(EventType.AGENT_FINISH, {"agent_name": agent_name, "payload": final}, run_id=run_id)
        if memory is not None:
            memory.add(env.task or "", final)
        return result

    async def stream(
        self,
        env: Envelope,
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
    ) -> AsyncIterator[str]:
        out = await self.run(env, tools=tools, output_type=output_type, memory=memory, session=session)
        yield out.text()

    # ------------------------------------------------------------------
    # REPL
    # ------------------------------------------------------------------

    def _get_input(self, prompt: str) -> str:
        if self.timeout is None:
            return self._input_fn(prompt)
        holder: list[str | None] = [self.default]

        def _ask() -> None:
            try:
                holder[0] = self._input_fn(prompt)
            except (EOFError, KeyboardInterrupt):
                holder[0] = self.default

        t = threading.Thread(target=_ask, daemon=True)
        t.start()
        t.join(timeout=self.timeout)
        if t.is_alive() and self.default is None:
            raise TimeoutError(f"Supervisor input timed out after {self.timeout}s")
        return holder[0] or ""

    def _show_header(self, task: str, agent_name: str, tools: dict) -> None:
        with _IO_LOCK:
            print(f"\n{'═' * 60}")
            print(f"[{agent_name}] Pipeline step — your turn")
            print(f"{'─' * 60}")
            print(f"Previous output:\n  {task[:500]}")
            if tools:
                print(f"\nAvailable tools: {', '.join(tools.keys())}")
            if self._agents:
                print(f"Retryable agents: {', '.join(self._agents.keys())}")
            if self._store is not None:
                keys = self._store.keys()
                if keys:
                    print(f"Store keys: {', '.join(keys[:10])}")
            print("\nCommands: continue | retry <agent>: <feedback> | store <key> | <tool>(<args>)")
            print(f"{'─' * 60}")

    # Accept ``name(args)`` with any surrounding whitespace and balanced
    # outer parentheses.  Anything between the first ``(`` and the last
    # ``)`` is the raw argument string; internal quoting is preserved so
    # the human can type ``search("AI news")`` or ``calc(a, b)``.
    _TOOL_CALL_RE = re.compile(r"^\s*(\w+)\s*\(\s*(.*?)\s*\)\s*$", re.DOTALL)

    def _try_tool_call(self, user_input: str, tools: dict) -> str | None:
        match = self._TOOL_CALL_RE.match(user_input)
        if not match:
            return None
        tool_name = match.group(1)
        if tool_name not in tools:
            return None
        args_str = match.group(2)
        # Best-effort quote-stripping only when the whole arg is a single
        # quoted token (don't strip from ``"foo", "bar"``).
        stripped = args_str.strip()
        if (
            len(stripped) >= 2
            and stripped[0] == stripped[-1]
            and stripped[0] in ("'", '"')
        ):
            args_str = stripped[1:-1]
        tool = tools[tool_name]
        try:
            defn = tool.definition() if callable(tool.definition) else tool.definition
            params = getattr(defn, "parameters", {}) or {}
            required = params.get("required") if isinstance(params, dict) else getattr(params, "required", None)
            first_param = (required or ["task"])[0]
            # Tool.run_sync expects kwargs; pass the raw arg string under the
            # first required parameter name.
            return str(tool.run_sync(**{first_param: args_str}))
        except Exception as exc:
            return f"Tool error: {exc}"

    def _parse_retry(self, user_input: str) -> tuple[str, str]:
        rest = user_input[6:].strip()
        if ":" in rest:
            name, feedback = rest.split(":", 1)
            return name.strip(), feedback.strip()
        return rest.strip(), ""

    def _find_agent(self, name: str) -> Any:
        if name in self._agents:
            return self._agents[name]
        for key, agent in self._agents.items():
            if key.lower() == name.lower():
                return agent
        raise ValueError(
            f"Unknown agent '{name}'. Available: {', '.join(self._agents.keys()) or '(none)'}"
        )

    def _run_retry(self, agent: Any, task: str, feedback: str) -> str:
        prompt = f"{task}\n\nFeedback: {feedback}" if feedback else task
        env_out = agent(prompt)
        if isinstance(env_out, Envelope):
            return env_out.text()
        if hasattr(env_out, "content"):
            return str(env_out.content)
        return str(env_out)

    def _emit_decision(
        self,
        session: Session | None,
        run_id: str,
        agent_name: str,
        kind: str,
        command: str,
        result: str | None = None,
    ) -> None:
        """Record a HIL decision on the session's event log.

        Kinds: ``continue`` | ``retry`` | ``store`` | ``tool`` |
        ``unknown`` | ``empty``.  Each decision is its own event so
        auditing a multi-step REPL session is just a sequential read
        of the event list.
        """
        if session is None:
            return
        payload = {"agent_name": agent_name, "kind": kind, "command": command}
        if result is not None:
            payload["result"] = result[:500]
        session.emit(EventType.HIL_DECISION, payload, run_id=run_id)

    def _handle_command(
        self,
        user_input: str,
        task: str,
        tools: dict,
        agent_name: str,
        last_output: str,
        session: Session | None,
        run_id: str,
    ) -> tuple[str | None, str]:
        """Process one REPL command.

        Returns ``(final, updated_last_output)`` where ``final`` is
        non-None iff the REPL should exit (the ``continue`` command).
        Everything else (retry / store / tool / unknown) returns
        ``(None, new_last_output)`` and the caller re-prompts.
        """
        if not user_input.strip():
            self._emit_decision(session, run_id, agent_name, "empty", user_input)
            return None, last_output

        lower = user_input.lower()

        if lower.startswith("continue"):
            custom = user_input[8:].strip().lstrip(":").strip()
            final = custom if custom else last_output
            self._emit_decision(session, run_id, agent_name, "continue", user_input, final)
            return final, last_output

        if lower.startswith("retry "):
            try:
                name, feedback = self._parse_retry(user_input)
                agent = self._find_agent(name)
                new_output = self._run_retry(agent, task, feedback)
                with _IO_LOCK:
                    print(f"\n[{name} re-run] {new_output[:300]}")
                self._emit_decision(session, run_id, agent_name, "retry", user_input, new_output)
                return None, new_output
            except Exception as exc:
                with _IO_LOCK:
                    print(f"Retry failed: {exc}")
                self._emit_decision(session, run_id, agent_name, "retry", user_input, f"error: {exc}")
                return None, last_output

        if lower.startswith("store "):
            key = user_input[6:].strip()
            if self._store is None:
                with _IO_LOCK:
                    print("No store configured — pass store= to SupervisorEngine.")
                self._emit_decision(session, run_id, agent_name, "store", user_input, "no store")
            else:
                value = self._store.read(key)
                with _IO_LOCK:
                    print(f"Store[{key}]: {value}")
                self._emit_decision(session, run_id, agent_name, "store", user_input, str(value))
            return None, last_output

        tool_result = self._try_tool_call(user_input, tools)
        if tool_result is not None:
            with _IO_LOCK:
                print(f"Result: {tool_result[:500]}")
            self._emit_decision(session, run_id, agent_name, "tool", user_input, tool_result)
            return None, tool_result

        with _IO_LOCK:
            print(f"Unknown command: {user_input}")
            print("Available: continue | retry <agent>: <feedback> | store <key> | <tool>(<args>)")
        self._emit_decision(session, run_id, agent_name, "unknown", user_input)
        return None, last_output

    def _run_repl(
        self,
        task: str,
        tools: dict,
        agent_name: str,
        session: Session | None = None,
        run_id: str = "",
    ) -> str:
        self._show_header(task, agent_name, tools)
        last_output = task

        while True:
            user_input = self._get_input(f"[{agent_name}] > ").strip()
            final, last_output = self._handle_command(
                user_input, task, tools, agent_name, last_output, session, run_id,
            )
            if final is not None:
                return final

    async def _run_repl_async(
        self,
        task: str,
        tools: dict,
        agent_name: str,
        *,
        session: Session | None,
        run_id: str,
    ) -> str:
        """Event-loop-native REPL — used when ``ainput_fn`` was supplied.

        Blocks cooperatively on ``ainput_fn`` instead of occupying a
        worker thread, so asyncio cancellation propagates naturally.
        """
        self._show_header(task, agent_name, tools)
        last_output = task
        # ``assert`` is disabled by ``-O`` / ``PYTHONOPTIMIZE=1`` — raise
        # explicitly so the guard survives optimised production deployments.
        if self._ainput_fn is None:
            raise RuntimeError(
                "_run_repl_async called without ainput_fn — this is a "
                "programming error; only call this path when ainput_fn was "
                "supplied to SupervisorEngine.__init__."
            )

        while True:
            user_input = (await self._ainput_fn(f"[{agent_name}] > ")).strip()
            final, last_output = self._handle_command(
                user_input, task, tools, agent_name, last_output, session, run_id,
            )
            if final is not None:
                return final
