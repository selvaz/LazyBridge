"""LLMEngine — agentic tool-calling loop over any provider."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal

from lazybridge.core.executor import Executor
from lazybridge.core.types import (
    CompletionRequest,
    CompletionResponse,
    Message,
    Role,
    StructuredOutputConfig,
    ToolCall,
)
from lazybridge.envelope import Envelope, EnvelopeMetadata, ErrorInfo
from lazybridge.session import EventType

if TYPE_CHECKING:
    from lazybridge.memory import Memory
    from lazybridge.session import Session
    from lazybridge.tools import Tool


class LLMEngine:
    """Drives the LLM ↔ tool-call loop for a single agent invocation.

    Wraps the provider-agnostic Executor and manages:
    - Multi-turn agentic loop with configurable max_turns
    - Parallel tool execution (tool_choice="parallel")
    - Structured output via provider API (output != str → Pydantic model)
    - Memory message injection
    - Session event emission (8 event types)
    """

    def __init__(
        self,
        model: str,
        *,
        thinking: bool = False,
        max_turns: int = 10,
        tool_choice: Literal["auto", "any", "parallel"] = "auto",
        temperature: float | None = None,
        system: str | None = None,
    ) -> None:
        self.model = model
        self.thinking = thinking
        self.max_turns = max_turns
        self.tool_choice = tool_choice
        self.temperature = temperature
        self.system = system
        self._executor: Executor | None = None

    def _get_executor(self) -> Executor:
        if self._executor is None:
            self._executor = Executor(self.model, model=None)
        return self._executor

    @staticmethod
    def _infer_provider(model: str) -> str:
        m = model.lower()
        if "claude" in m:
            return "anthropic"
        if "gpt" in m or "o1" in m or "o3" in m:
            return "openai"
        if "gemini" in m:
            return "google"
        if "deepseek" in m:
            return "deepseek"
        return "anthropic"

    def _make_executor(self) -> Executor:
        provider = self._infer_provider(self.model)
        return Executor(provider, model=self.model)

    async def run(
        self,
        env: Envelope,
        *,
        tools: list["Tool"],
        output_type: type,
        memory: "Memory | None",
        session: "Session | None",
    ) -> Envelope:
        run_id = str(uuid.uuid4())
        t_start = time.monotonic()
        agent_name = getattr(self, "_agent_name", "agent")

        if session:
            session.emit(EventType.AGENT_START, {"agent_name": agent_name, "task": env.task}, run_id=run_id)

        try:
            result = await self._loop(env, tools=tools, output_type=output_type, memory=memory, session=session, run_id=run_id)
        except Exception as exc:
            error_env = Envelope.error_envelope(exc)
            if session:
                session.emit(EventType.AGENT_FINISH, {"agent_name": agent_name, "error": str(exc)}, run_id=run_id)
            return error_env

        latency_ms = (time.monotonic() - t_start) * 1000
        result.metadata.latency_ms = latency_ms
        result.metadata.run_id = run_id

        if session:
            session.emit(
                EventType.AGENT_FINISH,
                {"agent_name": agent_name, "payload": result.text(), "latency_ms": latency_ms},
                run_id=run_id,
            )

        return result

    async def _loop(
        self,
        env: Envelope,
        *,
        tools: list["Tool"],
        output_type: type,
        memory: "Memory | None",
        session: "Session | None",
        run_id: str,
    ) -> Envelope:
        from lazybridge.core.types import ThinkingConfig
        from pydantic import BaseModel

        executor = self._make_executor()

        # Build initial messages
        messages: list[Message] = []
        if memory:
            messages.extend(memory.messages())

        # Inject context as system or user message
        system = self.system or ""
        if env.context:
            system = f"{system}\n\nContext:\n{env.context}".strip() if system else f"Context:\n{env.context}"

        task_text = env.task or env.text()
        messages.append(Message(role=Role.USER, content=task_text))

        tool_defs = [t.definition() for t in tools]
        tool_map = {t.name: t for t in tools}

        structured_cfg: StructuredOutputConfig | None = None
        if output_type is not str and isinstance(output_type, type):
            structured_cfg = StructuredOutputConfig(schema=output_type)

        thinking_cfg = ThinkingConfig(enabled=True) if self.thinking else None

        total_in = total_out = 0
        cost = 0.0
        model_used: str | None = None

        for turn in range(self.max_turns):
            if session:
                session.emit(EventType.LOOP_STEP, {"turn": turn, "messages": len(messages)}, run_id=run_id)

            req = CompletionRequest(
                messages=messages,
                system=system or None,
                temperature=self.temperature,
                tools=tool_defs,
                tool_choice=self.tool_choice if tool_defs else None,
                structured_output=structured_cfg if not tool_defs else None,
                thinking=thinking_cfg,
            )

            if session:
                session.emit(
                    EventType.MODEL_REQUEST,
                    {"provider": executor._provider.__class__.__name__, "model": self.model, "turns": turn},
                    run_id=run_id,
                )

            resp: CompletionResponse = await executor.aexecute(req)
            model_used = resp.model or self.model
            total_in += resp.usage.input_tokens
            total_out += resp.usage.output_tokens
            if resp.usage.cost_usd:
                cost += resp.usage.cost_usd

            if session:
                session.emit(
                    EventType.MODEL_RESPONSE,
                    {
                        "content": resp.content[:500],
                        "input_tokens": resp.usage.input_tokens,
                        "output_tokens": resp.usage.output_tokens,
                        "cost_usd": resp.usage.cost_usd,
                        "stop_reason": resp.stop_reason,
                    },
                    run_id=run_id,
                )

            if not resp.tool_calls:
                # Done — build final Envelope
                payload: Any
                if structured_cfg and resp.parsed:
                    payload = resp.parsed
                elif structured_cfg and not tool_defs:
                    # Try parsing content as structured output
                    payload = resp.parsed or resp.content
                else:
                    payload = resp.content

                mem_text = memory.text() if memory else ""
                if memory:
                    memory.add(task_text, resp.content, tokens=total_in + total_out)

                return Envelope(
                    task=env.task,
                    context=env.context,
                    payload=payload,
                    metadata=EnvelopeMetadata(
                        input_tokens=total_in,
                        output_tokens=total_out,
                        cost_usd=cost,
                        model=model_used,
                        provider=executor._provider.__class__.__name__,
                    ),
                )

            # Append assistant message with tool calls
            from lazybridge.core.types import TextContent, ToolUseContent

            assistant_blocks: list[Any] = []
            if resp.content:
                assistant_blocks.append(TextContent(text=resp.content))
            for tc in resp.tool_calls:
                assistant_blocks.append(ToolUseContent(id=tc.id, name=tc.name, input=tc.arguments))
            messages.append(Message(role=Role.ASSISTANT, content=assistant_blocks))

            # Execute tool calls (parallel if tool_choice="parallel", else sequential)
            if self.tool_choice == "parallel":
                results = await asyncio.gather(
                    *[self._exec_tool(tc, tool_map, session=session, run_id=run_id) for tc in resp.tool_calls],
                    return_exceptions=True,
                )
                tool_results = list(results)
            else:
                tool_results = []
                for tc in resp.tool_calls:
                    res = await self._exec_tool(tc, tool_map, session=session, run_id=run_id)
                    tool_results.append(res)

            # Append tool results
            from lazybridge.core.types import ToolResultContent

            result_blocks: list[Any] = []
            for tc, tr in zip(resp.tool_calls, tool_results):
                is_err = isinstance(tr, Exception)
                result_blocks.append(
                    ToolResultContent(
                        tool_use_id=tc.id,
                        content=str(tr),
                        tool_name=tc.name,
                        is_error=is_err,
                    )
                )
            messages.append(Message(role=Role.USER, content=result_blocks))

        # max_turns exhausted
        return Envelope(
            task=env.task,
            payload=messages[-1].to_text() if messages else "",
            error=ErrorInfo(type="MaxTurnsExceeded", message=f"Reached max_turns={self.max_turns}", retryable=False),
            metadata=EnvelopeMetadata(input_tokens=total_in, output_tokens=total_out, cost_usd=cost, model=model_used),
        )

    async def _exec_tool(
        self,
        tc: ToolCall,
        tool_map: dict[str, "Tool"],
        *,
        session: "Session | None",
        run_id: str,
    ) -> Any:
        if session:
            session.emit(EventType.TOOL_CALL, {"tool": tc.name, "arguments": tc.arguments}, run_id=run_id)

        tool = tool_map.get(tc.name)
        if tool is None:
            err = RuntimeError(f"Unknown tool: {tc.name!r}")
            if session:
                session.emit(EventType.TOOL_ERROR, {"tool": tc.name, "error": str(err)}, run_id=run_id)
            return err

        try:
            result = await tool.run(**tc.arguments)
            if session:
                session.emit(EventType.TOOL_RESULT, {"tool": tc.name, "result": str(result)[:500]}, run_id=run_id)
            return result
        except Exception as exc:
            if session:
                session.emit(EventType.TOOL_ERROR, {"tool": tc.name, "error": str(exc), "type": type(exc).__name__}, run_id=run_id)
            return exc

    async def stream(
        self,
        env: Envelope,
        *,
        tools: list["Tool"],
        output_type: type,
        memory: "Memory | None",
        session: "Session | None",
    ) -> AsyncIterator[str]:
        executor = self._make_executor()
        task_text = env.task or env.text()
        system = self.system or ""
        if env.context:
            system = f"{system}\n\nContext:\n{env.context}".strip() if system else f"Context:\n{env.context}"

        messages: list[Message] = []
        if memory:
            messages.extend(memory.messages())
        messages.append(Message(role=Role.USER, content=task_text))

        req = CompletionRequest(
            messages=messages,
            system=system or None,
            temperature=self.temperature,
            stream=True,
        )

        async for chunk in executor.astream(req):
            if chunk.delta:
                yield chunk.delta
