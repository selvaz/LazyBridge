"""HumanEngine — human-in-the-loop engine with terminal and web UI."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Callable, Literal

from lazybridge.envelope import Envelope, EnvelopeMetadata, ErrorInfo
from lazybridge.session import EventType

if TYPE_CHECKING:
    from lazybridge.memory import Memory
    from lazybridge.session import Session
    from lazybridge.tools import Tool


class _UIProtocol:
    """Minimal protocol for custom UI adapters."""

    async def prompt(self, task: str, *, tools: list[Any], output_type: type) -> str:
        raise NotImplementedError


class _TerminalUI(_UIProtocol):
    def __init__(self, timeout: float | None = None, default: str | None = None) -> None:
        self._timeout = timeout
        self._default = default

    async def prompt(self, task: str, *, tools: list[Any], output_type: type) -> str:
        from pydantic import BaseModel

        print(f"\n[Human Input Required]\n{task}")

        if tools:
            tool_names = [t.name for t in tools]
            print(f"Available actions: {', '.join(tool_names)}")

        if issubclass(output_type, BaseModel) if isinstance(output_type, type) else False:
            return await self._prompt_model(output_type)

        prompt_str = "Your response: "
        if self._timeout:
            try:
                loop = asyncio.get_event_loop()
                fut = loop.run_in_executor(None, input, prompt_str)
                return await asyncio.wait_for(fut, timeout=self._timeout)
            except asyncio.TimeoutError:
                if self._default is not None:
                    print(f"[Timeout — using default: {self._default!r}]")
                    return self._default
                raise
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, input, prompt_str)

    async def _prompt_model(self, model_type: type) -> str:
        import json
        from pydantic import BaseModel

        print(f"Please fill in the following fields for {model_type.__name__}:")
        data: dict[str, Any] = {}
        for field_name, field_info in model_type.model_fields.items():
            annotation = field_info.annotation or str
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(None, input, f"  {field_name} ({annotation.__name__ if hasattr(annotation, '__name__') else str(annotation)}): ")
            try:
                if annotation is int:
                    data[field_name] = int(raw)
                elif annotation is float:
                    data[field_name] = float(raw)
                elif annotation is bool:
                    data[field_name] = raw.lower() in ("yes", "true", "1", "y")
                elif annotation is list or (hasattr(annotation, "__origin__") and annotation.__origin__ is list):
                    data[field_name] = [x.strip() for x in raw.split(",")]
                else:
                    data[field_name] = raw
            except (ValueError, TypeError):
                data[field_name] = raw
        return json.dumps(data)


class HumanEngine:
    """Presents the task to a human and returns their response as an Envelope.

    With output=PydanticModel, terminal prompts each field; web renders a form.
    Emits the same 8 event types as LLMEngine for transparent observability.
    """

    def __init__(
        self,
        *,
        timeout: float | None = None,
        ui: Literal["terminal", "web"] | _UIProtocol = "terminal",
        default: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.default = default
        if isinstance(ui, str):
            if ui == "terminal":
                self._ui: _UIProtocol = _TerminalUI(timeout=timeout, default=default)
            elif ui == "web":
                raise NotImplementedError("Web UI is not yet implemented — use ui='terminal'")
            else:
                raise ValueError(f"Unknown UI type: {ui!r}")
        else:
            self._ui = ui

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
        agent_name = getattr(self, "_agent_name", "human")

        if session:
            session.emit(EventType.AGENT_START, {"agent_name": agent_name, "task": env.task}, run_id=run_id)

        try:
            task_text = env.task or env.text()
            if env.context:
                task_text = f"{task_text}\n\nContext:\n{env.context}"

            raw = await self._ui.prompt(task_text, tools=tools, output_type=output_type)

            payload: Any = raw
            from pydantic import BaseModel
            import json

            if isinstance(output_type, type) and issubclass(output_type, BaseModel):
                try:
                    data = json.loads(raw) if raw.strip().startswith("{") else {"response": raw}
                    payload = output_type(**data)
                except Exception:
                    payload = raw

        except Exception as exc:
            error_env = Envelope.error_envelope(exc)
            if session:
                session.emit(EventType.AGENT_FINISH, {"agent_name": agent_name, "error": str(exc)}, run_id=run_id)
            return error_env

        latency_ms = (time.monotonic() - t_start) * 1000
        result = Envelope(
            task=env.task,
            context=env.context,
            payload=payload,
            metadata=EnvelopeMetadata(latency_ms=latency_ms, run_id=run_id),
        )

        if session:
            session.emit(EventType.AGENT_FINISH, {"agent_name": agent_name, "payload": result.text()}, run_id=run_id)

        if memory:
            task_str = env.task or ""
            memory.add(task_str, result.text())

        return result

    async def stream(self, env: Envelope, *, tools: list, output_type: type, memory: Any, session: Any) -> AsyncIterator[str]:
        env_out = await self.run(env, tools=tools, output_type=output_type, memory=memory, session=session)
        yield env_out.text()
