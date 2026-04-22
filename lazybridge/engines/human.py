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
        # ``get_running_loop`` is the 3.10+ forward-compatible primitive;
        # ``get_event_loop`` is deprecated and errors on 3.13+ when no
        # loop is already running.  ``prompt`` is always awaited from an
        # active loop, so this is safe.
        loop = asyncio.get_running_loop()
        if self._timeout:
            try:
                fut = loop.run_in_executor(None, input, prompt_str)
                return await asyncio.wait_for(fut, timeout=self._timeout)
            except asyncio.TimeoutError:
                if self._default is not None:
                    print(f"[Timeout — using default: {self._default!r}]")
                    return self._default
                raise
        else:
            return await loop.run_in_executor(None, input, prompt_str)

    async def _prompt_model(self, model_type: type) -> str:
        """Prompt each field of a Pydantic model and coerce via TypeAdapter.

        On ``ValidationError`` the field is re-prompted up to
        ``_MAX_FIELD_RETRIES`` times with the validator's error message
        shown, so a typo (``"abc"`` for an ``int``) is caught
        interactively instead of falling through to a ``BaseModel``
        validation crash at the end of the form.
        """
        import json

        print(f"Please fill in the following fields for {model_type.__name__}:")
        data: dict[str, Any] = {}
        for field_name, field_info in model_type.model_fields.items():
            annotation = field_info.annotation or str
            type_label = getattr(annotation, "__name__", str(annotation))
            data[field_name] = await self._prompt_field(field_name, annotation, type_label)
        return json.dumps(data, default=str)

    _MAX_FIELD_RETRIES = 3

    async def _prompt_field(self, field_name: str, annotation: Any, type_label: str) -> Any:
        from pydantic import ValidationError

        loop = asyncio.get_running_loop()
        last_exc: str | None = None
        for attempt in range(self._MAX_FIELD_RETRIES):
            prefix = f"  {field_name} ({type_label}): "
            if last_exc is not None:
                prefix = f"  [invalid — {last_exc}]\n{prefix}"
            raw = await loop.run_in_executor(None, input, prefix)
            try:
                return self._coerce_field_strict(annotation, raw)
            except ValidationError as exc:
                # Compact single-line summary of the first error.
                err = exc.errors()[0] if exc.errors() else {}
                last_exc = f"{err.get('msg', 'invalid')} ({err.get('type', '?')})"
        # Out of retries — return whatever _coerce_field would have
        # produced in lenient mode (matches the previous behaviour).
        return self._coerce_field(annotation, raw)

    @staticmethod
    def _coerce_field_strict(annotation: Any, raw: str) -> Any:
        """Same as ``_coerce_field`` but re-raises ValidationError.

        Used by the interactive loop so the human is re-prompted with
        a readable error message instead of ending up with a string
        payload that fails Pydantic at the end of the form.
        """
        import json
        from pydantic import TypeAdapter

        origin = getattr(annotation, "__origin__", None)
        args = getattr(annotation, "__args__", ())
        if raw.strip() == "" and type(None) in args:
            return None
        if (origin is list or annotation is list) and not raw.lstrip().startswith("["):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            inner = args[0] if args else str
            return TypeAdapter(list[inner]).validate_python(parts)  # type: ignore[valid-type]
        trimmed = raw.strip()
        if trimmed and trimmed[0] in "{[" or trimmed in ("true", "false", "null"):
            parsed = json.loads(trimmed)
            return TypeAdapter(annotation).validate_python(parsed)
        return TypeAdapter(annotation).validate_python(raw)

    @staticmethod
    def _coerce_field(annotation: Any, raw: str) -> Any:
        """Coerce a CLI-entered string to ``annotation`` via TypeAdapter.

        Fallback order:
        1. Empty string ⇒ ``None`` iff the annotation accepts None.
        2. ``json.loads`` if the raw string looks like JSON (``{``, ``[``,
           ``true``/``false``/``null``, or a number).  Lets users paste
           lists / nested objects verbatim.
        3. Comma-split if the annotation is a list-ish origin and the
           input doesn't start with ``[``.
        4. ``TypeAdapter(annotation).validate_python(raw)`` — Pydantic's
           native coercion (handles int / float / bool / datetime /
           Optional / Union / Enum / ...).
        5. On any failure, fall back to the raw string; the outer
           ``BaseModel(**data)`` will emit a clear ValidationError.
        """
        import json
        from pydantic import TypeAdapter, ValidationError

        # Optional + empty → None.
        origin = getattr(annotation, "__origin__", None)
        args = getattr(annotation, "__args__", ())
        if raw.strip() == "" and type(None) in args:
            return None

        # Comma-list sugar for list[T] when the user didn't type JSON.
        if (origin is list or annotation is list) and not raw.lstrip().startswith("["):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            inner = args[0] if args else str
            try:
                return TypeAdapter(list[inner]).validate_python(parts)  # type: ignore[valid-type]
            except ValidationError:
                return parts

        # Try JSON first for everything that could be complex.
        trimmed = raw.strip()
        if trimmed and trimmed[0] in "{[" or trimmed in ("true", "false", "null"):
            try:
                parsed = json.loads(trimmed)
                return TypeAdapter(annotation).validate_python(parsed)
            except (json.JSONDecodeError, ValidationError, TypeError):
                pass   # fall through to plain-string validation

        # Final: let Pydantic handle the raw string (int "42", bool "yes", …).
        try:
            return TypeAdapter(annotation).validate_python(raw)
        except (ValidationError, TypeError):
            return raw


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
            from pydantic import BaseModel, ValidationError
            import json

            if isinstance(output_type, type) and issubclass(output_type, BaseModel):
                try:
                    data = json.loads(raw) if raw.strip().startswith("{") else {"response": raw}
                    payload = output_type(**data)
                except (json.JSONDecodeError, ValidationError, TypeError) as coerce_exc:
                    # Coercion failed — the raw string is still a usable
                    # payload for non-strict downstream code, but swallow
                    # the error silently would make structured-output
                    # failures indistinguishable from free-text answers.
                    # Record a warning event so the audit trail is honest.
                    if session:
                        session.emit(
                            EventType.TOOL_ERROR,
                            {
                                "agent_name": agent_name,
                                "kind": "structured_output_coercion",
                                "output_type": getattr(output_type, "__name__", str(output_type)),
                                "error_type": type(coerce_exc).__name__,
                                "error": str(coerce_exc),
                            },
                            run_id=run_id,
                        )
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
            # Emit a HIL_DECISION for the single human input so the audit
            # trail distinguishes human answers from LLM responses.
            session.emit(
                EventType.HIL_DECISION,
                {
                    "agent_name": agent_name,
                    "kind": "input",
                    "command": env.task or "",
                    "result": result.text()[:500],
                },
                run_id=run_id,
            )
            session.emit(EventType.AGENT_FINISH, {"agent_name": agent_name, "payload": result.text()}, run_id=run_id)

        if memory:
            task_str = env.task or ""
            memory.add(task_str, result.text())

        return result

    async def stream(self, env: Envelope, *, tools: list, output_type: type, memory: Any, session: Any) -> AsyncIterator[str]:
        env_out = await self.run(env, tools=tools, output_type=output_type, memory=memory, session=session)
        yield env_out.text()
