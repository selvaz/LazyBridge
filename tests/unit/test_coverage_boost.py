"""Coverage-targeted tests for exporters.py, human.py, structured.py,
tool_schema.py, agent.py, and engines/llm.py."""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from io import StringIO
from pathlib import Path

import pytest
from pydantic import BaseModel

from lazybridge import Agent, Envelope, Session, Tool
from lazybridge.core.structured import (
    _enum_match,
    _validate_schema,
    _validate_schema_subset,
    normalize_json_schema,
    validate_payload_against_output_type,
)
from lazybridge.exporters import (
    CallbackExporter,
    ConsoleExporter,
    FilteredExporter,
    JsonFileExporter,
    StructuredLogExporter,
)
from lazybridge.ext.hil.human import (
    _TerminalUI,
)

# =============================================================================
# exporters.py
# =============================================================================


class TestExporters:
    def test_callback_exporter(self):
        received = []
        exp = CallbackExporter(fn=received.append)
        exp.export({"event_type": "test", "value": 1})
        assert received == [{"event_type": "test", "value": 1}]

    def test_filtered_exporter_passes_matching(self):
        received = []
        inner = CallbackExporter(fn=received.append)
        exp = FilteredExporter(inner=inner, event_types={"agent_start"})
        exp.export({"event_type": "agent_start", "x": 1})
        assert len(received) == 1

    def test_filtered_exporter_blocks_non_matching(self):
        received = []
        inner = CallbackExporter(fn=received.append)
        exp = FilteredExporter(inner=inner, event_types={"agent_start"})
        exp.export({"event_type": "tool_call", "x": 1})
        assert len(received) == 0

    def test_structured_log_exporter(self, caplog):
        exp = StructuredLogExporter(logger_name="lazybridge.test")
        with caplog.at_level(logging.INFO, logger="lazybridge.test"):
            exp.export({"event_type": "agent_start", "agent_name": "x"})
        assert any("agent_start" in r.message for r in caplog.records)

    def test_json_file_exporter(self):
        with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            exp = JsonFileExporter(path=path)
            exp.export({"event_type": "test", "a": 1})
            exp.export({"event_type": "test2", "b": 2})
            exp.close()
            lines = Path(path).read_text().strip().split("\n")
            assert len(lines) == 2
            assert json.loads(lines[0])["event_type"] == "test"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_json_file_exporter_close_idempotent(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            exp = JsonFileExporter(path=path)
            exp.close()
            exp.close()  # second call should not raise
        finally:
            Path(path).unlink(missing_ok=True)

    def test_console_exporter_output(self):
        stream = StringIO()
        exp = ConsoleExporter(stream=stream)
        exp.export({"event_type": "agent_start", "agent_name": "bob", "task": "test"})
        output = stream.getvalue()
        assert "agent_start" in output
        assert "bob" in output

    def test_console_exporter_truncates_long_values(self):
        stream = StringIO()
        exp = ConsoleExporter(stream=stream)
        long_val = "x" * 200
        exp.export({"event_type": "test", "key": long_val})
        output = stream.getvalue()
        assert "..." in output

    def test_console_exporter_skips_none_values(self):
        stream = StringIO()
        exp = ConsoleExporter(stream=stream)
        exp.export({"event_type": "test", "key": None, "other": ""})
        output = stream.getvalue()
        assert "key" not in output

    def test_console_exporter_no_agent_name(self):
        stream = StringIO()
        exp = ConsoleExporter(stream=stream)
        exp.export({"event_type": "test"})
        output = stream.getvalue()
        assert "test" in output

    def test_session_uses_callback_exporter(self):
        events = []
        sess = Session(exporters=[CallbackExporter(fn=events.append)])
        from lazybridge.session import EventType

        sess.emit(EventType.AGENT_START, {"agent_name": "a"}, run_id="r1")
        assert any(e.get("event_type") == "agent_start" for e in events)


# =============================================================================
# human.py — _TerminalUI
# =============================================================================


class TestTerminalUI:
    @pytest.mark.asyncio
    async def test_prompt_returns_input(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "user answer")
        ui = _TerminalUI()
        result = await ui.prompt("task", tools=[], output_type=str)
        assert result == "user answer"

    @pytest.mark.asyncio
    async def test_prompt_with_timeout_default(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "answer")
        ui = _TerminalUI(timeout=5.0)
        result = await ui.prompt("task", tools=[], output_type=str)
        assert result == "answer"

    @pytest.mark.asyncio
    async def test_prompt_shows_tools(self, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda _: "x")
        t = Tool(lambda: None, name="my_tool")
        ui = _TerminalUI()
        await ui.prompt("task", tools=[t], output_type=str)
        captured = capsys.readouterr()
        assert "my_tool" in captured.out

    @pytest.mark.asyncio
    async def test_prompt_pydantic_model_simple(self, monkeypatch):
        responses = iter(["Alice", "42"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))

        class UserForm(BaseModel):
            name: str
            age: int

        ui = _TerminalUI()
        result = await ui.prompt("Fill form", tools=[], output_type=UserForm)
        data = json.loads(result)
        assert data["name"] == "Alice"
        assert data["age"] == 42

    @pytest.mark.asyncio
    async def test_coerce_field_strict_none_optional(self):
        from lazybridge.ext.hil.human import _TerminalUI

        _TerminalUI()
        result = _TerminalUI._coerce_field_strict(int | None, "")
        assert result is None

    @pytest.mark.asyncio
    async def test_coerce_field_strict_list_comma_separated(self):
        result = _TerminalUI._coerce_field_strict(list[str], "a, b, c")
        assert result == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_coerce_field_strict_json(self):
        result = _TerminalUI._coerce_field_strict(dict, '{"x": 1}')
        assert result == {"x": 1}

    @pytest.mark.asyncio
    async def test_coerce_field_fallback(self):
        result = _TerminalUI._coerce_field(str, "hello")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_coerce_field_optional_empty(self):
        result = _TerminalUI._coerce_field(str | None, "")
        assert result is None

    @pytest.mark.asyncio
    async def test_coerce_field_list_comma(self):
        result = _TerminalUI._coerce_field(list[int], "1, 2, 3")
        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_coerce_field_json_object(self):
        result = _TerminalUI._coerce_field(dict, '{"key": "val"}')
        assert result == {"key": "val"}

    @pytest.mark.asyncio
    async def test_coerce_field_bool(self):
        result = _TerminalUI._coerce_field(bool, "true")
        assert result is True

    @pytest.mark.asyncio
    async def test_coerce_field_fallback_on_bad_json(self):
        result = _TerminalUI._coerce_field(str, "{bad}")
        assert result == "{bad}"


# =============================================================================
# core/structured.py
# =============================================================================


class TestValidatePayload:
    def test_str_passthrough(self):
        assert validate_payload_against_output_type("hello", str) == "hello"

    def test_any_passthrough(self):
        from typing import Any as _Any

        assert validate_payload_against_output_type(42, _Any) == 42

    def test_pydantic_model_already_instance(self):
        class M(BaseModel):
            x: int

        m = M(x=1)
        result = validate_payload_against_output_type(m, M)
        assert result is m

    def test_pydantic_model_from_dict(self):
        class M(BaseModel):
            x: int

        result = validate_payload_against_output_type({"x": 5}, M)
        assert isinstance(result, M)
        assert result.x == 5

    def test_pydantic_model_from_json_string(self):
        class M(BaseModel):
            x: int

        result = validate_payload_against_output_type('{"x": 7}', M)
        assert isinstance(result, M)
        assert result.x == 7

    def test_pydantic_model_last_resort_coerce(self):
        class M(BaseModel):
            x: int

        result = validate_payload_against_output_type({"x": 3}, M)
        assert result.x == 3

    def test_generic_list_from_str(self):
        result = validate_payload_against_output_type("[1, 2, 3]", list[int])
        assert result == [1, 2, 3]

    def test_generic_list_from_python(self):
        result = validate_payload_against_output_type([1, 2], list[int])
        assert result == [1, 2]

    def test_fallthrough_unknown_type(self):
        result = validate_payload_against_output_type("x", 42)  # type: ignore
        assert result == "x"


class TestNormalizeJsonSchema:
    def test_adds_additional_properties_false(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = normalize_json_schema(schema)
        assert result["additionalProperties"] is False

    def test_recursively_normalizes_properties(self):
        schema = {
            "type": "object",
            "properties": {"child": {"type": "object", "properties": {"y": {"type": "int"}}}},
        }
        result = normalize_json_schema(schema)
        assert result["properties"]["child"]["additionalProperties"] is False

    def test_normalizes_defs(self):
        schema = {
            "type": "object",
            "$defs": {"MyModel": {"type": "object", "properties": {}}},
        }
        result = normalize_json_schema(schema)
        assert result["$defs"]["MyModel"]["additionalProperties"] is False

    def test_anyof_list_normalized(self):
        schema = {
            "anyOf": [
                {"type": "object", "properties": {"x": {"type": "string"}}},
                {"type": "null"},
            ]
        }
        result = normalize_json_schema(schema)
        assert result["anyOf"][0]["additionalProperties"] is False

    def test_items_schema_normalized(self):
        schema = {
            "type": "array",
            "items": {"type": "object", "properties": {"x": {}}},
        }
        result = normalize_json_schema(schema)
        assert result["items"]["additionalProperties"] is False

    def test_non_dict_passthrough(self):
        assert normalize_json_schema("string") == "string"  # type: ignore

    def test_existing_additional_properties_not_overwritten(self):
        schema = {"type": "object", "additionalProperties": True}
        result = normalize_json_schema(schema)
        assert result["additionalProperties"] is True


class TestValidateSchema:
    def test_valid_string(self):
        assert _validate_schema("hello", {"type": "string"}) is None

    def test_invalid_type(self):
        result = _validate_schema(42, {"type": "string"})
        assert result is not None

    def test_enum_match(self):
        assert _validate_schema("a", {"enum": ["a", "b", "c"]}) is None

    def test_enum_mismatch(self):
        result = _validate_schema("z", {"enum": ["a", "b"]})
        assert result is not None

    def test_required_fields(self):
        data = {"x": 1}
        result = _validate_schema(data, {"type": "object", "required": ["x", "y"]})
        assert result is not None and "y" in result

    def test_additional_properties_not_allowed(self):
        data = {"x": 1, "extra": 2}
        schema = {
            "type": "object",
            "properties": {"x": {}},
            "additionalProperties": False,
        }
        result = _validate_schema(data, schema)
        assert result is not None

    def test_array_items_type_check(self):
        result = _validate_schema([1, 2, "bad"], {"type": "array", "items": {"type": "integer"}})
        assert result is not None

    def test_boolean_vs_integer_distinct(self):
        result = _validate_schema(True, {"type": "integer"})
        assert result is not None

    def test_bool_int_enum_distinct(self):
        assert not _enum_match(True, 1)
        assert _enum_match(True, True)
        assert _enum_match(1, 1)


class TestValidateSchemaSubset:
    def test_no_schema_type(self):
        # No type constraint — anything passes
        assert _validate_schema_subset("hello", {}) is None

    def test_number_type(self):
        assert _validate_schema_subset(3.14, {"type": "number"}) is None

    def test_null_type(self):
        assert _validate_schema_subset(None, {"type": "null"}) is None

    def test_nested_property_validation(self):
        schema = {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
        }
        result = _validate_schema_subset({"n": "bad"}, schema)
        assert result is not None and "n" in result


# =============================================================================
# core/tool_schema.py — basic schema building
# =============================================================================


class TestToolSchemaBuilder:
    def test_simple_function_schema(self):
        from lazybridge.core.tool_schema import ToolSchemaBuilder, ToolSchemaMode

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        builder = ToolSchemaBuilder()
        defn = builder.build(add, name="add", description=None, strict=False, mode=ToolSchemaMode.SIGNATURE)
        assert defn.name == "add"
        assert defn.description is not None

    def test_annotated_function_schema(self):
        from typing import Annotated

        from lazybridge.core.tool_schema import ToolSchemaBuilder, ToolSchemaMode

        def greet(name: Annotated[str, "The person's name"]) -> str:
            """Greet someone."""
            return f"Hello {name}"

        builder = ToolSchemaBuilder()
        defn = builder.build(
            greet, name="greet", description="Greet a person", strict=False, mode=ToolSchemaMode.SIGNATURE
        )
        assert defn.name == "greet"

    def test_no_params_function(self):
        from lazybridge.core.tool_schema import ToolSchemaBuilder, ToolSchemaMode

        def ping() -> str:
            """Return pong."""
            return "pong"

        builder = ToolSchemaBuilder()
        defn = builder.build(ping, name="ping", description=None, strict=False, mode=ToolSchemaMode.SIGNATURE)
        assert defn.name == "ping"

    def test_pydantic_model_param(self):
        from lazybridge.core.tool_schema import ToolSchemaBuilder, ToolSchemaMode

        class Params(BaseModel):
            query: str
            limit: int = 10

        def search(params: Params) -> list:
            return []

        builder = ToolSchemaBuilder()
        defn = builder.build(search, name="search", description=None, strict=False, mode=ToolSchemaMode.SIGNATURE)
        assert defn.name == "search"

    def test_strict_mode(self):
        from lazybridge.core.tool_schema import ToolSchemaBuilder, ToolSchemaMode

        def fn(x: int) -> int:
            return x

        builder = ToolSchemaBuilder()
        defn = builder.build(fn, name="fn", description=None, strict=True, mode=ToolSchemaMode.SIGNATURE)
        assert defn.name == "fn"


# =============================================================================
# engines/llm.py — with a fake provider (no real API calls)
# =============================================================================


def _make_fake_executor(response_text: str = "fake response"):
    """Build an Executor-like object with a fake provider, bypassing resolution."""
    from lazybridge.core.executor import Executor
    from lazybridge.core.providers.base import BaseProvider
    from lazybridge.core.types import (
        CompletionRequest,
        CompletionResponse,
        StreamChunk,
        UsageStats,
    )

    class _FP(BaseProvider):
        default_model = "fake-model"

        def _init_client(self, **kwargs) -> None:
            self._text = response_text

        def complete(self, request: CompletionRequest) -> CompletionResponse:
            return CompletionResponse(
                content=self._text,
                usage=UsageStats(),
                model=self.model,
            )

        def stream(self, request: CompletionRequest):
            yield StreamChunk(delta=self._text, stop_reason="stop")

        async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
            return self.complete(request)

        async def astream(self, request: CompletionRequest):
            yield StreamChunk(delta=self._text, stop_reason="stop")

    provider = _FP(model="fake-model")
    return Executor(provider, max_retries=0)


def _make_llm_engine(response_text: str = "fake response"):
    """Build an LLMEngine wired to a fake provider (no real API calls).

    We can't pass a non-string provider to LLMEngine directly because __init__
    stores the provider name string and resolves it lazily in _make_executor.
    Instead we patch _make_executor to return our fake executor.
    """
    from lazybridge.engines.llm import LLMEngine

    # Use "anthropic" as provider name — it will be overridden before any API call.
    engine = LLMEngine.__new__(LLMEngine)
    engine._agent_name = "test"
    engine.model = "fake-model"
    engine.provider = "anthropic"
    engine.system = None
    engine.max_turns = 3
    engine.tool_choice = "auto"
    engine.temperature = None
    engine.thinking = False
    engine.request_timeout = None
    engine.max_retries = 0
    engine.retry_delay = 0.0
    engine.native_tools = []

    fake_executor = _make_fake_executor(response_text)
    engine._make_executor = lambda: fake_executor  # type: ignore[method-assign]
    return engine


class TestLLMEngine:
    @pytest.mark.asyncio
    async def test_run_returns_envelope(self):
        engine = _make_llm_engine()
        env = Envelope.from_task("hello")
        result = await engine.run(env, tools=[], output_type=str, memory=None, session=None)
        assert isinstance(result, Envelope)
        assert result.ok
        assert "fake response" in result.text()

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        engine = _make_llm_engine()
        env = Envelope.from_task("stream me")
        chunks = []
        async for chunk in engine.stream(env, tools=[], output_type=str, memory=None, session=None):
            chunks.append(chunk)
        assert any(c for c in chunks)

    @pytest.mark.asyncio
    async def test_run_with_session_emits_events(self):
        events: list[dict] = []
        sess = Session(exporters=[CallbackExporter(fn=events.append)])
        engine = _make_llm_engine()
        env = Envelope.from_task("emit events")
        await engine.run(env, tools=[], output_type=str, memory=None, session=sess)
        event_types = [e.get("event_type") for e in events]
        assert "agent_start" in event_types
        assert "agent_finish" in event_types

    @pytest.mark.asyncio
    async def test_run_with_memory(self):
        from lazybridge.memory import Memory

        engine = _make_llm_engine()
        mem = Memory()
        env = Envelope.from_task("remember this")
        result = await engine.run(env, tools=[], output_type=str, memory=mem, session=None)
        assert result.ok

    @pytest.mark.asyncio
    async def test_run_with_system_prompt(self):
        engine = _make_llm_engine()
        engine.system = "You are helpful."
        env = Envelope.from_task("task")
        result = await engine.run(env, tools=[], output_type=str, memory=None, session=None)
        assert result.ok

    @pytest.mark.asyncio
    async def test_run_with_context_envelope(self):
        engine = _make_llm_engine()
        env = Envelope(task="task", context="some context", payload="task")
        result = await engine.run(env, tools=[], output_type=str, memory=None, session=None)
        assert result.ok

    @pytest.mark.asyncio
    async def test_run_with_tool_call(self):
        """LLMEngine correctly runs a tool when the provider returns a tool_call."""
        from lazybridge.core.executor import Executor
        from lazybridge.core.providers.base import BaseProvider
        from lazybridge.core.types import (
            CompletionRequest,
            CompletionResponse,
            StreamChunk,
            ToolCall,
            UsageStats,
        )
        from lazybridge.engines.llm import LLMEngine

        calls = []

        class _ToolCallProvider(BaseProvider):
            default_model = "tool-model"

            def _init_client(self, **kwargs) -> None:
                self._call_count = 0

            def complete(self, request: CompletionRequest) -> CompletionResponse:
                self._call_count += 1
                if self._call_count == 1:
                    # First call: emit a tool call
                    tc = ToolCall(id="c1", name="adder", arguments={"a": 1, "b": 2})
                    return CompletionResponse(
                        content="",
                        tool_calls=[tc],
                        usage=UsageStats(),
                        model=self.model,
                    )
                # Second call: final answer
                return CompletionResponse(
                    content="result is 3",
                    usage=UsageStats(),
                    model=self.model,
                )

            def stream(self, request: CompletionRequest):
                yield StreamChunk(delta="x", stop_reason="stop")

            async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
                return self.complete(request)

            async def astream(self, request: CompletionRequest):
                yield StreamChunk(delta="x", stop_reason="stop")

        def adder(a: int, b: int) -> int:
            """Add two numbers."""
            calls.append((a, b))
            return a + b

        engine = LLMEngine.__new__(LLMEngine)
        engine._agent_name = "test"
        engine.model = "tool-model"
        engine.provider = "anthropic"
        engine.system = None
        engine.max_turns = 5
        engine.tool_choice = "auto"
        engine.temperature = None
        engine.thinking = False
        engine.request_timeout = None
        engine.max_retries = 0
        engine.retry_delay = 0.0
        engine.native_tools = []
        fake_exec = Executor(_ToolCallProvider(model="tool-model"), max_retries=0)
        engine._make_executor = lambda: fake_exec  # type: ignore[method-assign]

        tool = Tool(adder, name="adder")
        env = Envelope.from_task("add 1 and 2")
        result = await engine.run(env, tools=[tool], output_type=str, memory=None, session=None)
        assert result.ok
        assert calls == [(1, 2)]


# =============================================================================
# agent.py — uncovered paths
# =============================================================================


class _EchoEngine2:
    model = "echo"
    _agent_name = "echo"

    async def run(self, env, *, tools, output_type, memory, session):
        return Envelope(task=env.task, payload=env.task or "")

    async def stream(self, env, *, tools, output_type, memory, session):
        yield env.task or ""


class TestAgentUncoveredPaths:
    @pytest.mark.asyncio
    async def test_run_with_sources(self):
        agent = Agent(engine=_EchoEngine2(), sources=["source context"])
        result = await agent.run("task")
        assert result.ok

    @pytest.mark.asyncio
    async def test_agent_session_propagation_to_tool_agents(self):
        sess = Session()
        inner = Agent(engine=_EchoEngine2(), name="inner")
        Agent(engine=_EchoEngine2(), name="outer", tools=[inner], session=sess)
        # inner should have inherited the session
        assert inner.session is sess

    def test_agent_name_fallback_chain(self):
        # When no name is provided, falls back to model name from engine
        agent = Agent(engine=_EchoEngine2())
        assert agent.name == "echo"

    @pytest.mark.asyncio
    async def test_agent_timeout_wraps_run(self):

        class _SlowEngine:
            model = "slow"
            _agent_name = "slow"

            async def run(self, env, *, tools, output_type, memory, session):
                await asyncio.sleep(10)
                return Envelope(payload="never")

            async def stream(self, env, *, tools, output_type, memory, session):
                yield ""

        agent = Agent(engine=_SlowEngine(), timeout=0.05)
        result = await agent.run("task")
        assert not result.ok
        assert "timeout" in result.error.message.lower()

    def test_chain_creates_agent(self):
        a = Agent(engine=_EchoEngine2(), name="a")
        b = Agent(engine=_EchoEngine2(), name="b")
        chain = Agent.chain(a, b, name="ab_chain")
        assert chain.name == "ab_chain"

    def test_parallel_creates_agent(self):
        a = Agent(engine=_EchoEngine2(), name="a")
        b = Agent(engine=_EchoEngine2(), name="b")
        par = Agent.parallel(a, b)
        assert par is not None

    @pytest.mark.asyncio
    async def test_agent_stream_works(self):
        agent = Agent(engine=_EchoEngine2(), name="streamer")
        chunks = []
        async for chunk in agent.stream("hello"):
            chunks.append(chunk)
        assert any(chunks)

    def test_as_tool_returns_tool(self):
        agent = Agent(engine=_EchoEngine2(), name="sub")
        t = agent.as_tool("sub_tool", "Does sub things")
        assert isinstance(t, Tool)
        assert t.name == "sub_tool"

    def test_definition_method(self):
        agent = Agent(engine=_EchoEngine2(), name="def_agent")
        defn = agent.definition()
        assert defn is not None

    @pytest.mark.asyncio
    async def test_agent_with_output_model(self):
        class Out(BaseModel):
            result: str

        class _JsonEngine:
            model = "json"
            _agent_name = "json"

            async def run(self, env, *, tools, output_type, memory, session):
                return Envelope(task=env.task, payload=Out(result="hello"))

            async def stream(self, env, *, tools, output_type, memory, session):
                yield ""

        agent = Agent(engine=_JsonEngine(), output=Out)
        result = await agent.run("task")
        # payload should be an Out instance or passed through
        assert result.ok


# =============================================================================
# core/tool_schema.py — annotation_to_schema branches
# =============================================================================


class TestAnnotationToSchema:
    """Tests for _annotation_to_schema covering all type branches."""

    def _schema(self, annotation):
        from lazybridge.core.tool_schema import _annotation_to_schema

        return _annotation_to_schema(annotation)

    def test_optional_x(self):
        s = self._schema(int | None)
        assert s == {"anyOf": [{"type": "integer"}, {"type": "null"}]}

    def test_union_two_types(self):
        from typing import Union

        s = self._schema(Union[int, str])
        assert "anyOf" in s
        assert len(s["anyOf"]) == 2

    def test_annotated_with_string(self):
        from typing import Annotated

        s = self._schema(Annotated[int, "The count"])
        assert s["type"] == "integer"
        assert s["description"] == "The count"

    def test_annotated_with_description_attr(self):
        from typing import Annotated

        class Meta:
            description = "Some param"

        s = self._schema(Annotated[str, Meta()])
        assert s.get("description") == "Some param"

    def test_annotated_no_metadata(self):
        from typing import Annotated

        s = self._schema(Annotated[str, 42])
        assert "type" in s

    def test_literal_strings(self):
        from typing import Literal

        s = self._schema(Literal["a", "b"])
        assert s["type"] == "string"
        assert set(s["enum"]) == {"a", "b"}

    def test_literal_ints(self):
        from typing import Literal

        s = self._schema(Literal[1, 2, 3])
        assert s["type"] == "integer"
        assert 1 in s["enum"]

    def test_literal_mixed(self):
        from typing import Literal

        s = self._schema(Literal[1, "a"])
        assert "enum" in s

    def test_list_with_item_type(self):
        s = self._schema(list[int])
        assert s["type"] == "array"
        assert s["items"]["type"] == "integer"

    def test_set_with_item_type(self):
        s = self._schema(set[str])
        assert s["type"] == "array"
        assert s.get("uniqueItems") is True

    def test_frozenset_with_item_type(self):
        s = self._schema(frozenset[str])
        assert s["type"] == "array"
        assert s.get("uniqueItems") is True

    def test_tuple_homogeneous(self):
        s = self._schema(tuple[int, ...])
        assert s["type"] == "array"
        assert s["items"]["type"] == "integer"

    def test_tuple_fixed_length(self):
        s = self._schema(tuple[int, str, bool])
        assert s["type"] == "array"
        assert "prefixItems" in s
        assert s["items"] is False

    def test_tuple_empty(self):
        s = self._schema(tuple[()])
        # no args → plain array
        assert s["type"] == "array"

    def test_dict(self):
        s = self._schema(dict[str, int])
        assert s["type"] == "object"

    def test_enum_str_values(self):
        from enum import Enum

        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        s = self._schema(Color)
        assert s["type"] == "string"
        assert "red" in s["enum"]

    def test_enum_int_values(self):
        from enum import Enum

        class Status(Enum):
            A = 1
            B = 2

        s = self._schema(Status)
        assert s["type"] == "integer"
        assert 1 in s["enum"]

    def test_enum_mixed_values(self):
        from enum import Enum

        class Mixed(Enum):
            A = 1
            B = "two"

        s = self._schema(Mixed)
        assert "enum" in s

    def test_pydantic_model(self):
        class Sub(BaseModel):
            x: int

        s = self._schema(Sub)
        assert "properties" in s

    def test_unknown_type_returns_string(self):
        s = self._schema(object)
        assert s == {"type": "string"}


# =============================================================================
# core/tool_schema.py — InMemoryArtifactStore
# =============================================================================


class TestInMemoryArtifactStore:
    def _make_store(self):
        from lazybridge.core.tool_schema import InMemoryArtifactStore

        return InMemoryArtifactStore()

    def _make_artifact(self, fp="abc123"):
        from lazybridge.core.tool_schema import (
            ToolCompileArtifact,
            ToolSchemaMode,
            ToolSourceStatus,
        )
        from lazybridge.core.types import ToolDefinition

        defn = ToolDefinition(name="fn", description="", parameters={})
        return ToolCompileArtifact(
            fingerprint=fp,
            compiler_version="1",
            prompt_version="1",
            mode=ToolSchemaMode.SIGNATURE,
            source_status=ToolSourceStatus.BASELINE_ONLY,
            definition=defn,
            baseline_definition=None,
            llm_enriched_fields=frozenset(),
            warnings=(),
        )

    def test_get_miss_returns_none(self):
        store = self._make_store()
        assert store.get("nope") is None

    def test_put_then_get(self):
        store = self._make_store()
        art = self._make_artifact("xyz")
        store.put(art)
        assert store.get("xyz") is art

    def test_len(self):
        store = self._make_store()
        assert len(store) == 0
        store.put(self._make_artifact("a"))
        assert len(store) == 1
        store.put(self._make_artifact("b"))
        assert len(store) == 2

    def test_clear(self):
        store = self._make_store()
        store.put(self._make_artifact("a"))
        store.clear()
        assert len(store) == 0
        assert store.get("a") is None


# =============================================================================
# core/tool_schema.py — _flatten_refs and cache helpers
# =============================================================================


class TestFlattenRefs:
    def test_no_defs_passthrough(self):
        from lazybridge.core.tool_schema import _flatten_refs

        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = _flatten_refs(schema)
        assert result == schema

    def test_inline_ref(self):
        from lazybridge.core.tool_schema import _flatten_refs

        schema = {
            "type": "object",
            "properties": {"item": {"$ref": "#/$defs/MyDef"}},
            "$defs": {"MyDef": {"type": "integer"}},
        }
        result = _flatten_refs(schema)
        assert "$defs" not in result
        assert result["properties"]["item"] == {"type": "integer"}

    def test_cache_hit(self):
        from lazybridge.core.tool_schema import (
            _flatten_cache_clear,
            _flatten_cache_stats,
            _flatten_refs,
        )

        _flatten_cache_clear()
        schema = {
            "$defs": {"X": {"type": "string"}},
            "properties": {"v": {"$ref": "#/$defs/X"}},
        }
        _flatten_refs(schema)
        before = _flatten_cache_stats()
        _flatten_refs(schema)
        after = _flatten_cache_stats()
        assert after[1] > before[1]  # hit count increased

    def test_circular_ref_left_intact(self):
        from lazybridge.core.tool_schema import _flatten_refs

        schema = {
            "$defs": {"Node": {"properties": {"child": {"$ref": "#/$defs/Node"}}}},
            "properties": {"root": {"$ref": "#/$defs/Node"}},
        }
        result = _flatten_refs(schema)
        # Should not infinite-loop; the circular $ref is left as-is
        assert result is not None

    def test_list_items_inlined(self):
        from lazybridge.core.tool_schema import _flatten_refs

        schema = {
            "$defs": {"Item": {"type": "boolean"}},
            "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}},
        }
        result = _flatten_refs(schema)
        assert result["properties"]["items"]["items"] == {"type": "boolean"}


class TestFlattenCacheHelpers:
    def test_stats_returns_tuple(self):
        from lazybridge.core.tool_schema import _flatten_cache_stats

        stats = _flatten_cache_stats()
        assert isinstance(stats, tuple)
        assert len(stats) == 2

    def test_clear_resets_hits(self):
        from lazybridge.core.tool_schema import (
            _flatten_cache_clear,
            _flatten_cache_stats,
            _flatten_refs,
        )

        schema = {
            "$defs": {"T": {"type": "string"}},
            "properties": {"v": {"$ref": "#/$defs/T"}},
        }
        _flatten_refs(schema)
        _flatten_refs(schema)
        _flatten_cache_clear()
        stats = _flatten_cache_stats()
        assert stats[0] == 0  # size is 0 after clear
        assert stats[1] == 0  # hits reset to 0


# =============================================================================
# core/tool_schema.py — _validate_and_coerce_arguments + _make_arg_model
# =============================================================================


class TestValidateAndCoerceArguments:
    def test_valid_args_coerced(self):
        from lazybridge.core.tool_schema import _validate_and_coerce_arguments

        def add(x: int, y: int) -> int:
            return x + y

        result = _validate_and_coerce_arguments(add, {"x": "3", "y": "4"})
        assert result["x"] == 3
        assert result["y"] == 4

    def test_invalid_args_raises(self):
        from lazybridge.core.tool_schema import (
            ToolArgumentValidationError,
            _validate_and_coerce_arguments,
        )

        def add(x: int, y: int) -> int:
            return x + y

        with pytest.raises(ToolArgumentValidationError):
            _validate_and_coerce_arguments(add, {"x": "not_an_int", "y": 1})

    def test_zero_arg_passthrough(self):
        from lazybridge.core.tool_schema import _validate_and_coerce_arguments

        def noop() -> None:
            pass

        result = _validate_and_coerce_arguments(noop, {})
        assert result == {}

    def test_extra_args_rejected_without_kwargs(self):
        from lazybridge.core.tool_schema import (
            ToolArgumentValidationError,
            _validate_and_coerce_arguments,
        )

        def fn(x: int) -> int:
            return x

        with pytest.raises(ToolArgumentValidationError):
            _validate_and_coerce_arguments(fn, {"x": 1, "extra": "oops"})

    def test_kwargs_func_allows_extra(self):
        from lazybridge.core.tool_schema import _validate_and_coerce_arguments

        def fn(x: int, **kwargs) -> int:
            return x

        result = _validate_and_coerce_arguments(fn, {"x": 1, "extra": "ok"})
        assert result["x"] == 1


class TestMakeArgModel:
    def test_function_with_params(self):
        from lazybridge.core.tool_schema import _make_arg_model

        def fn(x: int, y: str = "default") -> None:
            pass

        model = _make_arg_model(fn)
        assert model is not None

    def test_function_cached(self):
        from lazybridge.core.tool_schema import _make_arg_model

        def fn(x: int) -> None:
            pass

        m1 = _make_arg_model(fn)
        m2 = _make_arg_model(fn)
        assert m1 is m2  # same cached object

    def test_zero_arg_returns_none(self):
        from lazybridge.core.tool_schema import _make_arg_model

        def noop() -> None:
            pass

        assert _make_arg_model(noop) is None

    def test_varargs_only_returns_none(self):
        from lazybridge.core.tool_schema import _make_arg_model

        def fn(*args) -> None:
            pass

        assert _make_arg_model(fn) is None

    def test_func_with_self_param_skipped(self):
        from lazybridge.core.tool_schema import _make_arg_model

        def fn(self, x: int) -> None:
            pass

        model = _make_arg_model(fn)
        # self should be skipped; only x matters
        assert model is not None


# =============================================================================
# core/tool_schema.py — _get_source_or_stub / _sig_required_params
# =============================================================================


class TestGetSourceOrStub:
    def test_builtin_returns_stub(self):
        import inspect
        from unittest.mock import patch

        from lazybridge.core.tool_schema import _get_source_or_stub

        def my_fn(x: int) -> int:
            return x

        with patch.object(inspect, "getsource", side_effect=OSError("no source")):
            src = _get_source_or_stub(my_fn)
        assert "my_fn" in src

    def test_regular_function_returns_source(self):
        from lazybridge.core.tool_schema import _get_source_or_stub

        def my_func(x: int) -> int:
            return x

        src = _get_source_or_stub(my_func)
        assert "my_func" in src


class TestSigRequiredParams:
    def test_required_and_optional(self):
        from lazybridge.core.tool_schema import _sig_required_params

        def fn(x: int, y: str = "default") -> None:
            pass

        required = _sig_required_params(fn)
        assert "x" in required
        assert "y" not in required

    def test_all_required(self):
        from lazybridge.core.tool_schema import _sig_required_params

        def fn(a: int, b: str) -> None:
            pass

        assert _sig_required_params(fn) == {"a", "b"}

    def test_varargs_excluded(self):
        from lazybridge.core.tool_schema import _sig_required_params

        def fn(x: int, *args, **kwargs) -> None:
            pass

        required = _sig_required_params(fn)
        assert "args" not in required
        assert "kwargs" not in required
        assert "x" in required


# =============================================================================
# core/tool_schema.py — build_artifact with store + flatten_refs opt-in
# =============================================================================


class TestToolSchemaBuildArtifact:
    def test_build_artifact_with_store(self):
        from lazybridge.core.tool_schema import (
            InMemoryArtifactStore,
            ToolSchemaBuilder,
            ToolSchemaMode,
        )

        def add(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        store = InMemoryArtifactStore()
        builder = ToolSchemaBuilder(artifact_store=store)
        art1 = builder.build_artifact(add, name="add", description=None, strict=False, mode=ToolSchemaMode.SIGNATURE)
        assert not art1.cache_hit
        assert len(store) == 1

        art2 = builder.build_artifact(add, name="add", description=None, strict=False, mode=ToolSchemaMode.SIGNATURE)
        assert art2.cache_hit

    def test_build_artifact_with_flatten_refs(self):
        from lazybridge.core.tool_schema import ToolSchemaBuilder, ToolSchemaMode

        class Inner(BaseModel):
            v: int

        def fn(item: Inner) -> None:
            pass

        builder = ToolSchemaBuilder(flatten_refs=True)
        art = builder.build_artifact(fn, name="fn", description=None, strict=False, mode=ToolSchemaMode.SIGNATURE)
        assert art is not None
        # $defs should be flattened out of the parameters
        assert "$defs" not in art.definition.parameters

    def test_build_with_docstring_description(self):
        from lazybridge.core.tool_schema import ToolSchemaBuilder, ToolSchemaMode

        def fn(x: int) -> int:
            """Compute something useful."""
            return x

        builder = ToolSchemaBuilder()
        art = builder.build_artifact(fn, name="fn", description=None, strict=False, mode=ToolSchemaMode.SIGNATURE)
        assert "Compute" in art.definition.description

    def test_build_with_docstring_param_description(self):
        from lazybridge.core.tool_schema import ToolSchemaBuilder, ToolSchemaMode

        def fn(x: int) -> int:
            """Do thing.

            Args:
                x: The input value.
            """
            return x

        builder = ToolSchemaBuilder()
        art = builder.build_artifact(fn, name="fn", description=None, strict=False, mode=ToolSchemaMode.SIGNATURE)
        props = art.definition.parameters.get("properties", {})
        assert props.get("x", {}).get("description") == "The input value."


# =============================================================================
# exporters.py — JsonFileExporter.close idempotency
# =============================================================================


class TestExporterEdgeCases:
    def test_json_file_exporter_close_idempotent(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        exp = JsonFileExporter(path=path)
        exp.export({"event_type": "test", "data": "hello"})
        exp.close()
        exp.close()  # second close must not raise

    def test_json_file_exporter_already_closed_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        exp = JsonFileExporter(path=path)
        exp._fh.close()
        exp.close()  # should not raise


# =============================================================================
# core/tool_schema.py — _parse_docstring_params Sphinx style + helpers
# =============================================================================


class TestParseDocstringParams:
    def test_sphinx_style(self):
        from lazybridge.core.tool_schema import _parse_docstring_params

        doc = "Do a thing.\n\n:param x: The first arg\n:param y: The second arg\n"
        params = _parse_docstring_params(doc)
        assert params["x"] == "The first arg"
        assert params["y"] == "The second arg"

    def test_sphinx_style_with_type(self):
        from lazybridge.core.tool_schema import _parse_docstring_params

        doc = ":param int x: An integer\n"
        params = _parse_docstring_params(doc)
        assert params["x"] == "An integer"

    def test_google_style(self):
        from lazybridge.core.tool_schema import _parse_docstring_params

        doc = "Summary.\n\nArgs:\n    x: First param\n    y: Second param\n"
        params = _parse_docstring_params(doc)
        assert params["x"] == "First param"
        assert params["y"] == "Second param"

    def test_empty_returns_empty(self):
        from lazybridge.core.tool_schema import _parse_docstring_params

        assert _parse_docstring_params("") == {}


class TestFuncSourceHash:
    def test_fallback_when_no_source(self):
        import inspect
        from unittest.mock import patch

        from lazybridge.core.tool_schema import _func_source_hash

        def my_fn(x: int) -> int:
            """Return x."""
            return x

        with patch.object(inspect, "getsource", side_effect=OSError("no source")):
            h = _func_source_hash(my_fn)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_regular_function(self):
        from lazybridge.core.tool_schema import _func_source_hash

        def fn():
            pass

        h = _func_source_hash(fn)
        assert len(h) == 16


class TestSchemaLlmId:
    def test_none_returns_empty(self):
        from lazybridge.core.tool_schema import _schema_llm_id

        assert _schema_llm_id(None) == ""

    def test_function_returns_qualname(self):
        from lazybridge.core.tool_schema import _schema_llm_id

        def my_fn():
            pass

        result = _schema_llm_id(my_fn)
        assert "my_fn" in result

    def test_instance_without_qualname(self):
        from lazybridge.core.tool_schema import _schema_llm_id

        class MyClass:
            pass

        obj = MyClass()
        result = _schema_llm_id(obj)
        assert "MyClass" in result
