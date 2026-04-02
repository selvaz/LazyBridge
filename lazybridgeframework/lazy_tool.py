"""LazyTool — unified tool abstraction.

A LazyTool can wrap:
  A) A plain Python function  → ``LazyTool.from_function(fn)``
  B) A LazyAgent (delegated)  → ``LazyTool.from_agent(agent)`` or ``agent.as_tool(...)``

Three layers:
  1. Schema  — ToolDefinition built from type hints (or LLM/HYBRID)
  2. Guidance — optional string injected into the system prompt
  3. Execution — run() / arun() returning a value directly to the caller

Communication is always via return value, not via shared store.
The store is for optional side-effects that other agents may read later.

Schema modes (from core.tool_schema):
  SIGNATURE  — introspect type hints (default, fast, deterministic)
  LLM        — let an LLM generate the schema
  HYBRID     — type hints for types, LLM for descriptions
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from lazybridgeframework.core.tool_schema import (
    _DEFAULT_BUILDER,
    ArtifactStore,
    InMemoryArtifactStore,
    ToolArgumentValidationError,
    ToolCompileArtifact,
    ToolSchemaBuilder,
    ToolSchemaBuildError,
    ToolSchemaMode,
    ToolSourceStatus,
    _validate_and_coerce_arguments,
)
from lazybridgeframework.core.types import ToolDefinition

if TYPE_CHECKING:
    from lazybridgeframework.lazy_agent import LazyAgent

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# params shorthand helper  {name: type} → JSON Schema
# ---------------------------------------------------------------------------

_PYTHON_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _check_unique(name: str, seen: set[str]) -> None:
    """Raise ValueError if name is already in seen, otherwise register it."""
    if name in seen:
        raise ValueError(f"Duplicate tool name: '{name}'")
    seen.add(name)


def _params_to_schema(params: dict[str, Any]) -> dict:
    """Convert ``{name: type | schema_dict}`` to a JSON Schema object."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, hint in params.items():
        if isinstance(hint, dict):
            properties[name] = hint
        else:
            properties[name] = {"type": _PYTHON_TYPE_MAP.get(hint, "string")}
        required.append(name)
    return {"type": "object", "properties": properties, "required": required}


# ---------------------------------------------------------------------------
# _DelegateConfig — internal config for agent-backed tools
# ---------------------------------------------------------------------------

@dataclass
class _DelegateConfig:
    agent: Any  # LazyAgent instance
    output_schema: type | dict | None = None
    native_tools: list | None = None
    system_prompt: str | None = None


# ---------------------------------------------------------------------------
# LazyTool
# ---------------------------------------------------------------------------

@dataclass
class LazyTool:
    """A self-contained tool: schema + execution + optional guidance.

    Do not construct directly — use the factory methods.
    """

    name: str
    description: str
    func: Callable[..., Any] | None = None
    guidance: str | None = None
    schema_mode: ToolSchemaMode = ToolSchemaMode.SIGNATURE
    strict: bool = False
    schema_builder: ToolSchemaBuilder | None = None
    schema_llm: Any | None = None
    _delegate: _DelegateConfig | None = field(default=None, repr=False)
    _compiled: ToolDefinition | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Factory: from plain Python function
    # ------------------------------------------------------------------

    @classmethod
    def from_function(
        cls,
        func: Callable,
        *,
        name: str | None = None,
        description: str | None = None,
        guidance: str | None = None,
        schema_mode: ToolSchemaMode = ToolSchemaMode.SIGNATURE,
        strict: bool = False,
        schema_builder: ToolSchemaBuilder | None = None,
        schema_llm: Any | None = None,
    ) -> LazyTool:
        """Wrap a Python callable as a LazyTool.

        Schema is auto-generated from type hints by default (SIGNATURE mode).
        Name and description default to the function's name and docstring.
        """
        tool_name = name or func.__name__
        tool_desc = description or (inspect.getdoc(func) or "").split("\n")[0] or tool_name
        return cls(
            name=tool_name,
            description=tool_desc,
            func=func,
            guidance=guidance,
            schema_mode=schema_mode,
            strict=strict,
            schema_builder=schema_builder,
            schema_llm=schema_llm,
        )

    # ------------------------------------------------------------------
    # Factory: from LazyAgent (delegated execution)
    # ------------------------------------------------------------------

    @classmethod
    def from_agent(
        cls,
        agent: LazyAgent,
        *,
        name: str | None = None,
        description: str | None = None,
        guidance: str | None = None,
        output_schema: type | dict | None = None,
        native_tools: list | None = None,
        system_prompt: str | None = None,
        strict: bool = False,
    ) -> LazyTool:
        """Wrap a LazyAgent as a tool with schema ``{"task": str}``.

        When the parent calls this tool, the task string is forwarded to the
        agent's loop() or chat() depending on its configuration.
        The agent's return value is passed directly back to the caller.
        """
        tool_name = name or getattr(agent, "name", None) or getattr(agent, "id", "agent_tool")
        tool_desc = description or getattr(agent, "description", None) or f"Delegate task to {tool_name}"
        delegate = _DelegateConfig(
            agent=agent,
            output_schema=output_schema,
            native_tools=native_tools,
            system_prompt=system_prompt,
        )
        # Schema is always {"task": str} for delegated agents
        fixed_def = ToolDefinition(
            name=tool_name,
            description=tool_desc,
            parameters=_params_to_schema({"task": str}),
            strict=strict,
        )
        tool = cls(
            name=tool_name,
            description=tool_desc,
            guidance=guidance,
            strict=strict,
            _delegate=delegate,
            _compiled=fixed_def,
        )
        return tool

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def definition(self, schema_llm: Any = None) -> ToolDefinition:
        """Return the ToolDefinition for this tool (cached after first call)."""
        if self._compiled is not None:
            return self._compiled
        builder = self.schema_builder or _DEFAULT_BUILDER
        effective_llm = schema_llm or self.schema_llm
        defn = builder.build(
            self.func,
            name=self.name,
            description=self.description,
            strict=self.strict,
            mode=self.schema_mode,
            schema_llm=effective_llm,
        )
        self._compiled = defn
        return defn

    def compile(self, schema_llm: Any = None) -> LazyTool:
        """Freeze schema and return self (for chaining)."""
        self.definition(schema_llm)
        return self

    # ------------------------------------------------------------------
    # Execution — sync
    # ------------------------------------------------------------------

    def run(self, arguments: dict[str, Any], *, parent: Any = None) -> Any:
        """Execute the tool with the given arguments.

        For delegated tools, ``parent`` is the executing LazyAgent (used to
        wire memory lineage if needed).
        Returns the result directly — no store writes, no side effects.
        """
        if self._delegate is not None:
            return self._run_delegate(arguments, parent=parent)

        if self.func is None:
            raise RuntimeError(f"LazyTool '{self.name}' has no function to execute.")

        validated = _validate_and_coerce_arguments(self.func, arguments)
        return self.func(**validated)

    def _run_delegate(self, arguments: dict[str, Any], parent: Any) -> Any:
        task = arguments.get("task", "")
        agent = self._delegate.agent
        tools = getattr(agent, "tools", None) or []
        native = self._delegate.native_tools or []
        sys = self._delegate.system_prompt

        if self._delegate.output_schema:
            if tools or native:
                resp = agent.loop(task, tools=tools, native_tools=native,
                                  output_schema=self._delegate.output_schema,
                                  **({"system": sys} if sys else {}))
            else:
                resp = agent.chat(task, output_schema=self._delegate.output_schema,
                                  **({"system": sys} if sys else {}))
            if resp.validated is False:
                raise ValueError(
                    f"Delegate agent '{agent.name}' failed to produce valid structured "
                    f"output for schema {self._delegate.output_schema!r}: "
                    f"{resp.validation_error}"
                )
            return resp.parsed if resp.parsed is not None else resp.content

        if tools or native:
            resp = agent.loop(task, tools=tools, native_tools=native,
                              **({"system": sys} if sys else {}))
        else:
            resp = agent.chat(task, **({"system": sys} if sys else {}))
        return resp.content

    # ------------------------------------------------------------------
    # Execution — async
    # ------------------------------------------------------------------

    async def arun(self, arguments: dict[str, Any], *, parent: Any = None) -> Any:
        if self._delegate is not None:
            return await self._arun_delegate(arguments, parent=parent)

        if self.func is None:
            raise RuntimeError(f"LazyTool '{self.name}' has no function to execute.")

        validated = _validate_and_coerce_arguments(self.func, arguments)
        result = self.func(**validated)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _arun_delegate(self, arguments: dict[str, Any], parent: Any) -> Any:
        task = arguments.get("task", "")
        agent = self._delegate.agent
        tools = getattr(agent, "tools", None) or []
        native = self._delegate.native_tools or []
        sys = self._delegate.system_prompt

        if self._delegate.output_schema:
            if tools or native:
                resp = await agent.aloop(task, tools=tools, native_tools=native,
                                         output_schema=self._delegate.output_schema,
                                         **({"system": sys} if sys else {}))
            else:
                resp = await agent.achat(task, output_schema=self._delegate.output_schema,
                                         **({"system": sys} if sys else {}))
            if resp.validated is False:
                raise ValueError(
                    f"Delegate agent '{agent.name}' failed to produce valid structured "
                    f"output for schema {self._delegate.output_schema!r}: "
                    f"{resp.validation_error}"
                )
            return resp.parsed if resp.parsed is not None else resp.content

        if tools or native:
            resp = await agent.aloop(task, tools=tools, native_tools=native,
                                     **({"system": sys} if sys else {}))
        else:
            resp = await agent.achat(task, **({"system": sys} if sys else {}))
        return resp.content

    # ------------------------------------------------------------------
    # Specialisation (clone with overrides)
    # ------------------------------------------------------------------

    def specialize(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        guidance: str | None = None,
        schema_mode: ToolSchemaMode | None = None,
        strict: bool | None = None,
    ) -> LazyTool:
        """Return a copy with selective overrides. Clears cached schema."""
        overrides = {k: v for k, v in {
            "name": name, "description": description, "guidance": guidance,
            "schema_mode": schema_mode, "strict": strict,
        }.items() if v is not None}
        # Clear the cached schema so it is rebuilt with the new name/description.
        if self._delegate is None:
            overrides["_compiled"] = None
        elif self._compiled is not None:
            # Delegate tools have fixed {task: str} parameters — only name/description/strict
            # can meaningfully change.  Patch the cached ToolDefinition directly so that
            # definition() reflects the new metadata without re-running the builder.
            td_patch: dict[str, Any] = {}
            if name is not None:
                td_patch["name"] = name
            if description is not None:
                td_patch["description"] = description
            if strict is not None:
                td_patch["strict"] = strict
            if td_patch:
                overrides["_compiled"] = replace(self._compiled, **td_patch)
        return replace(self, **overrides)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        kind = "delegate" if self._delegate else "function"
        return f"LazyTool(name={self.name!r}, kind={kind})"


# ---------------------------------------------------------------------------
# NormalizedToolSet — internal helper used by LazyAgent
# ---------------------------------------------------------------------------

@dataclass
class NormalizedToolSet:
    definitions: list[ToolDefinition]
    bridges: list[LazyTool]
    registry: dict[str, LazyTool]  # name → LazyTool, O(1) lookup

    @classmethod
    def from_list(cls, tools: list[LazyTool | ToolDefinition | dict]) -> NormalizedToolSet:
        """Normalise a mixed list of LazyTool / ToolDefinition / dict items.

        LazyTool items appear in all three collections (definitions, bridges, registry).
        ToolDefinition and dict items go into definitions only — they have no callable
        to invoke, so they rely on the tool_runner fallback or native provider handling.
        """
        definitions: list[ToolDefinition] = []
        bridges: list[LazyTool] = []
        registry: dict[str, LazyTool] = {}
        seen: set[str] = set()

        for item in tools:
            if isinstance(item, LazyTool):
                defn = item.definition()
                _check_unique(defn.name, seen)
                definitions.append(defn)
                bridges.append(item)
                registry[defn.name] = item
            elif isinstance(item, ToolDefinition):
                _check_unique(item.name, seen)
                definitions.append(item)
            elif isinstance(item, dict):
                defn = ToolDefinition(
                    name=item["name"],
                    description=item.get("description", ""),
                    parameters=item.get("parameters", {}),
                    strict=item.get("strict", False),
                )
                _check_unique(defn.name, seen)
                definitions.append(defn)
            else:
                raise TypeError(f"Expected LazyTool, ToolDefinition or dict, got {type(item)}")

        return cls(definitions=definitions, bridges=bridges, registry=registry)


__all__ = [
    "LazyTool",
    "NormalizedToolSet",
    "_params_to_schema",
    # re-exported from core.tool_schema
    "ToolArgumentValidationError",
    "ToolSchemaBuildError",
    "ToolSchemaMode",
    "ToolSourceStatus",
    "ToolCompileArtifact",
    "ArtifactStore",
    "InMemoryArtifactStore",
    "ToolSchemaBuilder",
]
