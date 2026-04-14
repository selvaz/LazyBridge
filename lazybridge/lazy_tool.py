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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lazybridge.core.tool_schema import (
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
from lazybridge.core.types import ToolDefinition

if TYPE_CHECKING:
    from lazybridge.lazy_agent import LazyAgent

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


@dataclass
class _PipelineConfig:
    """Metadata for pipeline tools (chain/parallel/agent_tool).

    Stored on the LazyTool so the pipeline can be serialized.
    Closures capture participants at runtime; this records the topology.
    """
    mode: str  # "chain" | "parallel" | "agent_tool"
    participants: tuple = ()
    native_tools: list = field(default_factory=list)
    combiner: str | None = None             # parallel only
    concurrency_limit: int | None = None    # parallel only
    step_timeout: float | None = None
    guidance: str | None = None
    # agent_tool-specific
    input_schema: type | None = None
    agent_tool_func: Callable | None = None
    agent_tool_provider: str | None = None
    agent_tool_model: str | None = None
    agent_tool_system: str | None = None


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
    _is_pipeline_tool: bool = field(default=False, repr=False, compare=False)
    _pipeline: _PipelineConfig | None = field(default=None, repr=False, compare=False)

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

    # ------------------------------------------------------------------
    # Persistence — save() / load()
    # ------------------------------------------------------------------

    _SENTINEL = "# LAZYBRIDGE_GENERATED_TOOL v1"
    _SENTINEL_V2 = "# LAZYBRIDGE_GENERATED_TOOL v2"

    def save(self, path: str) -> None:
        """Serialise this tool to a human-readable Python file.

        The generated file can be reloaded with :meth:`load`.  It contains
        the original function source (for ``from_function`` tools) or a
        ``LazyAgent`` constructor call (for ``from_agent`` tools), followed by
        a ``tool = LazyTool.from_function(...)`` or ``tool = agent.as_tool(...)``
        expression that recreates the tool on load.

        Parameters
        ----------
        path:
            Destination ``.py`` file.  Parent directories are created
            automatically.  Path must not contain ``..`` components.

        Raises
        ------
        ValueError
            If ``path`` contains ``..``, if the function cannot be
            introspected (lambda, REPL-defined, closure), or if called on
            an unsupported tool type.

        WARNING
        -------
        Never expose :meth:`load` as a ``LazyTool`` or agent tool.
        ``load()`` executes the target file — passing an LLM-controlled
        path to it is a code-execution vulnerability.
        """
        _validate_save_path(path)
        if self._is_pipeline_tool:
            if self._pipeline is None:
                raise ValueError(
                    f"LazyTool '{self.name}' is a pipeline tool without metadata. "
                    "Only pipelines created via chain()/parallel()/agent_tool() "
                    "on this version can be saved."
                )
            self._save_pipeline(path)
        elif self._delegate is not None:
            self._save_agent(path)
        else:
            self._save_function(path)

    def _save_function(self, path: str) -> None:
        if self.func is None:
            raise ValueError("Cannot save a LazyTool with no function.")
        # Reject lambdas up-front — they have no recoverable source identity
        if self.func.__name__ == "<lambda>":
            raise ValueError(
                "Cannot save lambda functions. Define the function in a .py "
                "file and use LazyTool.from_function() on it."
            )
        try:
            source = inspect.getsource(self.func)
        except (OSError, TypeError) as exc:
            raise ValueError(
                f"Cannot retrieve source for '{self.func.__name__}'. "
                "Functions defined in the REPL or as closures cannot be saved. "
                "Define the function in a .py file."
            ) from exc

        # Dedent: inspect.getsource may include class/method indentation
        source = inspect.cleandoc.__func__(source) if False else source  # noqa: SIM108
        import textwrap
        source = textwrap.dedent(source)

        imports = _extract_imports(self.func)

        # Build from_function() call — only include non-default args
        func_name = self.func.__name__
        call_args: list[str] = [f"    {func_name}"]
        if self.name != func_name:
            call_args.append(f"    name={self.name!r}")
        call_args.append(f"    description={self.description!r}")
        if self.guidance is not None:
            call_args.append(f"    guidance={self.guidance!r}")
        if self.schema_mode != ToolSchemaMode.SIGNATURE:
            call_args.append(f"    schema_mode=ToolSchemaMode.{self.schema_mode.name}")
        if self.strict:
            call_args.append(f"    strict=True")

        schema_mode_import = (
            "\nfrom lazybridge.lazy_tool import ToolSchemaMode"
            if self.schema_mode != ToolSchemaMode.SIGNATURE
            else ""
        )

        lines = [
            self._SENTINEL,
            f"# source: {_source_location(self.func)}",
            "",
        ]
        if imports:
            lines.extend(imports)
            lines.append("")
        lines += [
            f"from lazybridge import LazyTool{schema_mode_import}",
            "",
            "",
            source,
            "",
            "",
            "tool = LazyTool.from_function(",
        ]
        lines.extend(f"{arg}," for arg in call_args)
        lines.append(")")
        lines.append("")

        _write_file(path, "\n".join(lines))

    def _save_agent(self, path: str) -> None:
        agent = self._delegate.agent
        provider_alias = _provider_alias(agent)
        model = getattr(agent._executor, "model", None) or ""

        agent_args: list[str] = [f"    {provider_alias!r}"]
        agent_name = getattr(agent, "name", None)
        if agent_name:
            agent_args.append(f"    name={agent_name!r}")
        if model:
            agent_args.append(f"    model={model!r}")
        system = getattr(agent, "system", None)
        if system:
            agent_args.append(f"    system={system!r}")

        tool_args: list[str] = [f"    name={self.name!r}", f"    description={self.description!r}"]
        if self.guidance is not None:
            tool_args.append(f"    guidance={self.guidance!r}")
        if self.strict:
            tool_args.append(f"    strict=True")
        if self._delegate.output_schema is not None:
            schema_name = getattr(self._delegate.output_schema, "__name__", repr(self._delegate.output_schema))
            tool_args.append(f"    # output_schema={schema_name}  # re-attach after load if needed")
        if self._delegate.system_prompt is not None:
            tool_args.append(f"    system_prompt={self._delegate.system_prompt!r}")

        lines = [
            self._SENTINEL,
            "# NOTE: API keys are not serialized.",
            "# Set the appropriate environment variable before loading",
            "# (e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY, ...).",
            "",
            "from lazybridge import LazyAgent, LazyTool",
            "",
            "",
            "agent = LazyAgent(",
        ]
        lines.extend(f"{arg}," for arg in agent_args)
        lines += [
            ")",
            "",
            "tool = agent.as_tool(",
        ]
        lines.extend(f"{arg}," for arg in tool_args)
        lines += [")", ""]

        _write_file(path, "\n".join(lines))

    def _save_pipeline(self, path: str) -> None:
        """Code-gen a .py file that reconstructs a pipeline tool."""
        cfg = self._pipeline
        import textwrap

        # Collect imports and body separately, then merge at the end.
        # This avoids fragile index-based insertion.
        extra_imports: set[str] = set()
        body_lines: list[str] = []
        reconnect_needed: list[str] = []

        if cfg.mode == "agent_tool":
            # agent_tool: emit func source + schema source + LazyTool.agent_tool()
            # Do NOT emit participants — agent_tool() creates them internally.
            func = cfg.agent_tool_func
            schema = cfg.input_schema
            func_source = self._get_source_safe(func)
            schema_source = self._get_source_safe(schema)

            if func_source is None and func is not None:
                raise ValueError(
                    f"Cannot save agent_tool pipeline '{self.name}': "
                    f"function '{func.__name__}' source is not available "
                    "(closure, REPL, or C extension). Define it in a .py file."
                )
            if schema_source is None and schema is not None:
                raise ValueError(
                    f"Cannot save agent_tool pipeline '{self.name}': "
                    f"input_schema '{schema.__name__}' source is not available. "
                    "Define it in a .py file."
                )

            # Emit schema source (before func, since func may reference it)
            if schema_source:
                extra_imports.add("from pydantic import BaseModel, Field")
                # Also extract imports from the schema's module
                try:
                    extra_imports.update(_extract_imports(schema))
                except Exception:
                    pass
                body_lines.append(textwrap.dedent(schema_source))
                body_lines.append("")

            # Emit func source
            if func_source:
                try:
                    extra_imports.update(_extract_imports(func))
                except Exception:
                    pass
                body_lines.append(textwrap.dedent(func_source))
                body_lines.append("")

            func_name = func.__name__ if func else "None"
            schema_name = schema.__name__ if schema else "None"

            at_args = [f"    {func_name},"]
            at_args.append(f"    input_schema={schema_name},")
            if cfg.agent_tool_provider:
                at_args.append(f"    provider={cfg.agent_tool_provider!r},")
            if cfg.agent_tool_model:
                at_args.append(f"    model={cfg.agent_tool_model!r},")
            at_args.append(f"    name={self.name!r},")
            at_args.append(f"    description={self.description!r},")
            if cfg.guidance is not None:
                at_args.append(f"    guidance={cfg.guidance!r},")
            if cfg.step_timeout is not None:
                at_args.append(f"    step_timeout={cfg.step_timeout!r},")

            body_lines.append("tool = LazyTool.agent_tool(")
            body_lines.extend(at_args)
            body_lines.append(")")

        else:
            # chain / parallel: emit each participant, then the pipeline call
            var_names: list[str] = []
            emitted_funcs: set[str] = set()

            for i, p in enumerate(cfg.participants):
                var = f"participant_{i}"
                result = self._emit_participant(p, var, extra_imports, emitted_funcs)
                if result is None:
                    # Unsaveable — emit reconnect placeholder
                    p_name = getattr(p, "name", f"unknown_{i}")
                    body_lines.append(f"# {var}: <unsaveable: {p_name}>")
                    body_lines.append(f"# Pass via reconnect={{'{p_name}': ...}} when loading")
                    body_lines.append(f'{var} = reconnect["{p_name}"]')
                    body_lines.append("")
                    reconnect_needed.append(p_name)
                else:
                    body_lines.extend(result)
                    body_lines.append("")
                var_names.append(var)

            # Emit pipeline reconstruction call
            body_lines.append("")
            if cfg.mode == "parallel":
                call_args = [f"    {v}," for v in var_names]
                call_args.append(f"    name={self.name!r},")
                call_args.append(f"    description={self.description!r},")
                if cfg.combiner:
                    call_args.append(f"    combiner={cfg.combiner!r},")
                if cfg.concurrency_limit is not None:
                    call_args.append(f"    concurrency_limit={cfg.concurrency_limit!r},")
                if cfg.step_timeout is not None:
                    call_args.append(f"    step_timeout={cfg.step_timeout!r},")
                if cfg.guidance is not None:
                    call_args.append(f"    guidance={cfg.guidance!r},")
                body_lines.append("tool = LazyTool.parallel(")
                body_lines.extend(call_args)
                body_lines.append(")")

            elif cfg.mode == "chain":
                call_args = [f"    {v}," for v in var_names]
                call_args.append(f"    name={self.name!r},")
                call_args.append(f"    description={self.description!r},")
                if cfg.step_timeout is not None:
                    call_args.append(f"    step_timeout={cfg.step_timeout!r},")
                if cfg.guidance is not None:
                    call_args.append(f"    guidance={cfg.guidance!r},")
                body_lines.append("tool = LazyTool.chain(")
                body_lines.extend(call_args)
                body_lines.append(")")

        # Assemble final file: header + imports + body
        header = [
            self._SENTINEL_V2,
            f"# pipeline_mode: {cfg.mode}",
            "# NOTE: API keys are not serialized.",
            "# Set the appropriate environment variable before loading.",
        ]
        if reconnect_needed:
            header.append(f"# REQUIRES reconnect: {reconnect_needed}")
        header.append("")

        import_lines = ["from lazybridge import LazyAgent, LazyTool"]
        import_lines.extend(sorted(extra_imports))
        import_lines.append("")

        lines = header + import_lines + [""] + body_lines + [""]
        _write_file(path, "\n".join(lines))

    def _emit_participant(
        self, p: Any, var: str, extra_imports: set[str],
        emitted_funcs: set[str] | None = None,
    ) -> list[str] | None:
        """Generate code lines for a single participant, or None if unsaveable.

        Mutates ``extra_imports`` to add any imports the participant needs.
        Mutates ``emitted_funcs`` to track which function defs have already
        been emitted (avoids duplicate ``def`` blocks for reused functions).
        """
        import textwrap

        if emitted_funcs is None:
            emitted_funcs = set()

        # LazyAgent
        if hasattr(p, "_last_output"):
            provider_alias = _provider_alias(p)
            raw_model = getattr(p._executor, "model", None)
            model = raw_model if isinstance(raw_model, str) else None
            raw_system = getattr(p, "system", None)
            system = raw_system if isinstance(raw_system, str) else None
            raw_name = getattr(p, "name", None)
            name = raw_name if isinstance(raw_name, str) else None

            agent_args = [f"    {provider_alias!r}"]
            if name:
                agent_args.append(f"    name={name!r}")
            if model:
                agent_args.append(f"    model={model!r}")
            if system:
                agent_args.append(f"    system={system!r}")

            lines = [f"{var} = LazyAgent("]
            lines.extend(f"{arg}," for arg in agent_args)
            lines.append(")")
            return lines

        # LazyTool with function (from_function)
        if isinstance(p, LazyTool) and p.func is not None and not p._is_pipeline_tool:
            # Reject closures: free variables mean the source isn't self-contained
            if getattr(p.func, "__code__", None) and p.func.__code__.co_freevars:
                return None  # closure with captured variables — unsaveable
            source = self._get_source_safe(p.func)
            if source is None:
                return None  # source not retrievable — unsaveable
            source = textwrap.dedent(source)

            # Extract function's imports into the shared import set
            try:
                for imp in _extract_imports(p.func):
                    extra_imports.add(imp)
            except Exception:
                pass

            func_name = p.func.__name__

            # Only emit the function def once (skip if same function reused)
            func_id = id(p.func)
            lines: list[str] = []
            if func_id not in emitted_funcs:
                emitted_funcs.add(func_id)
                lines.append(source)
                lines.append("")

            call_args = [f"    {func_name}"]
            if p.name != func_name:
                call_args.append(f"    name={p.name!r}")
            call_args.append(f"    description={p.description!r}")
            if p.guidance is not None:
                call_args.append(f"    guidance={p.guidance!r}")

            lines.append(f"{var} = LazyTool.from_function(")
            lines.extend(f"{arg}," for arg in call_args)
            lines.append(")")
            return lines

        # LazyTool with delegate (from_agent)
        if isinstance(p, LazyTool) and p._delegate is not None:
            agent = p._delegate.agent
            inner_lines = self._emit_participant(
                agent, f"{var}_agent", extra_imports, emitted_funcs,
            )
            if inner_lines is None:
                return None
            tool_args = [f"    name={p.name!r}", f"    description={p.description!r}"]
            if p.guidance is not None:
                tool_args.append(f"    guidance={p.guidance!r}")
            lines = inner_lines + ["", f"{var} = {var}_agent.as_tool("]
            lines.extend(f"{arg}," for arg in tool_args)
            lines.append(")")
            return lines

        # Nested pipeline — recursive save
        if isinstance(p, LazyTool) and p._is_pipeline_tool and p._pipeline is not None:
            return self._emit_nested_pipeline(p, var, extra_imports, emitted_funcs)

        # Unknown — unsaveable
        return None

    def _emit_nested_pipeline(
        self, p: "LazyTool", var: str, extra_imports: set[str],
        emitted_funcs: set[str] | None = None,
    ) -> list[str] | None:
        """Recursively emit a nested pipeline as inline code."""
        cfg = p._pipeline
        lines: list[str] = []
        sub_vars: list[str] = []

        for i, sub_p in enumerate(cfg.participants):
            sub_var = f"{var}_p{i}"
            result = self._emit_participant(sub_p, sub_var, extra_imports, emitted_funcs)
            if result is None:
                return None  # unsaveable participant → whole nested pipeline unsaveable
            lines.extend(result)
            lines.append("")
            sub_vars.append(sub_var)

        if cfg.mode == "parallel":
            call_args = [f"    {v}," for v in sub_vars]
            call_args.append(f"    name={p.name!r},")
            call_args.append(f"    description={p.description!r},")
            if cfg.combiner:
                call_args.append(f"    combiner={cfg.combiner!r},")
            if cfg.concurrency_limit is not None:
                call_args.append(f"    concurrency_limit={cfg.concurrency_limit!r},")
            lines.append(f"{var} = LazyTool.parallel(")
            lines.extend(call_args)
            lines.append(")")
        elif cfg.mode == "chain":
            call_args = [f"    {v}," for v in sub_vars]
            call_args.append(f"    name={p.name!r},")
            call_args.append(f"    description={p.description!r},")
            lines.append(f"{var} = LazyTool.chain(")
            lines.extend(call_args)
            lines.append(")")
        else:
            return None  # unsupported nested mode

        return lines

    @staticmethod
    def _get_source_safe(obj: Any) -> str | None:
        """Try to get source code; return None on failure."""
        if obj is None:
            return None
        try:
            return inspect.getsource(obj)
        except (OSError, TypeError):
            return None

    @classmethod
    def load(
        cls,
        path: str,
        *,
        reconnect: dict[str, Any] | None = None,
        base_dir: str | None = None,
    ) -> "LazyTool":
        """Load a :class:`LazyTool` from a file previously created by :meth:`save`.

        Parameters
        ----------
        path:
            Path to the ``.py`` file.  Must contain ``..``-free path and must
            have been generated by :meth:`save` (sentinel header required).
        base_dir:
            If provided, the resolved *path* must reside inside this
            directory.  Use this when the path may originate from
            user-controlled input to prevent out-of-scope file access.

        Returns
        -------
        LazyTool
            The ``tool`` variable defined in the file.

        Raises
        ------
        ValueError
            If the path contains ``..``, resolves outside *base_dir*, if the
            sentinel header is missing, or if no ``tool`` variable of type
            ``LazyTool`` is found.

        WARNING
        -------
        Never expose this method as a ``LazyTool`` or agent tool.
        ``load()`` executes the target file — passing an LLM-controlled
        path to it is a code-execution vulnerability.
        """
        _validate_load_path(path, base_dir=base_dir)

        resolved = str(Path(path).resolve())
        with open(resolved, encoding="utf-8") as fh:
            first_line = fh.readline().rstrip("\n")

        if first_line not in (cls._SENTINEL, cls._SENTINEL_V2):
            raise ValueError(
                f"File {path!r} does not appear to have been generated by "
                "LazyTool.save() (missing sentinel header). "
                "load() only accepts files produced by LazyTool.save()."
            )

        import importlib.util
        spec = importlib.util.spec_from_file_location("_lazybridge_tool_module", resolved)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load module from {path!r}")
        module = importlib.util.module_from_spec(spec)
        # Inject reconnect dict for pipeline files that reference it
        module.reconnect = reconnect or {}  # type: ignore[attr-defined]
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        tool = getattr(module, "tool", None)
        if not isinstance(tool, cls):
            raise ValueError(
                f"No LazyTool named 'tool' found in {path!r}. "
                "The file must assign a LazyTool to a variable named 'tool'."
            )
        return tool

    def __repr__(self) -> str:
        kind = "delegate" if self._delegate else "function"
        return f"LazyTool(name={self.name!r}, kind={kind})"

    # ------------------------------------------------------------------
    # Pipeline factory methods
    # ------------------------------------------------------------------

    @classmethod
    def parallel(
        cls,
        *participants: Any,
        name: str,
        description: str,
        combiner: str = "concat",
        native_tools: list | None = None,
        session: Any | None = None,
        guidance: str | None = None,
        concurrency_limit: int | None = None,
        step_timeout: float | None = None,
    ) -> "LazyTool":
        """Fan-out pipeline tool: all participants run concurrently on the same task.

        Participants are cloned per invocation for call-state isolation —
        ``participant._last_output`` on the original is unchanged after the run.
        No ``LazySession`` required.

        Parameters
        ----------
        *participants:
            LazyAgent or LazyTool instances.
        name, description:
            Tool name and description exposed to the orchestrator LLM.
        combiner:
            ``"concat"`` (default) — outputs joined with agent-name headers.
            ``"last"`` — only the last result is returned.
        native_tools:
            Optional list of NativeTool values passed to all agent participants.
        session:
            *Validation-only.* If provided, raises ``ValueError`` if any agent
            participant is bound to a conflicting session. Does **not** register
            agents. Does **not** modify the graph.
        guidance:
            Optional hint injected into the tool description for the LLM.
        concurrency_limit:
            Maximum number of participants that execute simultaneously.
            ``None`` (default) — all run at once.
            Use when API rate limits or resource constraints apply.
        step_timeout:
            Per-participant timeout in seconds.  Exceeded participants return
            ``"[ERROR: TimeoutError: ...]"`` in concat mode.
            ``None`` (default) — no timeout.
        """
        from lazybridge.pipeline_builders import (
            build_parallel_func,
            _resolve_participant,
            _validate_session_compatibility,
        )
        if not participants:
            raise ValueError("parallel() requires at least one participant.")
        if combiner not in ("concat", "last"):
            raise ValueError(f"Invalid combiner {combiner!r}. Use 'concat' or 'last'.")
        _validate_session_compatibility(participants, session)
        _native = list(native_tools or [])

        def _run(task: str) -> str:
            inv = [_resolve_participant(p) for p in participants]
            return build_parallel_func(inv, _native, combiner, concurrency_limit, step_timeout)(task)

        tool = cls.from_function(_run, name=name, description=description, guidance=guidance)
        tool._is_pipeline_tool = True
        tool._pipeline = _PipelineConfig(
            mode="parallel",
            participants=participants,
            native_tools=_native,
            combiner=combiner,
            concurrency_limit=concurrency_limit,
            step_timeout=step_timeout,
            guidance=guidance,
        )
        return tool

    @classmethod
    def chain(
        cls,
        *participants: Any,
        name: str,
        description: str,
        native_tools: list | None = None,
        session: Any | None = None,
        guidance: str | None = None,
        step_timeout: float | None = None,
    ) -> "LazyTool":
        """Sequential pipeline tool: participants run in order, each receiving
        the previous output as context (agent→agent) or as the new task (tool→agent).

        Participants are cloned per invocation for call-state isolation —
        ``participant._last_output`` on the original is unchanged after the run.
        No ``LazySession`` required.

        Parameters
        ----------
        *participants:
            LazyAgent or LazyTool instances, in execution order.
        name, description:
            Tool name and description exposed to the orchestrator LLM.
        native_tools:
            Optional list of NativeTool values passed to all agent steps.
        session:
            *Validation-only.* If provided, raises ``ValueError`` if any agent
            participant is bound to a conflicting session. Does **not** register
            agents. Does **not** modify the graph.
        guidance:
            Optional hint injected into the tool description for the LLM.

        Notes
        -----
        Handoff semantics (source: pipeline_builders.build_achain_func):
            agent → agent: previous agent's output is injected as context;
                           the original task string is kept as the message.
            tool  → agent: tool's output becomes the next agent's task directly.

        Async-under-the-hood: the chain runs via ``build_achain_func`` (uses
        ``achat``/``aloop``/``ajson``) so the event loop is never blocked.
        ``run()`` drives it with ``run_async()``; ``arun()`` awaits it directly.

        Because clones execute the run, ``participant._last_output`` on the
        original object is ``None`` after the call. Use the return value of
        ``tool.run()`` or ``output_schema`` on the last step instead.

        Parameters (additional)
        -----------------------
        step_timeout:
            Per-step timeout in seconds.  ``asyncio.TimeoutError`` is raised
            if a step exceeds the limit.  ``None`` (default) — no timeout.
        """
        from lazybridge.pipeline_builders import (
            build_achain_func,
            _resolve_participant,
            _validate_session_compatibility,
        )
        if not participants:
            raise ValueError("chain() requires at least one participant.")
        _validate_session_compatibility(participants, session)
        _native = list(native_tools or [])

        def _run(task: str) -> Any:
            from lazybridge.lazy_run import run_async
            inv = [_resolve_participant(p) for p in participants]
            return run_async(build_achain_func(inv, _native, step_timeout)(task))

        tool = cls.from_function(_run, name=name, description=description, guidance=guidance)
        tool._is_pipeline_tool = True
        tool._pipeline = _PipelineConfig(
            mode="chain",
            participants=participants,
            native_tools=_native,
            step_timeout=step_timeout,
            guidance=guidance,
        )
        return tool

    @classmethod
    def agent_tool(
        cls,
        func: Callable,
        *,
        input_schema: type,
        provider: str = "anthropic",
        model: str | None = None,
        name: str | None = None,
        description: str | None = None,
        guidance: str | None = None,
        system: str | None = None,
        step_timeout: float | None = None,
        **agent_kwargs: Any,
    ) -> "LazyTool":
        """Create an intelligent tool by chaining a sub-agent → function.

        The sub-agent receives a natural-language task, produces structured
        output matching ``input_schema`` (a Pydantic BaseModel whose fields
        correspond to ``func``'s parameters), and the chain automatically
        passes ``model_dump()`` to the function for deterministic execution.

        This is the universal pattern for turning any function into an
        LLM-callable tool with validated parameter construction::

            from lazybridge.lazy_tool import LazyTool
            from pydantic import BaseModel, Field

            class FitModelInput(BaseModel):
                family: str = Field(description="Model family: ols, arima, garch, markov")
                target_col: str = Field(description="Target column in the dataset")
                dataset_name: str | None = None

            fit_tool = LazyTool.agent_tool(
                fit_model,
                input_schema=FitModelInput,
                provider="anthropic",
                name="fit_model",
                description="Fit a statistical model with LLM-guided parameter selection",
            )

            # The orchestrator calls fit_tool with {"task": "fit GARCH to SPY returns"}
            # → sub-agent produces FitModelInput(family="garch", target_col="value", ...)
            # → fit_model(**input.model_dump()) executes deterministically

        Parameters
        ----------
        func:
            The function to execute.  Its signature must accept the fields
            of ``input_schema`` as keyword arguments.
        input_schema:
            Pydantic BaseModel class.  The sub-agent's ``output_schema``.
        provider:
            LLM provider for the sub-agent.
        model:
            Model override for the sub-agent.
        name:
            Tool name exposed to the orchestrator (defaults to func.__name__).
        description:
            Tool description (defaults to func docstring).
        guidance:
            Optional guidance hint for the orchestrator.
        system:
            System prompt for the sub-agent.  Defaults to a prompt built
            from ``input_schema``'s field descriptions.
        step_timeout:
            Per-step timeout in seconds for the chain.
        **agent_kwargs:
            Extra keyword arguments forwarded to LazyAgent constructor.

        Returns
        -------
        LazyTool
            A pipeline tool with schema ``{"task": str}``.
        """
        from lazybridge.lazy_agent import LazyAgent

        tool_name = name or getattr(func, "__name__", "agent_tool")
        tool_desc = description or getattr(func, "__doc__", None) or f"Intelligent {tool_name} tool"

        # Build a default system prompt from the schema if not provided
        if system is None:
            schema_info = input_schema.model_json_schema()
            props = schema_info.get("properties", {})
            required = schema_info.get("required", [])
            field_lines = []
            for fname, finfo in props.items():
                desc = finfo.get("description", "")
                req = "(required)" if fname in required else "(optional)"
                field_lines.append(f"  - {fname} {req}: {desc}")
            fields_text = "\n".join(field_lines) if field_lines else "  (no fields)"

            system = (
                f"You are a parameter extraction agent for the '{tool_name}' function.\n"
                f"Given a natural language task, produce the correct parameters.\n\n"
                f"Parameters:\n{fields_text}\n\n"
                f"Output ONLY the structured parameters. Do not explain or narrate."
            )

        # Create the sub-agent (no tools — pure structured output)
        sub_agent = LazyAgent(
            provider,
            model=model,
            name=f"{tool_name}_params",
            description=f"Parameter builder for {tool_name}",
            system=system,
            output_schema=input_schema,
            **agent_kwargs,
        )

        # Wrap the function as a LazyTool
        func_tool = cls.from_function(func, name=f"{tool_name}_exec", description=tool_desc)

        # Chain: sub_agent (produces typed Pydantic) → func_tool (receives model_dump())
        pipeline = cls.chain(
            sub_agent,
            func_tool,
            name=tool_name,
            description=tool_desc,
            guidance=guidance,
            step_timeout=step_timeout,
        )
        # Override the chain's _pipeline with agent_tool metadata
        pipeline._pipeline = _PipelineConfig(
            mode="agent_tool",
            participants=(sub_agent, func_tool),
            native_tools=[],
            step_timeout=step_timeout,
            guidance=guidance,
            input_schema=input_schema,
            agent_tool_func=func,
            agent_tool_provider=provider,
            agent_tool_model=model,
            agent_tool_system=system,
        )
        return pipeline


# ---------------------------------------------------------------------------
# save/load helpers — module-private
# ---------------------------------------------------------------------------

def _validate_save_path(path: str) -> None:
    p = Path(path)
    if ".." in p.parts:
        raise ValueError(
            f"Path {path!r} contains '..' components. "
            "Paths with '..' are not allowed in LazyTool.save()."
        )
    if p.suffix.lower() != ".py":
        raise ValueError(f"Path {path!r} must end in '.py'.")


def _validate_load_path(path: str, *, base_dir: str | None = None) -> None:
    p = Path(path)
    if ".." in p.parts:
        raise ValueError(
            f"Path {path!r} contains '..' components. "
            "Paths with '..' are not allowed in LazyTool.load()."
        )
    if p.suffix.lower() != ".py":
        raise ValueError(f"Path {path!r} must end in '.py'.")
    if not p.exists():
        raise FileNotFoundError(f"No such file: {path!r}")
    if base_dir is not None:
        resolved = p.resolve()
        base = Path(base_dir).resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            raise ValueError(
                f"Path {path!r} resolves outside the allowed "
                f"base directory {base_dir!r}."
            ) from None


def _write_file(path: str, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _source_location(func: Callable) -> str:
    try:
        src_file = inspect.getfile(func)
        line = inspect.getsourcelines(func)[1]
        return f"{src_file}::{func.__name__} (line {line})"
    except (OSError, TypeError):
        return f"<unknown>::{func.__name__}"


def _extract_imports(func: Callable) -> list[str]:
    """Extract imports from ``func``'s source file that ``func`` actually uses.

    Uses AST analysis: parse the function/class source to find all referenced
    names, then intersect with the file's top-level imports.  Only imports
    whose provided names appear in the function body are returned.

    This avoids polluting generated files with unrelated imports from the
    source file (e.g. ``import pytest`` when the function only uses ``int``).
    """
    import ast
    import textwrap

    # 1. Get function source and find referenced names
    try:
        func_source = inspect.getsource(func)
    except (OSError, TypeError):
        return []
    func_source = textwrap.dedent(func_source)
    try:
        func_tree = ast.parse(func_source)
    except SyntaxError:
        return []

    referenced_names: set[str] = set()
    for node in ast.walk(func_tree):
        if isinstance(node, ast.Name):
            referenced_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            # For 'os.path.join', extract root name 'os'
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                referenced_names.add(root.id)

    if not referenced_names:
        return []

    # 2. Parse the source file's top-level imports
    try:
        src_file = inspect.getfile(func)
    except (OSError, TypeError):
        return []
    try:
        with open(src_file, encoding="utf-8") as fh:
            file_source = fh.read()
    except OSError:
        return []
    try:
        file_tree = ast.parse(file_source)
    except SyntaxError:
        return []

    # Map: provided_name -> import source line
    # An import line is included if ANY of the names it provides are referenced.
    import_line_names: list[tuple[str, set[str]]] = []
    for node in ast.iter_child_nodes(file_tree):
        if isinstance(node, ast.Import):
            line = ast.get_source_segment(file_source, node)
            if line is None:
                continue
            if "lazybridge" in line or "from __future__" in line:
                continue
            provided = {a.asname or a.name.split(".")[0] for a in node.names}
            import_line_names.append((line, provided))
        elif isinstance(node, ast.ImportFrom):
            line = ast.get_source_segment(file_source, node)
            if line is None:
                continue
            if node.module and ("lazybridge" in node.module or "__future__" in node.module):
                continue
            provided = {a.asname or a.name for a in node.names}
            import_line_names.append((line, provided))

    # 3. Keep only imports whose provided names intersect with referenced names
    needed: list[str] = []
    seen: set[str] = set()
    for line, provided in import_line_names:
        if provided & referenced_names and line not in seen:
            seen.add(line)
            needed.append(line)

    return needed


def _provider_alias(agent: Any) -> str:
    """Derive the string alias for the agent's provider (e.g. 'anthropic')."""
    try:
        class_name = type(agent._executor.provider).__name__
        # e.g. "AnthropicProvider" → "anthropic"
        return class_name.replace("Provider", "").lower()
    except AttributeError:
        return "anthropic"  # safe fallback


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


def _clone_delegate_tool_for_invocation(tool: "LazyTool") -> "LazyTool":
    """Clone the delegate agent inside a LazyTool.from_agent() tool.

    FRIEND MODULE CONTRACT:
      Called exclusively by pipeline_builders._resolve_participant().
      Encapsulates _DelegateConfig access here in its home module.
      If _DelegateConfig is renamed or restructured, only this function needs updating.

    Parameters
    ----------
    tool:
        A LazyTool with a non-None _delegate (created via from_agent()).
    """
    from lazybridge.pipeline_builders import _clone_for_invocation
    from dataclasses import replace
    agent_clone = _clone_for_invocation(tool._delegate.agent)  # type: ignore[union-attr]
    new_delegate = replace(tool._delegate, agent=agent_clone)   # type: ignore[union-attr]
    return replace(tool, _delegate=new_delegate)


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
