"""lazybridge — Agent = Engine + Tools + State.

Every Agent has the same shape.  Only the engine changes::

    from lazybridge import Agent, LLMEngine, Plan, Step, Session, Tool, from_step

    # --- Wrap Python functions as tools with explicit names ---

    search = Tool.wrap(search_web, name="search", description="Search the web.")

    # --- Build sub-agents with explicit names ---

    researcher = Agent(
        name="research",
        engine=LLMEngine("claude-haiku-4-5", system="You are a research expert."),
        tools=[search],
    )
    writer = Agent(
        name="write",
        engine=LLMEngine("gpt-5.4-mini", system="You are a concise technical writer."),
    )

    # --- Deterministic orchestrator: Plan engine ---

    pipeline = Agent(
        name="pipeline",
        engine=Plan(
            Step("research"),
            Step("write", task=from_prev, context=from_step("research")),
        ),
        tools=[researcher, writer],   # sub-agents passed directly
        session=Session(),
    )

    # --- Dynamic orchestrator: LLM engine ---

    orchestrator = Agent(
        name="orchestrator",
        engine=LLMEngine("claude-opus-4-7"),
        tools=[researcher, writer],
        session=Session(),
    )

    result = pipeline("AI trends 2026").text()

**String shortcut** — ``Agent("claude-opus-4-7")`` expands to
``Agent(engine=LLMEngine("claude-opus-4-7"))``.  Use the explicit
``LLMEngine(...)`` form when you need ``system=``, ``max_turns=``,
``thinking=``, or other engine-level config.

**The name chain** — ``Agent(name=...)`` is the authoritative key that
connects every part of the system::

    Agent(name="research")      →  tool map key when passed in tools=[researcher]
    Step("research")            →  looks up "research" in tool map ✓
    routes={"research": pred}   →  routes to the step named "research" ✓
    from_step("research")       →  reads output of step "research" (in-Plan) ✓
    from_agent("research")      →  reads last stored output of "research" (cross-run) ✓
    from_memory("research")     →  reads live memory of "research" ✓

**Tool.wrap classmethod** — the canonical way to wrap a Python function::

    search = Tool.wrap(search_web, name="search", description="...")

``Tool.wrap`` is an alternative constructor on the :class:`Tool` class
(same idiom as :meth:`dict.fromkeys` or :meth:`Path.cwd`).  It accepts
a callable, an :class:`Agent`, or an existing :class:`Tool` and
dispatches to the right wrapping path.  ``name`` is required for
callables and becomes the stable key in the tool map.  Raw callables
in ``tools=[fn]`` still work for backward compatibility, but
``Tool.wrap`` is the preferred form in new code.

The module-level :func:`tool` (lowercase) is a backwards-compat alias
that simply calls ``Tool.wrap``; it stays indefinitely so existing
imports keep working, but new code should reach for ``Tool.wrap``.

**Advanced alias / backward compat** — ``.as_tool("alias")`` remains
available when you need a name different from the agent's own::

    tools=[researcher.as_tool("deep_research")]

Direct ``tools=[agent]`` is the canonical composition style;
``.as_tool()`` is the advanced / compatibility path.

**Choosing between sentinels** — inside a single Plan, ``from_step`` is
the standard choice: it reads from in-memory step history with no
external dependency.  ``from_agent`` is for cross-run or cross-plan
data: last known output persisted in a shared Store.
``from_memory`` reads the agent's live conversation history.
"""

# Single-source the version from the installed distribution metadata so
# ``__version__`` and ``importlib.metadata.version("lazybridge")`` can
# never disagree.  Falls back to a literal only when the package isn't
# installed (running from a source tree without ``pip install -e .``).
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    try:
        __version__ = _dist_version("lazybridge")
    except PackageNotFoundError:  # pragma: no cover — uninstalled source tree
        __version__ = "0.7.9"
    del _dist_version, PackageNotFoundError
except ImportError:  # pragma: no cover — Python < 3.8, not supported
    __version__ = "0.7.9"

__stability__ = "alpha"

# Public API.  Symbols a user constructs, passes as a kwarg, or catches as
# an exception are re-exported from this top-level module.  Internals
# (provider request/response types, dispatcher helpers like _wrap_tool /
# build_tool_map, attribute-only types like EnvelopeMetadata / StoreEntry,
# rarely-subclassed Protocols) live under their submodules and stay
# importable as ``from lazybridge.X import Y`` for advanced callers.

# Primary surface
from lazybridge.agent import Agent, ParallelAgent

# Provider entry points
from lazybridge.core.providers import BaseProvider, UnsupportedFeatureError
from lazybridge.core.types import (
    AudioContent,
    CacheConfig,
    ImageContent,
    NativeTool,
)

# Engines (HumanEngine, SupervisorEngine, eval helpers, and OTelExporter
# live under ``lazybridge.ext.{hil,evals,otel}``).
from lazybridge.engines.llm import LLMEngine, StreamStallError, ToolTimeoutError

#: Snapshot of the model-string → provider routing aliases recognised
#: by ``LLMEngine`` at import time.  Documenting the canonical surface
#: at the top level so callers (and LLM assistants) can validate a
#: user-supplied alias without reaching into the engine internals.
#: For runtime extension use ``LLMEngine.register_provider_alias`` /
#: ``LLMEngine.provider_aliases()`` instead — this constant is a
#: snapshot, not a live view.
PROVIDER_ALIASES: dict[str, str] = LLMEngine.provider_aliases()
from lazybridge.engines.plan import (
    ConcurrentPlanRunError,
    Plan,
    PlanCompileError,
    PlanPaused,
    PlanRuntimeError,
    Step,
)
from lazybridge.envelope import Envelope

# Exporters (core).  ``OTelExporter`` lives in ``lazybridge.ext.otel``.
from lazybridge.exporters import (
    CallbackExporter,
    ConsoleExporter,
    EventExporter,
    FilteredExporter,
    JsonFileExporter,
    StructuredLogExporter,
)

# Graph
from lazybridge.graph import GraphSchema
from lazybridge.guardrails import ContentGuard, Guard, GuardAction, GuardChain, GuardError, LLMGuard
from lazybridge.memory import Memory
from lazybridge.predicates import when
from lazybridge.sentinels import (
    from_agent,
    from_memory,
    from_parallel,
    from_parallel_all,
    from_prev,
    from_start,
    from_step,
)
from lazybridge.session import EventLog, EventType, Session
from lazybridge.store import Store
from lazybridge.testing import MockAgent
from lazybridge.tools import Tool, ToolProvider, tool

__all__ = [
    # Primary API
    "Agent",
    "ParallelAgent",
    # Envelope
    "Envelope",
    # Sentinels
    "from_prev",
    "from_start",
    "from_step",
    "from_parallel",
    "from_parallel_all",
    "from_memory",
    "from_agent",
    # Tools
    "Tool",
    "tool",
    "ToolProvider",
    # Native tools (provider-hosted, e.g. web search)
    "NativeTool",
    # Predicates DSL (for Step.routes)
    "when",
    # State
    "Memory",
    "Store",
    # Session / Observability
    "Session",
    "EventLog",
    "EventType",
    # Guardrails
    "Guard",
    "GuardAction",
    "GuardError",
    "ContentGuard",
    "GuardChain",
    "LLMGuard",
    # Engines (HumanEngine, SupervisorEngine in lazybridge.ext.hil)
    "LLMEngine",
    "PROVIDER_ALIASES",
    "Plan",
    "Step",
    "ConcurrentPlanRunError",
    "PlanCompileError",
    "PlanPaused",
    "PlanRuntimeError",
    "ToolTimeoutError",
    "StreamStallError",
    # Graph
    "GraphSchema",
    # Exporters
    "EventExporter",
    "CallbackExporter",
    "ConsoleExporter",
    "FilteredExporter",
    "JsonFileExporter",
    "StructuredLogExporter",
    # Provider entry point (custom adapters)
    "BaseProvider",
    "UnsupportedFeatureError",
    # Multimodal content blocks
    "ImageContent",
    "AudioContent",
    # Cache configuration (kept — internal repr for LLMEngine cache)
    "CacheConfig",
    # Testing
    "MockAgent",
]
