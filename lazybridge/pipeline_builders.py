"""lazybridge/pipeline_builders.py

Pipeline execution logic extracted from lazy_session.py.
Neutral module to break circular imports.

Import topology (no cycles):
  lazy_session.py   → pipeline_builders  (as_tool delegates here)
  lazy_tool.py      → pipeline_builders  (parallel/chain use builders — lazy import)
  pipeline_builders → lazy_run           (lazy import inside closures)
  pipeline_builders → lazy_context       (lazy import inside closures)
  pipeline_builders → lazy_tool          (lazy import inside _resolve_participant)

NOTE: This module must NOT import from lazy_agent, lazy_session, or lazy_tool
at module load time. All cross-module imports are deferred inside function bodies.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

_logger = logging.getLogger(__name__)
from collections.abc import Callable

# ---------------------------------------------------------------------------
# _ChainState — moved from lazy_session.py
# ---------------------------------------------------------------------------


class _ChainState:
    """Internal state propagated between chain steps.

    Moved from lazy_session.py. lazy_session re-exports this class for
    backward compatibility (existing test imports remain valid).

    Attributes
    ----------
    text : str
        Text representation of this step's output. Always available.
        Used as the next agent's task when ctx is None (tool → agent).
    typed : Any | None
        Typed Pydantic object produced by this step, or None.
        Only set when the step was an agent with output_schema active.
    ctx : LazyContext | None
        Not None when the previous step was an agent — inject context
        into the next agent's system prompt, keep original task.
        None when the previous step was a tool (or first step) — use
        text directly as the next agent's task.

    Handoff semantics (decided by ctx):
        ctx is not None  →  agent → agent  →  inject context, keep original task
        ctx is None      →  tool  → agent  →  use text as new task
    """

    __slots__ = ("ctx", "text", "typed")

    def __init__(self, text: str, typed: Any, ctx: Any) -> None:
        self.text = text
        self.typed = typed
        self.ctx = ctx


# ---------------------------------------------------------------------------
# Shared checkpoint helpers (used by both sync and async chain builders)
# ---------------------------------------------------------------------------


def _restore_checkpoint(store: Any, ckpt_key: str, task: str) -> tuple[int, _ChainState]:
    """Restore chain state from checkpoint. Returns (start_step, state).

    Handles validation and semantic resume (handoff_mode).
    Falls back to step 0 on missing or malformed checkpoints.
    """
    if store is None:
        return 0, _ChainState(text=task, typed=None, ctx=None)

    saved = store.read(ckpt_key)
    if saved is None:
        return 0, _ChainState(text=task, typed=None, ctx=None)

    if not (isinstance(saved, dict) and isinstance(saved.get("step"), int) and "output" in saved):
        _logger.warning("Ignoring malformed checkpoint for %r: %r", ckpt_key, saved)
        return 0, _ChainState(text=task, typed=None, ctx=None)

    from lazybridge.lazy_context import LazyContext

    start_step = saved["step"] + 1
    _handoff = saved.get("handoff_mode", "text_task")
    if _handoff == "agent_context":
        _orig = saved.get("original_task", saved["output"])
        state = _ChainState(
            text=str(_orig),
            typed=None,
            ctx=LazyContext.from_text(f"[resumed previous output]\n{saved['output']}"),
        )
    else:
        state = _ChainState(text=saved["output"], typed=None, ctx=None)
    return start_step, state


def _save_checkpoint(store: Any, ckpt_key: str, step: int, state: _ChainState, task: str) -> None:
    """Save checkpoint with handoff semantics."""
    if store is None:
        return
    store.write(
        ckpt_key,
        {
            "step": step,
            "output": state.text,
            "original_task": task,
            "handoff_mode": "agent_context" if state.ctx is not None else "text_task",
        },
    )


def _clear_checkpoint(store: Any, ckpt_key: str) -> None:
    """Remove checkpoint after successful completion."""
    if store is None:
        return
    store.write(ckpt_key, None)


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def build_parallel_func(
    parts: list,
    native_tools: list,
    combiner: str,
    concurrency_limit: int | None = None,
    step_timeout: float | None = None,
) -> Callable[[str], str]:
    """Return a closure that runs parts concurrently when called with (task).

    Dispatch is character-for-character identical to LazySession.as_tool(mode='parallel'):
      output_schema set       → ajson
      has_tools/native_tools  → aloop
      bare agent              → achat
      LazyTool                → arun({"task": task})

    Parameters
    ----------
    concurrency_limit:
        Maximum number of participants that run simultaneously.
        None (default) means no limit — all run at once.
        Use when API rate limits or resource constraints apply.
    step_timeout:
        Per-participant timeout in seconds.  If a participant exceeds this,
        its result becomes ``asyncio.TimeoutError`` (rendered as
        ``"[ERROR: TimeoutError: ...]"`` in concat mode).
        None (default) means no timeout.
    """

    def _run_parallel(task: str) -> str:
        from lazybridge.lazy_run import run_async

        async def _gather() -> str:
            coros = []
            for p in parts:
                if hasattr(p, "achat"):  # LazyAgent
                    kw = {"native_tools": native_tools} if native_tools else {}
                    schema = getattr(p, "output_schema", None)
                    has_tools = bool(
                        getattr(p, "tools", None) or getattr(p, "native_tools", None) or kw.get("native_tools")
                    )
                    if schema is not None:
                        coros.append(p.ajson(task, schema, **kw))
                    elif has_tools:
                        coros.append(p.aloop(task, **kw))
                    else:
                        coros.append(p.achat(task, **kw))
                else:  # LazyTool
                    coros.append(p.arun({"task": task}))

            if step_timeout is not None:
                coros = [asyncio.wait_for(c, timeout=step_timeout) for c in coros]

            if concurrency_limit is not None:
                sem = asyncio.Semaphore(concurrency_limit)

                async def _guarded(c: Any) -> Any:
                    async with sem:
                        return await c

                coros = [_guarded(c) for c in coros]

            results = await asyncio.gather(*coros, return_exceptions=True)
            if not results:
                return ""

            def _to_text(r: Any) -> str:
                if isinstance(r, BaseException):
                    return f"[ERROR: {type(r).__name__}: {r}]"
                if hasattr(r, "model_dump_json"):
                    return r.model_dump_json(indent=2)
                if hasattr(r, "content"):
                    return r.content
                return str(r)

            if combiner == "last":
                return _to_text(results[-1])
            return "\n\n".join(f"[{getattr(p, 'name', '?')}]\n{_to_text(r)}" for p, r in zip(parts, results))

        return run_async(_gather())

    return _run_parallel


def build_chain_func(
    parts: list,
    native_tools: list,
    *,
    store: Any | None = None,
    chain_id: str = "chain",
    run_id: str | None = None,
) -> Callable[[str], Any]:
    """Return a closure that runs parts sequentially when called with (task).

    Handoff semantics identical to LazySession.as_tool(mode='chain'):
      prev = agent (ctx is not None) → inject context, keep original task
      prev = tool  (ctx is None)     → state.text becomes task

    Parameters
    ----------
    store:
        Optional LazyStore for checkpoint persistence.  When provided,
        state is saved after each step so the chain can resume from the
        last completed step on re-execution.
    chain_id:
        Namespace for checkpoint keys in the store.  Use distinct ids
        when multiple chains share the same store.
    run_id:
        Optional run identifier for checkpoint isolation.  When provided,
        the checkpoint key becomes ``_ckpt:{chain_id}:{run_id}`` instead
        of ``_ckpt:{chain_id}``.  Use this to isolate concurrent or
        repeated runs that share the same chain_id and store.  The caller
        must pass the same run_id to resume a specific run.

    Limitations
    -----------
    Checkpoint resume preserves handoff semantics (agent→agent context
    injection vs tool→agent text-as-task) via a synthetic context
    reconstructed from the saved output text.  However, the reconstructed
    context is a plain text snapshot — it does not carry the live agent's
    ``_last_output`` reference or any ``LazyContext`` composition that was
    active at checkpoint time.
    """
    _ckpt_key = f"_ckpt:{chain_id}:{run_id}" if run_id else f"_ckpt:{chain_id}"

    def _run_chain(task: str) -> Any:
        from lazybridge.lazy_context import LazyContext

        start_step, state = _restore_checkpoint(store, _ckpt_key, task)
        if start_step >= len(parts):
            _logger.warning(
                "Checkpoint step %d exceeds chain length %d for %r; restarting from step 0",
                start_step,
                len(parts),
                _ckpt_key,
            )
            start_step = 0
            state = _ChainState(text=task, typed=None, ctx=None)

        for i, p in enumerate(parts):
            if i < start_step:
                continue

            if hasattr(p, "chat"):  # LazyAgent
                kw: dict = {}
                if native_tools:
                    kw["native_tools"] = native_tools
                if state.ctx is not None:
                    kw["context"] = state.ctx
                    current_task = task
                else:
                    current_task = state.text
                schema = getattr(p, "output_schema", None)
                has_tools = bool(
                    getattr(p, "tools", None) or getattr(p, "native_tools", None) or kw.get("native_tools")
                )
                if schema is not None:
                    result = p.json(current_task, schema, **kw)
                    state = _ChainState(
                        text=result.model_dump_json() if hasattr(result, "model_dump_json") else str(result),
                        typed=result,
                        ctx=LazyContext.from_agent(p),
                    )
                elif has_tools:
                    resp = p.loop(current_task, **kw)
                    state = _ChainState(
                        text=resp.content if hasattr(resp, "content") else str(resp),
                        typed=None,
                        ctx=LazyContext.from_agent(p),
                    )
                else:
                    resp = p.chat(current_task, **kw)
                    state = _ChainState(
                        text=resp.content if hasattr(resp, "content") else str(resp),
                        typed=None,
                        ctx=LazyContext.from_agent(p),
                    )
            elif hasattr(p, "run"):  # LazyTool (nested pipeline)
                # When the previous step produced a typed Pydantic object
                # (e.g. an agent with output_schema), pass its fields as
                # the tool's arguments so agent→function chains work:
                #   Agent(output_schema=FitModelInput) → LazyTool(fit_model)
                # The agent produces FitModelInput; model_dump() becomes
                # {"family": "garch", "target_col": "value", ...} which
                # maps directly to fit_model's signature.
                if state.typed is not None and hasattr(state.typed, "model_dump"):
                    args = state.typed.model_dump()
                else:
                    args = {"task": state.text}
                result = p.run(args)
                state = _ChainState(text=str(result), typed=None, ctx=None)
            else:
                raise TypeError(f"Participant {p!r} must be a LazyAgent (has .chat) or LazyTool (has .run).")

            _save_checkpoint(store, _ckpt_key, i, state, task)

        _clear_checkpoint(store, _ckpt_key)
        return state.typed if state.typed is not None else state.text

    return _run_chain


def build_achain_func(
    parts: list,
    native_tools: list,
    step_timeout: float | None = None,
    *,
    store: Any | None = None,
    chain_id: str = "chain",
    run_id: str | None = None,
) -> Callable[[str], Any]:
    """Return an async closure for sequential execution (mirrors build_chain_func).

    Uses achat()/aloop()/ajson() instead of their sync counterparts so the
    event loop is never blocked.  Used by LazyTool.chain() — same async-under-
    the-hood pattern as build_parallel_func.

    The inner coroutine is constructed inline (not inside a nested def) to
    avoid Python's closure-capture-by-reference hazard in loops.

    Parameters
    ----------
    step_timeout:
        Per-step timeout in seconds.  asyncio.TimeoutError is raised if a
        step exceeds the limit.  None means no timeout.
    store:
        Optional LazyStore for checkpoint persistence (see build_chain_func).
    chain_id:
        Namespace for checkpoint keys in the store.  See build_chain_func
        for concurrency constraints and resume-semantics limitations.
    run_id:
        Optional run identifier for checkpoint isolation (see build_chain_func).
    """
    _ckpt_key = f"_ckpt:{chain_id}:{run_id}" if run_id else f"_ckpt:{chain_id}"

    async def _run_achain(task: str) -> Any:
        from lazybridge.lazy_context import LazyContext

        start_step, state = _restore_checkpoint(store, _ckpt_key, task)
        if start_step >= len(parts):
            _logger.warning(
                "Checkpoint step %d exceeds chain length %d for %r; restarting from step 0",
                start_step,
                len(parts),
                _ckpt_key,
            )
            start_step = 0
            state = _ChainState(text=task, typed=None, ctx=None)

        for i, p in enumerate(parts):
            if i < start_step:
                continue

            if hasattr(p, "achat"):  # LazyAgent
                kw: dict = {}
                if native_tools:
                    kw["native_tools"] = native_tools
                if state.ctx is not None:
                    kw["context"] = state.ctx
                    current_task = task
                else:
                    current_task = state.text

                schema = getattr(p, "output_schema", None)
                has_tools = bool(
                    getattr(p, "tools", None) or getattr(p, "native_tools", None) or kw.get("native_tools")
                )

                # Build the coroutine inline — avoids closure-in-loop capture issues.
                # Coroutines capture their arguments at creation time.
                if schema is not None:
                    coro = p.ajson(current_task, schema, **kw)
                elif has_tools:
                    coro = p.aloop(current_task, **kw)
                else:
                    coro = p.achat(current_task, **kw)

                _skip_timeout = step_timeout is None or getattr(p, "_is_human", False)
                result = await coro if _skip_timeout else await asyncio.wait_for(coro, timeout=step_timeout)

                if schema is not None:
                    state = _ChainState(
                        text=result.model_dump_json() if hasattr(result, "model_dump_json") else str(result),
                        typed=result,
                        ctx=LazyContext.from_agent(p),
                    )
                else:
                    state = _ChainState(
                        text=result.content if hasattr(result, "content") else str(result),
                        typed=None,
                        ctx=LazyContext.from_agent(p),
                    )

            elif hasattr(p, "arun"):  # LazyTool (nested pipeline)
                # When the previous step produced a typed Pydantic object
                # (e.g. an agent with output_schema), pass its fields as
                # the tool's arguments so agent→function chains work:
                #   Agent(output_schema=FitModelInput) → LazyTool(fit_model)
                # The agent produces FitModelInput; model_dump() becomes
                # {"family": "garch", "target_col": "value", ...} which
                # maps directly to fit_model's signature.
                if state.typed is not None and hasattr(state.typed, "model_dump"):
                    args = state.typed.model_dump()
                else:
                    args = {"task": state.text}
                coro = p.arun(args)
                result_str = (
                    await asyncio.wait_for(coro, timeout=step_timeout) if step_timeout is not None else await coro
                )
                state = _ChainState(text=str(result_str), typed=None, ctx=None)

            else:
                raise TypeError(f"Participant {p!r} must be a LazyAgent (has .achat) or LazyTool (has .arun).")

            _save_checkpoint(store, _ckpt_key, i, state, task)

        _clear_checkpoint(store, _ckpt_key)

        return state.typed if state.typed is not None else state.text

    return _run_achain


# ---------------------------------------------------------------------------
# Type discriminators (no circular imports — attribute checks only)
# ---------------------------------------------------------------------------


def _is_agent_instance(p: Any) -> bool:
    """True if p is a LazyAgent instance.
    Discriminator: _last_output is set in LazyAgent.__init__ (line 233).
    Not present on LazyTool — no circular import needed.
    """
    return hasattr(p, "_last_output")


def _is_delegate_tool(p: Any) -> bool:
    """True if p is a LazyTool.from_agent() instance (has a non-None delegate).
    Discriminator: _delegate is a LazyTool dataclass field, not present on LazyAgent.
    """
    return hasattr(p, "_delegate") and getattr(p, "_delegate", None) is not None


def _clone_for_invocation(agent: Any) -> Any:
    """Shallow-copy agent with call-state reset for per-invocation isolation.

    Tracking policy (source-verified lazy_agent.py line 366):
      _track() gates on self._log, NOT self.session.
      Clone gets _log bound to original session EventLog under a new agent_id.
      clone.session = None: clone is not registered in session._agents.
      If original had no session: clone._log = None (tracking OFF).

    Shares (safe — never mutated during execution):
      _executor, system, context, tools, output_schema, native_tools

    Resets (mutable call state):
      id → new UUID
      _last_output → None
      _last_response → None
      session → None
      _log → rebound to original session EventLog under new id, or None
    """
    import copy
    import uuid as _uuid

    clone = copy.copy(agent)
    clone.id = str(_uuid.uuid4())
    clone._last_output = None
    clone._last_response = None
    clone.session = None
    original_session = getattr(agent, "session", None)
    clone._log = original_session.events.agent_log(clone.id, agent.name) if original_session is not None else None
    return clone


def _resolve_participant(p: Any) -> Any:
    """Return invocation-isolated version of p.

    Dispatch:
      LazyAgent (_last_output present)            → _clone_for_invocation(p)
      LazyTool.from_agent() (_delegate not None)  → _clone_delegate_tool_for_invocation(p)
      LazyTool.from_function() or pipeline tool   → p unchanged (stateless)
      Unknown type                                → TypeError

    Note: _clone_delegate_tool_for_invocation lives in lazy_tool.py (friend module)
    to avoid accessing _DelegateConfig from outside its home module.
    """
    if _is_agent_instance(p):
        return _clone_for_invocation(p)
    if _is_delegate_tool(p):
        from lazybridge.lazy_tool import _clone_delegate_tool_for_invocation

        return _clone_delegate_tool_for_invocation(p)
    if hasattr(p, "run") and hasattr(p, "arun"):
        return p  # LazyTool.from_function or pipeline tool — stateless
    raise TypeError(
        f"Participant {p!r} is not a LazyAgent or LazyTool. "
        "Chain/parallel participants must be LazyAgent or LazyTool instances."
    )


def _validate_session_compatibility(participants: tuple, session: Any) -> None:
    """Validation-only: raises ValueError on cross-session conflict. No registration.
    Covers LazyAgent instances AND LazyTool.from_agent() delegate tools.
    """
    if session is None:
        return
    for p in participants:
        if _is_agent_instance(p):
            agent = p
        elif _is_delegate_tool(p):
            agent = p._delegate.agent
        else:
            continue
        agent_session = getattr(agent, "session", None)
        if agent_session is not None and agent_session is not session:
            raise ValueError(
                f"Agent '{getattr(agent, 'name', repr(agent))}' is bound to a different session. "
                "Pass the same session= to both the LazyAgent constructor and factory method, "
                "or omit session= from the factory."
            )
