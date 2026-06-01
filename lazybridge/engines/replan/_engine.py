"""ReplanEngine — guardian of the dynamic replan loop.

The counterpart to :class:`lazybridge.engines.plan.Plan` for pipelines
whose shape is decided at runtime by a planner agent.

``Plan`` is the guardian for fixed-step pipelines — it compiles the DAG at
construction, checkpoints after every step, and resumes from the last
checkpoint on crash or pause.  ``ReplanEngine`` applies the same guarantees
to the *replan loop*: the planner is called every round, its output drives
which tools run, and a Store-backed checkpoint is written after every round so
a restart picks up from the correct round without re-executing completed work.

Architecture
------------
``ReplanEngine`` follows LazyBridge's "everything is a tool" principle:

- The **planner** is a Tool in the tool_map (no constructor injection).
  Pass ``Agent(output=PlanRound, name="planner")`` in the parent Agent's
  ``tools=[]``; ReplanEngine finds it by ``planner_name``.
- **Worker tools** (agents, functions, pools) are also in the tool_map.
  ReplanEngine dispatches tasks via ``tool.run(**task.kwargs)`` — no
  special-casing for pools or agents.
- The resulting ``Agent(engine=ReplanEngine(...))`` is itself usable as a
  tool by a parent supervisor (no extra plumbing needed).

Checkpoint / resume
-------------------
Pass ``store=`` and ``checkpoint_key=`` to persist round state after every
round.  Pass ``resume=True`` to continue from the last checkpoint on the
next call.  Semantics match :class:`Plan`: the first call claims the key via
CAS; a second concurrent call raises :class:`ConcurrentPlanRunError`.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from lazybridge.engines.plan._serialisation import _first_arg_kwargs
from lazybridge.engines.plan._types import ConcurrentPlanRunError
from lazybridge.engines.replan._types import PlanRound, Task
from lazybridge.envelope import Envelope
from lazybridge.session import EventType
from lazybridge.signals import ConcludeSignal

if TYPE_CHECKING:
    from lazybridge.memory import Memory
    from lazybridge.session import Session
    from lazybridge.store import Store
    from lazybridge.tools import Tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_history(history: list[dict[str, Any]]) -> str:
    """Render the accumulated round history as a compact string for the planner."""
    if not history:
        return "(no prior rounds)"
    lines: list[str] = []
    for i, entry in enumerate(history, 1):
        t = entry.get("task", {})
        kw = t.get("kwargs", {})
        kw_str = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        out = str(entry.get("output", ""))
        lines.append(f"{i}. [{t.get('tool', '?')}]({kw_str}) → {out[:300]}")
    return "\n".join(lines)


def _reinterleave(
    tasks: list[Task],
    par_results: list[Any],
    seq_results: list[Any],
) -> list[Any]:
    """Re-interleave parallel and sequential results back to original task order."""
    out: list[Any] = []
    p_it = iter(par_results)
    s_it = iter(seq_results)
    for t in tasks:
        out.append(next(p_it) if t.parallel else next(s_it))
    return out


# ---------------------------------------------------------------------------
# ReplanEngine
# ---------------------------------------------------------------------------


class ReplanEngine:
    """Engine that guards the dynamic replan loop with checkpoint/resume.

    The planner and all worker tools are resolved from the tool_map at run
    time — nothing is injected at construction except configuration::

        guardian = Agent(
            engine=ReplanEngine(
                store=Store(db="project.sqlite"),
                checkpoint_key="my-project",
                resume=True,
            ),
            tools=[planner, analyst, coder, pool.as_tool("route")],
        )

    The planner tool must have ``output=PlanRound``.  ReplanEngine builds the
    planner's input dynamically (tool schemas + history) so the planner does
    not need a static system prompt that lists worker names.

    See :class:`lazybridge.engines.replan.PlanRound` and
    :class:`lazybridge.engines.replan.Task` for the planner output schema.
    """

    def __init__(
        self,
        *,
        planner_name: str = "planner",
        store: Store | None = None,
        checkpoint_key: str | None = None,
        resume: bool = False,
        max_rounds: int = 20,
    ) -> None:
        self.planner_name = planner_name
        self.store = store
        self.checkpoint_key = checkpoint_key
        self.resume = resume
        self.max_rounds = max_rounds

    # ------------------------------------------------------------------
    # Engine protocol
    # ------------------------------------------------------------------

    async def run(
        self,
        env: Envelope[Any],
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
        store: Any | None = None,
        plan_state: Any | None = None,
    ) -> Envelope[Any]:
        run_id = str(uuid.uuid4())
        agent_name = getattr(self, "_agent_name", "replan")
        if session:
            session.emit(
                EventType.AGENT_START,
                {"agent_name": agent_name, "task": env.task},
                run_id=run_id,
            )

        result_env: Envelope[Any] | None = None
        error_msg: str | None = None
        try:
            result_env = await self._run_impl(
                env,
                tools=tools,
                session=session,
                run_id=run_id,
                agent_name=agent_name,
            )
            return result_env
        except BaseException as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if session:
                payload: dict[str, Any] = {"agent_name": agent_name}
                if error_msg is not None:
                    payload["error"] = error_msg
                elif result_env is not None and result_env.error is not None:
                    payload["error"] = result_env.error.message
                elif result_env is not None:
                    payload["payload"] = (result_env.text() or "")[:500]
                session.emit(EventType.AGENT_FINISH, payload, run_id=run_id)

    async def stream(
        self,
        env: Envelope[Any],
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
    ) -> AsyncIterator[str]:
        result = await self.run(env, tools=tools, output_type=output_type, memory=memory, session=session)
        yield result.text()

    # ------------------------------------------------------------------
    # Checkpoint helpers (same CAS pattern as Plan)
    # ------------------------------------------------------------------

    def _checkpoint_store(self) -> Store | None:
        return self.store if self.checkpoint_key else None

    def _effective_key(self, run_uid: str) -> str | None:
        return self.checkpoint_key  # single-writer semantics (no "fork" mode yet)

    def _claim_checkpoint(self, effective_key: str | None, run_uid: str) -> dict[str, Any] | None:
        store = self._checkpoint_store()
        if store is None or effective_key is None:
            return None
        existing = store.read(effective_key)
        claimed: dict[str, Any] = {
            "round": 0,
            "history": [],
            "status": "claimed",
            "run_uid": run_uid,
            "final_answer": None,
        }
        if not isinstance(existing, dict):
            if not store.compare_and_swap(effective_key, None, claimed):
                raise ConcurrentPlanRunError(
                    f"Lost race claiming {effective_key!r} — another run wrote "
                    f"the key concurrently.  Use a unique checkpoint_key per run."
                )
            return claimed
        status = existing.get("status")
        if status == "done":
            if self.resume:
                return existing  # short-circuit in _run_impl
            if not store.compare_and_swap(effective_key, existing, claimed):
                raise ConcurrentPlanRunError(f"Lost race re-claiming completed key {effective_key!r}.  Retry.")
            return claimed
        if status is None:
            raise ConcurrentPlanRunError(
                f"Checkpoint key {effective_key!r} holds a value with no 'status' field; "
                f"refusing to overwrite.  Use a different checkpoint_key."
            )
        if not self.resume:
            raise ConcurrentPlanRunError(
                f"Checkpoint {effective_key!r} is already held by "
                f"run_uid={existing.get('run_uid')!r} (status={status!r}).  "
                f"Pass resume=True to continue, or use a unique checkpoint_key."
            )
        adopted = {**existing, "run_uid": run_uid}
        if not store.compare_and_swap(effective_key, existing, adopted):
            raise ConcurrentPlanRunError(f"Lost race adopting {effective_key!r} for resume.  Retry.")
        return adopted

    def _load_checkpoint(self, effective_key: str | None) -> dict[str, Any] | None:
        store = self._checkpoint_store()
        if store is None or effective_key is None or not self.resume:
            return None
        saved = store.read(effective_key)
        return saved if isinstance(saved, dict) else None

    def _save_checkpoint(
        self,
        last_snap: dict[str, Any] | None,
        effective_key: str | None,
        run_uid: str,
        *,
        round: int,
        history: list[dict[str, Any]],
        status: str,
        final_answer: str | None = None,
    ) -> dict[str, Any] | None:
        store = self._checkpoint_store()
        if store is None or effective_key is None:
            return None
        new_snap: dict[str, Any] = {
            "round": round,
            "history": list(history),
            "status": status,
            "run_uid": run_uid,
            "final_answer": final_answer,
        }
        if not store.compare_and_swap(effective_key, last_snap, new_snap):
            raise ConcurrentPlanRunError(
                f"Checkpoint {effective_key!r} was modified by another writer "
                f"mid-run (our run_uid={run_uid!r}).  Use a unique checkpoint_key "
                f"per concurrent run."
            )
        return new_snap

    # ------------------------------------------------------------------
    # Planner input builder
    # ------------------------------------------------------------------

    def _build_planner_input(
        self,
        task: str,
        history: list[dict[str, Any]],
        tool_map: dict[str, Tool],
    ) -> str:
        """Build the planner's input: tool schemas + task + history.

        The planner receives the available worker tool schemas dynamically so
        it doesn't need a static system prompt listing tool names.
        """
        lines: list[str] = []
        for name, t in tool_map.items():
            if name == self.planner_name:
                continue
            props = t.definition().parameters.get("properties", {})
            sig = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
            desc = t.description or ""
            entry = f"- {name}({sig})"
            if desc:
                entry += f": {desc}"
            lines.append(entry)
        roster = "\n".join(lines) if lines else "(no tools available)"
        hist = _format_history(history)
        return f"Available tools:\n{roster}\n\nTask: {task}\n\nHistory:\n{hist}"

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    async def _run_impl(
        self,
        env: Envelope[Any],
        *,
        tools: list[Tool],
        session: Session | None,
        run_id: str,
        agent_name: str,
    ) -> Envelope[Any]:
        tool_map: dict[str, Tool] = {t.name: t for t in tools}
        run_uid = uuid.uuid4().hex
        effective_key = self._effective_key(run_uid)

        last_snap = self._claim_checkpoint(effective_key, run_uid)
        checkpoint = self._load_checkpoint(effective_key)

        # Resume on already-completed run → return cached answer immediately.
        if isinstance(checkpoint, dict) and checkpoint.get("status") == "done":
            return Envelope(task=env.task, payload=checkpoint.get("final_answer", ""))

        history: list[dict[str, Any]] = list((checkpoint or {}).get("history") or [])
        round_start: int = int((checkpoint or {}).get("round") or 0)

        planner_tool = tool_map.get(self.planner_name)
        if planner_tool is None:
            return Envelope.error_envelope(
                RuntimeError(
                    f"ReplanEngine: planner tool {self.planner_name!r} not found "
                    f"in tool_map {list(tool_map)!r}.  "
                    f"Pass Agent(output=PlanRound, name={self.planner_name!r}) "
                    f"in the parent Agent's tools=[]."
                )
            )

        for round_num in range(round_start, self.max_rounds):
            # ── Ask the planner ───────────────────────────────────────────
            planner_input = self._build_planner_input(env.task or "", history, tool_map)
            try:
                planner_raw = await planner_tool.run(
                    **{"task": planner_input}
                    if "task" in planner_tool.definition().parameters.get("properties", {})
                    else _first_arg_kwargs(planner_tool, planner_input)
                )
            except ConcludeSignal:
                raise
            except Exception as exc:
                self._save_checkpoint(
                    last_snap,
                    effective_key,
                    run_uid,
                    round=round_num,
                    history=history,
                    status="failed",
                )
                return Envelope.error_envelope(exc)

            plan: PlanRound | None = planner_raw.payload if isinstance(planner_raw, Envelope) else planner_raw

            if plan is None:
                err_info = planner_raw.error if isinstance(planner_raw, Envelope) else None
                self._save_checkpoint(
                    last_snap,
                    effective_key,
                    run_uid,
                    round=round_num,
                    history=history,
                    status="failed",
                )
                if err_info is not None:
                    return Envelope(
                        task=env.task,
                        error=err_info,
                    )
                return Envelope.error_envelope(
                    RuntimeError(
                        f"ReplanEngine: planner returned no PlanRound payload "
                        f"at round {round_num}.  "
                        f"Ensure the planner Agent is built with output=PlanRound."
                    )
                )

            if plan.done:
                # P2: reject missing final_answer before writing a permanent "done" checkpoint.
                # A None answer cached as "done" would silently short-circuit every future
                # resume=True call with an empty payload.
                if plan.final_answer is None:
                    self._save_checkpoint(
                        last_snap,
                        effective_key,
                        run_uid,
                        round=round_num,
                        history=history,
                        status="failed",
                    )
                    return Envelope.error_envelope(
                        RuntimeError(
                            "ReplanEngine: planner set done=True but omitted final_answer.  "
                            "Set final_answer to a non-None string when done=True."
                        )
                    )
                self._save_checkpoint(
                    last_snap,
                    effective_key,
                    run_uid,
                    round=round_num,
                    history=history,
                    status="done",
                    final_answer=plan.final_answer,
                )
                return Envelope(task=env.task, payload=plan.final_answer)

            if not plan.tasks:
                self._save_checkpoint(
                    last_snap,
                    effective_key,
                    run_uid,
                    round=round_num,
                    history=history,
                    status="failed",
                )
                return Envelope.error_envelope(
                    RuntimeError(
                        f"ReplanEngine: round {round_num}: planner emitted no tasks "
                        f"and done=False.  Possible infinite loop — check the planner."
                    )
                )

            # ── Dispatch round ────────────────────────────────────────────
            parallel_tasks = [t for t in plan.tasks if t.parallel]
            seq_tasks = [t for t in plan.tasks if not t.parallel]

            # Parallel band — same gather/error-scan pattern as Plan.
            par_results: list[Any] = []
            if parallel_tasks:
                par_raw = await asyncio.gather(
                    *[self._dispatch(t, tool_map, session, run_id, agent_name) for t in parallel_tasks],
                    return_exceptions=True,
                )
                for r in par_raw:
                    if isinstance(r, ConcludeSignal):
                        raise r
                for r in par_raw:
                    if isinstance(r, BaseException):
                        self._save_checkpoint(
                            last_snap,
                            effective_key,
                            run_uid,
                            round=round_num,
                            history=history,
                            status="failed",
                        )
                        return Envelope.error_envelope(r)
                par_results = list(par_raw)

            # Sequential tasks.
            seq_results: list[Any] = []
            for t in seq_tasks:
                try:
                    seq_results.append(await self._dispatch(t, tool_map, session, run_id, agent_name))
                except ConcludeSignal:
                    raise
                except Exception as exc:
                    self._save_checkpoint(
                        last_snap,
                        effective_key,
                        run_uid,
                        round=round_num,
                        history=history,
                        status="failed",
                    )
                    return Envelope.error_envelope(exc)

            outputs = _reinterleave(plan.tasks, par_results, seq_results)
            history = list(history) + [{"task": t.model_dump(), "output": o} for t, o in zip(plan.tasks, outputs)]

            last_snap = self._save_checkpoint(
                last_snap,
                effective_key,
                run_uid,
                round=round_num + 1,
                history=history,
                status="running",
            )

        # max_rounds exceeded.
        self._save_checkpoint(
            last_snap,
            effective_key,
            run_uid,
            round=self.max_rounds,
            history=history,
            status="failed",
        )
        return Envelope.error_envelope(
            RuntimeError(
                f"ReplanEngine: max_rounds={self.max_rounds} exceeded without "
                f"the planner setting done=True.  Increase max_rounds or check "
                f"the planner's termination condition."
            )
        )

    async def _dispatch(
        self,
        task: Task,
        tool_map: dict[str, Tool],
        session: Session | None,
        run_id: str,
        agent_name: str,
    ) -> str:
        """Call one tool and return its text output."""
        tool = tool_map.get(task.tool)
        if tool is None:
            raise RuntimeError(
                f"ReplanEngine: unknown tool {task.tool!r} in round task.  "
                f"Available tools: {list(tool_map)!r}.  "
                f"The planner must only reference tool names present in tool_map."
            )
        if session:
            session.emit(
                EventType.TOOL_CALL,
                {
                    "step": task.tool,
                    "task": str(task.kwargs)[:200],
                    "agent_name": agent_name,
                },
                run_id=run_id,
            )
        raw = await tool.run(**task.kwargs)
        # P1: propagate tool-level errors rather than silently converting them to empty
        # strings.  An error Envelope's .text() returns "" which looks like a successful
        # but empty result to the planner — it would replan from phantom empty output.
        if isinstance(raw, Envelope) and raw.error is not None:
            raise RuntimeError(
                f"ReplanEngine: tool {task.tool!r} returned an error: {raw.error.message}"
            )
        result = raw if isinstance(raw, str) else (raw.text() if isinstance(raw, Envelope) else str(raw))
        if session:
            session.emit(
                EventType.TOOL_RESULT,
                {
                    "step": task.tool,
                    "result": result[:200],
                    "agent_name": agent_name,
                },
                run_id=run_id,
            )
        return result
