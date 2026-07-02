"""Sentinel resolution + parallel-band aggregation for :class:`Plan`.

Carved out of ``_plan.py`` in the v1-stabilization refactor.  Pure
functions of ``(prev, start, history, kv, tool_map)`` — no checkpoint or
scheduling state.  Behaviour is unchanged — ``Plan`` inherits this
mixin, so every method keeps its original name and signature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lazybridge.engines.plan._types import PlanRuntimeError, Step, StepResult
from lazybridge.envelope import Envelope
from lazybridge.sentinels import (
    _AGENT_OUTPUT_KEY_PREFIX,
    Sentinel,
    _FromAgent,
    _FromMemory,
    _FromParallel,
    _FromParallelAll,
    _FromPrev,
    _FromStart,
    _FromStep,
)

if TYPE_CHECKING:
    pass


class ResolveMixin:
    """Sentinel/band resolution shared into :class:`Plan` by inheritance.

    Expects the host class to provide ``self.steps``.
    """

    steps: list[Step]

    def _resolve_sentinel(
        self,
        sentinel: Sentinel | str,
        prev: Envelope[Any],
        start: Envelope[Any],
        history: list[StepResult],
        kv: dict[str, Any],
        tool_map: dict[str, Any],
    ) -> Envelope[Any]:
        # ``from_prev`` means "the previous step's *output* becomes the next
        # step's task" — i.e. real chain semantics.  Without this promotion
        # the default behaviour was for every step to receive the original
        # user task, so ``Plan(Step(a), Step(b))`` was NOT actually a chain.
        # We carry metadata AND ``.error`` through so downstream code can
        # short-circuit rather than silently feed an errored payload into
        # the next step.
        if isinstance(sentinel, _FromPrev):
            # Multimodal: preserve attachments — for step 0 ``prev`` is
            # the user-supplied input envelope and the user expects
            # images / audio to reach the first agent.  For step N>0
            # ``prev`` is an upstream LLM result which doesn't populate
            # attachments, so this defaults to ``None`` automatically.
            return Envelope(
                task=prev.text(),
                context=prev.context,
                images=prev.images,
                audio=prev.audio,
                payload=prev.payload,
                metadata=prev.metadata,
                error=prev.error,
            )
        if isinstance(sentinel, _FromStart):
            return start
        # from_step / from_parallel also forward the referenced step's
        # output as the next step's task for consistency with from_prev.
        if isinstance(sentinel, _FromStep):
            for r in reversed(history):
                if r.step_name == sentinel.name:
                    e = r.envelope
                    return Envelope(
                        task=e.text(),
                        context=e.context,
                        images=e.images,
                        audio=e.audio,
                        payload=e.payload,
                        metadata=e.metadata,
                        error=e.error,
                    )
            # Compile-time validation already rejects ``from_step`` whose
            # target isn't a Step name; getting here means the named step
            # exists in the Plan but didn't run (routing skipped it, or
            # the run is mid-band on a not-yet-completed parallel sibling).
            # Either way, falling back to ``start`` would silently mask
            # the misconfigured wiring — raise instead.
            known = sorted({r.step_name for r in history})
            raise PlanRuntimeError(
                f"from_step({sentinel.name!r}) found no matching step in this run's history.\n"
                f"  History so far: {known}.\n"
                f"  Likely cause: the named step was skipped by routing, or you're inside a\n"
                f"  parallel band trying to read from a sibling that hasn't completed.\n"
                f"  Fix: reorder the Plan so the referenced step runs before this one, or use\n"
                f"  ``from_parallel_all`` to aggregate parallel siblings after the band."
            )
        if isinstance(sentinel, _FromParallel):
            for r in reversed(history):
                if r.step_name == sentinel.name:
                    e = r.envelope
                    return Envelope(
                        task=e.text(),
                        context=e.context,
                        images=e.images,
                        audio=e.audio,
                        payload=e.payload,
                        metadata=e.metadata,
                        error=e.error,
                    )
            known = sorted({r.step_name for r in history})
            raise PlanRuntimeError(
                f"from_parallel({sentinel.name!r}) found no matching step in this run's history.\n"
                f"  History so far: {known}.\n"
                f"  Fix: reorder the Plan so the referenced parallel step runs before this one."
            )
        if isinstance(sentinel, _FromParallelAll):
            return self._aggregate_parallel_band(sentinel.name, history, fallback=start)
        if isinstance(sentinel, _FromMemory):
            # Resolved at execution time — reads the live memory of the agent
            # registered under sentinel.name in the tool map.  Empty if the
            # agent has no memory or hasn't run yet (silent no-op, no error).
            tool = tool_map.get(sentinel.name)
            memory = getattr(tool, "agent_memory", None) if tool else None
            if memory is not None:
                mem_text = memory.text()
                if mem_text:
                    return Envelope(task=mem_text, context=mem_text, payload=mem_text)
            return Envelope(task="", context=None, payload="")
        if isinstance(sentinel, _FromAgent):
            # Reads the last output of the named agent from the shared Store.
            # Written by Agent._run_body after a successful run under key
            # "__agent_output__:{name}".  Silent no-op if not yet written.
            tool = tool_map.get(sentinel.name)
            agent_store = getattr(tool, "agent_store", None) if tool else None
            if agent_store is not None:
                value = agent_store.read(_AGENT_OUTPUT_KEY_PREFIX + sentinel.name)
                if value is not None:
                    text = str(value)
                    return Envelope(task=text, context=text, payload=text)
            return Envelope(task="", context=None, payload="")
        if isinstance(sentinel, str):
            return Envelope(task=sentinel, payload=sentinel)
        return prev

    def _aggregate_parallel_band(
        self,
        start_name: str,
        history: list[StepResult],
        *,
        fallback: Envelope[Any],
    ) -> Envelope[Any]:
        """Build a single Envelope from every consecutive parallel sibling
        starting at ``start_name`` (in declared order).

        Returns an envelope where ``task`` and ``payload`` are both a
        labelled-text join — ``"[branch_a]\\n<text>\\n\\n[branch_b]\\n<text>..."``
        — so the next step's agent reads all branches via ``env.text()``
        without any change to its tool. The first non-None branch error
        is propagated so downstream can short-circuit.

        Per-branch cost is already accumulated by the engine via
        ``_aggregate_nested_metadata`` from ``history``, so this function
        does not re-sum tokens (would double-count).

        If the named step isn't found in the plan, returns ``fallback``.
        Compile-time validation rejects non-parallel start names, so the
        runtime degenerate case (band of size 1) shouldn't fire — but the
        function tolerates it to keep the engine non-crashy.
        """
        # Find the contiguous parallel band starting at start_name.
        names = [s.name or "" for s in self.steps]
        try:
            start_idx = names.index(start_name)
        except ValueError:
            return fallback

        # Walk forward while consecutive steps remain parallel.
        band_names: list[str] = []
        for i in range(start_idx, len(self.steps)):
            s = self.steps[i]
            if i > start_idx and not s.parallel:
                break
            band_names.append(s.name or "")
            if not s.parallel:
                # Degenerate single-step case (compiler should have rejected).
                break

        # For each band member, pick the most recent matching history entry.
        branch_envs: list[tuple[str, Envelope[Any]]] = []
        for n in band_names:
            for r in reversed(history):
                if r.step_name == n:
                    branch_envs.append((n, r.envelope))
                    break

        if not branch_envs:
            return fallback

        # Labelled-text join — both ``task`` (next step's prompt) and
        # ``payload`` (so ``Envelope.text()`` returns it) carry the join.
        sections = [f"[{n}]\n{e.text() if not e.error else f'(error) {e.error.message}'}" for n, e in branch_envs]
        joined = "\n\n".join(sections)

        # Short-circuit: first error wins so downstream can detect failure.
        first_error = next((e.error for _, e in branch_envs if e.error), None)

        return Envelope(
            task=joined,
            context=None,
            payload=joined,
            error=first_error,
        )
