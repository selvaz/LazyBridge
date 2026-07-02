"""Concurrent fan-out (``run_many`` / ``arun_many``) for :class:`Plan`.

Carved out of ``_plan.py`` in the v1-stabilization refactor.  Behaviour
is unchanged — ``Plan`` inherits this mixin, so both methods keep their
original names and signatures.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from lazybridge.envelope import Envelope


class FanoutMixin:
    """Fan-out helpers shared into :class:`Plan` by inheritance.

    Expects the host class to provide the Engine-protocol ``run``.
    """

    if TYPE_CHECKING:

        async def run(
            self,
            env: Envelope[Any],
            *,
            tools: list[Any],
            output_type: type,
            memory: Any,
            session: Any,
            store: Any | None = None,
            plan_state: Any | None = None,
        ) -> Envelope[Any]: ...

    def run_many(
        self,
        tasks: list[str | Envelope[Any]],
        *,
        concurrency: int | None = None,
        tools: list[Any] | None = None,
        memory: Any = None,
        session: Any = None,
        output_type: type = str,
    ) -> list[Envelope[Any]]:
        """Run this Plan concurrently against ``N`` inputs; sync return.

        Each ``task`` is dispatched as its own ``Plan.run`` invocation
        on a fresh asyncio task; results are returned as a list in
        input order.  Pair with ``Plan(on_concurrent="fork", ...)`` for
        true fan-out workflows where each input claims its own
        per-run keyspace.

        Errors are returned as error envelopes in the corresponding
        slot — the call never raises (matches ``Agent.parallel``
        semantics).

        ``concurrency`` caps the number of in-flight runs via an
        asyncio semaphore.  ``None`` (default) lets every task fire
        immediately.

        Pass ``tools`` when the Plan's steps use string-name targets that
        must be resolved against a live tool map.  Omitting ``tools``
        (or passing ``[]``) works only when every step target is an
        ``Agent`` object rather than a string alias.

        See :meth:`arun_many` for the async variant when the caller is
        already inside an event loop.
        """
        # Re-use the shared sync↔async bridge — it propagates contextvars
        # (OTel spans, request ids, …) into the worker loop so observability
        # flows through fan-outs, and handles nest_asyncio / loop-closed
        # cleanup uniformly.  See ``lazybridge._asyncbridge``.
        from lazybridge._asyncbridge import run_coroutine_blocking

        result: list[Envelope[Any]] = run_coroutine_blocking(
            lambda: self.arun_many(
                tasks,
                concurrency=concurrency,
                tools=tools,
                memory=memory,
                session=session,
                output_type=output_type,
            )
        )
        return result

    async def arun_many(
        self,
        tasks: list[str | Envelope[Any]],
        *,
        concurrency: int | None = None,
        tools: list[Any] | None = None,
        memory: Any = None,
        session: Any = None,
        output_type: type = str,
    ) -> list[Envelope[Any]]:
        """Async counterpart to :meth:`run_many`.

        Use this directly when you're already inside an event loop and
        want to ``await`` the fan-out without the sync-bridge overhead.

        Pass ``tools`` when the Plan's steps use string-name targets that
        must be resolved against a live tool map.  Omitting ``tools``
        (or passing ``[]``) works only when every step target is an
        ``Agent`` object rather than a string alias.
        """
        sem = asyncio.Semaphore(concurrency) if concurrency else None
        resolved_tools: list[Any] = tools or []

        async def _one(task: str | Envelope[Any]) -> Envelope[Any]:
            # ``Envelope.from_task`` populates BOTH ``task`` and
            # ``payload`` so the first step's ``from_prev`` resolves to
            # the user's input rather than an empty string.
            env = task if isinstance(task, Envelope) else Envelope.from_task(str(task))

            async def _go() -> Envelope[Any]:
                return await self.run(
                    env,
                    tools=resolved_tools,
                    output_type=output_type,
                    memory=memory,
                    session=session,
                )

            if sem is None:
                return await _go()
            async with sem:
                return await _go()

        raw = await asyncio.gather(
            *[_one(t) for t in tasks],
            return_exceptions=True,
        )
        # Wrap raised exceptions as error envelopes so the contract is
        # "list of envelopes in input order".  Plan.run normally
        # returns an error envelope itself, so this branch only fires
        # for genuine framework bugs / cancellations.
        return [
            r
            if isinstance(r, Envelope)
            else Envelope.error_envelope(r if isinstance(r, BaseException) else RuntimeError(str(r)))
            for r in raw
        ]
