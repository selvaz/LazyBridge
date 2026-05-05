"""Cross-cutting correctness regressions.

* Plan parallel-band atomicity: failure in one parallel branch leaves
  NO partial writes from sibling branches.
* Plan checkpoint claim race: two concurrent fresh runs on the same
  key fail fast at claim time (before either step executes), not
  after the loser wasted side-effects.
* Retryable exception classifier: exceptions whose class name matches
  a known transient family are retried even when the message is
  empty / non-English / SDK-mangled.
* Sync ``__call__`` contextvars propagation: a contextvar set in the
  outer event loop is visible inside the agent's worker loop.
* ``fallback=`` / ``verify=`` agents inherit the outer session and
  appear on the session graph with labelled edges.
"""

from __future__ import annotations

import asyncio
import contextvars

import pytest

from lazybridge import Agent
from lazybridge.core.executor import _is_retryable
from lazybridge.engines.plan import ConcurrentPlanRunError, Plan, Step
from lazybridge.envelope import Envelope, ErrorInfo
from lazybridge.session import Session
from lazybridge.store import Store
from lazybridge.testing import MockAgent

# ---------------------------------------------------------------------------
# E1 — Plan parallel-band atomicity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e1_parallel_band_failure_does_not_commit_sibling_writes() -> None:
    """When one parallel branch errors, branches earlier in the
    declared order MUST NOT have committed their ``writes=`` to the
    Store — the engine scans every branch for failure first and
    returns without applying any writes if any branch errored.
    """
    store = Store()

    # ``a`` succeeds quickly and would write to ``kv["a_out"]``.
    # ``b`` always fails.  Both run in the same parallel band.
    a = MockAgent("a-result", name="a", delay_ms=5)
    b = MockAgent(RuntimeError("b boom"), name="b")

    plan = Plan(
        Step(target=a, name="a", writes="a_out", parallel=True),
        Step(target=b, name="b", writes="b_out", parallel=True),
        store=store,
        checkpoint_key="atomicity",
    )
    env = await Agent(engine=plan, name="p").run("task")
    assert not env.ok, "expected failure envelope"

    # No sibling writes survive a band failure.
    assert store.read("a_out") is None
    assert store.read("b_out") is None
    # Checkpoint points at the BAND START (not the failing step) so a
    # resume re-runs the whole band cleanly rather than skipping earlier
    # siblings whose writes were never committed.
    cp = store.read("atomicity")
    assert cp is not None
    assert cp["status"] == "failed"
    assert cp["next_step"] == "a"   # band-start, not the failing step
    assert "a_out" not in cp["kv"]


@pytest.mark.asyncio
async def test_e1_parallel_band_success_still_commits_writes() -> None:
    """Sanity: when every branch succeeds, writes apply in declared
    order.
    """
    store = Store()
    a = MockAgent("a-out", name="a")
    b = MockAgent("b-out", name="b")

    plan = Plan(
        Step(target=a, name="a", writes="a_kv", parallel=True),
        Step(target=b, name="b", writes="b_kv", parallel=True),
        store=store,
        checkpoint_key="happy",
    )
    env = await Agent(engine=plan, name="p").run("task")
    assert env.ok, env.error
    assert store.read("a_kv") == "a-out"
    assert store.read("b_kv") == "b-out"


# ---------------------------------------------------------------------------
# E2 — Checkpoint claim race
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2_two_fresh_runs_collide_at_claim_not_at_first_save() -> None:
    """Two concurrent fresh Plan runs on the same checkpoint key
    collide at the up-front claim — the loser raises
    ``ConcurrentPlanRunError`` without first wasting side-effects on
    step 0.
    """
    store = Store()

    # Slow first step gives the second run a fair chance to hit
    # ``_claim_checkpoint`` while the first is mid-step.
    a = MockAgent(lambda env: f"a({env.task})", name="a", delay_ms=40)
    b = MockAgent(lambda env: f"b({env.text()})", name="b")

    def _make() -> Agent:
        return Agent(
            engine=Plan(
                Step(target=a, name="a"),
                Step(target=b, name="b"),
                store=store,
                checkpoint_key="shared-key",
            ),
            name="p",
        )

    first = _make()
    raised: list[BaseException] = []

    async def _second() -> None:
        # Wait long enough for ``first`` to claim, but short enough
        # that ``first`` is still inside step ``a``'s 40ms delay.
        await asyncio.sleep(0.005)
        try:
            await _make().run("second")
        except ConcurrentPlanRunError as exc:
            raised.append(exc)

    await asyncio.gather(first.run("first"), _second())

    assert raised, "second concurrent fresh run did not raise ConcurrentPlanRunError"
    # Step a only fired once — for the first run.  The second run
    # never made it past the up-front claim, so its planned call
    # to ``a`` was suppressed entirely (no double side-effects).
    assert a.call_count == 1, (
        f"second run should have failed at claim before executing step a, but a.call_count={a.call_count}"
    )


@pytest.mark.asyncio
async def test_e2_done_key_can_be_reclaimed_by_fresh_run() -> None:
    """A completed checkpoint (``status=done``) doesn't permanently
    poison the key — a fresh run on the same key still works.
    """
    store = Store()
    a = MockAgent("a", name="a")
    b = MockAgent("b", name="b")

    def _make() -> Agent:
        return Agent(
            engine=Plan(
                Step(target=a, name="a"),
                Step(target=b, name="b"),
                store=store,
                checkpoint_key="reclaim",
            ),
            name="p",
        )

    env1 = await _make().run("one")
    assert env1.ok
    cp = store.read("reclaim")
    assert cp["status"] == "done"

    env2 = await _make().run("two")
    assert env2.ok


# ---------------------------------------------------------------------------
# E4 — Retryable exception classification
# ---------------------------------------------------------------------------


def test_e4_retryable_by_class_name_without_message() -> None:
    """Class-name matching catches transients even when the message
    is empty / non-English / SDK-mangled.
    """

    class RateLimitError(Exception):  # mimics openai/anthropic naming
        pass

    class APITimeoutError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    assert _is_retryable(RateLimitError(""))
    assert _is_retryable(APITimeoutError("límite de tasa"))  # non-English
    assert _is_retryable(APIConnectionError("中文"))  # CJK message


def test_e4_retryable_walks_mro_for_subclasses() -> None:
    """Subclasses of well-known transient families inherit retryability
    via MRO walk — no need to enumerate every concrete subclass."""

    class MySpecialConnectionReset(ConnectionResetError):
        pass

    assert _is_retryable(MySpecialConnectionReset())


def test_e4_non_transient_still_not_retryable() -> None:
    """ValueErrors / TypeErrors / etc. with no transient class name
    or status code stay non-retryable."""

    class ValidationError(Exception):
        pass

    assert not _is_retryable(ValidationError("bad input"))
    assert not _is_retryable(ValueError("invalid"))
    assert not _is_retryable(TypeError("nope"))


def test_e4_status_code_path_still_wins() -> None:
    """Pre-existing status_code path is unchanged — 429 / 5xx remain
    retryable regardless of class name."""

    class CustomError(Exception):
        def __init__(self, code: int) -> None:
            self.status_code = code

    assert _is_retryable(CustomError(429))
    assert _is_retryable(CustomError(503))
    assert not _is_retryable(CustomError(400))


# ---------------------------------------------------------------------------
# A1 — Sync __call__ contextvars propagation
# ---------------------------------------------------------------------------


_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id",
    default=None,
)


class _ProbeEngine:
    """Minimal Engine that captures ``_request_id.get()`` during run().

    Used by the A1 test to assert that the caller's contextvars
    context survives the sync-call → worker-thread → asyncio.run hop.
    """

    def __init__(self, sink: list[str | None]) -> None:
        self.sink = sink
        self.model = "probe"

    async def run(
        self,
        env: Envelope,
        *,
        tools: list[object],
        output_type: type,
        memory: object,
        session: object,
    ) -> Envelope:
        self.sink.append(_request_id.get())
        return env


@pytest.mark.asyncio
async def test_a1_sync_call_propagates_contextvars_into_worker_loop() -> None:
    """A contextvar set in the outer event loop must be visible
    inside ``Agent.__call__``'s worker-thread loop, so observability
    state (OTel spans, request IDs) survives when callers use the
    sync façade from inside an async framework (FastAPI, Starlette,
    Jupyter).
    """
    seen: list[str | None] = []
    outer = Agent(engine=_ProbeEngine(seen), name="probe")

    _request_id.set("trace-42")
    # Calling __call__ from inside an async test forces the
    # worker-thread path (``get_running_loop`` succeeds, so __call__
    # cannot use the simple ``asyncio.run`` branch).
    outer("hello")

    assert seen == ["trace-42"], f"contextvar lost across worker-loop boundary: saw {seen!r}"


# ---------------------------------------------------------------------------
# A2 — Fallback / verify session propagation
# ---------------------------------------------------------------------------


def test_a2_fallback_agent_inherits_outer_session() -> None:
    """``fallback=`` Agents now inherit the outer session and appear
    on the session graph with a 'fallback' edge label."""
    sess = Session()
    primary = MockAgent("p", name="primary")
    fb = MockAgent("f", name="fb")

    Agent(engine=primary, fallback=fb, session=sess, name="outer")

    assert fb.session is sess
    edge_labels = {(e.from_id, e.to_id, e.label) for e in sess.graph.edges()}
    assert ("outer", "fb", "fallback") in edge_labels


def test_a2_verify_agent_inherits_outer_session() -> None:
    """``verify=`` Agents now inherit the outer session and appear
    on the session graph with a 'verify' edge label."""
    sess = Session()
    primary = MockAgent("p", name="primary")
    judge = MockAgent("approved", name="judge")

    Agent(engine=primary, verify=judge, session=sess, name="outer")

    assert judge.session is sess
    edge_labels = {(e.from_id, e.to_id, e.label) for e in sess.graph.edges()}
    assert ("outer", "judge", "verify") in edge_labels


def test_a2_fallback_with_existing_session_is_not_overwritten() -> None:
    """If the fallback agent already carries a session, the outer
    session is not stomped — preserves the contract that nested-tool
    Agents are also opt-in: explicit session wins."""
    outer_sess = Session()
    inner_sess = Session()

    primary = MockAgent("p", name="primary")
    fb = MockAgent("f", name="fb")
    fb.session = inner_sess  # pre-existing

    Agent(engine=primary, fallback=fb, session=outer_sess, name="outer")
    assert fb.session is inner_sess  # untouched


def test_a2_verify_callable_is_not_treated_as_agent() -> None:
    """``verify=`` accepts plain callables too — those have no
    ``_is_lazy_agent`` marker and should not trigger the registration
    path (which would crash on missing attributes).
    """
    sess = Session()
    primary = MockAgent("p", name="primary")

    def judge(_text: str) -> ErrorInfo | None:
        return None

    # Should not raise — callable verify skips the agent-registration branch.
    Agent(engine=primary, verify=judge, session=sess, name="outer")
