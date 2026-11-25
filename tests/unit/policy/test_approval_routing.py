from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from contextvars import Context, copy_context
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lazybridge import Agent, AgentPool, Envelope, Session, Store, Tool
from lazybridge.ext.approval import (
    ApprovalQueue,
    ApprovalTarget,
    ApprovalTicket,
    ApproverContext,
    EscalationSummary,
    ExecutorContext,
    ServiceAgent,
    TypedApprovalRequest,
    route_ticket,
)
from lazybridge.policy import (
    ApprovalTier,
    CardMatcher,
    PolicyBudgetExceeded,
    PolicyContext,
    PolicyError,
    PolicyWidening,
    TaskGrant,
    TreeBudget,
    child_within_parent,
    compile_claude,
    compile_codex,
    current_approver_credential,
    current_policy_context,
    policy_scope,
    resolve,
)
from lazybridge.policy.context import _policy_scope
from lazybridge.testing import MockAgent

AUTHORITY = frozenset({"run_tests", "package_install", "publish_pr"})
OWNER = ApproverContext(actor="owner", role="owner")


def provision(store: Store, principal: str = "owner", *roles: str) -> str:
    issuer = ApprovalQueue(store, allow_credential_issuance=True)
    return issuer.issue_approver_credential(principal, roles or ("owner",))


def invocation_credential() -> str:
    credential = current_approver_credential()
    assert credential
    return credential


class Clock:
    now = datetime(2026, 11, 24, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def grant(folder: Path, authority: frozenset[str] = frozenset()) -> TaskGrant:
    return TaskGrant(folder, {"tool_classes": ["read"]}, "off", "local", approval_authority=authority)


@contextmanager
def tree(
    folder: Path,
    *,
    worker_authority: frozenset[str] = AUTHORITY,
    worker_profile: str = "edit-and-run",
    worker_label: str = "worker",
    requester_label: str = "sub-worker",
) -> Iterator[tuple[tuple[PolicyContext, ...], Callable[[PolicyContext], None]]]:
    root_policy = resolve("edit-and-run", grant(folder, AUTHORITY))
    with ExitStack() as cleanup:
        scopes: list[tuple[PolicyContext, ExitStack]] = []
        worker_policy = child_within_parent(root_policy, grant(folder, worker_authority), profile=worker_profile)
        for name, label, policy in (
            ("orchestrator", "orchestrator", root_policy),
            ("worker", worker_label, worker_policy),
            ("sub-worker", requester_label, grant(folder, worker_authority)),
        ):
            stack = cleanup.enter_context(ExitStack())
            context = stack.enter_context(_policy_scope(policy, approval_actor=label, agent_id=name))
            assert context
            scopes.append((context, stack))

        def unwind_to(context: PolicyContext) -> None:
            while scopes[-1][0] is not context:
                scopes.pop()[1].close()

        yield tuple(context for context, _ in scopes), unwind_to


def request(
    requester: PolicyContext, action_class: str = "run_tests", tier: ApprovalTier = "routine"
) -> TypedApprovalRequest:
    return TypedApprovalRequest.create(
        task_id="task",
        attempt=1,
        action=action_class,
        target=ApprovalTarget(),
        requested_by=requester.actor,
        policy_revision="revision",
        tier=tier,
        action_class=action_class,
    )


def assignee(ticket: ApprovalTicket) -> str:
    grant_used = route_ticket(ticket)
    assert grant_used is not None
    return grant_used.actor


@pytest.mark.parametrize(
    ("worker_label", "requester_label"),
    [("worker", "sub-worker"), ("owner", "sub-worker"), ("worker", "owner"), ("worker", "worker")],
)
def test_a1_nearest_authorized_parent_decides(tmp_path: Path, worker_label: str, requester_label: str) -> None:
    store = Store()
    owner_credential = provision(store)
    queue = ApprovalQueue(store)
    with tree(tmp_path, worker_label=worker_label, requester_label=requester_label) as (chain, unwind_to):
        orchestrator, worker, requester = chain
        req = request(requester)
        ticket = queue.create_typed_ticket(req, requester=requester)
        assert (ticket.tier, ticket.action_class) == ("routine", "run_tests")
        assert ticket.requester_path == tuple(context.invocation_id for context in chain)
        assert assignee(ticket) == worker.invocation_id
        assert ticket.route is not None
        assert [grant.actor for grant in ticket.route.remaining] == [orchestrator.invocation_id, "owner"]
        assert ticket.route.hops[0].from_actor == requester.invocation_id
        view = ApproverContext.from_invocation(worker).model_copy(update={"actor": "unrelated-label"})
        assert queue.list_pending_tickets(assignee=view) == [ticket]
        assert queue.list_pending_tickets(assignee=OWNER) == []
        assert not queue.decide_typed_ticket(ticket.approval_id, credential=invocation_credential(), verdict="approved")
        assert not queue.decide_typed_ticket(ticket.approval_id, credential=owner_credential, verdict="approved")
        unwind_to(worker)
        assert queue.decide_typed_ticket(ticket.approval_id, credential=invocation_credential(), verdict="approved")
        decided = queue.get_ticket(ticket.approval_id)
        assert decided and decided.decision
        assert decided.decision.grant == route_ticket(ticket)
        assert decided.decision.decider == worker.invocation_id
        unwind_to(orchestrator)
        executor = ExecutorContext(
            actor=requester.actor,
            task_id=req.task_id,
            attempt=req.attempt,
            digest=req.digest,
            policy_revision=req.policy_revision,
            target=req.target,
        )
        assert queue.consume_typed_ticket(ticket.approval_id, executor=executor)
        assert not queue.consume_typed_ticket(ticket.approval_id, executor=executor)


def test_a2_read_parent_without_authority_is_skipped(tmp_path: Path) -> None:
    queue = ApprovalQueue(Store())
    with tree(tmp_path, worker_authority=frozenset(), worker_profile="read") as (chain, unwind_to):
        orchestrator, worker, requester = chain
        ticket = queue.create_typed_ticket(request(requester), requester=requester)
        assert assignee(ticket) == orchestrator.invocation_id
        assert queue.list_pending_tickets(assignee=ApproverContext.from_invocation(worker)) == []
        unwind_to(orchestrator)
        assert queue.decide_typed_ticket(ticket.approval_id, credential=invocation_credential(), verdict="approved")


def test_a3_sensitive_actions_skip_direct_parent(tmp_path: Path) -> None:
    queue = ApprovalQueue(Store())
    with tree(tmp_path) as (chain, unwind_to):
        orchestrator, worker, requester = chain
        ticket = queue.create_typed_ticket(request(requester, "package_install", "sensitive"), requester=requester)
        assert assignee(ticket) == orchestrator.invocation_id
        unwind_to(worker)
        assert not queue.decide_typed_ticket(ticket.approval_id, credential=invocation_credential(), verdict="approved")
        unwind_to(orchestrator)
        assert queue.decide_typed_ticket(ticket.approval_id, credential=invocation_credential(), verdict="approved")


def test_a4_publication_requires_owner(tmp_path: Path) -> None:
    store = Store()
    owner_credential = provision(store)
    without_role = provision(store, "owner", "orchestrator")
    queue = ApprovalQueue(store)
    with tree(tmp_path) as (chain, unwind_to):
        orchestrator, _, requester = chain
        req = request(requester, "publish_pr", "external")
        ticket = queue.create_typed_ticket(req, requester=requester)
        assert assignee(ticket) == "owner"
        with pytest.raises(ValueError, match="tier"):
            queue.create_typed_ticket(request(requester, "publish_pr", "routine"), requester=requester)
        unwind_to(orchestrator)
        assert not queue.decide_typed_ticket(ticket.approval_id, credential=invocation_credential(), verdict="approved")
    assert not queue.decide_typed_ticket(ticket.approval_id, credential=without_role, verdict="approved")
    assert queue.decide_typed_ticket(ticket.approval_id, credential=owner_credential, verdict="approved")
    decided = queue.get_ticket(ticket.approval_id)
    assert decided and decided.decision and decided.decision.grant
    assert decided.decision.grant.source == "owner"


@pytest.mark.asyncio
async def test_a5_independent_approval_is_disabled(tmp_path: Path) -> None:
    store = Store()
    owner_credential = provision(store)
    independent = provision(store, "release-reviewer", "independent")
    queue = ApprovalQueue(store, independent_approvers={"publish_pr": "release-reviewer"})
    policy = resolve("read", grant(tmp_path))
    with policy_scope(policy, approval_actor="requester") as requester:
        assert requester
        req = request(requester, "publish_pr", "external")
        ticket = queue.create_typed_ticket(req, requester=requester)
        assert assignee(ticket) == "owner"

        def impersonate(_env: Envelope) -> str:
            child = current_policy_context()
            assert child and child.actor == "release-reviewer" and child.agent_id != requester.agent_id
            assert not queue.decide_typed_ticket(
                ticket.approval_id, credential=invocation_credential(), verdict="approved"
            )
            for run in (lambda fn: fn(), Context().run):
                with pytest.raises(PermissionError):
                    run(lambda: queue.issue_approver_credential("release-reviewer", {"independent"}))
            return "denied"

        child = MockAgent(impersonate, name="release-reviewer")
        assert (await child.run("try approval")).text() == "denied"
    assert not queue.decide_typed_ticket(ticket.approval_id, credential=independent, verdict="approved")
    assert queue.decide_typed_ticket(ticket.approval_id, credential=owner_credential, verdict="approved")

    pathless = queue.create_typed_ticket(req)
    assert not pathless.requester_path and assignee(pathless) == "owner"
    assert not queue.decide_typed_ticket(pathless.approval_id, credential=independent, verdict="approved")
    assert queue.decide_typed_ticket(pathless.approval_id, credential=owner_credential, verdict="approved")


def test_a5_requester_cannot_approve_even_with_the_grant(tmp_path: Path) -> None:
    queue = ApprovalQueue(Store())
    with tree(tmp_path) as (chain, _):
        requester = chain[-1]
        ticket = queue.create_typed_ticket(request(requester), requester=requester)
        assert not queue.decide_typed_ticket(ticket.approval_id, credential=invocation_credential(), verdict="approved")
        forged = replace(requester, approval_actor="someone-else", agent_id="other-agent")
        with pytest.raises(ValueError, match="active invocation"):
            queue.create_typed_ticket(request(forged), requester=forged)
        with policy_scope(None, approval_actor="alias") as alias:
            assert alias and alias.agent_id == requester.agent_id
            alias_ticket = queue.create_typed_ticket(request(alias), requester=alias)
        assert not queue.decide_typed_ticket(
            alias_ticket.approval_id, credential=invocation_credential(), verdict="approved"
        )


@pytest.mark.parametrize("label", [None, "owner", "release-reviewer", "unrelated-label"])
@pytest.mark.parametrize(
    ("action_class", "tier"), [("run_tests", "routine"), ("package_install", "sensitive"), ("publish_pr", "external")]
)
def test_requester_label_cannot_remove_owner_fallback_or_grant_authority(
    tmp_path: Path, label: str | None, action_class: str, tier: ApprovalTier
) -> None:
    store = Store()
    owner_credential = provision(store)
    queue = ApprovalQueue(store)
    with policy_scope(resolve("read", grant(tmp_path, AUTHORITY)), approval_actor=label) as requester:
        assert requester
        req = request(requester, action_class, tier)
        ticket = queue.create_typed_ticket(req, requester=requester)
        pathless = queue.create_typed_ticket(req)
        assert assignee(ticket) == assignee(pathless) == "owner"
        assert not queue.decide_typed_ticket(ticket.approval_id, credential=invocation_credential(), verdict="approved")
        assert not queue.decide_typed_ticket(pathless.approval_id, credential=invocation_credential(), verdict="approved")

    assert {pending.approval_id for pending in queue.list_pending_tickets(assignee=OWNER)} == {
        ticket.approval_id,
        pathless.approval_id,
    }
    assert queue.decide_typed_ticket(ticket.approval_id, credential=owner_credential, verdict="approved")
    assert queue.decide_typed_ticket(pathless.approval_id, credential=owner_credential, verdict="approved")


def test_a5_public_ticket_values_cannot_authenticate_an_approver(tmp_path: Path) -> None:
    queue = ApprovalQueue(Store())
    with tree(tmp_path) as (chain, unwind_to):
        _, worker, requester = chain
        ticket = queue.create_typed_ticket(request(requester), requester=requester)
        public = queue.get_ticket(ticket.approval_id)
        assert public and public.route and public.route.assignee
        for grant_used in (public.route.assignee, *public.route.remaining):
            for exposed in grant_used.model_dump().values():
                if isinstance(exposed, str):
                    assert not queue.decide_typed_ticket(public.approval_id, credential=exposed, verdict="approved")
        assert queue.get_ticket(public.approval_id) == public

        unwind_to(worker)
        credential = invocation_credential()
        assert credential not in public.model_dump_json()
        assert credential not in repr(worker)
        assert Context().run(current_approver_credential) is None
        assert not Context().run(
            queue.decide_typed_ticket, public.approval_id, credential=credential, verdict="approved"
        )
        connection = copy_context()
        assert queue.decide_typed_ticket(public.approval_id, credential=credential, verdict="approved")
    assert connection.run(current_approver_credential) is None


def test_a6_manual_forward_carries_summaries_to_the_owner(tmp_path: Path) -> None:
    store = Store()
    owner_credential = provision(store)
    queue = ApprovalQueue(store)
    worker_summary = EscalationSummary(
        what="Run integration tests",
        why="Confirm a regression fix",
        risk="The test touches a shared fixture",
        alternatives=("Run the unit tests only",),
    )
    orchestrator_summary = EscalationSummary(
        what="Run integration tests", why="Owner controls the fixture", risk="Fixture contention", alternatives=()
    )
    with tree(tmp_path) as (chain, unwind_to):
        orchestrator, worker, requester = chain
        ticket = queue.create_typed_ticket(request(requester), requester=requester)
        assert not queue.escalate(ticket.approval_id, worker_summary, credential=owner_credential)
        assert not queue.escalate(ticket.approval_id, worker_summary, credential=invocation_credential())
        unwind_to(worker)
        worker_credential = invocation_credential()
        assert queue.escalate(ticket.approval_id, worker_summary, credential=worker_credential)
        assert not queue.decide_typed_ticket(ticket.approval_id, credential=worker_credential, verdict="approved")
        unwind_to(orchestrator)
        assert queue.escalate(ticket.approval_id, orchestrator_summary, credential=invocation_credential())
    [at_owner] = queue.list_pending_tickets(assignee=OWNER)
    assert at_owner.route is not None
    assert [hop.to.actor for hop in at_owner.route.hops] == [worker.invocation_id, orchestrator.invocation_id, "owner"]
    assert [hop.summary for hop in at_owner.route.hops] == [None, worker_summary, orchestrator_summary]


def test_a7_timeout_advances_one_eligible_hop_and_preserves_ttl(tmp_path: Path) -> None:
    clock = Clock()
    timeout = timedelta(seconds=5)
    store = Store(db=str(tmp_path / "approvals.db"))
    queue = ApprovalQueue(store, clock=clock, escalation_timeout=timeout)
    with tree(tmp_path) as (chain, unwind_to):
        orchestrator, worker, requester = chain
        ticket = queue.create_typed_ticket(request(requester), requester=requester)
        clock.now += 3 * timeout
        with ThreadPoolExecutor(max_workers=4) as pool:
            observations = list(pool.map(queue.get_ticket, [ticket.approval_id] * 8))
        assert all(observed and assignee(observed) == orchestrator.invocation_id for observed in observations)
        unwind_to(worker)
        assert not queue.decide_typed_ticket(ticket.approval_id, credential=invocation_credential(), verdict="approved")
        reread = ApprovalQueue(store, clock=clock, escalation_timeout=timeout)
        first = reread.get_ticket(ticket.approval_id)
        assert first and first.route
        assert assignee(first) == orchestrator.invocation_id
        assert first.route.hops[-1].reason == "timeout"
        assert first.route.hops[-1].summary is not None
        assert first.expires_at == ticket.expires_at
        clock.now += timeout
        [second] = reread.list_pending_tickets(assignee=OWNER)
        assert second.route is not None
        assert [hop.reason for hop in second.route.hops] == ["initial", "timeout", "timeout"]
        assert second.route.deadline is None


def test_a8_authority_is_delegable_and_changes_only_the_policy_digest(tmp_path: Path) -> None:
    base = resolve("read", grant(tmp_path))
    parent = resolve("read", grant(tmp_path, frozenset({"run_tests"})))
    assert base.digest != parent.digest
    for compile_policy in (compile_claude, compile_codex):
        base_settings, parent_settings = compile_policy(base), compile_policy(parent)
        assert base_settings.pop("policy_digest") == base.digest
        assert parent_settings.pop("policy_digest") == parent.digest
        assert base_settings == parent_settings
    child = child_within_parent(parent, grant(tmp_path, frozenset({"run_tests"})))
    assert child.approval_authority == frozenset({"run_tests"})
    with pytest.raises(PolicyWidening, match="approval authority"):
        child_within_parent(parent, grant(tmp_path, AUTHORITY))
    with policy_scope(parent), policy_scope(None) as default_child:
        assert default_child and default_child.effective_policy
        assert default_child.effective_policy.approval_authority == frozenset()
        assert default_child.effective_policy.digest == base.digest
    with (
        policy_scope(None, tree_budget=TreeBudget()),
        pytest.raises(PolicyWidening, match="parent grant"),
        policy_scope(parent),
    ):
        pytest.fail("a budget-only parent delegated approval authority it did not hold")
    with policy_scope(replace(parent, digest=base.digest)) as context:
        assert context and context.effective_policy
        assert context.effective_policy.digest == parent.digest


def test_historical_typed_ticket_keeps_its_original_request_digest() -> None:
    old_body = {
        "task_id": "old-task",
        "attempt": 1,
        "action": "git.push",
        "target": ApprovalTarget().model_dump(mode="json"),
        "requested_by": "old-worker",
        "policy_revision": "old-revision",
        "payload": {},
    }
    old_digest = hashlib.sha256(json.dumps(old_body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    store = Store()
    queue = ApprovalQueue(store)
    credential = provision(store)
    request = TypedApprovalRequest.model_validate({**old_body, "digest": old_digest})
    assert request.canonical_digest() == old_digest
    ticket = queue.create_typed_ticket(request)
    old_raw = ticket.model_dump(
        mode="json",
        exclude={
            "tier",
            "action_class",
            "requester_path",
            "requester_agent_id",
            "route",
            "card_proposal",
            "proposal_fingerprint",
            "proposal_parent",
            "proposal_parent_invocation_id",
        },
    )
    old_raw["request"] = {**old_body, "digest": old_digest}
    store.write(f"approval:{ticket.approval_id}", old_raw)
    assert queue.decide_typed_ticket(ticket.approval_id, credential=credential, verdict="approved")
    assert queue.consume_typed_ticket(
        ticket.approval_id,
        executor=ExecutorContext(
            actor=request.requested_by,
            task_id=request.task_id,
            attempt=request.attempt,
            digest=old_digest,
            policy_revision=request.policy_revision,
            target=request.target,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", ["max_depth", "max_total_children", "max_work_units"])
async def test_a13_pool_cycle_never_recovers_grants_or_routes_to_same_agent(tmp_path: Path, bound: str) -> None:
    store = Store()
    credential = provision(store)
    queue = ApprovalQueue(store)
    original = queue.issue_card(
        credential=credential,
        action_class="run_tests",
        tier="routine",
        matcher=CardMatcher(command=("pytest",)),
        transferable=True,
    )
    root_grant = TaskGrant(
        tmp_path,
        {"tool_classes": ["read", "edit", "shell"]},
        "on",
        "local",
        approval_authority=AUTHORITY,
        approval_cards=(original,),
    )
    root_policy = resolve("edit-and-run", root_grant)
    pool = AgentPool()
    outer = None
    asked = []

    async def agent_a(env: Envelope) -> str:
        nonlocal outer
        ctx = current_policy_context()
        assert ctx and ctx.effective_policy
        if env.task == "outer":
            outer = ctx
            return await pool.route("B", "narrow")
        assert outer and outer.agent_id == ctx.agent_id and outer.invocation_id != ctx.invocation_id
        assert ctx.effective_policy.tool_classes == {"read"}
        assert ctx.effective_policy.network == "off"
        assert ctx.effective_policy.approval_authority == {"package_install"}
        assert ctx.effective_policy.approval_cards == ()
        ticket = queue.create_typed_ticket(request(ctx), requester=ctx)
        assert ticket.requester_path[-3:] == (outer.invocation_id, ctx.parent_invocation_id, ctx.invocation_id)
        assert assignee(ticket) == root.invocation_id
        assert not queue.decide_typed_ticket(ticket.approval_id, credential=invocation_credential(), verdict="approved")
        with pytest.raises(PolicyBudgetExceeded, match=bound):
            await pool.route("B", "one more hop")
        asked.append(ticket)
        return "routed"

    async def agent_b(_env: Envelope) -> str:
        ctx = current_policy_context()
        assert ctx and ctx.effective_policy and ctx.effective_policy.approval_cards == (b_card,)
        return await pool.route("A", "return")

    with policy_scope(root_policy, tree_budget=TreeBudget(**{bound: 3}), approval_actor="orchestrator") as root:
        assert root
        a_card = queue.copy_card(original.card_id, holder=root, transferable=True)
        a_grant = replace(root_grant, approval_cards=(a_card,))
        with policy_scope(a_grant, approval_actor="setup") as setup:
            assert setup
            b_card = queue.copy_card(a_card.card_id, holder=setup)
        b_grant = replace(grant(tmp_path, frozenset({"package_install"})), approval_cards=(b_card,))
        a = MockAgent(agent_a, name="A", policy=resolve("edit-and-run", a_grant))
        b = MockAgent(agent_b, name="B", policy=resolve("read", b_grant))
        pool.register(a, b)
        assert (await a.run("outer")).text() == "routed"
        [ticket] = asked
        assert queue.decide_typed_ticket(ticket.approval_id, credential=invocation_credential(), verdict="approved")


@pytest.mark.asyncio
async def test_a13_pool_review_requires_a_distinct_agent_and_session(tmp_path: Path) -> None:
    session = Session()
    pool = AgentPool()
    reviewer = MockAgent("approved", name="reviewer", budget_role="reviewer", budget_engine="claude")
    reviewer.session = session
    pool.register(reviewer)

    async def review(_env: Envelope) -> str:
        with pytest.raises(PolicyError, match="different agent and session"):
            await pool.route("reviewer", "review")
        reviewer.session = Session()
        return await pool.route("reviewer", "review independently")

    writer = MockAgent(review, name="writer", reserve_review=True, budget_engine="codex")
    writer.session = session
    with policy_scope(resolve("read", grant(tmp_path)), tree_budget=TreeBudget(max_active_children=2)):
        assert (await writer.run("write")).text() == "approved"
    reviewer.assert_call_count(1)


@pytest.mark.asyncio
async def test_a14_wider_tools_are_attenuated_and_services_run_only_after_approval(tmp_path: Path) -> None:
    queue = ApprovalQueue(Store())
    observations = []
    wide = resolve("edit-and-run", TaskGrant(tmp_path, {"tool_classes": ["read", "edit", "shell"]}, "on", "local"))
    narrow = resolve("read", grant(tmp_path))

    class CaptureEngine:
        async def run(self, env, *, tools, **kwargs):
            ctx = current_policy_context()
            assert ctx and ctx.effective_policy
            observations.append((env.task, ctx, {tool.name for tool in tools}))
            return Envelope(payload="done")

    b = Agent(
        engine=CaptureEngine(),
        name="B",
        policy=wide,
        tools=[Tool(lambda: "edit", name="edit", policy_grant="edit")],
    )
    tool = b.as_tool(policy_grant="read")
    service = ServiceAgent(queue, b, action_class="run_tests", tier="routine")
    requests = []

    async def caller(_env: Envelope) -> str:
        ctx = current_policy_context()
        assert ctx
        assert (await tool.run(task="ordinary call")).ok
        ticket = service.request("approved call", task_id="test-service")
        requests.append(ticket)
        assert ticket.request and ticket.request.service_call
        assert ticket.requester_path[-1] == ctx.invocation_id
        assert assignee(ticket) == ctx.parent_invocation_id
        assert not queue.decide_typed_ticket(ticket.approval_id, credential=invocation_credential(), verdict="approved")
        with pytest.raises(ValueError, match="configured service agent"):
            queue.create_typed_ticket(ticket.request, requester=ctx)
        with pytest.raises(PermissionError):
            await service.execute(ticket.approval_id)
        return "requested"

    a = MockAgent(caller, name="A", policy=narrow)
    caller_policy = resolve("read", grant(tmp_path, frozenset({"run_tests"})))
    with policy_scope(caller_policy, approval_actor="orchestrator"):
        assert (await a.run("work")).ok
        [ticket] = requests
    assert await service.execute(ticket.approval_id) is None
    [(task, ctx, tools)] = observations
    assert task == "ordinary call" and ctx.parent.actor == "A"
    assert ctx.effective_policy.tool_classes == {"read"} and ctx.effective_policy.network == "off"
    assert tools == set()

    # The original invocation can decide only while it is active.
    with policy_scope(caller_policy, approval_actor="orchestrator") as coordinator:
        assert not queue.decide_typed_ticket(ticket.approval_id, credential=invocation_credential(), verdict="approved")
        assert coordinator
        requests.clear()
        assert (await a.run("work")).ok
        [ticket] = requests
        assert queue.decide_typed_ticket(ticket.approval_id, credential=invocation_credential(), verdict="approved")
    b.policy = narrow
    assert await service.execute(ticket.approval_id) is None
    b.policy = wide
    results = await asyncio.gather(service.execute(ticket.approval_id), service.execute(ticket.approval_id))
    assert sum(result is not None and result.ok for result in results) == 1
    assert len(observations) == 3
    task, ctx, tools = observations[-1]
    assert task == "approved call" and ctx.parent is None
    assert ctx.effective_policy == wide and tools == {"edit"}
    receipt = queue.get_ticket(ticket.approval_id)
    assert receipt and receipt.request and receipt.decision and receipt.effect_outcome
    assert (receipt.request.requested_by, receipt.decision.decider, receipt.effect_outcome.executor) == (
        "A",
        coordinator.invocation_id,
        "B",
    )
