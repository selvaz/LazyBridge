"""Routed calls to agents that execute under their own policy."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from lazybridge.policy import ApprovalTier, EffectivePolicy, current_policy_context
from lazybridge.policy.context import _require_operator_context
from lazybridge.policy.resolve import _with_policy_digest

from .queue import (
    DEFAULT_TTL,
    ApprovalQueue,
    ApprovalTarget,
    ApprovalTicket,
    ExecutorContext,
    ServiceCall,
    TypedApprovalRequest,
    _hash_prompt,
)

if TYPE_CHECKING:
    from lazybridge import Agent, Envelope


class ServiceAgent:
    """An operator-configured service with a request step and a one-shot dispatch."""

    def __init__(
        self,
        queue: ApprovalQueue,
        agent: Agent,
        *,
        action_class: str,
        tier: ApprovalTier,
    ) -> None:
        _require_operator_context()
        if not isinstance(agent.policy, EffectivePolicy):
            raise ValueError("a service agent requires a standalone effective policy")
        if not agent.name.strip() or not action_class.strip():
            raise ValueError("a service agent and its action class require names")
        self._queue = queue
        self._agent = agent
        self._name = agent.name
        self._policy = _with_policy_digest(agent.policy)
        self._action_class = action_class
        self._tier = tier

    def _binding(self, task: str) -> ServiceCall:
        return ServiceCall(agent=self._name, policy_digest=self._policy.digest, task_digest=_hash_prompt(task))

    def request(self, task: str, *, task_id: str, attempt: int = 1, ttl: timedelta = DEFAULT_TTL) -> ApprovalTicket:
        """File the current invocation's service request for human or ancestor approval."""
        requester = current_policy_context()
        if requester is None or not task.strip():
            raise ValueError("a service request requires an active invocation and a nonblank task")
        request = TypedApprovalRequest.create(
            task_id=task_id,
            attempt=attempt,
            action=f"Run the {self._name} service",
            target=ApprovalTarget(),
            requested_by=requester.actor,
            policy_revision=self._policy.revision,
            tier=self._tier,
            action_class=self._action_class,
            service_call=self._binding(task),
        )
        request = self._queue._validate_typed_request(request, ttl=ttl, requester=requester)
        return self._queue._file_typed_ticket(
            request, prompt=task, kind="approval", ttl=ttl, operator_only=False, requester=requester, paths=None
        )

    async def execute(self, approval_id: str) -> Envelope | None:
        """Dispatch the approved input as the service principal, once per ticket."""
        _require_operator_context()
        ticket = self._queue.get_ticket(approval_id)
        if ticket is None or ticket.request is None:
            return None
        request = ticket.request
        binding = self._binding(ticket.prompt)
        policy = self._agent.policy
        if (
            request.service_call != binding
            or request.tier != self._tier
            or request.action_class != self._action_class
            or not isinstance(policy, EffectivePolicy)
            or _with_policy_digest(policy).digest != self._policy.digest
            or self._agent.name != self._name
        ):
            return None
        executor = ExecutorContext(
            actor=request.requested_by,
            task_id=request.task_id,
            attempt=request.attempt,
            digest=request.digest,
            policy_revision=request.policy_revision,
            target=request.target,
        )
        if not self._queue._consume_typed_ticket(approval_id, executor=executor, service_call=binding):
            return None
        try:
            result = await self._agent.run(ticket.prompt)
        except BaseException as exc:
            self._queue._record_effect_outcome(
                approval_id, executor=executor, service_call=binding, succeeded=False, detail=type(exc).__name__
            )
            raise
        self._queue._record_effect_outcome(approval_id, executor=executor, service_call=binding, succeeded=result.ok)
        return result
