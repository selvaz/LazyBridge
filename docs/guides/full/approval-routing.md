# Approval routing and cards

Use the typed `ApprovalQueue` to authorize actions requested by an invocation.
The registry either spends a standing approval card or records a ticket for a
specific approver. Both outcomes are tied to the typed request's digest.

## Grant approval authority

Set `TaskGrant.approval_authority` to the action classes that the invocation
may approve. A child can receive a subset of its parent's classes. A child
without explicit approval grants starts with an empty hand and no approval
authority. Approval authority and cards are included in the effective policy
digest. Compiled engine settings still govern execution permissions.

| Tier | Examples | Eligible assignee |
| --- | --- | --- |
| Routine | Task-local commands and cleanup, tests, local commits, scratch environments | Nearest ancestor with the action class |
| Sensitive | Network access, package installs, pushes to non-default branches | Nearest ancestor with the action class above the direct parent |
| External | Pull requests, production, spending, deletion, messages | Configured independent approver or owner |

The built-in class names are `command`, `cleanup`, `run_tests`, `local_commit`,
`scratch_env`, `network`, `package_install`, `push_branch`, `publish_pr`,
`production`, `money`, `deletion`, and `message`. Configure additional classes
with `ApprovalQueue(action_tiers=...)`. Unclassified actions require the
external tier. The registry rejects tiers below a configured class's minimum.

Pass the requester's trusted `PolicyContext` to `create_typed_ticket` for a
manual-only flow, or to `submit_typed_request` to try a card first. Both use the
invocation ancestry recorded by `policy_scope`. The request's `requested_by`
must equal the invocation's `actor`. The actor defaults to its invocation ID.
A trusted caller can bind a stable principal with `approval_actor` when opening
the scope.

For example, a test runner can request approval from its parent.

```python
from pathlib import Path

from lazybridge import Store
from lazybridge.ext.approval import ApprovalQueue, ApprovalTarget, ApproverContext, TypedApprovalRequest
from lazybridge.policy import TaskGrant, policy_scope, resolve

folder = Path.cwd()
queue = ApprovalQueue(Store())
parent_policy = resolve(
    "edit-and-run",
    TaskGrant(folder, {"tool_classes": ["shell"]}, "off", "local",
              approval_authority=frozenset({"run_tests"})),
)
with (
    policy_scope(parent_policy, approval_actor="coordinator") as parent,
    policy_scope(TaskGrant(folder, {"tool_classes": ["shell"]}, "off", "local")) as runner,
):
    assert parent is not None and runner is not None
    request = TypedApprovalRequest.create(
        task_id="verify-release",
        attempt=1,
        action="run unit tests",
        target=ApprovalTarget(path="tests"),
        requested_by=runner.actor,
        policy_revision=parent_policy.revision,
        tier="routine",
        action_class="run_tests",
    )
    ticket = queue.create_typed_ticket(request, requester=runner)
    queue.decide_typed_ticket(
        ticket.approval_id,
        approver=ApproverContext.from_invocation(parent),
        verdict="approved",
    )
```

Use `route_ticket(ticket)` to inspect the assigned grant and
`list_pending_tickets(assignee=context)` to list work for an authenticated
approver. For ancestor decisions, the registry checks the actor, invocation ID,
and effective policy digest. For owner decisions, it checks the configured
`owner_actor` and authenticated `owner` role. To designate an independent
approver, configure `independent_approvers` with an external action class and
the approver's authenticated actor. That actor must be outside the requester
chain. The owner is the final fallback. If the owner made the request and
there is nobody else eligible, the ticket stays unassigned.

An approver can forward a pending ticket with
`escalate(approval_id, summary, approver=context)`. An `EscalationSummary`
carries what is requested, why, the risk, and the alternatives considered.
Each forwarding or timeout moves to the next eligible approver and preserves
the prior hops. The default per-hop timeout is five minutes. Reading, listing,
or deciding a due ticket applies its timeout automatically. The original ticket
expiry still applies. The final decision records the actor and grant used.
With a manual approval, call `consume_typed_ticket` with the authenticated
executor and the original request bindings immediately before the effect.

## Issue and specialise cards

The owner issues an original with `issue_card`. A matcher holds a fixed
executable and subcommand prefix, optional exact trailing arguments, optional
branch and path prefix, and the required network state. Cards can carry a task
scope, project scope, expiry, maximum uses, and the right to transfer. Add
issued cards to the holder's `TaskGrant.approval_cards` at provisioning time.
External actions always use a ticket.

Use `copy_card` as the parent to make a child card from one it holds. You can
add restrictions to the matcher and scope, shorten the expiry, reduce the use
limit, or remove transfer rights. For example, a parent can specialise a
`("git", "commit")` card to branch `topic` and the absolute `src/` directory
under the task folder, then give the copy a maximum of three uses and an expiry
two hours from now. Include the returned copy in the child's task grant.
Each use counts against the card and its ancestors. Revoking any card
invalidates the downstream copies in its provenance chain.

Use `set_card_template` to select original cards for the `coder`, `reviewer`,
or `researcher` template. `copy_card_template` draws from the parent's held,
transferable cards, preserving any restrictions already applied. Templates
start empty until the owner configures their contents.

For a card check, include a `CardAction` on the typed request. The trusted
execution layer supplies the actual argument vector, current directory, every
affected path, network state, and branch or project identity when applicable.
Relative paths are resolved from the command's working directory. The core
requires that directory and all resolved paths to stay within the task folder,
including after following symlinks. It also checks the effective shell and
network permissions, rejects shell composition, and checks every restriction
on the card. Launch exactly the verified argument vector directly, with shell
evaluation disabled. Check and execute against the same filesystem, branch,
and sandbox state. Keep credentials out of stored commands and summaries.

`submit_typed_request` returns a `CardApproval` when it records a matching
card use. The receipt contains the card ID, provenance, policy digest, and
request digest. The approval requires no ticket or token. Submit once immediately
before each execution. When the request needs a ticket, three identical,
card-eligible requests within the same task make a proposal available to the
direct parent through `list_card_proposals`. The comparison uses the
requester's actor and the parent's invocation, and ignores attempt numbers.
The parent can use the proposal to decide whether to delegate a card for
future work.

## Concepts it builds on

- **Object capabilities and attenuation.** A card is an object carrying a
  specific permission. A delegated copy keeps its chain of authority and can
  reduce what its holder can do.
- **Privilege-escalation prevention.** The policy resolver checks grants
  against the immediate parent. Card use also requires the registered card and
  every upstream grant to remain valid.
- **Permission boundaries.** Folder, tool, network, and card restrictions all
  apply at execution. Each boundary must allow the operation.
- **Separation of duties.** A requester cannot decide its ticket. Sensitive
  and external actions require organizational distance from the requester.
- **Least privilege.** Grant only the action classes, card scope, lifetime,
  and uses the invocation needs. Every extra restriction can survive further
  delegation.
