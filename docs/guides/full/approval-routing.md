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
| External | Pull requests, production, spending, deletion, messages | Owner |

The built-in class names are `command`, `cleanup`, `run_tests`, `local_commit`,
`scratch_env`, `network`, `package_install`, `push_branch`, `publish_pr`,
`production`, `money`, `deletion`, and `message`. Configure additional classes
with `ApprovalQueue(action_tiers=...)`. Unclassified actions require the
external tier. The registry rejects tiers below a configured class's minimum.

Pass the current requester's `PolicyContext` to `create_typed_ticket` for a
manual flow or `submit_typed_request` to try a card first. Agent names and the
optional `approval_actor` label on low-level scopes are display hints. They
can appear in request metadata and card receipts. The request digest binds
the original `requested_by` metadata for later execution checks. Routing and
decision checks use the core's invocation IDs, agent identities, and effective
policy grants, plus the owner principal backed by a registry credential.
Nested low-level scopes retain the enclosing agent's identity. The ticket
records the requester path and the route's policy grants without storing
credentials.

An ancestor uses the credential of its active invocation. For example, after
a test runner files a ticket, its coordinator resumes and decides it.

```python
from pathlib import Path

from lazybridge import Agent, Envelope, Store
from lazybridge.ext.approval import ApprovalQueue, ApprovalTarget, TypedApprovalRequest
from lazybridge.policy import TaskGrant, current_approver_credential, current_policy_context, policy_scope, resolve

folder = Path.cwd()
queue = ApprovalQueue(Store())
parent_policy = resolve(
    "edit-and-run",
    TaskGrant(
        folder, {"tool_classes": ["shell"]}, "off", "local",
        approval_authority=frozenset({"run_tests"}),
    ),
)


class TestRunner:
    async def run(self, env, **kwargs):
        context = current_policy_context()
        assert context is not None
        request = TypedApprovalRequest.create(
            task_id="verify-release",
            attempt=1,
            action=env.task,
            target=ApprovalTarget(path="tests"),
            requested_by=context.actor,
            policy_revision=parent_policy.revision,
            tier="routine",
            action_class="run_tests",
        )
        ticket = queue.create_typed_ticket(request, requester=context)
        return Envelope(payload=ticket.approval_id)


runner = Agent(engine=TestRunner(), name="runner")
with policy_scope(parent_policy, approval_actor="coordinator"):
    result = runner("run unit tests")
    assert result.ok
    credential = current_approver_credential()
    assert credential is not None
    assert queue.decide_typed_ticket(result.text(), credential=credential, verdict="approved")
```

Use `route_ticket(ticket)` to inspect the assigned grant and
`list_pending_tickets(assignee=context)` to filter work by routing identity.
Use `escalate(approval_id, summary, credential=credential)` to forward a
pending ticket from its current assignee. Each manual or timeout hop records
its reason, destination, and a summary with what, why, risk, and alternatives.
The default per-hop timeout is five minutes. Reading, listing, or deciding a
due ticket applies the timeout. The ticket's original expiry still applies.
The decision records the approver and the grant used. Call
`consume_typed_ticket` with the authenticated executor and original request
bindings immediately before carrying out an approved effect.

## Provision operator credentials

Provision owner credentials outside agent execution.
The operator CLI writes the secret to a new file and stores its hash with the
principal and roles in the registry. Choose an operator-controlled directory.
On Windows the file inherits the directory's access permissions. For the
default owner on the default queue prefix, run

```sh
python -m lazybridge.ext.approval --db approvals.sqlite issue-approver-credential owner --role owner --output owner.credential
```

Use `--prefix` for a custom queue prefix. From host Python, use
`ApprovalQueue(store, allow_credential_issuance=True)` and call
`issue_approver_credential(principal, roles)`. Keep that handle and write
access to the backing registry under operator control. Give agents the
default queue with issuance disabled. Keep owner credentials out of agent
contexts, tool arguments, and logs. The core prevents an invocation from
approving its own ticket under this boundary. In-process code holding the
owner's credential acts as the owner.

Set `owner_actor` to the owner principal. Tickets without a requester path
go to the owner, including historical unclassified tickets. NC-14 disables
independent approval pending the task 1b authentication review. The
`independent_approvers` option is accepted for compatibility and has no effect.
Previously filed tickets retain their recorded hops. A ticket assigned to an
independent approver can reach its recorded owner hop on timeout. Refile any
pending ticket whose stored route omitted the owner.

## Pool calls and service agents

Ordinary agent tools and pool routes apply the intersection of the caller's
delegable policy and the callee's configuration at every hop. Expose these
tools to a policy-enabled caller with an explicit grant, such as
`agent.as_tool(policy_grant="read")` or `pool.as_tool(policy_grant="read")`.
The callee keeps only
configured cards that are valid transfers from the caller's hand. In a cycle,
the returning invocation keeps the same agent identity but receives a new
invocation ID. Routing skips every ancestor with the requester's agent
identity. A configured `TreeBudget` charges each tool or pool hop against
its depth, child, and work limits. A reserved review requires a separate agent
identity and session from the writer.

When a callee needs its full configured policy, register it as a
`ServiceAgent(queue, agent, action_class=..., tier=...)` in the host. Give the
caller access to the service's request operation. From inside an invocation,
`service.request(task, task_id=...)` files a typed ticket for the exact task
and service policy. The tier routes the ticket through the ordinary approval
rules. After an eligible approver decides, the host calls
`await service.execute(ticket.approval_id)` from outside the caller's
invocation. The registry consumes the approval atomically before dispatch.
The service starts a separate invocation with its configured policy. The
ticket records the requester, the decision records the approver, and the
effect outcome records the service that ran. The task is visible in the
ticket prompt, so pass any service secrets through its host configuration.

## Issue and specialise cards

The owner issues an original with `issue_card(credential=..., ...)`. A matcher holds a fixed
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

Call `prune_card_copies()` to remove expired or revoked copies, including
copies invalidated through an ancestor. The queue retains originals, template
selections, and approval receipts. Existing card registries migrate on first
access. Upgrade all workers that share a registry before the first access.

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
requester's agent identity and the parent's invocation. It ignores display
labels and attempt numbers.
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
