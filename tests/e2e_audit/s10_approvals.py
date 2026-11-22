"""Scoped native permissions and digest-bound durable writer approvals."""

import asyncio
import contextlib
import json
import sys
from datetime import timedelta
from pathlib import Path

from _harness import PYTHON, scenario, scenario_dir, start_trace

from lazybridge import Envelope, Store
from lazybridge.engines.codex.app_server import CodexAppServerClient
from lazybridge.engines.coding import ApprovalDecision, remembering_gate
from lazybridge.ext.approval.queue import ApprovalQueue, StoreApprovalChannel
from lazybridge.ext.delegation import ExtraParam, JobRegistry, make_codex_writer

NAME = Path(__file__).stem
REQUESTED = {"network": {"enabled": True}, "fileSystem": {"write": ["D:/example"]}}
GRANTED = {"network": {"enabled": False}}


def server(directory):
    with contextlib.redirect_stdout(sys.stderr):
        trace = start_trace(directory)

    def read():
        return json.loads(sys.stdin.readline())

    def send(message):
        print(json.dumps(message), flush=True)

    try:
        init = read()
        assert init["method"] == "initialize"
        send({"id": init["id"], "result": {}})
        assert read()["method"] == "initialized"
        thread = read()
        assert thread["method"] == "thread/start"
        send({"id": thread["id"], "result": {"thread": {"id": "thread-1"}, "model": "fake"}})
        turn = read()
        assert turn["method"] == "turn/start"
        send({"id": turn["id"], "result": {"turn": {"id": "turn-1", "status": "inProgress"}}})
        responses = []
        for request_id in (101, 102):
            send(
                {
                    "id": request_id,
                    "method": "item/permissions/requestApproval",
                    "params": {"permissions": REQUESTED, "threadId": "thread-1", "turnId": "turn-1"},
                }
            )
            response = read()
            assert response["id"] == request_id
            responses.append(response["result"])
        (directory / "permission-responses.json").write_text(json.dumps(responses), encoding="utf-8")
        send({"method": "turn/completed", "params": {"turn": {"id": "turn-1", "status": "completed", "items": []}}})
    finally:
        sys.setprofile(None)
        trace.close()


async def check(directory):
    requests = []

    async def gate(request):
        requests.append(request)
        return ApprovalDecision("allow_session", permissions=GRANTED) if len(requests) == 1 else ApprovalDecision.deny()

    async def unexpected_tool(*args):
        raise AssertionError("unexpected tool call")

    client = CodexAppServerClient(command=(str(PYTHON), str(Path(__file__).resolve()), "server"))
    await asyncio.wait_for(
        client.run(
            prompt="work",
            model=None,
            cwd=str(directory),
            dynamic_tools=[],
            on_tool_call=unexpected_tool,
            approval_gate=remembering_gate(gate, set()),
            approval_policy="on-request",
        ),
        20,
    )
    responses = json.loads((directory / "permission-responses.json").read_text())
    assert len(requests) == 2 and all(
        r.kind == "permissions" and r.arguments["permissions"] == REQUESTED for r in requests
    )
    assert responses == [{"permissions": GRANTED, "scope": "session"}, {"permissions": {}, "scope": "turn"}], responses

    launches = []

    class Writer:
        async def run(self, env, **kwargs):
            launches.append(env.task)
            return Envelope(task=env.task, payload="done")

    def factory(**kwargs):
        return Writer()

    with Store(db=str(directory / "approvals.sqlite")) as store:
        queue = ApprovalQueue(store)
        channel = StoreApprovalChannel(
            queue,
            task_id="task",
            poll_seconds=0.03,
            ttl=timedelta(seconds=30),
            wait_budget=timedelta(seconds=0.3),
            renotify_interval=None,
        )
        tasks = set()
        tool = make_codex_writer(
            workspace_root=directory,
            gate=gate,
            channel=channel,
            registry=JobRegistry(store),
            background_tasks=tasks,
            doc="write",
            engine_factory=factory,
            extra_params={"cwd": ExtraParam()},
        )

        async def invoke(objective, cwd):
            await tool.run(objective=objective, cwd=str(cwd))
            await asyncio.gather(*list(tasks))

        first = "a" * 2100 + "objective A" + "b" * 1200
        second = "a" * 2100 + "objective B" + "b" * 1200
        await invoke(first, directory / "one")
        [first_ticket] = queue.list_pending_tickets()
        assert not launches
        assert queue.approve_ticket(first_ticket.approval_id, actor="tester", channel="test")
        await invoke(second, directory / "one")
        await invoke(first, directory / "two")
        tickets = [first_ticket, *queue.list_pending_tickets()]
        assert not launches and len(tickets) == 3, "different request used stale approval"
        digests = [ticket.prompt.rsplit("Request SHA-256 ", 1)[1] for ticket in tickets]
        assert len(set(digests)) == 3, digests
        await invoke(first, directory / "one")
        assert launches == [first], "matching late approval was not consumed"
    return "Scoped session grant and subsequent denial reached adapter; changed digests refused stale approval; matching request launched."


if __name__ == "__main__":
    if len(sys.argv) > 1:
        server(scenario_dir(NAME))
    else:
        sys.exit(
            scenario(
                NAME,
                "A1/A2 approvals",
                "Apply gate scope and bind durable writer approval to objective and options.",
                lambda d: asyncio.run(check(d)),
            )
        )
