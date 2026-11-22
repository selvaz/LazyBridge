"""Uncertain outcomes do not trigger fallback, verification retry or replay."""

import asyncio
import sys
from pathlib import Path

from _harness import scenario

from lazybridge import Agent, Envelope, Plan, Step, Store
from lazybridge.engines.codex import CodexEngine
from lazybridge.engines.codex.app_server import CodexRunResult, CodexTurnUncertain
from lazybridge.engines.plan import ConcurrentPlanRunError


async def check(directory):
    counter = directory / "invocations.txt"

    class Client:
        def __init__(self, label, uncertain=True):
            self.label = label
            self.uncertain = uncertain

        async def run(self, **kwargs):
            with counter.open("a", encoding="utf-8") as stream:
                stream.write(self.label + "\n")
            if self.uncertain:
                raise CodexTurnUncertain("operation may have completed", thread_id="thread", turn_id="turn")
            return CodexRunResult(text="done", thread_id="thread")

    class Fallback:
        async def run(self, env, **kwargs):
            with counter.open("a", encoding="utf-8") as stream:
                stream.write("fallback\n")
            return Envelope(task=env.task, payload="done")

    def engine(client):
        return CodexEngine(client=client, persist_thread=True, max_retries=0)

    fallback = Agent(engine=Fallback(), name="fallback")
    result = await Agent(engine=engine(Client("fallback-primary")), name="primary", fallback=fallback).run("work")
    assert result.error and getattr(result.error, "outcome_uncertain", False)
    judgments = []

    def judge(text):
        judgments.append(text)
        return "rejected"

    result = await Agent(engine=engine(Client("verified")), name="verified", verify=judge, max_verify=3).run("work")
    assert result.error and result.error.outcome_uncertain and not judgments
    with Store(db=str(directory / "checkpoint.sqlite")) as store:
        client = Client("plan")
        plan_engine = engine(client)

        def make_plan():
            return Plan(
                Step(Agent(engine=plan_engine, name="worker"), name="worker"),
                store=store,
                checkpoint_key="checkpoint",
                resume=True,
            )

        plan = make_plan()
        result = await Agent(engine=plan, name="plan").run("work")
        assert result.error and result.error.outcome_uncertain
        assert store.read("checkpoint")["status"] == "uncertain"
        try:
            await Agent(engine=make_plan(), name="plan").run("work")
        except ConcurrentPlanRunError:
            pass
        else:
            raise AssertionError("uncertain checkpoint replayed")
        assert counter.read_text().splitlines() == ["fallback-primary", "verified", "plan"]
        assert plan.clear_checkpoint()
        client.uncertain = False
        assert (await Agent(engine=make_plan(), name="plan").run("work")).ok
    assert counter.read_text().splitlines() == ["fallback-primary", "verified", "plan", "plan"]
    return "No fallback or verification retry; replay blocked until reconciliation; counters matched."


if __name__ == "__main__":
    sys.exit(
        scenario(
            Path(__file__).stem,
            "A7 uncertain outcome",
            "Preserve uncertainty, prevent automatic replay and resume after reconciliation.",
            lambda directory: asyncio.run(check(directory)),
        )
    )
