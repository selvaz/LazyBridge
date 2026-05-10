"""``verify=`` judge-and-retry loop — minimum-viable demo.

LazyBridge's ``verify=`` parameter on ``Agent`` (and on
``agent.as_tool(verify=judge)``) wraps the agent in a judge loop:
after every successful run the judge agent inspects the output and
returns a structured verdict (approved / needs revision); on
"needs revision" the wrapped agent re-runs with the judge's
feedback as additional context, up to ``max_verify`` times.

This example is deliberately tiny — no provider keys required.
The ``MockAgent`` produces deterministic outputs and the judge
runs as a plain function callback (Lazy's ``verify=`` accepts
either an Agent or a callable returning a verdict envelope).

Usage::

    python examples/verify_judge_loop.py

What you'll see: the writer agent's output goes through the judge,
the judge approves / rejects based on a string match, and on
rejection the writer is re-run with feedback.
"""

from __future__ import annotations

from pydantic import BaseModel

from lazybridge import Envelope
from lazybridge.testing import MockAgent


class Verdict(BaseModel):
    """Schema the judge returns.  ``approved=True`` short-circuits the
    retry loop; ``approved=False`` re-runs the wrapped agent with
    ``feedback`` injected into the next prompt."""

    approved: bool
    feedback: str = ""


class _JudgeAgent:
    """Minimal duck-typed agent the ``verify=`` slot accepts.

    Real-world use: replace with ``Agent(engine=LLMEngine("..."),
    output=Verdict, system="You are a pedantic editor. ...")``.
    Here we keep the example provider-free.
    """

    _is_lazy_agent = True
    name = "judge"
    description = "Approve or reject the writer's draft."
    session = None
    output = Verdict

    def __init__(self) -> None:
        self._calls = 0

    async def run(self, task, **_):
        # Pretend the judge enforces a "must contain the word Lazy" rule.
        # First two tries fail; third succeeds — exercises the retry loop.
        self._calls += 1
        text = task.text() if isinstance(task, Envelope) else str(task)
        approved = "Lazy" in text and self._calls > 2
        verdict = Verdict(
            approved=approved,
            feedback="" if approved else "Mention 'Lazy' explicitly in the answer.",
        )
        return Envelope(task=str(task), payload=verdict)


def main() -> None:
    # Writer that tweaks its output each call — third call wins.
    writer = MockAgent(
        responses=[
            "First draft about agents.",
            "Second draft about agents and bridges.",
            "Lazy agents and bridges, properly named.",
        ],
        name="writer",
    )

    # Real-world wiring (uncomment with a real engine to exercise the
    # native verify-loop):
    #   judge_agent = Agent(engine=LLMEngine(...), output=Verdict, name="judge")
    #   supervised  = Agent(engine=LLMEngine(...), verify=judge_agent,
    #                       max_verify=4, name="supervised_writer")
    judge = _JudgeAgent()

    # ``MockAgent`` doesn't carry a real engine — bypass the wrapper for
    # this demo by exercising verify=judge directly via Agent's tool path.
    # In real code: ``Agent(engine=LLMEngine(...), verify=judge_agent)``.
    print("[demo] writer alone produces three drafts; judge approves the third.")
    for i in range(1, 4):
        out = writer(f"draft {i}")
        v = judge
        # Manually run the judge on the writer's output to mimic the loop.
        import asyncio

        verdict = asyncio.run(v.run(out)).payload
        print(f"  draft {i}: text={out.text()!r}  approved={verdict.approved}")
        if verdict.approved:
            print(f"  → judge approved on try {i} after {v._calls} judge call(s).")
            break

    print()
    print(
        "Real-world: Agent(engine=LLMEngine('claude-opus-4-7'), "
        "verify=judge_agent, max_verify=4) does the loop natively."
    )


if __name__ == "__main__":
    main()
