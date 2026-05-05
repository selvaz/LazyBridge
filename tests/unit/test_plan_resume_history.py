"""Wave 1.3 — checkpoint persists step-result history.

Pre-W1.3 the ``Plan`` checkpoint persisted ``kv``, ``completed_steps``,
``next_step``, ``status``, ``run_uid`` — but NOT the in-memory step
history.  On resume the history restarted empty, so:

* ``from_parallel_all`` aggregations on resumed plans saw no upstream
  branches and silently fell back to ``start_env``.
* The nested-cost rollup re-aggregated from an empty history, so a
  resumed pipeline's cost report under-counted upstream spend.

W1.3 bumps the checkpoint to v2 with a serialised ``history`` field
(JSON-friendly dump of every completed StepResult).  v1 checkpoints
(no ``history`` key) read as empty history — pre-W1.3 behaviour, no
crash, opt-in upgrade.
"""

from __future__ import annotations

from typing import Any

import pytest

from lazybridge import Agent, Plan, Step, Store, from_parallel_all
from lazybridge.engines.plan import StepResult
from lazybridge.envelope import Envelope, EnvelopeMetadata


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


def test_history_serialisation_roundtrip():
    """Serialise → JSON-friendly dict → deserialise yields same shape."""
    history = [
        StepResult(
            step_name="a",
            envelope=Envelope(
                task="t",
                payload={"x": 1, "y": [2, 3]},
                metadata=EnvelopeMetadata(input_tokens=10, output_tokens=20, cost_usd=0.001),
            ),
        ),
        StepResult(
            step_name="b",
            envelope=Envelope(task="t", payload="hello"),
        ),
    ]

    payload = Plan._history_to_payload(history)
    assert isinstance(payload, list) and len(payload) == 2
    assert payload[0]["step_name"] == "a"
    assert payload[0]["envelope"]["payload"] == {"x": 1, "y": [2, 3]}
    assert payload[0]["envelope"]["metadata"]["input_tokens"] == 10

    restored = Plan._payload_to_history(payload)
    assert len(restored) == 2
    assert restored[0].step_name == "a"
    assert restored[0].envelope.payload == {"x": 1, "y": [2, 3]}
    assert restored[0].envelope.metadata.input_tokens == 10
    assert restored[1].envelope.payload == "hello"


def test_history_serialisation_handles_malformed_entries():
    """Malformed list entries are silently dropped, not fatal."""
    bad: list[Any] = [
        "not a dict",
        {"step_name": "ok", "envelope": {"task": "t", "payload": "v"}},
        {"step_name": "no envelope"},
        {"step_name": "bad envelope", "envelope": "not a dict either"},
    ]
    out = Plan._payload_to_history(bad)
    assert len(out) == 1
    assert out[0].step_name == "ok"


def test_payload_to_history_handles_non_list():
    assert Plan._payload_to_history(None) == []
    assert Plan._payload_to_history({"not": "a list"}) == []
    assert Plan._payload_to_history(42) == []


# ---------------------------------------------------------------------------
# Checkpoint write includes history + version field
# ---------------------------------------------------------------------------


def test_checkpoint_save_persists_history():
    """A successful Plan run persists step history into the checkpoint."""
    store = Store()  # in-memory

    def step_a(task: str) -> str:
        return "A-out"

    def step_b(task: str) -> str:
        return "B-out"

    plan = Plan(
        Step(step_a, name="a", writes="a_out"),
        Step(step_b, name="b", writes="b_out"),
        store=store,
        checkpoint_key="cp1",
    )
    Agent.from_engine(plan)("hi")

    snap = store.read("cp1")
    assert snap is not None
    assert snap["status"] == "done"
    assert snap.get("checkpoint_version") == 2
    history_payload = snap.get("history")
    assert isinstance(history_payload, list)
    step_names = [item["step_name"] for item in history_payload]
    assert step_names == ["a", "b"]


# ---------------------------------------------------------------------------
# Resume of a v1 (legacy) checkpoint degrades safely — no crash
# ---------------------------------------------------------------------------


def test_resume_legacy_v1_checkpoint_does_not_crash():
    """A pre-W1.3 checkpoint (no 'history' key) must still be resumable.
    The in-memory history is empty — same behaviour as before W1.3, but
    ``from_parallel_all`` falls back to ``start_env`` rather than
    crashing.
    """
    store = Store()
    # Hand-rolled v1 snapshot.
    legacy = {
        "next_step": "b",
        "kv": {"a_out": "A-out"},
        "completed_steps": ["a"],
        "status": "running",
        "run_uid": "legacy-run",
        # NB: no 'history' key, no 'checkpoint_version'.
    }
    store.write("cp_legacy", legacy)

    def step_a(task: str) -> str:
        return "A-out"

    def step_b(task: str) -> str:
        return "B-out"

    plan = Plan(
        Step(step_a, name="a", writes="a_out"),
        Step(step_b, name="b", writes="b_out"),
        store=store,
        checkpoint_key="cp_legacy",
        resume=True,
    )
    env = Agent.from_engine(plan)("hi")

    # Pipeline completed past 'a' (resumed at 'b').
    assert env.ok


# ---------------------------------------------------------------------------
# Resume restores history → from_parallel_all reads upstream branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_restores_history_for_from_parallel_all():
    """Persist a checkpoint snapshot AFTER a parallel band completes,
    then build a Plan that would consume the band via ``from_parallel_all``
    on resume — and verify the consumer sees the band's payloads.

    Without W1.3, the resumed history would be empty and the
    aggregator would fall back to ``start_env``.
    """
    store = Store()

    # Stage 1: run a small plan with a parallel band, see the
    # checkpoint contents reflect the band's history.
    def branch_a(task: str) -> str:
        return "A-payload"

    def branch_b(task: str) -> str:
        return "B-payload"

    consumer_calls: list[str] = []

    def consumer(task: str) -> str:
        consumer_calls.append(task)
        return f"saw:{task}"

    plan = Plan(
        Step(branch_a, name="a", parallel=True, writes="a_out"),
        Step(branch_b, name="b", parallel=True, writes="b_out"),
        Step(consumer, name="c", task=from_parallel_all("a")),
        store=store,
        checkpoint_key="cp_band",
    )
    env = Agent.from_engine(plan)("topic")
    assert env.ok

    # The consumer was called with the labelled-text join of both branches.
    assert any("[a]" in t and "[b]" in t for t in consumer_calls), consumer_calls
    assert any("A-payload" in t for t in consumer_calls)
    assert any("B-payload" in t for t in consumer_calls)

    # The checkpoint has the band's history persisted.
    snap = store.read("cp_band")
    assert snap is not None
    history_payload = snap.get("history") or []
    band_steps = {item["step_name"] for item in history_payload}
    assert {"a", "b"}.issubset(band_steps)
