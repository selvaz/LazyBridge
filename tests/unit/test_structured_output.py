"""Structured output behaviour.

Covers:
* ``output=list[Model]`` / ``output=dict[str, Model]`` activate
  structured output (generic forms are accepted).
* ``validate_payload_against_output_type`` validates plain Pydantic,
  list[Model], dict[str, Model], str passthrough, mixed payloads.
* ``Agent(output=Model, output_validator=fn)`` applies validator after
  schema validation; validator can raise to force retry.
* Retry-on-ValidationError: when the engine returns a string payload
  for an Agent expecting a typed payload, the Agent reattempts with
  feedback (up to ``max_output_retries``).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from lazybridge import Agent, Envelope
from lazybridge.core.structured import validate_payload_against_output_type
from lazybridge.envelope import EnvelopeMetadata


class Hit(BaseModel):
    title: str
    score: float


# ---------------------------------------------------------------------------
# validate_payload_against_output_type
# ---------------------------------------------------------------------------


def test_validate_against_bare_pydantic_model_from_dict():
    out = validate_payload_against_output_type({"title": "a", "score": 0.9}, Hit)
    assert isinstance(out, Hit)
    assert out.title == "a"


def test_validate_against_bare_pydantic_model_already_instance():
    inst = Hit(title="x", score=1.0)
    assert validate_payload_against_output_type(inst, Hit) is inst


def test_validate_against_bare_pydantic_model_from_json_string():
    out = validate_payload_against_output_type('{"title": "j", "score": 0.5}', Hit)
    assert isinstance(out, Hit) and out.title == "j"


def test_validate_list_of_models_from_list_of_dicts():
    out = validate_payload_against_output_type(
        [{"title": "a", "score": 0.1}, {"title": "b", "score": 0.2}],
        list[Hit],
    )
    assert all(isinstance(h, Hit) for h in out)
    assert [h.title for h in out] == ["a", "b"]


def test_validate_list_of_models_from_json_string():
    out = validate_payload_against_output_type(
        '[{"title":"a","score":0.1},{"title":"b","score":0.2}]',
        list[Hit],
    )
    assert [h.title for h in out] == ["a", "b"]


def test_validate_dict_str_to_model():
    out = validate_payload_against_output_type(
        {"first": {"title": "a", "score": 0.1}}, dict[str, Hit],
    )
    assert isinstance(out["first"], Hit)


def test_validate_str_output_passes_through():
    assert validate_payload_against_output_type("anything", str) == "anything"
    assert validate_payload_against_output_type(42, str) == 42   # no validation


def test_validate_raises_on_mismatch():
    with pytest.raises(ValidationError):
        validate_payload_against_output_type({"title": "a"}, Hit)   # score missing


# ---------------------------------------------------------------------------
# Agent.output_validator + retry-on-ValidationError
# ---------------------------------------------------------------------------


class _EngineReturning:
    """Fake engine whose ``run`` yields pre-canned Envelopes in sequence.

    Lets us simulate "model returns junk on attempt 1, valid on attempt 2"
    without touching real providers.
    """

    def __init__(self, payloads: list[Any]) -> None:
        self._queue = list(payloads)
        self.runs = 0

    async def run(self, env, *, tools, output_type, memory, session):
        self.runs += 1
        payload = self._queue.pop(0) if self._queue else self._queue[-1]
        return Envelope(task=env.task, payload=payload, metadata=EnvelopeMetadata())

    async def stream(self, *a, **kw):  # pragma: no cover
        if False:
            yield ""


def test_agent_retries_on_invalid_structured_payload():
    """Engine returns a broken dict first, then a valid one; Agent's
    validation-retry loop should yield the valid Hit."""
    engine = _EngineReturning([
        {"title": "x"},                  # missing score — raises ValidationError
        {"title": "ok", "score": 0.9},   # valid
    ])
    agent = Agent(engine=engine, output=Hit, max_output_retries=2, name="a")

    env = agent("task")
    assert env.ok
    assert isinstance(env.payload, Hit)
    assert env.payload.title == "ok"
    assert engine.runs == 2


def test_agent_returns_last_invalid_payload_after_max_retries():
    """All attempts produce invalid payloads → Agent gives up and returns
    the last attempt unchanged (so the caller sees it in ``.payload``)."""
    engine = _EngineReturning([
        {"title": "a"},    # invalid
        {"title": "b"},    # invalid
        {"title": "c"},    # invalid
    ])
    agent = Agent(engine=engine, output=Hit, max_output_retries=2, name="a")

    env = agent("task")
    # Agent ran original + 2 retries = 3 times.
    assert engine.runs == 3
    # Payload is the raw dict from the last attempt (validation never passed).
    assert env.payload == {"title": "c"}


def test_agent_output_validator_rejects_forces_retry():
    """output_validator returning None/True is fine; raising forces a
    retry with the error fed back as feedback context."""
    engine = _EngineReturning([
        Hit(title="bad", score=0.0),
        Hit(title="good", score=0.9),
    ])

    def validator(hit: Hit) -> Hit:
        if hit.score < 0.5:
            raise ValueError("score too low")
        return hit

    agent = Agent(
        engine=engine, output=Hit,
        output_validator=validator,
        max_output_retries=2, name="a",
    )
    env = agent("task")
    assert isinstance(env.payload, Hit)
    assert env.payload.title == "good"
    assert engine.runs == 2


def test_agent_list_output_validates_collection():
    engine = _EngineReturning([
        [{"title": "a", "score": 0.1}, {"title": "b", "score": 0.2}],
    ])
    agent = Agent(engine=engine, output=list[Hit], name="a")
    env = agent("task")
    assert isinstance(env.payload, list)
    assert all(isinstance(h, Hit) for h in env.payload)


def test_agent_str_output_skips_validation_retry_loop():
    """When output=str, the validation retry path is bypassed entirely."""
    engine = _EngineReturning([
        "anything",
    ])
    agent = Agent(engine=engine, output=str, max_output_retries=2)
    env = agent("task")
    assert env.payload == "anything"
    assert engine.runs == 1
