from __future__ import annotations

import asyncio
import inspect
from itertools import product
from types import SimpleNamespace

import pytest

from lazybridge import Store
from lazybridge.core.tool_schema import ToolArgumentValidationError
from lazybridge.ext.delegation import JobRegistry, make_persistent_consultant


@pytest.mark.parametrize("fresh_enabled,model_enabled,effort_enabled", list(product([False, True], repeat=3)))
async def test_consultant_signature_schema_and_dispatch_only_enabled_overrides(
    tmp_path, fresh_enabled, model_enabled, effort_enabled
):
    calls = []

    async def consultant(
        question: str, thread_id: str | None = None, model: str | None = None, effort: str | None = None
    ) -> str:
        calls.append(dict(question=question, thread_id=thread_id, model=model, effort=effort))
        return "thread_id=handle-1\nanswer"

    tasks = set()
    registry = JobRegistry(Store(db=str(tmp_path / "jobs.db")))
    tool = make_persistent_consultant(
        lambda: SimpleNamespace(func=consultant),
        tool_name="ask",
        registry=registry,
        background_tasks=tasks,
        accept_fresh=fresh_enabled,
        accept_model_override=model_enabled,
        accept_effort_override=effort_enabled,
    )
    options = {"fresh": (fresh_enabled, True), "model": (model_enabled, "chosen"), "effort": (effort_enabled, "high")}
    enabled = [name for name, (accepted, _) in options.items() if accepted]
    assert list(inspect.signature(tool.func).parameters) == ["question", *enabled]
    schema = tool.definition().parameters
    assert list(schema["properties"]) == ["question", *enabled]
    assert schema["required"] == ["question"]
    if fresh_enabled:
        assert schema["properties"]["fresh"]["type"] == "boolean"
    for name, (accepted, value) in options.items():
        if not accepted:
            with pytest.raises(ToolArgumentValidationError):
                assert await tool.run(question="blocked", **{name: value}) is None
    assert calls == [] and tasks == set()
    assert list(registry._store.items(prefix=registry._prefix)) == []
    await tool.run(question="work", **{name: options[name][1] for name in enabled})
    await asyncio.gather(*tasks)
    assert calls == [
        dict(
            question="work",
            thread_id=None,
            model="chosen" if model_enabled else None,
            effort="high" if effort_enabled else None,
        )
    ]
