"""Plan ↔ dict round-trip helpers.

The functions here power :meth:`Plan.to_dict` / :meth:`Plan.from_dict`
and are intentionally module-level so users can import them to write
their own ``to_yaml`` / Mermaid renderers on top of the topology shape
without instantiating a Plan.

Carved out of the old monolithic ``plan.py`` (W3.1).  Public function
names are preserved exactly so the package ``__init__.py`` can
re-export them for backward compatibility (test suites import
``_sentinel_to_ref`` / ``_sentinel_from_ref`` directly).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from lazybridge.engines.plan._types import Step
from lazybridge.sentinels import (
    Sentinel,
    _FromParallel,
    _FromParallelAll,
    _FromPrev,
    _FromStart,
    _FromStep,
    from_prev,
)

if TYPE_CHECKING:
    from lazybridge.tools import Tool


def _first_arg_kwargs(tool: Tool, value: str) -> dict[str, str]:
    """Build kwargs dict using the first parameter name of the tool."""
    params = tool.definition().parameters.get("properties", {})
    if params:
        first = next(iter(params))
        return {first: value}
    return {"input": value}


def _target_to_ref(target: Any) -> dict[str, str]:
    """Serialise a Step.target to a ``{"kind": ..., "name": ...}`` ref.

    Tools referenced by name round-trip as-is; callables and Agents are
    recorded by their ``name`` attribute / ``__name__`` so a registry can
    rebind them on load.
    """
    if isinstance(target, str):
        return {"kind": "tool", "name": target}
    if hasattr(target, "_is_lazy_agent"):
        return {"kind": "agent", "name": getattr(target, "name", "agent")}
    if callable(target):
        return {"kind": "callable", "name": getattr(target, "__name__", "anon")}
    return {"kind": "unknown", "name": str(target)}


def _target_from_ref(ref: dict[str, str], registry: dict[str, Any]) -> Any:
    kind = ref.get("kind")
    name = ref.get("name", "")
    if kind == "tool":
        return name  # keep as string — tool_map resolves it at run time
    if name in registry:
        return registry[name]
    raise KeyError(
        f"Plan.from_dict: no entry in registry for {kind} target {name!r}. "
        f"Pass registry={{'{name}': <callable>}} to rebind."
    )


def _sentinel_to_ref(sentinel: Any) -> dict[str, Any] | None:
    if sentinel is None:
        return None
    if isinstance(sentinel, _FromPrev):
        return {"kind": "from_prev"}
    if isinstance(sentinel, _FromStart):
        return {"kind": "from_start"}
    if isinstance(sentinel, _FromStep):
        return {"kind": "from_step", "name": sentinel.name}
    if isinstance(sentinel, _FromParallel):
        return {"kind": "from_parallel", "name": sentinel.name}
    if isinstance(sentinel, _FromParallelAll):
        return {"kind": "from_parallel_all", "name": sentinel.name}
    if isinstance(sentinel, str):
        return {"kind": "literal", "value": sentinel}
    return None


def _sentinel_from_ref(ref: dict[str, Any] | None) -> Sentinel | str:
    if ref is None:
        return from_prev
    from lazybridge.sentinels import from_parallel, from_parallel_all, from_start, from_step

    kind = ref.get("kind")
    if kind == "from_prev":
        return from_prev
    if kind == "from_start":
        return from_start
    if kind == "from_step":
        return from_step(ref["name"])
    if kind == "from_parallel":
        return from_parallel(ref["name"])
    if kind == "from_parallel_all":
        return from_parallel_all(ref["name"])
    if kind == "literal":
        return ref["value"]
    return from_prev


def _step_to_dict(step: Step) -> dict[str, Any]:
    d: dict[str, Any] = {
        "name": step.name,
        "target": _target_to_ref(step.target),
        "task": _sentinel_to_ref(step.task),
        "parallel": step.parallel,
    }
    if step.context is not None:
        # ``context=`` is single-or-list — preserve the shape on disk so
        # ``from_dict`` round-trips faithfully.  A single sentinel/str
        # serialises to one ref dict; a list serialises to a list of
        # ref dicts.
        if isinstance(step.context, list):
            d["context"] = [_sentinel_to_ref(item) for item in step.context]
        else:
            d["context"] = _sentinel_to_ref(step.context)
    if step.writes:
        d["writes"] = step.writes
    if step.routes is not None:
        # Predicates can't be JSON-serialised — record only target step
        # names; ``from_dict`` rebinds via ``registry["routes:<step>:<target>"]``.
        d["routes"] = sorted(step.routes.keys())
    if step.routes_by is not None:
        d["routes_by"] = step.routes_by
    return d


def _step_from_dict(data: dict[str, Any], registry: dict[str, Any]) -> Step:
    target = _target_from_ref(data["target"], registry)
    task = _sentinel_from_ref(data.get("task"))
    context: Sentinel | str | list[Sentinel | str] | None
    if "context" not in data:
        context = None
    else:
        raw = data["context"]
        if isinstance(raw, list):
            context = [_sentinel_from_ref(item) for item in raw]
        else:
            context = _sentinel_from_ref(raw)
    routes: dict[str, Callable[[Any], bool]] | None = None
    if "routes" in data:
        # Predicates live in Python; rebind by registry key
        # ``f"routes:{step_name}:{target}"``.  Missing keys raise
        # ``KeyError`` so the load fails loud.
        step_name = data.get("name", "<unnamed>")
        routes = {}
        for target_name in data["routes"]:
            key = f"routes:{step_name}:{target_name}"
            if key not in registry:
                raise KeyError(
                    f"Plan.from_dict: no entry in registry for "
                    f"{key!r} (predicate for routes={{{target_name!r}: ...}}). "
                    f"Pass registry={{{key!r}: predicate}} to rebind."
                )
            routes[target_name] = registry[key]
    return Step(
        target=target,
        task=task,
        context=context,
        writes=data.get("writes"),
        parallel=data.get("parallel", False),
        name=data.get("name"),
        routes=routes,
        routes_by=data.get("routes_by"),
    )
