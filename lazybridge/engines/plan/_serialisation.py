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

import re
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

# Valid step name: alphanumeric, underscores, hyphens — no path separators
# or shell-special characters that could indicate injection attempts.
_STEP_NAME_RE = re.compile(r"^[\w][\w\-]*$")


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


def _type_to_name(tp: Any) -> str:
    """Serialise a Step ``output=`` / ``input=`` type annotation by name.

    Types (Pydantic models, builtins, generics) live in Python and cannot
    be JSON-serialised — like callables/Agents they are recorded by name
    and rebound at load time via the ``from_dict`` registry.
    """
    name = getattr(tp, "__name__", None)
    return name if isinstance(name, str) and name else str(tp)


def _type_from_name(
    name: str,
    registry: dict[str, Any],
    *,
    step_name: Any,
    slot: str,
) -> Any:
    """Rebind a serialised type name to a live type via the registry.

    Looks up ``f"type:{name}"`` first (collision-proof against target
    names), then the bare ``name``.  Missing entries raise ``KeyError``
    so the load fails loud instead of silently degrading the step to
    ``output=str`` — which used to break ``routes_by`` plans at
    recompile time and downgrade every structured step to a raw string.
    """
    for key in (f"type:{name}", name):
        if key in registry:
            return registry[key]
    raise KeyError(
        f"Plan.from_dict: no entry in registry for {slot} type {name!r} "
        f"of step {step_name!r}.  Pass registry={{'type:{name}': <class>}} "
        f"(or {name!r}: <class>) to rebind."
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
        name = ref.get("name", "")
        _validate_step_name(name, context="from_step sentinel")
        return from_step(name)
    if kind == "from_parallel":
        name = ref.get("name", "")
        _validate_step_name(name, context="from_parallel sentinel")
        return from_parallel(name)
    if kind == "from_parallel_all":
        name = ref.get("name", "")
        _validate_step_name(name, context="from_parallel_all sentinel")
        return from_parallel_all(name)
    if kind == "literal":
        value = ref["value"]
        return value if isinstance(value, str) else str(value)
    raise ValueError(
        f"Plan sentinel deserialisation: unknown kind={kind!r} in ref={ref!r}. "
        f"Expected one of: from_prev, from_start, from_step, from_parallel, "
        f"from_parallel_all, literal. "
        f"Fix: regenerate the serialised Plan with a current LazyBridge version, "
        f"or correct the 'kind' field manually."
    )


def _validate_step_name(name: str, *, context: str = "") -> None:
    """Raise ValueError if *name* is not a safe step identifier.

    Guards against empty strings or names with path separators / shell
    metacharacters that could indicate a tampered checkpoint payload.
    """
    if not name or not isinstance(name, str):
        raise ValueError(
            f"Plan.from_dict: invalid step name {name!r}"
            + (f" in {context}" if context else "")
            + " — must be a non-empty string."
        )
    if not _STEP_NAME_RE.match(name):
        raise ValueError(
            f"Plan.from_dict: step name {name!r}"
            + (f" in {context}" if context else "")
            + " contains disallowed characters (only \\w and - are permitted)."
        )


def validate_plan_refs(
    steps: list[dict[str, Any]],
) -> list[str]:
    """Validate that all sentinel step references resolve to declared step names.

    Call this after :func:`_step_from_dict` if you want to catch dangling
    sentinel references before running the plan.  Returns a list of error
    strings (empty list = no issues).

    Parameters
    ----------
    steps:
        The raw list of step dicts as produced by :func:`_step_to_dict`
        (i.e. the ``"steps"`` key from ``Plan.to_dict()``).
    """
    known_names: set[str] = {s.get("name", "") for s in steps}
    errors: list[str] = []

    def _check_sentinel(ref: Any, ctx: str) -> None:
        if not isinstance(ref, dict):
            return
        kind = ref.get("kind", "")
        if kind in ("from_step", "from_parallel", "from_parallel_all"):
            name = ref.get("name", "")
            if name not in known_names:
                errors.append(
                    f"{ctx}: sentinel kind={kind!r} references unknown step {name!r} (known: {sorted(known_names)})"
                )

    for step in steps:
        step_name = step.get("name", "<unnamed>")
        _check_sentinel(step.get("task"), f"step {step_name!r} task")
        ctx_raw = step.get("context")
        if isinstance(ctx_raw, list):
            for item in ctx_raw:
                _check_sentinel(item, f"step {step_name!r} context[]")
        else:
            _check_sentinel(ctx_raw, f"step {step_name!r} context")
        for route_target in step.get("routes", []):
            if isinstance(route_target, str) and route_target not in known_names:
                errors.append(
                    f"step {step_name!r} routes: target {route_target!r} not in known steps"
                    f" (known: {sorted(known_names)})"
                )
        after = step.get("after_branches")
        if isinstance(after, str) and after not in known_names:
            errors.append(
                f"step {step_name!r} after_branches: target {after!r} not in known steps (known: {sorted(known_names)})"
            )

    return errors


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
    # ``output=`` / ``input=`` are types — recorded by name (rebound via
    # the from_dict registry, exactly like callable/Agent targets).
    # Omitting them was silently lossy: every structured step degraded
    # to ``output=str`` on reload and ``routes_by`` plans failed to
    # recompile (routes_by requires a Pydantic model output).
    if step.output is not str:
        d["output"] = _type_to_name(step.output)
    if step.input is not Any:
        d["input"] = _type_to_name(step.input)
    if step.routes is not None:
        # Predicates can't be JSON-serialised — record only target step
        # names; ``from_dict`` rebinds via ``registry["routes:<step>:<target>"]``.
        d["routes"] = sorted(step.routes.keys())
    if step.routes_by is not None:
        d["routes_by"] = step.routes_by
    if step.after_branches is not None:
        d["after_branches"] = step.after_branches
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
    output: Any = str
    if "output" in data:
        output = _type_from_name(str(data["output"]), registry, step_name=data.get("name"), slot="output")
    input_: Any = Any
    if "input" in data:
        input_ = _type_from_name(str(data["input"]), registry, step_name=data.get("name"), slot="input")
    return Step(
        target=target,
        task=task,
        context=context,
        writes=data.get("writes"),
        input=input_,
        output=output,
        parallel=data.get("parallel", False),
        name=data.get("name"),
        routes=routes,
        routes_by=data.get("routes_by"),
        after_branches=data.get("after_branches"),
    )
