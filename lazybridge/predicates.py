"""``when`` — declarative predicates for ``Step(routes={...})``.

The Plan engine calls each ``routes`` value as ``predicate(envelope) ->
bool``.  Writing those predicates as raw lambdas works but is dense
and Python-flavoured; the ``when`` DSL replaces them with a chained,
discoverable, English-shaped form.

::

    # Lambda form (still supported as an escape hatch):
    Step(searcher, routes={"apology": lambda env: not env.payload.items})

    # DSL form (preferred):
    from lazybridge import when
    Step(searcher, routes={"apology": when.field("items").empty()})

The DSL is intentionally small — half a dozen verbs cover ~95% of
real-world routing decisions.  When the predicate is genuinely
complex, fall back to ``when.payload(callable)`` (or a plain lambda)
so the user keeps full Python expressivity without paying the
cognitive cost on the easy cases.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from lazybridge.envelope import Envelope

# ---------------------------------------------------------------------------
# Public API — accessed via the ``when`` singleton at the bottom of the file.
# ---------------------------------------------------------------------------


class _When:
    """Entry point for the predicate DSL.

    The methods on this class do not return ``bool`` — they return
    *predicates* (callables ``Envelope -> bool``) that ``Step.routes``
    invokes at runtime.  This is why ``when.field("x").empty()`` is
    written without parentheses around the chain: the trailing call
    constructs the predicate; ``Step`` calls it later.
    """

    @staticmethod
    def field(name: str) -> _FieldBuilder:
        """Inspect ``env.payload.<name>`` (typed when ``output=`` is set).

        The returned builder exposes verbs (``equals``, ``empty``,
        ``in_``, …) that each finalise the chain into a predicate.
        """
        return _FieldBuilder(name)

    @staticmethod
    def payload(predicate: Callable[[Any], bool]) -> Callable[[Envelope], bool]:
        """Escape hatch: pass a callable that inspects the whole payload.

        Useful when the predicate spans multiple fields or runs custom
        logic the DSL doesn't cover.  Equivalent to
        ``lambda env: predicate(env.payload)``.
        """

        def _predicate(env: Envelope) -> bool:
            return bool(predicate(env.payload))

        return _predicate

    @staticmethod
    def envelope(predicate: Callable[[Envelope], bool]) -> Callable[[Envelope], bool]:
        """Escape hatch: pass a callable that inspects the whole Envelope.

        For the rare case where the predicate needs metadata, error
        state, or context — not just the payload.
        """

        def _predicate(env: Envelope) -> bool:
            return bool(predicate(env))

        return _predicate

    @staticmethod
    def errored() -> Callable[[Envelope], bool]:
        """Predicate that fires when the step produced an error envelope."""

        def _predicate(env: Envelope) -> bool:
            return env.error is not None

        return _predicate


# ---------------------------------------------------------------------------
# Field builder — produced by ``when.field("name")``.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FieldBuilder:
    """Intermediate object — call one of the verbs below to finalise."""

    name: str

    # ----- equality / identity -----

    def equals(self, value: Any) -> Callable[[Envelope], bool]:
        """``env.payload.<name> == value``."""

        def _predicate(env: Envelope) -> bool:
            return _safe_get(env, self.name) == value

        return _predicate

    def not_equals(self, value: Any) -> Callable[[Envelope], bool]:
        """``env.payload.<name> != value``."""

        def _predicate(env: Envelope) -> bool:
            return _safe_get(env, self.name) != value

        return _predicate

    def is_(self, value: Any) -> Callable[[Envelope], bool]:
        """``env.payload.<name> is value`` — for ``True`` / ``False`` / ``None``.

        Use this for boolean / sentinel checks where ``equals`` would
        give a false positive on truthy objects.
        """

        def _predicate(env: Envelope) -> bool:
            return _safe_get(env, self.name) is value

        return _predicate

    # ----- emptiness -----

    def empty(self) -> Callable[[Envelope], bool]:
        """``not env.payload.<name>`` — empty list / dict / str / None."""

        def _predicate(env: Envelope) -> bool:
            return not _safe_get(env, self.name)

        return _predicate

    def not_empty(self) -> Callable[[Envelope], bool]:
        """``bool(env.payload.<name>)`` — truthy / non-empty."""

        def _predicate(env: Envelope) -> bool:
            return bool(_safe_get(env, self.name))

        return _predicate

    # ----- membership -----

    def in_(self, values: Iterable[Any]) -> Callable[[Envelope], bool]:
        """``env.payload.<name> in values``.

        ``values`` is materialised into a frozenset when possible (for
        O(1) membership) and falls back to a tuple for unhashable
        members.
        """
        try:
            haystack: Any = frozenset(values)
        except TypeError:
            haystack = tuple(values)

        def _predicate(env: Envelope) -> bool:
            return _safe_get(env, self.name) in haystack

        return _predicate

    def not_in_(self, values: Iterable[Any]) -> Callable[[Envelope], bool]:
        """``env.payload.<name> not in values``."""
        inner = self.in_(values)

        def _predicate(env: Envelope) -> bool:
            return not inner(env)

        return _predicate

    # ----- comparison -----

    def greater_than(self, threshold: Any) -> Callable[[Envelope], bool]:
        """``env.payload.<name> > threshold``."""

        def _predicate(env: Envelope) -> bool:
            value = _safe_get(env, self.name)
            return value is not None and value > threshold

        return _predicate

    def less_than(self, threshold: Any) -> Callable[[Envelope], bool]:
        """``env.payload.<name> < threshold``."""

        def _predicate(env: Envelope) -> bool:
            value = _safe_get(env, self.name)
            return value is not None and value < threshold

        return _predicate

    # ----- regex -----

    def matches(self, pattern: str) -> Callable[[Envelope], bool]:
        """``re.search(pattern, env.payload.<name>)`` — string fields only.

        A non-string field returns False (no match).
        """
        compiled = re.compile(pattern)

        def _predicate(env: Envelope) -> bool:
            value = _safe_get(env, self.name)
            return isinstance(value, str) and compiled.search(value) is not None

        return _predicate


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _safe_get(env: Envelope, name: str) -> Any:
    """Return ``env.payload.<name>`` or ``None`` if the path is missing.

    Falling back to ``None`` rather than raising lets predicates fire
    cleanly even when the payload is a string (no attribute) or a
    dict (no attribute access) — the DSL stays predictable across
    payload shapes.
    """
    payload = env.payload
    if payload is None:
        return None
    if hasattr(payload, name):
        return getattr(payload, name)
    if isinstance(payload, dict):
        return payload.get(name)
    return None


# Public singleton — ``from lazybridge import when``.
when: _When = _When()
