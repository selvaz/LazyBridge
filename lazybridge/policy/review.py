"""Select the coding engine for an independent review."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, NamedTuple

from .models import PolicyError

if TYPE_CHECKING:
    from lazybridge.agent import Agent
    from lazybridge.engines.claude_code.engine import ClaudeCodeEngine
    from lazybridge.engines.codex.engine import CodexEngine

ReviewMode = Literal["opposite", "opposite-or-fresh"]
ReviewIndependence = Literal["opposite-engine", "same-provider-fresh"]

_OPPOSITE_ENGINES = {"claude": "codex", "codex": "claude"}


class ReviewSelection(NamedTuple):
    """Reviewer engine and the independence required for that selection."""

    engine: str
    independence: ReviewIndependence


def _engine_name(engine: str) -> str:
    name = engine.strip().lower()
    return "claude" if name == "claude-code" else name


def opposite_engine_reviewer(writer_engine: str) -> str:
    """Return the independent reviewer engine for a supported writer."""

    try:
        return _OPPOSITE_ENGINES[_engine_name(writer_engine)]
    except (AttributeError, KeyError) as exc:
        raise PolicyError(f"unsupported writer engine: {writer_engine!r}") from exc


def select_reviewer(
    writer_engine: str,
    available_engines: Iterable[str],
    mode: ReviewMode = "opposite",
) -> ReviewSelection:
    """Prefer the opposite engine and require a fresh reviewer on fallback.

    Availability describes the configured coding engines. Same-provider
    fallback requires the writer's engine to be the only available engine.
    Raises PolicyError if the mode allows no available reviewer.
    """

    if mode not in ("opposite", "opposite-or-fresh"):
        raise PolicyError(f"unsupported review mode: {mode!r}")
    opposite = opposite_engine_reviewer(writer_engine)
    available = {_engine_name(engine) for engine in available_engines}
    if opposite in available:
        return ReviewSelection(opposite, "opposite-engine")
    writer = _engine_name(writer_engine)
    if mode == "opposite-or-fresh" and available == {writer}:
        return ReviewSelection(writer, "same-provider-fresh")
    raise PolicyError(f"opposite reviewer engine {opposite!r} is unavailable")


def make_reviewer(
    writer_engine: ClaudeCodeEngine | CodexEngine,
    available_engines: Iterable[str],
    mode: ReviewMode = "opposite",
    *,
    preferred_model: str | None = None,
) -> tuple[Agent, ReviewIndependence]:
    """Build a reviewer and return its independence level.

    Only the working directory and, for same-provider fallback without a
    model preference, the writer's model configuration carry over. Each
    reviewer gets its own engine and Session, with ephemeral native history
    and no Memory, Store, sources, or application tools. Pass the artifact
    and any review evidence explicitly to the returned Agent.

    Under an effective writer policy, each invocation derives an offline
    read policy for the writer's folder.

    For reserved reviews, configure the enclosing TreeBudget with the same
    mode and available engines. The returned Agent can claim that slot.

    If supplied for same-provider fallback, the preferred model must differ
    from the writer's configured model after engine alias resolution.
    """
    from lazybridge.agent import Agent
    from lazybridge.engines.claude_code.engine import ClaudeCodeEngine
    from lazybridge.engines.codex.engine import CodexEngine
    from lazybridge.engines.coding import CodingAgentConfig
    from lazybridge.session import Session

    from .context import review_engine

    writer = review_engine(writer_engine)
    if writer is None:
        raise PolicyError("reviewers require a claude or codex writer engine")
    selection = select_reviewer(writer, available_engines, mode)
    fresh = selection.independence == "same-provider-fresh"
    model = preferred_model
    if fresh and model is None:
        model = writer_engine.model
    config = CodingAgentConfig.reviewer()
    engine: ClaudeCodeEngine | CodexEngine
    if selection.engine == "claude":
        engine = ClaudeCodeEngine(model=model or "sonnet", cwd=writer_engine.cwd, config=config)
    else:
        engine = CodexEngine(model=model, cwd=writer_engine.cwd, config=config)
    if fresh and preferred_model is not None and engine.model == writer_engine.model:
        raise PolicyError("preferred reviewer model must differ from the writer model")

    reviewer = Agent(engine=engine, name="reviewer", session=Session(), budget_role="reviewer")
    reviewer._fresh_review = fresh
    reviewer._read_only_review = True
    return reviewer, selection.independence
