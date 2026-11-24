"""Engine-neutral execution and delegation policies."""

from .compile import compile_claude, compile_codex
from .context import PolicyBudgetExceeded, PolicyContext, TreeBudget, current_policy_context, policy_scope
from .models import EffectivePolicy, PolicyDenied, PolicyError, PolicyWidening, Profile, ProfileDocument, TaskGrant
from .resolve import child_within_parent, load_profiles, resolve


def opposite_engine_reviewer(writer_engine: str) -> str:
    """Return the independent reviewer engine for a supported writer."""

    engines = {"claude": "codex", "claude-code": "codex", "codex": "claude"}
    try:
        reviewer = engines[writer_engine.strip().lower()]
    except (AttributeError, KeyError) as exc:
        raise PolicyError(f"unsupported writer engine: {writer_engine!r}") from exc
    if reviewer == writer_engine.strip().lower():  # defensive invariant
        raise PolicyError("same-engine review is not allowed")
    return reviewer


__all__ = [
    "EffectivePolicy",
    "PolicyError",
    "PolicyBudgetExceeded",
    "PolicyDenied",
    "PolicyContext",
    "PolicyWidening",
    "Profile",
    "ProfileDocument",
    "TaskGrant",
    "TreeBudget",
    "child_within_parent",
    "compile_claude",
    "compile_codex",
    "current_policy_context",
    "load_profiles",
    "opposite_engine_reviewer",
    "policy_scope",
    "resolve",
]
