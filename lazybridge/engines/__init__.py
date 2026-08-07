"""Core engine implementations.

``HumanEngine`` and ``SupervisorEngine`` are extension surface — import
them from :mod:`lazybridge.ext.hil`.

``ClaudeCodeEngine`` runs the model/tool loop through the locally
authenticated Claude Code runtime instead of a raw provider API call. It
is always importable; actually constructing one requires the optional
``claude-agent-sdk``/``mcp`` dependencies (``pip install
"lazybridge[claude-code]"``) — see
:mod:`lazybridge.engines.claude_code` and
:doc:`/guides/full/claude-code-engine`.
"""

from lazybridge.engines.base import Engine
from lazybridge.engines.claude_code import ClaudeCodeEngine
from lazybridge.engines.llm import LLMEngine
from lazybridge.engines.plan import Plan, Step
from lazybridge.engines.replan import PlanRound, ReplanEngine, Task

__all__ = ["ClaudeCodeEngine", "Engine", "LLMEngine", "Plan", "ReplanEngine", "PlanRound", "Step", "Task"]
