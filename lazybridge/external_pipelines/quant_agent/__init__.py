#: Stability tag — see ``docs/guides/core-vs-ext.md``.
#: ``"domain"`` = worked example shipped with the framework; not part of
#: the LazyBridge framework contract and may be moved or removed.

"""lazybridge.external_pipelines.quant_agent — Pre-configured quantitative analysis agent (domain example).

Domain example shipped with LazyBridge — not part of the framework
contract. Pin to a specific lazybridge release if you depend on it.

Hybrid architecture: complex tools get dedicated sub-agent pipelines
(agent_tool) for intelligent parameter construction; simple tools use
direct tool calling (plain Tool) for efficiency.

Quick start::

    from lazybridge.external_pipelines.quant_agent import quant_agent

    agent, rt = quant_agent("anthropic")
    resp = agent("Download SPY and analyze its volatility")
    print(resp.text())
    rt.close()
"""

from lazybridge.external_pipelines.quant_agent.agent import QUANT_SYSTEM_PROMPT, quant_agent

__all__ = ["QUANT_SYSTEM_PROMPT", "quant_agent"]
