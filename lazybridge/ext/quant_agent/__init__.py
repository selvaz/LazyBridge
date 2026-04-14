"""lazybridge.ext.quant_agent — Pre-configured quantitative analysis agent.

Hybrid architecture: complex tools get dedicated sub-agent pipelines
(agent_tool) for intelligent parameter construction; simple tools use
direct tool calling (plain LazyTool) for efficiency.

Quick start::

    from lazybridge.ext.quant_agent import quant_agent

    agent, rt = quant_agent("anthropic")
    resp = agent.loop("Download SPY and analyze its volatility")
    print(resp.content)
    rt.close()
"""

from lazybridge.ext.quant_agent.agent import quant_agent, QUANT_SYSTEM_PROMPT  # noqa: F401

__all__ = ["quant_agent", "QUANT_SYSTEM_PROMPT"]
