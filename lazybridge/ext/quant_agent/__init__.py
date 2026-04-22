"""lazybridge.ext.quant_agent — Pre-configured quantitative analysis agent.

Hybrid architecture: complex tools get dedicated sub-agent pipelines
(agent_tool) for intelligent parameter construction; simple tools use
direct tool calling (plain Tool) for efficiency.

Quick start::

    from lazybridge.ext.quant_agent import quant_agent

    agent, rt = quant_agent("anthropic")
    resp = agent("Download SPY and analyze its volatility")
    print(resp.text())
    rt.close()
"""

from lazybridge.ext.quant_agent.agent import QUANT_SYSTEM_PROMPT, quant_agent

__all__ = ["QUANT_SYSTEM_PROMPT", "quant_agent"]
