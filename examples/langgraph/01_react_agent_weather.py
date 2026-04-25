"""LangGraph `create_react_agent` weather example, ported to LazyBridge.

Original (LangGraph):

    from langgraph.prebuilt import create_react_agent

    def check_weather(location: str) -> str:
        '''Return the weather forecast for the specified location.'''
        return f"It's always sunny in {location}"

    graph = create_react_agent(
        "anthropic:claude-3-7-sonnet-latest",
        tools=[check_weather],
        prompt="You are a helpful assistant",
    )
    inputs = {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    for chunk in graph.stream(inputs, stream_mode="updates"):
        print(chunk)

LazyBridge equivalent: a plain ``Agent`` is already a ReAct loop. Tools are
passed by reference — schemas are derived from type hints + docstring, no
``@tool`` decorator. ``verbose=True`` prints turn-by-turn updates to stdout,
which is the equivalent of LangGraph's ``graph.stream(stream_mode="updates")``.
"""

from lazybridge import Agent, LLMEngine


def check_weather(location: str) -> str:
    """Return the weather forecast for ``location``."""
    return f"It's always sunny in {location}"


def main() -> None:
    agent = Agent(
        engine=LLMEngine("claude-opus-4-7", system="You are a helpful assistant"),
        tools=[check_weather],
        verbose=True,
    )
    result = agent("what is the weather in sf")
    print("\nFinal answer:", result.text())


if __name__ == "__main__":
    main()
