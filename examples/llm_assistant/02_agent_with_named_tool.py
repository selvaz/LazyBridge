"""Agent + explicit ``tool()`` wrapping.

The bare ``tools=[get_weather]`` form works but the explicit form
pins the LLM-visible tool name even if the function is renamed
internally — keeps tool-maps and plan references stable.
"""

from __future__ import annotations

from lazybridge import tool
from lazybridge.testing import MockAgent


def get_weather(city: str) -> str:
    """Return the current weather for ``city``."""
    return f"{city}: 22°C and sunny."


def main() -> None:
    # Step 1 — wrap the function with an explicit LLM-visible name.
    # The factory introspects type hints + the docstring to build a
    # JSON Schema that the LLM uses for argument validation.
    weather = tool(get_weather, name="get_weather")
    defn = weather.definition()  # ToolDefinition — the LLM-facing contract
    print("Tool name (what the LLM sees):", weather.name)
    print("Tool description:", defn.description.strip())
    print("Tool JSON Schema (parameters keys):", list(defn.parameters.keys()))

    # Step 2 — pass it to a real Agent like any other tool.  In a
    # production example you'd swap MockAgent for ``LLMEngine(...)``
    # and ``Agent(engine=..., tools=[weather])``.  MockAgent is
    # standalone (no tools list) so we just demonstrate the wrap.
    mock = MockAgent(["Weather lookup result: 22°C and sunny."], name="weather_bot")
    print(mock("Check the weather in Paris.").text())


if __name__ == "__main__":
    main()
