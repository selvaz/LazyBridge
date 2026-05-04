"""LangGraph multi-agent supervisor (research + math), ported to LazyBridge.

Original (langgraph-supervisor-py README):

    from langchain_openai import ChatOpenAI
    from langgraph_supervisor import create_supervisor
    from langgraph.prebuilt import create_react_agent

    model = ChatOpenAI(model="gpt-4o")

    def add(a: float, b: float) -> float: ...
    def multiply(a: float, b: float) -> float: ...
    def web_search(query: str) -> str: ...

    math_agent = create_react_agent(
        model=model, tools=[add, multiply], name="math_expert",
        prompt="You are a math expert. Always use one tool at a time.",
    )
    research_agent = create_react_agent(
        model=model, tools=[web_search], name="research_expert",
        prompt="You are a world class researcher with access to web search...",
    )
    workflow = create_supervisor(
        [research_agent, math_agent], model=model,
        prompt="You are a team supervisor managing a research expert and a math expert...",
    )
    app = workflow.compile()
    result = app.invoke({"messages": [{"role": "user",
        "content": "what's the combined headcount of the FAANG companies in 2024?"}]})

LazyBridge equivalent: there's no separate ``create_supervisor`` primitive —
specialist agents are simply tools of the supervisor agent. The supervisor's
LLM picks which specialist to call (and may call them sequentially or in
parallel — that decision is the model's, not the framework's).
"""

from lazybridge import Agent, LLMEngine


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def web_search(query: str) -> str:
    """Search the web for ``query`` (stub matching the original example)."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )


def main() -> None:
    math_agent = Agent(
        engine=LLMEngine(
            "gpt-4o",
            system="You are a math expert. Always use one tool at a time.",
        ),
        tools=[add, multiply],
        name="math_expert",
        description="Solves arithmetic problems using add/multiply tools.",
    )

    research_agent = Agent(
        engine=LLMEngine(
            "gpt-4o",
            system=("You are a world class researcher with access to web search. Do not do any math."),
        ),
        tools=[web_search],
        name="research_expert",
        description="Looks up current facts via web_search; never does math.",
    )

    supervisor = Agent(
        engine=LLMEngine(
            "gpt-4o",
            system=(
                "You are a team supervisor managing a research expert and a "
                "math expert. For current events, use research_expert. "
                "For math problems, use math_expert."
            ),
        ),
        tools=[research_agent, math_agent],  # agents-as-tools, no special primitive
        name="supervisor",
        verbose=True,
    )

    result = supervisor("what's the combined headcount of the FAANG companies in 2024?")
    print("\nFinal answer:", result.text())


if __name__ == "__main__":
    main()
