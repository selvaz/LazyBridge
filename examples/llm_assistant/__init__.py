"""Examples sized for an LLM coding assistant.

Each module is one focused, runnable pattern that follows the
:doc:`canonical-form contract </for-llms/codegen-contract>`.  Run any
file directly::

    python examples/llm_assistant/01_single_agent.py

Every example uses :class:`lazybridge.testing.MockAgent` (or a
mock-backed engine) so it executes without provider credentials.
Replace the mock engine with :class:`lazybridge.LLMEngine` to hit a
real provider.
"""
